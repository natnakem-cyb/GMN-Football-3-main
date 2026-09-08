"""
GMN-Football-3 -- Multi-Agent PPO (MAPPO) with Cooperative Reward Shaping
Trains a parameter-shared policy network and centralized team critic on cooperative
football scenarios with explicit pass-chain and assisted-goal incentives.

CANONICAL-TRAINER NOTE (see TASKS.md Item 2):
- train_mappo.py        = canonical PLAIN trainer (baseline; no shaping).
- train_mappo_shaped.py = canonical SHAPED trainer (this file). Shaped rewards
  come from CooperativeRewardShaper, wired end-to-end and covered by
  training/tests/test_reward_shaper.py + training/test_reward_shape_e2e.py.
  Policies from the two trainers are NOT directly comparable: shaped
  checkpoints encode pass-chain incentives.

Hyperparameters:
- Timesteps: 500,000
- Rollout length (n_steps): 256
- Mini-batch size: 256
- PPO Epochs: 4
- Learning rate: 3e-4 (Adam)
- Discount (gamma): 0.99
- GAE lambda: 0.95
- PPO Clip range: 0.15
- Value coefficient: 0.5
- Entropy coefficient: 0.01 -> 0.005 cosine anneal

Reward Shaping:
- Pass completion bonus: +0.25
- Assisted goal bonus: +0.50 (shared across all active left-team agents)
- Solitary shot penalty: -0.30 (when pass_chain_length == 0)
- Ball-hogging penalty: -0.005/tick (after 30 ticks of continuous possession)
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from training.gmn_pettingzoo import GMNMultiAgentEnv, CooperativeRewardShaper, OBSERVATION_DIM, ACTION_SPACE_SIZE
from training.mappo_networks import SharedActor, CentralizedCritic
from training.mappo_rollout import collect_rollout, compute_gae
from training.mappo_update import ppo_update
from training.eval_progress import evaluate_checkpoint_progress, persist_trend_snapshots


def run_mappo_shaped_training(
    timesteps: int = 500000,
    scenario: str = "academy_3_vs_1_with_keeper",
    seed: int = 42,
    resume_path: str = None,
    enable_reward_shaping: bool = True,
    shaper_kwargs: Dict[str, Any] = None,
    log_dir: str = None,
    validation_interval: int = 25000,
    validation_episodes: int = 50,
) -> bool:
    """
    Train MAPPO with cooperative reward shaping and TensorBoard logging.
    """
    # Enforce seed-based naming to prevent overwrites.
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    os.makedirs(model_dir, exist_ok=True)

    shaped_suffix = "shaped"
    best_model_path = os.path.join(model_dir, f"mappo_{scenario}_seed{seed}_{shaped_suffix}_best.pt")
    latest_model_path = os.path.join(model_dir, f"mappo_{scenario}_seed{seed}_{shaped_suffix}_latest.pt")

    print("==================================================")
    print(f"GMN FOOTBALL -- MAPPO WITH COOPERATIVE REWARD SHAPING")
    print(f"Target Scenario: {scenario} | Timesteps: {timesteps} | Seed: {seed}")
    if resume_path:
        print(f"Resuming From Checkpoint: {resume_path}")
    print(f"Reward Shaping: {'ENABLED' if enable_reward_shaping else 'DISABLED'}")
    if enable_reward_shaping:
        shaper_kwargs = shaper_kwargs or {}
        print(f"  Pass completion bonus: +{shaper_kwargs.get('reward_pass_completion', 0.25)}")
        print(f"  Assisted goal bonus: +{shaper_kwargs.get('reward_assisted_goal_bonus', 0.50)}")
        print(f"  Solitary shot penalty: {shaper_kwargs.get('penalty_solitary_shot', -0.30)}")
        print(f"  Ball-hogging penalty: {shaper_kwargs.get('penalty_ball_hogging', -0.005)}/tick (after {shaper_kwargs.get('max_unassisted_hold_ticks', 30)} ticks)")
    print(f"Best Checkpoint: {best_model_path}")
    print(f"Latest Checkpoint: {latest_model_path}")
    print("==================================================")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # TensorBoard setup
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "runs", f"mappo_{scenario}_seed{seed}_shaped_{timestamp}")
        )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard log directory: {log_dir}")

    print("\n1. Initializing Multi-Agent Environment & MAPPO Networks...")
    env = GMNMultiAgentEnv(
        scenario=scenario,
        auto_start_bridge=True,
        enable_reward_shaping=enable_reward_shaping,
    )
    num_agents = len(env.possible_agents)
    obs_dim = OBSERVATION_DIM
    global_state_dim = obs_dim * num_agents
    action_dim = ACTION_SPACE_SIZE

    print(f"   Controllable Agents ({num_agents}): {env.possible_agents}")
    print(f"   Local Obs Dim (with Role One-Hot): {obs_dim} | Action Dim: {action_dim}")
    print(f"   Critic Architecture: Permutation-Invariant Deep Sets Pooling (O(1) parameter scaling)")

    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    critic = CentralizedCritic(obs_dim=obs_dim, hidden=64, mode="pool")

    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)

    total_steps_elapsed = 0
    start_update = 1

    if resume_path and os.path.exists(resume_path):
        print(f"\n   -> Loading checkpoint state from: {resume_path}...")
        ckpt = torch.load(resume_path, map_location="cpu")
        ckpt_obs_dim = ckpt.get("obs_dim", 115 if "actor" in ckpt and ckpt["actor"]["net.0.weight"].shape[1] == 115 else OBSERVATION_DIM)
        if ckpt_obs_dim != OBSERVATION_DIM:
            raise RuntimeError(
                f"[GMN Contract Mismatch] Checkpoint '{resume_path}' has obs_dim={ckpt_obs_dim}, "
                f"which is incompatible with current environment OBSERVATION_DIM={OBSERVATION_DIM} (simple115_v3_role). "
                f"Pre-migration 115-dim checkpoints cannot be loaded; please re-train or re-export."
            )
        if "actor" in ckpt:
            actor.load_state_dict(ckpt["actor"])
        if "critic" in ckpt:
            critic.load_state_dict(ckpt["critic"])
        if "actor_opt" in ckpt:
            actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            critic_opt.load_state_dict(ckpt["critic_opt"])
        total_steps_elapsed = int(ckpt.get("timesteps", 0))
        print(f"   [OK] Checkpoint loaded successfully. Resuming from step {total_steps_elapsed}.")

    n_steps = 256
    remaining_timesteps = max(0, timesteps - total_steps_elapsed)
    n_updates = remaining_timesteps // n_steps
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_opt, T_max=n_updates, eta_min=3e-5)
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(critic_opt, T_max=n_updates, eta_min=3e-5)
    check_freq_steps = 10000

    print(f"\n2. Configuration:")
    print(f"   Total Updates: {n_updates} ({n_steps} steps per rollout)")
    print(f"   Remaining Timesteps: {n_updates * n_steps} (Total Target: {timesteps})")
    print(f"   PPO Epochs: 4 | Mini-batch: 256 | LR: 3e-4 (cosine anneal to 3e-5) | Clip: 0.15")
    print(f"   GAE: gamma=0.99, lambda=0.95 | Value Coef: 0.5 | Entropy Coef: 0.01 -> 0.005")

    # Metrics tracking
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_goals: List[int] = []
    trend_snapshots: List[Tuple[int, int, float, float]] = []  # (step, num_eps, mean_rew, goal_rate)
    loss_history: List[Dict[str, Any]] = []

    # Cooperative shaping diagnostics tracking
    episode_shaper_diagnostics: List[Dict[str, Any]] = []

    is_rondo_scenario = scenario == "academy_rondo_4v1"

    # Best-checkpoint selection
    best_rolling_goal_rate: float = -1.0
    best_rolling_checkpoint_step: int = 0
    best_rolling_checkpoint_path: str = ""
    best_deterministic_goal_rate: float = -1.0
    best_deterministic_checkpoint_step: int = 0
    best_deterministic_checkpoint_path: str = ""
    _has_best_deterministic: bool = False

    last_check_step = total_steps_elapsed
    last_checkpoint_step = (total_steps_elapsed // 50_000) * 50_000
    last_validation_step = total_steps_elapsed

    print(f"\n3. Starting MAPPO Training with Reward Shaping for {remaining_timesteps} steps...")
    start_time = time.time()
    total_steps_elapsed_at_start = total_steps_elapsed

    for update_idx in range(start_update, start_update + n_updates):
        # 1. Collect Rollout
        buffer = collect_rollout(env, actor, critic, num_steps=n_steps)
        total_steps_elapsed += n_steps

        # Record completed episodes and shaper diagnostics
        for ep_info in buffer.get("completed_episodes", []):
            episode_rewards.append(ep_info["reward"])
            episode_lengths.append(ep_info["length"])
            episode_goals.append(ep_info["goal"])

            # Collect shaper diagnostics if available
            if enable_reward_shaping and hasattr(env, "reward_shaper") and env.reward_shaper is not None:
                episode_shaper_diagnostics.append(env.reward_shaper.get_diagnostics())
            else:
                episode_shaper_diagnostics.append({})

        # 2. Compute GAE Advantages and Returns
        advantages, returns = compute_gae(
            rewards=buffer["rewards"],
            values=buffer["values"],
            dones=buffer["dones"],
            gamma=0.99,
            lam=0.95,
            bootstrap_value=0.0,
            next_local_obs=buffer["next_local_obs"],
            critic=critic,
            per_agent_rewards=buffer.get("per_agent_rewards"),
        )

        # 3. PPO Update Step
        entropy_coef = 0.01 - (0.01 - 0.005) * min(total_steps_elapsed / timesteps, 1.0)
        metrics = ppo_update(
            actor=actor,
            critic=critic,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            buffer=buffer,
            advantages=advantages,
            returns=returns,
            clip_range=0.15,
            n_epochs=4,
            batch_size=256,
            value_coef=0.5,
            entropy_coef=entropy_coef,
            max_grad_norm=0.5,
        )
        actor_scheduler.step()
        critic_scheduler.step()
        metrics["step"] = total_steps_elapsed
        metrics["update"] = update_idx
        metrics["entropy_coef"] = entropy_coef
        metrics["learning_rate"] = float(actor_opt.param_groups[0]["lr"])
        loss_history.append(metrics)

        # Log training metrics to TensorBoard
        writer.add_scalar("Train/PolicyLoss", metrics["policy_loss"], total_steps_elapsed)
        writer.add_scalar("Train/ValueLoss", metrics["value_loss"], total_steps_elapsed)
        writer.add_scalar("Train/Entropy", metrics["entropy"], total_steps_elapsed)
        writer.add_scalar("Train/ApproxKL", metrics["approx_kl"], total_steps_elapsed)
        writer.add_scalar("Train/LearningRate", metrics["learning_rate"], total_steps_elapsed)
        writer.add_scalar("Train/EntropyCoef", entropy_coef, total_steps_elapsed)

        # Log cooperative shaping diagnostics for completed episodes in this update
        if episode_shaper_diagnostics:
            recent_diagnostics = episode_shaper_diagnostics[-n_steps:]
            if recent_diagnostics:
                # Compute episode-level aggregates
                pass_chain_lengths = [d.get("pass_chain_length", 0) for d in recent_diagnostics]
                pass_completed_counts = [d.get("pass_completed_count", 0) for d in recent_diagnostics]
                solitary_shot_counts = [d.get("solitary_shot_count", 0) for d in recent_diagnostics]
                assisted_goal_counts = [d.get("assisted_goal_count", 0) for d in recent_diagnostics]
                ball_hogging_counts = [d.get("ball_hogging_count", 0) for d in recent_diagnostics]

                # Filter out empty diagnostics (from episodes before shaping was active)
                pass_chain_lengths = [x for x in pass_chain_lengths if x > 0 or any(d.get("pass_completed_count", 0) > 0 for d in recent_diagnostics[:1])]

                writer.add_scalar("Cooperative/Pass_Chain_Max_Mean", float(np.mean(pass_chain_lengths)) if pass_chain_lengths else 0.0, total_steps_elapsed)
                writer.add_scalar("Cooperative/Completed_Passes_Total", float(np.sum(pass_completed_counts)), total_steps_elapsed)
                writer.add_scalar("Cooperative/Assisted_Goals_Total", float(np.sum(assisted_goal_counts)), total_steps_elapsed)
                writer.add_scalar("Cooperative/Solitary_Shots_Total", float(np.sum(solitary_shot_counts)), total_steps_elapsed)
                writer.add_scalar("Cooperative/Ball_Hogging_Penalties", float(np.sum(ball_hogging_counts)), total_steps_elapsed)

        # Log episode-level reward and goal rate
        if episode_rewards:
            recent_ep_rewards = episode_rewards[-50:]
            recent_ep_goals = episode_goals[-50:]
            mean_reward = float(np.mean(recent_ep_rewards))
            goal_rate = float(np.mean(recent_ep_goals)) * 100.0
            writer.add_scalar("Performance/MeanReward_Last50", mean_reward, total_steps_elapsed)
            writer.add_scalar("Performance/GoalRate_Last50", goal_rate, total_steps_elapsed)

        # Diagnostics check for numerical instability
        if np.isnan(metrics["policy_loss"]) or np.isnan(metrics["value_loss"]):
            raise RuntimeError(f"NaN loss detected at update {update_idx}: {metrics}")
        if np.isnan(metrics["approx_kl"]) or abs(metrics["approx_kl"]) > 1.5:
            print(f"   [WARNING] High KL divergence at update {update_idx}: approx_kl = {metrics['approx_kl']:.4f}")

        # Periodic snapshot logging
        if total_steps_elapsed - last_check_step >= check_freq_steps:
            last_check_step = total_steps_elapsed
            recent_ep = episode_rewards[-50:] if episode_rewards else [0.0]
            recent_goals = episode_goals[-50:] if episode_goals else [0]
            mean_rew = float(np.mean(recent_ep))
            goal_pct = float(np.mean(recent_goals)) * 100.0

            if is_rondo_scenario:
                retention_threshold = 10.0
                recent_retention = [1 if r > retention_threshold else 0 for r in recent_ep]
                goal_pct = float(np.mean(recent_retention)) * 100.0

            trend_snapshots.append((total_steps_elapsed, len(episode_rewards), mean_rew, goal_pct))

            # Update best rolling checkpoint
            if goal_pct > best_rolling_goal_rate:
                best_rolling_goal_rate = goal_pct
                best_rolling_checkpoint_step = total_steps_elapsed
                best_rolling_ckpt_name = os.path.join(
                    model_dir, f"mappo_{scenario}_seed{seed}_{shaped_suffix}_rolling_best.pt"
                )
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "actor_opt": actor_opt.state_dict(),
                        "critic_opt": critic_opt.state_dict(),
                        "obs_dim": obs_dim,
                        "global_state_dim": global_state_dim,
                        "action_dim": action_dim,
                        "timesteps": total_steps_elapsed,
                    },
                    best_rolling_ckpt_name,
                )
                best_rolling_checkpoint_path = best_rolling_ckpt_name
                print(
                    f"   [OK] New best rolling checkpoint saved: {best_rolling_ckpt_name} "
                    f"(rolling {'possession retention' if is_rondo_scenario else 'goal rate'}: {best_rolling_goal_rate:.1f}% at step {best_rolling_checkpoint_step})",
                    flush=True,
                )

            metric_label = "Possession Retention" if is_rondo_scenario else "Goal Rate"
            print(
                f"   [Step {total_steps_elapsed:7d} / {timesteps}] Update {update_idx:4d}/{n_updates} | "
                f"Completed Episodes: {len(episode_rewards):4d} | "
                f"Rolling Reward (last 50): {mean_rew:+.4f} | "
                f"{metric_label}: {goal_pct:5.1f}% | "
                f"Val Loss: {metrics['value_loss']:.5f} | "
                f"Entropy: {metrics['entropy']:.4f} | "
                f"EntropyCoef: {entropy_coef:.5f} | LR: {float(actor_opt.param_groups[0]['lr']):.6f}",
                flush=True,
            )

        # Periodic validation every validation_interval steps
        if total_steps_elapsed - last_validation_step >= validation_interval:
            last_validation_step = total_steps_elapsed
            print(f"\n   [Validation] Running deterministic {validation_episodes}-episode eval at step {total_steps_elapsed}...")
            try:
                # Save temporary checkpoint for validation
                temp_ckpt_path = os.path.join(model_dir, f"mappo_{scenario}_seed{seed}_{shaped_suffix}_temp.pt")
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "actor_opt": actor_opt.state_dict(),
                        "critic_opt": critic_opt.state_dict(),
                        "obs_dim": obs_dim,
                        "global_state_dim": global_state_dim,
                        "action_dim": action_dim,
                        "timesteps": total_steps_elapsed,
                    },
                    temp_ckpt_path,
                )
                eval_row = evaluate_checkpoint_progress(
                    checkpoint_path=temp_ckpt_path,
                    scenario=scenario,
                    algorithm="MAPPO",
                    step=total_steps_elapsed,
                    learning_rate=float(actor_opt.param_groups[0]["lr"]),
                    num_episodes=validation_episodes,
                    deterministic=True,
                )
                val_goal_rate = float(eval_row.get("goal_rate_pct", 0.0))
                val_pass_acc = float(eval_row.get("pass_completion_rate_pct", 0.0))
                writer.add_scalar("Validation/GoalRate", val_goal_rate, total_steps_elapsed)
                writer.add_scalar("Validation/PassCompletionRate", val_pass_acc, total_steps_elapsed)
                print(f"   [Validation] Goal Rate: {val_goal_rate:.1f}% | Pass Completion Rate: {val_pass_acc:.1f}%")
            except Exception as e:
                print(f"   [Validation Notice] {e}")

        # 50k-interval checkpoint saving
        if total_steps_elapsed - last_checkpoint_step >= 50_000:
            last_checkpoint_step = total_steps_elapsed
            milestone_ckpt_name = os.path.join(model_dir, f"mappo_{scenario}_seed{seed}_{shaped_suffix}_{total_steps_elapsed}.pt")
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "actor_opt": actor_opt.state_dict(),
                    "critic_opt": critic_opt.state_dict(),
                    "obs_dim": obs_dim,
                    "global_state_dim": global_state_dim,
                    "action_dim": action_dim,
                    "timesteps": total_steps_elapsed,
                },
                milestone_ckpt_name,
            )
            try:
                eval_row = evaluate_checkpoint_progress(
                    checkpoint_path=milestone_ckpt_name,
                    scenario=scenario,
                    algorithm="MAPPO",
                    step=total_steps_elapsed,
                    learning_rate=float(actor_opt.param_groups[0]["lr"]),
                    num_episodes=30,
                    deterministic=True,
                )
                milestone_goal_rate = float(eval_row.get("goal_rate_pct", 0.0))
                milestone_pass_acc = float(eval_row.get("pass_completion_rate_pct", 0.0))
                milestone_eval_id = eval_row.get("evaluation_id", "N/A")
                milestone_ckpt_sha = eval_row.get("checkpoint_sha256", "N/A")
                # Accept if goal rate improves OR pass completion increases significantly without sacrificing score
                if not _has_best_deterministic or milestone_goal_rate > best_deterministic_goal_rate + 2.0 or milestone_pass_acc > 5.0:
                    best_deterministic_goal_rate = milestone_goal_rate
                    _has_best_deterministic = True
                    best_deterministic_checkpoint_step = total_steps_elapsed
                    torch.save(
                        {
                            "actor": actor.state_dict(),
                            "critic": critic.state_dict(),
                            "actor_opt": actor_opt.state_dict(),
                            "critic_opt": critic_opt.state_dict(),
                            "obs_dim": obs_dim,
                            "global_state_dim": global_state_dim,
                            "action_dim": action_dim,
                            "timesteps": total_steps_elapsed,
                        },
                        best_model_path,
                    )
                    best_deterministic_checkpoint_path = best_model_path
                    print(
                        f"   [OK] New best shaped checkpoint saved: {best_model_path} "
                        f"(eval goal rate: {best_deterministic_goal_rate:.1f}%, pass acc: {milestone_pass_acc:.1f}%, "
                        f"evaluation_id={milestone_eval_id}, checkpoint_sha256={milestone_ckpt_sha})",
                        flush=True,
                    )
            except Exception as e:
                print(f"[Notice] MAPPO milestone eval notice: {e}")

    duration = time.time() - start_time
    steps_this_run = total_steps_elapsed - total_steps_elapsed_at_start
    fps = steps_this_run / max(0.001, duration)
    print(f"\n[OK] MAPPO Shaped Training completed in {duration:.2f}s ({fps:.1f} steps/sec)", flush=True)

    # 4. Save latest checkpoint
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_opt": actor_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(),
            "obs_dim": obs_dim,
            "global_state_dim": global_state_dim,
            "action_dim": action_dim,
            "timesteps": total_steps_elapsed,
        },
        latest_model_path,
    )
    print(f"[OK] Latest shaped checkpoint saved: {latest_model_path}", flush=True)

    # End-of-run milestone evaluation
    try:
        eval_row = evaluate_checkpoint_progress(
            checkpoint_path=latest_model_path,
            scenario=scenario,
            algorithm="MAPPO",
            step=total_steps_elapsed,
            learning_rate=float(actor_opt.param_groups[0]["lr"]),
            num_episodes=50,
            deterministic=True,
        )
        end_goal_rate = float(eval_row.get("goal_rate_pct", 0.0))
        end_pass_acc = float(eval_row.get("pass_completion_rate_pct", 0.0))
        end_eval_id = eval_row.get("evaluation_id", "N/A")
        end_ckpt_sha = eval_row.get("checkpoint_sha256", "N/A")
        if not _has_best_deterministic or end_goal_rate > best_deterministic_goal_rate + 2.0 or end_pass_acc > 5.0:
            best_deterministic_goal_rate = end_goal_rate
            best_deterministic_checkpoint_step = total_steps_elapsed
            _has_best_deterministic = True
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "actor_opt": actor_opt.state_dict(),
                    "critic_opt": critic_opt.state_dict(),
                    "obs_dim": obs_dim,
                    "global_state_dim": global_state_dim,
                    "action_dim": action_dim,
                    "timesteps": total_steps_elapsed,
                },
                best_model_path,
            )
            best_deterministic_checkpoint_path = best_model_path
            print(
                f"[OK] End-of-run eval updated best shaped checkpoint: {best_model_path} "
                f"(goal rate: {best_deterministic_goal_rate:.1f}%, pass acc: {end_pass_acc:.1f}%, "
                f"evaluation_id={end_eval_id}, checkpoint_sha256={end_ckpt_sha})",
                flush=True,
            )
    except Exception as e:
        print(f"[Notice] End-of-run MAPPO eval notice: {e}")

    # Preserve best deterministic checkpoint as durable artifact
    if best_deterministic_checkpoint_path and os.path.exists(best_deterministic_checkpoint_path):
        import shutil
        shutil.copy2(best_deterministic_checkpoint_path, latest_model_path)
        print(
            f"[OK] Best deterministic shaped checkpoint preserved as durable artifact: {latest_model_path} "
            f"(from {best_deterministic_checkpoint_path}, best deterministic goal rate: {best_deterministic_goal_rate:.1f}% at step {best_deterministic_checkpoint_step})",
            flush=True,
        )

    # Persist trend snapshots
    if trend_snapshots:
        persist_trend_snapshots(trend_snapshots, algorithm="MAPPO", scenario=scenario, seed=seed)

    # 5. Print Training Reward & Performance Trend Summary
    print("\n5. Training Reward & Performance Trend Summary:", flush=True)
    if trend_snapshots:
        metric_label = "Rolling Possession Retention" if is_rondo_scenario else "Rolling Goal Rate"
        print(f"   {'Timestep':>9} | {'Episodes':>8} | {'Rolling Reward':>15} | {metric_label:>25}", flush=True)
        print(f"   {'-'*9}-+-{'-'*8}-+-{'-'*15}-+-{'-'*25}", flush=True)
        for step, num_eps, rew, goal_rt in trend_snapshots:
            print(f"   {step:9d} | {num_eps:8d} | {rew:+15.4f} | {goal_rt:25.1f}%", flush=True)
    else:
        total_eps = len(episode_rewards)
        overall_rew = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        overall_goals = float(np.mean(episode_goals)) * 100.0 if episode_goals else 0.0
        print(f"   Total Episodes Completed: {total_eps}", flush=True)
        print(f"   Overall Mean Reward: {overall_rew:+.4f}", flush=True)
        if is_rondo_scenario:
            retention_threshold = 10.0
            recent_retention = [1 if r > retention_threshold else 0 for r in episode_rewards[-50:]]
            retention_pct = float(np.mean(recent_retention)) * 100.0 if recent_retention else 0.0
            print(f"   Possession Retention Rate (last 50): {retention_pct:.1f}%", flush=True)
        else:
            print(f"   Overall Goal Rate: {overall_goals:.1f}%", flush=True)

    best_metric_label = "possession retention" if is_rondo_scenario else "goal rate"
    print(
        f"\n   Best rolling {best_metric_label}: {best_rolling_goal_rate:.1f}% at step {best_rolling_checkpoint_step} "
        f"| Best deterministic {best_metric_label}: {best_deterministic_goal_rate:.1f}% at step {best_deterministic_checkpoint_step} (exported)",
        flush=True,
    )

    # 6. Sampled Loss Metrics Progression Across Training
    print("\n6. Sampled Loss & Diagnostic Metrics Progression Across Training:", flush=True)
    print(f"   {'Update':>7} | {'Timestep':>9} | {'Policy Loss':>12} | {'Value Loss':>12} | {'Entropy':>10} | {'Approx KL':>11}", flush=True)
    print(f"   {'-'*7}-+-{'-'*9}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*11}", flush=True)

    sample_indices = np.linspace(0, len(loss_history) - 1, num=min(10, len(loss_history)), dtype=int)
    for idx in sample_indices:
        m = loss_history[idx]
        print(
            f"   {m['update']:7d} | {m['step']:9d} | {m['policy_loss']:+12.6f} | "
            f"{m['value_loss']:12.6f} | {m['entropy']:10.4f} | {m['approx_kl']:+11.6f}",
            flush=True,
        )

    # 7. Checkpoint Reload Smoke Test
    print("\n7. Verifying Checkpoint Reload Smoke Test...", flush=True)
    loaded_checkpoint = torch.load(latest_model_path, map_location="cpu")
    eval_actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    eval_critic = CentralizedCritic(obs_dim=obs_dim, hidden=64, mode="pool")

    eval_actor.load_state_dict(loaded_checkpoint["actor"])
    eval_critic.load_state_dict(loaded_checkpoint["critic"])
    eval_actor.eval()
    eval_critic.eval()

    # Test dummy forward pass
    dummy_obs = torch.zeros((num_agents, obs_dim), dtype=torch.float32)
    dummy_dist = eval_actor(dummy_obs)
    dummy_act = dummy_dist.sample()
    dummy_joint = torch.zeros((1, num_agents, obs_dim), dtype=torch.float32)
    dummy_val = eval_critic(dummy_joint)

    assert dummy_act.shape == (num_agents,), f"Dummy action shape mismatch: {dummy_act.shape}"
    assert dummy_val.shape == (1,), f"Dummy value shape mismatch: {dummy_val.shape}"
    print(f"   [OK] Checkpoint loaded and forward pass executed cleanly (action shape: {dummy_act.shape}, val: {dummy_val.item():.4f}).", flush=True)

    writer.close()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAPPO with Cooperative Reward Shaping on GMN-Football-3")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper", help="Scenario name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-steps", type=int, default=500000, help="Total environment timesteps to train")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume from")
    parser.add_argument("--enable-reward-shaping", action="store_true", default=True, help="Enable cooperative reward shaping")
    parser.add_argument("--disable-reward-shaping", action="store_true", help="Disable cooperative reward shaping")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--validation-interval", type=int, default=25000, help="Validation interval in steps")
    parser.add_argument("--validation-episodes", type=int, default=50, help="Number of episodes per validation run")
    args = parser.parse_args()

    enable_shaping = args.enable_reward_shaping and not args.disable_reward_shaping

    run_mappo_shaped_training(
        timesteps=args.total_steps,
        scenario=args.scenario,
        seed=args.seed,
        resume_path=args.resume,
        enable_reward_shaping=enable_shaping,
        log_dir=args.log_dir,
        validation_interval=args.validation_interval,
        validation_episodes=args.validation_episodes,
    )
