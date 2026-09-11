"""
GMN-Football-3 -- Multi-Agent PPO (MAPPO) with Centralized Critic Training
Trains a parameter-shared policy network and centralized team critic on cooperative football scenarios.

Hyperparameters:
- Timesteps: 200,000 (apples-to-apples comparison against IPPO baseline)
- Rollout length (n_steps): 256
- Mini-batch size: 256
- PPO Epochs: 4
- Learning rate: 3e-4 (Adam)
- Discount (gamma): 0.99
- GAE lambda: 0.95
- PPO Clip range: 0.15
- Value coefficient: 0.5
- Entropy coefficient: 0.01
"""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time
from typing import List, Tuple, Dict, Any

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from training.gmn_pettingzoo import GMNMultiAgentEnv, OBSERVATION_DIM, ACTION_SPACE_SIZE
from training.mappo_networks import SharedActor, CentralizedCritic
from training.mappo_rollout import collect_rollout, collect_rollout_parallel, collect_rollout_batched, compute_gae
from training.mappo_update import ppo_update
from training.eval_progress import evaluate_checkpoint_progress, persist_trend_snapshots
from training.curriculum_scheduler import CurriculumScheduler, CURRICULUM_STAGES
from training.checkpoint_contract import create_experiment_manifest, compute_file_sha256


def run_mappo_training(
    timesteps: int = 200000,
    checkpoint_name: str = None,
    scenario: str = "academy_3_vs_1_with_keeper",
    seed: int = 42,
    resume_path: str = None,
    n_envs: int = 1,
    self_play: bool = False,
    opponent_difficulty: str = "medium",
    opponent_strategy: str = "uniform",
    curriculum: bool = False,
    curriculum_state_path: str = None,
    curriculum_window_size: int = 100,
    curriculum_promote_threshold: float = 0.6,
    curriculum_demote_threshold: float = 0.1,
    curriculum_min_episodes: int = 200,
) -> bool:
    is_smoke_test = timesteps < 50000
    if checkpoint_name is None:
        suffix = "smoke" if is_smoke_test else "quarantine"
        checkpoint_name = f"mappo_{scenario}_seed{seed}_{suffix}.pt"
    else:
        # Enforce seed-based naming to prevent cross-seed overwrites.
        if f"seed{seed}" not in checkpoint_name:
            raise ValueError(
                f"Checkpoint name '{checkpoint_name}' does not contain seed identifier 'seed{seed}'. "
                "Use None or a seed-unique name to prevent overwrites across parallel runs."
            )

    # Quarantine protocol: the final training checkpoint is saved with a quarantine
    # suffix first. Only after end-of-run evaluation confirms it is the best
    # deterministic checkpoint is it promoted to the durable _clean.pt artifact.
    clean_checkpoint_name = checkpoint_name.replace("_quarantine.pt", "_clean.pt").replace("_smoke.pt", "_clean.pt")
    if clean_checkpoint_name == checkpoint_name:
        clean_checkpoint_name = f"mappo_{scenario}_seed{seed}_clean.pt"

    print("==================================================")
    print(f"GMN FOOTBALL -- MULTI-AGENT PPO (MAPPO) {'SMOKE TEST' if is_smoke_test else 'REAL TRAINING RUN'}")
    print(f"Target Scenario: {scenario} | Timesteps: {timesteps}")
    if resume_path:
        print(f"Resuming From Checkpoint: {resume_path}")
    print("Architecture: SharedActor (Mlp 64x64) + CentralizedCritic (Global State / Set Pooling -> 1)")
    print(f"Checkpoint Output (quarantine): models/{checkpoint_name}")
    print(f"Checkpoint Output (clean, post-eval): models/{clean_checkpoint_name}")
    print("==================================================")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Continuous forensic logging: per-episode JSONL trace for the entire run.
    _forensic_dir = os.path.join(os.path.dirname(__file__), "results", "forensics")
    os.makedirs(_forensic_dir, exist_ok=True)

    # H4: Programmatic experiment manifest generation.
    _runs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs"))
    _experiment_name = f"mappo_{scenario}_seed{seed}_{'smoke' if is_smoke_test else 'full'}"
    _run_dir = os.path.join(_runs_dir, _experiment_name)
    os.makedirs(_run_dir, exist_ok=True)
    _manifest_path = os.path.join(_run_dir, "experiment_manifest.json")
    try:
        _git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        _git_commit = "unavailable"
    _quarantine_path = os.path.join(models_dir, checkpoint_name)
    _manifest = create_experiment_manifest(
        algorithm="MAPPO",
        scenario=scenario,
        training_steps=timesteps,
        seed=seed,
        checkpoint_path=_quarantine_path,
        hyperparameters={
            "rollout_length": 256,
            "minibatch_size": 256,
            "ppo_epochs": 4,
            "learning_rate": 3e-4,
            "clip_range": 0.15,
            "value_coef": 0.5,
            "entropy_coef_start": 0.01,
            "entropy_coef_end": 0.005,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "actor_hidden": 64,
            "critic_hidden": 64,
            "obs_dim": OBSERVATION_DIM,
            "action_dim": ACTION_SPACE_SIZE,
        },
        git_commit=_git_commit,
    )
    _manifest["experiment"] = _experiment_name
    _manifest["output_dir"] = _run_dir
    _manifest["checkpoint_names"] = [
        f"mappo_{scenario}_seed{seed}_50k.pt",
        f"mappo_{scenario}_seed{seed}_100k.pt",
        f"mappo_{scenario}_seed{seed}_150k.pt",
        f"mappo_{scenario}_seed{seed}_200k.pt",
    ]
    with open(_manifest_path, "w", encoding="utf-8") as _mf:
        json.dump(_manifest, _mf, indent=2)
    print(f"   [H4] Experiment manifest written to: {_manifest_path}")

    print("\n1. Initializing Multi-Agent Environment & MAPPO Networks...")
    # Self-play / opponent pool (Task: opponent generalization). When enabled,
    # a pool of rule-based difficulties (and periodic policy snapshots) is
    # sampled per episode; snapshots of the current policy are saved to
    # training/models/opponent_pool/ for future bridge-side execution.
    opponent_pool = None
    if self_play:
        from training.opponent_pool import OpponentPool

        opponent_pool = OpponentPool(
            strategy=opponent_strategy,
            seed=seed,
            snapshot_dir=os.path.join(models_dir, "opponent_pool"),
        )
        print(f"   Self-play enabled: {opponent_pool.describe()}")

    # Parallel mode: n_envs > 1 uses a single batched bridge with pooled engines
    # instead of N separate bridge processes. n_envs=1 keeps the legacy path.
    if n_envs > 1:
        env = GMNMultiAgentEnv(
            scenario=scenario,
            auto_start_bridge=True,
            batch_size=n_envs,
            opponent_difficulty=opponent_difficulty,
            opponent_pool=opponent_pool,
        )
        envs = [env]
    else:
        envs = []
        for i in range(max(1, n_envs)):
            envs.append(
                GMNMultiAgentEnv(
                    scenario=scenario,
                    auto_start_bridge=True,
                    port=5050 + i if i > 0 else None,
                    opponent_difficulty=opponent_difficulty,
                    opponent_pool=opponent_pool,
                )
            )
        env = envs[0]

    # Continuous forensic logging: per-episode JSONL trace for the entire run.
    _forensic_run_id = time.strftime("%Y%m%dT%H%M%S")
    _terminal_jsonl_path = os.path.join(
        _forensic_dir, f"terminal_tick_trace_{scenario}_seed{seed}_{_forensic_run_id}.jsonl"
    )
    # Expose on env so existing end-of-run logging can find it.
    env._last_terminal_jsonl = _terminal_jsonl_path

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

    # ------------------------------------------------------------------
    # Curriculum scheduler (performance-threshold-driven promotion/demotion)
    # ------------------------------------------------------------------
    if curriculum:
        default_start = CURRICULUM_STAGES[0]
        effective_start = scenario or default_start
        default_curriculum_path = os.path.join(
            logs_dir, f"curriculum_state_seed{seed}.json"
        )
        curriculum_state_path = curriculum_state_path or default_curriculum_path
        scheduler = CurriculumScheduler.load(curriculum_state_path) or CurriculumScheduler(
            stages=CURRICULUM_STAGES,
            window_size=curriculum_window_size,
            promote_threshold=curriculum_promote_threshold,
            demote_threshold=curriculum_demote_threshold,
            min_episodes_before_promotion=curriculum_min_episodes,
        )
        # Always start from the scheduler's current stage (which may differ from
        # the CLI --scenario if resuming a crashed run).
        scenario = scheduler.current_stage
        print(
            f"\n   [Curriculum] Enabled. Starting stage: {scenario} "
            f"(episodes so far: {scheduler.total_episodes})"
        )
    else:
        scheduler = None

    n_steps = 256
    effective_steps_per_update = n_steps * max(1, n_envs)
    remaining_timesteps = max(0, timesteps - total_steps_elapsed)
    n_updates = remaining_timesteps // effective_steps_per_update
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_opt, T_max=max(1, n_updates), eta_min=3e-5)
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(critic_opt, T_max=max(1, n_updates), eta_min=3e-5)
    check_freq_steps = 1000 if is_smoke_test else 10000

    print(f"\n2. Configuration:")
    print(f"   Total Updates: {n_updates} ({n_steps} steps per rollout)")
    print(f"   Remaining Timesteps: {n_updates * n_steps} (Total Target: {timesteps})")
    print(f"   PPO Epochs: 4 | Mini-batch: 256 | LR: 3e-4 (cosine anneal to 3e-5) | Clip: 0.15")
    print(f"   GAE: gamma=0.99, lambda=0.95 | Value Coef: 0.5 | Entropy Coef: 0.01")

    # Metrics tracking
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_goals: List[int] = []
    trend_snapshots: List[Tuple[int, int, float, float, Dict[str, Any]]] = []  # (step, num_eps, mean_rew, goal_rate, event_counters)
    loss_history: List[Dict[str, Any]] = []

    is_rondo_scenario = scenario == "academy_rondo_4v1"

    # Best-checkpoint selection: track rolling (stochastic) and deterministic eval goal rates separately.
    # Rolling stats are cheap/frequent but noisy; deterministic evals match deployed browser behavior.
    best_rolling_goal_rate: float = -1.0
    best_rolling_checkpoint_step: int = 0
    best_rolling_checkpoint_path: str = ""
    best_deterministic_goal_rate: float = -1.0
    best_deterministic_checkpoint_step: int = 0
    best_deterministic_checkpoint_path: str = ""
    _has_best_deterministic: bool = False

    last_check_step = total_steps_elapsed
    last_checkpoint_step = (total_steps_elapsed // 50_000) * 50_000
    last_logged_stage = scheduler.current_stage if scheduler else scenario

    print(f"\n3. Starting MAPPO Training for {remaining_timesteps} steps...")
    start_time = time.time()
    total_steps_elapsed_at_start = total_steps_elapsed

    for update_idx in range(start_update, start_update + n_updates):
        # Curriculum: evaluate and possibly switch scenario BEFORE this rollout.
        if scheduler is not None:
            new_stage = scheduler.evaluate_and_step()
            if new_stage != last_logged_stage:
                transition = next(
                    (h for h in reversed(scheduler.history) if h["to"] == new_stage),
                    None,
                )
                direction = transition["type"] if transition else "change"
                rate = transition["success_rate"] if transition else 0.0
                print(
                    f"\n   [Curriculum] {direction.capitalize()} → {new_stage} "
                    f"(episode {scheduler.total_episodes}, window success rate: {rate:.1%})\n",
                    flush=True,
                )
                # Save a stage-transition checkpoint so the exact policy at the
                # boundary is recoverable later.
                transition_ckpt = os.path.join(
                    models_dir,
                    f"mappo_curriculum_{direction}_to_{new_stage}_step{total_steps_elapsed}.pt",
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
                        "curriculum_stage": new_stage,
                        "curriculum_history": scheduler.history,
                    },
                    transition_ckpt,
                )
                print(
                    f"   [Curriculum] Transition checkpoint saved: {transition_ckpt}",
                    flush=True,
                )
                # Switch env scenario and reset so the new stage starts fresh.
                env.set_scenario(new_stage)
                env.reset()
                last_logged_stage = new_stage
                # Persist scheduler state after every transition.
                scheduler.save(curriculum_state_path)

        # 1. Collect Rollout
        if n_envs > 1:
            buffer = collect_rollout_batched(
                env, actor, critic, num_steps=n_steps, batch_size=n_envs,
                terminal_jsonl_path=_terminal_jsonl_path,
            )
            total_steps_elapsed += n_steps * n_envs
        else:
            buffer = collect_rollout(
                env, actor, critic, num_steps=n_steps,
                terminal_jsonl_path=_terminal_jsonl_path,
            )
            total_steps_elapsed += n_steps

        # Record completed episodes
        for ep_info in buffer.get("completed_episodes", []):
            episode_rewards.append(ep_info["reward"])
            episode_lengths.append(ep_info["length"])
            episode_goals.append(ep_info["goal"])
            if scheduler is not None:
                scheduler.record_result(ep_info.get("success", bool(ep_info["goal"])))

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
        # Linear entropy schedule: decays from 0.01 to 0.005 over the full training run.
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

        # Diagnostics check for numerical instability
        if np.isnan(metrics["policy_loss"]) or np.isnan(metrics["value_loss"]):
            raise RuntimeError(f"NaN loss detected at update {update_idx}: {metrics}")
        if np.isnan(metrics["approx_kl"]) or abs(metrics["approx_kl"]) > 1.5:
            print(f"   [WARNING] High KL divergence at update {update_idx}: approx_kl = {metrics['approx_kl']:.4f}")

        # Periodic snapshot logging (matching train_ippo.py cadence)
        if total_steps_elapsed - last_check_step >= check_freq_steps:
            last_check_step = total_steps_elapsed

            # Periodic self-play snapshot: add the current policy to the pool.
            if opponent_pool is not None:
                snap = opponent_pool.maybe_snapshot(actor, total_steps_elapsed)
                if snap:
                    print(f"   [SelfPlay] Policy snapshot added to opponent pool: {snap}", flush=True)

            recent_ep = episode_rewards[-50:] if episode_rewards else [0.0]
            recent_goals = episode_goals[-50:] if episode_goals else [0]
            mean_rew = float(np.mean(recent_ep))

            # Phase 4: FORENSIC_TRAINING_METRIC capture
            if getattr(env, "_forensic_debug", False):
                import sys as _sys
                last_50_sum = sum(recent_ep)
                last_50_count = len(recent_ep)
                print(
                    f"[FORENSIC_TRAINING_METRIC]\n"
                    f"global_step={total_steps_elapsed}\n"
                    f"total_episodes={len(episode_rewards)}\n"
                    f"last_50_count={last_50_count}\n"
                    f"last_50_sum={last_50_sum:.6f}\n"
                    f"rolling_mean={mean_rew:.6f}\n"
                    f"min_last_50={min(recent_ep):.6f}\n"
                    f"max_last_50={max(recent_ep):.6f}\n"
                    f"last_50_values={recent_ep}\n"
                    f"[END_FORENSIC_TRAINING_METRIC]",
                    flush=True,
                    file=_sys.stdout,
                )

            # Sanity bound: per-tick reward is bounded [-1.0, +2.17], and the
            # longest scenario is 1800 ticks, so any single episode reward
            # outside [-2000, +4000] indicates a bug in reward accumulation
            # (e.g., ep_rew not resetting, or logging math corruption). Flag
            # it loudly and dump the offending values for post-mortem.
            _EP_REWARD_LO, _EP_REWARD_HI = -2000.0, 4000.0
            for _i, _r in enumerate(recent_ep):
                if not (_EP_REWARD_LO <= _r <= _EP_REWARD_HI):
                    import sys as _sys
                    print(
                        f"   [WARN] episode reward #{_i} = {_r:.2f} outside plausible "
                        f"range [{_EP_REWARD_LO}, {_EP_REWARD_HI}] — reward accumulation "
                        f"bug suspected. recent_ep={recent_ep}",
                        flush=True,
                        file=_sys.stderr,
                    )
                    break
            goal_pct = float(np.mean(recent_goals)) * 100.0

            # For rondo, report possession retention rate instead of goal rate.
            # We approximate retention by episodes whose reward exceeds a
            # "minimum viable possession" threshold derived from the dense reward
            # structure (~0.01/tick for full 20s = ~12.0 baseline).
            if is_rondo_scenario:
                retention_threshold = 10.0
                recent_retention = [1 if r > retention_threshold else 0 for r in recent_ep]
                goal_pct = float(np.mean(recent_retention)) * 100.0

            # Log cumulative reward-shaping event counters alongside trend snapshots
            # so we can confirm event-code fixes reach live training, not just tests.
            _shaper_diag = {}
            _shaper_cum = {}
            if getattr(env, "reward_shaper", None) is not None:
                try:
                    _shaper_diag = env.reward_shaper.get_diagnostics()
                    _shaper_cum = env.reward_shaper.get_cumulative_diagnostics()
                except Exception:
                    pass
            trend_snapshots.append((total_steps_elapsed, len(episode_rewards), mean_rew, goal_pct, _shaper_cum))
            if _shaper_cum:
                print(
                    f"   [EVENT COUNTERS] total_pass_completed={_shaper_cum.get('total_pass_completed_count', 0)} "
                    f"| total_goals={_shaper_cum.get('total_goal_scored_count', 0)} "
                    f"| total_turnovers={_shaper_cum.get('total_turnover_conceded_count', 0)}",
                    flush=True,
                )

            # Update best rolling checkpoint based on stochastic rollout goal rate (same metric as the console log).
            # This is a fast, cheap signal for monitoring/early-stopping, but it does NOT reflect deterministic
            # deployed behavior and therefore should NOT be used for the exported ONNX policy.
            if goal_pct > best_rolling_goal_rate:
                best_rolling_goal_rate = goal_pct
                best_rolling_checkpoint_step = total_steps_elapsed
                best_rolling_ckpt_name = os.path.join(
                    models_dir, f"mappo_{scenario}_seed{seed}_rolling_best.pt"
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

        # 50k-interval checkpoint saving and milestone evaluation (cheaper, more frequent deterministic eval)
        if total_steps_elapsed - last_checkpoint_step >= 50_000:
            last_checkpoint_step = total_steps_elapsed
            milestone_ckpt_name = f"mappo_{scenario}_seed{seed}_{total_steps_elapsed}.pt"
            milestone_ckpt_path = os.path.join(models_dir, milestone_ckpt_name)
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
                milestone_ckpt_path,
            )
            try:
                eval_row = evaluate_checkpoint_progress(
                    checkpoint_path=milestone_ckpt_path,
                    scenario=scenario,
                    algorithm="MAPPO",
                    step=total_steps_elapsed,
                    learning_rate=float(actor_opt.param_groups[0]["lr"]),
                    num_episodes=30,
                    deterministic=True,
                )
                _raw_milestone_goal_rate = eval_row.get("goal_rate_pct", 0.0)
                milestone_goal_rate = float(_raw_milestone_goal_rate) if not isinstance(_raw_milestone_goal_rate, dict) else 0.0
                milestone_eval_id = eval_row.get("evaluation_id", "N/A")
                milestone_ckpt_sha = eval_row.get("checkpoint_sha256", "N/A")
                # First milestone is accepted unconditionally; subsequent milestones must beat
                # the current best by >= 2 percentage points to replace the exported checkpoint.
                if not _has_best_deterministic or milestone_goal_rate > best_deterministic_goal_rate + 2.0:
                    best_deterministic_goal_rate = milestone_goal_rate
                    _has_best_deterministic = True
                    best_deterministic_checkpoint_step = total_steps_elapsed
                    best_ckpt_name = os.path.join(
                        models_dir, f"mappo_{scenario}_seed{seed}_best.pt"
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
                        best_ckpt_name,
                    )
                    best_deterministic_checkpoint_path = best_ckpt_name
                    print(
                        f"   [OK] New best deterministic checkpoint saved: {best_ckpt_name} "
                        f"(eval goal rate: {best_deterministic_goal_rate:.1f}%, "
                        f"evaluation_id={milestone_eval_id}, checkpoint_sha256={milestone_ckpt_sha})",
                        flush=True,
                    )
            except Exception as e:
                import traceback as _traceback
                print(f"[Notice] MAPPO milestone eval notice: {e}")
                print("[DEBUG] Full traceback:\n" + _traceback.format_exc(), flush=True)

    duration = time.time() - start_time
    steps_this_run = total_steps_elapsed - total_steps_elapsed_at_start
    fps = steps_this_run / max(0.001, duration)
    print(f"\n[OK] MAPPO Training completed in {duration:.2f}s ({fps:.1f} steps/sec)", flush=True)

    # 4. Save model checkpoint
    checkpoint_path = os.path.join(models_dir, checkpoint_name)
    print(f"\n4. Saving MAPPO Model Checkpoint to: {checkpoint_path}...", flush=True)
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
        checkpoint_path,
    )
    print(
        f"[OK] Checkpoint saved successfully. File exists: {os.path.exists(checkpoint_path)} "
        f"(size: {os.path.getsize(checkpoint_path)} bytes)",
        flush=True,
    )

    # End-of-run milestone evaluation — run BEFORE preservation so this eval can become the best.
    try:
        eval_row = evaluate_checkpoint_progress(
            checkpoint_path=checkpoint_path,
            scenario=scenario,
            algorithm="MAPPO",
            step=total_steps_elapsed,
            learning_rate=float(actor_opt.param_groups[0]["lr"]),
            num_episodes=50,
            deterministic=True,
        )
        _raw_goal_rate = eval_row.get("goal_rate_pct", 0.0)
        end_goal_rate = float(_raw_goal_rate) if not isinstance(_raw_goal_rate, dict) else 0.0
        end_eval_id = eval_row.get("evaluation_id", "N/A")
        end_ckpt_sha = eval_row.get("checkpoint_sha256", "N/A")
        if not _has_best_deterministic or end_goal_rate > best_deterministic_goal_rate + 2.0:
            best_deterministic_goal_rate = end_goal_rate
            best_deterministic_checkpoint_step = total_steps_elapsed
            _has_best_deterministic = True
            best_ckpt_name = os.path.join(
                models_dir, f"mappo_{scenario}_seed{seed}_best.pt"
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
                best_ckpt_name,
            )
            best_deterministic_checkpoint_path = best_ckpt_name
            print(
                f"[OK] End-of-run eval updated best deterministic checkpoint: {best_ckpt_name} "
                f"(goal rate: {best_deterministic_goal_rate:.1f}% at step {best_deterministic_checkpoint_step}, "
                f"evaluation_id={end_eval_id}, checkpoint_sha256={end_ckpt_sha})",
                flush=True,
            )
    except Exception as e:
        import traceback as _traceback
        print(f"[Notice] End-of-run MAPPO eval notice: {e}")
        print("[DEBUG] Full traceback:\n" + _traceback.format_exc(), flush=True)

    # Preserve best deterministic checkpoint as the durable exported artifact.
    # The deployed browser policy (TrainedPolicyAgent) runs deterministically, so the checkpoint
    # shipped to export_onnx.py / public/models must be selected from deterministic evals,
    # NOT from stochastic rollout stats. The rolling best is tracked separately for monitoring.
    clean_checkpoint_path = os.path.join(models_dir, clean_checkpoint_name)
    if best_deterministic_checkpoint_path and os.path.exists(best_deterministic_checkpoint_path):
        import shutil
        shutil.copy2(best_deterministic_checkpoint_path, clean_checkpoint_path)
        print(
            f"[OK] Best deterministic checkpoint preserved as durable artifact: {clean_checkpoint_path} "
            f"(from {best_deterministic_checkpoint_path}, best deterministic goal rate: {best_deterministic_goal_rate:.1f}% at step {best_deterministic_checkpoint_step})",
            flush=True,
        )

    # Persist trend snapshots
    if trend_snapshots:
        # Extract per-snapshot event counters that were captured at each checkpoint.
        _snapshot_event_counters = [snap[4] if len(snap) > 4 else {} for snap in trend_snapshots]
        persist_trend_snapshots(
            trend_snapshots,
            algorithm="MAPPO",
            scenario=scenario,
            seed=seed,
            event_counters_per_snapshot=_snapshot_event_counters,
        )

    # 5. Print Training Reward & Performance Trend Summary
    print("\n5. Training Reward & Performance Trend Summary:", flush=True)
    if trend_snapshots:
        metric_label = "Rolling Possession Retention" if is_rondo_scenario else "Rolling Goal Rate"
        print(f"   {'Timestep':>9} | {'Episodes':>8} | {'Rolling Reward':>15} | {metric_label:>25}", flush=True)
        print(f"   {'-'*9}-+-{'-'*8}-+-{'-'*15}-+-{'-'*25}", flush=True)
        for step, num_eps, rew, goal_rt, _ in trend_snapshots:
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
    loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
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

    # Phase 7: persist raw episode telemetry artifacts
    if getattr(env, "_forensic_debug", False):
        _forensic_dir = os.path.join(models_dir, "..", "results", "forensics")
        os.makedirs(_forensic_dir, exist_ok=True)
        _run_id = time.strftime("%Y%m%dT%H%M%S")
        _csv_path = os.path.join(_forensic_dir, f"episode_reward_trace_{scenario}_seed{seed}_{_run_id}.csv")
        _jsonl_path = os.path.join(_forensic_dir, f"terminal_tick_trace_{scenario}_seed{seed}_{_run_id}.jsonl")
        # Write CSV
        with open(_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "commit", "scenario", "seed", "collector", "episode_index",
                "global_step", "episode_length", "episode_reward",
                "terminal", "truncated", "terminal_raw_reward", "terminal_shared_reward",
                "score_left", "score_right", "event_code",
                "terminal_checkpoint_reward", "terminal_distance_to_goal",
                "min_step_reward", "max_step_reward", "sum_step_rewards", "reward_count",
                "reward_sum_validation",
            ])
            for i, ep_reward in enumerate(episode_rewards):
                ep_len = episode_lengths[i] if i < len(episode_lengths) else 0
                writer.writerow([
                    "", scenario, seed, "collect_rollout" if n_envs == 1 else "collect_rollout_batched",
                    i, 0, ep_len, ep_reward,
                    False, False, 0.0, 0.0,
                    0, 0, 0,
                    0.0, 0.0,
                    0.0, 0.0, ep_reward, 1,
                    ep_reward,
                ])
        print(f"   [FORENSIC] Episode trace CSV: {_csv_path}", flush=True)
        # Write terminal tick JSONL from collector if available
        terminal_jsonl = getattr(env, "_last_terminal_jsonl", None)
        if terminal_jsonl:
            print(f"   [FORENSIC] Terminal tick JSONL: {terminal_jsonl}", flush=True)

    # H4: Update experiment manifest with final results.
    try:
        _final_manifest_path = os.path.join(_run_dir, "experiment_manifest.json")
        if os.path.exists(_final_manifest_path):
            with open(_final_manifest_path, "r", encoding="utf-8") as _mf:
                _final_manifest = json.load(_mf)
        else:
            _final_manifest = {}
        _final_manifest.update({
            "experiment": _experiment_name,
            "status": "completed",
            "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
            "quarantine_checkpoint": checkpoint_path,
            "clean_checkpoint": clean_checkpoint_path,
            "quarantine_checkpoint_hash": compute_file_sha256(checkpoint_path) if os.path.exists(checkpoint_path) else "FILE_NOT_FOUND",
            "clean_checkpoint_hash": compute_file_sha256(clean_checkpoint_path) if os.path.exists(clean_checkpoint_path) else "FILE_NOT_FOUND",
            "best_deterministic_goal_rate": best_deterministic_goal_rate,
            "best_deterministic_step": best_deterministic_checkpoint_step,
            "best_rolling_goal_rate": best_rolling_goal_rate,
            "best_rolling_step": best_rolling_checkpoint_step,
            "total_episodes": len(episode_rewards),
            "total_timesteps": total_steps_elapsed,
        })
        with open(_final_manifest_path, "w", encoding="utf-8") as _mf:
            json.dump(_final_manifest, _mf, indent=2)
        print(f"   [H4] Experiment manifest updated: {_final_manifest_path}", flush=True)
    except Exception as _manifest_err:
        print(f"   [H4] Manifest update notice: {_manifest_err}", flush=True)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAPPO on GMN-Football-3")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total environment timesteps to train")
    parser.add_argument("--checkpoint-name", type=str, default=None, help="Output checkpoint filename (.pt)")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper", help="Scenario name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume from")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments/bridges (1 = legacy single-env mode)")
    parser.add_argument("--self-play", action="store_true", help="Enable the self-play opponent pool for the right team")
    parser.add_argument("--opponent-difficulty", type=str, default="medium", choices=["easy", "medium", "hard", "master"], help="Fixed rule-based opponent difficulty when self-play is disabled")
    parser.add_argument("--opponent-strategy", type=str, default="uniform", choices=["uniform", "cyclic", "elo"], help="Opponent pool selection strategy")
    parser.add_argument("--curriculum", action="store_true", help="Enable performance-threshold-driven curriculum promotion")
    parser.add_argument("--curriculum-state-path", type=str, default=None, help="Path to persist/load curriculum scheduler state (JSON)")
    parser.add_argument("--curriculum-window-size", type=int, default=100, help="Rolling window size for curriculum success rate")
    parser.add_argument("--curriculum-promote-threshold", type=float, default=0.6, help="Success rate threshold to promote to next stage")
    parser.add_argument("--curriculum-demote-threshold", type=float, default=0.1, help="Success rate threshold to demote after regression")
    parser.add_argument("--curriculum-min-episodes", type=int, default=200, help="Minimum episodes before promotion is allowed")
    args = parser.parse_args()

    if args.curriculum and args.scenario:
        # --scenario is used as the starting stage; scheduler drives subsequent changes.
        pass

    run_mappo_training(
        timesteps=args.timesteps,
        checkpoint_name=args.checkpoint_name,
        scenario=args.scenario,
        seed=args.seed,
        resume_path=args.resume,
        n_envs=args.n_envs,
        self_play=args.self_play,
        opponent_difficulty=args.opponent_difficulty,
        opponent_strategy=args.opponent_strategy,
        curriculum=args.curriculum,
        curriculum_state_path=args.curriculum_state_path,
        curriculum_window_size=args.curriculum_window_size,
        curriculum_promote_threshold=args.curriculum_promote_threshold,
        curriculum_demote_threshold=args.curriculum_demote_threshold,
        curriculum_min_episodes=args.curriculum_min_episodes,
    )
