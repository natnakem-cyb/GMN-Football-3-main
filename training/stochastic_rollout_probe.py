"""
GMN-Football-3 — Stochastic On-Policy Rollout Credit Probe
Collects stochastic rollouts from existing fresh-training checkpoints to gather
PASS/SHOT/GOAL events for credit-assignment analysis.

Measurement-only: no reward, GAE, mask, network, or horizon changes.
Uses production GAE semantics (dones = terminated only).
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor, CentralizedCritic
from training.mappo_rollout import unwrap_obs, unwrap_masks, _mask_matrix, compute_gae

ACTION_NAMES = [
    "IDLE", "LEFT", "RIGHT", "UP", "DOWN",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
    "LONG_PASS", "HIGH_PASS", "SHORT_PASS",
    "SHOT", "SPRINT",
    "RELEASE_DIRECTION", "RELEASE_SPRINT",
    "SLIDING", "DRIBBLE", "RELEASE_DRIBBLE",
]
TACKLE_ACTION = 16
SHOT_ACTIONS = {12}
PASS_ACTIONS = {9, 10, 11}
BALL_ACTION_IDS = frozenset((9, 10, 11, 12, 17))

# Production hyperparameters
GAMMA = 0.99
LAM = 0.95


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    obs_dim = ckpt.get("obs_dim", 127)
    action_dim = ckpt.get("action_dim", 19)
    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    critic = None
    if "critic" in ckpt:
        critic = CentralizedCritic(obs_dim=obs_dim, hidden=64)
        critic.load_state_dict(ckpt["critic"])
        critic.eval()

    timesteps = ckpt.get("timesteps", None)
    return actor, critic, obs_dim, action_dim, timesteps


def collect_stochastic_rollouts(
    checkpoint_path: str,
    scenario: str = "academy_3_vs_1_with_keeper_onball",
    num_episodes: int = 200,
    base_seed: int = 600000,
    bridge_port: int = 5050,
) -> Dict[str, Any]:
    """Collect stochastic rollouts from a checkpoint for credit analysis."""

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_sha256 = sha256_of(checkpoint_path)
    actor, critic, obs_dim, action_dim, ckpt_timesteps = load_checkpoint(checkpoint_path)

    print("=" * 70)
    print("STOCHASTIC ROLLOUT CREDIT PROBE")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"SHA256     : {checkpoint_sha256[:16]}...")
    print(f"Timesteps  : {ckpt_timesteps}")
    print(f"Scenario   : {scenario}")
    print(f"Episodes   : {num_episodes}")
    print(f"Base seed  : {base_seed}")
    print(f"Gamma={GAMMA}, Lambda={LAM}")
    print("=" * 70)

    env = GMNMultiAgentEnv(scenario=scenario, auto_start_bridge=True, port=bridge_port)
    controllable_agents = list(env.possible_agents)

    episodes_data: List[Dict[str, Any]] = []

    try:
        for ep in range(num_episodes):
            ep_seed = base_seed + ep * 1009
            obs_dict, _ = env.reset(seed=ep_seed)
            current_ep_masks = unwrap_masks(obs_dict)
            obs_dict = unwrap_obs(obs_dict)

            ep_tick_log: List[Dict[str, Any]] = []
            ep_reward = 0.0
            ep_length = 0
            goal_scored = 0
            tackle_actions = 0
            shot_actions = 0
            pass_actions = 0
            action_counts = [0] * 19
            last_info = {}
            episode_ground_truth = {}

            # Buffers for offline GAE
            ep_values = []
            ep_rewards = []
            ep_dones = []  # terminated only
            ep_truncated = []
            ep_terminated = []
            ep_local_obs = []
            ep_actions = []
            ep_logprobs = []
            ep_action_masks = []

            while True:
                current_agents = list(env.agents if env.agents else controllable_agents)
                local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
                mask_matrix = _mask_matrix(current_ep_masks, current_agents)

                # Compute global state for critic
                global_state = local_obs.flatten().astype(np.float32)

                with torch.no_grad():
                    obs_tensor = torch.from_numpy(local_obs).float()
                    mask_tensor = torch.tensor(mask_matrix, dtype=torch.bool)
                    dist = actor(obs_tensor, mask_tensor)
                    # STOCHASTIC sampling
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                    probs = dist.probs

                    # Critic value
                    value = float(critic(torch.from_numpy(global_state).float().unsqueeze(0)).item())

                action_dict = {}
                for i, a in enumerate(current_agents):
                    act_int = int(actions[i].item())
                    action_dict[a] = act_int
                    if act_int == TACKLE_ACTION:
                        tackle_actions += 1
                    if act_int in SHOT_ACTIONS:
                        shot_actions += 1
                    if act_int in PASS_ACTIONS:
                        pass_actions += 1
                    action_counts[act_int] += 1

                obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
                obs_dict = unwrap_obs(obs_dict)
                current_ep_masks = unwrap_masks(obs_dict)

                shared_rew = float(rewards[current_agents[0]]) if current_agents and current_agents[0] in rewards else 0.0
                ep_reward += shared_rew
                ep_length += 1

                term = any(terms.values()) if terms else False
                trunc = any(truncs.values()) if truncs else False
                done = term or trunc or not env.agents

                # Extract event info
                event_code = None
                event_type = None
                ball_owner_agent_idx = None
                score = {}
                if infos:
                    for inf in infos.values():
                        last_info = inf
                        event_code = inf.get("event_code")
                        event_type = inf.get("event", {}).get("type") if isinstance(inf.get("event"), dict) else None
                        ball_owner_agent_idx = inf.get("ball_owner_agent_idx")
                        score = inf.get("score", {})
                        if done and "ground_truth" in inf:
                            episode_ground_truth = inf["ground_truth"]
                        break

                # Store buffers for offline GAE
                ep_values.append(value)
                ep_rewards.append(shared_rew)
                ep_dones.append(term)  # FIXED: terminated only
                ep_truncated.append(trunc)
                ep_terminated.append(term)
                ep_local_obs.append(local_obs)
                ep_actions.append(np.array([action_dict[a] for a in current_agents], dtype=np.int64))
                ep_action_masks.append(mask_matrix)

                tick_data = {
                    "tick": ep_length - 1,
                    "seed": ep_seed,
                    "episode": ep,
                    "actions": action_dict,
                    "action_counts": {ACTION_NAMES[i]: action_counts[i] for i in range(19)},
                    "shared_reward": shared_rew,
                    "ep_reward": ep_reward,
                    "event_code": event_code,
                    "event_type": event_type,
                    "ball_owner_agent_idx": ball_owner_agent_idx,
                    "score": score,
                    "value": value,
                    "terminated": term,
                    "truncated": trunc,
                    "done": done,
                    "action_probabilities": {
                        a: float(probs[i, act_int].item())
                        for i, (a, act_int) in enumerate(action_dict.items())
                    },
                }
                ep_tick_log.append(tick_data)

                if done:
                    break

            score_left = last_info.get("score", {}).get("left", 0)
            is_goal = score_left > 0

            # Compute offline GAE on recorded trajectory
            ep_values_arr = np.array(ep_values, dtype=np.float32)
            ep_rewards_arr = np.array(ep_rewards, dtype=np.float32)
            ep_dones_arr = np.array(ep_dones, dtype=np.bool_)
            ep_truncated_arr = np.array(ep_truncated, dtype=np.bool_)
            ep_terminated_arr = np.array(ep_terminated, dtype=np.bool_)
            ep_local_obs_arr = np.stack(ep_local_obs, axis=0).astype(np.float32)
            ep_actions_arr = np.stack(ep_actions, axis=0).astype(np.int64)
            ep_action_masks_arr = np.stack(ep_action_masks, axis=0).astype(np.int8)

            # Compute GAE with production semantics
            # For truncated episodes, bootstrap with critic
            if ep_truncated_arr[-1] and not ep_terminated_arr[-1]:
                with torch.no_grad():
                    bootstrap_value = float(critic(torch.from_numpy(ep_local_obs_arr[-1]).float().unsqueeze(0)).item())
            else:
                bootstrap_value = 0.0

            advantages, returns = compute_gae(
                rewards=ep_rewards_arr,
                values=ep_values_arr,
                dones=ep_dones_arr,  # FIXED: terminated only
                gamma=GAMMA,
                lam=LAM,
                bootstrap_value=bootstrap_value,
                next_local_obs=ep_local_obs_arr[-1] if len(ep_local_obs_arr) > 0 else None,
                critic=critic,
            )

            # Compute TD residuals
            td_residuals = np.zeros_like(ep_rewards_arr)
            for t in range(len(ep_rewards_arr)):
                r_t = ep_rewards_arr[t]
                V_t = ep_values_arr[t]
                if t < len(ep_values_arr) - 1:
                    V_next = ep_values_arr[t + 1]
                    next_nonterminal = 0.0 if ep_dones_arr[t] else 1.0
                else:
                    V_next = bootstrap_value
                    next_nonterminal = 0.0 if ep_terminated_arr[t] else 1.0
                td_residuals[t] = r_t + GAMMA * V_next * next_nonterminal - V_t

            # Attach computed metrics to tick log
            for t, tick in enumerate(ep_tick_log):
                tick["gae_advantage"] = float(advantages[t]) if t < len(advantages) else None
                tick["gae_return"] = float(returns[t]) if t < len(returns) else None
                tick["td_residual"] = float(td_residuals[t]) if t < len(td_residuals) else None
                tick["bootstrap_value"] = bootstrap_value if t == len(ep_tick_log) - 1 else None
                tick["episode_terminated"] = bool(ep_terminated_arr[t]) if t < len(ep_terminated_arr) else None
                tick["episode_truncated"] = bool(ep_truncated_arr[t]) if t < len(ep_truncated_arr) else None

            episode_summary = {
                "episode": ep,
                "seed": ep_seed,
                "goal": int(is_goal),
                "tackle_actions": tackle_actions,
                "shot_actions": shot_actions,
                "pass_actions": pass_actions,
                "total_reward": ep_reward,
                "length": ep_length,
                "action_distribution": {ACTION_NAMES[i]: action_counts[i] for i in range(19)},
                "tick_log": ep_tick_log,
                "gae_advantages": advantages.tolist(),
                "gae_returns": returns.tolist(),
                "td_residuals": td_residuals.tolist(),
                "values": ep_values_arr.tolist(),
                "rewards": ep_rewards_arr.tolist(),
                "dones": ep_dones_arr.tolist(),
                "terminated": ep_terminated_arr.tolist(),
                "truncated": ep_truncated_arr.tolist(),
                "bootstrap_value": bootstrap_value,
            }
            episodes_data.append(episode_summary)

            if (ep + 1) % 20 == 0:
                print(f"Ep {ep+1:3d}/{num_episodes} | seed={ep_seed} | "
                      f"Goal={is_goal} | Tackles={tackle_actions} | "
                      f"Shots={shot_actions} | Passes={pass_actions} | "
                      f"Reward={ep_reward:+.3f} | Length={ep_length} | "
                      f"Term={term} | Trunc={trunc}")

    finally:
        env.close()

    # Aggregate metrics
    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_timesteps": ckpt_timesteps,
        "scenario": scenario,
        "num_episodes": num_episodes,
        "base_seed": base_seed,
        "gamma": GAMMA,
        "lambda": LAM,
        "timestamp_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode().strip(),
        "overall": {
            "goal_rate_pct": 100.0 * sum(1 for ep in episodes_data if ep["goal"]) / max(len(episodes_data), 1),
            "mean_tackles_per_ep": float(np.mean([ep["tackle_actions"] for ep in episodes_data])) if episodes_data else 0.0,
            "mean_shots_per_ep": float(np.mean([ep["shot_actions"] for ep in episodes_data])) if episodes_data else 0.0,
            "mean_passes_per_ep": float(np.mean([ep["pass_actions"] for ep in episodes_data])) if episodes_data else 0.0,
            "mean_reward": float(np.mean([ep["total_reward"] for ep in episodes_data])) if episodes_data else 0.0,
            "mean_length": float(np.mean([ep["length"] for ep in episodes_data])) if episodes_data else 0.0,
        },
        "episodes": episodes_data,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stochastic Rollout Credit Probe")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MAPPO checkpoint")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper_onball")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=600000)
    parser.add_argument("--output-dir", type=str, default="training/results")
    args = parser.parse_args()

    summary = collect_stochastic_rollouts(
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        base_seed=args.base_seed,
    )

    # Save detailed JSON
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    json_path = os.path.join(args.output_dir, f"stochastic_rollout_{ckpt_name}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed results saved to: {json_path}")

    # Save summary CSV row
    csv_path = os.path.join(os.path.dirname(__file__), "results", "stochastic_rollout_summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "checkpoint", "checkpoint_sha256", "checkpoint_timesteps", "scenario", "num_episodes",
            "goal_rate_pct", "mean_tackles_per_ep", "mean_shots_per_ep", "mean_passes_per_ep",
            "mean_reward", "mean_length", "mean_value", "std_value",
            "mean_td_residual", "std_td_residual", "mean_gae_advantage", "std_gae_advantage",
            "truncated_count", "terminated_count", "timestamp_iso", "git_commit",
        ])
        if not file_exists:
            writer.writeheader()

        # Compute aggregate stats
        all_values = []
        all_td_residuals = []
        all_gae_advantages = []
        truncated_count = 0
        terminated_count = 0
        for ep in summary["episodes"]:
            all_values.extend(ep["values"])
            all_td_residuals.extend(ep["td_residuals"])
            all_gae_advantages.extend(ep["gae_advantages"])
            truncated_count += sum(ep["truncated"])
            terminated_count += sum(ep["terminated"])

        writer.writerow({
            "checkpoint": summary["checkpoint"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "checkpoint_timesteps": summary["checkpoint_timesteps"],
            "scenario": summary["scenario"],
            "num_episodes": summary["num_episodes"],
            "goal_rate_pct": summary["overall"]["goal_rate_pct"],
            "mean_tackles_per_ep": summary["overall"]["mean_tackles_per_ep"],
            "mean_shots_per_ep": summary["overall"]["mean_shots_per_ep"],
            "mean_passes_per_ep": summary["overall"]["mean_passes_per_ep"],
            "mean_reward": summary["overall"]["mean_reward"],
            "mean_length": summary["overall"]["mean_length"],
            "mean_value": float(np.mean(all_values)) if all_values else None,
            "std_value": float(np.std(all_values)) if all_values else None,
            "mean_td_residual": float(np.mean(all_td_residuals)) if all_td_residuals else None,
            "std_td_residual": float(np.std(all_td_residuals)) if all_td_residuals else None,
            "mean_gae_advantage": float(np.mean(all_gae_advantages)) if all_gae_advantages else None,
            "std_gae_advantage": float(np.std(all_gae_advantages)) if all_gae_advantages else None,
            "truncated_count": truncated_count,
            "terminated_count": terminated_count,
            "timestamp_iso": summary["timestamp_iso"],
            "git_commit": summary["git_commit"],
        })
    print(f"Summary CSV updated: {csv_path}")


if __name__ == "__main__":
    main()
