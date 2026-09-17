"""
GMN-Football-3 — Occupancy Gate Evaluation (Hypothesis F)
Measurement-only: logs per-tick occupancy, event, critic, and advantage metrics
for CTRL vs INT arms on frozen weights.
No reward, mask, engine, or training code is modified beyond the single μ change.
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
    "SHORT_PASS", "LONG_PASS", "HIGH_PASS",
    "SHOT", "SPRINT",
    "RELEASE_DIRECTION", "RELEASE_SPRINT",
    "SLIDING", "DRIBBLE", "RELEASE_DRIBBLE"
]
TACKLE_ACTION = 16
SHOT_ACTIONS = {12}
PASS_ACTIONS = {9, 10, 11}
BALL_ACTION_IDS = frozenset((9, 10, 11, 12, 17))

# Production GAE convention (must match train_mappo.py)
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


def _event_type_from_code(event_code: Optional[int]) -> Optional[str]:
    from training.gmn_pettingzoo import EVENT_CODE_MAP
    if event_code is None or event_code <= 0 or event_code >= len(EVENT_CODE_MAP):
        return None
    return EVENT_CODE_MAP[event_code]


def evaluate_occupancy_gate(
    checkpoint_path: str,
    scenario: str = "academy_3_vs_1_with_keeper",
    num_episodes: int = 50,
    deterministic: bool = True,
    base_seed: int = 500000,
    bridge_port: int = 5050,
) -> Dict[str, Any]:
    """Run instrumented occupancy-gate evaluation on a MAPPO checkpoint."""

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_sha256 = sha256_of(checkpoint_path)
    actor, critic, obs_dim, action_dim, ckpt_timesteps = load_checkpoint(checkpoint_path)

    print("=" * 70)
    print("OCCUPANCY GATE EVALUATION")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"SHA256     : {checkpoint_sha256[:16]}...")
    print(f"Timesteps  : {ckpt_timesteps}")
    print(f"Scenario   : {scenario}")
    print(f"Episodes   : {num_episodes}")
    print(f"Deterministic: {deterministic}")
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

            # Buffers for offline GAE (production-correct: terminated-only dones)
            ep_values = []
            ep_rewards = []
            ep_terminated = []
            ep_truncated = []
            ep_local_obs = []
            ep_actions = []
            ep_action_masks = []

            while True:
                current_agents = list(env.agents if env.agents else controllable_agents)
                local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
                mask_matrix = _mask_matrix(current_ep_masks, current_agents)

                # Compute global state for critic: flatten local_obs in agent_order
                global_state = local_obs.flatten().astype(np.float32)

                with torch.no_grad():
                    obs_tensor = torch.from_numpy(local_obs).float()
                    mask_tensor = torch.tensor(mask_matrix, dtype=torch.bool)
                    dist = actor(obs_tensor, mask_tensor)
                    logits = dist.logits
                    if deterministic:
                        actions = logits.argmax(dim=-1)
                    else:
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

                # Capture mask legality for off-ball left agents (pre-step)
                off_ball_left_masks_pre = {}
                for a in current_agents:
                    if a.startswith("left_") and a in current_ep_masks:
                        mask = current_ep_masks[a]
                        if mask is not None:
                            off_ball_left_masks_pre[a] = {
                                "tackle_legal": int(mask[TACKLE_ACTION]) if len(mask) > TACKLE_ACTION else 0,
                                "shot_legal": int(mask[12]) if len(mask) > 12 else 0,
                                "pass_legal": int(mask[9]) if len(mask) > 9 else 0,
                                "dribble_legal": int(mask[17]) if len(mask) > 17 else 0,
                                "mask_sum": int(mask.sum()),
                                "mask": mask.tolist(),
                            }

                obs_dict, rewards, terms, truncs, infos = env.step(action_dict)

                # Capture masks from RAW env output BEFORE unwrapping obs (post-step)
                raw_masks = {}
                if isinstance(obs_dict, dict):
                    for a in current_agents:
                        agent_obs = obs_dict.get(a)
                        if isinstance(agent_obs, dict) and agent_obs.get("action_mask") is not None:
                            raw_masks[a] = np.array(agent_obs["action_mask"], dtype=np.int8)

                post_step_masks = {}
                for a in current_agents:
                    if a.startswith("left_") and a in raw_masks:
                        mask = raw_masks[a]
                        post_step_masks[a] = {
                            "tackle_legal": int(mask[TACKLE_ACTION]) if len(mask) > TACKLE_ACTION else 0,
                            "shot_legal": int(mask[12]) if len(mask) > 12 else 0,
                            "pass_legal": int(mask[9]) if len(mask) > 9 else 0,
                            "dribble_legal": int(mask[17]) if len(mask) > 17 else 0,
                            "mask_sum": int(mask.sum()),
                            "mask": mask.tolist(),
                        }

                obs_dict = unwrap_obs(obs_dict)
                current_ep_masks = unwrap_masks(obs_dict)

                shared_rew = float(rewards[current_agents[0]]) if current_agents and current_agents[0] in rewards else 0.0
                ep_reward += shared_rew
                ep_length += 1

                term = any(terms.values()) if terms else False
                trunc = any(truncs.values()) if truncs else False
                done = term or trunc or not env.agents

                # Extract event info
                event_code = getattr(env, "_last_frame_event_code", None)
                event_type = _event_type_from_code(event_code)
                ball_owner_agent_idx = getattr(env, "_last_ball_owner_agent_idx", 255)
                score = {}
                if infos:
                    for inf in infos.values():
                        last_info = inf
                        score = inf.get("score", {})
                        break

                # Compute ball distances
                d_self_ball = None
                d_self_goal = None
                if obs_dict and current_agents:
                    first_obs = obs_dict[current_agents[0]]
                    if hasattr(first_obs, "__len__") and len(first_obs) >= 14:
                        try:
                            ball_x = float(first_obs[11])
                            ball_y = float(first_obs[12])
                            self_x = float(first_obs[0])
                            self_y = float(first_obs[1])
                            goal_x = 1.0
                            goal_y = 0.0
                            d_self_ball = ((self_x - ball_x)**2 + (self_y - ball_y)**2)**0.5
                            d_self_goal = ((self_x - goal_x)**2 + (self_y - goal_y)**2)**0.5
                        except (IndexError, ValueError, TypeError):
                            pass

                # Determine if controlled player (index 0) owns the ball
                controlled_has_ball = (ball_owner_agent_idx == 0)
                left_has_ball = (0 <= ball_owner_agent_idx < len(current_agents) and
                                 current_agents[ball_owner_agent_idx].startswith("left_"))

                tick_data = {
                    "tick": ep_length - 1,
                    "seed": ep_seed,
                    "episode": ep,
                    "actions": action_dict,
                    "action_counts": {ACTION_NAMES[i]: action_counts[i] for i in range(19)},
                    "off_ball_left_masks_pre": off_ball_left_masks_pre,
                    "off_ball_left_masks_post": post_step_masks,
                    "shared_reward": shared_rew,
                    "ep_reward": ep_reward,
                    "event_code": event_code,
                    "event_type": event_type,
                    "ball_owner_agent_idx": ball_owner_agent_idx,
                    "controlled_has_ball": controlled_has_ball,
                    "left_has_ball": left_has_ball,
                    "score": score,
                    "d_self_ball": round(d_self_ball, 4) if d_self_ball is not None else None,
                    "d_self_goal": round(d_self_goal, 4) if d_self_goal is not None else None,
                    "value": value,
                    "terminated": term,
                    "truncated": trunc,
                    "done": done,
                }
                ep_tick_log.append(tick_data)

                # Store buffers for offline GAE (production-correct)
                ep_values.append(value)
                ep_rewards.append(shared_rew)
                ep_terminated.append(term)
                ep_truncated.append(trunc)
                ep_local_obs.append(local_obs)
                ep_actions.append(np.array([action_dict[a] for a in current_agents], dtype=np.int64))
                ep_action_masks.append(mask_matrix)

                if done:
                    break

            score_left = last_info.get("score", {}).get("left", 0)
            is_goal = score_left > 0

            # Compute offline GAE on recorded trajectory (production-correct)
            ep_values_arr = np.array(ep_values, dtype=np.float32)
            ep_rewards_arr = np.array(ep_rewards, dtype=np.float32)
            ep_terminated_arr = np.array(ep_terminated, dtype=np.bool_)
            ep_truncated_arr = np.array(ep_truncated, dtype=np.bool_)
            ep_local_obs_arr = np.stack(ep_local_obs, axis=0).astype(np.float32)
            ep_actions_arr = np.stack(ep_actions, axis=0).astype(np.int64)
            ep_action_masks_arr = np.stack(ep_action_masks, axis=0).astype(np.int8)

            # Bootstrap on truncation (training-correct), 0.0 on termination
            if ep_truncated_arr[-1] and not ep_terminated_arr[-1]:
                with torch.no_grad():
                    bootstrap_value = float(critic(torch.from_numpy(ep_local_obs_arr[-1]).float().unsqueeze(0)).item())
            else:
                bootstrap_value = 0.0

            advantages, returns = compute_gae(
                rewards=ep_rewards_arr,
                values=ep_values_arr,
                dones=ep_terminated_arr,  # FIX: terminated only (production GAE convention)
                gamma=GAMMA,
                lam=LAM,
                bootstrap_value=bootstrap_value,
                next_local_obs=ep_local_obs_arr[-1] if len(ep_local_obs_arr) > 0 else None,
                critic=critic,
            )

            # Attach computed metrics to tick log
            for t, tick in enumerate(ep_tick_log):
                tick["gae_advantage"] = float(advantages[t]) if t < len(advantages) else None
                tick["gae_return"] = float(returns[t]) if t < len(returns) else None
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
                "values": ep_values_arr.tolist(),
                "rewards": ep_rewards_arr.tolist(),
                "terminated": ep_terminated_arr.tolist(),
                "truncated": ep_truncated_arr.tolist(),
                "bootstrap_value": bootstrap_value,
            }
            episodes_data.append(episode_summary)

            print(f"Ep {ep+1:3d}/{num_episodes} | seed={ep_seed} | "
                  f"Goal={is_goal} | Tackles={tackle_actions} | "
                  f"Shots={shot_actions} | Passes={pass_actions} | "
                  f"Reward={ep_reward:+.3f} | Length={ep_length} | "
                  f"Term={term} | Trunc={trunc}")

    finally:
        env.close()

    # ------------------- Aggregate metrics -------------------
    all_ticks = [tick for ep in episodes_data for tick in ep["tick_log"]]
    total_ticks = len(all_ticks)

    # Occupancy fractions
    left_possession_ticks = sum(1 for t in all_ticks if t.get("left_has_ball"))
    controlled_owner_ticks = sum(1 for t in all_ticks if t.get("controlled_has_ball"))
    pass_legal_ticks = sum(1 for t in all_ticks
                           for a, m in t.get("off_ball_left_masks_post", {}).items()
                           if m.get("pass_legal"))
    shot_legal_ticks = sum(1 for t in all_ticks
                          for a, m in t.get("off_ball_left_masks_post", {}).items()
                          if m.get("shot_legal"))

    # Distances
    d_self_ball_vals = [t["d_self_ball"] for t in all_ticks if t.get("d_self_ball") is not None]
    d_self_goal_vals = [t["d_self_goal"] for t in all_ticks if t.get("d_self_goal") is not None]

    # Events
    pass_completed_count = sum(1 for t in all_ticks if t.get("event_type") == "pass_completed")
    shot_event_count = sum(1 for t in all_ticks if t.get("event_type") in ("shot", "shot_saved", "shot_missed"))
    goal_count = sum(1 for t in all_ticks if t.get("event_type") == "goal")

    # Reward stats
    all_rewards = [t["shared_reward"] for t in all_ticks]
    reward_entropy = 0.0
    if all_rewards:
        unique, counts = np.unique(np.round(all_rewards, 4), return_counts=True)
        probs = counts / counts.sum()
        reward_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    # Value / advantage stats
    all_values = [t["value"] for t in all_ticks]
    all_advantages = [t["gae_advantage"] for t in all_ticks if t.get("gae_advantage") is not None]

    def _std(vals):
        return float(np.std(vals)) if vals else None

    def _mean(vals):
        return float(np.mean(vals)) if vals else None

    # On-ball vs off-ball vs timeout
    on_ball_values = [t["value"] for t in all_ticks if t.get("controlled_has_ball")]
    off_ball_values = [t["value"] for t in all_ticks if not t.get("controlled_has_ball")]
    timeout_values = [t["value"] for t in all_ticks if t.get("truncated")]

    on_ball_advantages = [t["gae_advantage"] for t in all_ticks if t.get("controlled_has_ball") and t.get("gae_advantage") is not None]
    off_ball_advantages = [t["gae_advantage"] for t in all_ticks if not t.get("controlled_has_ball") and t.get("gae_advantage") is not None]

    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_timesteps": ckpt_timesteps,
        "scenario": scenario,
        "arm": None,
        "num_episodes": num_episodes,
        "deterministic": deterministic,
        "base_seed": base_seed,
        "gamma": GAMMA,
        "lambda": LAM,
        "timestamp_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode().strip(),
        "overall": {
            "total_ticks": total_ticks,
            "left_possession_ticks": left_possession_ticks,
            "left_possession_fraction": left_possession_ticks / max(total_ticks, 1),
            "controlled_owner_ticks": controlled_owner_ticks,
            "controlled_owner_fraction": controlled_owner_ticks / max(total_ticks, 1),
            "pass_legal_ticks": pass_legal_ticks,
            "pass_legal_fraction": pass_legal_ticks / max(total_ticks, 1),
            "shot_legal_ticks": shot_legal_ticks,
            "shot_legal_fraction": shot_legal_ticks / max(total_ticks, 1),
            "mean_d_self_ball": _mean(d_self_ball_vals),
            "std_d_self_ball": _std(d_self_ball_vals),
            "mean_d_self_goal": _mean(d_self_goal_vals),
            "std_d_self_goal": _std(d_self_goal_vals),
            "pass_completed_count": pass_completed_count,
            "shot_event_count": shot_event_count,
            "goal_count": goal_count,
            "mean_reward": _mean(all_rewards),
            "std_reward": _std(all_rewards),
            "reward_entropy": reward_entropy,
            "mean_value": _mean(all_values),
            "std_value": _std(all_values),
            "mean_gae_advantage": _mean(all_advantages),
            "std_gae_advantage": _std(all_advantages),
            "std_value_on_ball": _std(on_ball_values),
            "std_value_off_ball": _std(off_ball_values),
            "std_value_timeout": _std(timeout_values),
            "mean_advantage_on_ball": _mean(on_ball_advantages),
            "mean_advantage_off_ball": _mean(off_ball_advantages),
            "std_advantage_on_ball": _std(on_ball_advantages),
            "std_advantage_off_ball": _std(off_ball_advantages),
        },
        "episodes": episodes_data,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Occupancy Gate Evaluation (Hypothesis F)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MAPPO checkpoint")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=500000)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="training/models")
    parser.add_argument("--arm", type=str, default="CTRL", choices=["CTRL", "INT"])
    args = parser.parse_args()

    summary = evaluate_occupancy_gate(
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        base_seed=args.base_seed,
    )
    summary["arm"] = args.arm

    # Save detailed JSON
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    json_name = f"occupancy_gate_{args.arm}_{ckpt_name}.json"
    json_path = os.path.join(args.output_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed results saved to: {json_path}")

    # Save summary CSV row
    csv_path = os.path.join(os.path.dirname(__file__), "results", "occupancy_gate_summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "arm", "checkpoint", "checkpoint_sha256", "checkpoint_timesteps", "scenario",
            "num_episodes", "deterministic", "base_seed",
            "left_possession_fraction", "controlled_owner_fraction",
            "pass_legal_fraction", "shot_legal_fraction",
            "mean_d_self_ball", "std_d_self_ball",
            "mean_d_self_goal", "std_d_self_goal",
            "pass_completed_count", "shot_event_count", "goal_count",
            "mean_reward", "std_reward", "reward_entropy",
            "mean_value", "std_value",
            "mean_gae_advantage", "std_gae_advantage",
            "std_value_on_ball", "std_value_off_ball", "std_value_timeout",
            "mean_advantage_on_ball", "mean_advantage_off_ball",
            "std_advantage_on_ball", "std_advantage_off_ball",
            "timestamp_iso", "git_commit",
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "arm": summary["arm"],
            "checkpoint": summary["checkpoint"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "checkpoint_timesteps": summary["checkpoint_timesteps"],
            "scenario": summary["scenario"],
            "num_episodes": summary["num_episodes"],
            "deterministic": summary["deterministic"],
            "base_seed": summary["base_seed"],
            "left_possession_fraction": summary["overall"]["left_possession_fraction"],
            "controlled_owner_fraction": summary["overall"]["controlled_owner_fraction"],
            "pass_legal_fraction": summary["overall"]["pass_legal_fraction"],
            "shot_legal_fraction": summary["overall"]["shot_legal_fraction"],
            "mean_d_self_ball": summary["overall"]["mean_d_self_ball"],
            "std_d_self_ball": summary["overall"]["std_d_self_ball"],
            "mean_d_self_goal": summary["overall"]["mean_d_self_goal"],
            "std_d_self_goal": summary["overall"]["std_d_self_goal"],
            "pass_completed_count": summary["overall"]["pass_completed_count"],
            "shot_event_count": summary["overall"]["shot_event_count"],
            "goal_count": summary["overall"]["goal_count"],
            "mean_reward": summary["overall"]["mean_reward"],
            "std_reward": summary["overall"]["std_reward"],
            "reward_entropy": summary["overall"]["reward_entropy"],
            "mean_value": summary["overall"]["mean_value"],
            "std_value": summary["overall"]["std_value"],
            "mean_gae_advantage": summary["overall"]["mean_gae_advantage"],
            "std_gae_advantage": summary["overall"]["std_gae_advantage"],
            "std_value_on_ball": summary["overall"]["std_value_on_ball"],
            "std_value_off_ball": summary["overall"]["std_value_off_ball"],
            "std_value_timeout": summary["overall"]["std_value_timeout"],
            "mean_advantage_on_ball": summary["overall"]["mean_advantage_on_ball"],
            "mean_advantage_off_ball": summary["overall"]["mean_advantage_off_ball"],
            "std_advantage_on_ball": summary["overall"]["std_advantage_on_ball"],
            "std_advantage_off_ball": summary["overall"]["std_advantage_off_ball"],
            "timestamp_iso": summary["timestamp_iso"],
            "git_commit": summary["git_commit"],
        })
    print(f"Summary CSV updated: {csv_path}")


if __name__ == "__main__":
    main()
