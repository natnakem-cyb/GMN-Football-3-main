"""
GMN-Football-3 — Paralysis Forensics Evaluation
Measurement-only: logs per-tick action distribution, masks, actor confidence,
critic values, and state/owner data for current checkpoint population.
No reward, mask, engine, or training code is modified.
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
from training.mappo_rollout import unwrap_obs, unwrap_masks, _mask_matrix

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


def evaluate_paralysis_forensics(
    checkpoint_path: str,
    scenario: str = "academy_3_vs_1_with_keeper",
    num_episodes: int = 50,
    deterministic: bool = True,
    base_seed: int = 500000,
    bridge_port: int = 5050,
) -> Dict[str, Any]:
    """Run instrumented paralysis forensics evaluation on a MAPPO checkpoint."""

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_sha256 = sha256_of(checkpoint_path)
    actor, critic, obs_dim, action_dim, ckpt_timesteps = load_checkpoint(checkpoint_path)

    print("=" * 70)
    print("PARALYSIS FORENSICS EVALUATION")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"SHA256     : {checkpoint_sha256[:16]}...")
    print(f"Timesteps  : {ckpt_timesteps}")
    print(f"Scenario   : {scenario}")
    print(f"Episodes   : {num_episodes}")
    print(f"Deterministic: {deterministic}")
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
            action_logits_hist: List[Dict[str, Any]] = []
            last_info = {}
            episode_ground_truth = {}

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
                    value = None
                    if critic is not None:
                        state_tensor = torch.from_numpy(global_state).float().unsqueeze(0)
                        value = float(critic(state_tensor).item())

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

                    # Log action distribution for this agent
                    agent_logits = logits[i].cpu().numpy().tolist()
                    agent_probs = probs[i].cpu().numpy().tolist()
                    agent_logprob = float(log_probs[i].item())
                    agent_max_prob = float(agent_probs[act_int])
                    agent_entropy = float(dist.entropy()[i].item())
                    sorted_indices = np.argsort(agent_probs)[::-1][:3]
                    top3 = {
                        "indices": sorted_indices.tolist(),
                        "actions": [ACTION_NAMES[j] for j in sorted_indices],
                        "probs": [round(float(agent_probs[j]), 6) for j in sorted_indices],
                    }
                    logit_margin = round(float(agent_probs[sorted_indices[0]] - agent_probs[sorted_indices[1]]), 6) if len(sorted_indices) >= 2 else None

                    action_logits_hist.append({
                        "agent": a,
                        "action_id": act_int,
                        "action_name": ACTION_NAMES[act_int],
                        "log_prob": round(agent_logprob, 6),
                        "max_prob": round(agent_max_prob, 6),
                        "entropy": round(agent_entropy, 6),
                        "logit_margin_top1_top2": logit_margin,
                        "top3": top3,
                        "mask": mask_matrix[i].tolist(),
                        "mask_sum": int(mask_matrix[i].sum()),
                        "mask_legal_count": int(mask_matrix[i].sum()),
                        "argmax_before_mask": int(np.argmax(agent_logits)),
                        "argmax_after_mask": int(act_int),
                    })

                # Capture mask legality for off-ball left agents
                off_ball_left_masks = {}
                for a in current_agents:
                    if a.startswith("left_") and a in current_ep_masks:
                        mask = current_ep_masks[a]
                        if mask is not None:
                            off_ball_left_masks[a] = {
                                "tackle_legal": int(mask[TACKLE_ACTION]) if len(mask) > TACKLE_ACTION else 0,
                                "shot_legal": int(mask[12]) if len(mask) > 12 else 0,
                                "pass_legal": int(mask[9]) if len(mask) > 9 else 0,
                                "dribble_legal": int(mask[17]) if len(mask) > 17 else 0,
                                "mask_sum": int(mask.sum()),
                                "mask": mask.tolist(),
                            }

                obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
                
                # Capture masks from RAW env output BEFORE unwrapping obs
                raw_masks = {}
                if isinstance(obs_dict, dict):
                    for a in current_agents:
                        agent_obs = obs_dict.get(a)
                        if isinstance(agent_obs, dict) and agent_obs.get("action_mask") is not None:
                            raw_masks[a] = np.array(agent_obs["action_mask"], dtype=np.int8)
                
                # Update off_ball_left_masks with masks from THIS tick (post-step)
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
                
                # Also capture ground_truth from infos on every tick
                tick_ground_truth = {}
                if infos:
                    for inf in infos.values():
                        if "ground_truth" in inf:
                            tick_ground_truth = inf["ground_truth"]
                        break
                
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

                # Possession info
                possession_left = tick_ground_truth.get("possession_left", None) if tick_ground_truth else None

                tick_data = {
                    "tick": ep_length - 1,
                    "seed": ep_seed,
                    "episode": ep,
                    "actions": action_dict,
                    "action_counts": {ACTION_NAMES[i]: action_counts[i] for i in range(19)},
                    "off_ball_left_masks_pre": off_ball_left_masks,
                    "off_ball_left_masks_post": post_step_masks,
                    "shared_reward": shared_rew,
                    "ep_reward": ep_reward,
                    "event_code": event_code,
                    "event_type": event_type,
                    "ball_owner_agent_idx": ball_owner_agent_idx,
                    "score": score,
                    "d_self_ball": round(d_self_ball, 4) if d_self_ball is not None else None,
                    "d_self_goal": round(d_self_goal, 4) if d_self_goal is not None else None,
                    "possession_left": possession_left,
                    "value": value,
                    "action_logits_hist": action_logits_hist,
                }
                ep_tick_log.append(tick_data)

                if done:
                    break

            score_left = last_info.get("score", {}).get("left", 0)
            is_goal = score_left > 0

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
            }
            episodes_data.append(episode_summary)

            print(f"Ep {ep+1:3d}/{num_episodes} | seed={ep_seed} | "
                  f"Goal={is_goal} | Tackles={tackle_actions} | "
                  f"Shots={shot_actions} | Passes={pass_actions} | "
                  f"Reward={ep_reward:+.3f} | Length={ep_length}")

    finally:
        env.close()

    # Aggregate metrics
    tackle_heavy_episodes = [ep for ep in episodes_data if ep["tackle_actions"] >= 3]

    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_timesteps": ckpt_timesteps,
        "scenario": scenario,
        "num_episodes": num_episodes,
        "deterministic": deterministic,
        "base_seed": base_seed,
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
        "tackle_heavy_episodes": {
            "count": len(tackle_heavy_episodes),
            "goal_rate_pct": 100.0 * sum(1 for ep in tackle_heavy_episodes if ep["goal"]) / max(len(tackle_heavy_episodes), 1),
            "mean_tackles_per_ep": float(np.mean([ep["tackle_actions"] for ep in tackle_heavy_episodes])) if tackle_heavy_episodes else 0.0,
        },
        "episodes": episodes_data,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Paralysis Forensics Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MAPPO checkpoint")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=500000)
    parser.add_argument("--output-dir", type=str, default="training/models")
    args = parser.parse_args()

    summary = evaluate_paralysis_forensics(
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        deterministic=True,
        base_seed=args.base_seed,
    )

    # Save detailed JSON
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    json_path = os.path.join(args.output_dir, f"paralysis_forensics_{ckpt_name}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed results saved to: {json_path}")

    # Save summary CSV row
    csv_path = os.path.join(os.path.dirname(__file__), "results", "paralysis_forensics_summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "checkpoint", "checkpoint_sha256", "checkpoint_timesteps", "scenario", "num_episodes",
            "goal_rate_pct", "mean_tackles_per_ep", "mean_shots_per_ep", "mean_passes_per_ep",
            "mean_reward", "mean_length", "tackle_heavy_count", "tackle_heavy_goal_rate_pct",
            "timestamp_iso", "git_commit",
        ])
        if not file_exists:
            writer.writeheader()
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
            "tackle_heavy_count": summary["tackle_heavy_episodes"]["count"],
            "tackle_heavy_goal_rate_pct": summary["tackle_heavy_episodes"]["goal_rate_pct"],
            "timestamp_iso": summary["timestamp_iso"],
            "git_commit": summary["git_commit"],
        })
    print(f"Summary CSV updated: {csv_path}")


if __name__ == "__main__":
    main()
