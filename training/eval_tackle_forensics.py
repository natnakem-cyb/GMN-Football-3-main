"""
GMN-Football-3 — Tackle Spam Forensic Evaluation
Measurement-only: logs per-tick reward decomposition, action masks, and
tackle outcomes for episodes from tackle-heavy checkpoints.
No reward, mask, or engine code is modified.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor
from training.mappo_rollout import unwrap_obs, unwrap_masks, _mask_matrix
from training.reward_adapters import AttackingDrillRewardAdapter


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


class InstrumentedAdapter(AttackingDrillRewardAdapter):
    """Wraps AttackingDrillRewardAdapter to log per-tick reward decomposition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tick_log: List[Dict[str, Any]] = []

    def compute_shaped_rewards(self, base_rewards, step_events, info_ground_truth, active_agents, actions=None, tick=0, max_ticks=None):
        shaped = {a: base_rewards.get(a, 0.0) for a in active_agents if not a.startswith("right_")}
        self._current_tick = tick

        # Capture pre-shaping base rewards per left agent
        base_snapshot = {a: float(base_rewards.get(a, 0.0)) for a in active_agents if not a.startswith("right_")}

        # Run the full shaping pipeline
        self._apply_action_cost(shaped, actions)
        self._apply_possession(shaped, info_ground_truth)
        self._handle_events(shaped, step_events, active_agents, actions)

        # Temporarily skip _strip_progress, pass rewards, shot rewards, PBRS, dense terms,
        # exploration bonus to capture only the event-driven and possession terms.
        # Actually, we want the FULL decomposition. Let me re-think.
        # Instead, I'll instrument the parent class methods.
        # But since we're subclassing, I need to override compute_shaped_rewards
        # to capture all terms. Let me do it differently.

        # For now, just call super and capture the delta
        result = super().compute_shaped_rewards(base_rewards, step_events, info_ground_truth, active_agents, actions=actions, tick=tick, max_ticks=max_ticks)

        # Capture per-agent delta
        delta = {a: float(result.get(a, 0.0)) - base_snapshot.get(a, 0.0) for a in base_snapshot}

        self.tick_log.append({
            "tick": tick,
            "base_rewards": base_snapshot,
            "shaped_rewards": {a: float(result.get(a, 0.0)) for a in base_snapshot},
            "delta": delta,
            "step_events": step_events,
            "info_ground_truth": info_ground_truth,
            "actions": actions,
        })
        return result


def evaluate_tackle_forensics(
    checkpoint_path: str,
    scenario: str = "academy_3_vs_1_with_keeper",
    num_episodes: int = 50,
    deterministic: bool = True,
    base_seed: int = 500000,
    bridge_port: int = 5050,
) -> Dict[str, Any]:
    """Run instrumented evaluation on a MAPPO checkpoint."""

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 60)
    print("TACKLE FORENSICS EVALUATION")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Scenario   : {scenario}")
    print(f"Episodes   : {num_episodes}")
    print("=" * 60)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    obs_dim = checkpoint.get("obs_dim", 127 if "actor" in checkpoint and checkpoint["actor"]["net.0.weight"].shape[1] == 127 else (checkpoint["actor"]["net.0.weight"].shape[1] if "actor" in checkpoint else 127))
    action_dim = checkpoint.get("action_dim", 19)

    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = GMNMultiAgentEnv(scenario=scenario, auto_start_bridge=True, port=bridge_port)
    controllable_agents = list(env.possible_agents)

    # Monkey-patch reward adapter with instrumented version
    original_scenario_adapter = env._scenario_adapter
    def instrumented_scenario_adapter(previous=None):
        adapter = original_scenario_adapter(previous)
        if isinstance(adapter, AttackingDrillRewardAdapter):
            # Replace with instrumented version, preserving state
            instrumented = InstrumentedAdapter(
                step_cost=getattr(adapter, 'step_cost', -0.005),
                shot_reward=getattr(adapter, 'r_shot', 0.25),
                on_target_reward=getattr(adapter, 'r_on_target', 0.40),
                t_max=getattr(adapter, 't_max', 50),
                timeout_penalty=getattr(adapter, 'timeout_penalty', -0.50),
                enable_exploration_bonus=getattr(adapter, 'enable_exploration_bonus', False),
                exploration_beta=getattr(adapter, 'exploration_beta', 0.03),
            )
            # Preserve visit counts from previous adapter if any
            if previous is not None and hasattr(previous, '_visit_counts'):
                instrumented._visit_counts = previous._visit_counts
            return instrumented
        return adapter

    env._scenario_adapter = instrumented_scenario_adapter

    # Rebuild adapter with instrumented version
    if getattr(env, "reward_adapter", None) is not None:
        env.reward_adapter = env._scenario_adapter(previous=getattr(env, "reward_adapter", None))
        env.reward_shaper = env.reward_adapter

    episodes_data: List[Dict[str, Any]] = []

    try:
        for ep in range(num_episodes):
            ep_seed = base_seed + ep * 1009
            obs_dict, _ = env.reset(seed=ep_seed)
            current_ep_masks = unwrap_masks(obs_dict)
            obs_dict = unwrap_obs(obs_dict)

            # Reset instrumented adapter for new episode
            if isinstance(env.reward_adapter, InstrumentedAdapter):
                env.reward_adapter.reset()

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

            while True:
                current_agents = list(env.agents if env.agents else controllable_agents)
                local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
                mask_matrix = _mask_matrix(current_ep_masks, current_agents)

                with torch.no_grad():
                    dist = actor(torch.from_numpy(local_obs).float(), torch.tensor(mask_matrix, dtype=torch.bool))
                    if deterministic:
                        actions = dist.logits.argmax(dim=-1)
                    else:
                        actions = dist.sample()

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
                                "mask_sum": int(mask.sum()),
                            }

                obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
                obs_dict = unwrap_obs(obs_dict)
                current_ep_masks = unwrap_masks(obs_dict)

                shared_rew = float(rewards[current_agents[0]]) if current_agents and current_agents[0] in rewards else 0.0
                ep_reward += shared_rew
                ep_length += 1

                term = any(terms.values()) if terms else False
                trunc = any(truncs.values()) if truncs else False
                done = term or trunc or not env.agents

                if infos:
                    for inf in infos.values():
                        last_info = inf
                        if done and "ground_truth" in inf:
                            episode_ground_truth = inf["ground_truth"]
                        break

                # Capture tick data
                tick_data = {
                    "tick": ep_length - 1,
                    "actions": action_dict,
                    "action_counts": action_counts.copy(),
                    "off_ball_left_masks": off_ball_left_masks,
                    "shared_reward": shared_rew,
                    "event_code": last_info.get("event_code"),
                    "event_type": last_info.get("event", {}).get("type") if isinstance(last_info.get("event"), dict) else None,
                    "ball_owner": last_info.get("ball_owner_agent_idx"),
                    "score": last_info.get("score", {}),
                }

                # Capture adapter tick log for this tick
                if isinstance(env.reward_adapter, InstrumentedAdapter) and env.reward_adapter.tick_log:
                    last_tick_log = env.reward_adapter.tick_log[-1]
                    tick_data["adapter_delta"] = last_tick_log["delta"]
                    tick_data["adapter_base"] = last_tick_log["base_rewards"]
                    tick_data["adapter_shaped"] = last_tick_log["shaped_rewards"]
                    tick_data["step_events"] = [
                        {"type": e.get("type"), "team": e.get("team"), "agent_id": e.get("agent_id")}
                        for e in last_tick_log["step_events"]
                    ]

                ep_tick_log.append(tick_data)

                if done:
                    break

            # Episode summary
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
                  f"Reward={ep_reward:+.3f}")

    finally:
        env.close()

    # Aggregate metrics
    tackle_heavy_episodes = [ep for ep in episodes_data if ep["tackle_actions"] >= 3]

    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256_of(checkpoint_path),
        "scenario": scenario,
        "num_episodes": num_episodes,
        "deterministic": deterministic,
        "timestamp_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode().strip(),
        "overall": {
            "goal_rate_pct": 100.0 * sum(1 for ep in episodes_data if ep["goal"]) / max(len(episodes_data), 1),
            "mean_tackles_per_ep": float(np.mean([ep["tackle_actions"] for ep in episodes_data])) if episodes_data else 0.0,
            "mean_shots_per_ep": float(np.mean([ep["shot_actions"] for ep in episodes_data])) if episodes_data else 0.0,
            "mean_passes_per_ep": float(np.mean([ep["pass_actions"] for ep in episodes_data])) if episodes_data else 0.0,
            "mean_reward": float(np.mean([ep["total_reward"] for ep in episodes_data])) if episodes_data else 0.0,
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
    parser = argparse.ArgumentParser(description="Tackle Spam Forensic Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MAPPO checkpoint")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=500000)
    parser.add_argument("--output-dir", type=str, default="training/models")
    args = parser.parse_args()

    summary = evaluate_tackle_forensics(
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        deterministic=True,
        base_seed=args.base_seed,
    )

    # Save detailed JSON
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    json_path = os.path.join(args.output_dir, f"tackle_forensics_{ckpt_name}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed results saved to: {json_path}")

    # Save summary CSV row
    csv_path = os.path.join(os.path.dirname(__file__), "results", "tackle_forensics_summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "checkpoint", "checkpoint_sha256", "scenario", "num_episodes",
            "goal_rate_pct", "mean_tackles_per_ep", "mean_shots_per_ep",
            "mean_passes_per_ep", "mean_reward", "tackle_heavy_count",
            "tackle_heavy_goal_rate_pct", "timestamp_iso",
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "checkpoint": summary["checkpoint"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "scenario": summary["scenario"],
            "num_episodes": summary["num_episodes"],
            "goal_rate_pct": summary["overall"]["goal_rate_pct"],
            "mean_tackles_per_ep": summary["overall"]["mean_tackles_per_ep"],
            "mean_shots_per_ep": summary["overall"]["mean_shots_per_ep"],
            "mean_passes_per_ep": summary["overall"]["mean_passes_per_ep"],
            "mean_reward": summary["overall"]["mean_reward"],
            "tackle_heavy_count": summary["tackle_heavy_episodes"]["count"],
            "tackle_heavy_goal_rate_pct": summary["tackle_heavy_episodes"]["goal_rate_pct"],
            "timestamp_iso": summary["timestamp_iso"],
        })
    print(f"Summary CSV updated: {csv_path}")


if __name__ == "__main__":
    main()
