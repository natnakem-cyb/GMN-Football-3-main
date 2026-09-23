"""
GMN-Football-3 — Forced Tackle Probe (measurement only)

Forces SLIDING (action 16) on off-ball left agents for every tick,
recording the resulting reward/event deltas. Does not modify production
code. Outputs a separate JSON from the policy-eval run.
"""

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

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
        self._current_tick = tick
        base_snapshot = {a: float(base_rewards.get(a, 0.0)) for a in active_agents if not a.startswith("right_")}
        result = super().compute_shaped_rewards(
            base_rewards, step_events, info_ground_truth, active_agents,
            actions=actions, tick=tick, max_ticks=max_ticks,
        )
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


def evaluate_forced_tackle_probe(
    checkpoint_path: str,
    scenario: str = "academy_3_vs_1_with_keeper",
    num_episodes: int = 10,
    base_seed: int = 500000,
    bridge_port: int = 5050,
) -> Dict[str, Any]:
    """Run forced-tackle probe on a MAPPO checkpoint."""

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 60)
    print("FORCED TACKLE PROBE")
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
            instrumented = InstrumentedAdapter(
                step_cost=getattr(adapter, 'step_cost', -0.005),
                shot_reward=getattr(adapter, 'r_shot', 0.25),
                on_target_reward=getattr(adapter, 'r_on_target', 0.40),
                t_max=getattr(adapter, 't_max', 50),
                timeout_penalty=getattr(adapter, 'timeout_penalty', -0.50),
                enable_exploration_bonus=getattr(adapter, 'enable_exploration_bonus', False),
                exploration_beta=getattr(adapter, 'exploration_beta', 0.03),
            )
            if previous is not None and hasattr(previous, '_visit_counts'):
                instrumented._visit_counts = previous._visit_counts
            return instrumented
        return adapter

    env._scenario_adapter = instrumented_scenario_adapter
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

            if isinstance(env.reward_adapter, InstrumentedAdapter):
                env.reward_adapter.reset()

            ep_tick_log: List[Dict[str, Any]] = []
            ep_reward = 0.0
            ep_length = 0
            tackle_ticks = []
            action_counts = [0] * 19
            last_info = {}

            while True:
                current_agents = list(env.agents if env.agents else controllable_agents)
                local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
                mask_matrix = _mask_matrix(current_ep_masks, current_agents)

                with torch.no_grad():
                    dist = actor(torch.from_numpy(local_obs).float(), torch.tensor(mask_matrix, dtype=torch.bool))
                    policy_actions = dist.logits.argmax(dim=-1)

                action_dict = {}
                for i, a in enumerate(current_agents):
                    # Force TACKLE (16) on off-ball left agents
                    if a.startswith("left_") and a in current_ep_masks and current_ep_masks[a] is not None:
                        mask = current_ep_masks[a]
                        if len(mask) > TACKLE_ACTION and mask[TACKLE_ACTION] == 1:
                            act_int = TACKLE_ACTION
                        else:
                            act_int = int(policy_actions[i].item())
                    else:
                        act_int = int(policy_actions[i].item())
                    action_dict[a] = act_int
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

                if infos:
                    for inf in infos.values():
                        last_info = inf
                        break

                tick_data = {
                    "tick": ep_length - 1,
                    "actions": action_dict,
                    "action_counts": action_counts.copy(),
                    "shared_reward": shared_rew,
                    "event_code": last_info.get("event_code"),
                    "event_type": last_info.get("event", {}).get("type") if isinstance(last_info.get("event"), dict) else None,
                    "ball_owner": last_info.get("ball_owner_agent_idx"),
                    "score": last_info.get("score", {}),
                    "forced_tackle": any(v == TACKLE_ACTION for v in action_dict.values()),
                }

                if isinstance(env.reward_adapter, InstrumentedAdapter) and env.reward_adapter.tick_log:
                    last_tick_log = env.reward_adapter.tick_log[-1]
                    tick_data["adapter_delta"] = last_tick_log["delta"]
                    tick_data["adapter_base"] = last_tick_log["base_rewards"]
                    tick_data["adapter_shaped"] = last_tick_log["shaped_rewards"]
                    tick_data["step_events"] = [
                        {"type": e.get("type"), "team": e.get("team"), "agent_id": e.get("agent_id")}
                        for e in last_tick_log["step_events"]
                    ]
                    if tick_data["forced_tackle"]:
                        tackle_ticks.append(tick_data)

                ep_tick_log.append(tick_data)

                if done:
                    break

            episode_summary = {
                "episode": ep,
                "seed": ep_seed,
                "goal": int(last_info.get("score", {}).get("left", 0) > 0),
                "tackle_actions": sum(1 for t in ep_tick_log if t.get("forced_tackle")),
                "shot_actions": sum(1 for t in ep_tick_log if any(v == 12 for v in t.get("actions", {}).values())),
                "pass_actions": sum(1 for t in ep_tick_log if any(v in (9, 10, 11) for v in t.get("actions", {}).values())),
                "total_reward": ep_reward,
                "length": ep_length,
                "action_distribution": {ACTION_NAMES[i]: action_counts[i] for i in range(19)},
                "tackle_ticks": tackle_ticks,
                "tick_log": ep_tick_log,
            }
            episodes_data.append(episode_summary)

            print(f"Ep {ep+1:3d}/{num_episodes} | seed={ep_seed} | "
                  f"Goal={episode_summary['goal']} | Tackles={episode_summary['tackle_actions']} | "
                  f"Shots={episode_summary['shot_actions']} | Passes={episode_summary['pass_actions']} | "
                  f"Reward={ep_reward:+.3f}")

    finally:
        env.close()

    # Aggregate metrics
    tackle_ticks_all = [t for ep in episodes_data for t in ep.get("tackle_ticks", [])]

    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256_of(checkpoint_path),
        "scenario": scenario,
        "num_episodes": num_episodes,
        "deterministic": True,
        "timestamp_iso": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode().strip(),
        "probe_type": "forced_tackle_offball_left",
        "overall": {
            "goal_rate_pct": 100.0 * sum(1 for ep in episodes_data if ep["goal"]) / max(len(episodes_data), 1),
            "mean_tackles_per_ep": float(np.mean([ep["tackle_actions"] for ep in episodes_data])) if episodes_data else 0.0,
            "mean_shots_per_ep": float(np.mean([ep["shot_actions"] for ep in episodes_data])) if episodes_data else 0.0,
            "mean_passes_per_ep": float(np.mean([ep["pass_actions"] for ep in episodes_data])) if episodes_data else 0.0,
            "mean_reward": float(np.mean([ep["total_reward"] for ep in episodes_data])) if episodes_data else 0.0,
        },
        "tackle_tick_events": tackle_ticks_all,
        "episodes": episodes_data,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Forced Tackle Probe")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MAPPO checkpoint")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=500000)
    parser.add_argument("--output-dir", type=str, default="training/models")
    args = parser.parse_args()

    summary = evaluate_forced_tackle_probe(
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        base_seed=args.base_seed,
    )

    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    json_path = os.path.join(args.output_dir, f"tackle_forensics_forced_probe_{ckpt_name}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nForced probe results saved to: {json_path}")


if __name__ == "__main__":
    main()
