"""
Forced-tackle probe for Experiment D.

Measures the reward/event consequences of a tackle when the policy itself
never selects tackle. Forces action 16 (SLIDING) on off-ball left agents
for a configurable number of ticks, while letting on-ball agents act normally.

Outputs:
  training/models/tackle_forensics_forced_probe_<ckpt>.json

This is measurement-only. It does not change production code.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor
from training.mappo_rollout import unwrap_obs, unwrap_masks, _mask_matrix
from training.reward_adapters import AttackingDrillRewardAdapter

TACKLE_ACTION = 16


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

        # Call super() exactly once
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
            "step_events": [
                {"type": e.get("type"), "team": e.get("team"), "agent_id": e.get("agent_id")}
                for e in step_events
            ],
            "info_ground_truth": info_ground_truth,
            "actions": actions,
        })
        return result


def run_forced_tackle_probe(
    checkpoint_path: str,
    scenario: str = "academy_3_vs_1_with_keeper",
    num_episodes: int = 5,
    force_ticks: int = 20,
    base_seed: int = 500000,
    bridge_port: int = 5050,
) -> Dict[str, Any]:
    """Run forced-tackle probe on a MAPPO checkpoint."""

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 60)
    print("FORCED-TACKLE PROBE")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Scenario   : {scenario}")
    print(f"Episodes   : {num_episodes}")
    print(f"Force ticks: {force_ticks} (off-ball left agents forced to SLIDING)")
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
            goal_scored = 0
            tackle_actions = 0
            forced_tackle_ticks = 0
            last_info = {}

            while True:
                current_agents = list(env.agents if env.agents else controllable_agents)
                local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
                mask_matrix = _mask_matrix(current_ep_masks, current_agents)

                with torch.no_grad():
                    dist = actor(torch.from_numpy(local_obs).float(), torch.tensor(mask_matrix, dtype=torch.bool))
                    actions_tensor = dist.logits.argmax(dim=-1)

                action_dict = {}
                for i, a in enumerate(current_agents):
                    act_int = int(actions_tensor[i].item())
                    # Force tackle on off-ball left agents for the first N ticks
                    if (a.startswith("left_") and a in current_ep_masks and
                            current_ep_masks[a] is not None and
                            current_ep_masks[a][TACKLE_ACTION] == 1 and
                            ep_length < force_ticks):
                        act_int = TACKLE_ACTION
                        forced_tackle_ticks += 1
                    action_dict[a] = act_int
                    if act_int == TACKLE_ACTION:
                        tackle_actions += 1

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

                # Capture tick data
                tick_data = {
                    "tick": ep_length - 1,
                    "actions": action_dict,
                    "forced_tackle": ep_length - 1 < force_ticks,
                    "shared_reward": shared_rew,
                    "event_code": last_info.get("event_code"),
                    "event_type": last_info.get("event", {}).get("type") if isinstance(last_info.get("event"), dict) else None,
                    "ball_owner": last_info.get("ball_owner_agent_idx"),
                    "score": last_info.get("score", {}),
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
                "forced_tackle_ticks": forced_tackle_ticks,
                "total_reward": ep_reward,
                "length": ep_length,
                "tick_log": ep_tick_log,
            }
            episodes_data.append(episode_summary)

            print(f"Ep {ep+1:3d}/{num_episodes} | seed={ep_seed} | "
                  f"Goal={is_goal} | Tackles={tackle_actions} (forced={forced_tackle_ticks}) | "
                  f"Reward={ep_reward:+.3f}")

    finally:
        env.close()

    # Aggregate metrics
    tackle_ticks = [t for ep in episodes_data for t in ep["tick_log"] if t.get("forced_tackle")]

    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256_of(checkpoint_path),
        "scenario": scenario,
        "num_episodes": num_episodes,
        "force_ticks": force_ticks,
        "deterministic": True,
        "timestamp_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode().strip(),
        "overall": {
            "goal_rate_pct": 100.0 * sum(1 for ep in episodes_data if ep["goal"]) / max(len(episodes_data), 1),
            "mean_tackles_per_ep": float(np.mean([ep["tackle_actions"] for ep in episodes_data])) if episodes_data else 0.0,
            "mean_reward": float(np.mean([ep["total_reward"] for ep in episodes_data])) if episodes_data else 0.0,
        },
        "tackle_tick_analysis": {
            "total_tackle_ticks": len(tackle_ticks),
            "event_code_histogram": {},
            "step_events_contain_turnover_conceded_team_right": 0,
            "mean_delta_on_tackle_ticks": 0.0,
            "mean_delta_on_non_tackle_offball_ticks": 0.0,
        },
        "episodes": episodes_data,
    }

    # Analyze tackle ticks
    if tackle_ticks:
        event_codes = {}
        turnover_count = 0
        tackle_deltas = []
        non_tackle_deltas = []

        for ep in episodes_data:
            for t in ep["tick_log"]:
                if t.get("forced_tackle"):
                    ec = t.get("event_code")
                    if ec is not None:
                        event_codes[ec] = event_codes.get(ec, 0) + 1
                    events = t.get("step_events", [])
                    for e in events:
                        if e.get("type") == "TURNOVER_CONCEDED" and e.get("team") == "right":
                            turnover_count += 1
                    # Sum delta across left agents
                    delta_sum = sum(t.get("adapter_delta", {}).get(a, 0.0) for a in t.get("adapter_delta", {}))
                    tackle_deltas.append(delta_sum)
                else:
                    delta_sum = sum(t.get("adapter_delta", {}).get(a, 0.0) for a in t.get("adapter_delta", {}))
                    non_tackle_deltas.append(delta_sum)

        summary["tackle_tick_analysis"]["event_code_histogram"] = event_codes
        summary["tackle_tick_analysis"]["step_events_contain_turnover_conceded_team_right"] = turnover_count
        summary["tackle_tick_analysis"]["mean_delta_on_tackle_ticks"] = float(np.mean(tackle_deltas)) if tackle_deltas else 0.0
        summary["tackle_tick_analysis"]["mean_delta_on_non_tackle_offball_ticks"] = float(np.mean(non_tackle_deltas)) if non_tackle_deltas else 0.0

    return summary


def main():
    parser = argparse.ArgumentParser(description="Forced-tackle probe for Experiment D")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MAPPO checkpoint")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--force-ticks", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=500000)
    parser.add_argument("--output-dir", type=str, default="training/models")
    args = parser.parse_args()

    summary = run_forced_tackle_probe(
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        force_ticks=args.force_ticks,
        base_seed=args.base_seed,
    )

    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    json_path = os.path.join(args.output_dir, f"tackle_forensics_forced_probe_{ckpt_name}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nForced-probe results saved to: {json_path}")


if __name__ == "__main__":
    main()
