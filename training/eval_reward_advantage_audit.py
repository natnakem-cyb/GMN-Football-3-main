"""
GMN-Football-3 — Reward / Advantage Audit around PASS–SHOT vs MOVE

Audit-only module. No training, no reward/GAE/mask/network changes.
Produces:
  - training/results/reward_advantage_audit_summary.csv
  - training/results/reward_advantage_audit_detail.json
  - training/results/REWARD_ADVANTAGE_AUDIT.md

Checkpoints audited:
  - Fresh 50k:  mappo_academy_3_vs_1_with_keeper_seed{N}_freshtrain_50176.pt
  - Mix final:  mappo_academy_3_vs_1_with_keeper_seed{N}_mixscript_50176.pt
  Seeds: 42, 123, 7, 999

Arms:
  - ONBALL-pi   : pure policy, 20 episodes
  - ONBALL-Pass : forced SHORT_PASS once at t=0, then policy, 20 episodes
  - ONBALL-Shot : forced SHOT once at t=0, then policy, 20 episodes

GAE convention matches training: gamma=0.99, lambda=0.95,
bootstrap = critic(next_local_obs) when rollout is truncated mid-episode,
else 0.0 on genuine termination.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTION_NAMES = [
    "IDLE", "LEFT", "RIGHT", "UP", "DOWN",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
    "LONG_PASS", "HIGH_PASS", "SHORT_PASS",
    "SHOT", "SPRINT",
    "RELEASE_DIRECTION", "RELEASE_SPRINT",
    "SLIDING", "DRIBBLE", "RELEASE_DRIBBLE",
]
BALL_ACTION_IDS = frozenset((9, 10, 11, 12, 16, 17, 18))
NON_BALL_ACTION_IDS = frozenset(range(9)) | frozenset((13, 14, 15))
PASS_ACTION_IDS = frozenset((9, 10, 11))
SHOT_ACTION_IDS = frozenset((12,))

FORCED_PASS_IDX = 11   # SHORT_PASS
FORCED_SHOT_IDX = 12  # SHOT
OBS_DIM = 127
ACTION_DIM = 19
OBS_BALL_OWNERSHIP_SLICE = slice(94, 97)

WINDOW_H = 5  # ticks after attempt to look for pass_completed / shot event

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    obs_dim = ckpt.get("obs_dim", OBS_DIM)
    action_dim = ckpt.get("action_dim", ACTION_DIM)
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


def _select_policy_actions(actor, local_obs, mask_matrix, deterministic=True):
    with torch.no_grad():
        obs_tensor = torch.from_numpy(local_obs).float()
        mask_tensor = torch.tensor(mask_matrix, dtype=torch.bool)
        dist = actor(obs_tensor, mask_tensor)
        logits = dist.logits
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
    return actions.cpu().numpy()


def _build_action_dict(current_agents, policy_actions, force_idx, force_agent_idx, force_this_tick):
    action_dict = {}
    for i, agent in enumerate(current_agents):
        if i == force_agent_idx and force_this_tick and force_idx is not None:
            action_dict[agent] = int(force_idx)
        else:
            action_dict[agent] = int(policy_actions[i])
    return action_dict


# ---------------------------------------------------------------------------
# Episode collector with full tick telemetry
# ---------------------------------------------------------------------------
def collect_audit_episodes(
    actor,
    critic,
    checkpoint_path: str,
    checkpoint_sha256: str,
    checkpoint_timesteps: int,
    seed: int,
    arm: str,
    scenario: str,
    num_episodes: int,
    deterministic: bool,
    base_seed: int,
) -> Dict[str, Any]:
    """Run episodes and return per-tick audit records + episode summaries."""
    force_idx = None
    if arm == "ONBALL-Pass":
        force_idx = FORCED_PASS_IDX
    elif arm == "ONBALL-Shot":
        force_idx = FORCED_SHOT_IDX

    env = GMNMultiAgentEnv(scenario=scenario, auto_start_bridge=True, port=5050)
    controllable_agents = list(env.possible_agents)

    episodes_data: List[Dict[str, Any]] = []
    all_ticks: List[Dict[str, Any]] = []

    try:
        for ep in range(num_episodes):
            ep_seed = base_seed + ep * 1009
            obs_dict, _ = env.reset(seed=ep_seed)
            current_ep_masks = unwrap_masks(obs_dict)
            obs_dict = unwrap_obs(obs_dict)

            reset_ball_owner_agent_idx = getattr(env, "_last_ball_owner_agent_idx", 255)
            t0_valid_possession = False
            t0_pass_legal = False
            t0_shot_legal = False
            first_agent = list(obs_dict.keys())[0]
            agent_obs = obs_dict[first_agent]
            if isinstance(agent_obs, dict) and "observation" in agent_obs:
                raw_obs = agent_obs["observation"]
                if hasattr(raw_obs, "__len__") and len(raw_obs) >= 97:
                    t0_valid_possession = bool(raw_obs[95] == 1.0)
                if "action_mask" in agent_obs:
                    mask = agent_obs["action_mask"]
                    if hasattr(mask, "__len__") and len(mask) >= 18:
                        t0_pass_legal = int(mask[FORCED_PASS_IDX])
                        t0_shot_legal = int(mask[FORCED_SHOT_IDX])
            else:
                # obs_dict is already unwrapped; use current_ep_masks
                if first_agent in current_ep_masks and current_ep_masks[first_agent] is not None:
                    mask = current_ep_masks[first_agent]
                    if hasattr(mask, "__len__") and len(mask) >= 18:
                        t0_pass_legal = int(mask[FORCED_PASS_IDX])
                        t0_shot_legal = int(mask[FORCED_SHOT_IDX])
                if hasattr(agent_obs, "__len__") and len(agent_obs) >= 97:
                    t0_valid_possession = bool(agent_obs[95] == 1.0)

            current_agents = list(env.agents if env.agents else controllable_agents)
            ball_owner_agent_idx = reset_ball_owner_agent_idx
            if not (0 <= ball_owner_agent_idx < len(current_agents)):
                ball_owner_agent_idx = 0

            if arm == "ONBALL-Pass":
                t0_valid_force = bool(t0_pass_legal) and t0_valid_possession
            elif arm == "ONBALL-Shot":
                t0_valid_force = bool(t0_shot_legal) and t0_valid_possession
            else:
                t0_valid_force = True

            force_this_episode = force_idx is not None and t0_valid_force
            force_tick = 0 if force_this_episode else -1
            force_agent_idx = ball_owner_agent_idx if force_this_episode else -1

            ep_ticks: List[Dict[str, Any]] = []
            ep_reward = 0.0
            ep_length = 0
            policy_actions_count = {"pass_short": 0, "pass_long": 0, "pass_high": 0, "shot": 0, "tackle": 0}
            invalid_reason = None
            if not t0_valid_possession:
                invalid_reason = "t0_possession_invalid"
            elif arm == "ONBALL-Pass" and not t0_pass_legal:
                invalid_reason = "t0_pass_illegal"
            elif arm == "ONBALL-Shot" and not t0_shot_legal:
                invalid_reason = "t0_shot_illegal"

            tick_idx = 0
            while True:
                current_agents = list(env.agents if env.agents else controllable_agents)
                local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
                mask_matrix = _mask_matrix(current_ep_masks, current_agents)

                policy_actions = _select_policy_actions(actor, local_obs, mask_matrix, deterministic)

                action_dict = _build_action_dict(
                    current_agents, policy_actions, force_idx, force_agent_idx,
                    force_this_episode and tick_idx == force_tick,
                )

                # Actual actions after potential force override
                actual_actions = np.array([action_dict[a] for a in current_agents], dtype=np.int64)

                with torch.no_grad():
                    obs_tensor = torch.from_numpy(local_obs).float()
                    mask_tensor = torch.tensor(mask_matrix, dtype=torch.bool)
                    dist = actor(obs_tensor, mask_tensor)
                    # Logprobs for the ACTUAL actions taken (policy or forced)
                    logprobs = dist.log_prob(torch.tensor(actual_actions, dtype=torch.long)).cpu().numpy()
                    value = float(critic(obs_tensor.unsqueeze(0)).item()) if critic is not None else None

                obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
                current_ep_masks = unwrap_masks(obs_dict)
                obs_dict = unwrap_obs(obs_dict)

                terminated = any(terms.values()) if terms else False
                truncated = any(truncs.values()) if truncs else False
                done = terminated or truncated or not env.agents

                shared_rew = float(rewards[current_agents[0]]) if current_agents and current_agents[0] in rewards else 0.0
                ep_reward += shared_rew
                ep_length += 1

                event_code = getattr(env, "_last_frame_event_code", None)
                event_type = _event_type_from_code(event_code)
                ball_owner_agent_idx_post = getattr(env, "_last_ball_owner_agent_idx", 255)

                controlled_has_ball = (ball_owner_agent_idx_post == 0)

                tick_data = {
                    "tick": tick_idx,
                    "seed": ep_seed,
                    "episode": ep,
                    "arm": arm,
                    "action_selected": int(actual_actions[0]) if len(actual_actions) > 0 else -1,
                    "all_actions": [int(a) for a in actual_actions],
                    "forced": bool(force_this_episode and tick_idx == force_tick),
                    "forced_action": force_idx if (force_this_episode and tick_idx == force_tick) else None,
                    "t0_valid": t0_valid_force,
                    "mask_sum": int(np.sum(mask_matrix[0])) if len(mask_matrix) > 0 else None,
                    "pass_legal": int(mask_matrix[0][FORCED_PASS_IDX]) if len(mask_matrix) > 0 and len(mask_matrix[0]) > FORCED_PASS_IDX else None,
                    "shot_legal": int(mask_matrix[0][FORCED_SHOT_IDX]) if len(mask_matrix) > 0 and len(mask_matrix[0]) > FORCED_SHOT_IDX else None,
                    "has_ball": controlled_has_ball,
                    "owner": current_agents[ball_owner_agent_idx_post] if 0 <= ball_owner_agent_idx_post < len(current_agents) else None,
                    "reward": shared_rew,
                    "event_code": event_code,
                    "event_type": event_type,
                    "terminated": terminated,
                    "truncated": truncated,
                    "done": done,
                    "value": value,
                    "logprob": float(logprobs[0]) if len(logprobs) > 0 else None,
                    "invalid_reason": invalid_reason if tick_idx == 0 else None,
                }
                ep_ticks.append(tick_data)
                all_ticks.append(tick_data)
                tick_idx += 1

                if done:
                    break

            # Goal detection
            score_left = infos.get(current_agents[0], {}).get("score", {}).get("left", 0) if infos else 0
            is_goal = score_left > 0

            episode_summary = {
                "seed": seed,
                "episode": ep,
                "arm": arm,
                "checkpoint": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_timesteps": checkpoint_timesteps,
                "scenario": scenario,
                "ep_seed": ep_seed,
                "valid_t0_possession": t0_valid_possession,
                "valid_force_mask": t0_valid_force,
                "invalid_reason": invalid_reason,
                "force_applied": force_this_episode,
                "force_tick": force_tick,
                "pass_completed": 1 if any(t.get("event_type") == "pass_completed" for t in ep_ticks) else 0,
                "shot_event": 1 if any(t.get("event_type") == "shot" for t in ep_ticks) else 0,
                "goal": int(is_goal),
                "policy_pass_count": sum(policy_actions_count.values()),
                "policy_shot_count": policy_actions_count["shot"],
                "policy_tackle_count": policy_actions_count["tackle"],
                "force_reward": 0.0,
                "mean_post_force_reward": None,
                "event_histogram": {},
                "episode_reward": ep_reward,
                "episode_length": ep_length,
            }
            episodes_data.append(episode_summary)

            if (ep + 1) % 5 == 0:
                print(f"  [{arm} seed={seed}] Ep {ep+1:3d}/{num_episodes} | "
                      f"valid={t0_valid_force} | force={'Y' if force_this_episode else 'N'} | "
                      f"PASS={episode_summary['pass_completed']} SHOT={episode_summary['shot_event']} "
                      f"GOAL={is_goal} | Reward={ep_reward:+.3f} | Len={ep_length}")

    finally:
        env.close()

    return {
        "seed": seed,
        "arm": arm,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_timesteps": checkpoint_timesteps,
        "scenario": scenario,
        "deterministic": deterministic,
        "base_seed": base_seed,
        "gamma": 0.99,
        "lambda": 0.95,
        "episodes": episodes_data,
        "ticks": all_ticks,
    }


# ---------------------------------------------------------------------------
# Cell classification
# ---------------------------------------------------------------------------
def classify_cells(episodes: List[Dict[str, Any]], ticks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Classify each tick into an audit cell.

    Returns a list of dicts, one per tick, with cell label and metadata.
    """
    # Build event timeline per episode for window lookback
    ep_event_index: Dict[int, List[Dict[str, Any]]] = {}
    for t in ticks:
        ep = t["episode"]
        ep_event_index.setdefault(ep, []).append(t)

    classified = []
    for t in ticks:
        ep = t["episode"]
        ep_ticks = ep_event_index.get(ep, [])
        tick_idx = t["tick"]
        action = t["action_selected"]
        forced = t["forced"]
        done = t["done"]
        terminated = t["terminated"]
        truncated = t["truncated"]
        event_type = t.get("event_type")
        reward = t.get("reward", 0.0)
        value = t.get("value")
        logprob = t.get("logprob")
        mask_sum = t.get("mask_sum")
        pass_legal = t.get("pass_legal")
        shot_legal = t.get("shot_legal")

        # Determine source
        if forced:
            source = "probe_force"
        else:
            source = "pure_pi"

        # Determine if on-ball (PASS/SHOT legal) at decision time
        on_ball = bool(pass_legal or shot_legal)

        # Any ball action this tick?
        any_ball_action = any(a in BALL_ACTION_IDS for a in t.get("all_actions", [t["action_selected"]]))

        # Skip terminal ticks for outcome windows
        is_terminal = done

        # Window after this tick for outcome events
        window_ticks = []
        for future in ep_ticks:
            if future["tick"] > tick_idx and future["tick"] <= tick_idx + WINDOW_H:
                window_ticks.append(future)
        window_event_types = {ft.get("event_type") for ft in window_ticks if ft.get("event_type")}
        window_has_pass_completed = "pass_completed" in window_event_types
        window_has_shot = any(et and et.lower().startswith("shot") for et in window_event_types)
        window_has_goal = any(ft.get("goal") for ft in window_ticks if ft.get("goal"))

        # Cell classification
        cell = None
        if action in PASS_ACTION_IDS:
            if window_has_pass_completed:
                cell = "PASS_complete"
            elif window_has_goal:
                cell = "PASS_complete"  # goal implies successful pass sequence
            else:
                cell = "PASS_fail"
        elif action in SHOT_ACTION_IDS:
            if window_has_shot or window_has_goal:
                cell = "SHOT_event"
            else:
                cell = "SHOT_fail"
        elif action in NON_BALL_ACTION_IDS and not is_terminal and not any_ball_action:
            cell = "MOVE_safe"
        else:
            cell = "OTHER"

        classified.append({
            "seed": t["seed"],
            "episode": ep,
            "tick": tick_idx,
            "arm": t["arm"],
            "source": source,
            "cell": cell,
            "action": action,
            "action_name": ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action),
            "forced": forced,
            "on_ball": on_ball,
            "any_ball_action": any_ball_action,
            "reward": reward,
            "value": value,
            "logprob": logprob,
            "advantage": t.get("advantage"),
            "event_type": event_type,
            "pass_legal": pass_legal,
            "shot_legal": shot_legal,
            "mask_sum": mask_sum,
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
        })
    return classified


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_cells(classified: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate statistics per cell, per source, and overall."""
    from collections import defaultdict
    cells: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"rewards": [], "values": [], "logprobs": [], "count": 0})
    source_cells: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: {"rewards": [], "values": [], "logprobs": [], "count": 0}))

    for row in classified:
        cell = row["cell"]
        source = row["source"]
        reward = row.get("reward", 0.0)
        value = row.get("value")
        logprob = row.get("logprob")

        cells[cell]["rewards"].append(reward)
        if value is not None:
            cells[cell]["values"].append(value)
        if logprob is not None:
            cells[cell]["logprobs"].append(logprob)
        cells[cell]["count"] += 1

        source_cells[source][cell]["rewards"].append(reward)
        if value is not None:
            source_cells[source][cell]["values"].append(value)
        if logprob is not None:
            source_cells[source][cell]["logprobs"].append(logprob)
        source_cells[source][cell]["count"] += 1

    def summarize(cell_data):
        rewards = np.array(cell_data["rewards"], dtype=np.float32)
        values = np.array(cell_data["values"], dtype=np.float32) if cell_data["values"] else np.array([])
        logprobs = np.array(cell_data["logprobs"], dtype=np.float32) if cell_data["logprobs"] else np.array([])
        return {
            "count": int(cell_data["count"]),
            "reward_mean": float(np.mean(rewards)) if len(rewards) else None,
            "reward_std": float(np.std(rewards)) if len(rewards) else None,
            "reward_median": float(np.median(rewards)) if len(rewards) else None,
            "value_mean": float(np.mean(values)) if len(values) else None,
            "value_std": float(np.std(values)) if len(values) else None,
            "value_median": float(np.median(values)) if len(values) else None,
            "logprob_mean": float(np.mean(logprobs)) if len(logprobs) else None,
            "logprob_std": float(np.std(logprobs)) if len(logprobs) else None,
            "logprob_median": float(np.median(logprobs)) if len(logprobs) else None,
        }

    result = {
        "overall": {cell: summarize(data) for cell, data in cells.items()},
        "by_source": {
            source: {cell: summarize(data) for cell, data in src_cells.items()}
            for source, src_cells in source_cells.items()
        },
    }
    return result


# ---------------------------------------------------------------------------
# GAE computation per episode
# ---------------------------------------------------------------------------
def compute_episode_gae(ep_ticks: List[Dict[str, Any]], gamma=0.99, lam=0.95) -> List[float]:
    """Compute GAE advantages for a single episode."""
    T = len(ep_ticks)
    if T == 0:
        return []

    rewards = np.array([t["reward"] for t in ep_ticks], dtype=np.float32)
    values = np.array([t["value"] for t in ep_ticks], dtype=np.float32)
    dones = np.array([t["done"] for t in ep_ticks], dtype=bool)

    # Bootstrap: if last step is truncated (not terminated), use last value as bootstrap
    # If terminated, bootstrap = 0.0
    if dones[-1]:
        bootstrap = 0.0
    else:
        bootstrap = values[-1]

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = bootstrap if t == T - 1 else values[t + 1]
        next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae

    return advantages.tolist()


def attach_gae_to_ticks(episodes: List[Dict[str, Any]], ticks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute GAE per episode and attach advantage to each tick."""
    ep_ticks_map: Dict[int, List[Dict[str, Any]]] = {}
    for t in ticks:
        ep = t["episode"]
        ep_ticks_map.setdefault(ep, []).append(t)

    for ep, ep_tick_list in ep_ticks_map.items():
        ep_tick_list.sort(key=lambda x: x["tick"])
        advantages = compute_episode_gae(ep_tick_list)
        for i, t in enumerate(ep_tick_list):
            t["advantage"] = float(advantages[i]) if i < len(advantages) else 0.0

    # Flatten back
    all_with_gae = []
    for ep in sorted(ep_ticks_map.keys()):
        all_with_gae.extend(ep_ticks_map[ep])
    return all_with_gae


# ---------------------------------------------------------------------------
# Main audit runner
# ---------------------------------------------------------------------------
def run_audit(seeds, checkpoints, scenario, num_episodes, deterministic, base_seed, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    all_classified = []

    # checkpoints is a list of (seed, ckpt_label, ckpt_path, ckpt_timesteps)
    for seed, ckpt_label, ckpt_path, ckpt_timesteps in checkpoints:
        print(f"\n{'='*70}")
        print(f"AUDIT: {ckpt_label} seed={seed}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*70}")

        if not os.path.exists(ckpt_path):
            print(f"  SKIP: checkpoint not found")
            continue

        ckpt_sha256 = sha256_of(ckpt_path)
        actor, critic, obs_dim, action_dim, ckpt_ts = load_checkpoint(ckpt_path)

        for arm in ["ONBALL-pi", "ONBALL-Pass", "ONBALL-Shot"]:
            print(f"\n  Arm: {arm}")
            result = collect_audit_episodes(
                actor=actor,
                critic=critic,
                checkpoint_path=ckpt_path,
                checkpoint_sha256=ckpt_sha256,
                checkpoint_timesteps=ckpt_timesteps or 0,
                seed=seed,
                arm=arm,
                scenario=scenario,
                num_episodes=num_episodes,
                deterministic=deterministic,
                base_seed=base_seed,
            )
            # Attach GAE
            result["ticks"] = attach_gae_to_ticks(result["episodes"], result["ticks"])
            all_results.append(result)

            # Classify cells
            classified = classify_cells(result["episodes"], result["ticks"])
            for row in classified:
                row["checkpoint_label"] = ckpt_label
                row["checkpoint_timesteps"] = ckpt_timesteps
                row["audit_seed"] = seed
            all_classified.extend(classified)

    # Aggregate
    agg = aggregate_cells(all_classified)

    # Write detail JSON
    detail_path = os.path.join(output_dir, "reward_advantage_audit_detail.json")
    with open(detail_path, "w") as f:
        json.dump({
            "results": all_results,
            "classified_ticks": all_classified,
            "aggregate": agg,
        }, f, indent=2, default=lambda o: None if o is None else float(o))
    print(f"\nDetail JSON: {detail_path}")

    # Write summary CSV
    csv_path = os.path.join(output_dir, "reward_advantage_audit_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "checkpoint_label", "seed", "source", "cell",
            "count", "reward_mean", "reward_std", "reward_median",
            "value_mean", "value_std", "value_median",
            "logprob_mean", "logprob_std", "logprob_median",
            "advantage_mean", "advantage_std", "advantage_median", "fraction_negative",
        ])
        for seed, ckpt_label, _, _ in checkpoints:
            # Filter classified ticks for this ckpt+seed
            subset = [r for r in all_classified if r.get("checkpoint_label") == ckpt_label and r.get("audit_seed") == seed]
            for source in ["probe_force", "pure_pi"]:
                src_subset = [r for r in subset if r.get("source") == source]
                cells = {}
                for r in src_subset:
                    cell = r["cell"]
                    cells.setdefault(cell, []).append(r)

                for cell, rows in cells.items():
                    advs = [r.get("advantage", 0.0) for r in rows]
                    rewards = [r.get("reward", 0.0) for r in rows]
                    values = [r.get("value") for r in rows if r.get("value") is not None]
                    logprobs = [r.get("logprob") for r in rows if r.get("logprob") is not None]
                    writer.writerow([
                        ckpt_label, seed, source, cell,
                        len(rows),
                        np.mean(rewards) if rewards else None,
                        np.std(rewards) if rewards else None,
                        np.median(rewards) if rewards else None,
                        np.mean(values) if values else None,
                        np.std(values) if values else None,
                        np.median(values) if values else None,
                        np.mean(logprobs) if logprobs else None,
                        np.std(logprobs) if logprobs else None,
                        np.median(logprobs) if logprobs else None,
                        np.mean(advs) if advs else None,
                        np.std(advs) if advs else None,
                        np.median(advs) if advs else None,
                        sum(1 for a in advs if a < 0) / max(len(advs), 1) if advs else None,
                    ])
    print(f"Summary CSV: {csv_path}")

    return all_results, all_classified, agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Reward / Advantage Audit")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper_onball")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7, 999])
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--base-seed", type=int, default=500000)
    parser.add_argument("--output-dir", type=str, default="training/results")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(base_dir, "training", "models")

    seeds = args.seeds
    checkpoints = []
    for seed in seeds:
        fresh_path = os.path.join(model_dir, f"mappo_academy_3_vs_1_with_keeper_seed{seed}_freshtrain_50176.pt")
        mix_path = os.path.join(model_dir, f"mappo_academy_3_vs_1_with_keeper_seed{seed}_mixscript_50176.pt")
        if os.path.exists(fresh_path):
            checkpoints.append((seed, "fresh", fresh_path, 50176))
        else:
            print(f"WARN: missing fresh checkpoint for seed {seed}: {fresh_path}")
        if os.path.exists(mix_path):
            checkpoints.append((seed, "mixscript", mix_path, 50176))
        else:
            print(f"WARN: missing mixscript checkpoint for seed {seed}: {mix_path}")

    if not checkpoints:
        print("ERROR: no checkpoints found")
        sys.exit(1)

    run_audit(
        seeds=seeds,
        checkpoints=checkpoints,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
