"""
GMN-Football-3 — Experiment F_act: Forced One-Tick PASS/SHOT on Validated μ-onball
Measurement-only: probes whether the environment can execute a commanded football
action while the frozen policy refuses to select that action.

Arms:
  A — ONBALL-π     : frozen policy only (sanity)
  B — ONBALL-Pass  : force PASS once at t=0, then policy
  C — ONBALL-Shot  : force SHOT once at t=0, then policy

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
    "LONG_PASS", "HIGH_PASS", "SHORT_PASS",
    "SHOT", "SPRINT",
    "RELEASE_DIRECTION", "RELEASE_SPRINT",
    "SLIDING", "DRIBBLE", "RELEASE_DRIBBLE"
]
TACKLE_ACTION = 16
SHOT_ACTIONS = {12}
PASS_ACTIONS = {9, 10, 11}
BALL_ACTION_IDS = frozenset((9, 10, 11, 12, 17))
FORCED_PASS_IDX = 11   # SHORT_PASS in production discrete action space
FORCED_SHOT_IDX = 12  # SHOT in production discrete action space

OBS_DIM = 127
ACTION_DIM = 19
OBS_BALL_OWNERSHIP_SLICE = slice(94, 97)  # [no-one, left, right]
OBS_BALL_POS_SLICE = slice(88, 91)        # ball (x, y, z)


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


def _capture_t0_telemetry(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Capture reset-time ownership and mask telemetry before first step."""
    t0_ball_ownership = None
    t0_controlled_has_ball = False
    t0_mask_sum = None
    t0_pass_legal = None
    t0_shot_legal = None
    t0_dribble_legal = None
    t0_mask = None

    if obs_dict:
        first_agent = list(obs_dict.keys())[0]
        agent_obs = obs_dict[first_agent]
        if isinstance(agent_obs, dict) and "observation" in agent_obs:
            raw_obs = agent_obs["observation"]
            if hasattr(raw_obs, "__len__") and len(raw_obs) >= 97:
                t0_ball_ownership = raw_obs[94:97].tolist()
                t0_controlled_has_ball = bool(raw_obs[95] == 1.0)
                if "action_mask" in agent_obs:
                    mask = agent_obs["action_mask"]
                    if hasattr(mask, "__len__") and len(mask) >= 18:
                        t0_mask_sum = int(np.sum(mask))
                        t0_pass_legal = int(mask[FORCED_PASS_IDX])
                        t0_shot_legal = int(mask[FORCED_SHOT_IDX])
                        t0_dribble_legal = int(mask[17])
                        t0_mask = mask.tolist()

    return {
        "t0_ball_ownership": t0_ball_ownership,
        "t0_controlled_has_ball": t0_controlled_has_ball,
        "t0_mask_sum": t0_mask_sum,
        "t0_pass_legal": t0_pass_legal,
        "t0_shot_legal": t0_shot_legal,
        "t0_dribble_legal": t0_dribble_legal,
        "t0_mask": t0_mask,
    }


def _select_policy_actions(
    actor,
    local_obs: np.ndarray,
    mask_matrix: np.ndarray,
    deterministic: bool = True,
) -> np.ndarray:
    """Return action indices from the frozen policy for all agents."""
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


def _build_action_dict(
    current_agents: List[str],
    policy_actions: np.ndarray,
    force_idx: Optional[int],
    force_agent_idx: int,
    force_this_tick: bool,
) -> Dict[str, int]:
    """Build action dict, overriding one agent with forced action if applicable."""
    action_dict = {}
    for i, agent in enumerate(current_agents):
        if i == force_agent_idx and force_this_tick and force_idx is not None:
            action_dict[agent] = int(force_idx)
        else:
            action_dict[agent] = int(policy_actions[i])
    return action_dict


def _event_histogram(ticks: List[Dict[str, Any]]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for tick in ticks:
        ec = tick.get("event_code")
        if ec is not None and ec > 0:
            key = f"event_{ec}"
            hist[key] = hist.get(key, 0) + 1
    return hist


def _has_event(ticks: List[Dict[str, Any]], event_name: str) -> bool:
    for tick in ticks:
        et = tick.get("event_type")
        if et == event_name:
            return True
    return False


def _get_obs_vector(obs_dict: Dict[str, Any], agent: str) -> Optional[np.ndarray]:
    if not obs_dict:
        return None
    agent_obs = obs_dict.get(agent)
    if isinstance(agent_obs, dict) and "observation" in agent_obs:
        return np.asarray(agent_obs["observation"])
    return None


def run_arm(
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
    post_force_window: int = 50,
    bridge_port: int = 5050,
) -> Dict[str, Any]:
    """Run one arm of the F_act experiment for a single seed."""
    force_idx = None
    if arm == "ONBALL-Pass":
        force_idx = FORCED_PASS_IDX
    elif arm == "ONBALL-Shot":
        force_idx = FORCED_SHOT_IDX

    env = GMNMultiAgentEnv(scenario=scenario, auto_start_bridge=True, port=bridge_port)
    controllable_agents = list(env.possible_agents)

    episodes_data: List[Dict[str, Any]] = []

    try:
        for ep in range(num_episodes):
            ep_seed = base_seed + ep * 1009
            obs_dict, _ = env.reset(seed=ep_seed)

            # ---- t=0 ownership capture BEFORE first step ----
            t0_capture = _capture_t0_telemetry(obs_dict)
            current_ep_masks = unwrap_masks(obs_dict)
            obs_dict = unwrap_obs(obs_dict)

            # Determine ball-owning agent index at t=0 from bridge reset state
            reset_ball_owner_agent_idx = getattr(env, "_last_ball_owner_agent_idx", 255)
            t0_valid_possession = bool(t0_capture["t0_controlled_has_ball"])
            if arm == "ONBALL-Pass":
                t0_valid_force = bool(t0_capture["t0_pass_legal"]) and t0_valid_possession
            elif arm == "ONBALL-Shot":
                t0_valid_force = bool(t0_capture["t0_shot_legal"]) and t0_valid_possession
            else:
                t0_valid_force = True

            current_agents = list(env.agents if env.agents else controllable_agents)
            ball_owner_agent_idx = reset_ball_owner_agent_idx
            if not (0 <= ball_owner_agent_idx < len(current_agents)):
                ball_owner_agent_idx = 0

            force_this_episode = force_idx is not None and t0_valid_force
            force_tick = 0 if force_this_episode else -1
            force_agent_idx = ball_owner_agent_idx if force_this_episode else -1

            ep_tick_log: List[Dict[str, Any]] = []
            ep_reward = 0.0
            ep_length = 0
            policy_actions_count = {
                "pass_short": 0,
                "pass_long": 0,
                "pass_high": 0,
                "shot": 0,
                "tackle": 0,
            }
            force_success = False
            force_event_type = None
            force_reward = 0.0
            post_force_rewards: List[float] = []
            possession_snapshots = {}
            ball_position_before = None
            ball_position_after = None
            ownership_before = None
            ownership_after = None
            invalid_reason = None

            if not t0_valid_possession:
                invalid_reason = "t0_possession_invalid"
            elif arm == "ONBALL-Pass" and not t0_capture["t0_pass_legal"]:
                invalid_reason = "t0_pass_illegal"
            elif arm == "ONBALL-Shot" and not t0_capture["t0_shot_legal"]:
                invalid_reason = "t0_shot_illegal"

            tick_idx = 0
            while True:
                current_agents = list(env.agents if env.agents else controllable_agents)
                local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
                mask_matrix = _mask_matrix(current_ep_masks, current_agents)

                policy_actions = _select_policy_actions(actor, local_obs, mask_matrix, deterministic)

                # Pre-step: capture ball state for forced tick from current observation
                if force_this_episode and tick_idx == force_tick:
                    pre_obs = _get_obs_vector(obs_dict, current_agents[0]) if current_agents else None
                    if pre_obs is not None and len(pre_obs) >= 91:
                        ball_position_before = {
                            "x": float(pre_obs[88]),
                            "y": float(pre_obs[89]),
                            "z": float(pre_obs[90]),
                        }
                    ownership_before = current_agents[ball_owner_agent_idx] if 0 <= ball_owner_agent_idx < len(current_agents) else None

                action_dict = _build_action_dict(
                    current_agents,
                    policy_actions,
                    force_idx,
                    force_agent_idx,
                    force_this_episode and tick_idx == force_tick,
                )

                obs_dict, rewards, terms, truncs, infos = env.step(action_dict)

                shared_rew = float(rewards[current_agents[0]]) if current_agents and current_agents[0] in rewards else 0.0
                ep_reward += shared_rew
                ep_length += 1

                term = any(terms.values()) if terms else False
                trunc = any(truncs.values()) if truncs else False
                done = term or trunc or not env.agents

                # Extract event info
                event_code = getattr(env, "_last_frame_event_code", None)
                event_type = _event_type_from_code(event_code)
                ball_owner_agent_idx_post = getattr(env, "_last_ball_owner_agent_idx", 255)

                # Post-step masks
                raw_masks = {}
                if isinstance(obs_dict, dict):
                    for a in current_agents:
                        agent_obs = obs_dict.get(a)
                        if isinstance(agent_obs, dict) and agent_obs.get("action_mask") is not None:
                            raw_masks[a] = np.array(agent_obs["action_mask"], dtype=np.int8)

                current_ep_masks = unwrap_masks(obs_dict)
                obs_dict = unwrap_obs(obs_dict)

                # Post-step ball state for forced tick
                if force_this_episode and tick_idx == force_tick:
                    post_obs = _get_obs_vector(obs_dict, current_agents[0]) if current_agents else None
                    if post_obs is not None and len(post_obs) >= 91:
                        ball_position_after = {
                            "x": float(post_obs[88]),
                            "y": float(post_obs[89]),
                            "z": float(post_obs[90]),
                        }
                    ownership_after = current_agents[ball_owner_agent_idx_post] if 0 <= ball_owner_agent_idx_post < len(current_agents) else None
                    force_reward = shared_rew
                    if event_type in ("pass", "pass_completed"):
                        force_success = True
                        force_event_type = event_type
                    if event_type == "shot":
                        force_success = True
                        force_event_type = event_type

                # Determine action source and policy counts (exclude forced tick)
                action_source = "forced" if (force_this_episode and tick_idx == force_tick) else "policy"
                if action_source == "policy":
                    for i, agent in enumerate(current_agents):
                        act_idx = int(policy_actions[i])
                        if act_idx == FORCED_PASS_IDX:
                            policy_actions_count["pass_short"] += 1
                        elif act_idx == 9:
                            policy_actions_count["pass_long"] += 1
                        elif act_idx == 10:
                            policy_actions_count["pass_high"] += 1
                        if act_idx == FORCED_SHOT_IDX:
                            policy_actions_count["shot"] += 1
                        if act_idx == TACKLE_ACTION:
                            policy_actions_count["tackle"] += 1

                # Post-force reward tracking
                if force_this_episode:
                    post_force_rewards.append(shared_rew)

                # Possession snapshots
                if force_this_episode:
                    if tick_idx == force_tick + 1:
                        possession_snapshots["t1"] = ownership_after
                    elif tick_idx == force_tick + 5:
                        possession_snapshots["t5"] = ownership_after
                    elif tick_idx == force_tick + 20:
                        possession_snapshots["t20"] = ownership_after

                # Compute distances
                d_self_ball = None
                if obs_dict and current_agents:
                    first_obs = _get_obs_vector(obs_dict, current_agents[0])
                    if first_obs is not None and len(first_obs) >= 14:
                        try:
                            ball_x = float(first_obs[88])
                            ball_y = float(first_obs[89])
                            self_x = float(first_obs[0])
                            self_y = float(first_obs[1])
                            d_self_ball = ((self_x - ball_x) ** 2 + (self_y - ball_y) ** 2) ** 0.5
                        except (IndexError, ValueError, TypeError):
                            pass

                controlled_has_ball = (ball_owner_agent_idx_post == 0)

                tick_data = {
                    "tick": tick_idx,
                    "seed": ep_seed,
                    "episode": ep,
                    "arm": arm,
                    "action_selected": int(policy_actions[0]) if len(policy_actions) > 0 else -1,
                    "action_source": action_source,
                    "forced_action": force_idx if (force_this_episode and tick_idx == force_tick) else None,
                    "forced_tick": force_tick if force_this_episode else -1,
                    "t0_valid": t0_valid_force,
                    "mask_sum": int(np.sum(mask_matrix[0])) if len(mask_matrix) > 0 else None,
                    "pass_legal": int(mask_matrix[0][FORCED_PASS_IDX]) if len(mask_matrix) > 0 and len(mask_matrix[0]) > FORCED_PASS_IDX else None,
                    "shot_legal": int(mask_matrix[0][FORCED_SHOT_IDX]) if len(mask_matrix) > 0 and len(mask_matrix[0]) > FORCED_SHOT_IDX else None,
                    "has_ball": controlled_has_ball,
                    "owner": current_agents[ball_owner_agent_idx_post] if 0 <= ball_owner_agent_idx_post < len(current_agents) else None,
                    "d_self_ball": round(d_self_ball, 4) if d_self_ball is not None else None,
                    "reward": shared_rew,
                    "event_code": event_code,
                    "event_type": event_type,
                    "terminated": term,
                    "truncated": trunc,
                    "ball_position": {
                        "x": float(post_obs[88]) if (post_obs is not None and len(post_obs) > 88) else None,
                        "y": float(post_obs[89]) if (post_obs is not None and len(post_obs) > 89) else None,
                        "z": float(post_obs[90]) if (post_obs is not None and len(post_obs) > 90) else None,
                    } if 'post_obs' in dir() else {"x": None, "y": None, "z": None},
                    "ball_owner": current_agents[ball_owner_agent_idx_post] if 0 <= ball_owner_agent_idx_post < len(current_agents) else None,
                    "invalid_reason": invalid_reason if tick_idx == 0 else None,
                }

                # Fix ball_position: compute from obs_dict directly
                post_obs_vec = _get_obs_vector(obs_dict, current_agents[0]) if current_agents else None
                tick_data["ball_position"] = {
                    "x": float(post_obs_vec[88]) if post_obs_vec is not None and len(post_obs_vec) > 88 else None,
                    "y": float(post_obs_vec[89]) if post_obs_vec is not None and len(post_obs_vec) > 89 else None,
                    "z": float(post_obs_vec[90]) if post_obs_vec is not None and len(post_obs_vec) > 90 else None,
                }

                # Attach t=0 telemetry on first tick
                if tick_idx == 0:
                    tick_data["t0_ball_ownership"] = t0_capture["t0_ball_ownership"]
                    tick_data["t0_controlled_has_ball"] = t0_capture["t0_controlled_has_ball"]
                    tick_data["t0_mask_sum"] = t0_capture["t0_mask_sum"]
                    tick_data["t0_pass_legal"] = t0_capture["t0_pass_legal"]
                    tick_data["t0_shot_legal"] = t0_capture["t0_shot_legal"]
                    tick_data["t0_dribble_legal"] = t0_capture["t0_dribble_legal"]

                ep_tick_log.append(tick_data)
                tick_idx += 1

                if done:
                    break

                # Enforce post-force window limit
                if force_this_episode and tick_idx > force_tick + post_force_window:
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
                "pass_completed": 1 if _has_event(ep_tick_log, "pass_completed") else 0,
                "shot_event": 1 if _has_event(ep_tick_log, "shot") else 0,
                "goal": int(is_goal),
                "ownership_before": ownership_before,
                "ownership_after": ownership_after,
                "ball_delta_x": round(ball_position_after["x"] - ball_position_before["x"], 4) if (ball_position_before and ball_position_after and ball_position_after.get("x") is not None and ball_position_before.get("x") is not None) else None,
                "policy_pass_count": policy_actions_count["pass_short"] + policy_actions_count["pass_long"] + policy_actions_count["pass_high"],
                "policy_shot_count": policy_actions_count["shot"],
                "policy_tackle_count": policy_actions_count["tackle"],
                "possession_t1": possession_snapshots.get("t1"),
                "possession_t5": possession_snapshots.get("t5"),
                "possession_t20": possession_snapshots.get("t20"),
                "force_reward": force_reward,
                "mean_post_force_reward": round(np.mean(post_force_rewards), 4) if post_force_rewards else None,
                "event_histogram": _event_histogram(ep_tick_log),
                "episode_reward": ep_reward,
                "episode_length": ep_length,
                "tick_log": ep_tick_log,
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
        "post_force_window": post_force_window,
        "episodes": episodes_data,
    }


def aggregate_summary(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-seed results into cross-seed summary."""
    rows = []
    for res in all_results:
        for ep in res["episodes"]:
            rows.append(ep)

    total_valid = sum(1 for r in rows if r["valid_force_mask"])
    total_force_applied = sum(1 for r in rows if r["force_applied"])
    total_pass_completed = sum(r["pass_completed"] for r in rows)
    total_shot_event = sum(r["shot_event"] for r in rows)
    total_goals = sum(r["goal"] for r in rows)
    total_policy_pass = sum(r["policy_pass_count"] for r in rows)
    total_policy_shot = sum(r["policy_shot_count"] for r in rows)
    total_policy_tackle = sum(r["policy_tackle_count"] for r in rows)
    valid_t0_possession = sum(1 for r in rows if r["valid_t0_possession"])
    valid_force_count = sum(1 for r in rows if r["valid_force_mask"])
    force_success_count = sum(1 for r in rows if (r["force_applied"] and (
        (r["arm"] == "ONBALL-Pass" and r["pass_completed"]) or
        (r["arm"] == "ONBALL-Shot" and r["shot_event"])
    )))

    pass_valid_rows = [r for r in rows if r["arm"] == "ONBALL-Pass" and r["valid_force_mask"]]
    shot_valid_rows = [r for r in rows if r["arm"] == "ONBALL-Shot" and r["valid_force_mask"]]

    pass_success = sum(1 for r in pass_valid_rows if r["pass_completed"])
    shot_success = sum(1 for r in shot_valid_rows if r["shot_event"])

    mean_force_rewards = [r["force_reward"] for r in rows if r["force_applied"]]
    mean_post_force_rewards = [r["mean_post_force_reward"] for r in rows if r["force_applied"] and r["mean_post_force_reward"] is not None]
    delta_ball_x = [r["ball_delta_x"] for r in rows if r["force_applied"] and r["ball_delta_x"] is not None]

    return {
        "total_episodes": len(rows),
        "valid_t0_possession_count": valid_t0_possession,
        "valid_t0_possession_fraction": valid_t0_possession / max(len(rows), 1),
        "valid_force_count": valid_force_count,
        "force_success_count": force_success_count,
        "force_success_fraction": force_success_count / max(valid_force_count, 1),
        "pass_valid_count": len(pass_valid_rows),
        "pass_success_count": pass_success,
        "pass_success_fraction": pass_success / max(len(pass_valid_rows), 1),
        "shot_valid_count": len(shot_valid_rows),
        "shot_success_count": shot_success,
        "shot_success_fraction": shot_success / max(len(shot_valid_rows), 1),
        "total_goals": total_goals,
        "total_policy_pass": total_policy_pass,
        "total_policy_shot": total_policy_shot,
        "total_policy_tackle": total_policy_tackle,
        "mean_force_reward": round(np.mean(mean_force_rewards), 4) if mean_force_rewards else None,
        "mean_post_force_reward": round(np.mean(mean_post_force_rewards), 4) if mean_post_force_rewards else None,
        "mean_delta_ball_x": round(np.mean(delta_ball_x), 4) if delta_ball_x else None,
        "std_delta_ball_x": round(np.std(delta_ball_x), 4) if delta_ball_x else None,
    }


def write_csv(all_results: List[Dict[str, Any]], csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    fieldnames = [
        "seed", "arm", "episodes_attempted", "valid_t0_count", "valid_force_count",
        "force_success_count", "force_success_rate",
        "policy_pass_count", "policy_shot_count", "policy_tackle_count", "goal_count",
        "possession_t1", "possession_t5", "possession_t20",
        "mean_force_reward", "mean_post_force_reward", "mean_delta_ball_x",
        "timesteps", "checkpoint_sha256",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for res in all_results:
            for ep in res["episodes"]:
                writer.writerow({
                    "seed": ep["seed"],
                    "arm": ep["arm"],
                    "episodes_attempted": 1,
                    "valid_t0_count": 1 if ep["valid_t0_possession"] else 0,
                    "valid_force_count": 1 if ep["valid_force_mask"] else 0,
                    "force_success_count": 1 if (ep["force_applied"] and (
                        (ep["arm"] == "ONBALL-Pass" and ep["pass_completed"]) or
                        (ep["arm"] == "ONBALL-Shot" and ep["shot_event"])
                    )) else 0,
                    "force_success_rate": 1.0 if (ep["force_applied"] and (
                        (ep["arm"] == "ONBALL-Pass" and ep["pass_completed"]) or
                        (ep["arm"] == "ONBALL-Shot" and ep["shot_event"])
                    )) else 0.0,
                    "policy_pass_count": ep["policy_pass_count"],
                    "policy_shot_count": ep["policy_shot_count"],
                    "policy_tackle_count": ep["policy_tackle_count"],
                    "goal_count": ep["goal"],
                    "possession_t1": ep.get("possession_t1"),
                    "possession_t5": ep.get("possession_t5"),
                    "possession_t20": ep.get("possession_t20"),
                    "mean_force_reward": ep.get("force_reward"),
                    "mean_post_force_reward": ep.get("mean_post_force_reward"),
                    "mean_delta_ball_x": ep.get("ball_delta_x"),
                    "timesteps": ep["checkpoint_timesteps"],
                    "checkpoint_sha256": ep["checkpoint_sha256"],
                })


def write_report(
    all_results: List[Dict[str, Any]],
    report_path: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    checkpoint_timesteps: int,
    scenario: str,
    deterministic: bool,
    base_seed: int,
    post_force_window: int,
):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    agg = aggregate_summary(all_results)
    rows = [ep for res in all_results for ep in res["episodes"]]

    lines = []
    lines.append("# F_act: Forced One-Tick PASS/SHOT\n")
    lines.append(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    lines.append(f"**HEAD at eval:** `{__import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__)).decode().strip()}`")
    lines.append(f"**Checkpoint:** `{checkpoint_path}`")
    lines.append(f"**Checkpoint SHA256:** `{checkpoint_sha256}`")
    lines.append(f"**Checkpoint timesteps:** {checkpoint_timesteps}")
    lines.append(f"**Scenario:** {scenario}")
    lines.append(f"**Deterministic:** {deterministic}")
    lines.append(f"**Base seed:** {base_seed}")
    lines.append(f"**Gamma / Lambda:** 0.99 / 0.95")
    lines.append(f"**K (force duration):** 1")
    lines.append(f"**Post-force window:** {post_force_window} ticks")
    lines.append("")

    # Section 1: Protocol
    lines.append("## 1. Protocol")
    lines.append("- ONBALL-π: 10 episodes/seed, no forced action")
    lines.append("- ONBALL-Pass: >=20 valid episodes/seed, force PASS once at t=0")
    lines.append("- ONBALL-Shot: >=20 valid episodes/seed, force SHOT once at t=0")
    lines.append("- Teammate sprint scripting: OFF")
    lines.append("- t=0 ownership: captured from obs[94:97] before first step")
    lines.append("- Invalid episodes excluded from conditional success denominator")
    lines.append("")

    # Section 2: Trigger Table
    lines.append("## 2. Trigger Table")
    lines.append("| Seed | Arm | Valid t0 possession | Valid PASS force | Valid SHOT force |")
    lines.append("|------|-----|---------------------|------------------|------------------|")
    for res in all_results:
        seed = res["seed"]
        arm = res["arm"]
        eps = res["episodes"]
        valid_t0 = sum(1 for e in eps if e["valid_t0_possession"])
        valid_pass = sum(1 for e in eps if e["arm"] == "ONBALL-Pass" and e["valid_force_mask"])
        valid_shot = sum(1 for e in eps if e["arm"] == "ONBALL-Shot" and e["valid_force_mask"])
        lines.append(f"| {seed} | {arm} | {valid_t0}/{len(eps)} | {valid_pass} | {valid_shot} |")
    lines.append("")

    # Section 3: Forced PASS Result
    lines.append("## 3. Forced PASS Result")
    pass_results = [r for r in all_results if r["arm"] == "ONBALL-Pass"]
    for res in pass_results:
        seed = res["seed"]
        eps = res["episodes"]
        valid = [e for e in eps if e["valid_force_mask"]]
        successes = [e for e in valid if e["pass_completed"]]
        rate = len(successes) / max(len(valid), 1)
        lines.append(f"### Seed {seed}")
        lines.append(f"- Valid / attempted: {len(valid)}/{len(eps)}")
        lines.append(f"- PASS_COMPLETED: {len(successes)} / {len(valid)} = {rate*100:.1f}%")
        if valid:
            deltas = [e["ball_delta_x"] for e in valid if e["ball_delta_x"] is not None]
            rewards = [e["force_reward"] for e in valid]
            lines.append(f"- Mean Delta ball_x: {np.mean(deltas):.4f}" if deltas else "- Mean Delta ball_x: N/A")
            lines.append(f"- Mean force-tick reward: {np.mean(rewards):+.4f}")
        lines.append("")

    # Section 4: Forced SHOT Result
    lines.append("## 4. Forced SHOT Result")
    shot_results = [r for r in all_results if r["arm"] == "ONBALL-Shot"]
    for res in shot_results:
        seed = res["seed"]
        eps = res["episodes"]
        valid = [e for e in eps if e["valid_force_mask"]]
        successes = [e for e in valid if e["shot_event"]]
        rate = len(successes) / max(len(valid), 1)
        lines.append(f"### Seed {seed}")
        lines.append(f"- Valid / attempted: {len(valid)}/{len(eps)}")
        lines.append(f"- SHOT event: {len(successes)} / {len(valid)} = {rate*100:.1f}%")
        if valid:
            deltas = [e["ball_delta_x"] for e in valid if e["ball_delta_x"] is not None]
            rewards = [e["force_reward"] for e in valid]
            lines.append(f"- Mean Delta ball_x: {np.mean(deltas):.4f}" if deltas else "- Mean Delta ball_x: N/A")
            lines.append(f"- Mean force-tick reward: {np.mean(rewards):+.4f}")
        lines.append("")

    # Section 5: ONBALL-π Sanity
    lines.append("## 5. ONBALL-π Sanity")
    pi_results = [r for r in all_results if r["arm"] == "ONBALL-π"]
    for res in pi_results:
        seed = res["seed"]
        eps = res["episodes"]
        total_pass = sum(e["policy_pass_count"] for e in eps)
        total_shot = sum(e["policy_shot_count"] for e in eps)
        total_tackle = sum(e["policy_tackle_count"] for e in eps)
        total_goal = sum(e["goal"] for e in eps)
        lines.append(f"- Seed {seed}: PASS={total_pass}, SHOT={total_shot}, TACKLE={total_tackle}, GOAL={total_goal}")
    lines.append("")

    # Section 6: Post-Force π Behavior
    lines.append("## 6. Post-Force pi Behavior")
    for res in all_results:
        if res["arm"] == "ONBALL-π":
            continue
        seed = res["seed"]
        arm = res["arm"]
        eps = res["episodes"]
        valid = [e for e in eps if e["valid_force_mask"]]
        total_pass = sum(e["policy_pass_count"] for e in valid)
        total_shot = sum(e["policy_shot_count"] for e in valid)
        total_tackle = sum(e["policy_tackle_count"] for e in valid)
        lines.append(f"- {arm} seed {seed}: post-force PASS={total_pass}, SHOT={total_shot}, TACKLE={total_tackle}")
    lines.append("")

    # Section 7: Environment vs Policy Decision
    lines.append("## 7. Environment vs Policy Decision")
    pass_rate = agg.get("pass_success_fraction", 0)
    shot_rate = agg.get("shot_success_fraction", 0)
    lines.append(f"- P(PASS_COMPLETED | valid force PASS): {agg.get('pass_success_count', 0)}/{agg.get('pass_valid_count', 0)} = {pass_rate*100:.1f}%")
    lines.append(f"- P(SHOT | valid force SHOT): {agg.get('shot_success_count', 0)}/{agg.get('shot_valid_count', 0)} = {shot_rate*100:.1f}%")

    if pass_rate > 0 or shot_rate > 0:
        lines.append("\n**Call: ENVIRONMENT CAPABLE**")
        lines.append("The environment/action path can execute the commanded football action under the validated mu-onball state.")
    else:
        lines.append("\n**Call: ACTION / ENGINE / ENCODING FAILURE**")
        lines.append("Valid on-ball state with legal forced action produces no corresponding engine event.")
    lines.append("")

    # Section 8: Cross-Seed Consistency
    lines.append("## 8. Cross-Seed Consistency")
    for res in all_results:
        if res["arm"] == "ONBALL-π":
            continue
        seed = res["seed"]
        eps = res["episodes"]
        valid = [e for e in eps if e["valid_force_mask"]]
        successes = [e for e in valid if (
            (res["arm"] == "ONBALL-Pass" and e["pass_completed"]) or
            (res["arm"] == "ONBALL-Shot" and e["shot_event"])
        )]
        rate = len(successes) / max(len(valid), 1)
        lines.append(f"- {res['arm']} seed {seed}: {len(successes)}/{len(valid)} = {rate*100:.1f}%")
    lines.append("")

    # Section 9: What This Does Not Imply
    lines.append("## 9. What This Does Not Imply")
    lines.append("- Not Experiment B success")
    lines.append("- Not historical tackle-spam reproduction")
    lines.append("- Not F_start")
    lines.append("- Not production GAE failure")
    lines.append("- Not reward correctness")
    lines.append("- Not critic architecture correctness")
    lines.append("- Not GNN evidence")
    lines.append("")

    # Section 10: Next Experiment
    lines.append("## 10. Next Experiment")
    lines.append("Only if environment capability is demonstrated:")
    lines.append("- mix-script OR")
    lines.append("- fresh policy initialization on mu-onball")
    lines.append("")
    lines.append("Do not recommend retraining the existing paralyzed checkpoint.")
    lines.append("")

    # Section 11: Artifacts
    lines.append("## 11. Artifacts")
    lines.append("- `training/eval_f_act.py`")
    lines.append("- `training/results/EXPERIMENT_F_ACT.md`")
    lines.append("- `training/results/f_act_summary.csv`")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Experiment F_act: Forced PASS/SHOT on mu-onball")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MAPPO checkpoint")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper_onball")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for this run")
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--arm", type=str, required=True, choices=["ONBALL-pi", "ONBALL-Pass", "ONBALL-Shot"])
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--base-seed", type=int, default=500000)
    parser.add_argument("--output-dir", type=str, default="training/results")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    checkpoint_sha256 = sha256_of(args.checkpoint)
    actor, critic, obs_dim, action_dim, ckpt_timesteps = load_checkpoint(args.checkpoint)

    print("=" * 70)
    print(f"F_ACT: {args.arm}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"SHA256     : {checkpoint_sha256[:16]}...")
    print(f"Timesteps  : {ckpt_timesteps}")
    print(f"Scenario   : {args.scenario}")
    print(f"Seed       : {args.seed}")
    print(f"Episodes   : {args.num_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print("=" * 70)

    result = run_arm(
        actor=actor,
        critic=critic,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_timesteps=ckpt_timesteps or 0,
        seed=args.seed,
        arm=args.arm,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        base_seed=args.base_seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    json_name = f"f_act_{args.arm}_seed{args.seed}_{ckpt_name}.json"
    json_path = os.path.join(args.output_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nForensics JSON: {json_path}")

    eps = result["episodes"]
    valid = [e for e in eps if e["valid_force_mask"]]
    pass_arm = args.arm == "ONBALL-Pass"
    shot_arm = args.arm == "ONBALL-Shot"
    successes = [e for e in valid if (pass_arm and e["pass_completed"]) or (shot_arm and e["shot_event"])]
    print(f"\nValid episodes: {len(valid)}/{len(eps)}")
    print(f"Successes: {len(successes)}/{len(valid)} = {len(successes)/max(len(valid),1)*100:.1f}%")


if __name__ == "__main__":
    main()
