"""
GMN-Football-3 — Experiment F_act PASS Diagnostic Extension
Measurement-only: native PASS diagnostic with explicit pre/post tick semantics,
stable teammate identity, aiming/trajectory/event-detection diagnostics, and
evidence-based Rank 1/2/3 classification.

Arms:
  PASS — force PASS once at t=0, then policy (50-tick window)

No reward, mask, engine, or training code is modified.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv, EVENT_CODE_MAP
from training.mappo_networks import SharedActor, CentralizedCritic
from training.mappo_rollout import unwrap_obs, unwrap_masks, _mask_matrix

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
FORCED_PASS_IDX = 11  # SHORT_PASS
FORCED_SHOT_IDX = 12
OBS_DIM = 127
FORCE_TICK = 0
POST_FORCE_WINDOW = 50
TEAMMATE_1_ID = "left_2"
TEAMMATE_2_ID = "left_3"
TEAMMATE_1_OBS_IDX = 1   # index in controllableAgentIds list
TEAMMATE_2_OBS_IDX = 2


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
    if event_code is None or event_code <= 0 or event_code >= len(EVENT_CODE_MAP):
        return None
    return EVENT_CODE_MAP[event_code]


def _get_obs_vector(obs_dict: Dict[str, Any], agent: str) -> Optional[np.ndarray]:
    """Extract raw observation array from either envelope or raw format."""
    if not obs_dict:
        return None
    agent_obs = obs_dict.get(agent)
    if agent_obs is None:
        return None
    if isinstance(agent_obs, dict) and "observation" in agent_obs:
        return np.asarray(agent_obs["observation"], dtype=np.float32)
    if isinstance(agent_obs, np.ndarray):
        return agent_obs.astype(np.float32)
    return None


def _select_policy_actions(
    actor,
    local_obs: np.ndarray,
    mask_matrix: np.ndarray,
    deterministic: bool = True,
) -> np.ndarray:
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
    action_dict: Dict[str, int] = {}
    for i, agent in enumerate(current_agents):
        if i == force_agent_idx and force_this_tick and force_idx is not None:
            action_dict[agent] = int(force_idx)
        else:
            action_dict[agent] = int(policy_actions[i])
    return action_dict


def _extract_teammate_slice(obs_vec: Optional[np.ndarray], teammate_obs_idx: int) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return (x, y, vel_x, vel_y) for a teammate from the 127-dim observation."""
    if obs_vec is None or len(obs_vec) < OBS_DIM:
        return None, None, None, None
    pos_offset = teammate_obs_idx * 2
    vel_offset = 22 + teammate_obs_idx * 2
    if pos_offset + 1 >= len(obs_vec) or vel_offset + 1 >= len(obs_vec):
        return None, None, None, None
    x = float(obs_vec[pos_offset])
    y = float(obs_vec[pos_offset + 1])
    vx = float(obs_vec[vel_offset]) / 50.0
    vy = float(obs_vec[vel_offset + 1]) / 50.0
    if x == -1.0 and y == -1.0:
        return None, None, None, None
    return x, y, vx, vy


def _ball_to_teammate_distance(
    ball_x: Optional[float],
    ball_y: Optional[float],
    tm_x: Optional[float],
    tm_y: Optional[float],
) -> Optional[float]:
    if ball_x is None or ball_y is None or tm_x is None or tm_y is None:
        return None
    return math.hypot(ball_x - tm_x, ball_y - tm_y)


def _angular_error(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Angle in radians between two vectors."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = math.hypot(v1[0], v1[1])
    norm2 = math.hypot(v2[0], v2[1])
    if norm1 == 0 or norm2 == 0:
        return math.pi
    cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.acos(cos_theta)


def _perpendicular_distance_to_ray(
    point: Tuple[float, float],
    ray_origin: Tuple[float, float],
    ray_dir: Tuple[float, float],
) -> float:
    """Shortest distance from point to infinite ray (origin + t*dir)."""
    dx = point[0] - ray_origin[0]
    dy = point[1] - ray_origin[1]
    dir_len = math.hypot(ray_dir[0], ray_dir[1])
    if dir_len == 0:
        return math.hypot(dx, dy)
    # Project (dx,dy) onto dir, then compute perpendicular component
    proj = (dx * ray_dir[0] + dy * ray_dir[1]) / (dir_len * dir_len)
    closest_x = ray_origin[0] + proj * ray_dir[0]
    closest_y = ray_origin[1] + proj * ray_dir[1]
    return math.hypot(point[0] - closest_x, point[1] - closest_y)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class PassDiagnosticEvaluator:
    def __init__(
        self,
        actor,
        checkpoint_path: str,
        checkpoint_sha256: str,
        checkpoint_timesteps: int,
        scenario: str = "academy_3_vs_1_with_keeper_onball",
        deterministic: bool = True,
        base_seed: int = 500000,
        post_force_window: int = POST_FORCE_WINDOW,
        bridge_port: int = 5050,
    ):
        self.actor = actor
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_timesteps = checkpoint_timesteps
        self.scenario = scenario
        self.deterministic = deterministic
        self.base_seed = base_seed
        self.post_force_window = post_force_window
        self.bridge_port = bridge_port
        self.run_id = f"pass_diag_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

    # -- episode data collection ------------------------------------------

    def _collect_pre_step_state(
        self, obs_dict: Dict[str, Any], current_agents: List[str], ball_owner_agent_idx: int
    ) -> Dict[str, Any]:
        first_obs = _get_obs_vector(obs_dict, current_agents[0]) if current_agents else None
        pre_ball_pos = None
        pre_ball_vel = None
        if first_obs is not None and len(first_obs) >= 94:
            pre_ball_pos = {
                "x": float(first_obs[88]),
                "y": float(first_obs[89]),
                "z": float(first_obs[90]),
            }
            pre_ball_vel = {
                "x": float(first_obs[91]),
                "y": float(first_obs[92]),
                "z": float(first_obs[93]),
            }

        pre_owner_id = (
            current_agents[ball_owner_agent_idx]
            if 0 <= ball_owner_agent_idx < len(current_agents)
            else None
        )

        tm1_x, tm1_y, tm1_vx, tm1_vy = _extract_teammate_slice(first_obs, TEAMMATE_1_OBS_IDX)
        tm2_x, tm2_y, tm2_vx, tm2_vy = _extract_teammate_slice(first_obs, TEAMMATE_2_OBS_IDX)

        return {
            "pre_ball_pos": pre_ball_pos,
            "pre_ball_vel": pre_ball_vel,
            "pre_owner_id": pre_owner_id,
            "pre_tm1_x": tm1_x,
            "pre_tm1_y": tm1_y,
            "pre_tm1_vx": tm1_vx,
            "pre_tm1_vy": tm1_vy,
            "pre_tm2_x": tm2_x,
            "pre_tm2_y": tm2_y,
            "pre_tm2_vx": tm2_vx,
            "pre_tm2_vy": tm2_vy,
        }

    def _collect_post_step_state(
        self,
        obs_dict: Dict[str, Any],
        infos: Dict[str, Any],
        current_agents: List[str],
        event_code: Optional[int],
        ball_owner_agent_idx_post: int,
    ) -> Dict[str, Any]:
        first_obs = _get_obs_vector(obs_dict, current_agents[0]) if current_agents else None
        post_ball_pos = None
        post_ball_vel = None
        if first_obs is not None and len(first_obs) >= 94:
            post_ball_pos = {
                "x": float(first_obs[88]),
                "y": float(first_obs[89]),
                "z": float(first_obs[90]),
            }
            post_ball_vel = {
                "x": float(first_obs[91]),
                "y": float(first_obs[92]),
                "z": float(first_obs[93]),
            }

        post_owner_id = (
            current_agents[ball_owner_agent_idx_post]
            if 0 <= ball_owner_agent_idx_post < len(current_agents)
            else None
        )

        # last_kicked_by from bridge ground_truth
        post_last_kicked_by = None
        if current_agents and current_agents[0] in infos:
            gt = infos[current_agents[0]].get("ground_truth", {})
            lkb = gt.get("last_kicked_by")
            if lkb is not None:
                post_last_kicked_by = str(lkb)

        tm1_x, tm1_y, tm1_vx, tm1_vy = _extract_teammate_slice(first_obs, TEAMMATE_1_OBS_IDX)
        tm2_x, tm2_y, tm2_vx, tm2_vy = _extract_teammate_slice(first_obs, TEAMMATE_2_OBS_IDX)

        # mask data
        mask_sum = None
        pass_legal = None
        shot_legal = None
        has_ball = ball_owner_agent_idx_post == 0
        if current_agents and current_agents[0] in infos:
            am = infos[current_agents[0]].get("action_mask")
            if am is not None and hasattr(am, "__len__") and len(am) >= 13:
                mask_sum = int(np.sum(am))
                pass_legal = int(am[FORCED_PASS_IDX])
                shot_legal = int(am[FORCED_SHOT_IDX])

        return {
            "post_event_code": event_code,
            "post_event_name": _event_type_from_code(event_code),
            "post_ball_pos": post_ball_pos,
            "post_ball_vel": post_ball_vel,
            "post_owner_id": post_owner_id,
            "post_last_kicked_by": post_last_kicked_by,
            "post_tm1_x": tm1_x,
            "post_tm1_y": tm1_y,
            "post_tm1_vx": tm1_vx,
            "post_tm1_vy": tm1_vy,
            "post_tm2_x": tm2_x,
            "post_tm2_y": tm2_y,
            "post_tm2_vx": tm2_vx,
            "post_tm2_vy": tm2_vy,
            "mask_sum": mask_sum,
            "pass_legal": pass_legal,
            "shot_legal": shot_legal,
            "has_ball": has_ball,
        }

    # -- diagnostics ------------------------------------------------------

    def _compute_aiming_diagnostics(
        self,
        pre_ball_pos: Optional[Dict[str, float]],
        pre_pass_direction: Optional[Dict[str, float]],
        pre_tm1_x: Optional[float],
        pre_tm1_y: Optional[float],
        pre_tm2_x: Optional[float],
        pre_tm2_y: Optional[float],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "pass_direction_unit_vector": None,
            "nearest_teammate_at_force": None,
            "nearest_teammate_initial_distance": None,
            "angular_error_to_nearest_teammate": None,
            "perpendicular_distance_to_pass_ray_tm1": None,
            "perpendicular_distance_to_pass_ray_tm2": None,
            "distance_to_teammate_at_force_tm1": None,
            "distance_to_teammate_at_force_tm2": None,
            "angular_error_to_pass_direction_tm1": None,
            "angular_error_to_pass_direction_tm2": None,
        }

        if pre_ball_pos is None or pre_pass_direction is None:
            return result

        bx, by = pre_ball_pos["x"], pre_ball_pos["y"]
        dx, dy = pre_pass_direction["x"], pre_pass_direction["y"]
        dir_len = math.hypot(dx, dy)
        if dir_len == 0:
            return result
        dir_norm = (dx / dir_len, dy / dir_len)
        result["pass_direction_unit_vector"] = {"x": dir_norm[0], "y": dir_norm[1]}

        teammates = []
        if pre_tm1_x is not None and pre_tm1_y is not None:
            teammates.append(("left_2", pre_tm1_x, pre_tm1_y))
        if pre_tm2_x is not None and pre_tm2_y is not None:
            teammates.append(("left_3", pre_tm2_x, pre_tm2_y))

        if not teammates:
            return result

        # distances and angles for each teammate
        best_dist = None
        best_tm = None
        for tid, tx, ty in teammates:
            dist = math.hypot(bx - tx, by - ty)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_tm = tid

        result["nearest_teammate_at_force"] = best_tm
        result["nearest_teammate_initial_distance"] = round(best_dist, 6) if best_dist is not None else None

        for tid, tx, ty in teammates:
            vec_to_tm = (tx - bx, ty - by)
            ang_err = _angular_error(dir_norm, vec_to_tm)
            perp_dist = _perpendicular_distance_to_ray((tx, ty), (bx, by), dir_norm)
            dist = math.hypot(bx - tx, by - ty)
            if tid == "left_2":
                result["angular_error_to_pass_direction_tm1"] = round(ang_err, 6)
                result["perpendicular_distance_to_pass_ray_tm1"] = round(perp_dist, 6)
                result["distance_to_teammate_at_force_tm1"] = round(dist, 6)
            else:
                result["angular_error_to_pass_direction_tm2"] = round(ang_err, 6)
                result["perpendicular_distance_to_pass_ray_tm2"] = round(perp_dist, 6)
                result["distance_to_teammate_at_force_tm2"] = round(dist, 6)
            if tid == best_tm:
                result["angular_error_to_nearest_teammate"] = round(ang_err, 6)

        return result

    def _compute_trajectory_diagnostics(
        self, tick_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "min_ball_to_teammate_distance": None,
            "tick_of_minimum_ball_to_teammate_distance": None,
            "teammate_id_at_min_distance": None,
            "terminal_ball_to_nearest_teammate_distance": None,
            "owner_before_min_distance": None,
            "owner_after_min_distance": None,
            "last_kicked_by_at_min_distance": None,
            "pass_completed_event_seen": False,
            "min_distance_ball_to_any_left_teammate": None,
            "tick_of_min_distance": None,
            "teammate_id_at_min_distance": None,
            "owner_before_min_distance_full": None,
            "owner_after_min_distance_full": None,
        }

        if not tick_log:
            return result

        # Check for pass_completed event
        for tick in tick_log:
            if tick.get("post_event_name") == "pass_completed":
                result["pass_completed_event_seen"] = True
                break

        # Compute ball-to-teammate distances for each tick
        best_dist = None
        best_tick = None
        best_tm = None
        best_owner_before = None
        best_owner_after = None
        best_lkb = None

        for idx, tick in enumerate(tick_log):
            bx = tick.get("post_ball_pos_x")
            by = tick.get("post_ball_pos_y")
            tm1_x = tick.get("post_teammate_1_x")
            tm1_y = tick.get("post_teammate_1_y")
            tm2_x = tick.get("post_teammate_2_x")
            tm2_y = tick.get("post_teammate_2_y")

            candidates = []
            if tm1_x is not None and tm1_y is not None:
                candidates.append(("left_2", tm1_x, tm1_y))
            if tm2_x is not None and tm2_y is not None:
                candidates.append(("left_3", tm2_x, tm2_y))

            tick_min_dist = None
            tick_min_tm = None
            for tid, tx, ty in candidates:
                d = math.hypot(bx - tx, by - ty) if bx is not None and by is not None else None
                if d is not None and (tick_min_dist is None or d < tick_min_dist):
                    tick_min_dist = d
                    tick_min_tm = tid

            if tick_min_dist is not None:
                if best_dist is None or tick_min_dist < best_dist:
                    best_dist = tick_min_dist
                    best_tick = tick.get("tick")
                    best_tm = tick_min_tm
                    # owner before this tick = owner at previous tick
                    prev_tick = tick_log[idx - 1] if idx > 0 else tick
                    best_owner_before = prev_tick.get("post_owner_id")
                    best_owner_after = tick.get("post_owner_id")
                    best_lkb = tick.get("post_last_kicked_by")

        result["min_distance_ball_to_any_left_teammate"] = round(best_dist, 6) if best_dist is not None else None
        result["tick_of_min_distance"] = best_tick
        result["teammate_id_at_min_distance"] = best_tm
        result["owner_before_min_distance_full"] = best_owner_before
        result["owner_after_min_distance_full"] = best_owner_after
        result["last_kicked_by_at_min_distance"] = best_lkb

        # Also alias to plan's expected names
        result["min_ball_to_teammate_distance"] = result["min_distance_ball_to_any_left_teammate"]
        result["tick_of_minimum_ball_to_teammate_distance"] = result["tick_of_min_distance"]
        result["teammate_id_at_min_distance"] = result["teammate_id_at_min_distance"]
        result["terminal_ball_to_nearest_teammate_distance"] = self._terminal_distance(tick_log)
        result["owner_before_min_distance"] = best_owner_before
        result["owner_after_min_distance"] = best_owner_after

        return result

    def _terminal_distance(self, tick_log: List[Dict[str, Any]]) -> Optional[float]:
        if not tick_log:
            return None
        last = tick_log[-1]
        bx = last.get("post_ball_pos_x")
        by = last.get("post_ball_pos_y")
        tm1_x = last.get("post_teammate_1_x")
        tm1_y = last.get("post_teammate_1_y")
        tm2_x = last.get("post_teammate_2_x")
        tm2_y = last.get("post_teammate_2_y")
        dists = []
        for tx, ty in [(tm1_x, tm1_y), (tm2_x, tm2_y)]:
            if tx is not None and ty is not None and bx is not None and by is not None:
                dists.append(math.hypot(bx - tx, by - ty))
        return min(dists) if dists else None

    def _compute_teammate_response(self, tick_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        speeds_tm1: List[float] = []
        speeds_tm2: List[float] = []
        nonzero_count = 0
        total_teammates = 0

        for tick in tick_log:
            tm1_vx = tick.get("post_teammate_1_vel_x")
            tm1_vy = tick.get("post_teammate_1_vel_y")
            tm2_vx = tick.get("post_teammate_2_vel_x")
            tm2_vy = tick.get("post_teammate_2_vel_y")

            if tm1_vx is not None and tm1_vy is not None:
                s1 = math.hypot(tm1_vx, tm1_vy)
                speeds_tm1.append(s1)
                total_teammates += 1
                if s1 > 0.01:
                    nonzero_count += 1

            if tm2_vx is not None and tm2_vy is not None:
                s2 = math.hypot(tm2_vx, tm2_vy)
                speeds_tm2.append(s2)
                total_teammates += 1
                if s2 > 0.01:
                    nonzero_count += 1

        all_speeds = speeds_tm1 + speeds_tm2
        median_speed = float(np.median(all_speeds)) if all_speeds else None
        fraction_nonzero = nonzero_count / max(total_teammates, 1) if total_teammates > 0 else None

        return {
            "median_teammate_speed_after_force": round(median_speed, 6) if median_speed is not None else None,
            "fraction_teammates_with_nonzero_response": round(fraction_nonzero, 6) if fraction_nonzero is not None else None,
            "speeds_tm1": [round(s, 6) for s in speeds_tm1],
            "speeds_tm2": [round(s, 6) for s in speeds_tm2],
        }

    def _classify_rank(self, ep_diagnostics: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        evidence: Dict[str, Any] = {}
        ang_err = ep_diagnostics.get("angular_error_to_nearest_teammate")
        perp_tm1 = ep_diagnostics.get("perpendicular_distance_to_pass_ray_tm1")
        perp_tm2 = ep_diagnostics.get("perpendicular_distance_to_pass_ray_tm2")
        min_dist = ep_diagnostics.get("min_ball_to_teammate_distance")
        median_speed = ep_diagnostics.get("median_teammate_speed_after_force")
        pass_completed = ep_diagnostics.get("pass_completed_event_seen", False)
        ownership_changed = ep_diagnostics.get("owner_after_min_distance") != ep_diagnostics.get("owner_before_min_distance")

        perp_dist = min(p for p in [perp_tm1, perp_tm2] if p is not None) if any(p is not None for p in [perp_tm1, perp_tm2]) else None

        evidence["angular_error_rad"] = ang_err
        evidence["perpendicular_distance"] = perp_dist
        evidence["min_ball_to_teammate_distance"] = min_dist
        evidence["median_teammate_speed"] = median_speed
        evidence["pass_completed_event_seen"] = pass_completed
        evidence["ownership_changed"] = ownership_changed

        # Rank 1: Directional aiming defect
        if (
            ang_err is not None
            and ang_err > math.radians(45)
            and perp_dist is not None
            and perp_dist > 0.2
            and min_dist is not None
            and min_dist > 0.3
        ):
            return "Rank 1", evidence

        # Rank 2: Passive/non-reactive teammates
        if (
            ang_err is not None
            and ang_err < math.radians(30)
            and perp_dist is not None
            and perp_dist < 0.15
            and min_dist is not None
            and min_dist < 0.3
            and median_speed is not None
            and median_speed < 0.1
            and not pass_completed
        ):
            return "Rank 2", evidence

        # Rank 3: Completion/event detection
        if (
            min_dist is not None
            and min_dist < 0.3
            and ownership_changed
            and not pass_completed
        ):
            return "Rank 3", evidence

        return "Unclassified", evidence

    # -- main loop --------------------------------------------------------

    def run_episode(self, ep: int, base_seed: int) -> Dict[str, Any]:
        ep_seed = base_seed + ep * 1009
        env = GMNMultiAgentEnv(
            scenario=self.scenario,
            auto_start_bridge=True,
            port=self.bridge_port,
            enable_reward_shaping=True,
        )

        try:
            obs_dict, _ = env.reset(seed=ep_seed)
            controllable_agents = list(env.possible_agents)
            current_ep_masks = unwrap_masks(obs_dict)
            obs_dict = unwrap_obs(obs_dict)

            # t=0 ownership capture BEFORE first step
            reset_ball_owner_agent_idx = getattr(env, "_last_ball_owner_agent_idx", 255)
            t0_valid_possession = bool(reset_ball_owner_agent_idx == 0)
            t0_pass_legal = True
            if controllable_agents and controllable_agents[0] in current_ep_masks:
                mask = np.array(current_ep_masks[controllable_agents[0]], dtype=np.int8)
                if len(mask) >= 12:
                    t0_pass_legal = bool(mask[FORCED_PASS_IDX])

            current_agents = list(env.agents if env.agents else controllable_agents)
            ball_owner_agent_idx = reset_ball_owner_agent_idx
            if not (0 <= ball_owner_agent_idx < len(current_agents)):
                ball_owner_agent_idx = 0

            force_this_episode = t0_valid_possession and t0_pass_legal
            force_tick = FORCE_TICK if force_this_episode else -1
            force_agent_idx = ball_owner_agent_idx if force_this_episode else -1

            # Pre-step state at t=0 (from reset observation)
            pre_step_state = self._collect_pre_step_state(obs_dict, current_agents, ball_owner_agent_idx)

            # Pre-pass direction from bridge diagnostic hook
            pre_pass_direction = None
            if force_this_episode and current_agents:
                first_info_key = current_agents[0]
                # We don't have infos from reset, so we'll capture it after the first step
                pass

            ep_tick_log: List[Dict[str, Any]] = []
            ep_reward = 0.0
            ep_length = 0
            policy_actions_count = {
                "pass_short": 0, "pass_long": 0, "pass_high": 0,
                "shot": 0, "tackle": 0,
            }
            invalid_reason = None

            if not t0_valid_possession:
                invalid_reason = "t0_possession_invalid"
            elif not t0_pass_legal:
                invalid_reason = "t0_pass_illegal"

            tick_idx = 0
            while True:
                current_agents = list(env.agents if env.agents else controllable_agents)
                local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
                mask_matrix = _mask_matrix(current_ep_masks, current_agents)

                policy_actions = _select_policy_actions(self.actor, local_obs, mask_matrix, self.deterministic)

                # Build action dict
                action_dict = _build_action_dict(
                    current_agents,
                    policy_actions,
                    FORCED_PASS_IDX,
                    force_agent_idx,
                    force_this_episode and tick_idx == force_tick,
                )

                # Capture pre-pass direction from infos if this is the forced tick
                resolved_pass_direction = None
                if force_this_episode and tick_idx == force_tick:
                    pass  # resolved_pass_direction captured after env.step() from infos

                obs_dict, rewards, terms, truncs, infos = env.step(action_dict)

                # Capture resolved_pass_direction from infos after step
                if force_this_episode and tick_idx == force_tick:
                    if current_agents and current_agents[0] in infos:
                        rpd = infos[current_agents[0]].get("resolved_pass_direction")
                        if rpd is not None:
                            pre_pass_direction = {
                                "x": float(rpd.get("x", 0.0)),
                                "y": float(rpd.get("y", 0.0)),
                            }

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

                # Collect post-step state
                post_state = self._collect_post_step_state(
                    obs_dict, infos, current_agents, event_code, ball_owner_agent_idx_post
                )

                # Action source
                action_source = "forced" if (force_this_episode and tick_idx == force_tick) else "policy"

                # Policy counts (exclude forced tick)
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
                        if act_idx == 16:
                            policy_actions_count["tackle"] += 1

                tick_data = {
                    "run_id": self.run_id,
                    "episode_id": ep,
                    "seed": ep_seed,
                    "tick": tick_idx,
                    "action_source": action_source,
                    "action_selected": int(policy_actions[0]) if len(policy_actions) > 0 else -1,
                    "forced_action": FORCED_PASS_IDX if (force_this_episode and tick_idx == force_tick) else None,
                    # pre-step (only populated for forced tick)
                    "pre_ball_pos_x": pre_step_state["pre_ball_pos"]["x"] if (force_this_episode and tick_idx == force_tick and pre_step_state["pre_ball_pos"]) else None,
                    "pre_ball_pos_y": pre_step_state["pre_ball_pos"]["y"] if (force_this_episode and tick_idx == force_tick and pre_step_state["pre_ball_pos"]) else None,
                    "pre_ball_pos_z": pre_step_state["pre_ball_pos"]["z"] if (force_this_episode and tick_idx == force_tick and pre_step_state["pre_ball_pos"]) else None,
                    "pre_ball_vel_x": pre_step_state["pre_ball_vel"]["x"] if (force_this_episode and tick_idx == force_tick and pre_step_state["pre_ball_vel"]) else None,
                    "pre_ball_vel_y": pre_step_state["pre_ball_vel"]["y"] if (force_this_episode and tick_idx == force_tick and pre_step_state["pre_ball_vel"]) else None,
                    "pre_ball_vel_z": pre_step_state["pre_ball_vel"]["z"] if (force_this_episode and tick_idx == force_tick and pre_step_state["pre_ball_vel"]) else None,
                    "pre_owner_id": pre_step_state["pre_owner_id"] if (force_this_episode and tick_idx == force_tick) else None,
                    "pre_teammate_1_x": pre_step_state["pre_tm1_x"] if (force_this_episode and tick_idx == force_tick) else None,
                    "pre_teammate_1_y": pre_step_state["pre_tm1_y"] if (force_this_episode and tick_idx == force_tick) else None,
                    "pre_teammate_2_x": pre_step_state["pre_tm2_x"] if (force_this_episode and tick_idx == force_tick) else None,
                    "pre_teammate_2_y": pre_step_state["pre_tm2_y"] if (force_this_episode and tick_idx == force_tick) else None,
                    "pre_pass_direction_x": pre_pass_direction["x"] if (force_this_episode and tick_idx == force_tick and pre_pass_direction) else None,
                    "pre_pass_direction_y": pre_pass_direction["y"] if (force_this_episode and tick_idx == force_tick and pre_pass_direction) else None,
                    # post-step
                    "post_event_code": post_state["post_event_code"],
                    "post_event_name": post_state["post_event_name"],
                    "post_ball_pos_x": post_state["post_ball_pos"]["x"] if post_state["post_ball_pos"] else None,
                    "post_ball_pos_y": post_state["post_ball_pos"]["y"] if post_state["post_ball_pos"] else None,
                    "post_ball_pos_z": post_state["post_ball_pos"]["z"] if post_state["post_ball_pos"] else None,
                    "post_ball_vel_x": post_state["post_ball_vel"]["x"] if post_state["post_ball_vel"] else None,
                    "post_ball_vel_y": post_state["post_ball_vel"]["y"] if post_state["post_ball_vel"] else None,
                    "post_ball_vel_z": post_state["post_ball_vel"]["z"] if post_state["post_ball_vel"] else None,
                    "post_owner_id": post_state["post_owner_id"],
                    "post_last_kicked_by": post_state["post_last_kicked_by"],
                    "post_teammate_1_x": post_state["post_tm1_x"],
                    "post_teammate_1_y": post_state["post_tm1_y"],
                    "post_teammate_1_vel_x": post_state["post_tm1_vx"],
                    "post_teammate_1_vel_y": post_state["post_tm1_vy"],
                    "post_teammate_2_x": post_state["post_tm2_x"],
                    "post_teammate_2_y": post_state["post_tm2_y"],
                    "post_teammate_2_vel_x": post_state["post_tm2_vx"],
                    "post_teammate_2_vel_y": post_state["post_tm2_vy"],
                    "mask_sum": post_state["mask_sum"],
                    "pass_legal": post_state["pass_legal"],
                    "shot_legal": post_state["shot_legal"],
                    "has_ball": post_state["has_ball"],
                    "reward": round(shared_rew, 6),
                    "terminated": term,
                    "truncated": trunc,
                }

                ep_tick_log.append(tick_data)
                tick_idx += 1

                if done:
                    break

                if force_this_episode and tick_idx > force_tick + self.post_force_window:
                    break

            # Episode-level diagnostics
            aiming = self._compute_aiming_diagnostics(
                pre_step_state["pre_ball_pos"],
                pre_pass_direction,
                pre_step_state["pre_tm1_x"],
                pre_step_state["pre_tm1_y"],
                pre_step_state["pre_tm2_x"],
                pre_step_state["pre_tm2_y"],
            )
            trajectory = self._compute_trajectory_diagnostics(ep_tick_log)
            teammate_resp = self._compute_teammate_response(ep_tick_log)

            ep_diagnostics = {**aiming, **trajectory, **teammate_resp}
            rank, rank_evidence = self._classify_rank(ep_diagnostics)

            episode_summary = {
                "run_id": self.run_id,
                "seed": self.base_seed if self.base_seed else ep_seed,
                "episode": ep,
                "ep_seed": ep_seed,
                "scenario": self.scenario,
                "deterministic": self.deterministic,
                "checkpoint": self.checkpoint_path,
                "checkpoint_sha256": self.checkpoint_sha256,
                "checkpoint_timesteps": self.checkpoint_timesteps,
                "force_action": "PASS",
                "force_tick": FORCE_TICK,
                "K": 1,
                "teammate_scripting": False,
                "direction_override": False,
                "clampPassDirection": False,
                "reward_modified": False,
                "production_code_behavior_changed": False,
                "valid_t0_possession": t0_valid_possession,
                "valid_force_mask": force_this_episode,
                "invalid_reason": invalid_reason,
                "force_applied": force_this_episode,
                "pass_completed": 1 if trajectory.get("pass_completed_event_seen") else 0,
                "shot_event": 0,
                "goal": 0,
                "ownership_before": pre_step_state["pre_owner_id"],
                "ownership_after": post_state["post_owner_id"],
                "resolved_pass_direction": pre_pass_direction,
                "policy_pass_count": policy_actions_count["pass_short"] + policy_actions_count["pass_long"] + policy_actions_count["pass_high"],
                "policy_shot_count": policy_actions_count["shot"],
                "policy_tackle_count": policy_actions_count["tackle"],
                "episode_reward": round(ep_reward, 6),
                "episode_length": ep_length,
                "rank": rank,
                "rank_evidence": rank_evidence,
                "tick_log": ep_tick_log,
                **ep_diagnostics,
            }

            if (ep + 1) % 5 == 0:
                print(
                    f"  [PASS seed={self.base_seed}] Ep {ep+1:3d}/{20} | "
                    f"valid={force_this_episode} | force={'Y' if force_this_episode else 'N'} | "
                    f"PASS_COMPLETED={episode_summary['pass_completed']} | "
                    f"Rank={rank} | Reward={ep_reward:+.3f} | Len={ep_length}"
                )

            return episode_summary

        finally:
            env.close()

    def run_seed(self, seed: int, num_episodes: int = 20) -> List[Dict[str, Any]]:
        episodes: List[Dict[str, Any]] = []
        for ep in range(num_episodes):
            episodes.append(self.run_episode(ep, seed))
        return episodes

    def run_all(self, seeds: List[int] = (42, 123, 7, 999), num_episodes_per_seed: int = 20) -> List[Dict[str, Any]]:
        all_results: List[Dict[str, Any]] = []
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"PASS Diagnostic: seed={seed}, episodes={num_episodes_per_seed}")
            print(f"{'='*60}")
            seed_results = self.run_seed(seed, num_episodes_per_seed)
            all_results.append({
                "seed": seed,
                "arm": "ONBALL-Pass",
                "checkpoint": self.checkpoint_path,
                "checkpoint_sha256": self.checkpoint_sha256,
                "checkpoint_timesteps": self.checkpoint_timesteps,
                "scenario": self.scenario,
                "deterministic": self.deterministic,
                "base_seed": self.base_seed,
                "post_force_window": self.post_force_window,
                "episodes": seed_results,
            })
        return all_results

    # -- output writers ---------------------------------------------------

    def write_trace_csv(self, all_results: List[Dict[str, Any]], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        fieldnames = [
            "run_id", "episode_id", "seed", "tick", "action_source", "action_selected", "forced_action",
            "pre_ball_pos_x", "pre_ball_pos_y", "pre_ball_pos_z",
            "pre_ball_vel_x", "pre_ball_vel_y", "pre_ball_vel_z",
            "pre_owner_id",
            "pre_teammate_1_x", "pre_teammate_1_y", "pre_teammate_2_x", "pre_teammate_2_y",
            "pre_pass_direction_x", "pre_pass_direction_y",
            "post_event_code", "post_event_name",
            "post_ball_pos_x", "post_ball_pos_y", "post_ball_pos_z",
            "post_ball_vel_x", "post_ball_vel_y", "post_ball_vel_z",
            "post_owner_id", "post_last_kicked_by",
            "post_teammate_1_x", "post_teammate_1_y", "post_teammate_1_vel_x", "post_teammate_1_vel_y",
            "post_teammate_2_x", "post_teammate_2_y", "post_teammate_2_vel_x", "post_teammate_2_vel_y",
            "mask_sum", "pass_legal", "shot_legal", "has_ball", "reward", "terminated", "truncated",
        ]

        # Group results by seed
        by_seed: Dict[int, List[Dict[str, Any]]] = {}
        for res in all_results:
            by_seed.setdefault(res["seed"], []).extend(res["episodes"])

        for seed, episodes in by_seed.items():
            csv_path = os.path.join(output_dir, f"f_act_pass_trace_seed{seed}.csv")
            file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for ep in episodes:
                    for tick in ep.get("tick_log", []):
                        writer.writerow({k: tick.get(k) for k in fieldnames})

    def write_manifest(self, all_results: List[Dict[str, Any]], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        valid_count = sum(
            1 for res in all_results for ep in res["episodes"] if ep.get("valid_force_mask")
        )
        manifest = {
            "run_id": self.run_id,
            "git_head": self._git_head(),
            "scenario": self.scenario,
            "deterministic": self.deterministic,
            "base_seed": self.base_seed,
            "episode_count_per_seed": 20,
            "checkpoint_paths": {
                str(res["seed"]): res["checkpoint"] for res in all_results
            },
            "checkpoint_sha256": {
                str(res["seed"]): res["checkpoint_sha256"] for res in all_results
            },
            "checkpoint_timesteps": self.checkpoint_timesteps,
            "force_action": "PASS",
            "force_tick": FORCE_TICK,
            "K": 1,
            "teammate_scripting": False,
            "direction_override": False,
            "clampPassDirection": False,
            "reward_modified": False,
            "production_code_behavior_changed": False,
            "valid_forced_passes": valid_count,
            "seeds": [res["seed"] for res in all_results],
        }
        manifest_path = os.path.join(output_dir, "pass_diagnostic_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def write_findings(self, all_results: List[Dict[str, Any]], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        agg = self.compute_summary(all_results)
        lines: List[str] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines.append("# PASS Diagnostic Findings — F_act Extension\n")
        lines.append(f"**Date:** {now}")
        lines.append(f"**Run ID:** {self.run_id}")
        lines.append(f"**HEAD:** `{self._git_head()}`")
        lines.append(f"**Checkpoint SHA256 (seed 42):** `{self.checkpoint_sha256[:16]}...`")
        lines.append(f"**Checkpoint timesteps:** {self.checkpoint_timesteps}")
        lines.append(f"**Scenario:** {self.scenario}")
        lines.append(f"**Seeds:** {', '.join(str(r['seed']) for r in all_results)}")
        lines.append(f"**Episodes per seed:** 20")
        lines.append("")

        # Section 1: Provenance
        lines.append("## 1. Provenance\n")
        lines.append(f"- **HEAD:** `{self._git_head()}`")
        for res in all_results:
            lines.append(f"- **Seed {res['seed']} checkpoint:** `{res['checkpoint']}`")
            lines.append(f"  - SHA256: `{res['checkpoint_sha256'][:16]}...`")
            lines.append(f"  - Timesteps: {res['checkpoint_timesteps']}")
        lines.append(f"- **Scenario:** {self.scenario}")
        lines.append(f"- **Deterministic:** {self.deterministic}")
        lines.append("")

        # Section 2: Protocol
        lines.append("## 2. Protocol\n")
        lines.append("- Native PASS only (SHORT_PASS, index 11)")
        lines.append("- K = 1 (forced action applied for exactly one tick)")
        lines.append("- Post-force window: 50 ticks")
        lines.append("- No clampPassDirection")
        lines.append("- No teammate scripting")
        lines.append("- No reward/GAE/mask/network/spawn changes")
        lines.append("- Production PASS semantics unchanged")
        lines.append("")

        # Section 3: Trigger validity
        lines.append("## 3. Trigger Validity\n")
        valid_rows = [ep for res in all_results for ep in res["episodes"] if ep.get("valid_force_mask")]
        invalid_rows = [ep for res in all_results for ep in res["episodes"] if not ep.get("valid_force_mask")]
        lines.append(f"- Valid forced PASS episodes: {len(valid_rows)}")
        lines.append(f"- Invalid episodes (excluded): {len(invalid_rows)}")
        if invalid_rows:
            reasons = {}
            for ep in invalid_rows:
                r = ep.get("invalid_reason", "unknown")
                reasons[r] = reasons.get(r, 0) + 1
            for r, c in reasons.items():
                lines.append(f"  - {r}: {c}")
        lines.append("")

        # Section 4: PASS initiation
        lines.append("## 4. PASS Initiation\n")
        pass_initiated = sum(1 for ep in valid_rows if ep.get("force_applied"))
        lines.append(f"- Forced PASS episodes: {pass_initiated}")
        lines.append(f"- PASS initiation events: {pass_initiated}")
        lines.append("")

        # Section 5: PASS trajectory
        lines.append("## 5. PASS Trajectory\n")
        angles = [ep["angular_error_to_nearest_teammate"] for ep in valid_rows if ep.get("angular_error_to_nearest_teammate") is not None]
        perp_dists = [ep["perpendicular_distance_to_pass_ray_tm1"] for ep in valid_rows if ep.get("perpendicular_distance_to_pass_ray_tm1") is not None]
        perp_dists += [ep["perpendicular_distance_to_pass_ray_tm2"] for ep in valid_rows if ep.get("perpendicular_distance_to_pass_ray_tm2") is not None]
        min_dists = [ep["min_ball_to_teammate_distance"] for ep in valid_rows if ep.get("min_ball_to_teammate_distance") is not None]
        if angles:
            lines.append(f"- Median angular error to nearest teammate: {np.median(angles):.3f} rad ({np.median(angles)*180/math.pi:.1f}°)")
        if perp_dists:
            lines.append(f"- Perpendicular distance to pass ray: median={np.median(perp_dists):.4f}, min={np.min(perp_dists):.4f}")
        if min_dists:
            lines.append(f"- Min ball-to-teammate distance: median={np.median(min_dists):.4f}, min={np.min(min_dists):.4f}")
        lines.append("")

        # Section 6: Teammate response
        lines.append("## 6. Teammate Response\n")
        speeds = [ep["median_teammate_speed_after_force"] for ep in valid_rows if ep.get("median_teammate_speed_after_force") is not None]
        fractions = [ep["fraction_teammates_with_nonzero_response"] for ep in valid_rows if ep.get("fraction_teammates_with_nonzero_response") is not None]
        if speeds:
            lines.append(f"- Median teammate speed after force: {np.median(speeds):.4f}")
        if fractions:
            lines.append(f"- Fraction teammates with non-zero response: {np.median(fractions):.2%}")
        lines.append("")

        # Section 7: Completion/event evidence
        lines.append("## 7. Completion / Event Evidence\n")
        pass_completed = sum(ep.get("pass_completed", 0) for ep in valid_rows)
        lines.append(f"- PASS_COMPLETED events seen: {pass_completed} / {len(valid_rows)}")
        ownership_changes = sum(
            1 for ep in valid_rows
            if ep.get("owner_before_min_distance") != ep.get("owner_after_min_distance")
            and ep.get("owner_before_min_distance") is not None
        )
        lines.append(f"- Ownership transitions at min-distance tick: {ownership_changes}")
        lines.append("")

        # Section 8: Rank assessment
        lines.append("## 8. Rank 1 / Rank 2 / Rank 3 Assessment\n")
        rank_counts: Dict[str, int] = {}
        for ep in valid_rows:
            r = ep.get("rank", "Unclassified")
            rank_counts[r] = rank_counts.get(r, 0) + 1
        for r, c in sorted(rank_counts.items()):
            lines.append(f"- **{r}:** {c} episodes ({c/len(valid_rows)*100:.1f}%)")
        lines.append("")

        # Section 9: Conclusion
        lines.append("## 9. Conclusion\n")
        if agg.get("pass_success_fraction", 0) > 0:
            lines.append("PASS_COMPLETED events were observed in this diagnostic run.")
        else:
            lines.append("No PASS_COMPLETED events were observed in this diagnostic run.")
        lines.append("")
        lines.append("This document diagnoses only. It does not propose or implement corrective overrides.")
        lines.append("")

        # Section 10: No proposed fix
        lines.append("## 10. No Proposed Fix\n")
        lines.append("This document is diagnostic-only. No corrective override is proposed or implemented.")
        lines.append("")

        with open(os.path.join(output_dir, "PASS_DIAGNOSTIC_FINDINGS.md"), "w") as f:
            f.write("\n".join(lines))

    def compute_summary(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        rows = [ep for res in all_results for ep in res["episodes"]]
        valid_rows = [r for r in rows if r.get("valid_force_mask")]
        pass_valid = [r for r in valid_rows if r.get("force_applied")]
        pass_completed = [r for r in pass_valid if r.get("pass_completed")]

        min_dists = [r["min_ball_to_teammate_distance"] for r in valid_rows if r.get("min_ball_to_teammate_distance") is not None]
        angles = [r["angular_error_to_nearest_teammate"] for r in valid_rows if r.get("angular_error_to_nearest_teammate") is not None]
        speeds = [r["median_teammate_speed_after_force"] for r in valid_rows if r.get("median_teammate_speed_after_force") is not None]
        fractions = [r["fraction_teammates_with_nonzero_response"] for r in valid_rows if r.get("fraction_teammates_with_nonzero_response") is not None]

        def percentile(data, p):
            if not data:
                return None
            return float(np.percentile(data, p))

        return {
            "valid_forced_passes": len(pass_valid),
            "pass_initiation_events": len(pass_valid),
            "pass_completed_events": len(pass_completed),
            "pass_completion_rate": len(pass_completed) / max(len(pass_valid), 1),
            "pass_success_fraction": len(pass_completed) / max(len(pass_valid), 1),
            "min_ball_to_teammate_distance": round(np.min(min_dists), 6) if min_dists else None,
            "median_ball_to_teammate_distance": round(np.median(min_dists), 6) if min_dists else None,
            "p10_ball_to_teammate_distance": round(percentile(min_dists, 10), 6) if min_dists else None,
            "p25_ball_to_teammate_distance": round(percentile(min_dists, 25), 6) if min_dists else None,
            "p50_ball_to_teammate_distance": round(percentile(min_dists, 50), 6) if min_dists else None,
            "p75_ball_to_teammate_distance": round(percentile(min_dists, 75), 6) if min_dists else None,
            "p90_ball_to_teammate_distance": round(percentile(min_dists, 90), 6) if min_dists else None,
            "median_angular_error": round(np.median(angles), 6) if angles else None,
            "median_teammate_speed_after_force": round(np.median(speeds), 6) if speeds else None,
            "fraction_teammates_with_nonzero_response": round(np.median(fractions), 6) if fractions else None,
        }

    # -- utilities --------------------------------------------------------

    @staticmethod
    def _git_head() -> str:
        try:
            return __import__("subprocess").check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            ).decode().strip()[:12]
        except Exception:
            return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Experiment F_act PASS Diagnostic")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MAPPO checkpoint")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper_onball")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for this run")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--base-seed", type=int, default=500000)
    parser.add_argument("--output-dir", type=str, default="training/results")
    parser.add_argument("--post-force-window", type=int, default=POST_FORCE_WINDOW)
    parser.add_argument("--diag-pass-trace", action="store_true", help="Diagnostic flag for PASS trace collection")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint_sha256 = sha256_of(args.checkpoint)
    actor, critic, obs_dim, action_dim, ckpt_timesteps = load_checkpoint(args.checkpoint)

    print("=" * 70)
    print(f"F_ACT PASS DIAGNOSTIC")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"SHA256     : {checkpoint_sha256[:16]}...")
    print(f"Timesteps  : {ckpt_timesteps}")
    print(f"Scenario   : {args.scenario}")
    print(f"Seed       : {args.seed}")
    print(f"Episodes   : {args.num_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Post-force window: {args.post_force_window}")
    print("=" * 70)

    evaluator = PassDiagnosticEvaluator(
        actor=actor,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_timesteps=ckpt_timesteps or 0,
        scenario=args.scenario,
        deterministic=args.deterministic,
        base_seed=args.base_seed,
        post_force_window=args.post_force_window,
    )

    seeds = [args.seed]
    all_results = evaluator.run_all(seeds=seeds, num_episodes_per_seed=args.num_episodes)

    evaluator.write_trace_csv(all_results, args.output_dir)
    evaluator.write_manifest(all_results, args.output_dir)
    evaluator.write_findings(all_results, args.output_dir)

    agg = evaluator.compute_summary(all_results)
    print("\n" + "=" * 70)
    print("PASS DIAGNOSTIC SUMMARY")
    print(f"Valid forced passes: {agg['valid_forced_passes']}")
    print(f"PASS_COMPLETED events: {agg['pass_completed_events']}")
    print(f"Pass completion rate: {agg['pass_completion_rate']*100:.1f}%")
    if agg["median_angular_error"] is not None:
        print(f"Median angular error: {agg['median_angular_error']*180/math.pi:.1f}°")
    if agg["median_ball_to_teammate_distance"] is not None:
        print(f"Median min ball-to-teammate distance: {agg['median_ball_to_teammate_distance']:.4f}")
    print(f"Median teammate speed: {agg['median_teammate_speed_after_force']}")
    print("=" * 70)

    # Save JSON result
    result_path = os.path.join(
        args.output_dir,
        f"pass_diag_seed{args.seed}_{os.path.splitext(os.path.basename(args.checkpoint))[0]}.json",
    )
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults JSON: {result_path}")


if __name__ == "__main__":
    main()
