"""
GMN-Football-3 — GNN Graph Builder (Phase 4)

Builds a schema-valid GNN graph from a single live engine observation.

Signature:
    build_graph(observation: dict, info: dict, scenario_id: str) -> dict

observation: per-agent dict from GMNMultiAgentEnv.step() / reset():
    {"observation": np.ndarray[127], "action_mask": np.ndarray[19]}

info: per-agent info dict from GMNMultiAgentEnv.step() / reset():
    {"score": {"left": int, "right": int}, "event": {"type": str} | None, ...}

scenario_id: string matching ScenarioConfig.id (e.g. "academy_3_vs_1_defender_3")

Returns: dict matching gnn_graph_schema.json v3 structure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLE_VOCABULARY = ["GK", "CB", "LB", "RB", "CDM", "CM", "LM", "RM", "LW", "RW", "CAM", "ST"]
ROLE_TO_INDEX = {r: i for i, r in enumerate(ROLE_VOCABULARY)}

NEAR_THRESHOLD = 0.065
LINE_LANE_GAP_THRESHOLD = 0.15

# ---------------------------------------------------------------------------
# Embedded scenario data (mirrors src/scenarios/ScenarioRegistry.ts)
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict[str, Any]] = {
    "academy_empty_goal": {
        "timeLimitSeconds": 15,
        "terminateOnOpponentPossession": True,
        "objectives": [
            {"id": "score_goal", "text": "Score a goal into the opponent net"},
            {"id": "within_time", "text": "Score within 15 seconds"},
        ],
        "rewards": {"scoring": 1.0, "completion": 100},
        "teamLeftPlayers": 1,
        "teamRightPlayers": 0,
        "hasGoalkeeperLeft": False,
        "hasGoalkeeperRight": False,
        "setup": {
            "leftPlayers": [{"role": "ST", "isControlled": True}],
            "rightPlayers": [],
        },
    },
    "academy_run_to_score": {
        "timeLimitSeconds": 20,
        "terminateOnOpponentPossession": True,
        "objectives": [
            {"id": "score_goal", "text": "Score past the goalkeeper"},
            {"id": "avoid_dispossess", "text": "Do not lose ball possession to the defender"},
        ],
        "rewards": {"scoring": 1.0, "completion": 200},
        "teamLeftPlayers": 1,
        "teamRightPlayers": 2,
        "hasGoalkeeperLeft": False,
        "hasGoalkeeperRight": True,
        "setup": {
            "leftPlayers": [{"role": "ST", "isControlled": True}],
            "rightPlayers": [
                {"role": "GK", "isControlled": False},
                {"role": "CB", "isControlled": False},
            ],
        },
    },
    "academy_pass_and_shoot_with_keeper": {
        "timeLimitSeconds": 25,
        "terminateOnOpponentPossession": True,
        "objectives": [
            {"id": "complete_pass", "text": "Complete at least one pass"},
            {"id": "score_goal", "text": "Score a goal"},
        ],
        "rewards": {"scoring": 1.0, "completion": 350},
        "teamLeftPlayers": 2,
        "teamRightPlayers": 2,
        "hasGoalkeeperLeft": False,
        "hasGoalkeeperRight": True,
        "setup": {
            "leftPlayers": [
                {"role": "LW", "isControlled": True},
                {"role": "ST", "isControlled": False},
            ],
            "rightPlayers": [
                {"role": "GK", "isControlled": False},
                {"role": "CB", "isControlled": False},
            ],
        },
    },
    "academy_3_vs_1_with_keeper": {
        "timeLimitSeconds": 30,
        "terminateOnOpponentPossession": True,
        "objectives": [
            {"id": "create_triangle", "text": "Complete 2+ passes in the episode"},
            {"id": "score_goal", "text": "Score past the goalkeeper"},
        ],
        "rewards": {"scoring": 1.0, "completion": 500},
        "teamLeftPlayers": 3,
        "teamRightPlayers": 2,
        "hasGoalkeeperLeft": False,
        "hasGoalkeeperRight": True,
        "setup": {
            "leftPlayers": [
                {"role": "CAM", "isControlled": True},
                {"role": "LW", "isControlled": False},
                {"role": "RW", "isControlled": False},
            ],
            "rightPlayers": [
                {"role": "GK", "isControlled": False},
                {"role": "CB", "isControlled": False},
            ],
        },
    },
    "academy_3_vs_1_defender_2": {
        "timeLimitSeconds": 30,
        "terminateOnOpponentPossession": True,
        "objectives": [
            {"id": "score_goal", "text": "Score past two defenders and goalkeeper"},
        ],
        "rewards": {"scoring": 1.0, "completion": 500},
        "teamLeftPlayers": 3,
        "teamRightPlayers": 3,
        "hasGoalkeeperLeft": False,
        "hasGoalkeeperRight": True,
        "setup": {
            "leftPlayers": [
                {"role": "CAM", "isControlled": True},
                {"role": "LW", "isControlled": False},
                {"role": "RW", "isControlled": False},
            ],
            "rightPlayers": [
                {"role": "GK", "isControlled": False},
                {"role": "CB", "isControlled": False},
                {"role": "CB", "isControlled": False},
            ],
        },
    },
    "academy_3_vs_1_defender_3": {
        "timeLimitSeconds": 30,
        "terminateOnOpponentPossession": True,
        "objectives": [
            {"id": "score_goal", "text": "Score past three defenders and goalkeeper"},
        ],
        "rewards": {"scoring": 1.0, "completion": 500},
        "teamLeftPlayers": 3,
        "teamRightPlayers": 4,
        "hasGoalkeeperLeft": False,
        "hasGoalkeeperRight": True,
        "setup": {
            "leftPlayers": [
                {"role": "CAM", "isControlled": True},
                {"role": "LW", "isControlled": False},
                {"role": "RW", "isControlled": False},
            ],
            "rightPlayers": [
                {"role": "GK", "isControlled": False},
                {"role": "CB", "isControlled": False},
                {"role": "CB", "isControlled": False},
                {"role": "CB", "isControlled": False},
            ],
        },
    },
    "academy_3_vs_1_keeper_aggressive": {
        "timeLimitSeconds": 30,
        "terminateOnOpponentPossession": True,
        "objectives": [
            {"id": "score_goal", "text": "Score past aggressive goalkeeper"},
        ],
        "rewards": {"scoring": 1.0, "completion": 500},
        "teamLeftPlayers": 3,
        "teamRightPlayers": 2,
        "hasGoalkeeperLeft": False,
        "hasGoalkeeperRight": True,
        "setup": {
            "leftPlayers": [
                {"role": "CAM", "isControlled": True},
                {"role": "LW", "isControlled": False},
                {"role": "RW", "isControlled": False},
            ],
            "rightPlayers": [
                {"role": "GK", "isControlled": False},
                {"role": "CB", "isControlled": False},
            ],
        },
    },
    "academy_3_vs_1_shifted": {
        "timeLimitSeconds": 30,
        "terminateOnOpponentPossession": True,
        "objectives": [
            {"id": "score_goal", "text": "Score past shifted defensive setup"},
        ],
        "rewards": {"scoring": 1.0, "completion": 500},
        "teamLeftPlayers": 3,
        "teamRightPlayers": 2,
        "hasGoalkeeperLeft": False,
        "hasGoalkeeperRight": True,
        "setup": {
            "leftPlayers": [
                {"role": "CAM", "isControlled": True},
                {"role": "LW", "isControlled": False},
                {"role": "RW", "isControlled": False},
            ],
            "rightPlayers": [
                {"role": "GK", "isControlled": False},
                {"role": "CB", "isControlled": False},
            ],
        },
    },
    "academy_3_vs_1_randomized": {
        "timeLimitSeconds": 30,
        "terminateOnOpponentPossession": True,
        "objectives": [
            {"id": "score_goal", "text": "Score past defender and goalkeeper under random jitter"},
        ],
        "rewards": {"scoring": 1.0, "completion": 500},
        "teamLeftPlayers": 3,
        "teamRightPlayers": 2,
        "hasGoalkeeperLeft": False,
        "hasGoalkeeperRight": True,
        "setup": {
            "leftPlayers": [
                {"role": "CAM", "isControlled": True},
                {"role": "LW", "isControlled": False},
                {"role": "RW", "isControlled": False},
            ],
            "rightPlayers": [
                {"role": "GK", "isControlled": False},
                {"role": "CB", "isControlled": False},
            ],
        },
    },
    "academy_rondo_4v1": {
        "timeLimitSeconds": 20,
        "terminateOnOpponentPossession": False,
        "objectives": [
            {"id": "retain_possession", "text": "Retain possession for the full 20 seconds"},
            {"id": "complete_passes", "text": "Complete 10+ passes"},
        ],
        "rewards": {"scoring": 0, "completion": 0},
        "teamLeftPlayers": 4,
        "teamRightPlayers": 1,
        "hasGoalkeeperLeft": False,
        "hasGoalkeeperRight": False,
        "setup": {
            "leftPlayers": [
                {"role": "CM", "isControlled": True},
                {"role": "CM", "isControlled": False},
                {"role": "LW", "isControlled": False},
                {"role": "RW", "isControlled": False},
            ],
            "rightPlayers": [
                {"role": "CB", "isControlled": False},
            ],
        },
    },
    "5_vs_5": {
        "timeLimitSeconds": 90,
        "terminateOnOpponentPossession": False,
        "objectives": [
            {"id": "win_match", "text": "Score more goals than the opponent"},
            {"id": "control_possession", "text": "Maintain > 50% possession"},
        ],
        "rewards": {"scoring": 1.0, "completion": 800},
        "teamLeftPlayers": 5,
        "teamRightPlayers": 5,
        "hasGoalkeeperLeft": True,
        "hasGoalkeeperRight": True,
        "setup": {
            "leftPlayers": [
                {"role": "GK", "isControlled": False},
                {"role": "CB", "isControlled": False},
                {"role": "LM", "isControlled": False},
                {"role": "RM", "isControlled": False},
                {"role": "ST", "isControlled": True},
            ],
            "rightPlayers": [
                {"role": "GK", "isControlled": False},
                {"role": "CB", "isControlled": False},
                {"role": "LM", "isControlled": False},
                {"role": "RM", "isControlled": False},
                {"role": "ST", "isControlled": False},
            ],
        },
    },
    "11_vs_11": {
        "timeLimitSeconds": 180,
        "terminateOnOpponentPossession": False,
        "objectives": [
            {"id": "win_match", "text": "Win the full 11v11 match"},
            {"id": "clean_sheet", "text": "Keep a clean sheet (concede 0 goals)"},
        ],
        "rewards": {"scoring": 1.0, "completion": 1200},
        "teamLeftPlayers": 11,
        "teamRightPlayers": 11,
        "hasGoalkeeperLeft": True,
        "hasGoalkeeperRight": True,
        "setup": {
            "leftPlayers": [],
            "rightPlayers": [],
        },
        "formation": "4-3-3",
    },
}

# ---------------------------------------------------------------------------
# Embedded formation template data (mirrors src/engine/Rules.ts FORMATIONS)
# ---------------------------------------------------------------------------

FORMATIONS: dict[str, list[dict[str, Any]]] = {
    "4-3-3": [
        {"role": "GK", "xRatio": 0.05, "yRatio": 0.5},
        {"role": "LB", "xRatio": 0.25, "yRatio": 0.15},
        {"role": "CB", "xRatio": 0.22, "yRatio": 0.38},
        {"role": "CB", "xRatio": 0.22, "yRatio": 0.62},
        {"role": "RB", "xRatio": 0.25, "yRatio": 0.85},
        {"role": "CM", "xRatio": 0.48, "yRatio": 0.3},
        {"role": "CM", "xRatio": 0.44, "yRatio": 0.5},
        {"role": "CM", "xRatio": 0.48, "yRatio": 0.7},
        {"role": "LW", "xRatio": 0.75, "yRatio": 0.15},
        {"role": "ST", "xRatio": 0.82, "yRatio": 0.5},
        {"role": "RW", "xRatio": 0.75, "yRatio": 0.85},
    ],
    "4-4-2": [
        {"role": "GK", "xRatio": 0.05, "yRatio": 0.5},
        {"role": "LB", "xRatio": 0.25, "yRatio": 0.15},
        {"role": "CB", "xRatio": 0.22, "yRatio": 0.38},
        {"role": "CB", "xRatio": 0.22, "yRatio": 0.62},
        {"role": "RB", "xRatio": 0.25, "yRatio": 0.85},
        {"role": "LM", "xRatio": 0.5, "yRatio": 0.15},
        {"role": "CM", "xRatio": 0.48, "yRatio": 0.38},
        {"role": "CM", "xRatio": 0.48, "yRatio": 0.62},
        {"role": "RM", "xRatio": 0.5, "yRatio": 0.85},
        {"role": "ST", "xRatio": 0.8, "yRatio": 0.4},
        {"role": "ST", "xRatio": 0.8, "yRatio": 0.6},
    ],
    "3-5-2": [
        {"role": "GK", "xRatio": 0.05, "yRatio": 0.5},
        {"role": "CB", "xRatio": 0.22, "yRatio": 0.25},
        {"role": "CB", "xRatio": 0.2, "yRatio": 0.5},
        {"role": "CB", "xRatio": 0.22, "yRatio": 0.75},
        {"role": "LM", "xRatio": 0.48, "yRatio": 0.1},
        {"role": "CM", "xRatio": 0.45, "yRatio": 0.35},
        {"role": "CAM", "xRatio": 0.58, "yRatio": 0.5},
        {"role": "CM", "xRatio": 0.45, "yRatio": 0.65},
        {"role": "RM", "xRatio": 0.48, "yRatio": 0.9},
        {"role": "ST", "xRatio": 0.8, "yRatio": 0.38},
        {"role": "ST", "xRatio": 0.8, "yRatio": 0.62},
    ],
    "5-3-2": [
        {"role": "GK", "xRatio": 0.05, "yRatio": 0.5},
        {"role": "LB", "xRatio": 0.25, "yRatio": 0.12},
        {"role": "CB", "xRatio": 0.2, "yRatio": 0.3},
        {"role": "CB", "xRatio": 0.18, "yRatio": 0.5},
        {"role": "CB", "xRatio": 0.2, "yRatio": 0.7},
        {"role": "RB", "xRatio": 0.25, "yRatio": 0.88},
        {"role": "CM", "xRatio": 0.48, "yRatio": 0.3},
        {"role": "CM", "xRatio": 0.46, "yRatio": 0.5},
        {"role": "CM", "xRatio": 0.48, "yRatio": 0.7},
        {"role": "ST", "xRatio": 0.78, "yRatio": 0.4},
        {"role": "ST", "xRatio": 0.78, "yRatio": 0.6},
    ],
    "1-2-1": [
        {"role": "GK", "xRatio": 0.05, "yRatio": 0.5},
        {"role": "CB", "xRatio": 0.25, "yRatio": 0.5},
        {"role": "LM", "xRatio": 0.5, "yRatio": 0.2},
        {"role": "RM", "xRatio": 0.5, "yRatio": 0.8},
        {"role": "ST", "xRatio": 0.75, "yRatio": 0.5},
    ],
    "1-1-1": [
        {"role": "GK", "xRatio": 0.05, "yRatio": 0.5},
        {"role": "CB", "xRatio": 0.3, "yRatio": 0.5},
        {"role": "CM", "xRatio": 0.55, "yRatio": 0.5},
        {"role": "ST", "xRatio": 0.8, "yRatio": 0.5},
    ],
    "1-0": [
        {"role": "GK", "xRatio": 0.05, "yRatio": 0.5},
    ],
}

# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

_SCHEMA_PATH = Path(__file__).parent / "gnn_graph_schema.json"
with open(_SCHEMA_PATH, "r", encoding="utf-8") as _f:
    _SCHEMA = json.load(_f)


# ---------------------------------------------------------------------------
# Core algorithms (also used by tests for independent verification)
# ---------------------------------------------------------------------------


def _gap_detection(values: list[float], threshold: float) -> list[int]:
    """Assign group IDs to sorted values using 1D gap detection.

    Returns a list of group_ids parallel to the input values.
    """
    if not values:
        return []

    indexed = sorted(enumerate(values), key=lambda iv: iv[1])
    group_ids: list[int] = [0] * len(values)
    current_group = 0
    group_ids[indexed[0][0]] = current_group

    for i in range(1, len(indexed)):
        idx, val = indexed[i]
        prev_val = indexed[i - 1][1]
        if val - prev_val > threshold:
            current_group += 1
        group_ids[idx] = current_group

    return group_ids


def _euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def _compute_team_shape(team_positions: list[dict[str, float]], line_ids: list[int]) -> dict[str, Any]:
    """Compute formation deviation metrics for a team.

    Mirrors Phase 2 §3.3 formulas.
    """
    n = len(team_positions)
    if n == 0:
        return {
            "inter_line_spacing_variance": 0.0,
            "mean_line_compactness": 0.0,
            "num_lines": 1,
            "num_lanes": 1,
        }

    xs = [p["x"] for p in team_positions]
    ys = [p["y"] for p in team_positions]

    # Group by line_id
    lines: dict[int, list[tuple[float, float]]] = {}
    for i, lid in enumerate(line_ids):
        lines.setdefault(lid, []).append((xs[i], ys[i]))

    num_lines = len(lines)

    # Inter-line spacing variance
    if num_lines < 2:
        inter_line_spacing_variance = 0.0
    else:
        line_centroids = sorted(
            (np.mean([p[0] for p in pts]), lid) for lid, pts in lines.items()
        )
        spacings = [line_centroids[i + 1][0] - line_centroids[i][0] for i in range(len(line_centroids) - 1)]
        s_bar = sum(spacings) / len(spacings)
        variance = sum((s - s_bar) ** 2 for s in spacings) / len(spacings)
        inter_line_spacing_variance = variance

    # Line compactness (per-line)
    team_width = max(ys) - min(ys) if n > 1 else 0.0
    compactness_sum = 0.0
    for pts in lines.values():
        if len(pts) < 2:
            compactness_sum += 0.0
        else:
            y_vals = [p[1] for p in pts]
            y_bar = sum(y_vals) / len(y_vals)
            sigma = math.sqrt(sum((y - y_bar) ** 2 for y in y_vals) / len(y_vals))
            compactness = sigma / team_width if team_width > 0 else 0.0
            compactness_sum += compactness

    mean_line_compactness = compactness_sum / num_lines if num_lines > 0 else 0.0

    # Lane count (reuse lane_ids if available, else compute)
    lane_ids = _gap_detection(ys, LINE_LANE_GAP_THRESHOLD)
    num_lanes = max(lane_ids) + 1 if lane_ids else 1

    return {
        "inter_line_spacing_variance": inter_line_spacing_variance,
        "mean_line_compactness": mean_line_compactness,
        "num_lines": num_lines,
        "num_lanes": num_lanes,
    }


# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------


def _parse_observation(obs_dict: dict[str, Any]) -> dict[str, Any]:
    """Parse the 127-dim observation vector into structured fields."""
    obs = np.asarray(obs_dict["observation"], dtype=np.float32)
    if obs.shape[-1] != 127:
        raise ValueError(f"Expected 127-dim observation, got shape {obs.shape}")

    # Left team: positions at 0-21, velocities at 22-43
    left_positions = []
    left_velocities = []
    for i in range(11):
        x, y = float(obs[2 * i]), float(obs[2 * i + 1])
        left_positions.append({"x": x, "y": y})
        vx, vy = float(obs[22 + 2 * i]), float(obs[22 + 2 * i + 1])
        left_velocities.append({"vx": vx, "vy": vy})

    # Right team: positions at 44-65, velocities at 66-87
    right_positions = []
    right_velocities = []
    for i in range(11):
        x, y = float(obs[44 + 2 * i]), float(obs[44 + 2 * i + 1])
        right_positions.append({"x": x, "y": y})
        vx, vy = float(obs[66 + 2 * i]), float(obs[66 + 2 * i + 1])
        right_velocities.append({"vx": vx, "vy": vy})

    # Ball: position 88-90, velocity 91-93
    ball_pos = {
        "x": float(obs[88]),
        "y": float(obs[89]),
        "z": float(obs[90]),
    }
    ball_vel = {
        "vx": float(obs[91]),
        "vy": float(obs[92]),
        "vz": float(obs[93]),
    }

    # Ball ownership one-hot: 94=none, 95=left, 96=right
    ball_owned_none = bool(obs[94] > 0.5)
    ball_owned_left = bool(obs[95] > 0.5)
    ball_owned_right = bool(obs[96] > 0.5)
    if ball_owned_left:
        ball_ownership = "left"
    elif ball_owned_right:
        ball_ownership = "right"
    else:
        ball_ownership = "none"

    # Active player one-hot: 97-107
    active_raw = obs[97:108]
    active_index = int(np.argmax(active_raw)) if active_raw.max() > 0.5 else -1

    # Game mode one-hot: 108-114
    game_mode_raw = obs[108:115]
    game_mode_names = [
        "Normal",
        "KickOff",
        "GoalKick",
        "FreeKick",
        "Corner",
        "ThrowIn",
        "Penalty",
    ]
    game_mode = game_mode_names[int(np.argmax(game_mode_raw))] if game_mode_raw.max() > 0.5 else "Normal"

    # Agent role one-hot: 115-126
    role_raw = obs[115:127]
    viewpoint_role = ROLE_VOCABULARY[int(np.argmax(role_raw))] if role_raw.max() > 0.5 else "UNKNOWN"

    return {
        "left_positions": left_positions,
        "left_velocities": left_velocities,
        "right_positions": right_positions,
        "right_velocities": right_velocities,
        "ball_pos": ball_pos,
        "ball_vel": ball_vel,
        "ball_ownership": ball_ownership,
        "active_index": active_index,
        "game_mode": game_mode,
        "viewpoint_role": viewpoint_role,
    }


def _count_active_players(positions: list[dict[str, float]]) -> int:
    """Count non-sentinel player slots."""
    count = 0
    for p in positions:
        if p["x"] != -1.0 or p["y"] != -1.0:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------


def _get_scenario_roles(scenario_id: str, team: str) -> list[str]:
    """Get ordered role list for a team from scenario config."""
    scenario = SCENARIOS.get(scenario_id)
    if scenario is None:
        raise KeyError(f"Unknown scenario_id: {scenario_id}")

    if scenario_id == "11_vs_11":
        formation = scenario.get("formation", "4-3-3")
        template = FORMATIONS.get(formation, FORMATIONS["4-3-3"])
        return [node["role"] for node in template[: scenario["teamLeftPlayers" if team == "left" else "teamRightPlayers"]]]

    setup_key = "leftPlayers" if team == "left" else "rightPlayers"
    return [p["role"] for p in scenario["setup"][setup_key]]


def _get_controlled_player_id(scenario_id: str, team: str) -> str | None:
    """Find the controlled player ID for a team in the scenario config."""
    scenario = SCENARIOS.get(scenario_id)
    if scenario is None:
        return None
    setup_key = "leftPlayers" if team == "left" else "rightPlayers"
    for i, p in enumerate(scenario["setup"].get(setup_key, [])):
        if p.get("isControlled", False):
            return f"{team}_{i}"
    return f"{team}_0"


def _is_gk_from_role(role: str) -> bool:
    return role == "GK"


def _role_one_hot(role: str) -> list[float]:
    vec = [0.0] * 12
    idx = ROLE_TO_INDEX.get(role)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _get_formation_slot_positions(formation: str, team: str, num_players: int) -> list[dict[str, Any]]:
    """Mirror of Rules.ts getFormationPositions."""
    template = FORMATIONS.get(formation, FORMATIONS["4-3-3"])
    nodes = template[:num_players]
    result = []
    for node in nodes:
        x_ratio = node["xRatio"]
        y_ratio = node["yRatio"]
        if team == "left":
            x = -1.0 + x_ratio * 1.2
            y = -0.42 + y_ratio * 0.84
        else:
            x = 1.0 - x_ratio * 1.2
            y = 0.42 - y_ratio * 0.84
        result.append({
            "role": node["role"],
            "xRatio": x_ratio,
            "yRatio": y_ratio,
            "x": x,
            "y": y,
        })
    return result


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------


def _build_player_nodes(
    parsed: dict[str, Any],
    scenario_id: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build PLAYER nodes. Returns (nodes, player_map) where player_map
    maps global_id -> node dict for edge construction."""
    scenario = SCENARIOS.get(scenario_id)
    if scenario is None:
        raise KeyError(f"Unknown scenario_id: {scenario_id}")

    left_roles = _get_scenario_roles(scenario_id, "left")
    right_roles = _get_scenario_roles(scenario_id, "right")
    left_count = scenario["teamLeftPlayers"]
    right_count = scenario["teamRightPlayers"]

    left_positions = parsed["left_positions"][:left_count]
    left_velocities = parsed["left_velocities"][:left_count]
    right_positions = parsed["right_positions"][:right_count]
    right_velocities = parsed["right_velocities"][:right_count]

    # Active index in observation maps to index within the viewpoint player's team.
    # We don't know the viewpoint team from the observation alone, so we
    # default to marking the first left player as active when ambiguous.
    # The active one-hot is at indices 97-107; its index is within the
    # viewpoint team's 0-based roster slot.
    active_idx = parsed["active_index"]

    # Determine viewpoint team: role one-hot at 115-126 is the viewpoint player's role.
    # We match it against known roles to infer team.
    viewpoint_role = parsed["viewpoint_role"]
    viewpoint_team = "left"
    if viewpoint_role in right_roles:
        viewpoint_team = "right"
    elif viewpoint_role in left_roles:
        viewpoint_team = "left"
    # If role is ambiguous (e.g., both teams have a ST), default to left.

    nodes: list[dict[str, Any]] = []
    player_map: dict[str, dict[str, Any]] = {}

    # Left team
    for i in range(left_count):
        role = left_roles[i] if i < len(left_roles) else "UNKNOWN"
        pos = left_positions[i] if i < len(left_positions) else {"x": -1.0, "y": -1.0}
        vel = left_velocities[i] if i < len(left_velocities) else {"vx": 0.0, "vy": 0.0}
        global_id = f"left_{i}"
        is_active = (viewpoint_team == "left" and active_idx == i)
        is_controlled = (i == 0 and scenario["setup"].get("leftPlayers", []) == []) or (
            i < len(scenario["setup"].get("leftPlayers", []))
            and scenario["setup"]["leftPlayers"][i].get("isControlled", False)
        )
        # For 11_vs_11, default left_0 as controlled (matches GameEngine.ts players[0] fallback).
        if scenario_id == "11_vs_11" and i == 0:
            is_controlled = True

        node = {
            "node_type": "PLAYER",
            "global_id": global_id,
            "team": "left",
            "team_index": i,
            "position": pos,
            "velocity": vel,
            "role": role,
            "role_one_hot": _role_one_hot(role),
            "is_active": is_active,
            "is_controlled": is_controlled,
            "is_goalkeeper": _is_gk_from_role(role),
            "features": {},  # populated below
            "line_id": 0,
            "lane_id": 0,
        }
        nodes.append(node)
        player_map[global_id] = node

    # Right team
    for i in range(right_count):
        role = right_roles[i] if i < len(right_roles) else "UNKNOWN"
        pos = right_positions[i] if i < len(right_positions) else {"x": -1.0, "y": -1.0}
        vel = right_velocities[i] if i < len(right_velocities) else {"vx": 0.0, "vy": 0.0}
        global_id = f"right_{i}"
        is_active = (viewpoint_team == "right" and active_idx == i)

        node = {
            "node_type": "PLAYER",
            "global_id": global_id,
            "team": "right",
            "team_index": i,
            "position": pos,
            "velocity": vel,
            "role": role,
            "role_one_hot": _role_one_hot(role),
            "is_active": is_active,
            "is_controlled": False,
            "is_goalkeeper": _is_gk_from_role(role),
            "features": {},
            "line_id": 0,
            "lane_id": 0,
        }
        nodes.append(node)
        player_map[global_id] = node

    # Compute per-player derived features and line/lane assignments
    for team in ("left", "right"):
        team_nodes = [n for n in nodes if n["team"] == team]
        team_positions = [n["position"] for n in team_nodes]
        xs = [p["x"] for p in team_positions]
        ys = [p["y"] for p in team_positions]

        # Filter out sentinel positions for feature computation
        valid_xs = [x for x in xs if x != -1.0]
        valid_ys = [y for y in ys if y != -1.0]
        valid_positions = [p for p in team_positions if p["x"] != -1.0 or p["y"] != -1.0]

        # Line/lane assignment (gap detection on teammates only)
        line_ids = _gap_detection(xs, LINE_LANE_GAP_THRESHOLD)
        lane_ids = _gap_detection(ys, LINE_LANE_GAP_THRESHOLD)

        for idx, node in enumerate(team_nodes):
            node["line_id"] = line_ids[idx]
            node["lane_id"] = lane_ids[idx]

        # Derived features (require >= 2 valid players)
        if len(valid_positions) >= 2:
            # Nearest teammate / opponent distances
            for node in team_nodes:
                if node["position"]["x"] == -1.0 and node["position"]["y"] == -1.0:
                    node["features"] = {
                        "nearest_teammate_dist": 0.0,
                        "nearest_opponent_dist": 0.0,
                        "team_width": 0.0,
                        "team_depth": 0.0,
                        "compactness": 0.0,
                        "stretch": 0.0,
                        "receiver_availability": 0.0,
                    }
                    continue

                # Nearest teammate
                team_dists = []
                for other in team_nodes:
                    if other is node:
                        continue
                    if other["position"]["x"] == -1.0 and other["position"]["y"] == -1.0:
                        continue
                    d = _euclidean(node["position"]["x"], node["position"]["y"], other["position"]["x"], other["position"]["y"])
                    team_dists.append(d)
                nearest_teammate = min(team_dists) if team_dists else 0.0

                # Nearest opponent
                opp_nodes = [n for n in nodes if n["team"] != team and not (n["position"]["x"] == -1.0 and n["position"]["y"] == -1.0)]
                opp_dists = [
                    _euclidean(node["position"]["x"], node["position"]["y"], o["position"]["x"], o["position"]["y"])
                    for o in opp_nodes
                ]
                nearest_opponent = min(opp_dists) if opp_dists else 0.0

                # Team width / depth
                team_width = max(valid_ys) - min(valid_ys) if valid_ys else 0.0
                team_depth = max(valid_xs) - min(valid_xs) if valid_xs else 0.0

                # Compactness (mean pairwise distance among teammates)
                if len(team_dists) >= 1:
                    compactness = sum(team_dists) / len(team_dists)
                else:
                    compactness = 0.0

                # Stretch = team_depth
                stretch = team_depth

                # Receiver availability
                receiver_availability = 1.0 if nearest_opponent > NEAR_THRESHOLD else 0.0

                node["features"] = {
                    "nearest_teammate_dist": nearest_teammate,
                    "nearest_opponent_dist": nearest_opponent,
                    "team_width": team_width,
                    "team_depth": team_depth,
                    "compactness": compactness,
                    "stretch": stretch,
                    "receiver_availability": receiver_availability,
                }
        else:
            for node in team_nodes:
                node["features"] = {
                    "nearest_teammate_dist": 0.0,
                    "nearest_opponent_dist": 0.0,
                    "team_width": 0.0,
                    "team_depth": 0.0,
                    "compactness": 0.0,
                    "stretch": 0.0,
                    "receiver_availability": 0.0,
                }

    return nodes, player_map


def _build_ball_node(parsed: dict[str, Any]) -> dict[str, Any]:
    """Build BALL node."""
    speed = math.hypot(parsed["ball_vel"]["vx"], parsed["ball_vel"]["vy"], parsed["ball_vel"]["vz"])
    return {
        "node_type": "BALL",
        "node_id": "ball",
        "position": {
            "x": parsed["ball_pos"]["x"],
            "y": parsed["ball_pos"]["y"],
            "z": parsed["ball_pos"]["z"],
        },
        "velocity": {
            "vx": parsed["ball_vel"]["vx"],
            "vy": parsed["ball_vel"]["vy"],
            "vz": parsed["ball_vel"]["vz"],
        },
        "ownership": parsed["ball_ownership"],
        "speed": speed,
    }


def _build_goal_nodes() -> list[dict[str, Any]]:
    """Build GOAL nodes (goal_left at x=-1.0, goal_right at x=1.0)."""
    return [
        {
            "node_type": "GOAL",
            "node_id": "goal_left",
            "team": "left",
            "position": {"x": -1.0, "y": 0.0},
            "width": 0.14,
            "height": 0.05,
            "depth": 0.04,
        },
        {
            "node_type": "GOAL",
            "node_id": "goal_right",
            "team": "right",
            "position": {"x": 1.0, "y": 0.0},
            "width": 0.14,
            "height": 0.05,
            "depth": 0.04,
        },
    ]


def _build_scenario_node(scenario_id: str, info: dict[str, Any]) -> dict[str, Any]:
    """Build SCENARIO node."""
    scenario = SCENARIOS.get(scenario_id)
    if scenario is None:
        raise KeyError(f"Unknown scenario_id: {scenario_id}")

    # Objective vocabulary: multi-hot over alphabetical objective ids
    vocab = [
        "avoid_dispossess",
        "clean_sheet",
        "complete_pass",
        "complete_passes",
        "control_possession",
        "create_triangle",
        "retain_possession",
        "score_goal",
        "within_time",
        "win_match",
    ]
    objective_ids = {obj["id"] for obj in scenario["objectives"]}
    objective_vocabulary = [1 if v in objective_ids else 0 for v in vocab]

    # Score from info if available
    score = info.get("score", {"left": 0, "right": 0})
    # reward_scoring / reward_completion from scenario config
    scoring_reward = scenario["rewards"]["scoring"]
    completion_reward = scenario["rewards"]["completion"]

    return {
        "node_type": "SCENARIO",
        "node_id": "scenario",
        "id": scenario_id,
        "time_limit_seconds": scenario["timeLimitSeconds"],
        "terminate_on_opponent_possession": scenario["terminateOnOpponentPossession"],
        "objectives": [
            {
                "id": obj["id"],
                "text": obj["text"],
                "is_completed": False,
                "is_failed": False,
            }
            for obj in scenario["objectives"]
        ],
        "rewards": {
            "scoring": scoring_reward,
            "completion": completion_reward,
        },
        "objective_vocabulary": objective_vocabulary,
        "reward_scoring": scoring_reward,
        "reward_completion": completion_reward,
    }


def _build_team_shape_nodes(
    left_players: list[dict[str, Any]],
    right_players: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build TEAM_SHAPE nodes (one per team)."""
    nodes = []

    for team, players in [("left", left_players), ("right", right_players)]:
        positions = [p["position"] for p in players if p["position"]["x"] != -1.0 or p["position"]["y"] != -1.0]
        line_ids = [p["line_id"] for p in players]

        shape = _compute_team_shape(positions, line_ids)
        nodes.append({
            "node_type": "TEAM_SHAPE",
            "team": team,
            "formation_deviation": shape,
        })

    return nodes


def _build_formation_slot_nodes(scenario_id: str) -> list[dict[str, Any]]:
    """Build FORMATION_SLOT nodes for 11_vs_11 only. Empty list otherwise."""
    if scenario_id != "11_vs_11":
        return []

    scenario = SCENARIOS.get(scenario_id)
    if scenario is None:
        return []

    formation = scenario.get("formation", "4-3-3")
    num_left = scenario["teamLeftPlayers"]
    num_right = scenario["teamRightPlayers"]

    nodes = []
    for team, num in [("left", num_left), ("right", num_right)]:
        slots = _get_formation_slot_positions(formation, team, num)
        for i, slot in enumerate(slots):
            nodes.append({
                "node_type": "FORMATION_SLOT",
                "slot_id": f"{team}_slot{i}",
                "team": team,
                "role": slot["role"],
                "xRatio": slot["xRatio"],
                "yRatio": slot["yRatio"],
                "x": slot["x"],
                "y": slot["y"],
            })

    return nodes


# ---------------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------------


def _build_edges(
    player_nodes: list[dict[str, Any]],
    ball_node_id: str,
    scenario_id: str,
) -> list[dict[str, Any]]:
    """Build all edges for the graph."""
    edges: list[dict[str, Any]] = []

    # Index by team
    left_players = [n for n in player_nodes if n["team"] == "left"]
    right_players = [n for n in player_nodes if n["team"] == "right"]

    # Helper to add edges with deterministic ordering
    def _add_edge(edge: dict[str, Any]) -> None:
        edges.append(edge)

    # TEAMMATE edges (within each team)
    for team_players in (left_players, right_players):
        for i in range(len(team_players)):
            for j in range(i + 1, len(team_players)):
                _add_edge({
                    "edge_type": "TEAMMATE",
                    "source": team_players[i]["global_id"],
                    "target": team_players[j]["global_id"],
                })

    # OPPONENT edges (cross-team)
    for lp in left_players:
        for rp in right_players:
            _add_edge({
                "edge_type": "OPPONENT",
                "source": lp["global_id"],
                "target": rp["global_id"],
            })

    # NEAR edges (distance < 0.065)
    all_players = left_players + right_players
    for i in range(len(all_players)):
        for j in range(i + 1, len(all_players)):
            if all_players[i]["team"] == all_players[j]["team"]:
                continue  # NEAR is cross-team only (same-team proximity is TEAMMATE)
            pi = all_players[i]["position"]
            pj = all_players[j]["position"]
            dist = _euclidean(pi["x"], pi["y"], pj["x"], pj["y"])
            if dist < NEAR_THRESHOLD:
                _add_edge({
                    "edge_type": "NEAR",
                    "source": all_players[i]["global_id"],
                    "target": all_players[j]["global_id"],
                    "distance": dist,
                    "threshold": NEAR_THRESHOLD,
                })

    # POSSESSES edges
    ball_ownership = None
    for n in player_nodes:
        # We need ball ownership from the parsed observation, but it's not
        # directly on the player node. We'll set it from the observation
        # parsing step. For now, this is handled in build_graph.
        pass

    # Note: POSSESSES edges are added in build_graph after we know ownership.

    # Formation edges (11_vs_11 only)
    if scenario_id == "11_vs_11":
        formation_slots = [n for n in player_nodes if n.get("node_type") == "FORMATION_SLOT"]
        # Wait, formation slots are separate nodes, not in player_nodes.
        # We'll handle this in build_graph.
        pass

    # Sort edges deterministically
    edges.sort(key=lambda e: (e["edge_type"], e["source"], e.get("target", "")))

    return edges


def _build_formation_edges(
    formation_slot_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build FORMATION_ADJACENCY, FORMATION_LINE, FORMATION_LANE edges for 11_vs_11."""
    edges: list[dict[str, Any]] = []

    if not formation_slot_nodes:
        return edges

    # Group by team
    by_team: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for slot in formation_slot_nodes:
        by_team[slot["team"]].append(slot)

    for team, slots in by_team.items():
        if len(slots) < 2:
            continue

        # FORMATION_LINE edges: group by xRatio bands
        # GK: xRatio < 0.10, Defense: 0.10 <= xRatio < 0.35, Midfield: 0.35 <= xRatio < 0.60, Attack: >= 0.60
        line_bands = [
            (0, 0.10),
            (1, 0.35),
            (2, 0.60),
            (3, float("inf")),
        ]

        def _line_id(x_ratio: float) -> int:
            for lid, threshold in line_bands:
                if x_ratio < threshold:
                    return lid
            return 3

        line_groups: dict[int, list[dict[str, Any]]] = {}
        for slot in slots:
            lid = _line_id(slot["xRatio"])
            line_groups.setdefault(lid, []).append(slot)

        for lid, group in line_groups.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    edges.append({
                        "edge_type": "FORMATION_LINE",
                        "source": group[i]["slot_id"],
                        "target": group[j]["slot_id"],
                        "line_id": lid,
                    })

        # FORMATION_LANE edges: group by yRatio bands
        # Left: yRatio < 0.33, Center: 0.33 <= yRatio <= 0.67, Right: yRatio > 0.67
        lane_bands = [
            (0, 0.33),
            (1, 0.67),
            (2, float("inf")),
        ]

        def _lane_id(y_ratio: float) -> int:
            for lid, threshold in lane_bands:
                if y_ratio < threshold:
                    return lid
            return 2

        lane_groups: dict[int, list[dict[str, Any]]] = {}
        for slot in slots:
            lid = _lane_id(slot["yRatio"])
            lane_groups.setdefault(lid, []).append(slot)

        for lid, group in lane_groups.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    edges.append({
                        "edge_type": "FORMATION_LANE",
                        "source": group[i]["slot_id"],
                        "target": group[j]["slot_id"],
                        "lane_id": lid,
                    })

        # FORMATION_ADJACENCY edges: k=2 nearest neighbors in (xRatio, yRatio) space
        for slot in slots:
            distances = []
            for other in slots:
                if other is slot:
                    continue
                dx = slot["xRatio"] - other["xRatio"]
                dy = slot["yRatio"] - other["yRatio"]
                distances.append((math.hypot(dx, dy), other))
            distances.sort(key=lambda x: x[0])
            for dist, neighbor in distances[:2]:
                edges.append({
                    "edge_type": "FORMATION_ADJACENCY",
                    "source": slot["slot_id"],
                    "target": neighbor["slot_id"],
                })

    edges.sort(key=lambda e: (e["edge_type"], e["source"], e.get("target", "")))
    return edges


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_graph(observation: dict[str, Any], info: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    """Build a GNN graph from a live engine observation.

    Args:
        observation: per-agent dict from env.step() / env.reset() with keys
            "observation" (np.ndarray[127]) and "action_mask".
        info: per-agent info dict from env.step() / env.reset().
        scenario_id: scenario identifier string.

    Returns:
        dict matching gnn_graph_schema.json v3.
    """
    if scenario_id not in SCENARIOS:
        raise KeyError(f"Unknown scenario_id: {scenario_id}")

    parsed = _parse_observation(observation)
    scenario = SCENARIOS[scenario_id]

    # Build PLAYER nodes
    player_nodes, player_map = _build_player_nodes(parsed, scenario_id)

    # Build BALL node
    ball_node = _build_ball_node(parsed)
    ball_node_id = ball_node["node_id"]

    # Build GOAL nodes
    goal_nodes = _build_goal_nodes()

    # Build SCENARIO node
    scenario_node = _build_scenario_node(scenario_id, info)

    # Build TEAM_SHAPE nodes
    left_players = [n for n in player_nodes if n["team"] == "left"]
    right_players = [n for n in player_nodes if n["team"] == "right"]
    team_shape_nodes = _build_team_shape_nodes(left_players, right_players)

    # Build FORMATION_SLOT nodes (11_vs_11 only)
    formation_slot_nodes = _build_formation_slot_nodes(scenario_id)

    # Assemble all nodes in deterministic order
    # Order: PLAYER (left_0..left_10, right_0..right_10), BALL, GOAL (left, right),
    #        TEAM_SHAPE (left, right), FORMATION_SLOT (left, right)
    # NOTE: SCENARIO node goes in the top-level "scenario" field, NOT in "nodes".
    all_nodes: list[dict[str, Any]] = []
    all_nodes.extend(player_nodes)
    all_nodes.append(ball_node)
    all_nodes.extend(goal_nodes)
    # scenario_node is NOT added to all_nodes — it lives at graph["scenario"]
    all_nodes.extend(team_shape_nodes)
    all_nodes.extend(formation_slot_nodes)

    # Build edges
    edges: list[dict[str, Any]] = []

    # TEAMMATE, OPPONENT, NEAR edges from player nodes
    all_player_nodes = player_nodes
    left_p = [n for n in all_player_nodes if n["team"] == "left"]
    right_p = [n for n in all_player_nodes if n["team"] == "right"]

    for team_players in (left_p, right_p):
        for i in range(len(team_players)):
            for j in range(i + 1, len(team_players)):
                edges.append({
                    "edge_type": "TEAMMATE",
                    "source": team_players[i]["global_id"],
                    "target": team_players[j]["global_id"],
                })

    for lp in left_p:
        for rp in right_p:
            edges.append({
                "edge_type": "OPPONENT",
                "source": lp["global_id"],
                "target": rp["global_id"],
            })

    for i in range(len(all_player_nodes)):
        for j in range(i + 1, len(all_player_nodes)):
            if all_player_nodes[i]["team"] == all_player_nodes[j]["team"]:
                continue
            pi = all_player_nodes[i]["position"]
            pj = all_player_nodes[j]["position"]
            dist = _euclidean(pi["x"], pi["y"], pj["x"], pj["y"])
            if dist < NEAR_THRESHOLD:
                edges.append({
                    "edge_type": "NEAR",
                    "source": all_player_nodes[i]["global_id"],
                    "target": all_player_nodes[j]["global_id"],
                    "distance": dist,
                    "threshold": NEAR_THRESHOLD,
                })

    # POSSESSES edges: connect ball to closest player of owning team
    if parsed["ball_ownership"] in ("left", "right"):
        owner_team = parsed["ball_ownership"]
        team_nodes = [n for n in all_player_nodes if n["team"] == owner_team]
        ball_x = parsed["ball_pos"]["x"]
        ball_y = parsed["ball_pos"]["y"]
        closest = None
        closest_dist = float("inf")
        for n in team_nodes:
            if n["position"]["x"] == -1.0 and n["position"]["y"] == -1.0:
                continue
            d = _euclidean(ball_x, ball_y, n["position"]["x"], n["position"]["y"])
            if d < closest_dist:
                closest_dist = d
                closest = n
        if closest is not None:
            edges.append({
                "edge_type": "POSSESSES",
                "source": closest["global_id"],
                "target": ball_node_id,
            })

    # Formation edges (11_vs_11)
    formation_edges = _build_formation_edges(formation_slot_nodes)
    edges.extend(formation_edges)

    # Deterministic sort
    edges.sort(key=lambda e: (e["edge_type"], e["source"], e.get("target", "")))

    graph = {
        "scenario": scenario_node,
        "nodes": all_nodes,
        "edges": edges,
    }

    # Schema validation
    jsonschema.validate(graph, _SCHEMA)

    return graph
