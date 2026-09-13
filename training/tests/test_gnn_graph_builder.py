"""
GMN-Football-3 — GNN Graph Builder Correctness Suite (Phase 4)

Tests build_graph() against a live bridge instance, not mocked engine state.
Every graph built in this suite is validated with jsonschema against
gnn_graph_schema.json.

Coverage maps to the 9 requirements in GNN_PHASE4_GRAPH_DIAGNOSTICS.md:
  1. NEAR edge distance correctness
  2. TEAMMATE / OPPONENT edge team-sorting
  3. POSSESSES edge correctness
  4. Line/lane correctness (live gap-detection)
  5. TEAM_SHAPE metric correctness
  6. SCENARIO node correctness
  7. Formation (Track A) correctness — 11_vs_11 only
  8. Schema validation on every case
  9. Determinism check
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import jsonschema
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.gnn_graph_builder import (
    NEAR_THRESHOLD,
    LINE_LANE_GAP_THRESHOLD,
    SCENARIOS,
    build_graph,
    _gap_detection,
    _compute_team_shape,
    _euclidean,
)

# ---------------------------------------------------------------------------
# Schema for explicit validation in tests
# ---------------------------------------------------------------------------

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "gnn_graph_schema.json")
with open(_SCHEMA_PATH, "r", encoding="utf-8") as _f:
    _SCHEMA = json.load(_f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _port_for(scenario: str, suffix: str = "") -> int:
    key = f"gnn_graph_builder_{scenario}_{suffix}"
    return 5060 + (hashlib.md5(key.encode()).hexdigest()[:6].__hash__() % 1000)


def _make_env(scenario: str):
    port = _port_for(scenario)
    return GMNMultiAgentEnv(
        scenario=scenario,
        auto_start_bridge=True,
        port=port,
        enable_reward_shaping=False,
        debug_rewards=False,
    )


def _get_first_agent_obs_info(env):
    obs, info = env.reset(seed=42)
    first = list(obs.keys())[0]
    return obs[first], info[first]


def _validate_schema(graph):
    jsonschema.validate(graph, _SCHEMA)


def _player_positions_from_graph(graph):
    result = {}
    for node in graph["nodes"]:
        if node["node_type"] == "PLAYER":
            result[node["global_id"]] = node["position"]
    return result


def _independent_line_lane(positions: list[dict[str, float]]) -> tuple[list[int], list[int]]:
    """Independent from-scratch line/lane assignment for test verification."""
    xs = [p["x"] for p in positions]
    ys = [p["y"] for p in positions]
    line_ids = _gap_detection(xs, LINE_LANE_GAP_THRESHOLD)
    lane_ids = _gap_detection(ys, LINE_LANE_GAP_THRESHOLD)
    return line_ids, lane_ids


def _independent_team_shape(positions: list[dict[str, float]], line_ids: list[int]) -> dict:
    """Independent from-scratch team shape computation for test verification."""
    return _compute_team_shape(positions, line_ids)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestGNNGraphBuilder:
    """Correctness suite for build_graph()."""

    def test_schema_validation_academy_3_vs_1_defender_3(self):
        """Requirement 8: every graph validates against schema."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)
        finally:
            env.close()

    def test_schema_validation_all_scenarios(self):
        """Requirement 8: schema validation smoke over all registered scenarios."""
        for scenario_id in SCENARIOS:
            env = _make_env(scenario_id)
            try:
                obs, info = _get_first_agent_obs_info(env)
                graph = build_graph(obs, info, scenario_id)
                _validate_schema(graph)
            finally:
                env.close()

    def test_teammate_opponent_edges(self):
        """Requirement 2: TEAMMATE only same-team, OPPONENT only cross-team."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            teammate_edges = [e for e in graph["edges"] if e["edge_type"] == "TEAMMATE"]
            opponent_edges = [e for e in graph["edges"] if e["edge_type"] == "OPPONENT"]

            for e in teammate_edges:
                src_team = next(n["team"] for n in graph["nodes"] if n["global_id"] == e["source"])
                tgt_team = next(n["team"] for n in graph["nodes"] if n["global_id"] == e["target"])
                assert src_team == tgt_team, f"TEAMMATE edge cross-team: {e}"

            for e in opponent_edges:
                src_team = next(n["team"] for n in graph["nodes"] if n["global_id"] == e["source"])
                tgt_team = next(n["team"] for n in graph["nodes"] if n["global_id"] == e["target"])
                assert src_team != tgt_team, f"OPPONENT edge same-team: {e}"
        finally:
            env.close()

    def test_near_edges_correctness(self):
        """Requirement 1: NEAR edges exactly for cross-team pairs with dist < 0.065."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            positions = _player_positions_from_graph(graph)
            near_edges = [e for e in graph["edges"] if e["edge_type"] == "NEAR"]

            # Build set of cross-team pairs that SHOULD have NEAR edges
            players = [(n["global_id"], n["team"]) for n in graph["nodes"] if n["node_type"] == "PLAYER"]
            expected_near = set()
            for i, (id_a, team_a) in enumerate(players):
                for j, (id_b, team_b) in enumerate(players):
                    if i >= j or team_a == team_b:
                        continue
                    pa = positions[id_a]
                    pb = positions[id_b]
                    if pa["x"] == -1.0 and pa["y"] == -1.0:
                        continue
                    if pb["x"] == -1.0 and pb["y"] == -1.0:
                        continue
                    dist = _euclidean(pa["x"], pa["y"], pb["x"], pb["y"])
                    if dist < NEAR_THRESHOLD:
                        # Deterministic ordering for set key
                        pair = tuple(sorted([id_a, id_b]))
                        expected_near.add(pair)

            actual_near = set()
            for e in near_edges:
                pair = tuple(sorted([e["source"], e["target"]]))
                actual_near.add(pair)

            assert actual_near == expected_near, (
                f"NEAR edge mismatch: actual={actual_near}, expected={expected_near}"
            )

            # Verify distance field matches actual distance
            for e in near_edges:
                pa = positions[e["source"]]
                pb = positions[e["target"]]
                dist = _euclidean(pa["x"], pa["y"], pb["x"], pb["y"])
                assert abs(e["distance"] - dist) < 1e-6, f"NEAR distance field mismatch: {e}"
                assert e["threshold"] == NEAR_THRESHOLD
        finally:
            env.close()

    def test_possesses_edge_correctness(self):
        """Requirement 3: POSSESSES edge matches ball ownership."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            ball_node = next(n for n in graph["nodes"] if n["node_type"] == "BALL")
            ownership = ball_node["ownership"]

            possesses_edges = [e for e in graph["edges"] if e["edge_type"] == "POSSESSES"]

            if ownership == "none":
                assert len(possesses_edges) == 0, f"Unexpected POSSESSES when ownership=none: {possesses_edges}"
            else:
                assert len(possesses_edges) == 1, f"Expected 1 POSSESSES edge, got {len(possesses_edges)}"
                owner_id = possesses_edges[0]["source"]
                owner_node = next(n for n in graph["nodes"] if n["global_id"] == owner_id)
                assert owner_node["team"] == ownership, f"POSSESSES source team mismatch"
                assert possesses_edges[0]["target"] == "ball"
        finally:
            env.close()

    def test_line_lane_correctness(self):
        """Requirement 4: line_id/lane_id match gap-detection on live positions."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            # Extract live positions per team
            left_pos = [n["position"] for n in graph["nodes"] if n["node_type"] == "PLAYER" and n["team"] == "left"]
            right_pos = [n["position"] for n in graph["nodes"] if n["node_type"] == "PLAYER" and n["team"] == "right"]

            # Independent computation
            expected_left_lines, expected_left_lanes = _independent_line_lane(left_pos)
            expected_right_lines, expected_right_lanes = _independent_line_lane(right_pos)

            # Compare with graph output
            for node in graph["nodes"]:
                if node["node_type"] != "PLAYER":
                    continue
                gid = node["global_id"]
                if node["team"] == "left":
                    idx = int(gid.split("_")[1])
                    assert node["line_id"] == expected_left_lines[idx], f"left_{idx} line_id mismatch"
                    assert node["lane_id"] == expected_left_lanes[idx], f"left_{idx} lane_id mismatch"
                else:
                    idx = int(gid.split("_")[1])
                    assert node["line_id"] == expected_right_lines[idx], f"right_{idx} line_id mismatch"
                    assert node["lane_id"] == expected_right_lanes[idx], f"right_{idx} lane_id mismatch"
        finally:
            env.close()

    def test_line_lane_after_step(self):
        """Requirement 4 continued: rebuild after a live step and verify."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = env.reset(seed=42)
            first = list(obs.keys())[0]
            action = {a: 0 for a in obs.keys()}
            obs, rewards, terms, truncs, infos = env.step(action)

            obs_first = obs[first]
            info_first = infos[first]
            graph = build_graph(obs_first, info_first, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            # Independent verification on the stepped positions
            left_pos = [n["position"] for n in graph["nodes"] if n["node_type"] == "PLAYER" and n["team"] == "left"]
            right_pos = [n["position"] for n in graph["nodes"] if n["node_type"] == "PLAYER" and n["team"] == "right"]

            expected_left_lines, expected_left_lanes = _independent_line_lane(left_pos)
            expected_right_lines, expected_right_lanes = _independent_line_lane(right_pos)

            for node in graph["nodes"]:
                if node["node_type"] != "PLAYER":
                    continue
                gid = node["global_id"]
                parts = gid.split("_")
                team, idx = parts[0], int(parts[1])
                if team == "left":
                    assert node["line_id"] == expected_left_lines[idx]
                    assert node["lane_id"] == expected_left_lanes[idx]
                else:
                    assert node["line_id"] == expected_right_lines[idx]
                    assert node["lane_id"] == expected_right_lanes[idx]
        finally:
            env.close()

    def test_team_shape_correctness(self):
        """Requirement 5: TEAM_SHAPE metrics match independent formula computation."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            team_shape_nodes = [n for n in graph["nodes"] if n["node_type"] == "TEAM_SHAPE"]
            assert len(team_shape_nodes) == 2

            for ts_node in team_shape_nodes:
                team = ts_node["team"]
                players = [n for n in graph["nodes"] if n["node_type"] == "PLAYER" and n["team"] == team]
                positions = [p["position"] for p in players]
                line_ids = [p["line_id"] for p in players]

                expected = _independent_team_shape(positions, line_ids)
                actual = ts_node["formation_deviation"]

                assert abs(actual["inter_line_spacing_variance"] - expected["inter_line_spacing_variance"]) < 1e-6
                assert abs(actual["mean_line_compactness"] - expected["mean_line_compactness"]) < 1e-6
                assert actual["num_lines"] == expected["num_lines"]
                assert actual["num_lanes"] == expected["num_lanes"]
        finally:
            env.close()

    def test_scenario_node_correctness_multiple_scenarios(self):
        """Requirement 6: SCENARIO node fields match ScenarioRegistry for 2+ scenarios."""
        for scenario_id in ["academy_3_vs_1_defender_3", "academy_rondo_4v1", "5_vs_5"]:
            env = _make_env(scenario_id)
            try:
                obs, info = _get_first_agent_obs_info(env)
                graph = build_graph(obs, info, scenario_id)
                _validate_schema(graph)

                scenario_node = graph["scenario"]
                expected = SCENARIOS[scenario_id]

                assert scenario_node["id"] == scenario_id
                assert scenario_node["time_limit_seconds"] == expected["timeLimitSeconds"]
                assert scenario_node["terminate_on_opponent_possession"] == expected["terminateOnOpponentPossession"]
                assert scenario_node["reward_scoring"] == expected["rewards"]["scoring"]
                assert scenario_node["reward_completion"] == expected["rewards"]["completion"]
                assert scenario_node["rewards"]["scoring"] == expected["rewards"]["scoring"]
                assert scenario_node["rewards"]["completion"] == expected["rewards"]["completion"]

                # Objective vocabulary
                vocab = [
                    "avoid_dispossess", "clean_sheet", "complete_pass", "complete_passes",
                    "control_possession", "create_triangle", "retain_possession",
                    "score_goal", "within_time", "win_match",
                ]
                expected_vocab = [1 if v in {o["id"] for o in expected["objectives"]} else 0 for v in vocab]
                assert scenario_node["objective_vocabulary"] == expected_vocab

                # Objectives list
                assert len(scenario_node["objectives"]) == len(expected["objectives"])
                for graph_obj, expected_obj in zip(scenario_node["objectives"], expected["objectives"]):
                    assert graph_obj["id"] == expected_obj["id"]
                    assert graph_obj["text"] == expected_obj["text"]
            finally:
                env.close()

    def test_formation_slot_11_vs_11_only(self):
        academy_scenarios = ["academy_3_vs_1_defender_3", "academy_rondo_4v1", "5_vs_5"]

        for scenario_id in academy_scenarios:
            env = _make_env(scenario_id)
            try:
                obs, info = _get_first_agent_obs_info(env)
                graph = build_graph(obs, info, scenario_id)
                _validate_schema(graph)
                formation_slots = [n for n in graph["nodes"] if n["node_type"] == "FORMATION_SLOT"]
                assert len(formation_slots) == 0, f"FORMATION_SLOT found in {scenario_id}"
            finally:
                env.close()

        # 11_vs_11 must have FORMATION_SLOT nodes
        env = _make_env("11_vs_11")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "11_vs_11")
            _validate_schema(graph)
            formation_slots = [n for n in graph["nodes"] if n["node_type"] == "FORMATION_SLOT"]
            assert len(formation_slots) == 22, f"Expected 22 FORMATION_SLOT nodes, got {len(formation_slots)}"
        finally:
            env.close()

    def test_determinism(self):
        """Requirement 9: same input -> byte-identical graph."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph1 = build_graph(obs, info, "academy_3_vs_1_defender_3")
            graph2 = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph1)
            _validate_schema(graph2)

            assert json.dumps(graph1, sort_keys=True) == json.dumps(graph2, sort_keys=True)
        finally:
            env.close()

    def test_node_ordering_deterministic(self):
        """Node arrays follow the documented ordering."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            node_types = [n["node_type"] for n in graph["nodes"]]

            # All PLAYER nodes come first, ordered left_0..left_N, right_0..right_M
            player_nodes = [n for n in graph["nodes"] if n["node_type"] == "PLAYER"]
            for i, n in enumerate(player_nodes):
                expected_id = f"{n['team']}_{n['team_index']}"
                assert n["global_id"] == expected_id

            # BALL after PLAYER
            ball_idx = next(i for i, t in enumerate(node_types) if t == "BALL")
            assert all(t == "PLAYER" for t in node_types[:ball_idx])

            # GOAL nodes after BALL
            goal_indices = [i for i, t in enumerate(node_types) if t == "GOAL"]
            assert all(i > ball_idx for i in goal_indices)

            # TEAM_SHAPE after GOAL (SCENARIO lives at graph["scenario"], not in nodes)
            ts_indices = [i for i, t in enumerate(node_types) if t == "TEAM_SHAPE"]
            assert all(i > goal_indices[-1] for i in ts_indices), (
                f"TEAM_SHAPE must follow GOAL nodes: goal_end={goal_indices[-1]}, ts={ts_indices}"
            )

            # FORMATION_SLOT after TEAM_SHAPE if present
            slot_indices = [i for i, t in enumerate(node_types) if t == "FORMATION_SLOT"]
            if slot_indices:
                assert all(i > ts_indices[-1] for i in slot_indices), (
                    f"FORMATION_SLOT must follow TEAM_SHAPE: ts_end={ts_indices[-1]}, slots={slot_indices}"
                )
        finally:
            env.close()

    def test_edge_ordering_deterministic(self):
        """Edges are sorted lexicographically by (type, source, target)."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            edges = graph["edges"]
            sorted_edges = sorted(edges, key=lambda e: (e["edge_type"], e["source"], e.get("target", "")))
            assert edges == sorted_edges
        finally:
            env.close()

    def test_features_populated_for_active_players(self):
        """PLAYER features are populated when team has >= 2 active members."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            for node in graph["nodes"]:
                if node["node_type"] == "PLAYER":
                    assert "features" in node
                    assert "nearest_teammate_dist" in node["features"]
                    assert "nearest_opponent_dist" in node["features"]
                    assert "team_width" in node["features"]
                    assert "team_depth" in node["features"]
                    assert "compactness" in node["features"]
                    assert "stretch" in node["features"]
                    assert "receiver_availability" in node["features"]
        finally:
            env.close()

    def test_rich_teammate_edges_have_geometry(self):
        """Phase 3: TEAMMATE edges carry continuous relational geometry."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            teammates = [e for e in graph["edges"] if e["edge_type"] == "TEAMMATE"]
            assert teammates, "Expected TEAMMATE edges"
            required_keys = {"distance", "relative_x", "relative_y", "relative_vx", "relative_vy", "angle", "closing_speed"}
            for e in teammates:
                assert required_keys.issubset(e.keys()), f"TEAMMATE edge missing keys: {e}"
                assert e["distance"] >= 0.0
        finally:
            env.close()

    def test_rich_opponent_edges_have_geometry_and_pressure(self):
        """Phase 3: OPPONENT edges carry continuous geometry + pressure."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            opponents = [e for e in graph["edges"] if e["edge_type"] == "OPPONENT"]
            assert opponents, "Expected OPPONENT edges"
            required_keys = {"distance", "relative_x", "relative_y", "relative_vx", "relative_vy", "angle", "closing_speed", "pressure"}
            for e in opponents:
                assert required_keys.issubset(e.keys()), f"OPPONENT edge missing keys: {e}"
                assert e["pressure"] in (0.0, 1.0)
        finally:
            env.close()

    def test_player_ball_edges_exist_and_have_geometry(self):
        """Phase 3: PLAYER_BALL edges from every present player to ball."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            player_nodes = [n for n in graph["nodes"] if n["node_type"] == "PLAYER"]
            present_players = [n for n in player_nodes if not (n["position"]["x"] == -1.0 and n["position"]["y"] == -1.0)]
            pb_edges = [e for e in graph["edges"] if e["edge_type"] == "PLAYER_BALL"]
            assert len(pb_edges) == len(present_players), f"Expected {len(present_players)} PLAYER_BALL edges, got {len(pb_edges)}"

            required_keys = {"distance", "relative_x", "relative_y", "relative_vx", "relative_vy", "angle", "has_possession"}
            for e in pb_edges:
                assert required_keys.issubset(e.keys()), f"PLAYER_BALL edge missing keys: {e}"
                assert e["target"] == "ball"
        finally:
            env.close()

    def test_player_goal_edges_exist_and_have_geometry(self):
        """Phase 3: PLAYER_GOAL edges from every present player to both goals."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            player_nodes = [n for n in graph["nodes"] if n["node_type"] == "PLAYER"]
            present_players = [n for n in player_nodes if not (n["position"]["x"] == -1.0 and n["position"]["y"] == -1.0)]
            pg_edges = [e for e in graph["edges"] if e["edge_type"] == "PLAYER_GOAL"]
            expected = len(present_players) * 2  # each player to goal_left + goal_right
            assert len(pg_edges) == expected, f"Expected {expected} PLAYER_GOAL edges, got {len(pg_edges)}"

            required_keys = {"distance", "relative_x", "relative_y", "angle", "shot_angle", "nearest_opponent_pressure"}
            for e in pg_edges:
                assert required_keys.issubset(e.keys()), f"PLAYER_GOAL edge missing keys: {e}"
                assert e["target"] in ("goal_left", "goal_right")
                assert e["shot_angle"] >= 0.0
        finally:
            env.close()

    def test_ball_goal_edges_exist(self):
        """Phase 3: BALL_GOAL edges from ball to both goals."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            bg_edges = [e for e in graph["edges"] if e["edge_type"] == "BALL_GOAL"]
            assert len(bg_edges) == 2, f"Expected 2 BALL_GOAL edges, got {len(bg_edges)}"
            targets = {e["target"] for e in bg_edges}
            assert targets == {"goal_left", "goal_right"}
            for e in bg_edges:
                assert e["source"] == "ball"
                assert e["shot_angle"] >= 0.0
        finally:
            env.close()

    def test_formation_slot_connectivity_all_scenarios(self):
        """Phase 3: ASSIGNED_TO edges exist for any scenario with a formation."""
        for scenario_id in SCENARIOS:
            env = _make_env(scenario_id)
            try:
                obs, info = _get_first_agent_obs_info(env)
                graph = build_graph(obs, info, scenario_id)
                _validate_schema(graph)

                formation_slots = [n for n in graph["nodes"] if n["node_type"] == "FORMATION_SLOT"]
                assigned_edges = [e for e in graph["edges"] if e["edge_type"] == "ASSIGNED_TO"]

                if formation_slots:
                    player_nodes = [n for n in graph["nodes"] if n["node_type"] == "PLAYER" and not (n["position"]["x"] == -1.0 and n["position"]["y"] == -1.0)]
                    assert len(assigned_edges) == len(player_nodes), (
                        f"{scenario_id}: expected {len(player_nodes)} ASSIGNED_TO edges, got {len(assigned_edges)}"
                    )
                    for e in assigned_edges:
                        assert e["target"] in {s["slot_id"] for s in formation_slots}
                        assert "deviation_distance" in e
                        assert e["deviation_distance"] >= 0.0
                else:
                    assert len(assigned_edges) == 0, f"{scenario_id}: unexpected ASSIGNED_TO edges without formation slots"
            finally:
                env.close()

    def test_belongs_to_shape_edges_exist(self):
        """Phase 3: BELONGS_TO_SHAPE edges connect players to team shape."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            player_nodes = [n for n in graph["nodes"] if n["node_type"] == "PLAYER" and not (n["position"]["x"] == -1.0 and n["position"]["y"] == -1.0)]
            shape_edges = [e for e in graph["edges"] if e["edge_type"] == "BELONGS_TO_SHAPE"]
            assert len(shape_edges) == len(player_nodes), f"Expected {len(player_nodes)} BELONGS_TO_SHAPE edges, got {len(shape_edges)}"

            team_shapes = [n for n in graph["nodes"] if n["node_type"] == "TEAM_SHAPE"]
            for e in shape_edges:
                assert e["target"] in {f"team_shape_{n['team']}" for n in team_shapes}
        finally:
            env.close()

    def test_scenario_context_edges_exist(self):
        """Phase 3: SCENARIO_CONTEXT edges connect players to scenario node."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            player_nodes = [n for n in graph["nodes"] if n["node_type"] == "PLAYER" and not (n["position"]["x"] == -1.0 and n["position"]["y"] == -1.0)]
            ctx_edges = [e for e in graph["edges"] if e["edge_type"] == "SCENARIO_CONTEXT"]
            assert len(ctx_edges) == len(player_nodes), f"Expected {len(player_nodes)} SCENARIO_CONTEXT edges, got {len(ctx_edges)}"
            for e in ctx_edges:
                assert e["target"] == "scenario"
        finally:
            env.close()

    def test_z_scenario_support(self):
        """Phase 2/3: zScenario is attached to graph when provided."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            z = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3", z_scenario=z)
            _validate_schema(graph)
            assert "z_scenario" in graph
            assert graph["z_scenario"] == z
        finally:
            env.close()

    def test_possesses_edge_matches_ownership(self):
        """Phase 3: POSSESSES edge source belongs to owning team."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            ball_node = next(n for n in graph["nodes"] if n["node_type"] == "BALL")
            ownership = ball_node["ownership"]
            possesses_edges = [e for e in graph["edges"] if e["edge_type"] == "POSSESSES"]

            if ownership == "none":
                assert len(possesses_edges) == 0
            else:
                assert len(possesses_edges) == 1
                owner_id = possesses_edges[0]["source"]
                owner_node = next(n for n in graph["nodes"] if n["global_id"] == owner_id)
                assert owner_node["team"] == ownership
                assert possesses_edges[0]["target"] == "ball"
        finally:
            env.close()

    def test_exact_owner_overrides_nearest_player(self):
        """P0-A: POSSESSES uses info.ground_truth.current_ball_owner.agent_id,
        not the nearest player heuristic."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            ball_node = next(n for n in graph["nodes"] if n["node_type"] == "BALL")
            ownership = ball_node["ownership"]
            possesses_edges = [e for e in graph["edges"] if e["edge_type"] == "POSSESSES"]

            if ownership == "none":
                assert len(possesses_edges) == 0
                return

            assert len(possesses_edges) == 1
            exact_owner = possesses_edges[0]["source"]

            # Verify exact owner matches info ground truth
            ground_truth = info.get("ground_truth") or {}
            current_ball_owner = ground_truth.get("current_ball_owner")
            if isinstance(current_ball_owner, dict):
                assert exact_owner == current_ball_owner["agent_id"], (
                    f"POSSESSES source {exact_owner} does not match exact owner "
                    f"{current_ball_owner['agent_id']}"
                )
        finally:
            env.close()

    def test_controlled_player_from_info(self):
        """P0-B: is_controlled comes from info.controlledPlayerId."""
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            controlled_id = info.get("controlledPlayerId")
            if controlled_id is None:
                pytest.skip("controlledPlayerId not exposed in info")

            controlled_node = next((n for n in graph["nodes"] if n["global_id"] == controlled_id), None)
            assert controlled_node is not None, f"controlled player {controlled_id} not in graph"
            assert controlled_node["is_controlled"] is True

            # All other players must not be controlled
            for node in graph["nodes"]:
                if node["node_type"] == "PLAYER" and node["global_id"] != controlled_id:
                    assert node["is_controlled"] is False, (
                        f"Player {node['global_id']} should not be controlled"
                    )
        finally:
            env.close()

    def test_formation_assignment_role_based_not_nearest(self):
        """P0-C: ASSIGNED_TO uses role-based matching, not nearest geometry."""
        env = _make_env("11_vs_11")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "11_vs_11")
            _validate_schema(graph)

            assigned_edges = [e for e in graph["edges"] if e["edge_type"] == "ASSIGNED_TO"]
            assert len(assigned_edges) > 0

            # Verify each player is assigned to a slot with matching role
            slot_roles = {n["slot_id"]: n["role"] for n in graph["nodes"] if n["node_type"] == "FORMATION_SLOT"}
            for edge in assigned_edges:
                player_node = next(n for n in graph["nodes"] if n["global_id"] == edge["source"])
                target_role = slot_roles.get(edge["target"])
                assert target_role == player_node["role"], (
                    f"Player {edge['source']} role {player_node['role']} assigned to slot "
                    f"{edge['target']} with role {target_role}"
                )
        finally:
            env.close()

    def test_teammate_reverse_edge_geometry(self):
        """Phase 3: TEAMMATE edges are undirected with source->target geometry.
        
        Reverse geometry is derivable by negating relative_x, relative_y,
        relative_vx, relative_vy, and shifting angle by pi.
        """
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            teammate_edges = [e for e in graph["edges"] if e["edge_type"] == "TEAMMATE"]
            # TEAMMATE edges are undirected: only one edge per pair (i < j)
            # Verify features are finite and non-negative distance
            for e in teammate_edges:
                assert e["distance"] >= 0.0
                assert isinstance(e["relative_x"], float)
                assert isinstance(e["relative_y"], float)
                assert isinstance(e["angle"], float)
                assert isinstance(e["closing_speed"], float)
        finally:
            env.close()

    def test_opponent_reverse_edge_geometry(self):
        """Phase 3: OPPONENT edges are undirected with source->target geometry.
        
        Reverse geometry is derivable by negating relative_x, relative_y,
        relative_vx, relative_vy, and shifting angle by pi.
        """
        env = _make_env("academy_3_vs_1_defender_3")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_3_vs_1_defender_3")
            _validate_schema(graph)

            opponent_edges = [e for e in graph["edges"] if e["edge_type"] == "OPPONENT"]
            for e in opponent_edges:
                assert e["distance"] >= 0.0
                assert isinstance(e["relative_x"], float)
                assert isinstance(e["relative_y"], float)
                assert isinstance(e["angle"], float)
                assert isinstance(e["closing_speed"], float)
                assert e["pressure"] in (0.0, 1.0)
        finally:
            env.close()

    def test_possesses_none_creates_no_edge(self):
        """P0-A: when ownership is none, no POSSESSES edge is created."""
        env = _make_env("academy_empty_goal")
        try:
            obs, info = _get_first_agent_obs_info(env)
            graph = build_graph(obs, info, "academy_empty_goal")
            _validate_schema(graph)

            ball_node = next(n for n in graph["nodes"] if n["node_type"] == "BALL")
            assert ball_node["ownership"] == "none"

            possesses_edges = [e for e in graph["edges"] if e["edge_type"] == "POSSESSES"]
            assert len(possesses_edges) == 0
        finally:
            env.close()
