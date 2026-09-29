"""
GMN-Football-3 — Ball/goal visibility tests (Fix A + Fix D)

Regression gate for training/results/OPEN_ITEM_GNN_BALL_VISIBILITY.md:

  Fix A — the player node vector carries ball-relative geometry, attacking-goal
          geometry, a nearest-to-ball flag, and a ground-truth possession flag on
          dims 32-38 (NODE_FEATURE_DIM grew 32 -> 39).
  Fix D — the pooled global embedding is fused into every agent embedding before
          the policy head, so ball/goal content reaches the actor even though the
          BALL and GOAL nodes are graph sinks.

Everything here is synthetic and hand-computed: no bridge, no Unity, no
checkpoint. Expected numbers follow the pitch convention used by
gnn_graph_builder (x in [-1, 1], y in [-0.42, 0.42]) and the normalizers in
gnn_graph_to_tensor (PITCH_LENGTH=2.0, PITCH_WIDTH=0.84, DISTANCE_NORM=1.414).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from training.gnn_encoders import create_encoder
from training.gnn_graph_builder import ROLE_VOCABULARY, build_graph
from training.gnn_graph_to_tensor import (
    DISTANCE_NORM,
    NODE_FEATURE_DIM,
    PITCH_LENGTH,
    PITCH_WIDTH,
    PLAYER_BALL_DISTANCE_DIM,
    PLAYER_BALL_REL_X_DIM,
    PLAYER_BALL_REL_Y_DIM,
    PLAYER_GOAL_REL_X_DIM,
    PLAYER_GOAL_REL_Y_DIM,
    PLAYER_HAS_POSSESSION_DIM,
    PLAYER_IS_NEAREST_TO_BALL_DIM,
    graph_to_tensors,
)

# ---------------------------------------------------------------------------
# Synthetic graph helpers
# ---------------------------------------------------------------------------

# Hand-placed geometry. Ball sits at (0.10, 0.20); left_0 is the carrier.
BALL_X, BALL_Y = 0.10, 0.20


def _player(global_id, team, x, y, is_controlled=False, has_possession=False):
    node = {
        "node_type": "PLAYER",
        "global_id": global_id,
        "team": team,
        "team_index": int(global_id.split("_")[1]),
        "position": {"x": x, "y": y},
        "velocity": {"vx": 0.0, "vy": 0.0},
        "role": "CM",
        "role_one_hot": [0.0] * 12,
        "is_active": False,
        "is_controlled": is_controlled,
        "is_goalkeeper": False,
        "features": {},
        "line_id": 0,
        "lane_id": 0,
    }
    if has_possession:
        node["has_possession"] = True
    return node


def _ball(x, y, z=0.0):
    return {
        "node_type": "BALL",
        "node_id": "ball",
        "position": {"x": x, "y": y, "z": z},
        "velocity": {"vx": 0.0, "vy": 0.0, "vz": 0.0},
        "ownership": "none",
        "speed": 0.0,
    }


def _goal(node_id, team, x):
    return {
        "node_type": "GOAL",
        "node_id": node_id,
        "team": team,
        "position": {"x": x, "y": 0.0},
        "width": 0.14,
        "height": 0.05,
        "depth": 0.04,
    }


def _teammate_edge(src, dst):
    return {"edge_type": "TEAMMATE", "source": src, "target": dst, "distance": 0.3}


def _fixture_graph(with_goals=True, with_possession=True):
    nodes = [
        _player("left_0", "left", -0.20, 0.10, is_controlled=True,
                has_possession=with_possession),
        _player("left_1", "left", 0.10, -0.30),
        _player("right_0", "right", 0.45, 0.05),
        _ball(BALL_X, BALL_Y),
    ]
    if with_goals:
        nodes.append(_goal("goal_left", "left", -1.0))
        nodes.append(_goal("goal_right", "right", 1.0))
    edges = [_teammate_edge("left_0", "left_1")]
    return {"scenario": {"id": "unit_test"}, "nodes": nodes, "edges": edges}


def _ball_lofted_graph():
    """Same graph, but the ball only differs in height/speed.

    Ball x/y are what every player-relative dim is built from, so this variant
    leaves the whole player vector — including Fix A — bit identical. Only a
    graph-level path (Fix D) can carry this change to the agent embedding.
    """
    graph = _fixture_graph()
    for node in graph["nodes"]:
        if node["node_type"] == "BALL":
            node["position"]["z"] = 0.8
            node["velocity"]["vz"] = 25.0
            node["speed"] = 25.0
    return graph


def _row(graph, ident):
    """Return the encoded feature vector of one node by global_id/node_id."""
    gt = graph_to_tensors(graph)
    for idx, node in enumerate(graph["nodes"]):
        if (node.get("global_id") or node.get("node_id")) == ident:
            return gt.node_features[idx]
    raise KeyError(ident)


def _expected(dx, dy):
    """Hand-calc the three ball-relative dims from a raw delta."""
    return (
        dx / PITCH_LENGTH,
        dy / PITCH_WIDTH,
        math.hypot(dx, dy) / DISTANCE_NORM,
    )


# ---------------------------------------------------------------------------
# Fix A — known-answer geometry
# ---------------------------------------------------------------------------


class TestBallRelativeFeatures:
    def test_node_width_grew_to_39(self):
        assert NODE_FEATURE_DIM == 39
        gt = graph_to_tensors(_fixture_graph())
        assert gt.node_features.shape == (6, 39)  # 3 players + ball + 2 goals

    def test_carrier_ball_geometry_matches_hand_calculation(self):
        row = _row(_fixture_graph(), "left_0")
        rel_x, rel_y, dist = _expected(BALL_X - (-0.20), BALL_Y - 0.10)
        assert rel_x == pytest.approx(0.15, abs=1e-6)
        assert row[PLAYER_BALL_REL_X_DIM].item() == pytest.approx(rel_x, abs=1e-6)
        assert row[PLAYER_BALL_REL_Y_DIM].item() == pytest.approx(rel_y, abs=1e-6)
        assert row[PLAYER_BALL_DISTANCE_DIM].item() == pytest.approx(dist, abs=1e-6)

    def test_ball_on_the_left_gives_negative_relative_x(self):
        # right_0 sits at x=0.45 and the ball at x=0.10, so the ball is toward -x.
        row = _row(_fixture_graph(), "right_0")
        rel_x, rel_y, dist = _expected(BALL_X - 0.45, BALL_Y - 0.05)
        assert rel_x == pytest.approx(-0.175, abs=1e-6)
        assert row[PLAYER_BALL_REL_X_DIM].item() == pytest.approx(rel_x, abs=1e-6)
        assert row[PLAYER_BALL_REL_Y_DIM].item() == pytest.approx(rel_y, abs=1e-6)
        assert row[PLAYER_BALL_DISTANCE_DIM].item() == pytest.approx(dist, abs=1e-6)
        assert row[PLAYER_BALL_DISTANCE_DIM].item() > 0.0

    def test_attacking_goal_is_the_far_end_for_each_team(self):
        # left attacks +x (goal_right); right attacks -x (goal_left).
        carrier = _row(_fixture_graph(), "left_0")
        opponent = _row(_fixture_graph(), "right_0")
        assert carrier[PLAYER_GOAL_REL_X_DIM].item() == pytest.approx(
            (1.0 - (-0.20)) / PITCH_LENGTH, abs=1e-6
        )
        assert carrier[PLAYER_GOAL_REL_Y_DIM].item() == pytest.approx(
            (0.0 - 0.10) / PITCH_WIDTH, abs=1e-6
        )
        assert carrier[PLAYER_GOAL_REL_X_DIM].item() > 0.0
        assert opponent[PLAYER_GOAL_REL_X_DIM].item() == pytest.approx(
            (-1.0 - 0.45) / PITCH_LENGTH, abs=1e-6
        )
        assert opponent[PLAYER_GOAL_REL_X_DIM].item() < 0.0

    def test_missing_goal_nodes_fall_back_to_pitch_defaults(self):
        # With no GOAL nodes the encoder must still know that the left team
        # attacks the +x goal rather than silently emitting zeros.
        row = _row(_fixture_graph(with_goals=False), "left_0")
        assert row[PLAYER_GOAL_REL_X_DIM].item() == pytest.approx(0.6, abs=1e-6)

    def test_nearest_to_ball_flag_is_global_and_unique(self):
        graph = _fixture_graph()
        # Distances: left_0 0.3162, right_0 0.3808, left_1 0.5 -> left_0 wins.
        assert _row(graph, "left_0")[PLAYER_IS_NEAREST_TO_BALL_DIM].item() == 1.0
        assert _row(graph, "right_0")[PLAYER_IS_NEAREST_TO_BALL_DIM].item() == 0.0
        assert _row(graph, "left_1")[PLAYER_IS_NEAREST_TO_BALL_DIM].item() == 0.0

    def test_nearest_to_ball_moves_with_the_ball(self):
        graph = _fixture_graph()
        for node in graph["nodes"]:
            if node["node_type"] == "BALL":
                node["position"].update({"x": 0.46, "y": 0.05})
        assert _row(graph, "right_0")[PLAYER_IS_NEAREST_TO_BALL_DIM].item() == 1.0
        assert _row(graph, "left_0")[PLAYER_IS_NEAREST_TO_BALL_DIM].item() == 0.0

    def test_has_possession_dim_is_the_ground_truth_flag(self):
        assert _row(_fixture_graph(), "left_0")[PLAYER_HAS_POSSESSION_DIM].item() == 1.0
        assert _row(_fixture_graph(), "right_0")[PLAYER_HAS_POSSESSION_DIM].item() == 0.0
        # An absent flag means "not the owner", never a distance guess.
        absent = _row(_fixture_graph(with_possession=False), "left_0")
        assert absent[PLAYER_HAS_POSSESSION_DIM].item() == 0.0

    def test_sentinel_player_gets_no_ball_geometry(self):
        graph = _fixture_graph()
        graph["nodes"].append(
            _player("left_2", "left", -1.0, -1.0, has_possession=True)
        )
        row = _row(graph, "left_2")
        assert row[29].item() == 1.0  # sentinel flag
        for dim in range(PLAYER_BALL_REL_X_DIM, PLAYER_HAS_POSSESSION_DIM + 1):
            assert row[dim].item() == 0.0

    def test_reserved_dims_30_and_31_stay_zero(self):
        # inject_poss in onball_rootcause_analysis.py writes dim 30 as an
        # out-of-distribution positive control; it must not be a real feature.
        gt = graph_to_tensors(_fixture_graph())
        assert torch.count_nonzero(gt.node_features[:, 30:32]).item() == 0

    def test_ball_row_does_not_carry_player_dims(self):
        gt = graph_to_tensors(_fixture_graph())
        assert torch.count_nonzero(gt.node_features[3, PLAYER_BALL_REL_X_DIM:]).item() == 0


# ---------------------------------------------------------------------------
# Fix D — pooled global embedding fused into the agent embedding
# ---------------------------------------------------------------------------


class TestGlobalFusion:
    @pytest.mark.parametrize("encoder_type", ["mlp", "gat", "geometry"])
    def test_agent_head_is_widened_by_the_global_embedding(self, encoder_type):
        encoder = create_encoder(encoder_type, hidden_dim=32, output_dim=16)
        assert encoder.agent_head.in_features == 32 * 2
        assert encoder.agent_head.out_features == 16

    @pytest.mark.parametrize("encoder_type", ["mlp", "gat", "geometry"])
    def test_agent_embedding_moves_when_only_the_ball_moves(self, encoder_type):
        """The BALL node has no outgoing edge, yet must still reach the agent.

        Before Fix D the agent row was a function of its own and its teammates'
        features only, so destroying the ball left the actor embedding bit
        identical (the root-cause report measured delta exactly 0.0).
        """
        encoder = create_encoder(encoder_type, hidden_dim=32, output_dim=16)
        encoder.eval()

        moved = _fixture_graph()
        for node in moved["nodes"]:
            if node["node_type"] == "BALL":
                node["position"].update({"x": -0.75, "y": -0.35})

        with torch.no_grad():
            base = encoder(graph_to_tensors(_fixture_graph()))[0]
            shifted = encoder(graph_to_tensors(moved))[0]

        assert base.shape == shifted.shape == (1, 16)
        delta = float((base - shifted).abs().max())
        assert math.isfinite(delta)
        assert delta > 1e-6, "agent embedding is still ball-blind"

    def test_pre_fix_path_was_exactly_ball_blind(self):
        """Isolates Fix D from Fix A.

        ``_ball_lofted_graph`` changes only ball z/speed, which no player dim
        reads. The legacy row — agent_head applied to the agent hidden state
        alone, reusing the first ``hidden_dim`` columns of the current weight —
        is therefore provably identical between the two graphs: before Fix D the
        ball could not reach the actor at all. The fused path must differ.
        """
        encoder = create_encoder("gat", hidden_dim=32, output_dim=16)
        encoder.eval()

        def legacy_row(graph):
            gt = graph_to_tensors(graph)
            with torch.no_grad():
                h = encoder.input_proj(gt.node_features)
                for layer in encoder.layers:
                    h = layer(h, gt.edge_index, gt.edge_features)
                    h = h * gt.node_mask.unsqueeze(-1)
            weight = encoder.agent_head.weight[:, : encoder.hidden_dim]
            return torch.nn.functional.linear(h[gt.agent_node_indices], weight,
                                              encoder.agent_head.bias)

        base, lofted = _fixture_graph(), _ball_lofted_graph()
        assert torch.equal(legacy_row(base), legacy_row(lofted)), (
            "the reconstructed pre-Fix-D row should be exactly ball-blind"
        )

        with torch.no_grad():
            fused_base = encoder(graph_to_tensors(base))[0]
            fused_lofted = encoder(graph_to_tensors(lofted))[0]
        assert not torch.equal(fused_base, fused_lofted), (
            "Fix D did not give the agent embedding a graph-level ball channel"
        )

    def test_agent_embedding_is_deterministic(self):
        encoder = create_encoder("gat", hidden_dim=32, output_dim=16)
        encoder.eval()
        with torch.no_grad():
            first = encoder(graph_to_tensors(_fixture_graph()))[0]
            second = encoder(graph_to_tensors(_fixture_graph()))[0]
        assert torch.equal(first, second)

    def test_global_embedding_still_shaped_like_output_dim(self):
        encoder = create_encoder("gat", hidden_dim=32, output_dim=16)
        encoder.eval()
        with torch.no_grad():
            agent_emb, global_emb = encoder(graph_to_tensors(_fixture_graph()))
        assert agent_emb.shape == (1, 16)
        assert global_emb.shape == (16,)


# ---------------------------------------------------------------------------
# Possession plumbing: engine ground truth -> graph node -> dim 38
#
# Uses a hand-built 127-dim observation (no bridge required) so the whole
# build_graph -> graph_to_tensors path is exercised offline.
# ---------------------------------------------------------------------------

SCENARIO = "academy_3_vs_1_with_keeper"


def _synthetic_observation():
    obs = np.zeros(127, dtype=np.float32)
    left = [(-0.20, 0.10), (0.10, -0.30), (-0.55, 0.25)]
    right = [(0.45, 0.05), (0.85, 0.0)]
    for i in range(11):
        lx, ly = left[i] if i < len(left) else (-1.0, -1.0)
        rx, ry = right[i] if i < len(right) else (-1.0, -1.0)
        obs[2 * i], obs[2 * i + 1] = lx, ly
        obs[44 + 2 * i], obs[44 + 2 * i + 1] = rx, ry
    obs[88], obs[89], obs[90] = BALL_X, BALL_Y, 0.0
    obs[95] = 1.0  # ball ownership one-hot: left
    obs[97] = 1.0  # active roster slot 0
    obs[115 + ROLE_VOCABULARY.index("CAM")] = 1.0  # viewpoint role (a left role)
    return {"observation": obs, "action_mask": []}


def _info(owner_agent_id):
    ground_truth = {}
    if owner_agent_id is not None:
        ground_truth["current_ball_owner"] = {
            "agent_id": owner_agent_id,
            "team": "left" if owner_agent_id.startswith("left") else "right",
        }
    return {"ground_truth": ground_truth, "controlledPlayerId": "left_0"}


class TestPossessionGroundTruth:
    def test_engine_owner_is_stamped_on_the_matching_player_node(self):
        graph = build_graph(_synthetic_observation(), _info("left_0"), SCENARIO)
        flags = {
            node["global_id"]: node.get("has_possession", False)
            for node in graph["nodes"]
            if node.get("node_type") == "PLAYER"
        }
        assert flags["left_0"] is True
        assert sum(1 for value in flags.values() if value) == 1

    def test_owner_also_lights_up_the_possession_edges(self):
        graph = build_graph(_synthetic_observation(), _info("left_0"), SCENARIO)
        possesses = [e for e in graph["edges"] if e["edge_type"] == "POSSESSES"]
        assert [(e["source"], e["target"]) for e in possesses] == [("left_0", "ball")]
        carrier_ball_edges = [
            e for e in graph["edges"]
            if e["edge_type"] == "PLAYER_BALL" and e["source"] == "left_0"
        ]
        assert carrier_ball_edges[0]["has_possession"] is True

    def test_dim_38_reads_the_ground_truth_owner(self):
        graph = build_graph(_synthetic_observation(), _info("left_2"), SCENARIO)
        gt = graph_to_tensors(graph)
        carrier = gt.node_features[2]
        assert carrier[PLAYER_HAS_POSSESSION_DIM].item() == 1.0
        others = torch.cat(
            [
                gt.node_features[idx : idx + 1]
                for idx, node in enumerate(graph["nodes"])
                if node.get("node_type") == "PLAYER" and idx != 2
            ]
        )
        assert torch.count_nonzero(others[:, PLAYER_HAS_POSSESSION_DIM]).item() == 0

    def test_unknown_owner_id_is_dropped_instead_of_dangling(self):
        # right_9 is not a player in this scenario: POSSESSES would dangle and
        # graph_to_tensors would raise, so the id must be discarded.
        graph = build_graph(_synthetic_observation(), _info("right_9"), SCENARIO)
        assert not [e for e in graph["edges"] if e["edge_type"] == "POSSESSES"]
        gt = graph_to_tensors(graph)
        assert torch.count_nonzero(gt.node_features[:, PLAYER_HAS_POSSESSION_DIM]).item() == 0

    def test_no_ground_truth_means_no_possession_anywhere(self):
        graph = build_graph(_synthetic_observation(), _info(None), SCENARIO)
        assert not [e for e in graph["edges"] if e["edge_type"] == "POSSESSES"]
        gt = graph_to_tensors(graph)
        assert torch.count_nonzero(gt.node_features[:, PLAYER_HAS_POSSESSION_DIM]).item() == 0
        # Geometry is still populated — possession and ball location are separate.
        assert torch.count_nonzero(gt.node_features[:, PLAYER_BALL_DISTANCE_DIM]).item() > 0

    def test_carrier_ball_geometry_survives_the_real_builder(self):
        graph = build_graph(_synthetic_observation(), _info("left_0"), SCENARIO)
        carrier = graph_to_tensors(graph).node_features[0]
        rel_x, rel_y, dist = _expected(BALL_X - (-0.20), BALL_Y - 0.10)
        assert carrier[PLAYER_BALL_REL_X_DIM].item() == pytest.approx(rel_x, abs=1e-6)
        assert carrier[PLAYER_BALL_REL_Y_DIM].item() == pytest.approx(rel_y, abs=1e-6)
        assert carrier[PLAYER_BALL_DISTANCE_DIM].item() == pytest.approx(dist, abs=1e-6)
        assert carrier[PLAYER_IS_NEAREST_TO_BALL_DIM].item() == 1.0
