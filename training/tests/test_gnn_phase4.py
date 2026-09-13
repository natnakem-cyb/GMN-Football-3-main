"""
GMN-Football-3 — Phase 4 Unit Tests

Tests for:
- graph_to_tensor adapter
- GNN encoders (MLP, GAT, geometry-aware)
- Diagnostic probes
- Integration: Phase 3 graph -> tensor -> encoder -> embeddings
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from training.gnn_graph_builder import build_graph, SCENARIOS
from training.gnn_graph_to_tensor import (
    graph_to_tensors,
    GraphTensor,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    Z_SCENARIO_DIM,
    NODE_TYPE_TO_INDEX,
    EDGE_TYPE_TO_INDEX,
)
from training.gnn_encoders import (
    MLPBaselineEncoder,
    GATEncoder,
    GeometryAwareEncoder,
    create_encoder,
)
from training.gnn_diagnostics import (
    SpatialGeometryProbe,
    DirectionProbe,
    PressureProbe,
    FormationProbe,
    GoalGeometryProbe,
    PassingProbe,
    ScenarioProbe,
    create_probe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env_and_graph(scenario_id: str = "academy_3_vs_1_defender_3"):
    """Create a live env and return (obs, info, graph)."""
    from training.gmn_pettingzoo import GMNMultiAgentEnv
    env = GMNMultiAgentEnv(scenario=scenario_id)
    obs, info = env.reset(seed=42)
    first_agent = list(obs.keys())[0]
    graph = build_graph(obs[first_agent], info[first_agent], scenario_id)
    env.close()
    return obs, info, graph


def _make_synthetic_graph(num_players: int = 4, num_edges: int = 6) -> dict:
    """Create a minimal synthetic graph for fast unit tests."""
    nodes = []
    edges = []

    for i in range(num_players):
        nodes.append({
            "node_type": "PLAYER",
            "global_id": f"left_{i}",
            "team": "left",
            "team_index": i,
            "position": {"x": -0.5 + i * 0.2, "y": 0.0},
            "velocity": {"vx": 0.0, "vy": 0.0},
            "role": ["GK", "CB", "CM", "ST"][i % 4],
            "role_one_hot": [0.0] * 12,
            "is_active": i == 0,
            "is_controlled": i == 0,
            "is_goalkeeper": i == 0,
            "features": {
                "nearest_teammate_dist": 0.5,
                "nearest_opponent_dist": 1.0,
                "team_width": 0.3,
                "team_depth": 0.6,
                "compactness": 0.4,
                "stretch": 0.6,
                "receiver_availability": 1.0,
            },
            "line_id": i % 2,
            "lane_id": i % 3,
        })

    nodes.append({
        "node_type": "BALL",
        "node_id": "ball",
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "velocity": {"vx": 0.0, "vy": 0.0, "vz": 0.0},
        "ownership": "left",
        "speed": 0.0,
    })

    for i in range(min(num_edges, num_players * (num_players - 1) // 2)):
        src = i % num_players
        tgt = (i + 1) % num_players
        if src != tgt:
            edges.append({
                "edge_type": "TEAMMATE",
                "source": f"left_{src}",
                "target": f"left_{tgt}",
                "distance": 0.5,
                "relative_x": 0.2,
                "relative_y": 0.0,
                "relative_vx": 0.0,
                "relative_vy": 0.0,
                "angle": 0.0,
                "closing_speed": 0.0,
            })

    return {
        "scenario": {
            "id": "academy_3_vs_1_defender_3",
            "node_type": "SCENARIO",
            "node_id": "scenario",
            "time_limit_seconds": 30,
            "terminate_on_opponent_possession": True,
            "objectives": [],
            "objective_vocabulary": [0] * 10,
            "rewards": {"scoring": 1.0, "completion": 100},
            "reward_scoring": 1.0,
            "reward_completion": 100,
        },
        "nodes": nodes,
        "edges": edges,
        "z_scenario": [0.1] * 8,
    }


# ---------------------------------------------------------------------------
# Graph-to-Tensor Tests
# ---------------------------------------------------------------------------

class TestGraphToTensor:
    """Tests for graph_to_tensors adapter."""

    def test_synthetic_graph_converts(self):
        graph = _make_synthetic_graph()
        gt = graph_to_tensors(graph)
        assert gt.node_features.shape == (5, NODE_FEATURE_DIM)
        assert gt.node_type.shape == (5,)
        assert gt.edge_index.shape[0] == 2
        assert gt.edge_features.shape[1] == EDGE_FEATURE_DIM
        assert gt.node_mask.shape == (5,)
        assert len(gt.agent_node_indices) == 1
        assert gt.graph_context is not None
        assert gt.graph_context.shape == (Z_SCENARIO_DIM,)

    def test_live_graph_converts(self):
        obs, info, graph = _make_env_and_graph("academy_3_vs_1_defender_3")
        gt = graph_to_tensors(graph)
        assert gt.node_features.shape[1] == NODE_FEATURE_DIM
        assert gt.edge_features.shape[1] == EDGE_FEATURE_DIM
        assert gt.node_features.shape[0] == len(graph["nodes"])
        # SCENARIO_CONTEXT edges are skipped because scenario node is graph["scenario"], not in nodes
        assert gt.edge_index.shape[1] <= len(graph["edges"])

    def test_node_type_encoding(self):
        graph = _make_synthetic_graph()
        gt = graph_to_tensors(graph)
        for i, node in enumerate(graph["nodes"]):
            expected_type = NODE_TYPE_TO_INDEX.get(node["node_type"], 0)
            assert gt.node_type[i].item() == expected_type

    def test_edge_type_encoding(self):
        graph = _make_synthetic_graph()
        gt = graph_to_tensors(graph)
        for i, edge in enumerate(graph["edges"]):
            expected_type = EDGE_TYPE_TO_INDEX.get(edge["edge_type"], 0)
            assert gt.edge_type[i].item() == expected_type

    def test_agent_node_indices(self):
        graph = _make_synthetic_graph()
        gt = graph_to_tensors(graph)
        assert len(gt.agent_node_indices) == 1
        assert gt.agent_node_indices[0] == 0  # left_0 is controlled

    def test_z_scenario_attached(self):
        graph = _make_synthetic_graph()
        gt = graph_to_tensors(graph)
        assert gt.graph_context is not None
        assert gt.graph_context.shape == (Z_SCENARIO_DIM,)
        assert torch.allclose(gt.graph_context, torch.tensor([0.1] * 8, dtype=torch.float32))

    def test_no_z_scenario_when_absent(self):
        graph = _make_synthetic_graph()
        del graph["z_scenario"]
        gt = graph_to_tensors(graph)
        assert gt.graph_context is None

    def test_empty_graph(self):
        graph = {"scenario": {"id": "test"}, "nodes": [], "edges": []}
        gt = graph_to_tensors(graph)
        assert gt.node_features.shape == (0, NODE_FEATURE_DIM)
        assert gt.edge_index.shape == (2, 0)
        assert gt.agent_node_indices == []

    def test_node_id_mapping(self):
        graph = _make_synthetic_graph()
        gt = graph_to_tensors(graph)
        assert "left_0" in gt.node_id_to_index
        assert "ball" in gt.node_id_to_index
        assert gt.node_id_to_index["left_0"] == 0

    def test_deterministic_conversion(self):
        graph = _make_synthetic_graph()
        gt1 = graph_to_tensors(graph)
        gt2 = graph_to_tensors(graph)
        assert torch.allclose(gt1.node_features, gt2.node_features)
        assert torch.allclose(gt1.edge_index, gt2.edge_index)
        assert torch.allclose(gt1.edge_features, gt2.edge_features)

    def test_to_device(self):
        graph = _make_synthetic_graph()
        gt = graph_to_tensors(graph)
        device = torch.device("cpu")
        gt_cpu = gt.to(device)
        assert gt_cpu.node_features.device == device

    def test_scenario_id_preserved(self):
        graph = _make_synthetic_graph()
        gt = graph_to_tensors(graph)
        assert gt.scenario_id == "academy_3_vs_1_defender_3"


# ---------------------------------------------------------------------------
# Encoder Tests
# ---------------------------------------------------------------------------

class TestEncoders:
    """Tests for GNN encoders."""

    @pytest.fixture
    def sample_graph_tensor(self):
        graph = _make_synthetic_graph(num_players=4, num_edges=4)
        return graph_to_tensors(graph)

    def test_mlp_baseline_forward(self, sample_graph_tensor):
        encoder = MLPBaselineEncoder()
        encoder.eval()
        with torch.no_grad():
            agent_emb, global_emb = encoder(sample_graph_tensor)
        assert agent_emb.shape[-1] == 128
        assert global_emb.shape[-1] == 128
        assert agent_emb.shape[0] == len(sample_graph_tensor.agent_node_indices)

    def test_gat_encoder_forward(self, sample_graph_tensor):
        encoder = GATEncoder(num_layers=2, num_heads=4)
        encoder.eval()
        with torch.no_grad():
            agent_emb, global_emb = encoder(sample_graph_tensor)
        assert agent_emb.shape[-1] == 128
        assert global_emb.shape[-1] == 128

    def test_geometry_encoder_forward(self, sample_graph_tensor):
        encoder = GeometryAwareEncoder(num_layers=2)
        encoder.eval()
        with torch.no_grad():
            agent_emb, global_emb = encoder(sample_graph_tensor)
        assert agent_emb.shape[-1] == 128
        assert global_emb.shape[-1] == 128

    def test_encoder_deterministic(self, sample_graph_tensor):
        encoder = GATEncoder(num_layers=2)
        encoder.eval()
        with torch.no_grad():
            agent1, global1 = encoder(sample_graph_tensor)
            agent2, global2 = encoder(sample_graph_tensor)
        assert torch.allclose(agent1, agent2)
        assert torch.allclose(global1, global2)

    def test_encoder_without_z_scenario(self, sample_graph_tensor):
        sample_graph_tensor.graph_context = None
        encoder = MLPBaselineEncoder()
        encoder.eval()
        with torch.no_grad():
            agent_emb, global_emb = encoder(sample_graph_tensor)
        assert agent_emb.shape[-1] == 128
        assert global_emb.shape[-1] == 128

    def test_create_encoder_factory(self):
        for enc_type in ["mlp", "gat", "geometry"]:
            encoder = create_encoder(enc_type)
            assert isinstance(encoder, nn.Module)

    def test_encoder_output_dim_match(self):
        encoder = GATEncoder(output_dim=64)
        graph = _make_synthetic_graph()
        gt = graph_to_tensors(graph)
        encoder.eval()
        with torch.no_grad():
            agent_emb, global_emb = encoder(gt)
        assert agent_emb.shape[-1] == 64
        assert global_emb.shape[-1] == 64


# ---------------------------------------------------------------------------
# Probe Tests
# ---------------------------------------------------------------------------

class TestProbes:
    """Tests for diagnostic probes."""

    def test_spatial_probe_forward(self):
        probe = SpatialGeometryProbe(input_dim=128, num_targets=3)
        x = torch.randn(4, 128)
        out = probe(x)
        assert out.shape == (4, 3)

    def test_direction_probe_forward(self):
        probe = DirectionProbe(input_dim=128, num_targets=2)
        x = torch.randn(4, 128)
        out = probe(x)
        assert out.shape == (4, 2)

    def test_pressure_probe_forward(self):
        probe = PressureProbe(input_dim=128)
        x = torch.randn(4, 128)
        out = probe(x)
        assert out.shape == (4, 1)

    def test_formation_probe_forward(self):
        probe = FormationProbe(input_dim=128, num_roles=12, num_lines=4, num_lanes=3)
        x = torch.randn(4, 128)
        out = probe(x)
        assert out.shape == (4, 1 + 12 + 4 + 3)

    def test_goal_geometry_probe_forward(self):
        probe = GoalGeometryProbe(input_dim=128)
        x = torch.randn(4, 128)
        out = probe(x)
        assert out.shape == (4, 2)

    def test_passing_probe_forward(self):
        probe = PassingProbe(input_dim=128)
        x = torch.randn(4, 128)
        out = probe(x)
        assert out.shape == (4, 4)

    def test_scenario_probe_forward(self):
        probe = ScenarioProbe(input_dim=128, num_scenarios=10)
        x = torch.randn(4, 128)
        out = probe(x)
        assert out.shape == (4, 10)

    def test_probe_factory(self):
        for probe_type in ["spatial", "direction", "pressure", "formation", "goal_geometry", "passing"]:
            probe = create_probe(probe_type, input_dim=64)
            assert isinstance(probe, nn.Module)
        # ScenarioProbe requires num_scenarios
        probe = create_probe("scenario", input_dim=64, num_scenarios=10)
        assert isinstance(probe, nn.Module)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestPhase4Integration:
    """End-to-end integration: Phase 3 graph -> tensor -> encoder -> embeddings."""

    def test_full_pipeline_3v1(self):
        obs, info, graph = _make_env_and_graph("academy_3_vs_1_defender_3")
        gt = graph_to_tensors(graph)
        assert gt.node_features.shape[0] > 0
        assert gt.edge_index.shape[1] > 0

        for enc_type in ["mlp", "gat", "geometry"]:
            encoder = create_encoder(enc_type)
            encoder.eval()
            with torch.no_grad():
                agent_emb, global_emb = encoder(gt)
            assert agent_emb.shape[-1] == 128
            assert global_emb.shape[-1] == 128
            assert agent_emb.shape[0] >= 1

    def test_full_pipeline_rondo(self):
        obs, info, graph = _make_env_and_graph("academy_rondo_4v1")
        gt = graph_to_tensors(graph)
        assert gt.node_features.shape[0] > 0

        encoder = MLPBaselineEncoder()
        encoder.eval()
        with torch.no_grad():
            agent_emb, global_emb = encoder(gt)
        assert agent_emb.shape[-1] == 128

    def test_full_pipeline_11v11(self):
        obs, info, graph = _make_env_and_graph("11_vs_11")
        gt = graph_to_tensors(graph)
        assert gt.node_features.shape[0] > 0

        encoder = GATEncoder(num_layers=2)
        encoder.eval()
        with torch.no_grad():
            agent_emb, global_emb = encoder(gt)
        assert agent_emb.shape[-1] == 128

    def test_z_scenario_affects_output(self):
        obs, info, graph = _make_env_and_graph("academy_3_vs_1_defender_3")
        gt1 = graph_to_tensors(graph)
        gt2 = graph_to_tensors(graph)
        # Ensure z_scenario exists before perturbing
        if gt2.graph_context is None:
            gt2.graph_context = torch.zeros(Z_SCENARIO_DIM, dtype=torch.float32)
        # Perturb zScenario
        gt2.graph_context = gt2.graph_context + 0.1

        encoder = MLPBaselineEncoder()
        encoder.eval()
        with torch.no_grad():
            _, global1 = encoder(gt1)
            _, global2 = encoder(gt2)
        assert not torch.allclose(global1, global2), "zScenario perturbation should change global embedding"

    def test_deterministic_end_to_end(self):
        obs, info, graph = _make_env_and_graph("academy_3_vs_1_defender_3")
        gt1 = graph_to_tensors(graph)
        gt2 = graph_to_tensors(graph)

        encoder = GATEncoder(num_layers=2)
        encoder.eval()
        with torch.no_grad():
            a1, g1 = encoder(gt1)
            a2, g2 = encoder(gt2)
        assert torch.allclose(a1, a2)
        assert torch.allclose(g1, g2)

    def test_no_edges_graph(self):
        graph = {
            "scenario": {"id": "test"},
            "nodes": [{
                "node_type": "PLAYER",
                "global_id": "left_0",
                "team": "left",
                "team_index": 0,
                "position": {"x": 0.0, "y": 0.0},
                "velocity": {"vx": 0.0, "vy": 0.0},
                "role": "ST",
                "role_one_hot": [0.0] * 12,
                "is_active": True,
                "is_controlled": True,
                "is_goalkeeper": False,
                "features": {k: 0.0 for k in ["nearest_teammate_dist", "nearest_opponent_dist", "team_width", "team_depth", "compactness", "stretch", "receiver_availability"]},
                "line_id": 0,
                "lane_id": 0,
            }],
            "edges": [],
        }
        gt = graph_to_tensors(graph)
        encoder = GATEncoder(num_layers=2)
        encoder.eval()
        with torch.no_grad():
            agent_emb, global_emb = encoder(gt)
        assert agent_emb.shape[-1] == 128
        assert global_emb.shape[-1] == 128
