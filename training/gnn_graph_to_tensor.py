"""
GMN-Football-3 — Graph-to-Tensor Adapter (Phase 4)

Converts a Phase 3 graph dict into PyTorch tensors suitable for GNN consumption.

Input: graph dict from training.gnn_graph_builder.build_graph()
Output: GraphTensor dataclass with node_features, edge_index, edge_features, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from training.gnn_graph_builder import SCENARIOS, ROLE_VOCABULARY, LINE_LANE_GAP_THRESHOLD

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_TYPE_TO_INDEX = {
    "PLAYER": 0,
    "BALL": 1,
    "GOAL": 2,
    "FORMATION_SLOT": 3,
    "TEAM_SHAPE": 4,
}

EDGE_TYPE_TO_INDEX = {
    "TEAMMATE": 0,
    "OPPONENT": 1,
    "NEAR": 2,
    "POSSESSES": 3,
    "PLAYER_BALL": 4,
    "PLAYER_GOAL": 5,
    "BALL_GOAL": 6,
    "ASSIGNED_TO": 7,
    "BELONGS_TO_SHAPE": 8,
    "SCENARIO_CONTEXT": 9,
    "FORMATION_ADJACENCY": 10,
    "FORMATION_LINE": 11,
    "FORMATION_LANE": 12,
}

# Continuous node feature indices in the packed node_features tensor
NODE_FEATURE_DIM = 32

# Edge feature dimension (max over all edge types)
EDGE_FEATURE_DIM = 10

# zScenario dimension
Z_SCENARIO_DIM = 8


# ---------------------------------------------------------------------------
# GraphTensor
# ---------------------------------------------------------------------------

@dataclass
class GraphTensor:
    """Tensorized graph ready for GNN consumption."""
    node_features: torch.Tensor          # (num_nodes, NODE_FEATURE_DIM)
    node_type: torch.Tensor              # (num_nodes,) int64
    edge_index: torch.Tensor             # (2, num_edges) int64
    edge_type: torch.Tensor              # (num_edges,) int64
    edge_features: torch.Tensor          # (num_edges, EDGE_FEATURE_DIM)
    node_mask: torch.Tensor              # (num_nodes,) float32
    agent_node_indices: List[int]        # indices of controllable player nodes
    graph_context: Optional[torch.Tensor]  # (z_dim,) or None
    node_id_to_index: Dict[str, int]     # mapping from node_id/global_id to node index
    scenario_id: str                     # scenario identifier

    def to(self, device: torch.device) -> "GraphTensor":
        return GraphTensor(
            node_features=self.node_features.to(device),
            node_type=self.node_type.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            edge_features=self.edge_features.to(device),
            node_mask=self.node_mask.to(device),
            agent_node_indices=self.agent_node_indices,
            graph_context=self.graph_context.to(device) if self.graph_context is not None else None,
            node_id_to_index=self.node_id_to_index,
            scenario_id=self.scenario_id,
        )

    def num_nodes(self) -> int:
        return int(self.node_mask.sum().item())

    def num_edges(self) -> int:
        return self.edge_index.shape[1]


# ---------------------------------------------------------------------------
# Node feature extraction
# ---------------------------------------------------------------------------

def _encode_player_node(node: Dict[str, Any], idx: int, all_nodes: List[Dict[str, Any]]) -> torch.Tensor:
    features = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float32)

    # Position (indices 0-1): already normalized [-1,1] x [-0.42,0.42]
    features[0] = float(node["position"]["x"])
    features[1] = float(node["position"]["y"])

    # Velocity (indices 2-3): scale down by 50
    features[2] = float(node["velocity"]["vx"]) / 50.0
    features[3] = float(node["velocity"]["vy"]) / 50.0

    # Role one-hot (indices 4-15): 12-dim
    role_one_hot = node.get("role_one_hot", [0.0] * 12)
    features[4:16] = torch.tensor(role_one_hot, dtype=torch.float32)

    # is_active (index 16), is_controlled (17), is_goalkeeper (18)
    features[16] = 1.0 if node.get("is_active", False) else 0.0
    features[17] = 1.0 if node.get("is_controlled", False) else 0.0
    features[18] = 1.0 if node.get("is_goalkeeper", False) else 0.0

    # Derived team features (indices 19-25)
    feat = node.get("features", {})
    features[19] = float(feat.get("nearest_teammate_dist", 0.0)) / 1.414  # normalize by max diagonal
    features[20] = float(feat.get("nearest_opponent_dist", 0.0)) / 1.414
    features[21] = float(feat.get("team_width", 0.0)) / 0.84  # normalize by pitch width
    features[22] = float(feat.get("team_depth", 0.0)) / 2.0   # normalize by pitch length
    features[23] = float(feat.get("compactness", 0.0)) / 1.414
    features[24] = float(feat.get("stretch", 0.0)) / 2.0
    features[25] = float(feat.get("receiver_availability", 0.0))

    # line_id, lane_id (indices 26-27)
    features[26] = float(node.get("line_id", 0))
    features[27] = float(node.get("lane_id", 0))

    # team_index (index 28)
    features[28] = float(node.get("team_index", 0))

    # sentinel flag (index 29): 1.0 if position is (-1.0, -1.0)
    x = float(node["position"]["x"])
    y = float(node["position"]["y"])
    features[29] = 1.0 if (x == -1.0 and y == -1.0) else 0.0

    return features


def _encode_ball_node(node: Dict[str, Any]) -> torch.Tensor:
    features = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float32)

    # Position (indices 0-2): x, y, z
    features[0] = float(node["position"]["x"])
    features[1] = float(node["position"]["y"])
    features[2] = float(node["position"]["z"])

    # Velocity (indices 3-5): scale down by 50
    features[3] = float(node["velocity"]["vx"]) / 50.0
    features[4] = float(node["velocity"]["vy"]) / 50.0
    features[5] = float(node["velocity"]["vz"]) / 50.0

    # Speed (index 6)
    features[6] = float(node.get("speed", 0.0)) / 50.0

    # Ownership one-hot (indices 7-9): none, left, right
    ownership = node.get("ownership", "none")
    if ownership == "none":
        features[7] = 1.0
    elif ownership == "left":
        features[8] = 1.0
    elif ownership == "right":
        features[9] = 1.0

    # Remaining indices unused (10-31) = 0.0
    return features


def _encode_goal_node(node: Dict[str, Any]) -> torch.Tensor:
    features = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float32)

    # Position (indices 0-1)
    features[0] = float(node["position"]["x"])
    features[1] = float(node["position"]["y"])

    # Fixed geometry (indices 2-4)
    features[2] = float(node.get("width", 0.14))
    features[3] = float(node.get("height", 0.05))
    features[4] = float(node.get("depth", 0.04))

    # Team one-hot (indices 5-6): left, right
    team = node.get("team", "left")
    if team == "left":
        features[5] = 1.0
    else:
        features[6] = 1.0

    # node_id encoded as goal_left=0, goal_right=1 (index 7)
    node_id = node.get("node_id", "goal_left")
    features[7] = 0.0 if node_id == "goal_left" else 1.0

    return features


def _encode_formation_slot_node(node: Dict[str, Any]) -> torch.Tensor:
    features = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float32)

    # Nominal position (indices 0-2): x, y, plus xRatio
    features[0] = float(node.get("x", 0.0))
    features[1] = float(node.get("y", 0.0))
    features[2] = float(node.get("xRatio", 0.0))
    features[3] = float(node.get("yRatio", 0.0))

    # Role one-hot (indices 4-15): 12-dim
    role = node.get("role", "UNKNOWN")
    role_one_hot = [0.0] * 12
    if role in ROLE_VOCABULARY:
        role_one_hot[ROLE_VOCABULARY.index(role)] = 1.0
    features[4:16] = torch.tensor(role_one_hot, dtype=torch.float32)

    # team one-hot (indices 16-17): left, right
    team = node.get("team", "left")
    if team == "left":
        features[16] = 1.0
    else:
        features[17] = 1.0

    return features


def _encode_team_shape_node(node: Dict[str, Any]) -> torch.Tensor:
    features = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float32)

    # Formation deviation stats (indices 0-3)
    fd = node.get("formation_deviation", {})
    features[0] = float(fd.get("inter_line_spacing_variance", 0.0))
    features[1] = float(fd.get("mean_line_compactness", 0.0))
    features[2] = float(fd.get("num_lines", 1))
    features[3] = float(fd.get("num_lanes", 1))

    # team one-hot (indices 4-5): left, right
    team = node.get("team", "left")
    if team == "left":
        features[4] = 1.0
    else:
        features[5] = 1.0

    return features


def _encode_node(node: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
    node_type = node.get("node_type", "PLAYER")
    node_type_idx = NODE_TYPE_TO_INDEX.get(node_type, 0)

    if node_type == "PLAYER":
        features = _encode_player_node(node, 0, [])
    elif node_type == "BALL":
        features = _encode_ball_node(node)
    elif node_type == "GOAL":
        features = _encode_goal_node(node)
    elif node_type == "FORMATION_SLOT":
        features = _encode_formation_slot_node(node)
    elif node_type == "TEAM_SHAPE":
        features = _encode_team_shape_node(node)
    else:
        features = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float32)

    return features, node_type_idx


# ---------------------------------------------------------------------------
# Edge feature extraction
# ---------------------------------------------------------------------------

def _encode_teammate_edge(edge: Dict[str, Any]) -> torch.Tensor:
    features = torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float32)
    features[0] = float(edge.get("distance", 0.0)) / 1.414
    features[1] = float(edge.get("relative_x", 0.0)) / 2.0
    features[2] = float(edge.get("relative_y", 0.0)) / 0.84
    features[3] = float(edge.get("relative_vx", 0.0)) / 50.0
    features[4] = float(edge.get("relative_vy", 0.0)) / 50.0
    features[5] = float(edge.get("angle", 0.0)) / 3.14159
    features[6] = float(edge.get("closing_speed", 0.0))
    return features


def _encode_opponent_edge(edge: Dict[str, Any]) -> torch.Tensor:
    features = _encode_teammate_edge(edge)
    features[7] = float(edge.get("pressure", 0.0))
    return features


def _encode_near_edge(edge: Dict[str, Any]) -> torch.Tensor:
    features = torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float32)
    features[0] = float(edge.get("distance", 0.0)) / 1.414
    features[1] = float(edge.get("threshold", 0.065))
    return features


def _encode_possesses_edge(edge: Dict[str, Any]) -> torch.Tensor:
    return torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float32)


def _encode_player_ball_edge(edge: Dict[str, Any]) -> torch.Tensor:
    features = torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float32)
    features[0] = float(edge.get("distance", 0.0)) / 1.414
    features[1] = float(edge.get("relative_x", 0.0)) / 2.0
    features[2] = float(edge.get("relative_y", 0.0)) / 0.84
    features[3] = float(edge.get("relative_vx", 0.0)) / 50.0
    features[4] = float(edge.get("relative_vy", 0.0)) / 50.0
    features[5] = float(edge.get("angle", 0.0)) / 3.14159
    features[6] = 1.0 if edge.get("has_possession", False) else 0.0
    return features


def _encode_player_goal_edge(edge: Dict[str, Any]) -> torch.Tensor:
    features = torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float32)
    features[0] = float(edge.get("distance", 0.0)) / 1.414
    features[1] = float(edge.get("relative_x", 0.0)) / 2.0
    features[2] = float(edge.get("relative_y", 0.0)) / 0.84
    features[3] = float(edge.get("angle", 0.0)) / 3.14159
    features[4] = float(edge.get("shot_angle", 0.0))
    features[5] = float(edge.get("nearest_opponent_pressure", 0.0)) / 1.414
    return features


def _encode_ball_goal_edge(edge: Dict[str, Any]) -> torch.Tensor:
    features = torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float32)
    features[0] = float(edge.get("distance", 0.0)) / 1.414
    features[1] = float(edge.get("relative_x", 0.0)) / 2.0
    features[2] = float(edge.get("relative_y", 0.0)) / 0.84
    features[3] = float(edge.get("angle", 0.0)) / 3.14159
    features[4] = float(edge.get("shot_angle", 0.0))
    return features


def _encode_assigned_to_edge(edge: Dict[str, Any]) -> torch.Tensor:
    features = torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float32)
    features[0] = float(edge.get("deviation_distance", 0.0)) / 1.414
    features[1] = float(edge.get("deviation_x", 0.0)) / 2.0
    features[2] = float(edge.get("deviation_y", 0.0)) / 0.84
    return features


def _encode_belongs_to_shape_edge(edge: Dict[str, Any]) -> torch.Tensor:
    features = torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float32)
    team = edge.get("team", "left")
    features[0] = 1.0 if team == "left" else 0.0
    features[1] = 1.0 if team == "right" else 0.0
    return features


def _encode_scenario_context_edge(edge: Dict[str, Any]) -> torch.Tensor:
    return torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float32)


def _encode_formation_adjacency_edge(edge: Dict[str, Any]) -> torch.Tensor:
    return torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float32)


def _encode_formation_line_edge(edge: Dict[str, Any]) -> torch.Tensor:
    features = torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float32)
    features[0] = float(edge.get("line_id", 0))
    return features


def _encode_formation_lane_edge(edge: Dict[str, Any]) -> torch.Tensor:
    features = torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float32)
    features[0] = float(edge.get("lane_id", 0))
    return features


_EDGE_ENCODERS = {
    "TEAMMATE": _encode_teammate_edge,
    "OPPONENT": _encode_opponent_edge,
    "NEAR": _encode_near_edge,
    "POSSESSES": _encode_possesses_edge,
    "PLAYER_BALL": _encode_player_ball_edge,
    "PLAYER_GOAL": _encode_player_goal_edge,
    "BALL_GOAL": _encode_ball_goal_edge,
    "ASSIGNED_TO": _encode_assigned_to_edge,
    "BELONGS_TO_SHAPE": _encode_belongs_to_shape_edge,
    "SCENARIO_CONTEXT": _encode_scenario_context_edge,
    "FORMATION_ADJACENCY": _encode_formation_adjacency_edge,
    "FORMATION_LINE": _encode_formation_line_edge,
    "FORMATION_LANE": _encode_formation_lane_edge,
}


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

def graph_to_tensors(graph: Dict[str, Any]) -> GraphTensor:
    """Convert a Phase 3 graph dict into PyTorch tensors.

    Args:
        graph: dict matching gnn_graph_schema.json v3, from build_graph()

    Returns:
        GraphTensor with all tensor fields populated
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    scenario_node = graph.get("scenario", {})
    z_scenario = graph.get("z_scenario")

    # Build node_id -> index mapping
    node_id_to_index: Dict[str, int] = {}
    for idx, node in enumerate(nodes):
        node_type = node.get("node_type", "PLAYER")
        if node_type == "PLAYER":
            node_id_to_index[node["global_id"]] = idx
        elif node_type == "BALL":
            node_id_to_index["ball"] = idx
        elif node_type == "GOAL":
            node_id_to_index[node["node_id"]] = idx
        elif node_type == "FORMATION_SLOT":
            node_id_to_index[node["slot_id"]] = idx
        elif node_type == "TEAM_SHAPE":
            node_id_to_index[f"team_shape_{node['team']}"] = idx

    # Encode nodes
    node_features_list = []
    node_type_list = []
    for node in nodes:
        feats, ntype = _encode_node(node)
        node_features_list.append(feats)
        node_type_list.append(ntype)

    node_features = torch.stack(node_features_list) if node_features_list else torch.zeros((0, NODE_FEATURE_DIM))
    node_type = torch.tensor(node_type_list, dtype=torch.long)
    node_mask = torch.ones(len(nodes), dtype=torch.float32)

    # Encode edges
    edge_index_list = []
    edge_type_list = []
    edge_features_list = []

    for edge in edges:
        edge_type_str = edge.get("edge_type", "TEAMMATE")
        edge_type_idx = EDGE_TYPE_TO_INDEX.get(edge_type_str, 0)

        source_id = edge.get("source", "")
        target_id = edge.get("target", "")

        # Skip SCENARIO_CONTEXT edges: scenario node is graph["scenario"], not in nodes list
        if edge_type_str == "SCENARIO_CONTEXT":
            continue

        source_idx = node_id_to_index.get(source_id)
        target_idx = node_id_to_index.get(target_id)

        if source_idx is None or target_idx is None:
            raise ValueError(
                f"Dangling edge reference in graph: {edge_type_str} {source_id} -> {target_id}"
            )

        edge_index_list.append([source_idx, target_idx])
        edge_type_list.append(edge_type_idx)

        encoder = _EDGE_ENCODERS.get(edge_type_str, _encode_teammate_edge)
        edge_features_list.append(encoder(edge))

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)
        edge_features = torch.stack(edge_features_list)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
        edge_features = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    # Identify agent nodes (controlled player)
    agent_node_indices = []
    for idx, node in enumerate(nodes):
        if node.get("node_type") == "PLAYER" and node.get("is_controlled", False):
            agent_node_indices.append(idx)

    # Graph context (zScenario)
    graph_context = None
    if z_scenario is not None:
        graph_context = torch.tensor(z_scenario, dtype=torch.float32)
        if graph_context.dim() == 0:
            graph_context = graph_context.unsqueeze(0)

    scenario_id = scenario_node.get("id", "unknown")

    return GraphTensor(
        node_features=node_features,
        node_type=node_type,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_features=edge_features,
        node_mask=node_mask,
        agent_node_indices=agent_node_indices,
        graph_context=graph_context,
        node_id_to_index=node_id_to_index,
        scenario_id=scenario_id,
    )
