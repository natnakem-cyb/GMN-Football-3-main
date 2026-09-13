"""
GMN-Football-3 — GNN Encoders (Phase 4)

Lightweight graph neural network encoders implemented in vanilla PyTorch.
No external graph libraries required.

Encoders:
- MLPBaseline: non-graph baseline, flattens and pools all node features
- GATEncoder: graph attention network with 2-3 message-passing layers
- GeometryAwareEncoder: EGNN-style with coordinate update (when spatial features present)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: masked mean pooling
# ---------------------------------------------------------------------------

def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pool over nodes, respecting mask."""
    mask = mask.unsqueeze(-1)  # (num_nodes, 1)
    masked_sum = (x * mask).sum(dim=0)
    mask_sum = mask.sum(dim=0).clamp(min=1.0)
    return masked_sum / mask_sum


# ---------------------------------------------------------------------------
# MLP Baseline Encoder
# ---------------------------------------------------------------------------

class MLPBaselineEncoder(nn.Module):
    """
    Non-graph baseline: flattens all node features and pools them.
    This establishes the representation quality floor without message passing.
    """

    def __init__(
        self,
        node_feat_dim: int = 32,
        z_dim: int = 8,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Node feature encoder
        layers = []
        in_dim = node_feat_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        self.node_encoder = nn.Sequential(*layers)

        # Graph context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Output heads
        self.agent_head = nn.Linear(hidden_dim, output_dim)
        self.global_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            graph_tensor: GraphTensor with node_features, node_mask, graph_context

        Returns:
            agent_embeddings: (num_agents, output_dim)
            global_embedding: (output_dim,)
        """
        x = graph_tensor.node_features  # (num_nodes, node_feat_dim)
        mask = graph_tensor.node_mask   # (num_nodes,)
        context = graph_tensor.graph_context  # (z_dim,) or None

        # Encode each node
        node_emb = self.node_encoder(x)  # (num_nodes, hidden_dim)

        # Global pooling
        global_emb = _masked_mean_pool(node_emb, mask)  # (hidden_dim,)

        # Fuse with graph context
        if context is not None:
            context = context.unsqueeze(0)  # (1, z_dim)
            global_emb = global_emb.unsqueeze(0)  # (1, hidden_dim)
            global_emb = self.context_fusion(torch.cat([global_emb, context], dim=-1)).squeeze(0)

        # Agent embeddings
        agent_indices = graph_tensor.agent_node_indices
        if agent_indices:
            agent_emb = node_emb[agent_indices]  # (num_agents, hidden_dim)
            agent_emb = self.agent_head(agent_emb)
        else:
            # No agents: use global embedding as fallback
            agent_emb = self.agent_head(global_emb.unsqueeze(0))

        global_emb = self.global_head(global_emb)

        return agent_emb, global_emb


# ---------------------------------------------------------------------------
# Graph Attention Encoder (GAT-like)
# ---------------------------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    """
    Single graph attention layer.
    Implements scaled dot-product attention over neighbors.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.key_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.value_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (num_nodes, node_feat_dim)
            edge_index: (2, num_edges)
            edge_features: (num_edges, edge_feat_dim)

        Returns:
            updated node features: (num_nodes, hidden_dim)
        """
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            # No edges: return zero-updated features with residual
            residual = node_features
            x = self.norm(self.output_proj(torch.zeros(num_nodes, self.hidden_dim, device=node_features.device)))
            return x + residual if residual.shape[-1] == x.shape[-1] else x

        # Project to Q, K, V
        Q = self.query_proj(node_features)  # (num_nodes, hidden_dim)
        K = self.key_proj(node_features)    # (num_nodes, hidden_dim)
        V = self.value_proj(node_features)  # (num_nodes, hidden_dim)
        E = self.edge_proj(edge_features)   # (num_edges, hidden_dim)

        # Reshape for multi-head attention
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)
        K = K.view(num_nodes, self.num_heads, self.head_dim)
        V = V.view(num_nodes, self.num_heads, self.head_dim)
        E = E.view(num_edges, self.num_heads, self.head_dim)

        # Gather source/target
        src = edge_index[0]  # (num_edges,)
        tgt = edge_index[1]  # (num_edges,)

        Q_tgt = Q[tgt]  # (num_edges, num_heads, head_dim)
        K_src = K[src]  # (num_edges, num_heads, head_dim)
        V_src = V[src]  # (num_edges, num_heads, head_dim)

        # Scaled dot-product attention with edge bias
        attn_logits = (Q_tgt * K_src).sum(dim=-1) * self.scale  # (num_edges, num_heads)
        attn_logits = attn_logits + E.sum(dim=-1) * self.scale   # add edge bias

        # Softmax over neighbors per target node
        attn_weights = torch.zeros(num_edges, self.num_heads, device=node_features.device)
        max_logits = torch.zeros(num_nodes, self.num_heads, device=node_features.device)
        max_logits.index_add_(0, tgt, attn_logits.clamp(max=0))
        attn_logits = attn_logits - max_logits[tgt]
        exp_weights = attn_logits.exp()

        # Group by target node
        sum_exp = torch.zeros(num_nodes, self.num_heads, device=node_features.device)
        sum_exp.index_add_(0, tgt, exp_weights)
        attn_weights = exp_weights / (sum_exp[tgt] + 1e-8)

        # Apply attention to values
        attn_output = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=node_features.device)
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # (num_edges, num_heads, 1)
        weighted_values = V_src * attn_weights_expanded     # (num_edges, num_heads, head_dim)
        attn_output.index_add_(0, tgt, weighted_values)

        # Reshape and project
        attn_output = attn_output.view(num_nodes, self.hidden_dim)
        attn_output = self.output_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # Residual + norm
        residual = node_features
        if residual.shape[-1] != attn_output.shape[-1]:
            residual = F.linear(residual, torch.eye(attn_output.shape[-1], device=residual.device)[: residual.shape[-1]])
        return self.norm(attn_output + residual)


class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder with 2-3 message-passing layers.
    """

    def __init__(
        self,
        node_feat_dim: int = 32,
        edge_feat_dim: int = 10,
        z_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # Message-passing layers
        self.layers = nn.ModuleList([
            GraphAttentionLayer(
                node_feat_dim=hidden_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Graph context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Output heads
        self.agent_head = nn.Linear(hidden_dim, output_dim)
        self.global_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = graph_tensor.node_features
        mask = graph_tensor.node_mask
        context = graph_tensor.graph_context
        edge_index = graph_tensor.edge_index
        edge_features = graph_tensor.edge_features

        # Input projection
        h = self.input_proj(x)

        # Message passing
        for layer in self.layers:
            h = layer(h, edge_index, edge_features)
            h = h * mask.unsqueeze(-1)  # mask out invalid nodes

        # Global pooling
        global_emb = _masked_mean_pool(h, mask)

        # Fuse with graph context
        if context is not None:
            context = context.unsqueeze(0)
            global_emb = global_emb.unsqueeze(0)
            global_emb = self.context_fusion(torch.cat([global_emb, context], dim=-1)).squeeze(0)

        # Agent embeddings
        agent_indices = graph_tensor.agent_node_indices
        if agent_indices:
            agent_emb = h[agent_indices]
            agent_emb = self.agent_head(agent_emb)
        else:
            agent_emb = self.agent_head(global_emb.unsqueeze(0))

        global_emb = self.global_head(global_emb)

        return agent_emb, global_emb


# ---------------------------------------------------------------------------
# Geometry-Aware Encoder (EGNN-style)
# ---------------------------------------------------------------------------

class GeometryAwareEncoder(nn.Module):
    """
    Geometry-aware graph encoder that updates both node features and coordinates.
    Inspired by EGNN but simplified for lightweight use.
    """

    def __init__(
        self,
        node_feat_dim: int = 32,
        edge_feat_dim: int = 10,
        z_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 128,
        coord_dim: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.coord_dim = coord_dim

        # Feature MLPs
        self.feature_mlp = nn.Sequential(
            nn.Linear(node_feat_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Coordinate update MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, coord_dim),
        )

        # Edge message MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Graph context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Output heads
        self.agent_head = nn.Linear(hidden_dim, output_dim)
        self.global_head = nn.Linear(hidden_dim, output_dim)

    def _get_relative_coords(self, coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        tgt = edge_index[1]
        rel_coords = coords[tgt] - coords[src]  # (num_edges, coord_dim)
        dist = torch.norm(rel_coords, dim=-1, keepdim=True).clamp(min=1e-6)
        return rel_coords, dist

    def forward(self, graph_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = graph_tensor.node_features
        mask = graph_tensor.node_mask
        context = graph_tensor.graph_context
        edge_index = graph_tensor.edge_index
        edge_features = graph_tensor.edge_features

        num_nodes = x.shape[0]

        # Initialize hidden features from input features
        h = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
        h = self.feature_mlp(torch.cat([x, h], dim=-1))

        # Extract coordinates from node features (indices 0-1 for x,y)
        coords = x[:, : self.coord_dim].clone()

        if edge_index.shape[1] > 0:
            rel_coords, dist = self._get_relative_coords(coords, edge_index)
            src = edge_index[0]
            tgt = edge_index[1]

            for _ in range(3):  # 3 message-passing steps
                # Edge messages
                h_src = h[src]
                h_tgt = h[tgt]
                edge_input = torch.cat([h_src, h_tgt, edge_features, dist], dim=-1)
                edge_msg = self.edge_mlp(edge_input)

                # Aggregate messages
                agg = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
                agg.index_add_(0, tgt, edge_msg)

                # Update features
                h = self.feature_mlp(torch.cat([x, h + agg], dim=-1))
                h = h * mask.unsqueeze(-1)

                # Update coordinates (equivariant)
                coord_update = self.coord_mlp(h[tgt])
                coords = coords + 0.1 * torch.zeros_like(coords).index_add_(0, tgt, coord_update) / max(1, edge_index.shape[1])
        else:
            # No edges: just refine features
            for _ in range(3):
                h = self.feature_mlp(torch.cat([x, h], dim=-1))
                h = h * mask.unsqueeze(-1)

        # Global pooling
        global_emb = _masked_mean_pool(h, mask)

        # Fuse with graph context
        if context is not None:
            context = context.unsqueeze(0)
            global_emb = global_emb.unsqueeze(0)
            global_emb = self.context_fusion(torch.cat([global_emb, context], dim=-1)).squeeze(0)

        # Agent embeddings
        agent_indices = graph_tensor.agent_node_indices
        if agent_indices:
            agent_emb = h[agent_indices]
            agent_emb = self.agent_head(agent_emb)
        else:
            agent_emb = self.agent_head(global_emb.unsqueeze(0))

        global_emb = self.global_head(global_emb)

        return agent_emb, global_emb


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------

def create_encoder(encoder_type: str, **kwargs) -> nn.Module:
    """Create a GNN encoder by name.

    Args:
        encoder_type: 'mlp', 'gat', or 'geometry'
        **kwargs: passed to encoder constructor

    Returns:
        nn.Module encoder
    """
    if encoder_type == "mlp":
        return MLPBaselineEncoder(**kwargs)
    elif encoder_type == "gat":
        return GATEncoder(**kwargs)
    elif encoder_type == "geometry":
        return GeometryAwareEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Choose 'mlp', 'gat', or 'geometry'.")
