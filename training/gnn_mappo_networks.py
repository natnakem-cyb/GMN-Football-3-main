"""MAPPO actor and centralized critic backed by the Phase 4 GNN encoders."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from training.gnn_encoders import create_encoder
from training.gnn_graph_to_tensor import graph_to_tensors


def _as_graph_tensor(graph: Any):
    """Accept a schema graph or an already tensorized graph."""
    if hasattr(graph, "node_features"):
        return graph
    if not isinstance(graph, dict):
        raise TypeError(f"Expected graph dict or GraphTensor, got {type(graph).__name__}")
    return graph_to_tensors(graph)


class GNNMAPPOActor(nn.Module):
    """Shared categorical actor that reads one controlled-player graph per row."""

    requires_graph_observations = True

    def __init__(self, action_dim: int = 19, encoder_type: str = "gat", hidden_dim: int = 128):
        super().__init__()
        self.action_dim = int(action_dim)
        self.encoder_type = encoder_type
        self.encoder = create_encoder(
            encoder_type,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=0.0,
        ) if encoder_type == "gat" else create_encoder(
            encoder_type,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.policy_head = nn.Linear(hidden_dim, self.action_dim)

    def forward(
        self,
        graphs: Sequence[Any],
        action_mask: Optional[torch.Tensor] = None,
    ) -> Categorical:
        logits = self.raw_logits(graphs)
        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
            if mask.shape != logits.shape:
                mask = mask.reshape(logits.shape)
            if not mask.any(dim=-1).all():
                raise ValueError("Action mask contains a row with no legal actions")
            logits = logits.masked_fill(~mask, float("-inf"))
        return Categorical(logits=logits)

    def raw_logits(self, graphs: Sequence[Any]) -> torch.Tensor:
        """Return unmasked logits for diagnostics and canonical evaluation."""
        if not graphs:
            raise ValueError("GNN actor requires at least one graph")
        embeddings = []
        for graph in graphs:
            agent_embeddings, _ = self.encoder(_as_graph_tensor(graph))
            if agent_embeddings.shape[0] != 1:
                raise ValueError(
                    "Each policy graph must mark exactly one controlled player; "
                    f"got {agent_embeddings.shape[0]}"
                )
            embeddings.append(agent_embeddings[0])
        return self.policy_head(torch.stack(embeddings))


class GNNMAPPOCritic(nn.Module):
    """Centralized scalar critic over graph-level encoder embeddings."""

    requires_graph_observations = True

    def __init__(self, encoder_type: str = "gat", hidden_dim: int = 128):
        super().__init__()
        self.encoder_type = encoder_type
        self.encoder = create_encoder(
            encoder_type,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=0.0,
        ) if encoder_type == "gat" else create_encoder(
            encoder_type,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graphs: Sequence[Any]) -> torch.Tensor:
        if not graphs:
            raise ValueError("GNN critic requires at least one graph")
        global_embeddings = [
            self.encoder(_as_graph_tensor(graph))[1]
            for graph in graphs
        ]
        return self.value_head(torch.stack(global_embeddings)).squeeze(-1)
