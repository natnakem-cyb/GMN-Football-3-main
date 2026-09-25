"""ONNX export and parity checks for graph-policy actor checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from training.checkpoint_contract import create_policy_checkpoint_contract
from training.gnn_graph_to_tensor import GraphTensor
from training.gnn_mappo_networks import GNNMAPPOActor


INPUT_NAMES = (
    "node_features",
    "edge_index",
    "edge_features",
    "node_mask",
    "graph_context",
    "agent_node_index",
)


class GNNActorOnnxModule(nn.Module):
    """Expose one tensorized graph and controlled-node index to ONNX."""

    def __init__(self, actor: GNNMAPPOActor):
        super().__init__()
        self.encoder = actor.encoder
        self.policy_head = actor.policy_head

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: torch.Tensor,
        graph_context: torch.Tensor,
        agent_node_index: torch.Tensor,
    ) -> torch.Tensor:
        graph = GraphTensor(
            node_features=node_features,
            node_type=torch.zeros(
                node_features.shape[0], dtype=torch.long, device=node_features.device
            ),
            edge_index=edge_index,
            edge_type=torch.zeros(
                edge_index.shape[1], dtype=torch.long, device=edge_index.device
            ),
            edge_features=edge_features,
            node_mask=node_mask,
            agent_node_indices=agent_node_index,
            graph_context=graph_context,
            node_id_to_index={},
            scenario_id="onnx",
        )
        agent_embeddings, _ = self.encoder(graph)
        return self.policy_head(agent_embeddings)


def sample_graph_inputs(
    seed: int = 123, num_nodes: int = 5, edge_index: torch.Tensor | None = None
) -> tuple[torch.Tensor, ...]:
    """Small valid graph used only to trace/export and verify the ONNX module."""
    generator = torch.Generator().manual_seed(seed)
    nodes = torch.randn(num_nodes, 32, generator=generator)
    nodes[:, 17] = 0.0
    nodes[1, 17] = 1.0
    if edge_index is None:
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 1], [1, 0, 3, 2, 1, 4]], dtype=torch.long)
    return (
        nodes,
        edge_index,
        torch.randn(edge_index.shape[1], 10, generator=generator),
        torch.ones(nodes.shape[0], dtype=torch.float32),
        torch.zeros(8, dtype=torch.float32),
        torch.tensor([1], dtype=torch.long),
    )


def export_gnn_actor_onnx(
    checkpoint: dict[str, Any],
    output_path: str,
    scenario_id: str,
    algorithm: str = "MAPPO",
) -> dict[str, Any]:
    """Export one GNN actor; validates ONNX and checks CPU runtime parity."""
    architecture = checkpoint.get("policy_architecture")
    if architecture not in ("gnn:mlp", "gnn:gat", "gnn:geometry"):
        raise ValueError(f"Unsupported GNN architecture: {architecture!r}")
    actor_state = checkpoint["actor"]
    hidden_dim = int(actor_state["policy_head.weight"].shape[1])
    actor = GNNMAPPOActor(
        encoder_type=architecture.split(":", 1)[1], hidden_dim=hidden_dim
    )
    actor.load_state_dict(actor_state)
    actor.eval()
    module = GNNActorOnnxModule(actor).eval()
    example = sample_graph_inputs()
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    dynamic_axes = {
        "node_features": {0: "num_nodes"},
        "edge_index": {1: "num_edges"},
        "edge_features": {0: "num_edges"},
        "node_mask": {0: "num_nodes"},
    }
    torch.onnx.export(
        module,
        example,
        output_path,
        input_names=list(INPUT_NAMES),
        output_names=["action_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    runtime = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    candidate_feeds = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(INPUT_NAMES, example)
    }
    runtime_input_names = [item.name for item in runtime.get_inputs()]
    feeds = {name: candidate_feeds[name] for name in runtime_input_names}
    actual = runtime.run(["action_logits"], feeds)[0]
    with torch.no_grad():
        expected = module(*example).cpu().numpy()
    max_abs_diff = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=1e-4, atol=1e-5):
        raise RuntimeError(
            f"ONNX/PyTorch action-logit parity failed (max_abs_diff={max_abs_diff:.3g})"
        )

    # Confirm that the declared node and edge axes really accept a second graph size.
    dynamic_example = sample_graph_inputs(
        seed=19,
        num_nodes=7,
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 1], [1, 0, 3, 2, 5, 4, 1, 6]],
            dtype=torch.long,
        ),
    )
    dynamic_candidates = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(INPUT_NAMES, dynamic_example)
    }
    dynamic_feeds = {
        name: dynamic_candidates[name] for name in runtime_input_names
    }
    dynamic_actual = runtime.run(["action_logits"], dynamic_feeds)[0]
    with torch.no_grad():
        dynamic_expected = module(*dynamic_example).cpu().numpy()
    if not np.allclose(dynamic_actual, dynamic_expected, rtol=1e-4, atol=1e-5):
        diff = float(np.max(np.abs(dynamic_actual - dynamic_expected)))
        raise RuntimeError(
            f"Dynamic ONNX/PyTorch parity failed (max_abs_diff={diff:.3g})"
        )

    with open(output_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    metadata = {
        "format": "gmn-gnn-actor-onnx-v1",
        "policy_architecture": architecture,
        "checkpoint_contract": checkpoint.get(
            "checkpoint_contract", create_policy_checkpoint_contract(architecture)
        ),
        "scenario_id": scenario_id,
        "algorithm": algorithm,
        "input_names": runtime_input_names,
        "input_dtypes": {
            "node_features": "float32",
            "edge_index": "int64",
            "edge_features": "float32",
            "node_mask": "float32",
            "graph_context": "float32",
            "agent_node_index": "int64",
        },
        "output_name": "action_logits",
        "action_dim": int(checkpoint.get("action_dim", 19)),
        "onnx_sha256": digest,
        "parity_max_abs_diff": max_abs_diff,
        "graph_preprocessing": (
            "Build graph with training.gnn_graph_builder.build_graph and encode it with "
            "training.gnn_graph_to_tensor.graph_to_tensors. Browser callers must supply "
            "these tensor fields; the game UI graph builder is not included in this export."
        ),
    }
    with open(output_path + ".json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata
