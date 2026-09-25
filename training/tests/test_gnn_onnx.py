"""Export/runtime contract checks for browser-consumable GNN ONNX actors."""

import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from training.checkpoint_contract import create_policy_checkpoint_contract
from training.gnn_mappo_networks import GNNMAPPOActor
from training.gnn_onnx import INPUT_NAMES, export_gnn_actor_onnx


@pytest.mark.parametrize("encoder_type", ["mlp", "gat", "geometry"])
def test_gnn_onnx_exports_dynamic_graph_inputs_and_matches_pytorch(
    tmp_path, encoder_type
):
    architecture = f"gnn:{encoder_type}"
    actor = GNNMAPPOActor(encoder_type=encoder_type, hidden_dim=16).eval()
    checkpoint = {
        "policy_architecture": architecture,
        "checkpoint_contract": create_policy_checkpoint_contract(architecture),
        "obs_dim": 127,
        "action_dim": 19,
        "actor": actor.state_dict(),
    }
    output_path = tmp_path / f"{encoder_type}.onnx"

    metadata = export_gnn_actor_onnx(
        checkpoint, str(output_path), "academy_empty_goal"
    )

    assert output_path.exists()
    assert (tmp_path / f"{encoder_type}.onnx.json").exists()
    assert metadata["policy_architecture"] == architecture
    assert {"node_features", "agent_node_index"}.issubset(metadata["input_names"])
    assert metadata["parity_max_abs_diff"] < 1e-5
