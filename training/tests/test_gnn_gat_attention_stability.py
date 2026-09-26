"""Regression coverage for the GAT attention softmax numerical-stability defect.

Background
----------
`GraphAttentionLayer.forward` previously computed its softmax denominator as::

    max_logits = _aggregate_by_index(attn_logits.clamp(max=0), tgt, num_nodes)
    attn_logits = attn_logits - max_logits[tgt]

Two compounding faults:

1. ``_aggregate_by_index`` is a *sum* scatter, not a *max*, so the per-target
   shift is a sum of neighbour logits rather than the largest one.
2. ``.clamp(max=0)`` forces that sum to be ``<= 0``, so subtracting it can only
   *increase* magnitude. The "stability" shift was unconditionally
   destabilising.

On live ``academy_3_vs_1_with_keeper`` graphs, edge features grow to about
+-77 (vs +-1.34 at reset). By the third message-passing layer this drove
``attn_logits`` to 109.56, past the float32 ``exp()`` overflow threshold
(~88.72). ``exp()`` saturated to ``inf`` and ``inf / (inf + 1e-8)`` produced
``NaN`` in all 10 node rows, which surfaced as NaN actor logits during the
first ``collect_rollout`` and blocked all ``gnn:gat`` training.

The 77 pre-existing GNN tests all use synthetic or reset-state graphs, where
edge features stay within +-1.34 and the overflow never triggers. That is the
coverage gap that let this ship. These tests close it two ways: a fully
deterministic overflow repro, and a multi-step *live* environment sweep.

Verified against the pre-fix code: these tests fail without the fix and pass
with it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from training.gnn_encoders import GraphAttentionLayer
from training.gnn_mappo_networks import GNNMAPPOActor, _as_graph_tensor

SCENARIO = "academy_3_vs_1_with_keeper"
N_EDGES = 12
# float32 exp() saturates to inf above ~88.72
EXP_OVERFLOW_THRESHOLD = 88.72


def _layer() -> GraphAttentionLayer:
    torch.manual_seed(0)
    # node_feat_dim == hidden_dim so the residual branch at
    # GraphAttentionLayer.forward is not taken. In the real encoder, input_proj
    # already maps 32 -> 128, so attention layers only ever see hidden_dim-wide
    # features; matching that here keeps the test on the attention path.
    layer = GraphAttentionLayer(
        node_feat_dim=128, edge_feat_dim=10, hidden_dim=128, num_heads=4, dropout=0.0
    )
    layer.eval()
    return layer


def _star_graph():
    """One target with N_EDGES incoming edges, so the softmax denominator is real."""
    node_features = torch.zeros(2, 128, dtype=torch.float32)
    src = torch.zeros(N_EDGES, dtype=torch.long)
    tgt = torch.ones(N_EDGES, dtype=torch.long)
    edge_index = torch.stack([src, tgt], dim=0)
    edge_features = torch.zeros(N_EDGES, 10, dtype=torch.float32)
    return node_features, edge_index, edge_features


def _force_constant_logits(layer: GraphAttentionLayer, target_logit: float) -> None:
    """Pin the layer so every edge gets exactly ``target_logit``.

    Zeroing query/key removes the Q.K term, then the edge-projection bias sets
    the logit directly. This makes the overflow condition exact and
    independent of random initialisation.
    """
    with torch.no_grad():
        layer.query_proj.weight.zero_()
        layer.query_proj.bias.zero_()
        layer.key_proj.weight.zero_()
        layer.key_proj.bias.zero_()
        # E is viewed as (num_edges, num_heads, head_dim) and summed over
        # head_dim, then scaled. head_dim = hidden_dim // num_heads = 32.
        per_head_dim = target_logit / (layer.scale * layer.head_dim)
        layer.edge_proj.weight.zero_()
        layer.edge_proj.bias.fill_(per_head_dim)


# ---------------------------------------------------------------------------
# 1. Deterministic overflow repro (seed-independent)
# ---------------------------------------------------------------------------


def test_attention_survives_large_positive_logits():
    """attn_logits above the exp() overflow threshold must not produce NaN.

    This is the exact failure that blocked gnn:gat training.
    """
    layer = _layer()
    node_features, edge_index, edge_features = _star_graph()
    _force_constant_logits(layer, 120.0)  # 120 >> 88.72

    out = layer(node_features, edge_index, edge_features)

    assert torch.isfinite(out).all(), (
        f"GraphAttentionLayer produced {int(torch.isnan(out).sum())} NaN / "
        f"{int(torch.isinf(out).sum())} Inf values at attn_logits=120"
    )


@pytest.mark.parametrize("logit", [89.0, 120.0, 500.0, 5000.0])
def test_attention_finite_across_overflow_range(logit):
    layer = _layer()
    node_features, edge_index, edge_features = _star_graph()
    _force_constant_logits(layer, logit)

    out = layer(node_features, edge_index, edge_features)

    assert torch.isfinite(out).all(), f"non-finite output at attn_logits={logit}"
    assert logit > EXP_OVERFLOW_THRESHOLD


def test_attention_backward_stays_finite():
    """Gradients must remain finite; inf in the forward graph poisons backward."""
    layer = _layer()
    node_features, edge_index, edge_features = _star_graph()
    _force_constant_logits(layer, 120.0)

    out = layer(node_features, edge_index, edge_features)
    out.sum().backward()

    grads = [p.grad for p in layer.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    for name, p in layer.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_isolated_node_yields_finite_output():
    """Zero in-degree nodes must not produce NaN (the +1e-8 guard must hold)."""
    layer = _layer()
    node_features, edge_index, edge_features = _star_graph()
    _force_constant_logits(layer, 120.0)
    node_features = torch.cat(
        [node_features, torch.full((1, 128), 3.0)], dim=0
    )  # node 2 has no edges at all

    out = layer(node_features, edge_index, edge_features)

    assert torch.isfinite(out).all(), (
        "isolated (zero-degree) node produced non-finite output"
    )


# ---------------------------------------------------------------------------
# 2. Multi-step LIVE environment graph sweep (the real coverage gap)
# ---------------------------------------------------------------------------


def test_live_env_multi_step_graphs_produce_finite_logits():
    """Drive a real scenario past reset and assert finite logits every step.

    The pre-fix encoder NaN'd reproducibly at rollout step 17-27 on this exact
    scenario, so 40 steps is comfortably beyond the observed trigger point.
    """
    from torch.distributions import Categorical

    from training.gmn_pettingzoo import GMNMultiAgentEnv
    from training.mappo_rollout import (
        _mask_matrix,
        unwrap_graph_observations,
        unwrap_masks,
        unwrap_obs,
    )

    torch.manual_seed(42)
    env = GMNMultiAgentEnv(
        scenario=SCENARIO,
        auto_start_bridge=True,
        batch_size=1,
        opponent_difficulty="medium",
        opponent_pool=None,
        training_mode=True,
        shot_clock_truncates=False,
        shot_clock_t_max=600,
        enable_exploration_bonus=True,
        exploration_beta=0.03,
        include_graph_observations=True,
    )
    try:
        actor = GNNMAPPOActor(action_dim=19, encoder_type="gat")
        actor.eval()
        obs_dict, infos = env.reset(seed=42)
        controllable = list(env.possible_agents)
        graphs = unwrap_graph_observations(infos, controllable)
        mask_matrix = _mask_matrix(unwrap_masks(obs_dict), controllable)

        max_edge_mag = 0.0
        n_steps = 40
        for step in range(n_steps):
            gts = [_as_graph_tensor(graphs[a]) for a in controllable]
            for g in gts:
                assert torch.isfinite(g.node_features).all(), f"NaN node features at step {step}"
                assert torch.isfinite(g.edge_features).all(), f"NaN edge features at step {step}"
                if g.edge_features.numel():
                    max_edge_mag = max(max_edge_mag, float(g.edge_features.abs().max()))

            with torch.no_grad():
                logits = actor.raw_logits(gts)
            assert torch.isfinite(logits).all(), (
                f"non-finite actor logits at step {step} "
                f"(NaN={int(torch.isnan(logits).sum())}, Inf={int(torch.isinf(logits).sum())})"
            )

            m = torch.as_tensor(mask_matrix, dtype=torch.bool).reshape(logits.shape)
            act = Categorical(logits=logits.masked_fill(~m, float("-inf"))).sample()
            obs_dict, rewards, terms, truncs, infos = env.step(
                {a: int(act[i]) for i, a in enumerate(controllable)}
            )
            graphs = unwrap_graph_observations(infos, controllable)
            mask_matrix = _mask_matrix(unwrap_masks(obs_dict), controllable)
            flat = unwrap_obs(obs_dict)
            arrs = list(flat.values()) if isinstance(flat, dict) else [flat]
            for arr in arrs:
                assert np.isfinite(np.asarray(arr)).all(), (
                    f"non-finite flat observation at step {step}"
                )

        # Document that live graphs really do exceed the reset-state range the
        # synthetic tests exercise, so this test guards a real regime.
        assert max_edge_mag > 10.0, (
            f"live edge features only reached {max_edge_mag:.3f}; the overflow "
            f"regime was not exercised by this sweep"
        )
    finally:
        env.close()

