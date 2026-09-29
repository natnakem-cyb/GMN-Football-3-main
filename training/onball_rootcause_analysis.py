"""Read-only root-cause analysis of the GNN on-ball action collapse.

Consumes the state captures written by training/onball_state_capture.py and
answers the four mechanism questions raised by the milestone sweep, WITHOUT any
training, weight change, or Unity bridge:

  S1 STRUCTURE  - can possession / ball information even reach the actor's
                  agent node in the graph? (in-degree by edge type, directed
                  ball->agent reachability at 1/2/3 hops, whether the graph
                  reports ownership on ticks where the engine reports a carrier)
  S2 ATTENTION  - are the GAT weights degenerate (near-uniform => plain
                  averaging), and how often is attention structurally bypassed
                  because the agent node has no incoming edge at all?
  S3 LOGIT      - on-ball vs off-ball differentiation for GNN vs Flat, plus a
                  CAUSAL ball/possession ablation (destroy ball information in
                  the input and measure how much the actor output moves) and a
                  positive control that injects a node-level possession bit.
  S4 INIT       - fresh randomly-initialised actors of both architectures on
                  the same captured states, to separate "learned pathology"
                  from "architecture + initialisation prior".
  S5 HEAD       - collapse audit of the trained heads (bias-only argmax, logit
                  spread across states, agent-embedding effective rank).

Nothing here writes model weights; outputs are one JSON report, one CSV summary
and a printed table. Logits are compared on RAW (unmasked) logits so that
masked -inf entries cannot contaminate deltas; argmax and pi(pass/shot) mass
use the masked logits, exactly as the probes do.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.gnn_graph_to_tensor import (
    GraphTensor,
    PLAYER_BALL_DISTANCE_DIM,
    PLAYER_BALL_REL_X_DIM,
    PLAYER_BALL_REL_Y_DIM,
    PLAYER_HAS_POSSESSION_DIM,
    PLAYER_IS_NEAREST_TO_BALL_DIM,
)
from training.gnn_mappo_networks import GNNMAPPOActor
from training.mappo_networks import SharedActor
from training.checkpoint_contract import load_mappo_actor

ACTION_NAMES = [
    "IDLE", "LEFT", "RIGHT", "UP", "DOWN", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT",
    "DOWN_RIGHT", "LONG_PASS", "HIGH_PASS", "SHORT_PASS", "SHOT", "SPRINT",
    "RELEASE_DIRECTION", "RELEASE_SPRINT", "SLIDING", "DRIBBLE",
    "RELEASE_DRIBBLE",
]
PASS_SHOT = (9, 10, 11, 12)
EDGE_TYPE_NAMES = [
    "TEAMMATE", "OPPONENT", "NEAR", "POSSESSES", "PLAYER_BALL", "PLAYER_GOAL",
    "BALL_GOAL", "ASSIGNED_TO", "BELONGS_TO_SHAPE", "SCENARIO_CONTEXT",
    "FORMATION_ADJACENCY", "FORMATION_LINE", "FORMATION_LANE",
]
NT_PLAYER, NT_BALL, NT_GOAL, NT_SLOT, NT_SHAPE = 0, 1, 2, 3, 4
NODE_TYPE_NAME = {NT_PLAYER: "PLAYER", NT_BALL: "BALL", NT_GOAL: "GOAL",
                  NT_SLOT: "FORMATION_SLOT", NT_SHAPE: "TEAM_SHAPE"}
BALL_INFO_EDGE_TYPES = {3, 4}          # POSSESSES, PLAYER_BALL
# The ball-visibility fix moved ball information *inside* the player vector, so
# the ablations below must reach those dims as well - otherwise "ball_gone"
# leaves the agent's own ball-relative geometry untouched and understates the
# effect. Dims 30-31 remain spare, which is where `inject_poss` writes.
BALL_GEOMETRY_PLAYER_DIMS = [
    PLAYER_BALL_REL_X_DIM, PLAYER_BALL_REL_Y_DIM, PLAYER_BALL_DISTANCE_DIM,
    PLAYER_IS_NEAREST_TO_BALL_DIM,
]
# Flat observation slices used for the ablation controls (simple115_v3_role).
FLAT_BALL_POSVEL = slice(88, 94)       # ball x,y,z + vx,vy,vz
FLAT_TEAM_OWNERSHIP = slice(94, 97)    # one-hot [no-one, left, right]


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def auc_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC of a scalar score against a binary label vector."""
    if labels.min() == labels.max():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s = scores[order]
    r = np.arange(1, len(scores) + 1, dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[i:j + 1] = r[i:j + 1].mean()
        i = j + 1
    ranks[order] = r
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def effective_rank(vecs: np.ndarray) -> float:
    """Entropy-based effective rank of a centred embedding matrix."""
    if vecs.shape[0] < 3:
        return float("nan")
    x = vecs - vecs.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(x, compute_uv=False)
    p = sv / (sv.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(math.exp(-(p * np.log(p)).sum()))


def mode_share(values: Iterable[int]) -> Tuple[float, int, Dict[str, int]]:
    c = Counter(int(v) for v in values)
    if not c:
        return float("nan"), 0, {}
    top, n = c.most_common(1)[0]
    total = sum(c.values())
    return n / total, len(c), {ACTION_NAMES[k]: v for k, v in sorted(c.items())}


# ---------------------------------------------------------------------------
# Capture loading / graph reconstruction
# ---------------------------------------------------------------------------

def load_capture(meta_path: str) -> Dict[str, Any]:
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["_npz"] = np.load(os.path.splitext(meta_path)[0] + ".npz")
    return meta


def rebuild_graph(z, key: str) -> GraphTensor:
    return GraphTensor(
        node_features=torch.from_numpy(z[f"{key}_nf"].astype(np.float32)),
        node_type=torch.from_numpy(z[f"{key}_nt"].astype(np.longlong)),
        edge_index=torch.from_numpy(z[f"{key}_ei"].astype(np.longlong)),
        edge_type=torch.from_numpy(z[f"{key}_et"].astype(np.longlong)),
        edge_features=torch.from_numpy(z[f"{key}_ef"].astype(np.float32)),
        node_mask=torch.from_numpy(z[f"{key}_nm"].astype(np.float32)),
        agent_node_indices=[int(i) for i in z[f"{key}_ani"]],
        graph_context=(None if z[f"{key}_ctx"].size == 0
                       else torch.from_numpy(z[f"{key}_ctx"].astype(np.float32))),
        node_id_to_index={},
        scenario_id="captured",
    )


def variant_graph(gt: GraphTensor, mode: str, agent_node: int) -> GraphTensor:
    """Ball/possession ablations plus a positive-control possession injection.

    as_is        : untouched capture.
    ball_zeroed  : the ball node's whole feature vector destroyed.
    poss_ablated : ball ownership one-hot forced to "none" and every
                   PLAYER_BALL edge has_possession flag forced to 0.
    ball_gone    : ball node zeroed AND every ball-touching edge feature zeroed.
                   Since the ball-visibility fix the ball is also described
                   inside every player row, so this additionally clears the
                   agent-agnostic ball geometry dims (32-34, 37) on all nodes.
                   Possession is deliberately NOT cleared here - that stays the
                   job of poss_ablated - so the two columns stay comparable with
                   the pre-fix report.
    inject_poss  : positive control - write a possession bit into the agent's
                   OWN node feature (unused dim 30), i.e. what a possession-aware
                   graph schema would hand the actor directly. Still dim 30
                   post-fix: the new features landed at 32-38 and left 30-31 spare.
    """
    nf = gt.node_features.clone()
    ef = gt.edge_features.clone()
    ball_rows = [i for i in range(nf.shape[0]) if int(gt.node_type[i]) == NT_BALL]
    if mode == "as_is":
        pass
    elif mode == "ball_zeroed":
        for b in ball_rows:
            nf[b] = 0.0
    elif mode == "poss_ablated":
        for b in ball_rows:
            nf[b, 7], nf[b, 8], nf[b, 9] = 1.0, 0.0, 0.0
        for e, t in enumerate(gt.edge_type.tolist()):
            if t == 4:                       # PLAYER_BALL dim 6 = has_possession
                ef[e, 6] = 0.0
        # Post-fix, possession is also a node feature now: an ablation that
        # stopped at the edge flag would leave the carrier still self-reporting.
        # Pre-fix captures are only 32 dims wide and have no such dim.
        if nf.shape[1] > PLAYER_HAS_POSSESSION_DIM:
            for i in range(nf.shape[0]):
                if int(gt.node_type[i]) == NT_PLAYER:
                    nf[i, PLAYER_HAS_POSSESSION_DIM] = 0.0
    elif mode == "ball_gone":
        for b in ball_rows:
            nf[b] = 0.0
        ball_set = set(ball_rows)
        for e, t in enumerate(gt.edge_type.tolist()):
            touches = (int(gt.edge_index[0, e]) in ball_set
                       or int(gt.edge_index[1, e]) in ball_set)
            if t in (3, 4, 6) or touches:
                ef[e] = 0.0
        if nf.shape[1] > PLAYER_IS_NEAREST_TO_BALL_DIM:
            nf[:, BALL_GEOMETRY_PLAYER_DIMS] = 0.0
    elif mode == "inject_poss":
        nf[agent_node, 30] = 1.0
    else:
        raise ValueError(mode)
    return GraphTensor(
        node_features=nf, node_type=gt.node_type, edge_index=gt.edge_index,
        edge_type=gt.edge_type, edge_features=ef, node_mask=gt.node_mask,
        agent_node_indices=list(gt.agent_node_indices),
        graph_context=gt.graph_context, node_id_to_index={},
        scenario_id=gt.scenario_id,
    )


def flat_variant(row: np.ndarray, mode: str) -> np.ndarray:
    x = row.astype(np.float32).copy()
    if mode == "as_is":
        return x
    if mode in ("ball_zeroed", "ball_gone"):
        x[FLAT_BALL_POSVEL] = 0.0
    if mode in ("poss_ablated", "ball_gone"):
        x[FLAT_TEAM_OWNERSHIP] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return x


# ---------------------------------------------------------------------------
# Faithful GAT forward that also exposes the attention weights
# ---------------------------------------------------------------------------

def gat_layer_with_attention(layer, h, edge_index, edge_features):
    """Re-derives GraphAttentionLayer.forward, returning per-edge attention.

    Returns (out, weights, indegree, uniform_out). uniform_out is the same
    aggregation with the softmax replaced by 1/indegree, i.e. plain neighbour
    averaging - comparing it with `out` says whether learned attention has
    actually moved away from averaging.
    """
    import torch.nn.functional as F
    from training.gnn_encoders import _aggregate_by_index

    num_nodes, num_edges = h.shape[0], edge_index.shape[1]
    heads = layer.num_heads
    deg = (_aggregate_by_index(torch.ones(num_edges, 1), edge_index[1], num_nodes)
           .squeeze(-1) if num_edges else torch.zeros(num_nodes))
    if num_edges == 0:
        out = layer.norm(layer.output_proj(torch.zeros(num_nodes, layer.hidden_dim)))
        if h.shape[-1] == out.shape[-1]:
            out = out + h
        return out, np.zeros((0, heads)), deg, out

    Q = layer.query_proj(h).view(num_nodes, heads, layer.head_dim)
    K = layer.key_proj(h).view(num_nodes, heads, layer.head_dim)
    V = layer.value_proj(h).view(num_nodes, heads, layer.head_dim)
    E = layer.edge_proj(edge_features).view(num_edges, heads, layer.head_dim)
    src, tgt = edge_index[0], edge_index[1]
    logits = (Q[tgt] * K[src]).sum(dim=-1) * layer.scale
    logits = logits + E.sum(dim=-1) * layer.scale
    mx = torch.full((num_nodes, heads), float("-inf"), dtype=logits.dtype)
    mx = mx.scatter_reduce(0, tgt.unsqueeze(-1).expand(-1, heads), logits,
                           reduce="amax", include_self=True)
    ex = (logits - mx[tgt]).exp()
    denom = _aggregate_by_index(ex, tgt, num_nodes)
    w = ex / (denom[tgt] + 1e-8)

    def _finish(vv):
        o = layer.output_proj(vv.view(num_nodes, layer.hidden_dim))
        o = layer.dropout(o)
        res = h
        if res.shape[-1] != o.shape[-1]:
            res = F.linear(res, torch.eye(o.shape[-1])[: res.shape[-1]])
        return layer.norm(o + res)

    out = _finish(_aggregate_by_index(V[src] * w.unsqueeze(-1), tgt, num_nodes))
    uni_w = (1.0 / deg.clamp(min=1.0))[tgt]
    uni = _finish(_aggregate_by_index(V[src] * uni_w.view(-1, 1, 1), tgt, num_nodes))
    return out, w.detach().cpu().numpy(), deg.detach().cpu().numpy(), uni


def topology_report(graphs: List[GraphTensor]) -> Dict[str, Any]:
    """Is the graph STRUCTURE (edge set / indices) state-dependent at all?

    A state-independent edge_index means the GNN can only ever mix a FIXED set of
    neighbours; relational facts such as "who is near the ball" cannot be
    expressed structurally, only through edge features on those fixed edges.
    """
    sigs: List[Tuple[Any, ...]] = []
    indeg_profiles: List[Tuple[int, ...]] = []
    agent_rows: List[Dict[str, Any]] = []
    for gt in graphs:
        ei = gt.edge_index
        sigs.append(tuple(map(tuple, ei.t().tolist())) + tuple(gt.edge_type.tolist()))
        n = gt.node_features.shape[0]
        indeg_profiles.append(tuple(int((ei[1] == i).sum()) for i in range(n)))
        agent = gt.agent_node_indices[0] if gt.agent_node_indices else 0
        edges_in = (ei[1] == agent).nonzero(as_tuple=True)[0].tolist()
        eia = ei.numpy()
        adj = np.zeros((gt.node_features.shape[0], gt.node_features.shape[0]), dtype=bool)
        for e in range(eia.shape[1]):
            adj[int(eia[0, e]), int(eia[1, e])] = True
        balls = [i for i in range(gt.node_features.shape[0])
                 if int(gt.node_type[i]) == NT_BALL]
        reaches = any(agent in reachable(adj, b, h) for b in balls for h in (1, 2, 3))
        agent_rows.append({
            "indegree": len(edges_in),
            "incoming_edge_types": Counter(
                EDGE_TYPE_NAMES[int(gt.edge_type[e])] for e in edges_in).most_common(),
            "reaches_ball": reaches,
        })
    uniq = len(set(sigs))
    # Which edge types account for the (small) structural variation, and does any
    # state-dependent NEAR edge terminate at the agent node / touch the ball?
    type_counts = [tuple(sorted(Counter(EDGE_TYPE_NAMES[int(t)]
                                         for t in gt.edge_type.tolist()).items()))
                   for gt in graphs]
    mode_counts = Counter(type_counts).most_common(1)[0][0] if type_counts else ()
    varying = sorted({n for tc in type_counts for n, c in tc
                      if c != dict(mode_counts).get(n, 0)})
    near_shapes: Counter = Counter()
    for gt in graphs:
        nt = gt.node_type.tolist()
        for e, t in enumerate(gt.edge_type.tolist()):
            if EDGE_TYPE_NAMES[int(t)] == "NEAR":
                s_i, t_i = int(gt.edge_index[0, e]), int(gt.edge_index[1, e])
                near_shapes[f"{NODE_TYPE_NAME.get(int(nt[s_i]), nt[s_i])}"
                            f"->{NODE_TYPE_NAME.get(int(nt[t_i]), nt[t_i])}"] += 1
    in_types: Counter = Counter()
    for r in agent_rows:
        in_types.update(dict(r["incoming_edge_types"]))
    return {
        "unique_edge_sets_across_captured_ticks": uniq,
        "share_ticks_with_first_edge_set": round(
            sum(1 for s in sigs if s == sigs[0]) / max(len(sigs), 1), 6),
        "unique_indegree_profiles": len(set(indeg_profiles)),
        "indegree_profile_first_tick": list(indeg_profiles[0]) if indeg_profiles else None,
        "agent_row_indegree_values": sorted({r["indegree"] for r in agent_rows}),
        "agent_row_max_indegree": max((r["indegree"] for r in agent_rows), default=None),
        "agent_row_incoming_edge_types_all_rows": in_types.most_common(),
        "varying_edge_types": varying,
        "near_edge_endpoints_by_node_type": near_shapes.most_common(),
        "agent_rows_that_reach_ball": sum(1 for r in agent_rows if r["reaches_ball"]),
        "agent_rows_total": len(agent_rows),
        "note": ((f"edge set is a FIXED template plus {varying} proximity edges; "
                  "no ball/possession edge ever terminates at an agent row")
                 if uniq <= 2 else "edge_index varies across ticks"),
    }


def attention_report(actor: GNNMAPPOActor, graphs: List[GraphTensor]) -> Dict[str, Any]:
    """S2: attention degeneracy, and how often attention is structurally bypassed."""
    per_layer = {i: {"ent": [], "maxw": [], "dev": [], "neff": [], "avg_gap": [],
                     "a_ent": [], "a_neff": []}
                 for i in range(len(actor.encoder.layers))}
    agent_indeg: List[int] = []
    fidelity = 0.0
    with torch.no_grad():
        for gt in graphs:
            h = actor.encoder.input_proj(gt.node_features)
            for li, layer in enumerate(actor.encoder.layers):
                ref = layer(h, gt.edge_index, gt.edge_features)
                out, w, deg, uni = gat_layer_with_attention(
                    layer, h, gt.edge_index, gt.edge_features)
                deg = torch.as_tensor(deg)
                fidelity = max(fidelity, float((ref - out).abs().max()))
                agent = gt.agent_node_indices[0] if gt.agent_node_indices else 0
                if li == 0:
                    agent_indeg.append(int(deg[agent]))
                keep = (deg >= 2).nonzero(as_tuple=True)[0].tolist()
                agent_rows_w = w[(gt.edge_index[1] == agent).nonzero(as_tuple=True)[0].numpy()]
                if agent_rows_w.size:
                    for hh in range(layer.num_heads):
                        ww = np.clip(agent_rows_w[:, hh].astype(np.float64), 1e-12, None)
                        ww = ww / ww.sum()
                        ent = float(-(ww * np.log(ww)).sum())
                        if len(ww) > 1:                     # log(1) == 0 -> single source
                            per_layer[li]["a_ent"].append(ent / math.log(len(ww)))
                        else:
                            per_layer[li]["a_ent"].append(1.0)   # trivially concentrated
                        per_layer[li]["a_neff"].append(math.exp(ent))
                else:
                    per_layer[li]["a_ent"].append(0.0)   # nothing to attend to
                    per_layer[li]["a_neff"].append(0.0)
                if keep:
                    tgt = gt.edge_index[1]
                    for node in keep:
                        rows = (tgt == node).nonzero(as_tuple=True)[0].numpy()
                        for hh in range(layer.num_heads):
                            ww = np.clip(w[rows, hh].astype(np.float64), 1e-12, None)
                            ww = ww / ww.sum()
                            ent = float(-(ww * np.log(ww)).sum())
                            per_layer[li]["ent"].append(ent / math.log(len(rows)))
                            per_layer[li]["maxw"].append(float(ww.max()))
                            per_layer[li]["dev"].append(
                                float(np.abs(ww - 1.0 / len(rows)).mean()))
                            per_layer[li]["neff"].append(math.exp(ent))
                    sel = deg >= 2
                    num = float(((out - uni)[sel] ** 2).sum(-1).sqrt().mean())
                    den = float((out[sel] ** 2).sum(-1).sqrt().mean()) + 1e-12
                    per_layer[li]["avg_gap"].append(num / den)
                h = out * gt.node_mask.unsqueeze(-1)
    stats = {}
    for li, d in per_layer.items():
        stats[f"layer{li}"] = {
            "target_node_heads_scored": len(d["ent"]),
            "mean_normalised_entropy": round(float(np.mean(d["ent"])), 6) if d["ent"] else None,
            "mean_max_weight": round(float(np.mean(d["maxw"])), 6) if d["maxw"] else None,
            "mean_abs_dev_from_uniform": round(float(np.mean(d["dev"])), 6) if d["dev"] else None,
            "mean_effective_fanin": round(float(np.mean(d["neff"])), 4) if d["neff"] else None,
            "mean_rel_gap_vs_plain_average": round(float(np.mean(d["avg_gap"])), 6) if d["avg_gap"] else None,
            "agent_row_mean_effective_fanin": round(float(np.mean(d["a_neff"])), 4) if d["a_neff"] else None,
            "agent_row_mean_normalised_entropy": round(float(np.mean(d["a_ent"])), 6) if d["a_ent"] else None,
        }
    ai = np.asarray(agent_indeg)
    return {
        "reimplementation_max_abs_error": fidelity,
        "agent_node_indegree_mean": round(float(ai.mean()), 4) if len(ai) else None,
        "agent_node_indegree_max": int(ai.max()) if len(ai) else None,
        "share_agent_rows_with_zero_incoming_edges":
            round(float((ai == 0).mean()), 6) if len(ai) else None,
        "per_layer": stats,
    }


@torch.no_grad()
def signal_attenuation(actor: GNNMAPPOActor, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How much cross-state variation survives each message-passing layer.

    Measured on the AGENT node row only (that is the row the actor reads).
    relative_state_variation = (mean across-state std) * sqrt(dim) / |mean vector|,
    i.e. 1.0 = the state still moves the vector by its own length, 0.0 = the
    node has collapsed to a constant vector.
    """
    gts = [r["gt"] for r in rows]

    def cv(vals):
        m = np.stack(vals)
        across = float(m.std(axis=0).mean())
        mean_norm = float(np.linalg.norm(m.mean(axis=0)))
        return {
            "across_state_std_mean": round(across, 8),
            "mean_vector_norm": round(mean_norm, 6),
            "relative_state_variation": round(
                across * math.sqrt(m.shape[1]) / (mean_norm + 1e-12), 8),
        }

    H = [actor.encoder.input_proj(gt.node_features) for gt in gts]
    stages = {"input_proj": cv([h[gt.agent_node_indices[0]].numpy() for h, gt in zip(H, gts)])}
    bias = {}
    for li, layer in enumerate(actor.encoder.layers):
        Hn = []
        for h, gt in zip(H, gts):
            o = layer(h, gt.edge_index, gt.edge_features)
            Hn.append(o * gt.node_mask.unsqueeze(-1))
        H = Hn
        stages[f"layer{li}"] = cv([h[gt.agent_node_indices[0]].numpy() for h, gt in zip(H, gts)])
        bias[f"layer{li}_output_proj_bias_norm"] = round(
            float(torch.linalg.norm(layer.output_proj.bias.detach())), 6)
    X = np.stack([gt.node_features.numpy()[gt.agent_node_indices[0]] for gt in gts])
    stages["raw_node_features"] = {
        "across_state_std_mean": round(float(X.std(axis=0).mean()), 8),
        "mean_vector_norm": round(float(np.linalg.norm(X.mean(axis=0))), 6),
        "relative_state_variation": round(
            float(X.std(axis=0).mean()) * math.sqrt(X.shape[1])
            / (float(np.linalg.norm(X.mean(axis=0))) + 1e-12), 8),
    }
    return {"stages": stages, "output_proj_bias_norms": bias}



# ---------------------------------------------------------------------------
# S1: structural audit of the captured graphs
# ---------------------------------------------------------------------------

def reachable(adj: np.ndarray, src: int, hops: int) -> set:
    """Nodes reachable from `src` following edge direction in <= `hops` hops."""
    seen: set = set()
    frontier = {src}
    for _ in range(hops):
        nxt: set = set()
        for u in frontier:
            for v in np.nonzero(adj[u])[0].tolist():
                if int(v) != src and int(v) not in seen:
                    nxt.add(int(v))
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    return seen


def structure_report(z, metas: List[Dict[str, Any]]) -> Dict[str, Any]:
    indeg_by_type: Counter = Counter()
    indeg_hist: Counter = Counter()
    reach = {1: 0, 2: 0, 3: 0}
    own_at_onball: Counter = Counter()
    poss_flag_rows = 0
    ball_info_in = 0
    nonzero_dims: Counter = Counter()
    self_report_rows = 0            # agent rows whose own dim 38 is lit
    ball_rel_rows = 0               # agent rows with ball-relative geometry present
    edge_type_counts: Counter = Counter()
    n_rows = 0
    for s, m in enumerate(metas):
        owner = int(m["owner"])
        for ai in range(len(m["agents"])):
            gt = rebuild_graph(z, f"s{s}a{ai}")
            nf, ei, et = (gt.node_features.numpy(), gt.edge_index.numpy(),
                          gt.edge_type.numpy())
            agent = gt.agent_node_indices[0] if gt.agent_node_indices else 0
            if nf.shape[1] > PLAYER_HAS_POSSESSION_DIM:
                # Post-fix population check: the ball-relative geometry and the
                # engine possession stamp must now be present in real captures.
                if abs(float(nf[agent, PLAYER_HAS_POSSESSION_DIM])) > 1e-9:
                    self_report_rows += 1
                if abs(float(nf[agent, PLAYER_BALL_DISTANCE_DIM])) > 1e-9:
                    ball_rel_rows += 1
            adj = np.zeros((nf.shape[0], nf.shape[0]), dtype=bool)
            for e in range(ei.shape[1]):
                adj[int(ei[0, e]), int(ei[1, e])] = True
                edge_type_counts[EDGE_TYPE_NAMES[int(et[e])]] += 1
            incoming = ei[1] == agent
            d = int(incoming.sum())
            indeg_hist[d] += 1
            n_rows += 1
            for t in et[incoming].tolist():
                indeg_by_type[EDGE_TYPE_NAMES[int(t)]] += 1
                if int(t) in BALL_INFO_EDGE_TYPES:
                    ball_info_in += 1
            balls = [i for i in range(nf.shape[0]) if int(gt.node_type[i]) == NT_BALL]
            for b in balls:
                for k in (1, 2, 3):
                    if agent in reachable(adj, b, k):
                        reach[k] += 1
                        break
                if ai == owner:
                    own_at_onball[["none", "left", "right"][
                        int(np.argmax(nf[b, 7:10]))]] += 1
            pb = (et == 4)
            if ai == owner:
                poss_flag_rows += int((gt.edge_features.numpy()[pb][:, 6] > 0).sum()) \
                    if pb.any() else 0
                for dim in range(nf.shape[1]):
                    if abs(float(nf[agent, dim])) > 1e-9:
                        nonzero_dims[dim] += 1
    return {
        "agent_rows": n_rows,
        "node_feature_width": int(nf.shape[1]) if n_rows else None,
        "agent_rows_self_reporting_possession": self_report_rows,
        "agent_rows_with_ball_relative_features": ball_rel_rows,
        "agent_indegree_histogram": {str(k): v for k, v in sorted(indeg_hist.items())},
        "share_agent_rows_zero_incoming": round(indeg_hist.get(0, 0) / n_rows, 6) if n_rows else None,
        "agent_incoming_edge_types": dict(indeg_by_type),
        "ball_reachable_to_agent_rows": {f"{k}_hop": reach[k] for k in (1, 2, 3)},
        "ball_info_edges_into_agent_rows": ball_info_in,
        "graph_ownership_on_engine_onball_rows": dict(own_at_onball),
        "has_possession_edge_flags_on_owner_rows": poss_flag_rows,
        "edge_type_counts": dict(edge_type_counts),
        "owner_node_nonzero_feature_dims": {str(k): v for k, v in sorted(nonzero_dims.items())},
    }


# ---------------------------------------------------------------------------
# S3/S4/S5: logit engine
# ---------------------------------------------------------------------------

def collect_graph_rows(z, metas) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s, m in enumerate(metas):
        owner = int(m["owner"])
        sidx = int(m["sidx"])
        for ai in range(len(m["agents"])):
            gt = rebuild_graph(z, f"s{s}a{ai}")
            agent = gt.agent_node_indices[0] if gt.agent_node_indices else 0
            rows.append({
                "gt": gt, "agent_node": agent, "agent_idx": ai,
                "is_owner": ai == owner, "tick": sidx, "ep": int(m["ep"]),
                "mask": z["mask"][sidx, ai].astype(bool),
            })
    return rows


def check_node_width(actor: GNNMAPPOActor, width: int, where: str) -> None:
    """Captures and checkpoints must be from the same feature generation.

    A pre-fix capture is 32 dims wide; a post-fix actor expects 39 (and vice
    versa). Say so plainly instead of letting the matmul raise an opaque error.
    """
    expect = int(actor.encoder.input_proj.in_features)
    if width != expect:
        raise SystemExit(
            f"{where}: capture carries {width}-dim node features but the actor "
            f"was built for {expect}. Captures taken before the ball-visibility "
            "fix (NODE_FEATURE_DIM 32 -> 39) are not comparable with post-fix "
            "checkpoints: re-run training/onball_state_capture.py against this "
            "checkpoint before analysing.")


@torch.no_grad()
def gnn_pass(actor: GNNMAPPOActor, rows: List[Dict[str, Any]], mode: str):
    raw, probs, emb, gemb = [], [], [], []
    if rows:
        check_node_width(actor, int(rows[0]["gt"].node_features.shape[1]), "gnn_pass")
    for r in rows:
        gt = variant_graph(r["gt"], mode, r["agent_node"])
        agent_e, global_e = actor.encoder(gt)
        lg = actor.policy_head(agent_e)[0]
        m = torch.from_numpy(r["mask"])
        p = torch.softmax(lg.masked_fill(~m, float("-inf")), dim=-1).numpy()
        raw.append(lg.numpy().astype(np.float64))
        probs.append(p.astype(np.float64))
        emb.append(agent_e[0].numpy().astype(np.float64))
        gemb.append(global_e.numpy().astype(np.float64))
    return (np.stack(raw), np.stack(probs), np.stack(emb), np.stack(gemb))


@torch.no_grad()
def flat_pass(actor: SharedActor, obs: np.ndarray, mask: np.ndarray, mode: str):
    """Forward the flat MLP row-by-row, exposing the penultimate embedding.

    SharedActor.net = [Linear, Tanh, Linear, Tanh, Linear] so the penultimate
    activation is tanh(net[2](tanh(net[0](x)))).
    """
    raw, probs, emb = [], [], []
    for row, m in zip(obs, mask):
        x = torch.from_numpy(flat_variant(row, mode)).float()
        pre = torch.tanh(actor.net[2](torch.tanh(actor.net[0](x))))
        lg = actor.net[4](pre)
        mt = torch.from_numpy(np.asarray(m).astype(bool))
        p = torch.softmax(lg.masked_fill(~mt, float("-inf")), dim=-1).numpy()
        raw.append(lg.numpy().astype(np.float64))
        probs.append(p.astype(np.float64))
        emb.append(pre.numpy().astype(np.float64))
    return np.stack(raw), np.stack(probs), np.stack(emb)



def mean_pairwise_l1(probs: np.ndarray) -> float:
    """Average L1 distance between policy distributions of DIFFERENT states."""
    if probs.shape[0] < 3:
        return float("nan")
    mu = probs.mean(axis=0, keepdims=True)
    return float(2.0 * np.abs(probs - mu).sum(axis=1).mean())


def differentiation(raw: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    on, off = raw[labels == 1], raw[labels == 0]
    if len(on) < 2 or len(off) < 2:
        return {}
    w = on.mean(axis=0) - off.mean(axis=0)
    scores = raw @ w
    between = float(np.linalg.norm(on.mean(0) - off.mean(0)))
    within = float(np.linalg.norm(0.5 * (on.std(0) + off.std(0))))
    return {
        "logit_centroid_distance": round(between, 6),
        "logit_within_group_norm": round(within, 6),
        "logit_fisher_ratio": round(between / within, 6) if within > 1e-12 else None,
        "linear_separability_auc": round(auc_scores(scores, labels), 6),
    }


def paired_contrast(probs: np.ndarray, rows: List[Dict[str, Any]],
                    raw: Optional[np.ndarray] = None,
                    mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Same tick, ball-carrier vs its own teammates: cleanest state-matched test.

    `probs` are masked with each row's OWN legal-action set, so a carrier and a
    non-carrier can differ purely because pass/shot is illegal for the
    non-carrier. To separate that from a genuine policy difference we recompute
    both distributions over the INTERSECTION of legal actions.
    """
    by_tick: Dict[int, Dict[str, List[int]]] = {}
    for i, r in enumerate(rows):
        d = by_tick.setdefault(r["tick"], {"on": [], "off": []})
        d["on" if r["is_owner"] else "off"].append(i)
    l1, agree, n, l1x = [], [], 0, []
    for tick, d in by_tick.items():
        if not d["on"] or not d["off"]:
            continue
        n += 1
        p_on = probs[d["on"]].mean(axis=0)
        p_off = probs[d["off"]].mean(axis=0)
        l1.append(float(np.abs(p_on - p_off).sum()))
        agree.append(int(np.argmax(p_on) == np.argmax(p_off)))
        if raw is not None and mask is not None:
            common = mask[d["on"]].all(axis=0) & mask[d["off"]].all(axis=0)
            if common.sum() >= 2:
                def _renorm(vecs):
                    v = np.exp(vecs[:, common] - vecs[:, common].max(axis=1, keepdims=True))
                    return (v / v.sum(axis=1, keepdims=True)).mean(axis=0)
                a = _renorm(raw[d["on"]])
                b = _renorm(raw[d["off"]])
                l1x.append(float(np.abs(a - b).sum()))
                agree_x = int(int(np.argmax(a)) == int(np.argmax(b)))
            else:
                agree_x = None
        else:
            agree_x = None
    return {
        "paired_ticks": n,
        "mean_l1_owner_vs_teammates": round(float(np.mean(l1)), 6) if l1 else None,
        "owner_teammate_argmax_agreement": round(float(np.mean(agree)), 6) if agree else None,
        "mean_l1_common_legal_actions": round(float(np.mean(l1x)), 6) if l1x else None,
    }



def summarise(raw, probs, labels, emb, rows) -> Dict[str, Any]:
    on = labels == 1
    am = probs.argmax(axis=-1)
    o = {
        "rows": int(len(raw)), "onball_rows": int(on.sum()),
        "onball_argmax_mode_share": None, "onball_argmax_hist": {},
        "onball_mean_pi_pass_shot": None, "offball_mean_pi_pass_shot": None,
    }
    if on.any():
        share, nuniq, hist = mode_share(am[on])
        o["onball_argmax_mode_share"] = round(share, 6)
        o["onball_argmax_n_distinct"] = nuniq
        o["onball_argmax_hist"] = hist
        o["onball_mean_pi_pass_shot"] = round(float(probs[on][:, list(PASS_SHOT)].sum(1).mean()), 6)
        o["onball_mean_pairwise_l1_probs"] = round(mean_pairwise_l1(probs[on]), 6)
        o["onball_logit_within_row_std"] = round(float(raw[on].std(axis=1).mean()), 6)
        o["onball_logit_across_row_std"] = round(float(raw[on].std(axis=0).mean()), 6)
        o["onball_embedding_effective_rank"] = round(effective_rank(emb[on]), 4)
        o["onball_embedding_mean_pairwise_l2"] = round(float(np.mean([
            np.linalg.norm(a - b) for a in emb[on][:40] for b in emb[on][:40]])), 6)
    if (~on).sum() > 1:
        o["offball_mean_pi_pass_shot"] = round(float(probs[~on][:, list(PASS_SHOT)].sum(1).mean()), 6)
        o["offball_argmax_mode_share"] = round(mode_share(am[~on])[0], 6)
    o["onball_vs_offball_logits"] = differentiation(raw, labels)
    masks = np.stack([r["mask"] for r in rows]) if all("mask" in r for r in rows) else None
    o["paired_owner_vs_teammates"] = paired_contrast(probs, rows, raw, masks)
    # Ownership confound audit: if the carrier is always the SAME agent index, then
    # "on-ball vs off-ball" rows differ by a fixed structural property (which player
    # row is marked controlled), not only by possession. Reported so that
    # differentiation metrics are read with the right caveat.
    o["owner_agent_index_histogram"] = dict(Counter(
        int(r["agent_idx"]) for r in rows if r["is_owner"]))
    o["teammate_agent_index_histogram"] = dict(Counter(
        int(r["agent_idx"]) for r in rows if not r["is_owner"]))
    if on.any():
        lp = np.log(np.clip(probs[on].astype(np.float64), 1e-12, None))
        top2 = np.sort(lp, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        o["onball_top1_top2_logit_margin_mean"] = round(float(margin.mean()), 6)
        o["onball_top1_top2_logit_margin_std"] = round(float(margin.std()), 6)
        o["onball_top1_top2_logit_margin_min"] = round(float(margin.min()), 6)
        o["onball_share_rows_margin_gt_0p5"] = round(float((margin > 0.5).mean()), 6)
    return o



def ablation_delta(base_raw, base_probs, alt_raw, alt_probs, labels) -> Dict[str, Any]:
    on = labels == 1
    if not on.any():
        return {}
    dr = np.abs(alt_raw[on] - base_raw[on])
    dp = np.abs(alt_probs[on] - base_probs[on])
    am_b, am_a = base_probs[on].argmax(-1), alt_probs[on].argmax(-1)
    pi_b = base_probs[on][:, list(PASS_SHOT)].sum(1)
    pi_a = alt_probs[on][:, list(PASS_SHOT)].sum(1)
    return {
        "mean_abs_logit_delta": round(float(dr.mean()), 8),
        "max_abs_logit_delta": round(float(dr.max()), 8),
        "mean_abs_prob_delta": round(float(dp.sum(axis=1).mean() / 2.0), 8),
        "argmax_change_rate": round(float((am_a != am_b).mean()), 6),
        "pi_pass_shot_shift_mean": round(float(np.mean(pi_a - pi_b)), 8),
        "pi_pass_shot_shift_abs_mean": round(float(np.mean(np.abs(pi_a - pi_b))), 8),
    }


def head_audit(actor, raw: np.ndarray, probs: np.ndarray, emb: np.ndarray,
               labels: np.ndarray) -> Dict[str, Any]:
    """S5: is the policy head effectively a constant (input-independent) map?"""
    if hasattr(actor, "policy_head"):
        W = actor.policy_head.weight.detach().numpy().astype(np.float64)
        b = actor.policy_head.bias.detach().numpy().astype(np.float64)
    else:
        W = actor.net[4].weight.detach().numpy().astype(np.float64)
        b = actor.net[4].bias.detach().numpy().astype(np.float64)
    on = labels == 1
    proj = emb[on] @ W.T if on.any() else emb @ W.T
    am = probs[on].argmax(axis=-1) if on.any() else probs.argmax(axis=-1)
    locked = int(Counter(am.tolist()).most_common(1)[0][0]) if len(am) else -1
    return {
        "head_weight_frobenius_norm": round(float(np.linalg.norm(W)), 6),
        "head_bias_l2_norm": round(float(np.linalg.norm(b)), 6),
        "head_bias_argmax_action": ACTION_NAMES[int(np.argmax(b))],
        "locked_onball_action": ACTION_NAMES[locked] if locked >= 0 else None,
        "bias_only_matches_locked_action": bool(int(np.argmax(b)) == locked),
        "bias_l2_over_projected_embedding_l2": round(
            float(np.linalg.norm(b)) / (float(np.linalg.norm(proj, axis=1).mean()) + 1e-12), 4),
        "projected_logit_across_row_std": round(float(proj.std(axis=0).mean()), 6),
        "embedding_cosine_to_mean": round(float(np.mean([
            float(v @ proj.mean(0) / (np.linalg.norm(v) * np.linalg.norm(proj.mean(0)) + 1e-12))
            for v in proj])), 6),
    }


def flat_dim_auc(obs: np.ndarray, labels: np.ndarray) -> List[List[Any]]:
    """Which flat observation dimensions separate on-ball from off-ball rows."""
    out = []
    for d in range(obs.shape[1]):
        try:
            a = auc_scores(obs[:, d].astype(np.float64), labels)
        except Exception:
            continue
        if not math.isnan(a):
            out.append([d, round(a, 6), round(float(np.abs(obs[labels == 1, d].mean()
                                                          - obs[labels == 0, d].mean())), 6)])
    out.sort(key=lambda r: -abs(r[1] - 0.5))
    return out[:12]


@torch.no_grad()
def fresh_init_probe(arch: str, rows, obs, mask, labels, n_inits: int) -> Dict[str, Any]:
    """S4: untrained actors of the same architecture on the SAME captured states."""
    recs = []
    for i in range(n_inits):
        torch.manual_seed(1000 + i)
        if arch == "gnn":
            actor = GNNMAPPOActor(action_dim=19, encoder_type="gat", hidden_dim=128).eval()
            r0, p0, e0, _ = gnn_pass(actor, rows, "as_is")
            r1, p1, _, _ = gnn_pass(actor, rows, "ball_gone")
        else:
            actor = SharedActor(obs_dim=obs.shape[1], action_dim=19, hidden=64).eval()
            r0, p0, e0 = flat_pass(actor, obs, mask, "as_is")
            r1, p1, _ = flat_pass(actor, obs, mask, "ball_gone")
        recs.append({
            "seed": 1000 + i,
            "onball_argmax_mode_share": round(mode_share(p0[labels == 1].argmax(-1))[0], 6)
                if (labels == 1).any() else None,
            "onball_argmax_action": ACTION_NAMES[int(Counter(
                p0[labels == 1].argmax(-1).tolist()).most_common(1)[0][0])]
                if (labels == 1).any() else None,
            "onball_mean_pairwise_l1_probs": round(mean_pairwise_l1(p0[labels == 1]), 6)
                if (labels == 1).sum() > 2 else None,
            "onball_mean_pi_pass_shot": round(float(p0[labels == 1][:, list(PASS_SHOT)].sum(1).mean()), 6)
                if (labels == 1).any() else None,
            "onball_logit_across_row_std": round(float(r0[labels == 1].std(axis=0).mean()), 6)
                if (labels == 1).any() else None,
            "ball_gone_mean_abs_logit_delta": ablation_delta(r0, p0, r1, p1, labels).get(
                "mean_abs_logit_delta"),
            "onball_vs_offball_auc": differentiation(r0, labels).get("linear_separability_auc"),
        })
    keys = [k for k in recs[0] if k != "seed"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in recs if isinstance(r[k], (int, float))]
        if vals:
            agg[k] = {"mean": round(float(np.mean(vals)), 6),
                      "min": round(float(np.min(vals)), 6),
                      "max": round(float(np.max(vals)), 6)}
    return {"n_inits": n_inits, "aggregate": agg, "per_init": recs}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

GNN_MODES = ["as_is", "ball_zeroed", "poss_ablated", "ball_gone", "inject_poss"]
FLAT_MODES = ["as_is", "ball_zeroed", "poss_ablated", "ball_gone"]


def flat_tick_rows(z, max_rows: int = 0):
    obs, mask, labels, idx, ticks = [], [], [], [], []
    for t in range(z["obs"].shape[0]):
        na = int(z["n_agents"][t])
        own = int(z["owner"][t])
        for ai in range(na):
            obs.append(z["obs"][t, ai])
            mask.append(z["mask"][t, ai])
            labels.append(1 if ai == own else 0)
            idx.append(ai)
            ticks.append(t)
    obs = np.stack(obs).astype(np.float32)
    mask = np.stack(mask).astype(bool)
    labels = np.asarray(labels, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    ticks = np.asarray(ticks, dtype=np.int64)
    if max_rows and obs.shape[0] > max_rows:
        sel = np.linspace(0, obs.shape[0] - 1, max_rows).round().astype(int)
        obs, mask, labels = obs[sel], mask[sel], labels[sel]
        idx, ticks = idx[sel], ticks[sel]
    return obs, mask, labels, idx, ticks


def analyse_capture(meta: Dict[str, Any], args) -> Dict[str, Any]:
    arch, z = meta["arch"], meta["_npz"]
    metas = meta["graph_stored_meta"]
    out: Dict[str, Any] = {
        "arch": arch, "seed": meta["seed"], "step": meta["step"],
        "checkpoint": os.path.basename(meta["checkpoint"]),
        "timesteps": meta.get("timesteps"), "capture_ticks": meta["n_ticks"],
        "capture_onball_ticks": meta["count_onball"],
        "graph_ticks": meta["graph_ticks_stored"],
    }
    rows = collect_graph_rows(z, metas)
    glabels = np.asarray([1 if r["is_owner"] else 0 for r in rows], dtype=np.int64)
    out["structure"] = structure_report(z, metas)
    if rows:
        out["topology"] = topology_report([r["gt"] for r in rows])

    ckpt = torch.load(meta["checkpoint"], map_location="cpu")
    actor = load_mappo_actor(ckpt)
    actor.eval()

    obs, mask, labels, flat_idx, flat_ticks = flat_tick_rows(z, args.max_flat_rows)

    if arch == "gnn" and rows:
        with torch.no_grad():
            ref = actor.raw_logits([rows[0]["gt"]])[0].numpy()
            man = gnn_pass(actor, rows[:1], "as_is")[0][0]
        out["embedding_path_fidelity"] = round(float(np.abs(ref - man).max()), 9)
        out["attention"] = attention_report(
            actor, [r["gt"] for r in rows][:args.max_attention_graphs])
        out["signal_attenuation"] = signal_attenuation(actor, rows)
        passes = {m: gnn_pass(actor, rows, m) for m in GNN_MODES}
        raw0, prob0, emb0, _ = passes["as_is"]
        out["trained"] = summarise(raw0, prob0, glabels, emb0, rows)
        out["trained"]["head_audit"] = head_audit(actor, raw0, prob0, emb0, glabels)
        out["ablations"] = {
            m: ablation_delta(passes["as_is"][0], passes["as_is"][1],
                              passes[m][0], passes[m][1], glabels)
            for m in GNN_MODES if m != "as_is"}
        out["fresh_init"] = fresh_init_probe("gnn", rows, None, None, glabels, args.fresh_inits)
    else:
        passes = {m: flat_pass(actor, obs, mask, m) for m in FLAT_MODES}
        raw0, prob0, emb0 = passes["as_is"]
        dummy_rows = [{"tick": int(flat_ticks[i]), "is_owner": bool(l),
                       "agent_idx": int(flat_idx[i]), "mask": mask[i]}
                      for i, l in enumerate(labels)]
        out["trained"] = summarise(raw0, prob0, labels, emb0, dummy_rows)
        out["trained"]["head_audit"] = head_audit(actor, raw0, prob0, emb0, labels)
        out["ablations"] = {
            m: ablation_delta(passes["as_is"][0], passes["as_is"][1],
                              passes[m][0], passes[m][1], labels)
            for m in FLAT_MODES if m != "as_is"}
        out["flat_top_dims_by_possession_auc"] = flat_dim_auc(obs, labels)
        out["fresh_init"] = fresh_init_probe("flat", None, obs, mask, labels, args.fresh_inits)
    return out


def write_markdown(runs: List[Dict[str, Any]], heads: List[Dict[str, Any]],
                   path: str, protocol: Dict[str, Any]) -> None:
    """Human-readable report with a data-driven verdict per capture."""
    L: List[str] = []
    A = L.append
    A("# On-ball policy collapse — root-cause analysis\n")
    A("Read-only offline analysis of captured checkpoints. No training was run.\n")
    A(f"- captures: {len(runs)}")
    A(f"- fresh untrained inits probed per capture: {protocol['fresh_inits']}")
    A(f"- capture files: {', '.join(protocol['captures'])}\n")
    A("**Scope note.** This report diagnoses why the `gnn` arm's on-ball actions collapse. "
      "Because the graph observation never routed ball/possession information to the actor's "
      "node (see Verdict), the GNN-vs-flat comparison is **moot as originally posed**: `gnn` "
      "reward/pass/shot numbers produced before the fix describe this defect, not the "
      "architecture, and are not an architecture verdict. Filed as an open item in "
      "`training/results/OPEN_ITEM_GNN_BALL_VISIBILITY.md`; recommendations A, C and D are now "
      "implemented (see 'Status after the ball-visibility fix'), so everything here reads as a "
      "**pre-fix baseline** unless the capture is itself post-fix - the section 1 table prints "
      "each capture's `node feature width` (32 = pre-fix, 39 = post-fix).\n")

    A("## Headline table\n")
    cols = ["arch", "seed", "step", "argmax_mode_share", "mean_pi", "state_sensitivity_l1",
            "ball_gone_dlogit", "poss_ablated_dlogit", "inject_poss_dlogit",
            "fresh_mode_share", "fresh_dlogit", "emb_effective_rank", "locked_action"]
    A("| " + " | ".join(cols) + " |")
    A("|" + "---|" * len(cols))
    for h in heads:
        A("| " + " | ".join(str(h.get(c)) for c in cols) + " |")
    A("")

    A("## 1. Structure — is possession reachable to the agent node?\n")
    A("Captures store the graph observation for BOTH arms (so the same ticks can be replayed), "
      "therefore these graph-side rows also appear for flat; only the gnn rows are load-bearing "
      "for that arm.\n")
    A("| arch | seed | step | node feature width | zero-in-degree agent rows | agent in-degree hist | "
      "incoming edge types | ball reachable (<=3 hops) | engine says ball owned | "
      "`has_possession` edges on owner rows | agent rows self-reporting possession | "
      "agent rows with ball-relative features |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        s = r["structure"]
        A(f"| {r['arch']} | {r['seed']} | {r['step']} | {s.get('node_feature_width')} | "
          f"{s['share_agent_rows_zero_incoming']} | `{s['agent_indegree_histogram']}` | "
          f"`{s['agent_incoming_edge_types']}` | `{s['ball_reachable_to_agent_rows']}` | "
          f"`{s['graph_ownership_on_engine_onball_rows']}` | "
          f"{s['has_possession_edge_flags_on_owner_rows']} | "
          f"{s.get('agent_rows_self_reporting_possession')} / {s['agent_rows']} | "
          f"{s.get('agent_rows_with_ball_relative_features')} / {s['agent_rows']} |")
    A("")

    A("## 1b. Topology — is the graph structure state-dependent at all?\n")
    A("| arch | seed | step | unique edge sets | unique in-degree profiles | agent-row in-degrees | "
      "agent-row in-edges (all rows) | varying edge types | agent rows reaching the ball |")
    A("|---|---|---|---|---|---|---|---|")
    for r in runs:
        t = r.get("topology")
        if not t:
            continue
        A(f"| {r['arch']} | {r['seed']} | {r['step']} | "
          f"{t['unique_edge_sets_across_captured_ticks']} | {t['unique_indegree_profiles']} | "
          f"`{t['agent_row_indegree_values']}` | "
          f"`{t['agent_row_incoming_edge_types_all_rows']}` | `{t['varying_edge_types']}` | "
          f"{t['agent_rows_that_reach_ball']}/{t['agent_rows_total']} |")
    A("")
    for r in runs:
        t = r.get("topology")
        if t:
            A(f"- {r['arch']} seed={r['seed']} step={r['step']}: {t['note']} "
              f"(NEAR-edge endpoints by node type: `{t['near_edge_endpoints_by_node_type']}`)")
    A("")

    A("## 2. Attention — degenerate / uniform?\n")
    for r in runs:
        if "attention" not in r:
            continue
        a = r["attention"]
        A(f"### {r['arch']} seed={r['seed']} step={r['step']} "
          f"(reimplementation fidelity {a['reimplementation_max_abs_error']:.2e})")
        A("| layer | norm. entropy | max weight | |dev from uniform| | effective fan-in | "
          "rel. gap vs plain average |")
        A("|---|---|---|---|---|---|")
        for li, st in a["per_layer"].items():
            A(f"| {li} | {st['mean_normalised_entropy']} | {st['mean_max_weight']} | "
              f"{st['mean_abs_dev_from_uniform']} | {st['mean_effective_fanin']} | "
              f"{st['mean_rel_gap_vs_plain_average']} |")
        A("")
    A("")


    A("## 3. Causal ablations on ball/possession information\n")
    A("| arch | seed | step | mode | mean |dlogit| | max |dlogit| | argmax change rate |")
    A("|---|---|---|---|---|---|---|")
    for r in runs:
        for mode, st in r["ablations"].items():
            A(f"| {r['arch']} | {r['seed']} | {r['step']} | {mode} | "
              f"{st.get('mean_abs_logit_delta')} | {st.get('max_abs_logit_delta')} | "
              f"{st.get('argmax_change_rate')} |")
    A("")

    A("## 4. Signal attenuation across the encoder (agent node only)\n")
    A("`relative_state_variation` = across-state std * sqrt(dim) / |mean vector|: "
      "1.0 = the state still moves the vector by its own length, 0.0 = constant.\n")
    A("| arch | seed | step | stage | rel. state variation | mean vector norm |")
    A("|---|---|---|---|---|---|")
    for r in runs:
        if "signal_attenuation" not in r:
            continue
        for stage, st in r["signal_attenuation"]["stages"].items():
            A(f"| {r['arch']} | {r['seed']} | {r['step']} | {stage} | "
              f"{st['relative_state_variation']} | {st['mean_vector_norm']} |")
    A("")

    A("## 5. Untrained actor probes on the same captured states\n")
    A("| arch | seed | step | fresh mode share | fresh pairwise L1 | fresh ball_gone |dlogit| | "
      "fresh on/off AUC |")
    A("|---|---|---|---|---|---|---|")
    for r in runs:
        fi = r.get("fresh_init", {}).get("aggregate", {})
        A(f"| {r['arch']} | {r['seed']} | {r['step']} | "
          f"{(fi.get('onball_argmax_mode_share') or {}).get('mean')} | "
          f"{(fi.get('onball_mean_pairwise_l1_probs') or {}).get('mean')} | "
          f"{(fi.get('ball_gone_mean_abs_logit_delta') or {}).get('mean')} | "
          f"{(fi.get('onball_vs_offball_auc') or {}).get('mean')} |")
    A("")

    A("## 6. Lock strength on the carrier rows (confound-free)\n")
    A("Owner rows are the same agent index in every captured tick for this drill, so these "
      "statistics are not contaminated by 'which row is which agent'.\n")
    A("| arch | seed | step | lock margin mean | margin std | margin min | share margin > 0.5 nats | "
      "owner index histogram |")
    A("|---|---|---|---|---|---|---|---|")
    for r in runs:
        t = r.get("trained", {})
        A(f"| {r['arch']} | {r['seed']} | {r['step']} | "
          f"{t.get('onball_top1_top2_logit_margin_mean')} | "
          f"{t.get('onball_top1_top2_logit_margin_std')} | "
          f"{t.get('onball_top1_top2_logit_margin_min')} | "
          f"{t.get('onball_share_rows_margin_gt_0p5')} | "
          f"`{t.get('owner_agent_index_histogram')}` |")
    A("")

    A("## 7. Confounds that must be kept in mind\n")
    for r in runs:
        t = r.get("trained", {})
        hist = t.get("owner_agent_index_histogram") or {}
        if len(hist) == 1:
            A(f"- {r['arch']} seed={r['seed']} step={r['step']}: the ball carrier is ALWAYS "
              f"agent index {list(hist)[0]} in this drill, so `on_off_*` and `paired_*` metrics "
              "compare agent 0 against agents 1/2, not possession-vs-no-possession with the "
              "agent index held fixed. Treat them as 'does the actor react to the row's "
              "identity', not as evidence of possession awareness.")
    A("- The `ball_*`/`poss_*` ablations and the fresh-init probes are NOT affected by that "
      "confound: they compare the same rows before and after destroying ball information, and "
      "they hold for every agent index.")
    A("")

    A("## Verdict\n")
    for r in runs:
        s, ab = r["structure"], r["ablations"]
        fi = r.get("fresh_init", {}).get("aggregate", {})
        t = r["trained"]
        parts = []
        if r["arch"] == "gnn":
            if s["ball_reachable_to_agent_rows"] and all(
                    v == 0 for v in s["ball_reachable_to_agent_rows"].values()):
                parts.append("ball node is NOT reachable to the agent within encoder depth")
            if s["share_agent_rows_zero_incoming"] > 0:
                parts.append(f"{s['share_agent_rows_zero_incoming']:.2f} of agent rows have ZERO "
                             "incoming edges (attention bypassed entirely)")
            if s["has_possession_edge_flags_on_owner_rows"] == 0:
                parts.append("no `has_possession` edge flag is ever set on the carrier")
        else:
            parts.append("flat observation carries the ball pose directly "
                         "(`obs[88:97]`), so the graph findings below do not apply to it")
        bite = lambda k: abs(ab.get(k, {}).get("mean_abs_logit_delta") or 0.0)
        if bite("ball_gone") == 0.0 and bite("ball_zeroed") == 0.0:
            parts.append("zeroing ALL ball state/edges changes logits by EXACTLY 0.0 "
                         "(representation is provably ball-independent)")
        if bite("poss_ablated") == 0.0:
            parts.append("removing possession flags/edges changes logits by EXACTLY 0.0")
        if bite("inject_poss") > 1e-4:
            parts.append(f"injecting a possession bit into the agent's own node DOES move logits "
                         f"(mean |dlogit| {bite('inject_poss'):.4f}): the head can read it, the "
                         "input path cannot deliver it")
        if (fi.get("onball_argmax_mode_share") or {}).get("mean", 0) >= 0.7:
            parts.append(f"untrained fresh actors are already "
                         f"{(fi['onball_argmax_mode_share']['mean']):.2f} mode-locked on the same "
                         "states (an initialisation prior, not a learned optimum)")
        er = t.get("onball_embedding_effective_rank")
        if er and er < 2:
            parts.append(f"on-ball embeddings collapse to effective rank {er} "
                         "(a single direction)")
        A(f"- **{r['arch']} seed={r['seed']} step={r['step']}** — argmax mode share "
          f"{t.get('onball_argmax_mode_share')}, locked action "
          f"`{t.get('head_audit', {}).get('locked_onball_action')}`. " + "; ".join(parts) + ".")

    def _mean_ball_bite(arch: str) -> Optional[float]:
        vals = [abs((r["ablations"].get("ball_gone") or {}).get("mean_abs_logit_delta") or 0.0)
                for r in runs if r["arch"] == arch]
        return round(float(np.mean(vals)), 8) if vals else None

    def _mean_ball_flip(arch: str) -> Optional[float]:
        vals = [(r["ablations"].get("ball_gone") or {}).get("argmax_change_rate") or 0.0
                for r in runs if r["arch"] == arch]
        return round(float(np.mean(vals)), 6) if vals else None

    g_bite, f_bite = _mean_ball_bite("gnn"), _mean_ball_bite("flat")
    A("")
    A("### Cross-arm contrast (the decisive test)\n")
    A(f"- mean carrier-row |Δlogit| when every ball state/edge is destroyed: "
      f"**gnn {g_bite}** vs **flat {f_bite}**")
    A(f"- mean share of carrier-row argmaxes that flip under the same ablation: "
      f"**gnn {_mean_ball_flip('gnn')}** vs **flat {_mean_ball_flip('flat')}**")
    A("- an untrained actor of each arm shows the same split (gnn 0.0 vs flat ~0.11), so this is "
      "a property of the observation/encoder, not of the weights.")
    A("- together with the structural row above (no ball edge ever terminates at an agent row, "
      "ball unreachable at any depth), the collapse is explained by **missing ball/possession "
      "information in the graph observation for the agent node**, not by attention degeneracy or "
      "by training dynamics.")
    A("")
    A("")
    A("## Mechanism: where ball information is lost, in code\n")
    A("1. `training/gnn_graph_to_tensor.py::_encode_player_node` (lines 101-143) builds a player "
      "node from position, velocity, role one-hot, `is_active`/`is_controlled`/`is_goalkeeper`, "
      "and team-shape scalars (`nearest_teammate_dist`, `nearest_opponent_dist`, `team_width`, "
      "`team_depth`, `compactness`, `stretch`, `receiver_availability`, `line_id`, `lane_id`, "
      "`team_index`). There is **no ball position, no ball distance, no possession flag, and no "
      "goal-relative geometry** anywhere in the 32-dim player vector (dims 30-31 are unused, "
      "which is exactly where `inject_poss` writes).")
    A("2. `PLAYER_BALL` edges are emitted player→ball and carry `has_possession` at edge-feature "
      "dim 6 (`_encode_player_ball_edge`, line 308). Possession can therefore only ever arrive at "
      "the BALL node - and the captured graphs show the flag is never set "
      "(`has_possession_edge_flags_on_owner_rows` = 0), while `POSSESSES` edges are not emitted "
      "at all and `_encode_possesses_edge` (line 296) returns an all-zero feature vector anyway.")
    A("3. `training/gnn_encoders.py::GATLayer` adds edge features to the **attention logits** "
      "(`logits = (Q[target]·K[source])·scale + edge_proj(edge_features)`) while the message "
      "value is `V[source]`. An edge flag can therefore only re-weight messages; it can never be "
      "read as a feature. With possession sitting on an edge that points into the ball, no "
      "re-weighting can move it to the actor.")
    A("4. The ball and the goals are **sinks**: `BALL_GOAL` is ball→goal and `PLAYER_GOAL` is "
      "player→goal, and goal nodes have zero outgoing edges. There is hence no directed path "
      "ball→player at ANY depth: measured ball reachability to the agent row is 0/180 rows at "
      "1, 2 and 3 hops, and zeroing the ball node plus every ball-touching edge changes the "
      "logits by exactly 0.0.")
    A("5. `TEAMMATE` edges form a fixed star/DAG (0→1, 0→2, 1→2 within the left team), which is "
      "why the agent rows have in-degree 0/1/2 and `agent_row_incoming_edge_types_all_rows` "
      "contains `TEAMMATE` only. The carrier's row (node 0) receives **no messages at all**: its "
      "embedding is three LayerNorms over a residual stream that only ever contains its own "
      "projected features.")
    A("6. Consequence: the actor's input cannot express 'where is the ball', 'do I have it', or "
      "'where is the goal'. The reachable behaviour is a fixed action ordering; training can only "
      "sharpen the margin on that ordering (observed: margin grows while entropy stays ~2.2-2.4 "
      "nats of 2.94 max, i.e. the policy stays diffuse but its tiny argmax bias is constant).")
    A("7. Why it looks like an initialization artefact rather than a training pathology: "
      "randomly initialised actors on the same states are already 80%+ mode-locked and equally "
      "ball-blind (`fresh_mode_share`, `fresh_dlogit` = 0.0), the collapse is present in the "
      "first captured milestone, and each seed locks onto a different constant action - a fixed "
      "ordering chosen by the random head, not a learned one.")
    A("")
    A("## Status after the ball-visibility fix (A, C and D implemented)\n")
    A("- **A is in.** The player node vector is 39 wide; dims 32-38 carry `ball_rel_x`, "
      "`ball_rel_y`, `ball_distance`, `goal_rel_x`, `goal_rel_y`, `is_nearest_to_ball`, "
      "`has_possession`, normalised with the same pitch constants as the edge encoders. "
      "`NODE_FEATURE_DIM` in `training/gnn_graph_to_tensor.py` is the single source of truth "
      "for the tensorizer, `checkpoint_contract`, `gnn_onnx.py` and `GnnOnnxPolicy.ts`.")
    A("- **C is in.** `has_possession` is stamped on the carrier's node from the engine's "
      "`ground_truth.current_ball_owner.agent_id` inside `gnn_graph_builder`, and only when "
      "that id exists in the player map - it is never inferred from proximity.")
    A("- **D is in.** `gnn_encoders._fuse_global` concatenates the masked-mean pooled global "
      "embedding onto every agent embedding before `agent_head`, so ball and goal node "
      "features reach the actor even though those nodes are still graph sinks. `training/"
      "tests/test_gnn_ball_visibility.py` pins this: a ball change that leaves every player "
      "row untouched (height/speed only) produces a bit-identical pre-fix agent row but a "
      "different post-fix one.")
    A("- Dims 30-31 are still spare, so `inject_poss` keeps writing dim 30 and that column "
      "remains comparable across generations.")
    A("- The ablations were widened to match the new feature layout: `poss_ablated` also "
      "clears player dim 38, `ball_gone` also clears player dims 32-34 and 37 (it still "
      "leaves possession alone, so `ball_gone` and `poss_ablated` keep measuring different "
      "things). Both branches no-op on 32-wide captures.")
    A("- **Offline verification (2026-09-29):** `pytest training/tests` = 389 passed, 1 skipped, "
      "0 failed (four stale 32-dim assumptions in `tests/test_gnn_policy_integration.py` had to be "
      "widened first); `npx tsc --noEmit` clean; `tests/test_gnn_onnx.py` confirms the 39-wide "
      "ONNX export still matches PyTorch for mlp/gat/geometry. On a synthetic post-fix graph a "
      "fresh-init actor moves under every control (`ball_gone` 0.109, `poss_ablated` 0.177, "
      "`inject_poss` 0.064 mean |dlogit|, lofted-ball 0.007 through D alone), and both widened "
      "ablations are verified no-ops on a 32-wide tensor - so a pre-fix row and a post-fix row "
      "cannot quietly mean different things.")
    A("- **Still required: a live post-fix capture.** The trained-checkpoint half of the acceptance "
      "test (items 1-4: `ball_gone` |dlogit| > 0, fresh-init mode share well below 1.0, pass/shot "
      "mass above the 0.21 uniform baseline) cannot be answered offline. Re-running this script "
      "against a pre-fix capture with a post-fix checkpoint is refused by `check_node_width`, so "
      "each table row must come from a capture/actor pair of the same generation.")
    A("- **Not done: recommendation B** (mirrored ball->player and goal->player edges). The "
      "structural rows above will keep reporting `ball reachable = 0` at every depth; that is "
      "expected and no longer blocks the actor because of D. Acceptance for the fix is the "
      "fresh-init sweep plus `ball_gone`/`poss_ablated` |dlogit| > 0 on post-fix captures.\n")
    A("## Recommended fixes (ranked by leverage per line of change)\n")
    A("A. **Compute ball/goal-relative features locally in `graph_to_tensors`** and append them to "
      "the player node vector (positions of both the player and the ball are already in the node "
      "features, so this needs no Unity rebuild): `ball_rel_x`, `ball_rel_y`, `ball_distance`, "
      "`goal_rel_x`, `goal_rel_y`, `is_nearest_to_ball`, `has_possession`. Keep `NODE_FEATURE_DIM` "
      "in sync (`input_proj` takes `NODE_FEATURE_DIM`).")
    A("B. **Reverse-augment the ball/goal edges** in the tensorizer: for each `PLAYER_BALL` edge "
      "add the mirrored ball→player edge, and for each `PLAYER_GOAL` edge add goal→player, so the "
      "ball and goal node features actually flow into the players.")
    A("C. **Set the possession flag properly** (env side, or as a node feature per A) so that the "
      "carrier's own node says it is the carrier; today `has_possession` is never set on any "
      "captured tick.")
    A("D. **One-line architecture fallback**: fuse the pooled `global_emb` (which already contains "
      "the ball and goal node features through masked mean pooling) into the per-agent embedding "
      "before the policy head, giving the actor ball visibility while A-C are implemented.")
    A("E. **Re-run the 4-seed sweep after the fix** and use these same probes as the acceptance "
      "test: `ball_gone` |Δlogit| > 0, fresh-init mode share well below 1.0, and pass/shot mass "
      "above the 4/19 ≈ 0.21 uniform baseline on carrier rows.")
    A("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def headline(r: Dict[str, Any]) -> Dict[str, Any]:
    t = r.get("trained", {})
    ab = r.get("ablations", {})
    fa = t.get("head_audit", {})
    fi = r.get("fresh_init", {}).get("aggregate", {})
    return {
        "arch": r["arch"], "seed": r["seed"], "step": r["step"],
        "onball_rows": t.get("onball_rows"),
        "argmax_mode_share": t.get("onball_argmax_mode_share"),
        "mean_pi": t.get("onball_mean_pi_pass_shot"),
        "state_sensitivity_l1": t.get("onball_mean_pairwise_l1_probs"),
        "on_off_auc": (t.get("onball_vs_offball_logits") or {}).get("linear_separability_auc"),
        "paired_l1": (t.get("paired_owner_vs_teammates") or {}).get("mean_l1_owner_vs_teammates"),
        "paired_argmax_agree": (t.get("paired_owner_vs_teammates") or {}).get("owner_teammate_argmax_agreement"),
        "ball_gone_dlogit": (ab.get("ball_gone") or {}).get("mean_abs_logit_delta"),
        "poss_ablated_dlogit": (ab.get("poss_ablated") or {}).get("mean_abs_logit_delta"),
        "inject_poss_dlogit": (ab.get("inject_poss") or {}).get("mean_abs_logit_delta"),
        "emb_effective_rank": t.get("onball_embedding_effective_rank"),
        "bias_argmax": fa.get("head_bias_argmax_action"),
        "locked_action": fa.get("locked_onball_action"),
        "fresh_mode_share": (fi.get("onball_argmax_mode_share") or {}).get("mean"),
        "fresh_dlogit": (fi.get("ball_gone_mean_abs_logit_delta") or {}).get("mean"),
        "margin_mean": t.get("onball_top1_top2_logit_margin_mean"),
        "margin_std": t.get("onball_top1_top2_logit_margin_std"),
        "margin_min": t.get("onball_top1_top2_logit_margin_min"),
        "owner_idx_hist": json.dumps(t.get("owner_agent_index_histogram")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rootcause-dir", default="runs/gnnflat_seq_20260927_210021/rootcause")
    ap.add_argument("--pattern", default="cap_*.json")
    ap.add_argument("--only", default="", help="substring filter on capture file names")
    ap.add_argument("--fresh-inits", type=int, default=8)
    ap.add_argument("--max-attention-graphs", type=int, default=24)
    ap.add_argument("--max-flat-rows", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.rootcause_dir, args.pattern)))
    if args.only:
        paths = [p for p in paths if args.only in os.path.basename(p)]
    if not paths:
        raise SystemExit(f"no captures matched {args.pattern} in {args.rootcause_dir}")
    runs = [analyse_capture(load_capture(p), args) for p in paths]
    out_path = args.out or os.path.join(args.rootcause_dir, "rootcause_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"protocol": {"fresh_inits": args.fresh_inits,
                                "max_flat_rows": args.max_flat_rows,
                                "captures": [os.path.basename(p) for p in paths]},
                   "runs": runs}, fh, indent=2)

    import csv as _csv
    heads = [headline(r) for r in runs]
    csv_path = os.path.splitext(out_path)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(heads[0].keys()))
        w.writeheader()
        for h in heads:
            w.writerow(h)

    md_path = os.path.splitext(out_path)[0] + ".md"
    write_markdown(runs, heads, md_path, {"fresh_inits": args.fresh_inits,
                                          "captures": [os.path.basename(p) for p in paths]})

    print("== headline (per capture) ==")
    for h in heads:
        print(json.dumps(h, default=str))

    print("\n== structure (per capture) ==")
    for r in runs:
        s = r["structure"]
        print(f"{r['arch']} seed={r['seed']} step={r['step']}: "
              f"zero_indeg_share={s['share_agent_rows_zero_incoming']} "
              f"indeg_hist={s['agent_indegree_histogram']} "
              f"incoming_types={s['agent_incoming_edge_types']} "
              f"ball_reachable={s['ball_reachable_to_agent_rows']} "
              f"ownership_on_onball={s['graph_ownership_on_engine_onball_rows']} "
              f"poss_edge_flags={s['has_possession_edge_flags_on_owner_rows']} "
              f"edge_types={s['edge_type_counts']}")

    print("\n== topology (is the graph structure state-dependent?) ==")
    for r in runs:
        if "topology" in r:
            print(f"{r['arch']} seed={r['seed']} step={r['step']}: "
                  + json.dumps(r["topology"]))

    print("\n== attention (gnn captures) ==")
    for r in runs:
        if "attention" in r:
            a = r["attention"]
            print(f"{r['arch']} seed={r['seed']}: reimpl_fidelity="
                  f"{a['reimplementation_max_abs_error']:.2e} "
                  f"zero_indeg_agent_rows={a['share_agent_rows_with_zero_incoming_edges']}")
            for li, st in a["per_layer"].items():
                print(f"    {li}: {st}")

    print("\n== ablations (on-ball rows) ==")
    for r in runs:
        print(f"{r['arch']} seed={r['step'] and r['seed']}: " + json.dumps(r["ablations"]))

    print("\n== fresh init (untrained, same states) ==")
    for r in runs:
        if "fresh_init" in r:
            print(f"{r['arch']} seed={r['seed']}: "
                  + json.dumps(r["fresh_init"]["aggregate"]))

    print("\n== head audit ==")
    for r in runs:
        print(f"{r['arch']} seed={r['seed']}: " + json.dumps(r["trained"]["head_audit"]))

    print(f"\nWROTE {out_path}\nWROTE {csv_path}")


if __name__ == "__main__":
    main()









