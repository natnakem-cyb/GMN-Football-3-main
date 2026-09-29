"""Read-only probe: does ball information reach the POOLED (critic) path but not
the per-agent node path the ACTOR reads?

The GNN encoder exposes two readouts:
  * `agent_emb = agent_head(h[agent_node_indices])` - what the policy head sees
  * `global_emb = global_head(masked_mean_pool(h))`  - what the critic head sees

Masked mean pooling touches every node, including BALL and GOAL, so the pooled
path *can* carry ball geometry even when the agent node cannot reach it. This
script quantifies that asymmetry on real captured states: destroy the ball (node
features + every ball-touching edge) and measure how much each readout moves.

**Status after the ball-visibility fix (2026-09-29).** The asymmetry below was a
pre-fix finding (agent delta exactly 0.0 on all 14 captures while the pooled
delta was 0.068-0.103). Fix D now concatenates the pooled global embedding onto
every agent row before the head, so on a *post-fix* checkpoint
`ball_gone_agent_emb_max_delta` is expected to be non-zero and the two readouts
should move together — that is the point of the probe now: it is the regression
check that D closes the gap, not a demonstration of the defect. Consequently
`pooled_over_agent_ratio` reports `null` whenever the agent delta is non-zero,
i.e. `null` is the healthy post-fix value, not a failure. Pre-fix captures
cannot be probed with post-fix checkpoints: the checkpoint contract rejects the
32/39 `node_feature_dim` mismatch, so re-capture first.

No training, no weight updates, no Unity bridge.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.checkpoint_contract import load_mappo_actor          # noqa: E402
from training.onball_rootcause_analysis import (                   # noqa: E402
    collect_graph_rows, load_capture, variant_graph,
)


@torch.no_grad()
def probe(meta: Dict[str, Any], max_rows: int) -> Dict[str, Any]:
    z = meta["_npz"]
    rows = collect_graph_rows(z, meta["graph_stored_meta"])[:max_rows]
    actor = load_mappo_actor(torch.load(meta["checkpoint"], map_location="cpu"))
    actor.eval()
    enc = actor.encoder

    d_agent, d_global, d_logit = [], [], []
    for r in rows:
        a0, g0 = enc(r["gt"])
        destroyed = variant_graph(r["gt"], "ball_gone", r["agent_node"])
        a1, g1 = enc(destroyed)
        d_agent.append(float((a1 - a0).abs().max()))
        d_global.append(float((g1 - g0).abs().max()))
        l0 = actor.policy_head(a0)[0]
        l1 = actor.policy_head(a1)[0]
        d_logit.append(float((l1 - l0).abs().max()))
    return {
        "arch": meta["arch"], "seed": meta["seed"], "step": meta["step"],
        "rows": len(rows),
        "ball_gone_agent_emb_max_delta": round(float(np.max(d_agent)), 9),
        "ball_gone_pooled_emb_max_delta": round(float(np.max(d_global)), 9),
        "ball_gone_logit_max_delta": round(float(np.max(d_logit)), 9),
        "pooled_over_agent_ratio": (round(float(np.max(d_global)) /
                                         max(float(np.max(d_agent)), 1e-12), 6)
                                    if np.max(d_agent) == 0 else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rootcause-dir", default="runs/gnnflat_seq_20260927_210021/rootcause")
    ap.add_argument("--pattern", default="cap_gnn_*.json")
    ap.add_argument("--max-rows", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.rootcause_dir, args.pattern)))
    results = [probe(load_capture(p), args.max_rows) for p in paths]
    for r in results:
        print(json.dumps(r))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
