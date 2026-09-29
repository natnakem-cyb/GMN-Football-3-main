"""Root-cause state capture for the GNN on-ball collapse (read-only, no training).

The milestone sweep established that the GNN arm locks onto ONE action for every
on-ball tick and gets worse with training. This script does NOT test that again;
it captures the *states* needed to explain WHY, so every downstream measurement
(attention structure, on-ball vs off-ball logit contrast, possession ablation,
fresh-initialisation control) can run offline against one capture instead of
re-rolling the Unity bridge.

What is written per checkpoint (one .npz + one .json sidecar):
  * EVERY tick: flat observation rows (A x 127), legal-action masks (A x 19),
    ball-owner index, episode/tick index. Full-resolution record used for the
    on-ball/off-ball logit contrast.
  * A SUBSAMPLE of ticks (on-ball ticks first, then evenly strided off-ball
    ticks): the tensorised graph of every controlled agent (node features, node
    types, edge_index, edge types, edge features, node mask, graph context,
    agent node index). This is what makes graph-reachability, attention and
    feature-ablation analysis possible without Unity.

Methodology matches training/probe_ckpt_onball.py: same scenario, same
BASE_SEED=500000 ladder, same pre-step owner semantics
(env._last_ball_owner_agent_idx read BEFORE env.step), same masking, same
deterministic argmax rollout. Only the episode count is smaller (3 by default),
because the goal is ~120 captured ticks, not a statistical rate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.checkpoint_contract import load_mappo_actor
from training.mappo_rollout import unwrap_obs, unwrap_masks, _mask_matrix
from training.gnn_graph_to_tensor import graph_to_tensors
from training.eval_f_act import sha256_of

BASE_SEED = 500000
SCENARIO = "academy_3_vs_1_with_keeper_onball"
PASS_SHOT = (9, 10, 11, 12)


def graph_arrays(gt) -> Dict[str, np.ndarray]:
    """Flatten one GraphTensor into storable numpy arrays."""
    return {
        "nf": gt.node_features.detach().cpu().numpy().astype(np.float32),
        "nt": gt.node_type.detach().cpu().numpy().astype(np.int16),
        "nm": gt.node_mask.detach().cpu().numpy().astype(np.float32),
        "ei": gt.edge_index.detach().cpu().numpy().astype(np.int16),
        "et": gt.edge_type.detach().cpu().numpy().astype(np.int16),
        "ef": gt.edge_features.detach().cpu().numpy().astype(np.float32),
        "ctx": (np.zeros(0, dtype=np.float32) if gt.graph_context is None
                else gt.graph_context.detach().cpu().numpy().astype(np.float32)),
        "ani": np.array([int(i) for i in gt.agent_node_indices], dtype=np.int16),
    }



def policy_actions(actor, lo: np.ndarray, mm, infos, agents) -> np.ndarray:
    """Deterministic masked-argmax action per agent, identical to the probes."""
    with torch.no_grad():
        mt = torch.tensor(np.asarray(mm, dtype=bool), dtype=torch.bool)
        if bool(getattr(actor, "requires_graph_observations", False)):
            gr = [infos[a]["graph_observation"] for a in agents]
            dist = actor(gr, mt)
        else:
            dist = actor(torch.from_numpy(lo).float(), mt)
        return dist.logits.argmax(dim=-1).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arch", default="gnn")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--bridge-port", type=int, default=6400)
    ap.add_argument("--max-onball-ticks", type=int, default=60)
    ap.add_argument("--max-offball-ticks", type=int, default=60)
    ap.add_argument("--off-stride", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    actor = load_mappo_actor(ckpt)
    actor.eval()
    ckpt_sha = sha256_of(args.checkpoint)

    # Graph observations are captured for BOTH arms so the same tick set can be
    # fed to either architecture offline.
    env = GMNMultiAgentEnv(scenario=SCENARIO, auto_start_bridge=True,
                           port=args.bridge_port, include_graph_observations=True)
    controllable = list(env.possible_agents)

    obs_rows: List[np.ndarray] = []
    mask_rows: List[np.ndarray] = []
    owner_col: List[int] = []
    ep_col: List[int] = []
    tick_col: List[int] = []
    nagents_col: List[int] = []

    stored: Dict[str, np.ndarray] = {}
    stored_meta: List[Dict[str, Any]] = []
    n_onball = 0
    n_sel = 0
    keep_on = 0
    keep_off = 0
    tick_ptr = 0
    graph_missing = 0

    try:
        for ep in range(args.num_episodes):
            ep_seed = BASE_SEED + ep * 1009
            obs_dict, infos = env.reset(seed=ep_seed)
            cur_masks = unwrap_masks(obs_dict)
            obs_dict = unwrap_obs(obs_dict)
            tick = 0
            while True:
                agents = list(env.agents if env.agents else controllable)
                lo = np.stack([obs_dict[a] for a in agents], axis=0).astype(np.float32)
                mm = np.asarray(_mask_matrix(cur_masks, agents), dtype=np.uint8)
                owner = int(getattr(env, "_last_ball_owner_agent_idx", 255))
                on_ball = 0 <= owner < len(agents)

                obs_rows.append(lo)
                mask_rows.append(mm)
                owner_col.append(owner)
                ep_col.append(ep)
                tick_col.append(tick)
                nagents_col.append(len(agents))

                want_graph = ((on_ball and keep_on < args.max_onball_ticks)
                              or (not on_ball and tick % args.off_stride == 0
                                  and keep_off < args.max_offball_ticks))
                if want_graph:
                    ok = True
                    for ai, a in enumerate(agents):
                        gobs = (infos.get(a) or {}).get("graph_observation")
                        if gobs is None:
                            ok = False
                            graph_missing += 1
                            break
                        gt = graph_to_tensors(gobs)
                        for key, arr in graph_arrays(gt).items():
                            stored[f"s{len(stored_meta)}a{ai}_{key}"] = arr
                    if ok:
                        stored_meta.append({"sidx": tick_ptr, "ep": ep, "tick": tick,
                                            "owner": owner, "agents": list(agents)})
                        if on_ball:
                            keep_on += 1
                        else:
                            keep_off += 1

                acts = policy_actions(actor, lo, mm, infos, agents)
                if on_ball:
                    n_onball += 1
                    if int(acts[owner]) in PASS_SHOT:
                        n_sel += 1
                ad = {a: int(acts[i]) for i, a in enumerate(agents)}

                obs_dict, rewards, terms, truncs, infos = env.step(ad)
                cur_masks = unwrap_masks(obs_dict)
                obs_dict = unwrap_obs(obs_dict)
                tick += 1
                tick_ptr += 1
                done = (any(terms.values()) if terms else False) or \
                       (any(truncs.values()) if truncs else False) or (not env.agents)
                if done:
                    break
    finally:
        env.close()

    meta = {
        "arch": args.arch, "seed": args.seed, "step": args.step,
        "checkpoint": args.checkpoint, "sha256": ckpt_sha,
        "timesteps": ckpt.get("timesteps"),
        "scenario": SCENARIO, "base_seed": BASE_SEED,
        "episodes": args.num_episodes,
        "n_ticks": tick_ptr, "count_onball": n_onball,
        "count_selected_pass_shot_onball": n_sel,
        "graph_ticks_stored": len(stored_meta),
        "graph_stored_meta": stored_meta,
        "graph_missing": graph_missing, "off_stride": args.off_stride,
        "created_by": "training/onball_state_capture.py",
    }
    npz = os.path.splitext(args.out)[0] + ".npz"
    tmp = npz + ".tmp.npz"
    payload = dict(stored)
    payload["obs"] = np.stack(obs_rows).astype(np.float32)
    payload["mask"] = np.stack(mask_rows).astype(np.uint8)
    payload["owner"] = np.asarray(owner_col, dtype=np.int16)
    payload["ep"] = np.asarray(ep_col, dtype=np.int16)
    payload["tick"] = np.asarray(tick_col, dtype=np.int16)
    payload["n_agents"] = np.asarray(nagents_col, dtype=np.int16)
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, npz)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps({k: meta[k] for k in ("arch", "seed", "step", "n_ticks",
                                           "count_onball",
                                           "count_selected_pass_shot_onball",
                                           "graph_ticks_stored",
                                           "graph_missing")}, indent=2))


if __name__ == "__main__":
    main()

