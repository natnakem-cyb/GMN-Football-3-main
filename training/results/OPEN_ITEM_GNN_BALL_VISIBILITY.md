# Open Item — GNN Actor Node Cannot See the Ball or Possession

**Opened:** 2026-09-28
**Status:** Fixed in code — fixes A, C and D implemented 2026-09-29 (section 11). Fix B (reverse ball→player / goal→player edges) deliberately **not** implemented; D covers the gap. **Training-side acceptance is still open** (section 7): it needs fresh post-fix captures from a live run, which no offline test can substitute.
**Severity:** High for the `gnn` arm — a hard ceiling on learnable on-ball behaviour. Inert for the `flat` arm.
**Evidence:** `runs/gnnflat_seq_20260927_210021/rootcause/rootcause_report.md` (also `.json`, `.csv`), `pooled_probe.json`
**Tools:** `training/onball_state_capture.py`, `training/onball_rootcause_analysis.py`, `training/onball_pooled_probe.py`, `training/rootcause_capture_all.ps1`

---

## 1. Read this before quoting any GNN-vs-flat comparison number

This item is filed **separately** from the GNN-vs-flat reward comparison, and must not be folded into it as one of its conclusions.

**The GNN-vs-flat comparison is moot as originally posed.** While the graph observation never routes ball or possession information to the actor's node, a `gnn` run does not measure "does message passing help on this task". It measures "what does message passing do when the graph omits the single variable that decides the action". Every reward, pass-rate, shot-rate, or entropy figure from the `gnn` arm describes **this defect**, not the architecture. Those numbers are not evidence for or against GNNs here, and should not be read as an architecture verdict.

Conversely: every `gnn` number captured before 2026-09-29 was produced by the blind architecture described below and keeps that caveat permanently — it is a **baseline**, not an architecture verdict. Post-fix checkpoints are a different input generation (`node_feature_dim` 39 vs 32) and must not be tabulated against pre-fix numbers; `onball_rootcause_analysis.py` now refuses to load a capture and an actor from different generations rather than let the shapes mismatch.

---

## 2. Defect

> **Reading note:** this section and every number in sections 3-5 describe the graph **as captured on 2026-09-28**, i.e. the pre-fix architecture. Fixes A and C have since grown the player vector to 39 dims and added exactly the ball/goal/possession terms named below, so the line references and the "32-dim" width are historical. They are kept verbatim because they are the evidence for the change, not a description of the current tree (section 11).

`training/gnn_graph_to_tensor.py::_encode_player_node` (lines 101-143) builds the 32-dim player node vector from: position, velocity, role one-hot, `is_active`/`is_controlled`/`is_goalkeeper`, and team-shape scalars (`nearest_teammate_dist`, `nearest_opponent_dist`, `team_width`, `team_depth`, `compactness`, `stretch`, `receiver_availability`, `line_id`, `lane_id`, `team_index`).

There is **no ball position, no ball distance, no possession flag, and no goal-relative geometry** in the player vector. Dims 30-31 are unused.

Because the graph is the actor's only input, the actor node carries no ball information — and, as shown below, no *edge* can deliver it either.

---

## 3. Evidence (all measured, read-only, 14 checkpoint captures)

Captures: `gnn` and `flat` × seed 42 at 2560/5120/7680/10240 steps, plus seeds 7/999/123 at 10240. 3 episodes each, up to 60 on-ball + 60 off-ball graph ticks per capture.

| Test | `gnn` (7 checkpoints) | `flat` (7 checkpoints) |
|---|---|---|
| Zero the ball node **and every ball-touching edge**, carrier-row mean \|Δlogit\| | **exactly 0.0** | **0.080** (0.014-0.113) |
| Same ablation, carrier-row argmax flip share | **0.0** | **0.141** (0.0-0.273) |
| Same ablation on **untrained** actors of each arm | **exactly 0.0** | 0.013-0.120 |
| Ablate possession flags/edges, mean \|Δlogit\| | **exactly 0.0** | 0.005-0.014 |
| **Positive control:** write one possession bit into the agent's *own* node (`inject_poss`) | mean \|Δlogit\| **0.092-0.109**, max 0.24-0.38, argmax flips 0-57% of rows | n/a |
| Ball reachability to the agent row (1/2/3 hops) | **0/180, 0/189, 0/210, 0/62-67 rows** — never reachable | (graph-side rows; not the flat arm's input) |
| Agent-row incoming edge types | `TEAMMATE` only | — |
| Agent rows with **zero** incoming edges | 0.333 (a third of rows, incl. the carrier's) | — |
| `has_possession` edge flags set on carrier rows | **0** in all 14 captures | — |
| Agent-row effective fan-in | 0.93-1.23 | — |
| Normalised attention entropy / max weight / deviation from uniform | 0.62-0.87 / 0.42-0.65 / 0.089-0.223 | — |
| Reimplementation fidelity vs `GATLayer` | **0.00e+00** (bit-identical) | — |
| Critic-side check (`pooled_probe.json`): ball destroyed | `agent_emb` Δ **0.0** vs `pooled_emb` Δ **0.068-0.103** | — |

Two controls make this airtight:

1. **Attention is not degenerate.** The offline reimplementation reproduces `GATLayer` bit-identically (error 0.00e+00), so the attention numbers are trustworthy: normalised entropy 0.62-0.87, max weight 0.42-0.65, deviation from uniform 0.089-0.223, all-node effective fan-in 2.71-3.53 of 3-6 neighbours. Attention works; the rows the actor reads have almost nothing in them.
2. **The head can use possession.** Injecting a single scalar into the agent's own node moves logits by 0.092-0.109 (max 0.24-0.38). So this is a data-pipeline problem, not a capacity problem. (Correction to an earlier verbal summary: the argmax flip rate under `inject_poss` is 0-57% of carrier rows, not a flat 27% — it is 57% on the most diffuse checkpoint (seed 42 @2560) and 0% where the lock margin already exceeds the perturbation. The logit movement, not the flip rate, is the load-bearing part of this control.)

Also, no attenuation inside the encoder: `relative_state_variation` on the agent node is ~0.34 at the raw features and 0.48-0.73 through all three layers. The signal was never attenuated because it was never there.

---

## 4. Mechanism — where the information is lost, in code

1. **Player nodes never contain the ball.** `_encode_player_node` has no ball/goal/possession term (section 2). `NODE_FEATURE_DIM` is 32 with dims 30-31 unused.
2. **Possession lives only on edges pointing the wrong way.** `PLAYER_BALL` edges are emitted player→ball and carry `has_possession` at edge-feature dim 6 (`_encode_player_ball_edge`, line 308). Possession can therefore only ever arrive at the **ball** node — and the captures show the flag is never set (0 flags on carrier rows on every tick). `POSSESSES` edges are never emitted, and `_encode_possesses_edge` (line 296) returns an all-zero feature vector anyway.
3. **GAT cannot promote an edge flag into a node feature.** `training/gnn_encoders.py::GATLayer` computes `logits = (Q[target]·K[source])·scale + edge_proj(edge_features)` and messages `V[source]`. Edge features re-weight messages; they cannot be read as content. With possession sitting on an edge that points *into the ball*, no amount of re-weighting can move it to the actor.
4. **Ball and goals are sinks.** `BALL_GOAL` is ball→goal, `PLAYER_GOAL` is player→goal, and goal nodes have zero outgoing edges. There is no directed path ball→player at any depth: measured reachability to the agent row is 0 rows at 1, 2 and 3 hops.
5. **The teammate set is a fixed star.** `TEAMMATE` edges are 0→1, 0→2, 1→2 within the left team, so agent rows have in-degree 0/1/2, incoming type `TEAMMATE` only, and unique edge-set count 1-2 across all ticks (the only variation being proximity `NEAR` edges, which are PLAYER→PLAYER). The carrier's row (node 0) receives **no messages at all**: its embedding is three LayerNorms over a residual stream containing only its own projected features.

**Consequence:** the actor's input cannot express "where is the ball", "do I have it", or "where is the goal". The reachable behaviour set is a fixed action ordering, and training can only sharpen the margin on that ordering.

---

## 5. Why this is not a training or hyperparameter issue

- **Lock present from the first checkpoint, sharpening, never escaping.** Seed 42 mode share: 0.567 (2560) → 0.733 (5120) → 0.733 (7680) → **1.000** (10240). Mean π(pass/shot) over legal carrier rows falls 0.141 → 0.066 while pass/shot actions are **100% legal** on every carrier row.
- **Entropy is not collapsed.** 2.67 → 2.22 nats against a maximum of ln 19 = 2.94. The policy stays diffuse; a constant logit ordering is simply being sharpened (margin 0.18 → 0.58 for seed 42; 0.97-1.43 for seeds 7/999).
- **Untrained actors already behave this way.** Fresh random inits are 75-89% mode-locked *and* exactly ball-blind (Δ = 0.0) on the same states — an initialisation prior over a blind input, not a learned optimum.
- **Different seeds lock onto different constants** (`RELEASE_DIRECTION`, `DOWN`, `UP_RIGHT`, `DOWN_RIGHT`, `IDLE`) — same totality, different random ordering.

**Therefore:** more training steps (e.g. a Phase 2A continuation) would spend hours sharpening a lock that is already proven to be input-driven. Do not run it before the fix.

---

## 6. Recommended fix — A + D together, as the starting point of a future task

**Implemented 2026-09-29: A, C and D. Not implemented: B** (the reverse-edge augmentation; D covers the same gap without a schema change, and B remains the cheaper next step if a post-fix capture shows residual ball-blindness).
These two changes were treated as one work unit, as recommended. The implementation record is section 11.

**A. Give the player node local ball/goal geometry (root cause, no Unity rebuild needed).**
In `graph_to_tensors`, both the player position and the ball position are already present in node features, so relative features can be computed offline: `ball_rel_x`, `ball_rel_y`, `ball_distance`, `goal_rel_x`, `goal_rel_y`, `is_nearest_to_ball`, `has_possession`. Append them to the player vector and keep `NODE_FEATURE_DIM` in sync (`input_proj` takes `NODE_FEATURE_DIM`), including the ONNX export path.

**D. Fuse the pooled global embedding into the per-agent embedding (cheap insurance, no schema change).**
`global_emb` is a masked mean pool over all nodes, so it already contains the ball and goal node features — the critic-side probe confirms it carries ball information (Δ 0.068-0.103 under ball destruction) while the agent readout does not (Δ 0.0). Fusing it into the per-agent embedding before the policy head gives the actor ball visibility even if A is incomplete, and it does not touch the graph schema at all.

Why both: A fixes the cause, D is defense in depth that keeps the actor from being blind if any single ball-relative feature is mis-populated. Do not ship A alone as a "one-line fix without a test" — the acceptance test below is the gate.

Optional follow-ups, lower priority than A+D: reverse-augment `PLAYER_BALL` (ball→player) and `PLAYER_GOAL` (goal→player) edges so ball/goal node features actually flow into players; set the `has_possession` flag properly env-side instead of relying on the node feature.

---

## 7. Acceptance test (reuse these probes as the gate)

After A+D land, re-run the same 4-seed sweep and re-run this analysis. The fix is accepted only if, on carrier rows:

1. `ball_gone` |Δlogit| > 0 on every checkpoint and every seed (today: exactly 0.0).
2. Fresh-init mode share well below 1.0 (today `gnn` 0.75-0.89).
3. Pass/shot probability mass above the 4/19 ≈ 0.21 uniform baseline on carrier rows (today 0.066-0.141 while all pass/shot actions are legal).
4. The positive control still holds: `inject_poss` continues to move logits.
5. Structural: ball reachability to the agent row is non-zero at ≥1 hop, or the ball-relative features are present at dim 32+.

**Status of each item (2026-09-29, offline evidence only):** item 5 is satisfied by construction — dims 32-38 exist and are populated, covered by `training/tests/test_gnn_ball_visibility.py`. Items 1-4 still need a post-fix capture. What offline work does establish: on a synthetic post-fix graph a fresh-init actor moves under every control (`ball_gone` mean |Δlogit| 0.109, `poss_ablated` 0.177, `inject_poss` 0.064), so the pipeline is now capable of carrying the signal — which is exactly what the pre-fix table could not say (0.0 on all 14 checkpoints). It says nothing about a *trained* checkpoint's mode share or pass/shot mass, so it is a pre-flight check, not an acceptance.

---

## 8. Reproduction (all read-only)

```powershell
# 14 captures, parallel, no training
powershell -ExecutionPolicy Bypass -File training\rootcause_capture_all.ps1

# full analysis -> rootcause_report.json / .csv / .md
.\.venv\Scripts\python.exe training\onball_rootcause_analysis.py --fresh-inits 8 `
  --out runs\gnnflat_seq_20260927_210021\rootcause\rootcause_report.json

# critic-path vs actor-path ball information
.\.venv\Scripts\python.exe training\onball_pooled_probe.py --max-rows 30 `
  --out runs\gnnflat_seq_20260927_210021\rootcause\pooled_probe.json
```

Captures and the generated report live under the gitignored `runs\gnnflat_seq_20260927_210021\rootcause\` directory, so they are not tracked; the scripts above regenerate them.

---

## 9. Limitations / confounds

- **Carrier identity confound.** In these drills the carrier is always agent index 0, so `on_off_*` and `paired_*` metrics compare agent indices rather than possession states (which is why `gnn` shows on/off AUC 0.97-1.00 while `flat` sits at 0.57-0.97). The `ball_*`/`poss_*` ablations and the fresh-init probes compare the *same* rows with and without ball information and are immune to this confound.
- **One scenario family, three episodes per capture.** The defect is a schema-level property (`_encode_player_node` and the edge directions are scenario-independent), so it is not expected to be scenario-specific, but this was not measured across scenarios.
- **Value/critic side not diagnosed.** The probe establishes that the critic's pooled input carries ball information; whether the critic's value function uses it well was not measured.

---

## 10. Non-goals (explicit)

- No fix in the session that produced this item.
- No Phase 2A training continuation.
- No change to `NODE_FEATURE_DIM`, edge schema, or ONNX export under this item.
- No verdict on GNN-vs-flat architecture quality (see section 1).

_(These were the boundaries of the diagnosis session that opened the item. The follow-up task lifted the `NODE_FEATURE_DIM` restriction — see section 11 — and deliberately left the edge schema and Phase 2A untouched.)_

---

## 11. Implementation record — what shipped on 2026-09-29

| Fix | Landed? | Where |
|---|---|---|
| A — ball/goal geometry in the player node | **Yes** | `training/gnn_graph_to_tensor.py`: `NODE_FEATURE_DIM` 32 → 39; `_encode_player_node` appends dims 32-38 |
| B — reverse ball→player / goal→player edges | No (deferred) | edge schema untouched; `ball_reachable_to_agent_rows` stays 0 by design |
| C — possession stamped from ground truth | **Yes** | `training/gnn_graph_builder.py`: owned-player flag driven by `current_ball_owner` / `ball.owner`, with `has_possession` kept as a fallback |
| D — pooled global embedding fused into every agent row | **Yes** | `training/gnn_encoders.py`: `_fuse_global` before the head; `agent_head` widened to `2 * hidden` in all three encoders |

**New node feature dims** (`training/gnn_graph_to_tensor.py`, single source of truth; `checkpoint_contract.GNN_NODE_FEATURE_DIM` and `src/agents/GnnOnnxPolicy.ts::NODE_FEATURE_DIM` must stay in sync):

| Dim | Name | Meaning |
|---|---|---|
| 32 | `ball_rel_x` | ball x minus player x, divided by pitch length |
| 33 | `ball_rel_y` | ball y minus player y, divided by pitch width |
| 34 | `ball_distance` | Euclidean ball distance, normalised by `DISTANCE_NORM` |
| 35 | `goal_rel_x` | attacking-goal relative x |
| 36 | `goal_rel_y` | attacking-goal relative y |
| 37 | `is_nearest_to_ball` | 1.0 for the player closest to the ball |
| 38 | `has_possession` | ground-truth possession stamp (Fix C) |

Dims **30-31 are intentionally left spare**, so the `inject_poss` positive control stays an out-of-distribution perturbation rather than an existing feature.

**Consequences for the toolchain**
- `checkpoint_contract.GNN_NODE_FEATURE_DIM = 39`: pre-fix (32-wide) GNN checkpoints are now *rejected* with a named `node_feature_dim` mismatch instead of crashing in a matmul. That is what makes the pre-fix captures in section 3 a frozen baseline.
- `onball_rootcause_analysis.py`: the diagnostic table carries a `node feature width` column (32 = pre-fix, 39 = post-fix); `check_node_width` aborts if a capture and an actor come from different generations; `ball_gone` now also clears the agent-agnostic ball geometry dims (32-34, 37) and `poss_ablated` also clears dim 38, so the two columns keep their original meaning across both generations.
- `onball_state_capture.py` needs no change: it stores `gt.node_features` verbatim, so new captures are 39-wide automatically.
- ONNX: `training/gnn_onnx.py` and `src/agents/GnnOnnxPolicy.ts` both take the width from the shared constant; export/PyTorch parity is asserted for `mlp`, `gat` and `geometry` in `training/tests/test_gnn_onnx.py`.

**Offline verification (this session)**
- New `training/tests/test_gnn_ball_visibility.py`: known-answer geometry for every new dim (hand-computed from the fixture pitch), ground-truth possession stamping, and the Fix D property that a ball differing only in height — invisible to Fix A — still changes the agent embedding.
- GNN subset (`ball_visibility`, `graph_builder`, `phase4`, `gat_attention_stability`, `onnx`, `encoding_hygiene`): 103 passed.
- Full `pytest training/tests`: 389 passed, 1 skipped, 0 failed. Four failures surfaced and were fixed; all four were stale 32-dim assumptions in `training/tests/test_gnn_policy_integration.py`'s synthetic graph helper, not architecture problems.
- `npx tsc --noEmit`: clean.
- Ablation controls through a fresh-init `GNNMAPPOActor(mlp)` on a synthetic post-fix graph: `ball_gone` |Δlogit| 0.109, `poss_ablated` 0.177, `inject_poss` 0.064, lofted-ball 0.007 — and the same two ablations are no-ops on a 32-wide tensor, confirming the widening cannot silently change meaning between generations.

**What is *not* verified** (see section 7): live post-fix captures, trained-checkpoint mode share, and pass/shot probability mass. Until those run, this item stays open.

