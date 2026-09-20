# RETEST_PROVENANCE — Actor-Loss Reweight Corrected-Gate Retest

**Date:** 2026-09-20  
**HEAD (before):** `880091ddf0805dc23213d9e433efeed95905f839`  
**Scenario:** `academy_3_vs_1_with_keeper_onball`

---

## A0 — Working Tree Truth

**Git state:**
- Branch `main`, up to date with `origin/main`.
- Uncommitted changes: `training/mappo_update.py`, `training/train_mappo.py`, run-directory manifests/CSVs, and retest artifacts.
- HEAD is still `880091d`; gate fix is not yet committed.

**Loss-path gate diff (vs 880091d):**

```text
-            pass_shot_mask = batch_masks[:, 9:13].any(dim=-1)  # indices 9,10,11,12
             selected_action = actions_t[batch_idx]
+            is_pass_shot_selected = (selected_action >= 9) & (selected_action <= 12)
             selected_legal = batch_masks.gather(1, selected_action.unsqueeze(-1)).squeeze(-1)
-            is_pass_shot = pass_shot_mask & selected_legal
+            is_pass_shot = is_pass_shot_selected & selected_legal
```

**A0 result:** PASS — optimizer path uses `selected ∈ {9..12} ∧ selected_legal`. Old `pass_shot_mask.any()` gate is gone from the loss block. A standalone sanity test (`training/tests/test_actor_reweight_gate.py`) passes.

---

## A1 — Checkpoint Inventory

**Scope:** 40 trajectory-table rows (seeds 42/123/7/999 × steps 5k/10k/15k/20k/25k/30k/35k/40k/45k/final).

**Findings:**
- All 40 retest checkpoints exist on disk under `training/models/`.
- Naming convention: `mappo_academy_3_vs_1_with_keeper_onball_seed{seed}_actorreweight.pt` (final) and `..._actorreweight_{step}.pt` (intermediate).
- No `*_actorreweight_50000.pt` files exist; the "final" step maps to the no-step quarantine file.
- Every checkpoint hash is distinct from every other checkpoint hash.
- The original 880091d final checkpoint for seed 42 (`ed2edb4b...`) is **not** present on disk; it was overwritten by the retest quarantine file.
- Seeds 123/7/999 have no original counterparts in the current tree (original 880091d only ran seed 42).

**CSV:** `training/results/retest_checkpoint_inventory.csv` (40 rows).

**A1 result:** PASS — all trajectory rows map to existing files; labeled set is `retest` throughout. Original 880091d originals were overwritten.

---

## A2 — Path Binding to Trajectory Table

**Investigation:**
The retest report (`ACTOR_LOSS_REWEIGHT_RETEST.md`) cites a trajectory table. The only eval summary CSV present at the time the report was written was `training/results/actor_reweight_eval_summary.csv` (mtime `2026-09-20 12:40:47 PM`).

**Critical finding:** That CSV was written **before** the retest training began (retest training ran 14:44–15:00). Its final-step values match the original 880091d eval rows in `eval_progress.csv`:

| Seed | actor_reweight_eval_summary.csv (12:40 PM) | Original 880091d eval_progress row | Retest eval_progress row (14:44) |
|------|-------------------------------------------|-----------------------------------|--------------------------------|
| 42 | mean_reward `-0.7368`, PASS+SHOT `0.76%` | hash `ed2edb4b...`, mean `-0.7368` | hash `eb9e1403...`, mean `-0.7690` |
| 123 | mean_reward `-0.7095`, PASS+SHOT `1.62%` | hash `127179b3...`, mean `-0.7095` | hash `72e56e38...`, mean `-0.7348` |
| 7 | mean_reward `-0.6873`, PASS+SHOT `0.00%` | hash `577c6df7...`, mean `-0.6873` | hash `7b29a3a4...`, mean `-0.7542` |
| 999 | mean_reward `-0.3354`, PASS+SHOT `2.34%` | hash `60fc82db...`, mean `-0.3354` | hash `0365a1d9...`, mean `-0.7459` |

**Conclusion:** The trajectory table in the original retest report was bound to the **original 880091d hashes** via a stale CSV, not to the retest hashes.

**A2 result:** FAIL — retest report trajectory table binds to 880091d eval data, not retest eval data.

---

## A3 — Force-Path / Cache Audit

**Eval script:** `training/eval_actor_reweight.py`
- Uses `evaluate_checkpoint_progress(..., force_reeval=True)`.
- Default output: `training/results/actor_reweight_eval_summary.csv`.

**Cache behavior:** `evaluate_checkpoint_progress` skips re-evaluation only when `force_reeval=False` and an exact `evaluation_id` match exists. With `force_reeval=True`, it always recomputes and writes to `win_rate_progress.csv` (global) and returns the row.

**Provenance of trajectory numbers:**
- The 12:40 PM `actor_reweight_eval_summary.csv` mtime predates retest training.
- No record of a post-retest run of `eval_actor_reweight.py` writing to that path was found in the working tree or git history.
- The retest eval did run (14:44–14:49) and wrote rows to per-run `eval_progress.csv` and to `win_rate_progress.csv`, but **not** to `actor_reweight_eval_summary.csv`.
- No wrapper calling eval without `force_reeval=True` was found; the failure is stale-CSV copy-paste, not cache replay.

**A3 result:** FAIL — the exact producing command for the reported Phase 2 numbers cannot be recovered for the retest; the only CSV with that shape was written before retest training.

---

## A4 — Independent Spot-Check (seed 999 final)

**Rebuild:** All 40 retest checkpoints were re-evaluated with `force_reeval=True` using `eval_actor_reweight.py`. Output written to `training/results/actor_reweight_retest_eval_summary.csv`.

**Rebuilt trajectory (PASS+SHOT%, from fresh eval):**

| Seed | 5k | 10k | 15k | 20k | 25k | 30k | 35k | 40k | 45k | 50k |
|------|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 42 | 0.75 | 0.75 | 0.75 | 0.75 | 0.78 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 123 | 1.02 | 0.78 | 1.52 | 1.28 | 1.42 | 1.76 | 1.82 | 1.42 | 1.65 | 1.67 |
| 7 | 0.80 | 0.76 | 0.82 | 1.05 | 0.80 | 0.90 | 0.85 | 0.99 | 0.86 | 0.86 |
| 999 | 0.75 | 0.73 | 0.75 | 0.73 | 0.73 | 1.02 | 1.03 | 1.03 | 1.03 | 1.03 |

**Standalone seed 999 final:**
- Path: `training/models/mappo_academy_3_vs_1_with_keeper_onball_seed999_actorreweight.pt`
- SHA256: `0365a1d95f9a31f9517bdd42401aa3f4a4bfed3e713be610ae91bcf95d526097`
- Standalone PASS+SHOT%: **1.03%**
- Rebuilt trajectory table value: **1.03%**
- Difference: **0.00 pp** (within ±0.15 pp tolerance)

**Original (stale) table claim:** 2.34%  
**Verdict:** The stale table was wrong. The rebuilt trajectory matches the standalone eval.

**A4 result:** PASS (after rebuild). The original table claim of 2.34% is rejected.

---

## A5 — Share-Log Consistency

**Files:** `training/logs/actor_loss_reweight_share_seed{42,123,7,999}_academy_3_vs_1_with_keeper_onball.csv`  
**Mtime window:** 2026-09-20 14:41–14:57 (retest training window)  
**Columns:** `step`, `update`, `surrogate_loss_share_M`, `surrogate_loss_share_M1`  
**Means (M=500 / M=1):**
- Seed 42: 41.2% / 0.5%
- Seed 123: 28.3% / 0.4%
- Seed 7: 43.6% / 0.6%
- Seed 999: 26.1% / 0.3%

**Predicate match:** The share computation in `mappo_update.py` uses `is_pass_shot_selected = (selected_action >= 9) & (selected_action <= 12)`, identical to the loss gate. The old `pass_shot_mask.any()` gate is absent.

**A5 result:** PASS — share logs fall in retest window, have correct columns, means match claimed 26–44% / 0.3–0.6% band, and predicate matches loss gate.

---

## A6 — Commit and Push

**Status:** Not yet committed or pushed. See Phase B branch decision before final commit.

**Planned commit contents:**
- Gate fix in `training/mappo_update.py`
- `training/tests/test_actor_reweight_gate.py`
- Share-log CSVs
- `training/results/retest_checkpoint_inventory.csv`
- `training/results/RETEST_PROVENANCE.md`
- `training/results/actor_reweight_retest_eval_summary.csv` (fresh rebuilt trajectory)
- Updated `training/results/ACTOR_LOSS_REWEIGHT_RETEST.md` with corrected numbers and provenance failure note

---

## Phase B Branch

**A2 and A3 failed.** The retest behavioral table was built from a stale CSV (`actor_reweight_eval_summary.csv`, mtime 12:40 PM) written before retest training, not from fresh retest eval data. This is a **copy-paste / wrong-CSV** failure.

**Branch: B1 — Retest behavioural table untrusted / incomplete provenance**

**Failure mode:** `wrong CSV / copy-paste`

**Next step:** Fix binding by using `actor_reweight_retest_eval_summary.csv` (fresh eval) and update the report. Do not proceed to basin forensics until the report is corrected and A4/A5 pass on the rebuilt data.

**Note:** The rebuilt trajectory (from fresh eval) shows markedly different behavior from the stale table:
- Seed 999 final: **1.03%** (not 2.34%)
- Seed 42: flat ~0.75% across all steps (no Phase 1 bump)
- Seed 123: 1.67% at 50k (not 1.62%, and baseline is 1.96% → reversion confirmed)
- Seed 7: 0.86% at 50k (not 0.00%)

This means the corrected-gate training did **not** produce the behavioral improvements claimed in the original retest report. The "starvation-relief real" verdict is unsupported by the verified retest eval data.

---

## Confirmation Checklist

- A0 Loss-path gate correct: **yes**
- A1 Inventory: **PASS**
- A2 Path binding: **FAIL** (stale CSV / wrong hashes)
- A3 Force-path / CSV source: **FAIL** (producing command cannot be recovered for retest)
- A4 Spot-check seed999: **PASS** (after rebuild; standalone 1.03% = rebuilt 1.03%)
- A5 Share logs: **PASS** (predicate matches, means in band)
- A6 Pushed: **no** (pending)

- No new multi-seed train campaign in Phase A: **yes**
- No reward/GAE/mask/network redesign: **yes**
- 880091d findings not overwritten: **yes** (correction section only)
