# CLOSE-OUT REPORT — SMALL-N CAVEATS, CANONICAL SCOPE TABLE, COMMIT/PUSH

**Date:** 2026-09-21  
**HEAD (before):** `472b92de6e768756f62669af58add3fce48b1091`  
**HEAD (after commit):** `2d4b6dd32a58c7ab0e6617ef8f3e36d65ac39ff5`  
**Pushed to origin/main:** yes

---

## TASK 1 — SMALL-N CAVEATS

**Section added to:** `BASE_SEED_AND_OCCUPANCY_RECONCILIATION.md` (Section 3.1.1, immediately after the canonical occupancy table)

**Per-seed 1-frame percentage-point shift:**
- Seed 42: n=13 → 1 frame = **7.69 pp**
- Seed 123: n=96 → 1 frame = 1.04 pp
- Seed 7: n=18 → 1 frame = **5.56 pp**
- Seed 999: n=42 → 1 frame = 2.38 pp

**Seeds flagged as most fragile:** 42, 7

---

## TASK 2 — CANONICAL SCOPE RECONCILIATION TABLE

**Tolerance used and justification:** ±0.05 percentage points. Pre-stated tolerance accounting for minor floating-point and eval-script variance between the two measurement paths. Not tight enough to force a match label; all four seeds reconcile within a much tighter margin (≤0.005 pp).

| Seed | n_decisions | n_pass_shot | canonical_rate | rebuilt_rate | delta_pp | match |
|------|-------------|-------------|----------------|--------------|----------|-------|
| 42 | 7650 | 57 | 0.7451% | 0.75% | −0.0049 pp | ✅ |
| 123 | 7650 | 128 | 1.6732% | 1.67% | +0.0032 pp | ✅ |
| 7 | 7650 | 66 | 0.8627% | 0.86% | +0.0027 pp | ✅ |
| 999 | 7650 | 79 | 1.0327% | 1.03% | +0.0027 pp | ✅ |

**Any seed NOT matching:** None. All four seeds reconcile within tolerance.

---

## TASK 3 — COMMIT / PUSH / FRESH-CLONE VERIFICATION

**Files committed (full list):**
- `training/eval_canonical_three_agent_measurement.py` — canonical measurement script (DEFAULT_BASE_SEED corrected to 500000)
- `training/generate_final_report.py` — gate report generator
- `training/results/BASE_SEED_AND_OCCUPANCY_RECONCILIATION.md` — updated reconciliation document
- `training/results/post_reweight_logit_prestep_reconciled_detail.json` — 30,600-frame detail JSON (Git LFS, 326 MB)
- `training/results/post_reweight_logit_prestep_reconciled_summary.csv` — per-seed aggregates
- `training/results/post_reweight_canonical_scope_reconciliation.csv` — 7650-decision reconciliation
- `training/verify_canonical_artifacts.py` — independent artifact verifier

**Ambiguous untracked files resolved:**
- `runs/*` — deleted (untracked scratch run directories from smoke/full test runs)
- `training/compute_event_stats.py` — deleted (standalone scratch script)
- `training/results/post_reweight_logit_prestep_reconciled_detail_seed999.json` — deleted (seed999 variant scratch artifact)
- `training/results/post_reweight_logit_prestep_reconciled_detail_smoke.json` — deleted (smoke variant scratch artifact)
- Partial files (`*_partial.json`, `*_partial.csv`) — deleted after full run completed

**Pushed:** yes

**Fresh-clone verification:**
- Commit SHA matches: **yes** (`2d4b6dd`)
- All files present, non-zero size: **yes**
- `verify_canonical_artifacts.py` passes in fresh clone: **yes** (13/13 checks)
- `verify_prestep_measurement.py` passes in fresh clone: **yes** (all 11 check categories across 4 seeds)

---

## TASK 4 — SUPERSESSION STATEMENT

**Superseded reports/commits:**
- `5908073` (`ONBALL_OCCUPANCY_CHECK.md`, `onball_occupancy_summary.csv`) — reason: used defective post-step ball-ownership measurement (`_last_ball_owner_agent_idx == 0`) and wrong base_seed=700000. Both method and seed are incorrect under the canonical contract.
- `085ec85` (`ONBALL_OCCUPANCY_PRESTEP_CHECK.md`, `onball_occupancy_prestep_summary.csv`) — reason: corrected measurement method but still used base_seed=700000, conflicting with `CANONICAL_METRICS_CONTRACT.md` §2.1 (base_seed=500000).
- `085ec85` (`post_reweight_logit_prestep_summary.csv`, `post_reweight_logit_prestep_detail.json`) — reason: corrected method but wrong base_seed=700000. The `n_onball` values (42=74, 123=121, 7=91, 999=92) are not comparable to canonical values.
- "Final Measurement Gate" report (`generate_final_report.py` output) — reason: used base_seed=500000 but was generated from artifacts that mixed wrong-base_seed data and did not include the canonical scope reconciliation table with explicit tolerance. This document replaces it.

**This report is now the canonical reference for:** occupancy, π-floor, and scope-reconciliation numbers under base_seed=500000.

---

## CONFIRMATIONS

- No training performed: **yes**
- No reward/GAE/mask/network/environment changes: **yes**
- No parameter changed to force a match in Task 2: **yes** (tolerance pre-stated at ±0.05 pp; all seeds passed within ≤0.005 pp)
- No new interpretation beyond Task 2's direct result: **yes**
- No files left in ambiguous staged/untracked state: **yes**

---

## FILES WRITTEN / COMMITTED

| File | Status |
|------|--------|
| `training/eval_canonical_three_agent_measurement.py` | Committed |
| `training/generate_final_report.py` | Committed |
| `training/results/BASE_SEED_AND_OCCUPANCY_RECONCILIATION.md` | Committed |
| `training/results/post_reweight_logit_prestep_reconciled_detail.json` | Committed (LFS) |
| `training/results/post_reweight_logit_prestep_reconciled_summary.csv` | Committed |
| `training/results/post_reweight_canonical_scope_reconciliation.csv` | Committed |
| `training/verify_canonical_artifacts.py` | Committed |

**Deleted (scratch):**
- `runs/` — 14 untracked scratch run directories
- `training/compute_event_stats.py` — standalone scratch script
- `training/results/post_reweight_logit_prestep_reconciled_detail_seed999.json` — variant scratch
- `training/results/post_reweight_logit_prestep_reconciled_detail_smoke.json` — variant scratch
- `*_partial.json`, `*_partial.csv` — intermediate measurement artifacts

**COMMIT:** `2d4b6dd32a58c7ab0e6617ef8f3e36d65ac39ff5`
