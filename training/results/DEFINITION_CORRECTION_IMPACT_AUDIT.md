# DEFINITION-CORRECTION IMPACT AUDIT

**Date:** 2026-09-21  
**HEAD:** `d4ed87099a0897102e59f163b27663ced3b45695`  
**Purpose:** Determine whether `d4ed870`'s canonical on-ball definition correction was a pure documentation fix (Branch A) or a silent second redefinition requiring recomputation (Branch B).

---

## TASK 1 — CODE VS DOCUMENTATION CHECK

**File/function inspected:** `training/eval_canonical_three_agent_measurement.py`, `_build_agent_decision_frame`, lines 255–257

**Exact retention/on-ball logic found:**
```python
team_has_ball = bool(float(obs_arr[OBS_L_TEAM_OWNERSHIP_INDEX]) == 1.0)
agent_has_ball = bool(pre_step_ball_owner_agent_idx == agent_index)
onball = agent_has_ball
```

**Definition implemented:** individual-carrier (`pre_step_ball_owner_agent_idx == agent_index`)

**d4ed870 documentation claims:** individual-carrier (`pre_step_ball_owner_agent_idx == agent_index`)

**Branch:** **A (match, doc-only fix)**

**Evidence:**

1. The frame-retention logic in `collect_canonical_measurement` (lines 609–633) does **not** filter frames based on on-ball status at all. It records **ALL** agent decisions unconditionally. The `onball` flag is a **derived per-frame property**, not a retention condition.

2. The `onball` flag is computed in `_build_agent_decision_frame` as `agent_has_ball`, which is `pre_step_ball_owner_agent_idx == agent_index` — the individual-carrier definition. This code was present in the original `2d4b6dd` commit and was **not changed** in `d4ed870`.

3. `git diff 2d4b6dd..d4ed870 -- training/eval_canonical_three_agent_measurement.py` shows only:
   - Docstring wording changes
   - Tolerance constant additions
   - No functional changes to `_build_agent_decision_frame` or any on-ball logic

4. The JSON schema written by the script (line 868) already encoded `"onball_source": "pre_step_ball_owner_agent_idx"` in `2d4b6dd`. The artifact schema and the code always matched; only the human-readable markdown documentation incorrectly described the method as team-level `obs[95]`.

**Cross-check:** The canonical `n_onball` values committed at `2d4b6dd` (13/96/18/42) and the scope-reconciliation table (deltas ≤0.005 pp) were computed using the individual-carrier logic. No numbers change.

---

## TASK 2 — BRANCH RESOLUTION

**Branch A confirmed.** No recomputation needed.

- **Canonical `n_onball` table (from `2d4b6dd`):** unchanged
  - Seed 42: **13**
  - Seed 123: **96**
  - Seed 7: **18**
  - Seed 999: **42**

- **Canonical scope-reconciliation table (from `2d4b6dd`):** unchanged
  - All 4 seeds match `actor_reweight_retest_eval_summary.csv` within ±0.05 pp (actual deltas ≤0.005 pp)

- **What was actually fixed:** The prose in `BASE_SEED_AND_OCCUPANCY_RECONCILIATION.md` incorrectly stated "Pre-step obs[95] team possession" as the measurement method. The actual code always used `pre_step_ball_owner_agent_idx == agent_index`. `d4ed870` corrected the documentation to match the code; it did not change the code or the numbers.

---

## TASK 3 — VERIFIER STATUS

**`verify_prestep_measurement.py` quarantine justified:** **yes**

- This script validates 085ec85-era artifacts that use team-level `obs[95]` as the retention condition.
- It explicitly checks `onball_source == 'obs[95]_pre_step'` and `obs95 == 1.0` for every retained frame.
- The canonical artifacts use `onball_source = "pre_step_ball_owner_agent_idx"` and record ALL frames (not just `obs95 == 1.0` frames).
- Running this script against canonical artifacts would fail or produce misleading results because the field-name expectations and retention semantics differ.
- The quarantine banner correctly identifies this as a historical/non-canonical verifier.

**`verify_canonical_artifacts.py` correctly authoritative:** **yes**

- It validates the reconciled canonical artifacts (detail JSON, summary CSV, scope reconciliation CSV).
- It enforces the standardized ±0.05 pp tolerance.
- It rejects 085ec85-era artifact file names.
- It checks for the canonical schema fields (`pre_step_ball_owner_agent_idx`, `agent_has_ball`, `onball`).
- **PASSED** (exit 0) on current artifacts.

**`VERIFIER_STATUS.md` accurate:** **yes**

- Correctly identifies `verify_canonical_artifacts.py` as authoritative.
- Correctly quarantines `verify_prestep_measurement.py` as historical.
- Documents the standardized `RECONCILIATION_TOLERANCE_PP = 0.05`.

---

## TASK 4 — FINAL CANONICAL STATEMENT

**Which numbers are authoritative now:** `2d4b6dd`'s tables remain authoritative and **unchanged**.

- The canonical `n_onball` values under `base_seed=500000` are: 42=13, 123=96, 7=18, 999=42.
- The canonical scope-reconciliation table shows all 4 seeds match `actor_reweight_retest_eval_summary.csv` within ±0.05 pp (actual deltas ≤0.005 pp).
- `d4ed870` was a **documentation-only correction** that aligned the human-readable markdown with the code that was already producing these numbers. No measurement was redone, no parameters were changed, and no numbers were altered.

**What `d4ed870` actually fixed:**
- Corrected the canonical on-ball definition in `BASE_SEED_AND_OCCUPANCY_RECONCILIATION.md` from the incorrect "Pre-step obs[95] team possession" to the correct `pre_step_ball_owner_agent_idx == agent_index` (individual carrier).
- Rewrote the 17→74→13 causal narrative to acknowledge that both measurement method and measurement population changed, not just base seed.
- Standardized reconciliation tolerance to ±0.05 pp in code and docs.
- Quarantined the historical verifier that encoded the old team-level description.
- Made the canonical verifier explicitly reject 085ec85-era artifact file names.

---

## CONFIRMATIONS

- No training performed: **yes**
- No reward/GAE/mask/network/environment changes: **yes**
- No base_seed change: **yes** (remains 500,000)
- No new on-ball definition introduced beyond the two named: **yes**
- No unnecessary recomputation (Branch A): **yes**
- No large-artifact duplication: **yes**

## FILES WRITTEN

- `training/results/DEFINITION_CORRECTION_IMPACT_AUDIT.md` (this document)
- No new measurement tables or CSVs (Branch A — no recomputation needed)
- No changes to `VERIFIER_STATUS.md` (already accurate)
