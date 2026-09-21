# Residual Work Completion Note

**Date:** 2026-09-21  
**Commit:** `2d4b6dd32a58c7ab0e6617ef8f3e36d65ac39ff5` (close-out)  
**Residual-work commit:** pending

## Files Modified

| File | Change |
|------|--------|
| `training/results/BASE_SEED_AND_OCCUPANCY_RECONCILIATION.md` | Corrected canonical on-ball definition throughout; added explicit `n_onball` footnote; rewrote 17→74→13 causal narrative to acknowledge dual change; updated verification section to make `verify_canonical_artifacts.py` primary and quarantine `verify_prestep_measurement.py`. |
| `training/eval_canonical_three_agent_measurement.py` | Added `RECONCILIATION_TOLERANCE_PP = 0.05` and `RECONCILIATION_TOLERANCE_ABS = 0.0005` constants; replaced hard-coded `0.001` with `RECONCILIATION_TOLERANCE_ABS`; updated docstring to describe individual-carrier onball definition. |
| `training/verify_canonical_artifacts.py` | Added `RECONCILIATION_TOLERANCE_PP = 0.05`; replaced hard-coded `0.5` with the constant; added guard rejecting 085ec85-era artifact file names. |
| `training/generate_final_report.py` | Removed stale "085ec85 pushed? No - branch is ahead" claim; replaced "obs[95] == 1.0 is the ONLY retention condition" with correct individual-carrier description; reads current HEAD dynamically via `git rev-parse HEAD`. |
| `training/verify_prestep_measurement.py` | Added quarantine banner at top of file declaring it historical/non-canonical and pointing to `verify_canonical_artifacts.py`. |
| `training/results/VERIFIER_STATUS.md` | New file documenting authoritative verifier, quarantined verifier, and standardized tolerance. |

## Tolerance

**`RECONCILIATION_TOLERANCE_PP = 0.05`** (percentage points)  
**`RECONCILIATION_TOLERANCE_ABS = 0.0005`** (absolute rate units)

Enforced in:
- `training/eval_canonical_three_agent_measurement.py`
- `training/verify_canonical_artifacts.py`
- `training/results/BASE_SEED_AND_OCCUPANCY_RECONCILIATION.md`
- `training/results/VERIFIER_STATUS.md`

## Canonical Verifier Default Paths

`training/verify_canonical_artifacts.py` defaults to:
- `training/results/post_reweight_logit_prestep_reconciled_detail.json`
- `training/results/post_reweight_logit_prestep_reconciled_summary.csv`
- `training/results/post_reweight_canonical_scope_reconciliation.csv`

Rejects 085ec85-era file names (`post_reweight_logit_prestep_detail.json`, `post_reweight_logit_prestep_summary.csv`, `onball_occupancy_summary.csv`, `onball_occupancy_prestep_summary.csv`).

## generate_final_report.py Status

- No longer emits "085ec85" or "branch is ahead".
- Reports live HEAD SHA via `git rev-parse HEAD`.
- Describes onball as `(pre_step_ball_owner_agent_idx == agent_index)`.
- Contains the dual-change wording for 74 → 13.

## Verification Commands Run

```bash
# Task 1 — obs[95] / team-possession wording check
Select-String -Path "training/results/BASE_SEED_AND_OCCUPANCY_RECONCILIATION.md" -Pattern 'obs\[95\]|team possession|on-ball measurement'
# Result: remaining hits correctly describe team-level vs individual-carrier semantics only

# Task 2 — 74→13 causal wording check
Select-String -Path "training/results/BASE_SEED_AND_OCCUPANCY_RECONCILIATION.md" -Pattern '74 → 13|base-seed change|solely|only by the seed'
# Result: no single-cause explanation remains

# Task 3 — tolerance standardization check
Select-String -Path "training/eval_canonical_three_agent_measurement.py","training/verify_canonical_artifacts.py" -Pattern '0\.001|0\.5|tolerance|TOLERANCE'
# Result: only 0.05 / RECONCILIATION_TOLERANCE constants appear

# Task 3 — verifier passes under tighter tolerance
python training/verify_canonical_artifacts.py
# Result: ALL VERIFICATION CHECKS PASSED

# Task 4 — canonical verifier rejects old file names
python -c "import sys; sys.path.insert(0, '.'); exec(open('training/verify_canonical_artifacts.py').read().replace('detail_path = \"training/results/post_reweight_logit_prestep_reconciled_detail.json\"', 'detail_path = \"training/results/post_reweight_logit_prestep_detail.json\"'))"
# Result: ERROR: post_reweight_logit_prestep_detail.json is a historical 085ec85-era artifact.

# Task 4 — old verifier banner present
python training/verify_prestep_measurement.py 2>&1 | Select-Object -First 1
# Result: HISTORICAL VERIFIER — DO NOT USE FOR CANONICAL CLOSE-OUT

# Task 5 — final report generator
python training/generate_final_report.py 2>&1 | Select-Object -First 12
# Result: no "085ec85", no "branch is ahead", correct HEAD SHA, correct onball definition
```

## Confirmations

- No training performed: yes
- No reward/GAE/mask/network/environment changes: yes
- No parameter changed to force a match: yes (tolerance pre-stated at ±0.05 pp)
- No new interpretation beyond corrected definitions: yes
- No files left in ambiguous state: yes
