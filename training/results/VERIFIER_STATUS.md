# Verifier Status

**Date:** 2026-09-21  
**Canonical close-out commit:** `2d4b6dd32a58c7ab0e6617ef8f3e36d65ac39ff5`

## Current authoritative verifier

**`training/verify_canonical_artifacts.py`** is the sole authoritative verifier for the canonical close-out artifacts committed at `2d4b6dd` and later. It validates:

- `post_reweight_logit_prestep_reconciled_detail.json`
- `post_reweight_logit_prestep_reconciled_summary.csv`
- `post_reweight_canonical_scope_reconciliation.csv`

Default paths are hard-coded to the reconciled file names. The script enforces the canonical tolerance of **±0.05 pp** for scope reconciliation.

## Quarantined verifier

**`training/verify_prestep_measurement.py`** is **HISTORICAL / NON-CANONICAL**.

- It validates the 085ec85-era artifacts (`post_reweight_logit_prestep_detail.json`, `post_reweight_logit_prestep_summary.csv`, `onball_occupancy_summary.csv`).
- Those artifacts used `base_seed=700000` and agent-0-only scope, which are not the canonical measurement.
- Running this script against the current reconciled artifacts will produce misleading results because it expects different field names and artifact schemas.
- **Do not cite its pass/fail result as validation of the canonical close-out.**

## Tolerance

| Constant | Value | Where enforced |
|----------|-------|----------------|
| `RECONCILIATION_TOLERANCE_PP` | 0.05 pp | `verify_canonical_artifacts.py`, `eval_canonical_three_agent_measurement.py`, this document |

## Invariant

Any future measurement run at `base_seed=500000` must replace all four canonical artifacts atomically and re-run `verify_canonical_artifacts.py`. The historical verifier is never to be used for the canonical gate.
