# PASS Fix Verification — Nearest-Teammate Aiming (Option A)

## Pre-fix baseline (commit `89fa215`, 80 episodes)

| Metric | Value |
|--------|-------|
| PASS_COMPLETED | **2 / 80 = 2.5 %** |
| Median angular error to nearest teammate | ~40° |
| Min ball-to-teammate distance | 0.067 – 0.199 |
| Root cause | Rank 1: default `action.direction || stickyDirection || Vec2.fromAngle(heading)` frequently does not point at any teammate |

## Post-fix run (to be filled)

| Metric | Pre-fix | Post-fix | Judgment |
|--------|---------|----------|----------|
| PASS_COMPLETED | 2/80 = 2.5 % | _TBD_ | Rise meaningfully above 2.5 %? (bar ≥ 25 %) |
| Median angular error | ~40° | _TBD_ | Should fall sharply if aim is fixed |
| Min ball-to-teammate distance | 0.067–0.199 | _TBD | — |

## Run metadata

- **Design option chosen:** (a) nearest teammate
- **Reason:** Minimal, verifiable change aligned with diagnostic distance analysis.
- **Scoped change:** PASS direction resolution only (SHORT, LONG, HIGH). Explicit `action.direction` still wins when present. Non-PASS actions unchanged. `pass_completed` detection unchanged.
- **Not touched:** reward, GAE, masks, networks, spawn logic, training loops.
- **Artifacts:**
  - `training/results/f_act_pass_trace_seed{42,123,7,999}_postfix.csv`
  - `training/results/pass_diagnostic_manifest_postfix.json`

## Verification commands

```bash
for SEED in 42 123 7 999; do
  python -m training.eval_pass_diagnostic \
    --checkpoint training/models/mappo_academy_3_vs_1_with_keeper_seed${SEED}_50176.pt \
    --seed ${SEED} \
    --num-episodes 20 \
    --output-suffix _postfix \
    --output-dir training/results
done
```

## Residual failure modes to watch

- **Nearest teammate marked / blocked:** completion improves but stays low → implement option (b) passing-lane logic.
- **Angular error still large:** bug in teammate selection or measurement mismatch.
- **Ball close but no `pass_completed`:** re-open Rank-3 under new geometry.

## Pass/fail judgment

_Pending post-fix run. Treat as pass if completion rate rises meaningfully above 2.5 % (suggested bar ≥ 25 %)._
