# PASS Fix Verification — Nearest-Teammate Aiming (Option A)

## Pre-fix baseline (commit `89fa215`, 80 episodes)

| Metric | Value |
|--------|-------|
| PASS_COMPLETED | **2 / 80 = 2.5 %** |
| Median angular error to nearest teammate | ~40° |
| Min ball-to-teammate distance | 0.067 – 0.199 |
| Root cause | Rank 1: default `action.direction || stickyDirection || Vec2.fromAngle(heading)` frequently does not point at any teammate |

## Post-fix results (HEAD `986b480` + uncommitted `resolvePassDirection` fix)

| Metric | Pre-fix (89fa215) | Post-fix | Δ |
|--------|------------------|-----------|---|
| PASS_COMPLETED | 2/80 = 2.5% | **21/80 = 26.25%** | +23.75 pp |
| Median angular error | ~40° | **0.13°** | −39.9° |
| Min ball-to-teammate distance (median) | 0.067–0.199 | **0.0532** | tighter cluster |

Per-seed breakdown:

| Seed | PASS_COMPLETED | Rate | Median ang. err | Median min dist |
|------|---------------|------|-----------------|-----------------|
| 42 | 1/20 | 5.0% | 0.0° | 0.0548 |
| 123 | 4/20 | 20.0% | 0.5° | 0.0557 |
| 7 | 13/20 | 65.0% | 0.0° | 0.0245 |
| 999 | 3/20 | 15.0% | 0.6° | 0.0557 |

**Pass/fail judgment:**
Completion rate is **21/80 = 26.25%**.
Threshold for PASS: meaningfully above 2.5% (suggested bar ≥ 25% = 20/80).
Result: **PASS**.

### Residual failure modes observed
- **High seed variance:** seed 7 = 65%, seed 42 = 5%. Remaining misses are not explained by aiming (angular error is near 0°); likely due to receiver availability / keeper interference / ball-to-teammate distance at force time (some episodes had max min-distance ≈ 0.30).
- **Ball-to-teammate distance still > 0.25 in tail:** ~10% of valid episodes had min distance > 0.20, which is outside easy reception range.
- **No option (b) passing-lane logic implemented:** nearest-teammate aim ignores obstructing defenders. Episodes where the nearest teammate is marked/blocked still fail despite correct aim.

## Run metadata

- **Git HEAD:** `986b4809f2e2` (template commit) + uncommitted `resolvePassDirection` fix in `src/engine/GameEngine.ts` and `training/bridge_server.ts`
- **Design option chosen:** (a) nearest teammate
- **Reason:** Minimal, verifiable change aligned with diagnostic distance analysis.
- **Scoped change:** PASS direction resolution only (SHORT, LONG, HIGH). Explicit `action.direction` still wins when present. Non-PASS actions unchanged. `pass_completed` detection unchanged.
- **Not touched:** reward, GAE, masks, networks, spawn logic, training loops.
- **Confirmation:** No training / reward / GAE / mask / network / spawn code was modified during this verification.
- **Artifacts:**
  - `training/results/f_act_pass_trace_seed{42,123,7,999}_postfix.csv`
  - `training/results/pass_diagnostic_manifest_postfix.json`
  - `training/results/PASS_FIX_VERIFICATION.md` (this file)

## Verification commands

```bash
for SEED in 42 123 7 999; do
  python -m training.eval_pass_diagnostic \
    --checkpoint training/models/mappo_academy_3_vs_1_with_keeper_seed${SEED}_50176.pt \
    --scenario academy_3_vs_1_with_keeper_onball \
    --seed ${SEED} \
    --num-episodes 20 \
    --base-seed 42 \
    --output-suffix _postfix \
    --output-dir training/results
done
```

## Pre-fix artefact integrity

- `training/results/f_act_pass_trace_seed{42,123,7,999}.csv`: **untouched** (byte-identical to pre-fix)
- `training/results/pass_diagnostic_manifest.json`: **untouched** (36 lines, unchanged)
- `training/results/PASS_DIAGNOSTIC_FINDINGS.md`: **untouched**

## Residual failure modes to watch

- **Nearest teammate marked / blocked:** completion improves but stays low → implement option (b) passing-lane logic.
- **Angular error still large:** bug in teammate selection or measurement mismatch.
- **Ball close but no `pass_completed`:** re-open Rank-3 under new geometry.

## Pass/fail judgment

**PASS.** Completion rate rose from 2.5% (2/80) to 26.25% (21/80), exceeding the ≥ 25% threshold. Median angular error fell from ~40° to 0.13°, confirming the nearest-teammate aim fix is mechanically effective. The remaining failures are dominated by distance/availability tail cases and seed variance, not by aiming error.
