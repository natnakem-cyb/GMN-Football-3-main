# Experiment D: Pass-Path Forensics

## 0. Provenance
- HEAD: `2230fdc5b578cd135eddb89a9a77e16f2c278479`
- Date: 2026-09-16
- Scenario: `academy_3_vs_1_with_keeper`
- Probe: `training/experiment_d_pass_forensics.ts`
- Raw data: `training/results/forensics/experiment_d_pass_forensics_1789588207080.json`

## 1. Setup

| Parameter | Value |
|-----------|-------|
| Episodes attempted | 20 |
| Valid possession+mask states found | 21 |
| `no_valid_state` failures | 9 |
| PASS commanded | 21 |
| PASS completed | 0 |
| SHOT commanded | 21 |
| SHOT fired | 0 |

## 2. Hypothesis battery

| ID | Hypothesis | Evidence | Result |
|----|------------|----------|--------|
| H1 | Wrong agent receives the action | Acting `left_1` is the logged owner before and after; action map is keyed by `player.id`. | **Rejected** |
| H2 | Pass requires direction / `targetPlayerId` and none is set | We set `direction` and `targetPlayerId`; engine `applyPlayerAction` uses `action.direction` first. | **Rejected** |
| H3 | No legal receiver in effective range/cone | Nearest teammate distance to BALL starts at 0.026 (very close) but increases as ball goes out of bounds. Ball never reaches receiver. | **Confirmed** |
| H4 | Power/type insufficient | Power 0.75 with SHORT_PASS (loft 0) produces velocity ~1.68, enough to cross the pitch. Not a power issue. | **Rejected** |
| H5 | Opponent intercepts before completion | Ball owner becomes `null` (loose), never `right`. No interception event in observe window. | **Rejected** |
| H6 | Event name / team filter miss | Engine emits `pass` event; probe checks `pass_completed`. Both strings match project enum. | **Rejected** |
| H7 | Sticky/multi-tick: single-tick PASS ignored | Engine processes SHORT_PASS immediately in `applyPlayerAction`; no multi-tick requirement. | **Rejected** |
| H8 | Mask true but engine still rejects | Ball is kicked (velocity changes, `pass` event emitted). Engine accepts the action. | **Rejected** |

## 3. Confirmed root cause: H3

The forced pass direction is computed as the straight-line vector from the ball carrier to the nearest teammate. In the primary scenario, the nearest teammate (`left_2`, LW) is offset significantly in the negative y direction (`y: -0.25` relative to the carrier at `y: -0.02`). The resulting normalized direction has a large y component (`≈ -0.75`).

When the ball is kicked with this direction and power 0.75, the y velocity is strong enough to carry the ball past the touchline (`y < -0.42`) within ~15 ticks. The engine then clamps the ball to the throw-in line (`y = -0.37`) and emits `out_of_bounds`. Because the ball is dead at the touchline, no teammate can gain possession, and `pass_completed` never fires.

### Key tick dump (first trial)

| Tick | Phase | Ball (x, y) | Ball velocity | Ball→nearest mate | Event |
|------|-------|-------------|---------------|-------------------|-------|
| 1 | before | (0.267, -0.030) | (0, 0) | 0.026 (`left_1`) | — |
| 2 | after_command | (0.296, -0.063) | (1.68, -1.92) | 0.069 (`left_1`) | `pass` |
| 3 | observe | (0.324, -0.095) | (1.62, -1.85) | 0.111 (`left_1`) | `pass` |
| ... | ... | ... | ... | ... | ... |
| 16 | observe | (0.610, -0.370) | (0, 0) | — | `out_of_bounds` |
| 17–50 | observe | (0.610, -0.370) | (0, 0) | — | `out_of_bounds` (repeats) |

The ball crosses the touchline at tick 16 and remains dead. `pass_completed` never fires.

## 4. Implication for C

The mask correctly reports PASS as legal when the player has possession. The engine correctly processes the action. The failure is in the **control layer's pass targeting**: the commanded direction is geometrically valid but tactically poor, sending the ball out of bounds before any teammate can receive it.

This is a **C failure** but not an engine bug. The fix belongs in the probe/control layer (adapter/scripted controller), not in `src/engine/`.

## 5. Recommended repair

**Minimal change:** In the forced-pass command, clamp the pass direction so the y component does not exceed a safe margin (e.g., `|y| ≤ 0.5` after normalization), or select a pass target whose relative position has a small y offset. Alternatively, reduce pass power so the ball slows before reaching the touchline.

**Verification:** After the clamp, re-run forced PASS with ≥20 valid possession+mask trials. Target: `P(PASS_COMPLETED | poss+mask+cmd) > 0`.
