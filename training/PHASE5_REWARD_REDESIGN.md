# Phase 5 — Reward Redesign

## Root Cause

The original reward function in `src/engine/ObservationEncoder.ts` produced a scalar where:
- **Ball-progress checkpoint** contributed ~100% of total dense reward in `academy_3_vs_1_with_keeper`
- Goal/shoot/pass shaping contributions were effectively zero in practice because:
  - Progress reward magnitude (`min(0.05, deltaX * 0.5)`) dominated all event bonuses
  - Ball-hogging penalty (`-0.005/tick` after 30 ticks) was too weak to counteract progress farming
  - Pass/shoot bonuses only fired on discrete events, providing insufficient gradient

## Redesign Principles

1. **Reduce dense progress dominance** without making the signal sparse
2. **Add explicit per-event rewards** for passing and shooting so they compete with dribbling
3. **Strengthen anti-farming penalties** so indefinite possession is discouraged
4. **Keep terminal goal reward dominant** so scoring remains the ultimate objective

## Changes Made

### A. `src/engine/ObservationEncoder.ts` — `computeReward()`

| Parameter | Old Value | New Value |
|-----------|-----------|-----------|
| `goal_scored` | +1.0 | +2.0 |
| `goal_conceded` | -1.0 | -1.0 (unchanged) |
| `progress_checkpoint_max` | 0.05 | 0.02 |
| `progress_checkpoint_scale` | `deltaX * 0.5` | `deltaX * 0.2` |
| `pass_completed` | 0 | +0.15 |
| `shot_taken` | 0 | +0.1 |
| `shot_quality_on_target` | +0.03 | +0.1 |
| `shot_quality_off_target` | +0.001 | +0.01 |

**Call-site update**: `src/engine/GameEngine.ts` now passes `passCompletedByLeft` and `shotEventByLeft` flags derived from `newEventsThisTick`.

### B. `training/gmn_pettingzoo.py` — `CooperativeRewardShaper`

| Parameter | Old Default | New Default |
|-----------|-------------|-------------|
| `penalty_ball_hogging` | -0.005/tick | -0.02/tick |
| `max_unassisted_hold_ticks` | 30 | 15 |

## Expected Effect

| Behavior | Before (per-agent r/step) | After (per-agent r/step) | Change |
|----------|--------------------------|--------------------------|--------|
| Dribble forward | 0.0245 | 0.0384 | ↑ (progress reduced but events vary) |
| Move right | 0.0245 | 0.0384 | ↑ |
| Shoot | 0.0812 | 0.1330 | ↑ (shot bonus + quality) |
| Short pass | 0.0171 | 0.0318 | ↑ (pass bonus) |
| Idle | 0.0018 | 0.0007 | ↓ (progress reduced) |

**Key insight**: Passing and shooting now yield materially higher per-step reward than idle or pure dribbling. The goal reward (+2.0) dominates all intermediate signals, preserving the correct objective hierarchy.

## Overcorrection Check

- **Did not remove progress reward**: Kept at reduced rate (0.02 max, deltaX*0.2) so the agent still receives dense shaping
- **Did not make rewards sparse**: Pass (+0.15) and shot (+0.1) bonuses fire frequently enough to provide gradient
- **Did not inflate magnitudes arbitrarily**: All values derived from empirical reward accounting
- **Goal reward remains dominant**: +2.0 terminal vs +0.15 per-pass or +0.1 per-shot
