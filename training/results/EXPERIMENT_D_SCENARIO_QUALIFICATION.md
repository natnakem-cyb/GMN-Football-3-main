# Experiment D: Scenario Qualification

## 0. Provenance
- HEAD hash: `31374243eea45984e20deea53c4a9a463000e9f2`
- Branch: `main`
- Date (UTC): 2026-09-16
- Probe script: `training/experiment_d_scenario_probes.ts`
- Raw results: `experiment_d_results_1789587008423.json`
- Scope: Probe-only qualification. No engine modifications, no reward edits, no training, no architecture expansion.
- Primary forensic case: `academy_3_vs_1_with_keeper`

## 1. Metric definitions

| Name | Meaning |
|------|---------|
| `pass_actions` | Discrete actions whose type is `SHORT_PASS`, `LONG_PASS`, or `HIGH_PASS` |
| `pass_completed` | Engine events with `type === "pass_completed"` and `team === "left"` |
| `shot_actions` | Discrete actions whose type is `SHOT` |
| `shot_events` | Engine events with `type === "shot"` and `team === "left"` |
| `goals` | Left-team goals detected via `engine.score.left` delta |
| `P(PASS_COMPLETED \| poss+mask+cmd)` | Forced-pass completion rate under verified left possession + action-mask legality + commanded `SHORT_PASS` |
| `P(SHOT \| poss+mask+cmd)` | Forced-shot firing rate under verified left possession + action-mask legality + commanded `SHOT` |

All tables below separate `pass_actions` from `pass_completed` and `shot_actions` from `shot_events`.

## 2. Scenario cards summary

| Scenario ID | Type | Success predicate |
|-------------|------|-------------------|
| `academy_empty_goal` | academy_drill | Score within 15s |
| `academy_run_to_score` | academy_drill | Score without losing possession |
| `academy_pass_and_shoot_with_keeper` | academy_drill | Complete 1+ pass and score |
| `academy_3_vs_1_with_keeper` | academy_drill | Complete 2+ passes and score |
| `academy_3_vs_1_defender_2` | academy_drill | Score |
| `academy_3_vs_1_defender_3` | academy_drill | Score |
| `academy_3_vs_1_keeper_aggressive` | academy_drill | Score |
| `academy_3_vs_1_shifted` | academy_drill | Score |
| `academy_3_vs_1_randomized` | academy_drill | Score under jitter |
| `academy_rondo_4v1` | keep_ball | Retain possession 20s + 10+ passes |
| `5_vs_5` | match | Win + >50% possession |
| `11_vs_11` | match | Win + clean sheet |

Legal actions (all scenarios): IDLE, MOVE (8 directions), SPRINT, RELEASE_SPRINT, SHORT_PASS, LONG_PASS, HIGH_PASS, SHOT, TACKLE, DRIBBLE, RELEASE_DRIBBLE, RELEASE_DIRECTION.

Mask behavior: movement + IDLE always valid. PASS/SHOT require possession. TACKLE requires no possession. DRIBBLE is valid off-ball as a sticky movement modifier.

## 3. Primary forensic case: `academy_3_vs_1_with_keeper`

### 3.1 Scripted feasibility
- Episodes: 20
- Goals: 1 (5.0%)
- Pass actions: 34 (1.7/ep)
- Pass completed: 31 (1.6/ep)
- Shot actions: 3 (0.1/ep)
- Shot events: 8 (0.4/ep)
- Turnovers: 19
- Timeouts: 0
- Crashes: 0
- Avg steps: 46.05
- Avg reward: 0.1311

**F assessment:** Feasibility is confirmed. Goals are achievable under deterministic scripted control.

### 3.2 Random baseline (reconciled split metrics)
- Episodes: 20
- Goals: 4 (20.0%)
- Pass actions: 11,250 (562.5/ep)
- Pass completed: 3,295 (164.8/ep)
- Shot actions: 3,774 (188.7/ep)
- Shot events: 1,808 (90.4/ep)
- Turnovers: 7
- Timeouts: 9
- Crashes: 0
- Avg reward: 0.4624

**F assessment:** Lower-bound feasibility confirmed. Non-zero goals and non-zero pass/shot volumes.

### 3.3 Forced PASS / SHOT (conditional rates under hard guarantees)
- Episodes attempted: 10
- Valid possession+mask states found: 9
- `no_valid_state` failures: 4 episodes ended before a valid left-possession state was encountered

| Metric | Count | Rate |
|--------|-------|------|
| PASS commanded | 9 | — |
| PASS completed | 0 | **0.0%** |
| SHOT commanded | 9 | — |
| SHOT fired | 2 | 22.2% |

**C assessment (PASS):** `P(PASS_COMPLETED | poss+mask+cmd) = 0.0%`. Under verified left possession with action-mask-legal `SHORT_PASS`, a forced pass never produced a `pass_completed` event in 9 trials. This is a **C failure**: the mask declares the action legal and the player has possession, yet the execution path does not yield the expected completion event.

**C assessment (SHOT):** `P(SHOT | poss+mask+cmd) = 22.2%`. Forced shots do fire occasionally. SHOT execution path is not broken.

**Hypothesized causes for PASS failure (not fixed in this brief):**
1. Pass direction is hardcoded to `{x: 1.0, y: 0.0}` which may not intersect a moving teammate.
2. Pass power `0.75` may be insufficient to reach the target before the ball slows.
3. Opponent defenders intercept or block the pass before the receiver touches it.
4. The `targetPlayerId` is set but physics delivery is unreliable under forced single-tick execution.

### 3.4 Possession consistency
- Episodes: 10
- Total ticks: 13,276
- Inconsistencies: 2 (0.0151%)
- Crashes: 0

**C assessment:** Possession invariants are robust. No systemic causal breaks.

### 3.5 Transition lookback (200-tick window)
- Goals observed: 2
- Preceded by shot/pass within 200 ticks: 2
- No prior shot/pass: 0

**C assessment:** Every goal is causally preceded by a shot or pass event within the 200-tick lookback. Transition integrity holds in the observed episodes.

### 3.6 Return ordering (I)
| Policy | Mean return | Std | Goals |
|--------|-------------|-----|-------|
| Spam-move | 0.2474 | 0.6294 | 2 |
| Spam-tackle | 0.0000 | 0.0000 | 0 |
| Scripted-best | 0.1311 | 0.4715 | 1 |

- Scripted-best > Spam-tackle: **TRUE** (0.1311 > 0.0000)
- Scripted-best > Spam-move: **FALSE** (0.1311 < 0.2474)
- **I pass: FALSE**

**I assessment:** The scripted policy returns less than random movement spam. Incentive alignment is not demonstrated.

### 3.7 Decision: UNQUALIFIED

Failed conjuncts: **C** (forced PASS completion = 0%), **I** (scripted return < spam-move return).

### 3.8 Failed conjuncts: {C, I}

## 4. Other academy scenarios

| Scenario | Random goals | Forced PASS completed | Forced SHOT fired | Verdict |
|----------|-------------|----------------------|-------------------|---------|
| `academy_empty_goal` | 3/20 (15%) | 0/20 (0%) | 1/20 (5%) | PARTIAL |
| `academy_run_to_score` | 2/20 (10%) | 0/8 (0%) | 0/8 (0%) | PARTIAL |
| `academy_pass_and_shoot_with_keeper` | 3/20 (15%) | 0/10 (0%) | 3/10 (30%) | PARTIAL |
| `academy_3_vs_1_with_keeper` | 4/20 (20%) | 0/9 (0%) | 2/9 (22%) | **UNQUALIFIED** |
| `academy_3_vs_1_defender_2` | 5/20 (25%) | 0/40 (0%) | 4/40 (10%) | PARTIAL |
| `academy_3_vs_1_defender_3` | 3/20 (15%) | 0/20 (0%) | 3/20 (15%) | PARTIAL |
| `academy_3_vs_1_keeper_aggressive` | 4/20 (20%) | 0/9 (0%) | 2/9 (22%) | PARTIAL |
| `academy_3_vs_1_shifted` | 1/20 (5%) | 0/8 (0%) | 1/8 (12%) | PARTIAL |
| `academy_3_vs_1_randomized` | 5/20 (25%) | 0/33 (0%) | 1/33 (3%) | PARTIAL |
| `academy_rondo_4v1` | 0/20 (0%) | 0/14 (0%) | 3/14 (21%) | PARTIAL |

**Verdict key:**
- **QUALIFIED:** F ∧ L ∧ C ∧ I ∧ E all pass
- **PARTIAL:** Some conjuncts pass; others need follow-up but are not fatal
- **UNQUALIFIED:** One or more fatal conjuncts fail

All scenarios share the same forced-PASS failure mode, so none can be marked QUALIFIED until the underlying cause is diagnosed.

## 5. Cross-cutting findings

### 5.1 Mask opportunity skew
Across all academy drills, TACKLE validity is 98–100% in random-walk samples, while PASS/SHOT validity is 0–1.7%. This is by design (PASS/SHOT require possession), but it means the action space is heavily skewed toward off-ball actions during random play.

### 5.2 Forced PASS = 0 across all scenarios
The most significant cross-cutting finding is that `P(PASS_COMPLETED | poss+mask+cmd) = 0.0%` in every academy scenario tested. This is not a mask bug (the mask correctly reports PASS as legal when the player has possession). It is an execution-path failure: the commanded pass does not result in a `pass_completed` event.

### 5.3 Transition integrity
With the 200-tick lookback, all observed goals are preceded by a shot or pass event. No C anomaly in transition causality.

### 5.4 Reward monotonicity
All non-rondo scenarios show total reward = 0.0000 over 120 random steps, with monotonic step-to-step ordering. Rondo shows total reward = 0.0400 over 97 steps, also monotonic. No non-monotonic reward violations detected.

### 5.5 Incentive alignment failure
Scripted-best return (0.1311) is less than spam-move return (0.2474) in the primary case. The reward stack does not sufficiently differentiate productive scripted behavior from random movement.

## 6. Recommendation

**Do not start Experiment B (horizon extension) or any reward redesign until the primary case is qualified.**

### Ranked repair list for primary case `academy_3_vs_1_with_keeper`:

1. **Diagnose forced-PASS execution path (C repair)**
   - Instrument pass action to log `targetPlayerId`, pass direction, pass power, receiver distance, and whether `pass_completed` fires within N ticks.
   - Determine if the failure is: (a) pass direction missing receiver, (b) pass power too low, (c) opponent intercepts before receiver touches, or (d) `targetPlayerId` is ignored by physics.
   - This is the highest-priority fix because it blocks C for all scenarios.

2. **Fix return ordering (I repair)**
   - Once PASS execution is restored, re-run scripted-best vs spam-move vs spam-tackle.
   - If scripted-best still underperforms spam-move, investigate whether step cost dominates or whether the scripted controller is suboptimal.
   - Consider whether the reward stack needs adjustment for sparse-goal scenarios.

3. **Re-qualify after repairs**
   - Re-run forced PASS/SHOT probes.
   - Re-run return ordering.
   - If both C and I pass, mark primary case QUALIFIED and propagate to other academy scenarios.

4. **Then consider Experiment B**
   - Only after primary case is QUALIFIED should horizon extension be explored.

### Explicit constraints honored
- No engine modifications
- No reward magnitude changes
- No training
- No architecture expansion
- All probes run on frozen engine at HEAD `31374243eea45984e20deea53c4a9a463000e9f2`
