# Diagnosis Note: 4-Seed 100k MAPPO Post-Strip Baseline

## 0. Provenance
- Baseline: `training/results/BASELINE_100k_POST_STRIP_4SEED.md`
- Checkpoints: `training/models/mappo_academy_3_vs_1_with_keeper_seed{42,123,999,7}_100k`
- Eval JSONs: `training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed{42,123,999,7}_100k.json`
- Git commit: `aaaec6338593ecbd8a131a73ce31601156d10289`

## 1. Per-seed diagnosis

### Seed 42
| Axis | Value |
|------|-------|
| Goals/ep | 0.00 |
| Shots/ep | 0.00 |
| Passes attempted/ep | 0.00 |
| Passes completed/ep | 0.00 |
| Possession % | 49.2% ± 50.0% |
| Turnovers/ep | 0.02 ± 0.14 |
| Mean length | 51.0 ± 0.0 steps |
| Mean reward | -0.619 ± 0.200 |
| Forward progress/ep | -0.012 ± 0.553 |
| Min distance to goal | 0.481 ± 0.286 |
| Tackles/ep | 0.22 |

**Top actions:** UP_LEFT 47.6%, SPRINT 19.5%, DOWN_LEFT 15.2%, DRIBBLE 9.1%, RELEASE_SPRINT 7.7%

**Failure mode:** Mode 1 (No exploration of finishing actions) + Mode 4 (Reward attribution mismatch). The policy never attempts shots or passes. Fixed episode length of exactly 51.0 steps indicates every episode hits the time limit. The action distribution is dominated by diagonal movement (UP_LEFT) and ball-handling spam (SPRINT, DRIBBLE) without any coherent attacking strategy. Negative forward progress confirms the ball is not advancing toward goal.

### Seed 123
| Axis | Value |
|------|-------|
| Goals/ep | 0.00 |
| Shots/ep | 0.26 |
| Passes attempted/ep | 0.00 |
| Passes completed/ep | 0.00 |
| Possession % | 0.0% |
| Turnovers/ep | 0.22 ± 0.65 |
| Mean length | 70.5 ± 101.3 steps |
| Mean reward | -0.888 ± 0.528 |
| Forward progress/ep | -0.520 ± 0.535 |
| Min distance to goal | 0.638 ± 0.189 |
| Tackles/ep | 130.44 |

**Top actions:** SLIDING 61.7%, UP 21.4%, UP_RIGHT 4.6%, DOWN 4.0%

**Failure mode:** Mode 2 (Actions present but non-productive) + Mode 3 (Possession/turnover collapse). The policy spams tackle actions (130/ep) and occasionally attempts shots (0.26/ep) but never completes them. 0% possession means the agent never gets the ball. The extreme tackle count and high episode-length variance (70.5 ± 101.3) suggest pathological exploration behavior.

### Seed 999
| Axis | Value |
|------|-------|
| Goals/ep | 0.00 |
| Shots/ep | 0.08 |
| Passes attempted/ep | 0.06 |
| Passes completed/ep | 0.00 |
| Possession % | 0.0% |
| Turnovers/ep | 0.18 ± 0.39 |
| Mean length | 49.4 ± 5.5 steps |
| Mean reward | -0.710 ± 0.235 |
| Forward progress/ep | -0.562 ± 0.347 |
| Min distance to goal | 0.639 ± 0.196 |
| Tackles/ep | 0.00 |

**Top actions:** RELEASE_DIRECTION 54.3%, RELEASE_SPRINT 36.7%, UP_LEFT 3.1%

**Failure mode:** Mode 1 (No exploration of finishing actions) + Mode 3 (Possession/turnover collapse). The policy oscillates between RELEASE_DIRECTION and RELEASE_SPRINT without any coherent strategy. 0% possession, 0% pass completion, and rare shot attempts that never succeed. This is a classic "button masher" failure mode.

### Seed 7
| Axis | Value |
|------|-------|
| Goals/ep | 0.00 |
| Shots/ep | 0.18 |
| Passes attempted/ep | 0.10 |
| Passes completed/ep | 0.00 |
| Possession % | 0.0% |
| Turnovers/ep | 0.20 ± 0.40 |
| Mean length | 49.1 ± 5.6 steps |
| Mean reward | -0.790 ± 0.279 |
| Forward progress/ep | -0.632 ± 0.678 |
| Min distance to goal | 0.643 ± 0.198 |
| Tackles/ep | 0.00 |

**Top actions:** RIGHT 58.4%, SPRINT 31.9%, RELEASE_SPRINT 5.1%

**Failure mode:** Mode 1 (No exploration of finishing actions) + Mode 3 (Possession/turnover collapse). The policy is dominated by RIGHT movement + SPRINT, with occasional shot/pass attempts that never complete. 0% possession. The action distribution shows no SHOT or PASS actions in the top actions despite the summary reporting 0.18 shots/ep, suggesting shot attempts are rare and distributed across agents.

## 2. Aggregate diagnosis

| Metric | Seed 42 | Seed 123 | Seed 999 | Seed 7 | Aggregate |
|--------|---------|----------|----------|--------|-----------|
| Goal rate | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** |
| Shots/ep | 0.00 | 0.26 | 0.08 | 0.18 | 0.13 |
| Passes attempted/ep | 0.00 | 0.00 | 0.06 | 0.10 | 0.04 |
| Pass completion | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** |
| Possession % | 49.2% | 0.0% | 0.0% | 0.0% | 12.3% |
| Forward progress/ep | -0.012 | -0.520 | -0.562 | -0.632 | **-0.432** |
| Mean length | 51.0 | 70.5 | 49.4 | 49.1 | 55.0 |
| Mean reward | -0.619 | -0.888 | -0.710 | -0.790 | **-0.752** |

**Dominant failure mode: Mode 1 + Mode 4**

The agents are not learning to explore finishing actions (shots/passes) in a productive way, and the reward signal is dominated by step cost/timeout penalties with no viable goal path. All seeds show negative forward progress, confirming the ball is not advancing toward the goal. The diversity of pathological action distributions (movement spam, tackle spam, button oscillation) across seeds suggests the policies are in early-training dynamics rather than having converged to a deliberate strategy.

**Secondary observation:** 2/4 seeds (999, 7) have 0% possession, meaning they never get the ball. This is a prerequisite failure: without possession, no finishing action is possible. Seed 123's 130 tackles/episode is clearly pathological and suggests the exploration bonus may be causing action spam rather than meaningful exploration.

## 3. Comparison to 200k PBRS baseline

| Metric | 100k (this) | 200k PBRS (`81031ae`) |
|--------|-------------|----------------------|
| Goal rate | 0.0% | 0.0% |
| Shots/ep | 0.13 (mean) | 0.00 |
| Passes/ep | 0.04 (mean) | 0.00 |
| Mean length | 55.0 | 18.5 |
| Mean reward | -0.752 | -0.109 |

The 100k policies are **more active** (longer episodes, some shot/pass attempts) but have **worse reward** than the 200k PBRS policies. This suggests that PBRS shaping improves the reward signal while also suppressing unproductive action exploration. The 100k policies are exploring more but not in a goal-directed way.

## 4. Recommendation

**Recommended next step: C (controlled exploration ablation)**

Rationale:
1. The exploration bonus (E=1, β=0.03) is already active and appears to be having mixed effects: it drives activity (longer episodes, some shot attempts) but also pathological behavior (seed 123's 130 tackles/episode, button oscillation in seeds 999/7).
2. The 200k PBRS baseline (which also included exploration) showed more passive behavior with 0 shots, suggesting that exploration alone is not sufficient and may interact with reward shaping in non-obvious ways.
3. Before committing to a 200k run (B), we should isolate whether the current exploration setting is helping or hurting. An ablation at E=0 vs E=1 (same 100k horizon) would answer this.
4. If E=0 performs similarly to E=1, then exploration is not the bottleneck and the issue is likely a bug or insufficient training signal (D). If E=0 performs worse, then exploration is helping and B is justified.

**Conditional path after C:**
- If E=1 is better than E=0: proceed to B (longer horizon with same freeze)
- If E=0 is similar to E=1: investigate D (action masking, reward attribution, or other bugs) before any further training
- If E=1 is worse than E=0: reduce β or disable exploration, then proceed to B with corrected exploration setting

## 5. Explicit statement

**Do not start 200k training (B) or a full exploration ablation (C) until the above diagnosis is confirmed.** If proceeding with C, run only the E=0 vs E=1 comparison at 100k timesteps with the same post-strip reward freeze. Do not run B and C in parallel.
