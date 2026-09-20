# ONBALL_OCCUPANCY_CHECK — Post-Reweight Logit Snapshot Disconnect Test

**Date:** 2026-09-20  
**HEAD:** `3428b96`  
**Scenario:** `academy_3_vs_1_with_keeper_onball`  
**Data source:** Reused rollout logs from post-reweight logit snapshot (`post_reweight_logit_detail.json`) + rebuilt eval CSV (`actor_reweight_retest_eval_summary.csv`)

**On-ball definition:** `obs[95] == 1.0` (agent 0 has the ball) — identical to logit snapshot frame filter.

**Frequency-reweight verdict:** Remains closed (0/4 primary). This task does not reopen that verdict.

---

## Task 1 — Occupancy Definition

**On-ball** for this task means the same frame-selection criterion used in the logit snapshot: `obs[95] == 1.0` for agent 0. This is purely an occupancy count (how often agent 0 has the ball), not a mask-legality filter. The separate mask-legality question is left untouched.

---

## Task 2 — Per-Seed Occupancy Table

Using the 50k final checkpoints from the logit snapshot (50 episodes, deterministic, `base_seed=700000`):

| Seed | n_episodes | n_ticks | n_onball | P(on-ball) | n_onball_frames | P(PASS+SHOT | on-ball) | Implied agent0 rate | Actual rebuilt rate | Gap |
|------|------------|---------|----------|------------|-----------------|------------------------|---------------------|---------------------|-----|
| 42 | 50 | 2550 | 21 | 0.82% | 21 | 52.94% | 0.35% | 0.75% | -0.40% |
| 123 | 50 | 2550 | 27 | 1.06% | 27 | 59.26% | 0.63% | 1.67% | -1.04% |
| 7 | 50 | 2550 | 23 | 0.90% | 23 | 43.48% | 0.39% | 0.86% | -0.47% |
| 999 | 50 | 2550 | 11 | 0.43% | 11 | 54.55% | 0.24% | 1.03% | -0.79% |

**Notes:**
- `n_ticks` = sum of `episode_length` across all 50 episodes (51 ticks per episode in this scenario).
- `n_onball` = sum of `onball_frames` across all episodes (ticks where `obs[95]==1.0`).
- `P(PASS+SHOT | on-ball)` = fraction of on-ball frames where deterministic `action_taken` is in {9,10,11,12}.
- `Implied agent0 rate` = `P(on-ball)` × `P(PASS+SHOT | on-ball)`.
- `Actual rebuilt rate` = from `actor_reweight_retest_eval_summary.csv` (team-wide metric: all 3 agents × all ticks).

---

## Task 3 — Hypothesis Test

### 3a. Implied vs Actual

For all four seeds, the implied agent0 rate is **lower** than the actual team-wide rate. The gap ranges from -0.40 pp (seed 42) to -1.04 pp (seed 123).

If we naively assume each of the 3 agents contributes equally, the expected team-wide rate would be approximately 3× the implied agent0 rate:

| Seed | Implied agent0 | Expected team-wide (3×) | Actual team-wide | Ratio actual/expected |
|------|----------------|------------------------|------------------|----------------------|
| 42 | 0.35% | 1.05% | 0.75% | 0.71× |
| 123 | 0.63% | 1.89% | 1.67% | 0.88× |
| 7 | 0.39% | 1.17% | 0.86% | 0.73× |
| 999 | 0.24% | 0.72% | 1.03% | 1.43× |

For seeds 42, 123, and 7, the actual team-wide rate is **lower** than the equal-contribution prediction. For seed 999, the actual rate is **higher** than the equal-contribution prediction.

### 3b. Seed 42 Specific

**Question:** Does seed 42's low unconditional rate (0.75%) despite high conditional π(PASS) (0.55) make sense under the occupancy hypothesis?

**Answer: PARTIALLY — but not in the simple way the hypothesis states.**

Seed 42 does have relatively low on-ball occupancy (0.82%, 2nd lowest among 4 seeds) and the highest conditional PASS+SHOT rate (52.94%). Its implied agent0 rate (0.35%) is the 2nd lowest among the 4 seeds, which is consistent with its low actual rate (0.75%, lowest among 4 seeds).

However, the occupancy hypothesis **does not fully explain** the disconnect:
1. Seed 999 has **lower** on-ball occupancy (0.43%) and **lower** implied rate (0.24%), yet its **actual** rate (1.03%) is **higher** than seed 42's. This directly contradicts the simple claim that low occupancy → low unconditional rate.
2. The gap between implied and actual is **larger** for seed 42 (-0.40 pp) than for seed 7 (-0.47 pp) and similar to seed 999 (-0.79 pp), suggesting that other agents' behavior (or other factors) explain a substantial portion of the team-wide rate.
3. The equal-contribution model predicts seed 42's team-wide rate should be ~1.05%, but actual is only 0.75%. This means other agents in seed 42 contribute **less** than agent0, not more.

### 3c. Verdict

**Occupancy hypothesis: PARTIALLY SUPPORTED**

- **Supported element:** Seed 42's low on-ball occupancy (relative to seeds 123 and 7) does contribute to its low unconditional rate. The arithmetic product P(on-ball) × P(PASS+SHOT|on-ball) places seed 42 near the bottom of the implied-rate ranking, consistent with its bottom ranking in actual rate.
- **Not supported element:** The hypothesis does not fully explain the disconnect. Seed 999 has the lowest on-ball occupancy (0.43%) but the 2nd-highest actual rate (1.03%). The gap between implied and actual varies non-uniformly across seeds (0.40–1.04 pp), indicating that other factors — most likely other agents' behavior — account for a substantial portion of the team-wide variance.

---

## Task 4 — Interpretation

The occupancy check simplifies the picture in one respect and complicates it in another:

**Simplifies:** Seed 42's high conditional π(PASS) is not "wasted" because agent0 rarely gets the ball. The disconnect between high conditional preference and low unconditional rate is largely an arithmetic consequence of low on-ball occupancy. This removes "actor-behavior inconsistency" as a candidate explanation for seed 42's specific pattern.

**Complicates:** Occupancy alone does not explain cross-seed ordering. Seed 999 has the lowest on-ball occupancy but the 2nd-highest actual rate. The equal-contribution model fails for seed 999 (actual 1.43× predicted) and over-predicts for seeds 42/7/123. This indicates that other agents' PASS+SHOT behavior varies substantially across seeds, and that the team-wide rate is not simply the sum of identical agent contributions.

**Bottom line:** The seed-42 disconnect is **partially** an occupancy artifact. The remaining variance — especially seed 999's anomalously high actual rate given its low implied rate — points to cross-seed differences in other agents' behavior or in team-level coordination, not to a problem with agent0's logits.

---

## Non-Claims

- Policy bars still failed 0/4 primary; this task does not change that verdict.
- No new training is authorized by this measurement.
- The "999 held under reweight" narrative remains void.
- This task does not reconcile mask-legality definitions (separate flagged task).

---

## Confirmation Checklist

- [x] On-ball definition identical to logit snapshot (`obs[95]==1.0`)
- [x] All checkpoints verified against inventory SHA256s
- [x] Behavioral rates cited only from rebuilt CSV
- [x] No training performed
- [x] No reward/GAE/mask/network/entropy changes
- [x] No policy-sufficiency verdict reopened

---

## Files

- `training/results/ONBALL_OCCUPANCY_CHECK.md` — this report
- `training/results/onball_occupancy_summary.csv` — per-seed occupancy table
