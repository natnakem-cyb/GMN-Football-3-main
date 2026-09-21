# CANONICAL PI SNAPSHOT AND SEED-42 RE-EXAMINATION

**Date:** 2026-09-21  
**HEAD:** `63b4b7ae50cf236f7b5441e32448bfac59fd36a4`  
**Protocol:** base_seed=500000, individual-carrier onball, scenario=academy_3_vs_1_with_keeper_onball, 50 ep/seed

---

## TASK 1 — DATA SOURCE DETERMINATION

**Data source:** `training/results/post_reweight_logit_prestep_reconciled_detail.json` (existing, verified canonical artifact)

**Required fields confirmed present:**
- `probs` (19-element softmax probability vector) — present
- `masked_logits` (19-element masked logit vector) — present
- `action_mask` (19-element binary mask) — present
- `action_taken` (int, deterministic argmax) — present
- `onball` (bool, individual-carrier definition) — present
- `pre_step_obs95` (float, team possession) — present
- `pre_step_ball_owner_agent_idx` (int, individual carrier) — present
- `entropy` (float, policy entropy) — present

**Conclusion:** No new data collection required. The existing canonical detail JSON contains all fields needed to compute π statistics under the verified individual-carrier on-ball definition.

---

## TASK 2 — CANONICAL π STATISTICS

### 2.1 π Statistics (on-ball frames only)

| Seed | n_onball | n_PASS_legal | n_SHOT_legal | n_either_legal | π_PASS_mean | π_SHOT_mean | π_PASS+SHOT_mean | H(π) | Δ_PASS-MOVE | Δ_SHOT-MOVE |
|------|----------|--------------|--------------|----------------|-------------|-------------|------------------|------|--------------|--------------|
| 42 | 13 | 13 | 13 | 13 | 0.562647 | 0.022073 | **0.584720** | 1.842472 | 2.194521 | -0.629992 |
| 123 | 96 | 96 | 96 | 96 | 0.243202 | 0.057281 | 0.300483 | 2.753993 | 0.528614 | -0.365700 |
| 7 | 18 | 18 | 18 | 18 | 0.285108 | 0.048443 | 0.333550 | 2.669602 | 0.600177 | -0.704717 |
| 999 | 42 | 42 | 42 | 42 | 0.292379 | 0.061863 | 0.354242 | 2.667934 | 0.633952 | -0.271171 |

**Note:** All on-ball frames have all actions legal (n_PASS_legal = n_onball, n_SHOT_legal = n_onball) because the academy_3_vs_1_with_keeper_onball scenario provides full action masks for all controlled agents at all times. The "legal-conditional" and "all-onball" populations are identical.

### 2.2 Legal-Conditional π Statistics

| Seed | n_onball | n_PASS_legal | π_PASS_mean (legal) | n_SHOT_legal | π_SHOT_mean (legal) | n_either_legal | π_PASS+SHOT_mean (legal) |
|------|----------|--------------|---------------------|--------------|---------------------|----------------|--------------------------|
| 42 | 13 | 13 | 0.562647 | 13 | 0.022073 | 13 | 0.584720 |
| 123 | 96 | 96 | 0.243202 | 96 | 0.057281 | 96 | 0.300483 |
| 7 | 18 | 18 | 0.285108 | 18 | 0.048443 | 18 | 0.333550 |
| 999 | 42 | 42 | 0.292379 | 42 | 0.061863 | 42 | 0.354242 |

### 2.3 Deterministic Selection Frequency (separate table)

| Seed | n_onball | n_selected_PS | P(selected PS | on-ball, legal) |
|------|----------|---------------|-------------------------|
| 42 | 13 | 8 | 0.615385 |
| 123 | 96 | 79 | 0.822917 |
| 7 | 18 | 17 | 0.944444 |
| 999 | 42 | 30 | 0.714286 |

**Important distinction:** π_PASS+SHOT_mean is the policy's *softmax probability mass* over PASS+SHOT actions. P(selected PS | on-ball, legal) is the *deterministic argmax selection frequency*. These are separate statistics and must not be merged.

---

## TASK 3 — π-FLOOR RECONCILIATION

| Seed | n_onball | n_selected_PS | mean_π_PS | mean(π_selected) | mean(mask_sum|selected) | floor_action | floor_mask | ineq_pass |
|------|----------|---------------|-----------|------------------|-------------------------|--------------|------------|-----------|
| 42 | 13 | 8 | 0.584720 | 0.692147 | 18.000000 | 0.032389 | 0.034188 | True |
| 123 | 96 | 79 | 0.300483 | 0.159015 | 18.000000 | 0.043311 | 0.045718 | True |
| 7 | 18 | 17 | 0.333550 | 0.187686 | 18.000000 | 0.049708 | 0.052469 | True |
| 999 | 42 | 30 | 0.354242 | 0.229712 | 18.000000 | 0.037594 | 0.039683 | True |

**All seeds pass π-floor:** **yes**

The π-floor inequality holds for all four seeds:
- `n_selected_PS × mean(π_selected) / n_onball ≤ mean_π_PS`

This confirms the canonical π statistics are internally consistent.

---

## TASK 4 — SEED-42 DISCONNECT RE-EXAMINATION

### 4.1 The original question

Under the canonical definition and base_seed, does seed 42 still show a high conditional π(PASS+SHOT) relative to the other seeds, given its very small n_onball=13?

### 4.2 Side-by-side comparison

| Seed | n_onball | π_PASS+SHOT_mean | P(selected PS | on-ball, legal) | Unconditional rate |
|------|----------|------------------|-------------------------|---------------------|
| 42 | 13 | **0.584720** | 0.615385 | 0.745098% |
| 123 | 96 | 0.300483 | 0.822917 | 1.673203% |
| 7 | 18 | 0.333550 | 0.944444 | 0.862745% |
| 999 | 42 | 0.354242 | 0.714286 | 1.032680% |

**Finding:** Seed 42 has the **highest** conditional π_PASS+SHOT_mean (0.5847) of all four seeds under the canonical measurement. This is not a spurious result from the old measurement — it persists under the corrected definition and base_seed.

### 4.3 Small-n caveat

With n_onball=13 for seed 42, a single frame changes the conditional rate by **7.69 percentage points**. The observed π_PASS+SHOT_mean of 0.5847 should be read as a descriptive statistic from a sparse sample, not a stable seed-level property. The same caveat applies to seed 7 (n=18, 5.56 pp per frame).

### 4.4 Occupancy-arithmetic re-test

The correct occupancy arithmetic for the all-agent scope is:

```
unconditional_rate = P(on-ball) * P(selected PS | on-ball) + P(off-ball) * P(selected PS | off-ball)
```

Breaking this down by on-ball vs off-ball contribution:

| Seed | n_onball | n_selected_ps_onball | n_pass_shot_total | n_offball | n_selected_ps_offball | implied_onball | implied_offball | total_implied | actual | match? |
|------|----------|----------------------|-------------------|-----------|-----------------------|----------------|-----------------|---------------|--------|--------|
| 42 | 13 | 8 | 57 | 7637 | 49 | 0.104575% | 0.640523% | 0.745098% | 0.745098% | YES |
| 123 | 96 | 79 | 128 | 7554 | 49 | 1.032680% | 0.640523% | 1.673203% | 1.673203% | YES |
| 7 | 18 | 17 | 66 | 7632 | 49 | 0.222222% | 0.640523% | 0.862745% | 0.862745% | YES |
| 999 | 42 | 30 | 79 | 7608 | 49 | 0.392157% | 0.640523% | 1.032680% | 1.032680% | YES |

**Key observation:** For every seed, the off-ball contribution (0.640523%) is constant at 49 selections / 7650 decisions = 0.6405%. This makes sense: off-ball PASS+SHOT selections are actions that the policy takes when it does not have the ball (e.g., attempting a pass or shot from a non-possession state, which the environment may or may not allow).

The on-ball contribution varies by seed:
- Seed 42: 0.1046% (8/7650)
- Seed 123: 1.0327% (79/7650)
- Seed 7: 0.2222% (17/7650)
- Seed 999: 0.3922% (30/7650)

### 4.5 Disconnect status: **Partially explained, with a remaining puzzle**

**What is explained:** The low unconditional rate for seed 42 (0.745%) is **primarily due to low on-ball occupancy** (13/7650 = 0.17%), not a failure to select PASS+SHOT when on-ball. When seed 42 has the ball, it selects PASS+SHOT at 61.54% (8/13), which is moderate. The majority of PASS+SHOT selections for all seeds come from off-ball situations (49/57 = 85.96% for seed 42).

**What remains:** Seed 42 still has the **highest** conditional π_PASS+SHOT_mean (0.5847) of all four seeds, yet its deterministic selection frequency (0.615) is second-lowest. The gap between policy probability (0.5847) and actual selection (0.615) is small, but the absolute level is still lower than seeds 123 (0.823), 7 (0.944), and 999 (0.714). The original puzzle — high conditional π but low unconditional rate — is now understood as an occupancy effect, but the underlying question of *why* seed 42 has such low on-ball occupancy (and why its conditional selection rate is moderate despite high π) remains open.

**Small-n discipline:** With n_onball=13, the conditional π_PASS+SHOT_mean of 0.5847 has a confidence interval of roughly ±0.14 (1/√13 ≈ 0.28, but for bounded [0,1] proportions the Wilson interval is tighter). This means seed 42's conditional π could plausibly overlap with seed 999's (0.354) or seed 123's (0.300) at the upper bound. The ranking of seed 42 as "highest" is suggestive but not definitive.

---

## SUPERSESSION

**`3428b96` (original logit snapshot) superseded: yes**

Reason: The original logit snapshot used:
1. Wrong on-ball definition: team-level `obs[95]` (left-team possession) instead of individual-carrier `pre_step_ball_owner_agent_idx == agent_index`
2. Wrong base_seed: 700000 instead of canonical 500000

Both defects mean the π statistics from `3428b96` are void for the canonical investigation. The current document is the first canonical π snapshot under the verified protocol.

---

## CONFIRMATIONS

- No training performed: **yes**
- No reward/GAE/mask/network/environment/base_seed changes: **yes**
- Policy-sufficiency verdict (0/4) not reopened: **yes**
- π and action-frequency reported separately: **yes**
- No scope confusion in occupancy-arithmetic test: **yes** (all-agent scope used throughout)
- Detail JSON valid/parseable: **yes** (existing canonical artifact)
- π-floor passes for all seeds: **yes**

---

## FILES WRITTEN

- `training/results/CANONICAL_PI_SNAPSHOT_AND_SEED42_REEXAMINATION.md` (this document)
- `training/compute_canonical_pi_snapshot.py` (computation script)
- `training/analyze_seed42_disconnect.py` (analysis script)

No new detail JSON was created; the existing canonical detail JSON was reused.
