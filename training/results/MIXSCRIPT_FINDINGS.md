# Mix-Script Findings — µ-onball, Bounded Scripted-Assistance Intervention

**Date:** 2026-09-18  
**HEAD:** `1f8b3f2`  
**Protocol:** `training/results/MIXSCRIPT_PROTOCOL.md`  
**Scenario:** `academy_3_vs_1_with_keeper_onball`  
**Seeds:** 42, 123, 7, 999  
**Total timesteps:** 50,176 per seed (fresh initialization, no warm-start)

---

## 1. Evaluation Artifacts

| Checkpoint type | Timesteps | Checkpoint pattern |
|-----------------|-----------|-------------------|
| Mix-script-end | 15,000 | `mappo_academy_3_vs_1_with_keeper_onball_seed{N}_mixscript_15000.pt` |
| Final | 50,176 | `mappo_academy_3_vs_1_with_keeper_seed{N}_mixscript_50176.pt` |

> **Note:** Checkpoint filenames are inconsistent across timesteps. The 15k checkpoints include `_onball` in the name, while the 50k checkpoints do not. This reflects the training CLI invocation at the time of each save, not a model difference.

Evaluation JSONs:
- `training/results/f_act_ONBALL-pi_seed{N}_mappo_academy_3_vs_1_with_keeper_onball_seed{N}_mixscript_15000.json`
- `training/results/f_act_ONBALL-pi_seed{N}_mappo_academy_3_vs_1_with_keeper_seed{N}_mixscript_50176.json`

---

## 2. Pure-π Evaluation Results

Protocol: `eval_f_act.py`, arm `ONBALL-pi`, 20 episodes/seed, deterministic=True, no forced actions, full-episode rollout.

### 2.1 t=0 Validity

| Seed | Mix-script-end valid | Final valid |
|------|---------------------:|------------:|
| 42   | 20 / 20              | 20 / 20     |
| 123  | 20 / 20              | 20 / 20     |
| 7    | 20 / 20              | 20 / 20     |
| 999  | 20 / 20              | 20 / 20     |

Aggregate: **80 / 80 = 100%** valid on-ball state at reset for both checkpoints.

### 2.2 Action Selection Rates (per tick)

| Seed | Fresh baseline | Mix-script-end | Final |
|------|---------------:|---------------:|------:|
| 42   | 0.59%          | 0.00%          | 0.00% |
| 123  | 3.63%          | 3.28%          | 4.71% |
| 7    | 0.00%          | 0.88%          | 0.00% |
| 999  | 0.20%          | 1.96%          | 0.10% |
| **Pooled** | **1.10%** | **1.52%** | **1.20%** |

### 2.3 Event Counts

| Seed | Fresh pass_comp | Fresh shot | Fresh goal | Mix-end pass_comp | Mix-end shot | Mix-end goal | Final pass_comp | Final shot | Final goal |
|------|----------------:|-----------:|-----------:|------------------:|-------------:|-------------:|----------------:|-----------:|-----------:|
| 42   | 0               | 0          | 0          | 0                 | 0            | 0            | 0               | 0          | 0          |
| 123  | 0               | 0          | 0          | 10                | 1            | 1            | 18              | 0          | 0          |
| 7    | 0               | 0          | 0          | 0                 | 0            | 0            | 0               | 0          | 0          |
| 999  | 0               | 0          | 0          | 0                 | 0            | 0            | 0               | 0          | 0          |
| **Pooled** | **0** | **0** | **0** | **10** | **1** | **1** | **18** | **0** | **0** |

---

## 3. Judgment Against Pre-Registered Criteria

### 3.1 Primary Criterion

**Criterion:** Post-script unscripted π shows PASS+SHOT combined selection rate exceeding the fresh-training baseline by at least 0.50 percentage points, in at least 3 of 4 seeds independently.

| Seed | Fresh baseline | Mix-script-end | Mix-end target | Mix-end Pass? | Final | Final target | Final Pass? |
|------|---------------:|---------------:|---------------:|:-------------:|------:|-------------:|:-----------:|
| 42   | 0.59%          | 0.00%          | ≥ 1.09%        | **FAIL**      | 0.00% | ≥ 1.09%      | **FAIL**    |
| 123  | 3.63%          | 3.28%          | ≥ 4.13%        | **FAIL**      | 4.71% | ≥ 4.13%      | **PASS**    |
| 7    | 0.00%          | 0.88%          | ≥ 0.50%        | **PASS**      | 0.00% | ≥ 0.50%      | **FAIL**    |
| 999  | 0.20%          | 1.96%          | ≥ 0.70%        | **PASS**      | 0.10% | ≥ 0.70%      | **FAIL**    |

**Mix-script-end result:** 2 of 4 seeds pass. **Primary criterion FAILS** (requires ≥ 3/4 seeds).  
**Final result:** 1 of 4 seeds pass. **Primary criterion FAILS** (requires ≥ 3/4 seeds).

### 3.2 Secondary Criterion

**Criterion:** At least one non-forced `pass_completed` and one non-forced `shot` event under pure π, in at least 2 of 4 seeds.

| Seed | Mix-script-end pass_comp | Mix-script-end shot | Mix-script-end Pass? | Final pass_comp | Final shot | Final Pass? |
|------|-------------------------:|--------------------:|:--------------------:|----------------:|-----------:|:-----------:|
| 42   | 0                       | 0                   | **FAIL**             | 0               | 0          | **FAIL**    |
| 123  | 10                      | 1                   | **PASS**             | 18              | 0          | **FAIL**    |
| 7    | 0                       | 0                   | **FAIL**             | 0               | 0          | **FAIL**    |
| 999  | 0                       | 0                   | **FAIL**             | 0               | 0          | **FAIL**    |

**Mix-script-end result:** 1 of 4 seeds pass. **Secondary criterion FAILS** (requires ≥ 2/4 seeds).  
**Final result:** 0 of 4 seeds pass. **Secondary criterion FAILS** (requires ≥ 2/4 seeds).

---

## 4. Explicit Conclusion

**This is a negative result.**

Bounded mix-script intervention (5% override probability during steps 0–14,999, 60% PASS / 40% SHOT) does not produce sustained PASS/SHOT preference after scripting is removed. The behavior change observed at the mix-script-end checkpoint does not consolidate through the 35,176-step unscripted tail for 3 of 4 seeds.

### 4.1 What the result shows

- **Seed 123 is the only consistent improver.** It shows elevated PASS selection at both checkpoints (3.28% at 15k, 4.71% at 50k) and is the only seed to produce `pass_completed` events (10 at 15k, 18 at 50k). However, it never produces a `shot` event at the final checkpoint.
- **Seeds 42, 7, and 999 revert to or below baseline.** Seed 42 selects zero PASS/SHOT actions at both checkpoints. Seed 7 shows a brief elevation at 15k (0.88%) but collapses to 0.00% by 50k. Seed 999 drops from 1.96% at 15k to 0.10% at 50k, below its own fresh-training baseline of 0.20%.
- **TACKLE dominates the fallback behavior.** At the final checkpoint, seed 42 selects TACKLE 1183 times across 20 episodes (≈58 per episode), while seeds 7 and 999 show zero TACKLE but also zero PASS/SHOT. This indicates the policy finds high-entropy local optima rather than sustaining the scripted football actions.
- **Pooled result is flat.** Pooled PASS+SHOT rate is 1.52% at mix-script-end and 1.20% at final, versus the fresh-training baseline of 1.10%. The intervention produces no meaningful pooled improvement.

### 4.2 What the result does not show

- No seed achieves the pre-registered primary threshold except seed 123 at the final checkpoint (and only barely: 4.71% vs. 4.13% target).
- No seed produces both a `pass_completed` and a `shot` event under pure π at the final checkpoint. Seed 123 produces `pass_completed` events but zero `shot` events.
- No goals are scored under pure π in any seed at either checkpoint.
- The behavior change is not durable: the unscripted tail (35,176 steps) is insufficient to consolidate the scripted demonstrations for most seeds.

---

## 5. Pattern Diagnosis

The results match the protocol's "reversion" judgment rule:

> "If the mix-script-end checkpoint shows elevated PASS/SHOT selection but the final checkpoint reverts toward baseline: the learner adopts the behavior only while scripting biases the data, then reverts. This suggests the unscripted tail is insufficient to consolidate the behavior, or the reward signal is too weak to sustain it."

Evidence:
- **At mix-script-end (15k):** Seeds 123, 7, and 999 all show elevated PASS+SHOT rates relative to baseline. This indicates the policy can represent and select these actions when the rollout distribution is biased.
- **At final (50k):** Only seed 123 sustains elevated PASS selection. Seeds 42, 7, and 999 all regress. The 35,176-step unscripted tail does not consolidate the behavior change for the majority of seeds.
- **Reward signal weakness:** The `AttackingDrillRewardAdapter` provides sparse reward for football events. With exploration bonus β=0.03 and entropy coefficient annealed from 0.01 to 0.005, the signal-to-noise ratio for PASS/SHOT learning is low. The scripted overrides provide dense supervised signal for 15k steps, but once removed, the policy optimizes toward higher-frequency, lower-variance behaviors (TACKLE, IDLE, MOVE) that accumulate more reward ticks.

---

## 6. Comparison to Fresh-Training Baseline

| Metric | Fresh baseline (pooled) | Mix-script-end (pooled) | Final (pooled) |
|--------|------------------------:|------------------------:|---------------:|
| PASS+SHOT rate | 1.10% | 1.52% | 1.20% |
| PASS rate | 1.05% | 1.25% | 1.12% |
| SHOT rate | 0.05% | 0.27% | 0.07% |
| TACKLE rate | 1.86% | 19.49% | 30.34% |
| pass_completed events | 0 | 10 | 18 |
| shot events | 0 | 1 | 0 |
| goals | 0 | 1 | 0 |

The mix-script intervention produces a transient increase in SHOT selection at 15k (0.27% vs. 0.05% baseline) and a small number of `pass_completed` and `shot` events, but these do not survive into the final checkpoint. TACKLE selection explodes at the final checkpoint (30.34% vs. 1.86% baseline), suggesting the policy collapses into a high-frequency local optimum once the scripted demonstrations are removed.

---

## 7. Recommendation

Per the experimental hierarchy and the protocol's judgment rules, the mix-script intervention fails both primary and secondary criteria. Investigation should redirect toward **reward/GAE-level causes** rather than further rollout-distribution interventions.

Specific next steps:
1. **Diagnose reward signal sparsity.** The `AttackingDrillRewardAdapter` provides reward only on goal-scoring events and a small exploration bonus. PASS and SHOT actions are not directly rewarded, so the policy has little gradient signal to prefer them over TACKLE or MOVE.
2. **Examine advantage estimation for rare actions.** With γ=0.99 and λ=0.95, advantages for actions that lead to rare events (PASS, SHOT) may be dominated by bootstrapped value estimates that favor frequent, low-variance actions.
3. **Consider reward shaping for football actions.** Before concluding that paralysis is structural, test whether small positive rewards for legal PASS and SHOT selections (not just completions) change the optimization landscape.
4. **Do not extend the unscripted tail.** The 35,176-step tail already demonstrates that the behavior does not consolidate. Further training without reward/GAE changes is unlikely to change the outcome.

---

## 8. Artifacts

| Artifact | Location |
|----------|----------|
| Protocol | `training/results/MIXSCRIPT_PROTOCOL.md` |
| Checkpoints (15k) | `training/models/mappo_academy_3_vs_1_with_keeper_onball_seed{N}_mixscript_15000.pt` |
| Checkpoints (50k) | `training/models/mappo_academy_3_vs_1_with_keeper_seed{N}_mixscript_50176.pt` |
| Evaluation JSONs (15k) | `training/results/f_act_ONBALL-pi_seed{N}_mappo_academy_3_vs_1_with_keeper_onball_seed{N}_mixscript_15000.json` |
| Evaluation JSONs (50k) | `training/results/f_act_ONBALL-pi_seed{N}_mappo_academy_3_vs_1_with_keeper_seed{N}_mixscript_50176.json` |
| Findings | `training/results/MIXSCRIPT_FINDINGS.md` |

---

## 9. Confirmations

- No reward / GAE / mask / network / spawn changes were made during mix-script training.
- Mix-script only altered the behavior policy during rollout collection (steps 0–14,999).
- Evaluation protocol matches fresh-training and Phase B exactly (ONBALL-pi, 20 episodes/seed, deterministic, full-episode rollout).
- No warm-start from existing checkpoints was used.
