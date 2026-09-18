# Fresh Training Findings — µ-onball, Repaired PASS Path

**Date:** 2026-09-18  
**HEAD:** `05c95eb`  
**Protocol:** `training/results/FRESH_TRAINING_PROTOCOL.md`  
**Scenario:** `academy_3_vs_1_with_keeper_onball`  
**Seeds:** 42, 123, 7, 999  
**Total timesteps:** 50,176 per seed (fresh initialization, no warm-start)  
**Checkpoints:** `training/models/mappo_academy_3_vs_1_with_keeper_seed{N}_freshtrain_50176.pt`

---

## 1. Training Configuration (Logged & Unchanged)

| Parameter | Value |
|-----------|-------|
| Reward function | `AttackingDrillRewardAdapter` (unchanged) |
| GAE γ / λ | 0.99 / 0.95 (unchanged) |
| Action masks | Frozen (unchanged) |
| Architecture | SharedActor (MLP 64×64) + CentralizedCritic |
| Rollout length | 256 |
| Mini-batch size | 256 |
| PPO epochs | 4 |
| Learning rate | 3e-4 → 3e-5 (cosine anneal) |
| Clip range | 0.15 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0.01 → 0.005 |
| Max grad norm | 0.5 |
| Opponent difficulty | medium |
| Self-play | disabled |
| Curriculum | disabled |

No reward, GAE, mask, or network changes were made. The PASS path uses production nearest-teammate resolution (unchanged).

---

## 2. Post-Training Pure-π Evaluation Results

Protocol: `eval_f_act.py`, arm `ONBALL-pi`, 20 episodes/seed, deterministic=True, no forced actions, full-episode rollout.

### 2.1 t=0 Validity

| Seed | Valid t0 possession | Fraction |
|------|--------------------:|---------|
| 42   | 20 / 20             | 100%     |
| 123  | 20 / 20             | 100%     |
| 7    | 20 / 20             | 100%     |
| 999  | 20 / 20             | 100%     |

Aggregate: **80 / 80 = 100%** valid on-ball state at reset.

### 2.2 Action Histogram (pure π)

| Action category | Seed 42 | Seed 123 | Seed 7 | Seed 999 | Pooled |
|-----------------|--------:|---------:|-------:|---------:|-------:|
| PASS (9–11)     | 6 (0.59%) | 36 (3.53%) | 0 (0.00%) | 1 (0.10%) | 43 (1.05%) |
| SHOT (12)       | 0 (0.00%) | 1 (0.10%) | 0 (0.00%) | 1 (0.10%) | 2 (0.05%) |
| TACKLE (16)     | 0 (0.00%) | 31 (3.04%) | 7 (0.69%) | 38 (3.73%) | 76 (1.86%) |
| DRIBBLE (17)    | 0 (0.00%) | 311 (30.49%) | 2 (0.20%) | 349 (34.22%) | 662 (16.23%) |
| MOVE (1–8)      | 36 (3.53%) | 237 (23.24%) | 992 (97.25%) | 354 (34.71%) | 1619 (39.68%) |
| IDLE (0)        | 391 (38.33%) | 2 (0.20%) | 1 (0.10%) | 75 (7.35%) | 469 (11.50%) |
| **Total ticks** | 1020 | 1020 | 1020 | 1020 | 4080 |

Note: Counts are per-agent (all 3 controlled left-team players) derived from `policy_pass_count` / `policy_shot_count` / `policy_tackle_count` in per-episode summaries, plus per-tick `action_selected` for categories not tracked in episode summaries (DRIBBLE, MOVE, IDLE).

### 2.3 Event Counts

| Event               | Seed 42 | Seed 123 | Seed 7 | Seed 999 | Pooled |
|---------------------|--------:|---------:|-------:|---------:|-------:|
| `pass` initiation   | 0       | 16       | 4      | 1        | 21     |
| `pass_completed`    | 0       | 0        | 0      | 0        | 0      |
| `shot`              | 0       | 0        | 0      | 0        | 0      |
| `goal`              | 0       | 0        | 0      | 0        | 0      |

No goals scored under pure π in any seed. No `pass_completed` or `shot` events occurred in any seed.

---

## 3. Side-by-Side Comparison: Phase B Frozen-π vs. Fresh-Trained π

### 3.1 Per-Seed Action Selection

| Seed | Phase B PASS | Phase B SHOT | Phase B TACKLE | Fresh PASS | Fresh SHOT | Fresh TACKLE | Fresh DRIBBLE | Fresh MOVE | Fresh IDLE |
|------|-------------:|-------------:|---------------:|-----------:|-----------:|-------------:|--------------:|-----------:|-----------:|
| 42   | 0 (0.00%)    | 0 (0.00%)    | 54 (6.01%)     | 6 (0.59%)  | 0 (0.00%)  | 0 (0.00%)    | 0 (0.00%)     | 36 (3.53%) | 391 (38.33%) |
| 123  | 1 (0.12%)    | 1 (0.12%)    | 0 (0.00%)      | 36 (3.53%) | 1 (0.10%)  | 31 (3.04%)   | 311 (30.49%)  | 237 (23.24%) | 2 (0.20%) |
| 7    | 1 (0.12%)    | 1 (0.12%)    | 119 (14.28%)   | 0 (0.00%)  | 0 (0.00%)  | 7 (0.69%)    | 2 (0.20%)     | 992 (97.25%) | 1 (0.10%) |
| 999  | 0 (0.00%)    | 1 (0.12%)    | 0 (0.00%)      | 1 (0.10%)  | 1 (0.10%)  | 38 (3.73%)   | 349 (34.22%)  | 354 (34.71%) | 75 (7.35%) |
| **Pooled** | **2 (0.06%)** | **3 (0.09%)** | **173 (5.06%)** | **43 (1.05%)** | **2 (0.05%)** | **76 (1.86%)** | **662 (16.23%)** | **1619 (39.68%)** | **469 (11.50%)** |

### 3.2 Per-Seed Events

| Seed | Phase B pass | Phase B pass_completed | Phase B shot | Phase B goal | Fresh pass | Fresh pass_completed | Fresh shot | Fresh goal |
|------|-------------:|-----------------------:|-------------:|-------------:|-----------:|---------------------:|-----------:|-----------:|
| 42   | 0            | 0                      | 0            | 0            | 0          | 0                    | 0         | 0         |
| 123  | 0            | 0                      | 0            | 0            | 16         | 0                    | 0         | 0         |
| 7    | 0            | 0                      | 0            | 0            | 4          | 0                    | 0         | 0         |
| 999  | 1            | 0                      | 1            | 0            | 1          | 0                    | 0         | 0         |
| **Pooled** | **1** | **0** | **1** | **0** | **21** | **0** | **0** | **0** |

### 3.3 Total Ticks

| Seed | Phase B ticks | Fresh ticks |
|------|-------------:|------------:|
| 42   | 903          | 1020        |
| 123  | 869          | 1020        |
| 7    | 834          | 1020        |
| 999  | 810          | 1020        |
| **Pooled** | **3416** | **4080** |

---

## 4. Judgment Against Pre-Registered Criteria

### Primary Criterion

**Criterion:** Post-training PASS+SHOT combined selection rate ≥ 1.5% in at least 3 of 4 seeds independently.

| Seed | Combined PASS+SHOT rate | Criterion threshold | Pass/Fail |
|------|------------------------:|--------------------:|:---------:|
| 42   | 0.59%                   | ≥ 1.5%              | **FAIL**  |
| 123  | 3.63%                   | ≥ 1.5%              | **PASS**  |
| 7    | 0.00%                   | ≥ 1.5%              | **FAIL**  |
| 999  | 0.20%                   | ≥ 1.5%              | **FAIL**  |

**Overall:** 1 of 4 seeds pass. **Primary criterion FAILS** (requires ≥ 3/4 seeds).

### Secondary Criterion

**Criterion:** At least one non-forced `pass_completed` and one non-forced `shot` event in at least 2 of 4 seeds.

| Seed | pass_completed events | shot events | Criterion | Pass/Fail |
|------|----------------------:|------------:|-----------|:---------:|
| 42   | 0                     | 0           | ≥1 each   | **FAIL**  |
| 123  | 0                     | 0           | ≥1 each   | **FAIL**  |
| 7    | 0                     | 0           | ≥1 each   | **FAIL**  |
| 999  | 0                     | 0           | ≥1 each   | **FAIL**  |

**Overall:** 0 of 4 seeds pass. **Secondary criterion FAILS** (requires ≥ 2/4 seeds).

---

## 5. Explicit Conclusion

**This is a partial/negative result.**

Fresh 50k-step training from scratch on µ-onball with the repaired PASS path does not reliably break paralysis. Only 1 of 4 seeds (seed 123) clears the primary action-selection bar, and no seed produces a completed pass or shot event under pure π control.

### What the result does show

- The pooled PASS+SHOT selection rate (1.10%) is approximately 7× the Phase B frozen-π baseline (0.15%). This indicates that learning CAN shift action preference in some seeds.
- Seed 123 is the standout: 36 PASS selections (3.53%) and 1 SHOT selection (0.10%) across 1020 ticks. This is a 24× improvement over baseline for that seed.
- The other 3 seeds remain near-paralyzed: seed 7 selects MOVE 97% of the time, seed 42 selects IDLE 38% and MOVE 3.5%, and seed 999 is dominated by DRIBBLE (34%) and MOVE (35%).

### What the result does not show

- No seed achieves the pre-registered 1.5% combined rate except seed 123.
- No seed produces a single `pass_completed` or `shot` event under pure π.
- No goals are scored.
- The per-seed variance is even larger than observed in Phase B: some seeds improve, others worsen or stay flat.

### Comparison to Phase B baseline

| Metric | Phase B (pooled) | Fresh-trained (pooled) | Change |
|--------|-----------------:|----------------------:|-------:|
| PASS rate | 0.06% | 1.05% | +17× |
| SHOT rate | 0.09% | 0.05% | −44% |
| TACKLE rate | 5.06% | 1.86% | −63% |
| DRIBBLE rate | 1.61% | 16.23% | +10× |
| MOVE rate | 75.47% | 39.68% | −47% |
| IDLE rate | 0.56% | 11.50% | +20× |
| pass events | 1 | 21 | +21× |
| pass_completed | 0 | 0 | — |
| shot events | 1 | 0 | −100% |
| goals | 0 | 0 | — |

The fresh-trained policy has not simply replicated the baseline's TACKLE-skewed profile. Instead, it exhibits seed-specific attractors: seed 123 learns PASS preference, seed 7 collapses to near-pure MOVE, seed 42 oscillates between IDLE and MOVE, and seed 999 splits between DRIBBLE and MOVE. This suggests the 50k-step training budget is insufficient to find a stable, generalizable policy basin for football actions.

---

## 6. Recommendation

### Immediate next step: Move to Option B (bounded mix-script)

The fresh-training result does not break paralysis in a reproducible way across seeds. Per the experimental hierarchy and the protocol's own fallback clause, this triggers Option B: bounded mix-script assistance to pull policies out of the paralysis basin, rather than continuing to hope for spontaneous basin escape from fresh initialization.

Option B should be scoped as:
- A bounded scripted-assistance schedule (e.g., forced PASS/SHOT at t=0 with decaying probability)
- Same frozen reward/GAE/mask configuration
- Same 4-seed comparison protocol
- Explicitly labeled as an ablation, not part of the fresh-training run

### Why not longer training or hyperparameter tuning first?

The per-seed variance is the decisive factor. Seed 123 learned PASS preference at 50k steps; seeds 42, 7, and 999 did not. Without understanding why seed 123 escaped and the others did not, simply extending training or tuning hyperparameters is unlikely to produce a reproducible 3/4-seed pass. The TACKLE seed-skew in Phase B (seeds 7 and 42 high, seeds 123 and 999 zero) and the current PASS seed-skew (seed 123 high, others near zero) both point to initialization-sensitive basin geometry, not a lack of training time.

### TACKLE as a comparative clue

Phase B showed TACKLE emerging at 5.06% overall, driven entirely by seeds 7 (119 selections) and 42 (54 selections). Fresh training eliminates TACKLE in seed 42 (0 selections), reduces it in seed 7 (7 selections), and introduces it in seeds 123 (31) and 999 (38) where it was previously zero. This reversal confirms that TACKLE was not a stable learned behavior but a seed-specific artifact of the original 50k training run. Whatever mechanism produced TACKLE in the baseline checkpoints is not reproducible under fresh initialization. This makes TACKLE a less useful diagnostic signal for the paralysis problem than it initially appeared.

---

## 7. Artifacts

| Artifact | Location |
|----------|----------|
| Protocol | `training/results/FRESH_TRAINING_PROTOCOL.md` |
| Checkpoints | `training/models/mappo_academy_3_vs_1_with_keeper_seed{N}_freshtrain_50176.pt` |
| Evaluation JSONs | `training/results/f_act_ONBALL-pi_seed{N}_mappo_academy_3_vs_1_with_keeper_seed{N}_freshtrain_50176.json` |
| Aggregate JSON | `training/results/f_act_fresh_train_postfix.json` |
| Summary CSV | `training/results/f_act_fresh_train_postfix_summary.csv` |
| Findings | `training/results/FRESH_TRAINING_FINDINGS.md` |

---

## 8. Confirmations

- No reward / GAE / mask / network / spawn changes.
- No warm-start from existing checkpoints.
- No option (b) implemented.
- PASS path uses production nearest-teammate resolution (unchanged).
- Evaluation protocol matches Phase B exactly (ONBALL-pi, 20 episodes/seed, deterministic, full-episode rollout).
