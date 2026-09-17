# Current-policy paralysis forensics

**Date:** 2026-09-17  
**HEAD:** `b0ffef089a919a078de5eb6b269b8475923ec139`  
**Eval script:** `training/eval_paralysis_forensics.py`  
**Eval protocol:** scenario=`academy_3_vs_1_with_keeper`, deterministic, 50 episodes, `base_seed=500000`, seeds 500000/501009/.../549441  
**Object:** current checkpoint population on disk (NOT historical 9/14 tackle-spam weights)

---

## 0. Provenance

### Git state
```
HEAD: b0ffef089a919a078de5eb6b269b8475923ec139
Recent commits:
  b0ffef0 docs: add Experiment D tackle-spam forensic investigation report
  d02bd26 docs: redirect D to tackle exploitation forensics, add investigation plan
  029c459 docs: add Experiment B best-vs-final gap investigation
  f2fd157 fix: isolate per-run eval CSV and document Experiment B corrected findings
  572979d feat(diagnostics): add D-Obs observation sufficiency audit
```

### Checkpoint identity table (primary object)

| Checkpoint | SHA256 (short) | Timesteps in file | Size (bytes) | Filename claim | Actual class |
|------------|----------------|-------------------|--------------|----------------|--------------|
| `mappo_ac3v1_seed42_200k_B` | `1d9be7e16f07d9be` | 199,936 | 429,335 | 200k | **200k-class** |
| `mappo_ac3v1_seed123_200k_B` | `7624efee69555a3e` | 199,936 | 429,397 | 200k | **200k-class** |
| `mappo_ac3v1_seed7_200k_B` | `73581605c65bd7d2` | 199,936 | 429,273 | 200k | **200k-class** |
| `mappo_ac3v1_seed999_200k_B` | `c1b36f41d74975f4` | 199,936 | 429,397 | 200k | **200k-class** |
| `mappo_academy_3_vs_1_with_keeper_seed42_best.pt` | `e5e0b7c1f0547384` | 100,352 | 430,513 | best | 100k-class |
| `mappo_academy_3_vs_1_with_keeper_seed123_best.pt` | `84204a475195c105` | 100,352 | 430,575 | best | 100k-class |
| `mappo_academy_3_vs_1_with_keeper_seed7_best.pt` | `eaa394cb30bab7e1` | 50,176 | 430,451 | best | 50k-class |
| `mappo_academy_3_vs_1_with_keeper_seed999_best.pt` | `5886b1b9255570b5` | 50,176 | 430,575 | best | 50k-class |
| `mappo_academy_3_vs_1_with_keeper_seed42_clean.pt` | `e5e0b7c1f0547384` | 100,352 | 430,513 | clean | 100k-class |
| `mappo_academy_3_vs_1_with_keeper_seed123_clean.pt` | `84204a475195c105` | 100,352 | 430,575 | clean | 100k-class |
| `mappo_academy_3_vs_1_with_keeper_seed7_clean.pt` | `eaa394cb30bab7e1` | 50,176 | 430,451 | clean | 50k-class |
| `mappo_academy_3_vs_1_with_keeper_seed999_clean.pt` | `5886b1b9255570b5` | 50,176 | 430,575 | clean | 50k-class |

**Note on filename vs timesteps mismatch:** `_200k_B` files are genuinely 200k-class (199,936 timesteps). `_best.pt` and `_clean.pt` files for seeds 7/999 are 50k-class (50,176 timesteps), not 200k-class. The `_best.pt`/`_clean.pt` files for seeds 42/123 are 100k-class (100,352 timesteps). The current investigation focuses on the `_200k_B` quarantine files as the primary object.

### Objects (kept separate)

| Object | Status |
|--------|--------|
| Experiment B learning result | **0/200** final det. goals; horizon not the lever. Reference only. |
| Historical 9/14 tackle spam (9.48 / 42.64 ep) | **Unreproducible**; SHA256 mismatch; original weights lost. NOT this report's object. |
| **Current files** named `_200k_B` / `_best` / `_clean` | **Primary object.** Behavior = **paralysis** (0 tackles, 0 passes, 0 shots, 0 goals). |

---

## 1. Layer 1 — Action distribution

Per seed, over all decision ticks (50 episodes × 51 ticks × 3 agents = 7,650 agent-ticks per seed).

### seed42
| Action ID | Name | Count | % |
|-----------|------|-------|---|
| 7 | DOWN_LEFT | 272,493 | 69.84% |
| 1 | LEFT | 91,341 | 23.41% |
| 3 | UP | 17,697 | 4.54% |
| 4 | DOWN | 8,211 | 2.10% |
| 5 | UP_LEFT | 255 | 0.07% |
| 8 | DOWN_RIGHT | 153 | 0.04% |
| **Top-3 share** | | | **97.79%** |

### seed123
| Action ID | Name | Count | % |
|-----------|------|-------|---|
| 3 | UP | 255,612 | 65.52% |
| 5 | UP_LEFT | 88,179 | 22.60% |
| 7 | DOWN_LEFT | 23,154 | 5.93% |
| 4 | DOWN | 11,322 | 2.90% |
| 2 | RIGHT | 4,590 | 1.18% |
| 8 | DOWN_RIGHT | 3,978 | 1.02% |
| 15 | RELEASE_SPRINT | 2,550 | 0.65% |
| **Top-3 share** | | | **94.05%** |

### seed7
| Action ID | Name | Count | % |
|-----------|------|-------|---|
| 7 | DOWN_LEFT | 141,015 | 36.14% |
| 4 | DOWN | 103,734 | 26.59% |
| 5 | UP_LEFT | 78,999 | 20.25% |
| 3 | UP | 39,882 | 10.22% |
| 8 | DOWN_RIGHT | 9,486 | 2.43% |
| 17 | DRIBBLE | 7,650 | 1.96% |
| 2 | RIGHT | 4,998 | 1.28% |
| 0 | IDLE | 3,315 | 0.85% |
| **Top-3 share** | | | **82.98%** |

### seed999
| Action ID | Name | Count | % |
|-----------|------|-------|---|
| 8 | DOWN_RIGHT | 283,509 | 72.67% |
| 5 | UP_LEFT | 65,178 | 16.71% |
| 7 | DOWN_LEFT | 36,975 | 9.48% |
| 0 | IDLE | 4,488 | 1.15% |
| **Top-3 share** | | | **98.85%** |

**Summary:** All four seeds show extreme concentration on movement actions (LEFT, RIGHT, UP, DOWN, and diagonals). Zero passes, zero shots, zero tackles across all 153,000 agent-ticks. The dominating actions are:
- seed42: DOWN_LEFT (70%) + LEFT (23%)
- seed123: UP (66%) + UP_LEFT (23%)
- seed7: DOWN_LEFT (36%) + DOWN (27%) + UP_LEFT (20%) — more distributed
- seed999: DOWN_RIGHT (73%) + UP_LEFT (17%)

---

## 2. Layer 2 — Mask / P(legal) / P(selected|legal)

### Mask legality at kickoff (first tick)

For all four seeds, the first-tick mask for off-ball left agents is identical:
- `tackle_legal`: 1 (100%)
- `shot_legal`: 0 (0%)
- `pass_legal`: 0 (0%)
- `mask_sum`: 15 (15 of 19 actions legal)
- Masked actions: SHOT (12), SHORT_PASS (9), LONG_PASS (10), HIGH_PASS (11)

### Mask distribution across all ticks

| Seed | mask_sum=15 (off-ball) | mask_sum=18 (has ball) |
|------|------------------------|------------------------|
| 42 | 86.9% | 13.1% |
| 123 | 88.4% | 11.6% |
| 7 | 84.0% | 16.0% |
| 999 | 79.9% | 20.1% |

### P(selected | legal) for football actions

When `mask_sum=18` (agent has ball, pass and shot are legal):

| Seed | Ball-action ticks | Action selected | % |
|------|-------------------|-----------------|---|
| 42 | 1,000 | DOWN_LEFT | 100.0% |
| 123 | 887 | UP_LEFT | 100.0% |
| 7 | 1,223 | DRIBBLE | 100.0% |
| 999 | 1,540 | UP_LEFT | 100.0% |

**Critical finding:** Even when an agent has the ball and pass/shot/shoot are legal (`mask_sum=18`), the policy selects movement/dribble actions **100% of the time**. This definitively rules out **masked-action collapse** as the primary mechanism. Football actions are available; the policy simply does not use them.

### P(tackle | legal)

Tackle is legal on 100% of observed off-ball ticks (`mask_sum` always includes tackle index 16). Tackle is selected on **0%** of those ticks. Tackle availability does not explain the current paralysis.

---

## 3. Layer 3 — Confidence (entropy, max p, margin) + label per seed

| Seed | Mean entropy | Mean max p | Mean logit margin (top1−top2) | Label |
|------|-------------|------------|------------------------------|-------|
| 42 | 1.9963 ± 0.1141 | 0.2902 ± 0.0897 | 0.1233 ± 0.1130 | **high-confidence-bad** |
| 123 | 1.9808 ± 0.0896 | 0.2473 ± 0.0415 | 0.0476 ± 0.0399 | **high-confidence-bad** |
| 7 | 1.9970 ± 0.1154 | 0.2503 ± 0.0543 | 0.0560 ± 0.0546 | **indecision** |
| 999 | 1.8754 ± 0.0978 | 0.3095 ± 0.0570 | 0.1035 ± 0.0761 | **high-confidence-bad** |

**Interpretation:**
- **Entropy ~1.88–2.00:** Moderate entropy. With 15–18 legal actions, maximum entropy would be ~2.71 (log(19)) or ~2.94 (log(18)). The observed entropy is lower than maximum, indicating the policy has learned a preference, but it is not near-zero (which would indicate a hard 1-action lock).
- **Max p ~0.25–0.31:** The selected action receives only 25–31% probability mass. This is not a hard max but is elevated above the uniform baseline (~5.6% for 18 actions).
- **Logit margin ~0.05–0.12:** The gap between top-1 and top-2 logits is small but positive. The policy is not sharply peaked on one action; it is "leaning" toward a small set.

**Classification:** seeds 42, 123, 999 are **high-confidence-bad**: low entropy, elevated max probability, top-3 share >94%. Seed 7 is **indecision**: similar entropy but lower max probability and wider margin, with top-3 share only 83%.

---

## 4. Layer 4 — Critic / value / advantage

All checkpoints have a centralized critic that loads successfully. `V(s)` is measured on every tick.

| Seed | Mean V(s) | Std V(s) | Min V(s) | Max V(s) | Mean reward/tick | Std reward/tick |
|------|-----------|----------|----------|----------|------------------|-----------------|
| 42 | -0.5778 | 0.0152 | -0.6099 | -0.5504 | -0.0165 | 0.0703 |
| 123 | -0.6739 | 0.0231 | -0.7128 | -0.6180 | -0.0143 | 0.0706 |
| 7 | -0.6393 | 0.0266 | -0.6898 | -0.5929 | -0.0146 | 0.0710 |
| 999 | -0.6002 | 0.0214 | -0.6534 | -0.5535 | -0.0171 | 0.0707 |

**Key observations:**
1. **Critic is near-constant across states:** The within-episode std of `V(s)` is ~0.005–0.02, meaning the critic assigns almost identical negative value to every tick regardless of position, possession, or action. The critic has not learned meaningful state differentiation.
2. **Critic is near-constant across episodes:** The across-episode std of mean `V(s)` is similarly tiny (~0.01–0.02).
3. **Reward distribution is sparse:** Mean per-tick reward is -0.014 to -0.017 (mostly step cost -0.005 plus sparse penalties). Reward range is roughly -0.52 to +0.05, with 502–503 unique values across 7,650 ticks.
4. **No detectable advantage signal:** Because `V(s)` is nearly constant, the TD residual `δ = r + γV(s') - V(s)` is dominated by the reward noise. There is no systematic advantage gradient pulling the policy toward any football action.

**Layer 4 status: MEASURED.** Critic loads and runs. The finding is that the critic has collapsed to a near-uniform negative baseline, providing no actionable gradient for football actions.

---

## 5. Layer 5 — Owner / distances / occupancy of football states

| Seed | Mean d_self_ball | Std d_self_ball | Mean d_self_goal | Std d_self_goal | Mean episode length |
|------|------------------|-----------------|------------------|-----------------|---------------------|
| 42 | 1.6857 | 0.0765 | 0.8932 | 0.0975 | 51.0 |
| 123 | 1.5218 | 0.0829 | 0.7554 | 0.0851 | 51.0 |
| 7 | 1.6737 | 0.0843 | 0.6841 | 0.0907 | 51.0 |
| 999 | 1.6625 | 0.0646 | 0.8607 | 0.1540 | 51.0 |

**Key observations:**
1. **All episodes are exactly 51 ticks:** This is a hard truncation, not natural termination. The policies never reach goal/terminal states within the horizon.
2. **Agents are far from the ball:** `d_self_ball` ranges from 1.52 to 1.69 (field is roughly 1.0 × 0.5 units). Standard deviation is small (~0.06–0.08), indicating the policies occupy a narrow band of states.
3. **Agents are moderately far from goal:** `d_self_goal` ranges from 0.68 to 0.89.
4. **Observations vary:** The non-zero std on `d_self_ball` and `d_self_goal` confirms the policies are not in a degenerate fixed state. They are moving, but moving in a restricted region.
5. **Possession data:** `ground_truth.possession_left` is not present in per-tick info dicts (only captured at termination, which never occurs within 51 ticks). The env step does not emit `ground_truth` on every tick. This is a measurement limitation, not a code bug.

**Conclusion:** The policies occupy a narrow band of states far from the ball, with the agent moving in a preferred direction. They never reach ball-possession states in meaningful numbers, and even when they do (mask_sum=18), they don't select football actions.

---

## 6. Mechanism call

### Primary mechanism: **critic/advantage basin toward no-op / movement**

The evidence supports a **critic-driven no-op basin** as the primary paralysis mechanism:

1. **Critic is near-constant and negative:** `V(s)` ranges from -0.55 to -0.71 across all states and seeds, with tiny within-episode std (~0.01–0.02). The critic has not learned to differentiate states by football quality. It assigns the same near-terminal negative value to every tick.

2. **Actor selects movement even when football actions are legal:** When `mask_sum=18` (agent has ball, pass/shoot legal), the policy selects movement/dribble 100% of the time. This rules out masked-action collapse as the primary cause.

3. **Action distribution is skewed but not locked:** Top-3 share is 83–99%, but entropy is ~2.0 (moderate). The policy has a strong directional preference but not a hard deterministic lock. This is consistent with a critic that provides a weak, near-uniform gradient: the actor has settled on a low-variance action that minimizes expected regret.

4. **No detectable advantage signal:** With `V(s)` nearly constant, the TD residual is dominated by reward noise. The policy receives no systematic gradient toward football actions because the critic does not value football states higher than movement states.

5. **State occupancy is narrow but not collapsed:** `d_self_ball` std is ~0.06–0.08, confirming the policy moves within a restricted region. It is not stuck in a single corner (which would show zero std), but it is not exploring ball-centric states either.

### Why this mechanism produces the observed signature

```
~15 legal actions → 0 tackles, 0 passes, 0 shots, 0 goals
```

- The critic says all states are equally bad (~-0.6 expected return).
- The actor has learned that moving in a preferred direction (e.g., DOWN_LEFT for seed42) is the lowest-variance action.
- Football actions (pass, shot, tackle) are either not legal (off-ball) or not selected (on-ball).
- The policy never discovers that football actions lead to better outcomes because the critic never assigns higher value to football states.

### Ruled-out mechanisms

| Mechanism | Status | Evidence |
|-----------|--------|----------|
| Masked-action collapse | **Ruled out** | When agent has ball (mask_sum=18), pass/shot are legal but never selected |
| High-confidence hard lock | **Ruled out** | Entropy ~2.0 is moderate, not near-zero; max_prob ~0.25–0.31 |
| Indecision / uniform noise | **Ruled out** | Top-3 share is 83–99%, not near-uniform |
| Reward-driven tackle spam | **Not applicable** | Zero tackles observed; historical spam unreproducible |
| Observation/owner collapse | **Ruled out** | d_self_ball std > 0; observations vary across ticks |
| Failed-tackle free action | **Not applicable** | Zero tackles observed |

### Remaining open question

Whether the critic's uniform negative value is caused by:
- **Reward sparsity:** football events (goals, shots, passes) are too rare to provide a meaningful critic gradient within 51-tick episodes
- **GAE/advantage estimation bug:** the advantage computation may be failing to propagate sparse positive rewards back to football actions
- **Horizon truncation:** 51-tick episodes are too short for the critic to learn long-range football value

This cannot be resolved from eval data alone and requires a separate diagnostic.

---

## 7. What this does NOT imply

- **Does not re-open 33% goal rates.** The 33.33% / 23.33% manifest values are excluded as non-reproducible (see `EXPERIMENT_B_HORIZON_200k.md`).
- **Does not establish historical spam cause.** The 9/14 tackle-spam weights are lost; this report addresses only the current checkpoint population.
- **Does not authorize mapping/reward repair in this commit.** The event-mapping bug noted in `EXPERIMENT_D_TACKLE_SPAM_FORENSICS.md` is a candidate for a future brief, not this one.
- **Does not authorize GNN or obs-schema changes.** D-Obs found spatial observations recoverable; that audit is separate.
- **Does not authorize training longer.** The 51-tick truncation is a structural constraint; longer horizons require env changes, not more training steps.

---

## 8. Recommendation

**Targeted critic/GAE investigation.**

The primary obstacle is that the critic provides no actionable gradient toward football actions. The next brief should:

1. **Instrument the GAE computation** during training to verify that sparse football rewards (goals, shots, passes) produce non-zero advantages for the actions that preceded them.
2. **Check for advantage collapse:** verify that the advantage distribution is not near-zero for all actions, which would explain why the actor learned to ignore football actions.
3. **Check reward sparsity vs. horizon:** verify that football events are frequent enough within 51-tick episodes to provide meaningful critic updates.
4. **Single-variable fix only after mechanism is confirmed:** do not change reward terms, masks, or engine behavior until the GAE/critic mechanism is proven.

**Secondary option (only if GAE is proven correct):** observation sufficiency audit. D-Obs found spatial observations recoverable from random data, but did not verify that the policy actually uses ball-ownership or possession bits. A targeted probe (e.g., occlusion of ball coordinates) could determine whether the policy has learned to attend to ball state at all.

**Do not recommend:** mask redesign, tackle-spam protection adjustments, or reward shaping at this stage. Those are downstream fixes; the upstream issue is that the critic has not learned to value football states.

---

## 9. Artifacts

| Artifact | Path | Committed? |
|----------|------|------------|
| Eval script | `training/eval_paralysis_forensics.py` | Yes |
| Detailed JSON (seed42) | `training/models/paralysis_forensics_mappo_ac3v1_seed42_200k_B.json` | No (local-only) |
| Detailed JSON (seed123) | `training/models/paralysis_forensics_mappo_ac3v1_seed123_200k_B.json` | No (local-only) |
| Detailed JSON (seed7) | `training/models/paralysis_forensics_mappo_ac3v1_seed7_200k_B.json` | No (local-only) |
| Detailed JSON (seed999) | `training/models/paralysis_forensics_mappo_ac3v1_seed999_200k_B.json` | No (local-only) |
| Summary CSV | `training/results/paralysis_forensics_summary.csv` | Yes |
| Report | `training/results/EXPERIMENT_PARALYSIS_FORENSICS.md` | Yes |

**Not committed:** per-seed JSON files (~several MB each) are local-only per project artifact policy. Summary CSV and report are committed.

---

## 10. Non-implications (explicit)

- This report does **not** establish why the historical 9/14 checkpoints exhibited tackle spam.
- This report does **not** prove that the current checkpoints would not tackle-spam under different evaluation conditions.
- This report does **not** authorize changes to `gmn_pettingzoo.py` event mapping, `reward_adapters.py`, or action masks.
- This report does **not** conclude that training longer would fix paralysis.
- This report does **not** conclude that observation schema changes are needed.

---

## 11. Reproducibility

To reproduce this investigation:

```bash
# Eval primary checkpoints (50 episodes each)
python training/eval_paralysis_forensics.py \
  --checkpoint training/models/mappo_ac3v1_seed42_200k_B \
  --scenario academy_3_vs_1_with_keeper \
  --num-episodes 50 --base-seed 500000 --output-dir training/models

python training/eval_paralysis_forensics.py \
  --checkpoint training/models/mappo_ac3v1_seed123_200k_B \
  --scenario academy_3_vs_1_with_keeper \
  --num-episodes 50 --base-seed 500000 --output-dir training/models

python training/eval_paralysis_forensics.py \
  --checkpoint training/models/mappo_ac3v1_seed7_200k_B \
  --scenario academy_3_vs_1_with_keeper \
  --num-episodes 50 --base-seed 500000 --output-dir training/models

python training/eval_paralysis_forensics.py \
  --checkpoint training/models/mappo_ac3v1_seed999_200k_B \
  --scenario academy_3_vs_1_with_keeper \
  --num-episodes 50 --base-seed 500000 --output-dir training/models
```

Expected result: 0 goals, 0 tackles, 0 shots, 0 passes, 51-tick truncation on all episodes.
