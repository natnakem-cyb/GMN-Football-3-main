# Mix-Script Protocol (Option B) — µ-onball, Repaired PASS Path

**Date:** 2026-09-18  
**HEAD:** `05c95eb`  
**Scenario:** `academy_3_vs_1_with_keeper_onball`  
**Seeds:** 42, 123, 7, 999  
**Total timesteps:** 50,176 per seed (fresh initialization, no warm-start)  
**Checkpoints:** `mappo_academy_3_vs_1_with_keeper_seed{N}_mixscript_50176.pt`  
**Mix-script-end checkpoint:** `mappo_academy_3_vs_1_with_keeper_seed{N}_mixscript_15000.pt`

---

## 1. Frozen Configuration (Unchanged from Fresh-Training Baseline)

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

**Mix-script only alters the behavior policy used to generate rollout data.** Reward computation, GAE advantage estimation, and action masks are identical to the fresh-training run.

---

## 2. Script Mechanism

### 2.1 Trigger

During the **mix-script window** (global steps 0 through 14,999 inclusive), each on-ball tick where the left team (controlled agents) has ball possession is subject to a scripted override with probability **p = 0.05 (5%)**.

"On-ball tick" is defined as any rollout tick where at least one controlled agent's observation slice `obs[94:97]` indicates left-team ball ownership (`obs[95] == 1.0`). This is a conservative proxy: the left team has the ball, and one of the controlled agents is the carrier.

### 2.2 Override Action Selection

When a tick is selected for override:

1. **Identify legal scripted actions:** Examine the action mask of each controlled agent. Collect agents with legal PASS (indices 9, 10, or 11) and agents with legal SHOT (index 12).
2. **Choose action type:**
   - If both PASS and SHOT are legal for at least one agent: choose PASS with probability 0.60, SHOT with probability 0.40.
   - If only PASS is legal: choose PASS.
   - If only SHOT is legal: choose SHOT.
   - If neither is legal: skip override for this tick (count as a non-override).
3. **Choose target agent:** Uniformly at random among controlled agents for whom the chosen action type is legal.
4. **Execute override:** Replace the policy's sampled action for that agent with the scripted action index.

### 2.3 What "Successful" Means for Scripted Actions

A scripted override is considered **successful** if, on the environment tick immediately following the override:
- The engine emits a `pass` or `pass_completed` event for a PASS override, OR
- The engine emits a `shot` event for a SHOT override.

This is the same success definition used in the force-arm diagnostics (`eval_f_act.py`). No new mechanics are introduced; the production nearest-teammate PASS resolution and existing SHOT mechanics are used unchanged.

### 2.4 Logprob Handling

When an action is overridden:
- The stored `logprob` for that agent is recomputed by a forward pass through the actor with the overridden action. This keeps the surrogate objective approximately on-policy for the overridden transition.
- Non-overridden agents retain their original policy logprobs.

---

## 3. Bound on Duration

| Phase | Global Steps | Description |
|-------|-------------|-------------|
| **Mix-script window** | 0 – 14,999 | Scripted overrides active at 5% per on-ball tick |
| **Unscripted tail** | 15,000 – 50,176 | No overrides; pure policy rollouts only |

The unscripted tail is **35,176 steps** (≈70% of total training). This is a substantial period for the learner's own optimization to carry forward any behavior change induced by the scripted demonstrations.

---

## 4. What Stays Frozen

- **Reward function:** `AttackingDrillRewardAdapter` — identical hyperparameters, identical event detection.
- **GAE γ/λ:** 0.99 / 0.95 — identical to fresh-training baseline.
- **Action masks:** Engine-computed legality masks — unchanged. Scripted actions are only injected when they are already legal per the mask.
- **Network architecture:** SharedActor (MLP 64×64) + CentralizedCritic — unchanged.
- **Hyperparameters:** All PPO hyperparameters identical to fresh-training run.

**Mix-script is solely an intervention on the behavior policy during rollout collection.** It does not modify reward shaping, advantage computation, or the learned policy architecture.

---

## 5. Seeds

Same four seeds as fresh-training baseline: **42, 123, 7, 999**.

All four seeds receive identical mix-script treatment. No seed is excluded or treated differently based on its fresh-training result.

---

## 6. Checkpoints

| Checkpoint | When Saved | Purpose |
|------------|-----------|---------|
| `_mixscript_15000.pt` | Global step 15,000 (end of mix-script window) | Isolate "does behavior differ immediately when crutch is removed" |
| `_mixscript_50176.pt` | Global step 50,176 (end of training) | Final unscripted evaluation; primary success criterion |

---

## 7. Pre-Registered Success Criteria

### 7.1 Primary Criterion

Post-script unscripted π (evaluated identically to fresh-training Phase C: pure-π, no force, 20 episodes/seed, deterministic) shows **PASS+SHOT combined selection rate exceeding the fresh-training baseline (pooled 1.10%) by at least 0.50 percentage points** (i.e., ≥1.60% combined), **in at least 3 of 4 seeds independently**.

| Seed | Fresh-training baseline | Mix-script target | Margin |
|------|------------------------:|------------------:|-------:|
| 42   | 0.59%                  | ≥ 1.09%          | +0.50pp |
| 123  | 3.63%                  | ≥ 4.13%          | +0.50pp |
| 7    | 0.00%                  | ≥ 0.50%          | +0.50pp |
| 999  | 0.20%                  | ≥ 0.70%          | +0.50pp |

**Pooled target:** ≥1.60% combined PASS+SHOT (vs. fresh-training pooled 1.10%).

Per-seed judgment is primary. Pooled result is supplementary context only.

### 7.2 Secondary Criterion

At least one non-forced `pass_completed` and one non-forced `shot` event under pure π, in **at least 2 of 4 seeds**.

This is the bar the fresh-training run failed at 0/4. Meeting it at all is meaningful progress even before hitting the primary threshold.

### 7.3 Judgment Rules

- **Primary criterion PASS:** ≥3 seeds individually exceed their per-seed target.
- **Secondary criterion PASS:** ≥2 seeds individually produce ≥1 `pass_completed` AND ≥1 `shot` event.
- If the primary criterion passes but the secondary fails: the behavior change is in action selection (policy prefers PASS/SHOT) but execution success remains broken. This points to a **masking/engine interaction** issue, not a pure learning failure.
- If both criteria fail, and the mix-script-end checkpoint also shows no difference from the fresh-training baseline: this is a strong signal that paralysis is **structural** (advantage estimation, reward shaping, or exploration-bonus related) rather than a pure "never saw a good example" problem. Investigation should redirect toward reward/GAE-level causes rather than further rollout-distribution interventions.
- If the mix-script-end checkpoint shows elevated PASS/SHOT selection but the final checkpoint reverts toward baseline: the learner adopts the behavior only while scripting biases the data, then reverts. This suggests the unscripted tail is insufficient to consolidate the behavior, or the reward signal is too weak to sustain it.
- If the mix-script-end checkpoint shows elevated PASS/SHOT selection and the final checkpoint sustains or builds on it: the intervention is successful and should be scaled (longer unscripted tail, tuned override rate/duration).

---

## 8. Evaluation Protocol

Identical to fresh-training and Phase B:
- **Script:** `training/eval_f_act.py`, arm `ONBALL-pi`
- **Episodes:** 20 per seed
- **Deterministic:** True
- **No forced actions:** Pure policy only
- **Metrics:** t=0 validity, per-action selection rates/counts, event counts (`pass` initiation, `pass_completed`, `shot`, `goal`)

Evaluations are run on:
1. `_mixscript_15000.pt` checkpoint (mix-script-end)
2. `_mixscript_50176.pt` checkpoint (final, post-unscripted-tail)

---

## 9. Explicit Non-Goals

- No changes to reward function, GAE γ/λ, or masks.
- No new PASS/SHOT mechanics.
- No permanent or default-on scripting.
- No warm-starting from fresh-training or original checkpoints.
- No modification of existing `training/results/` files from prior phases.
