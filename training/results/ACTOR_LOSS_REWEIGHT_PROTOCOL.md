# ACTOR_LOSS_REWEIGHT_PROTOCOL — Fixed Multiplier on Legal PASS/SHOT

**Date:** 2026-09-20  
**HEAD:** `d2452e0`  
**Phase:** Actor-Loss Frequency-Balancing Ablation (Measurement + Training)

---

## 1. Mechanism

### 1.1 Exact Formula

During the actor-loss computation in `ppo_update`, each transition's contribution to the clipped surrogate objective is:

```
L_t^actor = -min(ratio_t × A_t, clip(ratio_t) × A_t)
```

For transitions where the selected action was a **legal** PASS or SHOT at that state, the actor-loss term is reweighted:

```
L_t^actor = -M × min(ratio_t × A_t, clip(ratio_t) × A_t)   if action ∈ {9,10,11,12} AND mask[action] = True
L_t^actor = -min(ratio_t × A_t, clip(ratio_t) × A_t)        otherwise
```

Where:
- `M` = fixed multiplier (see Section 2)
- `action` = the action actually taken by the agent at that transition
- `mask[action]` = True if the action was legal in the action mask at that state
- `ratio_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` = probability ratio
- `A_t` = GAE advantage (unchanged)

### 1.2 Where Applied

The reweighting is applied **inside `mappo_update.py` `ppo_update()`**, at the point where the per-transition surrogate objective is computed:

```python
# Pseudocode insertion in ppo_update()
for each transition in batch:
    if action_is_legal_pass_or_shot(actions_t[batch_idx], batch_masks[batch_idx]):
        surr1 = ratio * advantages_t[batch_idx] * M
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages_t[batch_idx] * M
    else:
        surr1 = ratio * advantages_t[batch_idx]
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages_t[batch_idx]
    policy_loss = -torch.min(surr1, surr2).mean()
```

**Important:** The multiplier `M` is applied to the surrogate objective terms **before** the `mean()` aggregation. This means:
- The policy gradient for PASS/SHOT transitions is multiplied by `M`
- The entropy bonus and value loss are **not** reweighted
- The critic's loss is computed from **unweighted** transitions

### 1.3 Actor-Loss Only — Critic Isolation

The critic's loss computation is **completely isolated** from the reweighting:

```python
# Critic loss — unweighted, unchanged
values_pred = critic(joint_obs_t[batch_idx])
value_loss = ((values_pred - returns_t[batch_idx]) ** 2).mean()
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

The critic sees the same `joint_obs_t`, `returns_t`, and `values_pred` as an unmodified run. The reweighting only affects `policy_loss`, which only affects the actor's parameter update via `actor_opt.step()`.

**Confirmation:** After each update step, we verify that `critic_opt.step()` is called with gradients derived solely from `value_loss`, which uses unweighted data.

---

## 2. Multiplier Value M

### 2.1 Selection

**M = 500**

### 2.2 Reasoning

From the actor-optimization audit (`d2452e0`), the 50k checkpoint's per-action-family gradient norms and frequencies are:

| Action Family | Frequency (%) | Per-sample grad norm | Aggregate contribution |
|---------------|---------------|----------------------|------------------------|
| MOVE | ~95% | ~0.025 | ~2.4×10⁻² |
| IDLE | ~5% | ~0.02 | ~1.0×10⁻³ |
| TACKLE (SLIDING) | ~0.8% | ~0.002 | ~1.6×10⁻⁵ |
| PASS | ~0.1% | ~0.001 | ~1.0×10⁻⁶ |
| SHOT | ~0.02% | ~0.001 | ~2.0×10⁻⁸ |

The inverse of combined PASS+SHOT frequency (~0.12%) is ~833. Starting conservatively below this:

- **M = 500** makes PASS/SHOT aggregate gradient contribution ~600× larger than raw
- Weighted PASS/SHOT aggregate ≈ 7.2×10⁻⁴, which is within an order of magnitude of IDLE (~1.0×10⁻³) and TACKLE (~1.6×10⁻⁵)
- This is "comparable in order of magnitude" to the non-MOVE actions without being wildly dominant

**Conservative rationale:** M = 500 is below the theoretical inverse-frequency value (~833), reducing the risk of destabilizing the policy while still making PASS/SHOT gradient signal detectable above the noise floor.

---

## 3. Schedule

### 3.1 Phase 1 — Intervention Window

| Parameter | Value |
|-----------|-------|
| Steps | 0 – 15,000 |
| Multiplier M | 500 (constant) |
| Description | Actor-loss reweighting active for all legal PASS/SHOT transitions |

### 3.2 Phase 2 — Unscripted Tail

| Parameter | Value |
|-----------|-------|
| Steps | 15,000 – 50,000 |
| Multiplier M | 1.0 (no reweighting) |
| Description | Standard PPO, no reweighting. This tests whether the intervention created a persistent behavioral change or immediately reverted. |

### 3.3 Total Training Length

50,000 steps per seed, matching the fresh-training baseline checkpoints.

---

## 4. Checkpointing

### 4.1 Intervals

Checkpoints saved every 5,000 steps through both phases:

| Step | Phase | Label |
|------|-------|-------|
| 0 | Initialization | `_actorreweight_step0.pt` |
| 5,000 | Phase 1 | `_actorreweight_step5000.pt` |
| 10,000 | Phase 1 | `_actorreweight_step10000.pt` |
| 15,000 | Phase 1/2 boundary | `_actorreweight_step15000.pt` |
| 20,000 | Phase 2 | `_actorreweight_step20000.pt` |
| 25,000 | Phase 2 | `_actorreweight_step25000.pt` |
| 30,000 | Phase 2 | `_actorreweight_step30000.pt` |
| 35,000 | Phase 2 | `_actorreweight_step35000.pt` |
| 40,000 | Phase 2 | `_actorreweight_step40000.pt` |
| 45,000 | Phase 2 | `_actorreweight_step45000.pt` |
| 50,000 | Phase 2 end | `_actorreweight_step50000.pt` |

### 4.2 Naming Convention

```
mappo_academy_3_vs_1_with_keeper_seed{N}_actorreweight_step{K}.pt
```

Example: `mappo_academy_3_vs_1_with_keeper_seed42_actorreweight_step15000.pt`

### 4.3 Seeds

All 4 seeds: 42, 123, 7, 999.

---

## 5. Frozen Configurations

The following are **unchanged** from the fresh-training baseline:

| Component | Status |
|-----------|--------|
| Reward function | Frozen — `AttackingDrillRewardAdapter` unchanged |
| GAE γ | Frozen — 0.99 |
| GAE λ | Frozen — 0.95 |
| Action masks | Frozen — production masks from bridge |
| Environment | Frozen — `academy_3_vs_1_with_keeper_onball` |
| µ-onball setup | Frozen — unchanged |
| PASS mechanics | Frozen — unchanged |
| Critic architecture | Frozen — `CentralizedCritic` unchanged |
| Critic training | Frozen — loss/update computed from unweighted transitions |
| Actor architecture | Frozen — `SharedActor` unchanged |
| Optimizer | Frozen — Adam, lr=3e-4 |
| Batch size | Frozen — 256 |
| PPO clip range | Frozen — 0.15 |
| Entropy coefficient | Frozen — 0.01 |
| Value loss coefficient | Frozen — 0.5 |
| Max grad norm | Frozen — 0.5 |
| Football entropy bonus | Frozen — 0.0 (Form A disabled) |
| Mix-script override | Frozen — disabled |

**Only change:** Fixed multiplier `M = 500` on actor-loss terms for legal PASS/SHOT transitions during Phase 1 (steps 0–15k).

---

## 6. Success Criteria

### 6.1 Primary Criterion

Pure-π evaluation (canonical protocol: deterministic, base_seed=500000, 50 episodes) on the **Phase 2 tail-end checkpoint** (step 50,000) shows PASS+SHOT combined selection rate exceeding the canonical fresh-training baseline by **≥1.5 percentage points** in **at least 3 of 4 seeds independently**.

| Seed | Canonical Baseline | Target (baseline + 1.5pp) |
|------|-------------------|---------------------------|
| 42 | 0% (0/50 episodes) | ≥1.5% |
| 123 | 1.96% (50/2,550) | ≥3.46% |
| 7 | 0% (0/50 episodes) | ≥1.5% |
| 999 | 0% (0/50 episodes) | ≥1.5% |

### 6.2 Secondary Criterion

At least one non-forced `pass_completed` event and at least one non-forced `shot` event under pure π (canonical protocol), in **at least 2 of 4 seeds**, on the Phase 2 tail-end checkpoint.

### 6.3 Guard Criterion (Reversion Check)

Report per-seed whether Phase 2 tail-end selection rate falls below the seed's own canonical fresh-training baseline. Not a pass/fail gate alone, but must be reported explicitly.

### 6.4 Required Secondary Measurement (Mechanistic Check)

Aggregate actor-loss/gradient contribution **share by action class** (not per-sample gradient norm) during Phase 1, compared to an unweighted control. This directly tests whether the multiplier actually shifted `G_PASS`/`G_SHOT` as a share of total actor-loss mass.

**Report this regardless of whether primary/secondary behavioral criteria pass.**

### 6.5 Diagnostic-Only Reporting

Phase 1 intervention-window π(PASS)/π(SHOT) rates, explicitly labeled non-evidentiary for the primary/secondary criteria.

---

## 7. Evaluation Protocol

### 7.1 Canonical Deterministic Evaluation

Per `CANONICAL_METRICS_CONTRACT.md`:

| Parameter | Value |
|-----------|-------|
| Script | `training/eval_critic_gae_forensics.py` |
| Scenario | `academy_3_vs_1_with_keeper_onball` |
| Episodes per seed | 50 |
| Base seed | 500,000 |
| Episode seeds | 500000, 501009, 502018, ..., 549441 |
| Policy mode | Deterministic (argmax) |
| Action masks | Production masks from bridge |
| GAE computation | `compute_gae` with `dones = terminated` only |
| Bootstrap on truncation | Critic(next_obs) |
| Bootstrap on termination | 0.0 |
| Gamma | 0.99 |
| Lambda | 0.95 |

### 7.2 Gradient-Share Measurement During Training

During Phase 1 training, log per-action-class aggregate gradient contribution:

```python
# After each ppo_update, before actor_opt.step():
total_grad_norm = 0.0
action_class_grads = {family: 0.0 for family in ACTION_FAMILIES}
for t in range(T_total):
    for a in range(num_agents):
        act_int = actions[t, a]
        family = classify_action(act_int)
        # Gradient contribution = |∇_θ log π(a_t|s_t)| × |A_t|
        # Approximated by: grad_norm of the taken action's logit × |advantage|
        logit_grad = final_grad[act_int].norm().item()
        adv = abs(advantages[t, a])
        action_class_grads[family] += logit_grad * adv
        total_grad_norm += logit_grad * adv

# Normalize to shares
for family in ACTION_FAMILIES:
    action_class_grads[family] /= total_grad_norm
```

Compare these shares against the unweighted fresh-training baseline (from `actor_optimization_audit_task3.json`).

---

## 8. Decision Rule

| Interpretation | Condition | Next Step |
|----------------|-----------|-----------|
| **Starvation causal** | Primary/secondary criteria pass AND gradient share for PASS/SHOT rose during Phase 1 | Recommend scaling (longer tail, tuning M, or promoting to main recipe) |
| **Starvation descriptive** | Gradient share rose during Phase 1 but π did not hold on Phase 2 tail | Redirect to deeper optimization dynamics (second-order effects, representation learning, basin dependence — note seed 123's partial success as relevant clue) |
| **Implementation failure** | Gradient share did NOT rise during Phase 1 despite M = 500 being applied | Treat as implementation bug, not hypothesis result |

---

## 9. Non-Goals

- No reward, GAE γ/λ, mask, network, environment, or critic-training changes.
- No entropy coefficient changes.
- No annealed/curriculum schedule or batch-oversampling approach.
- No warm-starting from existing checkpoints.
- No judging success on Phase 1 numbers alone.
- No new evaluation protocol — canonical metrics contract only.
- No resolving the TACKLE/SLIDING temporal-alignment question in this task.

---

## 10. Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-09-20 | Initial protocol for actor-loss reweighting ablation |
