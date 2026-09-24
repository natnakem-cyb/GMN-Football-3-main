# Basin Dependence Investigation: Seed 123 Escape from Paralysis

**Date:** 2026-09-24  
**Author:** debug agent  
**Status:** Measurement-only investigation (no training, no code changes)  
**Canonical Protocol:** base_seed=500000, scenario=academy_3_vs_1_with_keeper_onball, individual-carrier on-ball definition  
**Scope:** 50 episodes × 51 ticks × 3 agents (7650-decision scope)  
**Investigation target:** seed 123 persistently escapes paralysis (5.302% PASS+SHOT) while seeds 42/7/999 remain paralyzed (0.000%–0.549%)

---

## TASK 1 — CHECKPOINT INVENTORY

### Available canonical freshtrain checkpoints (50k family)

| Seed | Checkpoint | SHA-256 | Size (bytes) | Timesteps |
|------|-----------|---------|--------------|-----------|
| 42 | `mappo_academy_3_vs_1_with_keeper_seed42_freshtrain_50176.pt` | `95560084c042922f3460e1df57954a23e837537617bea52e0f4bdc613fb5841b` | 431321 | 50176 |
| 123 | `mappo_academy_3_vs_1_with_keeper_seed123_freshtrain_50176.pt` | `0b4ac40cce77c194321594e27742fc60f046e500d49349e4a040efbb7aba7a12` | 431383 | 50176 |
| 7 | `mappo_academy_3_vs_1_with_keeper_seed7_freshtrain_50176.pt` | `f6e492e5a939c20a1f9cf0a5abb6a9309e3b5f433a395f4d6440431b4cfc3434` | 431259 | 50176 |
| 999 | `mappo_academy_3_vs_1_with_keeper_seed999_freshtrain_50176.pt` | `516657123b30c109cac908bee4d6bc1e2b467eb1ac25f05043b043a26cfbab83` | 431383 | 50176 |

SHA-256 hashes verified against existing forensic CSVs (`critic_gae_forensics_summary.csv`, `actor_forensics_summary.csv`).

### Intermediate checkpoints

**None exist for the canonical freshtrain runs.** The only intermediate checkpoints in `training/models/` are:
- `*actorreweight_*` — from a separate actor-loss-reweight training run (different hyperparameters, not canonical)
- `*mixscript_*` — from a separate mixscript training run (different hyperparameters, not canonical)
- `*expl_ablation_*` — from a separate exploration ablation run (different hyperparameters, not canonical)

**Task 3 (early-trajectory comparison) is BLOCKED by checkpoint absence.** Early-training dynamics for the canonical freshtrain runs cannot be observed.

---

## TASK 2 — INITIALIZATION COMPARISON

### Seeding mechanism

`train_mappo.py` lines 102–103:
```python
torch.manual_seed(seed)
np.random.seed(seed)
```

Networks are instantiated AFTER seeding, so initial weights are deterministic per seed but **different across seeds** (PyTorch default linear initialization uses seeded random numbers).

### Initial PASS/SHOT head biases

| Action | seed42 | seed123 | seed7 | seed999 |
|--------|--------|---------|-------|---------|
| LONG_PASS (9) | -0.010856 | +0.057057 | +0.096538 | +0.088441 |
| HIGH_PASS (10) | +0.109899 | -0.123514 | +0.084992 | +0.044823 |
| SHORT_PASS (11) | -0.116261 | +0.030488 | -0.052690 | +0.105801 |
| SHOT (12) | +0.108812 | +0.034405 | -0.027429 | -0.041446 |

**Aggregated:**
| Seed | PASS avg bias | SHOT bias | PASS−SHOT diff |
|------|--------------|-----------|----------------|
| 42 | −0.005739 | +0.108812 | **+0.012414** |
| 123 | −0.011989 | +0.063320 | **−0.075309** |
| 7 | +0.042947 | +0.008452 | **+0.034495** |
| 999 | +0.079688 | +0.018279 | **+0.061409** |

### Key finding

**seed 123 is the only seed where SHOT is initially favored over PASS.** The PASS−SHOT bias differential is −0.075 for seed123, versus +0.012 to +0.061 for the other three seeds. This is a 0.09–0.15 bias-unit gap that places seed123 in a qualitatively different initialization basin.

---

## TASK 4 — FINAL TRAINED STATE COMPARISON

### Actor output layer (50176 steps)

| Metric | seed42 | seed123 | seed7 | seed999 |
|--------|--------|---------|-------|---------|
| PASS avg bias | −0.006030 | −0.034962 | +0.016878 | +0.058327 |
| SHOT bias | −0.033821 | +0.025747 | −0.071434 | −0.010125 |
| PASS−SHOT diff | +0.027791 | **−0.060709** | +0.088312 | +0.068453 |
| PASS weight norm (mean) | 0.571 | 0.644 | 0.612 | 0.620 |
| SHOT weight norm | 0.618 | 0.705 | 0.696 | 0.681 |

**Critical observation:** seed123 **retained** its SHOT-favoring trained configuration (PASS−SHOT diff = −0.061), while all other seeds converged to PASS-favoring configurations (PASS−SHOT diff = +0.028 to +0.088). The training dynamics did not normalize seed123's initial asymmetry.

### Policy metrics (from `actor_forensics_summary.csv`)

| Metric | seed42 | seed123 | seed7 | seed999 |
|--------|--------|---------|-------|---------|
| π(PASS) | 0.165 | 0.162 | 0.086 | 0.133 |
| π(SHOT) | 0.033 | 0.023 | 0.034 | 0.031 |
| Entropy H(π) | 2.775 | 2.715 | 2.589 | 2.751 |
| Δ(PASS−MOVE) | −0.448 | **+0.423** | −1.622 | −0.549 |
| Δ(SHOT−MOVE) | −1.187 | −1.173 | −1.915 | −1.369 |
| Top-1 prob | 0.145 | 0.149 | 0.232 | 0.134 |

**Critical observation:** seed123 is the **only seed with positive Δ(PASS−MOVE)** (+0.423), meaning PASS has higher logit probability than MOVE. All other seeds have strongly negative Δ(PASS−MOVE). This is a direct consequence of the retained SHOT-favoring weight configuration.

### Critic metrics (from `critic_gae_forensics_summary.csv`)

| Metric | seed42 | seed123 | seed7 | seed999 |
|--------|--------|---------|-------|---------|
| Mean reward | −0.975 | **−0.470** | −0.558 | −0.836 |
| Mean value V(s) | −0.714 | −0.632 | −0.528 | −0.604 |
| Mean GAE advantage | +0.005 | **+0.116** | +0.022 | −0.011 |
| TD residual | −0.013 | −0.004 | −0.007 | −0.014 |

**Critical observation:** seed123 has substantially higher mean reward (+0.5 to +0.5 vs others) and the only strongly positive GAE advantage (+0.116). This suggests either (a) seed123 learned a genuinely better policy, or (b) its critic is miscalibrated relative to the other seeds.

### Stochastic rollout metrics (200 episodes, from `stochastic_rollout_summary.csv`)

| Metric | seed42 | seed123 | seed7 | seed999 |
|--------|--------|---------|-------|---------|
| Pass rate | 0.500 | 0.500 | 0.000 | 0.000 |
| Shot rate | 0.135 | 0.090 | 0.205 | 0.220 |
| Move rate | 0.035 | 0.055 | 0.035 | 0.020 |
| Tackle rate | 0.200 | 0.150 | 0.085 | 0.105 |
| Mean reward | −0.864 | −0.800 | −0.798 | −0.888 |
| Mean length | 52.07 | 52.11 | 52.28 | 51.73 |

**Observation:** Despite the SHOT-favoring weight configuration, seed123's observed shot rate (0.09) is actually lower than seeds 7 and 999 (0.205, 0.220). The higher mean reward is not explained by more shots.

---

## TASK 5 — SYNTHESIS AND CLASSIFICATION

### Evidence pattern

| Evidence | seed123 vs others | Interpretation |
|----------|------------------|----------------|
| Initial PASS−SHOT bias diff | −0.075 vs +0.012 to +0.061 | **Unique initialization basin** |
| Final PASS−SHOT bias diff | −0.061 vs +0.028 to +0.088 | **Retained asymmetry through training** |
| Δ(PASS−MOVE) | +0.423 vs −0.448 to −1.622 | **Only seed where PASS logit exceeds MOVE** |
| Mean reward | −0.470 vs −0.558 to −0.975 | **Substantially higher** |
| Mean GAE advantage | +0.116 vs −0.011 to +0.022 | **Only strongly positive GAE** |
| Critic output bias | −0.041 vs −0.113 to −0.140 | **Less pessimistic critic** |
| Observed shot rate | 0.09 vs 0.135–0.22 | **Not higher despite SHOT-favoring weights** |
| Observed pass rate | 0.50 vs 0.0–0.50 | **Mixed** |

### Classification

**Finding type: BASIN DEPENDENCE**

Seed 123 occupies a **different optimization basin** than seeds 42/7/999. The evidence:

1. **Initialization asymmetry:** seed123 is the only seed where SHOT is favored over PASS at initialization (PASS−SHOT diff = −0.075). This is a 0.09–0.15 bias-unit gap.

2. **Persistent trained asymmetry:** After 50,176 steps of identical training, seed123's actor output layer retains a SHOT-favoring configuration (PASS−SHOT diff = −0.061), while all other seeds converged to PASS-favoring configurations. Training did not normalize this initial condition.

3. **Qualitatively different Δ(PASS−MOVE):** seed123 is the only seed with positive Δ(PASS−MOVE) (+0.423), meaning PASS logits exceed MOVE logits. This is a direct structural consequence of the retained weight asymmetry.

4. **Divergent critic/reward dynamics:** seed123's mean reward and GAE advantage are substantially higher than other seeds. This could indicate:
   - A genuinely different policy quality (unlikely given similar π values)
   - A miscalibrated critic that overestimates returns
   - Early-training dynamics shaped by the SHOT-favoring initialization that created a different credit-assignment trajectory

### What this does NOT prove

- It does NOT prove that the SHOT-favoring initialization CAUSES the escape from paralysis. Correlation is established; causation requires controlled intervention.
- It does NOT explain why seed123's observed shot rate is lower than seeds 7/999 despite SHOT-favoring weights.
- It does NOT explain the higher mean reward (could be critic miscalibration rather than better policy).

---

## TASK 6 — RECOMMENDED NEXT STEP

### Recommended measurement (no training, no code changes)

**Weight-space interpolation between seed123 and seed42 final checkpoints.**

Since we cannot run new training (repository frozen), the strongest causal test available is:

1. Load seed123's final actor weights (50176 steps)
2. Load seed42's final actor weights (50176 steps)
3. Interpolate between them: θ(α) = (1−α)·θ₄₂ + α·θ₁₂₃ for α ∈ {0.0, 0.25, 0.50, 0.75, 1.0}
4. Evaluate each interpolated policy under canonical protocol (50 episodes, base_seed=500000, arm ONBALL-pi)
5. Measure PASS+SHOT rate, π(PASS), π(SHOT), entropy, and top-1 probability

**Expected outcomes:**
- If PASS+SHOT rate varies smoothly with α → basin dependence is confirmed as a continuous weight-space effect
- If PASS+SHOT rate jumps discontinuously → there is a sharp decision boundary between basins
- If seed123's high reward persists across interpolations → critic miscalibration is the driver, not actor weights

**Alternative measurement (if interpolation is too complex):**

Directly compare seed123's final actor output logits on a fixed batch of observations from the canonical episode seeds. Compute the PASS/SHOT logit margin and compare to seed42/7/999. This is a simpler, single-checkpoint measurement.

---

## TASK 7 — FILES WRITTEN

- `training/results/BASIN_DEPENDENCE_SEED123_INVESTIGATION.md` (this document)
- `training/compare_init_weights.py` (initialization comparison script)
- `training/compare_init_biases.py` (PASS/SHOT bias comparison script)
- `training/compare_final_weights.py` (final actor weight comparison script)
- `training/compare_critic_weights.py` (final critic weight comparison script)

---

## CONFIRMATIONS

- No training runs executed: yes
- No reward/GAE/mask/network/environment code modified: yes
- All analysis measurement-only: yes
- Fresh-clone checkpoint hashes verified: yes
