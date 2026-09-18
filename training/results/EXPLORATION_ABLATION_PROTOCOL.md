# Exploration Ablation Protocol — Form A: Targeted On-Ball Entropy Bonus
Date: 2026-09-18  
Status: Pre-registered (locked before training begins)

## 1. Mechanism

**Form:** Targeted entropy bonus added at the policy-optimization level.

At each PPO update step, compute the standard policy entropy over the full 19-action distribution, then add an **extra entropy term** that is nonzero only when:
- The mask indicates PASS (indices 9, 10, 11) or SHOT (index 12) is legal for the controlled agent, **and**
- The agent is in an on-ball state (left-ownership slot active in obs[95]).

The extra term is:

```
bonus_entropy = -sum(π(a | legal football actions) * log π(a | legal football actions))
```

This is the Shannon entropy of the *marginal* distribution over legal PASS/SHOT actions only, excluding MOVE/IDLE/TACKLE/DRIBBLE.

The total entropy signal used in the PPO loss is:

```
H_total = H(full_19) + alpha * bonus_entropy
```

where `alpha` is the bonus coefficient (see Section 2).

**Important:** This is **not** a reward-shape change. The base reward function, GAE γ/λ, and masks are untouched. The bonus is an additional term in the policy optimization objective that encourages the actor to spread probability mass over legal football actions during Phase 1 only.

**Implementation location:** `training/mappo_update.py` — `ppo_update()` gains a new optional argument `onball_football_entropy_bonus` (default 0.0). When > 0, the function computes the extra entropy term from the same masked distribution already used for the policy loss, scoped to PASS/SHOT-legal rows. The bonus is added to the standard entropy before applying `entropy_coef`.

**Mask contract:** The bonus uses the *same* `action_masks` already passed to `actor()` during the update. If a row has PASS or SHOT legal, the bonus applies to that row's marginal over `{9,10,11,12}`. If neither is legal, the bonus term is zero for that row. No mask logic is changed.

## 2. Magnitude

**Bonus coefficient:** `alpha = 0.50`

Reasoning:
- The standard `entropy_coef` is 0.01 and applies to the full 19-action entropy (~2.75 nats). The football-action marginal entropy is typically smaller (subset of actions), so a coefficient of 0.50 gives the football-action entropy roughly comparable weight to the global entropy term without overwhelming the surrogate objective.
- This is a conservative first test. If no effect is seen, the next step would be to increase alpha or switch to a count-based exploration bonus; if effect is seen but unstable, decrease alpha.
- The bonus is **constant** during Phase 1 (no ramp/decay). The collapse shape from forensics is not established enough to justify a curve.

## 3. Schedule

| Phase | Steps | Bonus | Checkpoints |
|-------|-------|-------|-------------|
| 1 | 0 – 15,000 | `alpha = 0.50` (constant) | 5k, 10k, 15k |
| 2 | 15,000 – 30,000 | `alpha = 0.0` (fully off) | 20k, 25k, 30k |

**Total training:** 30,000 steps per seed.

**Rationale for Phase 2 length:** 15,000 additional steps is roughly 59 update cycles (256 steps/cycle). This is long enough for the policy's own optimization dynamics to either consolidate or erase the Phase 1 preference shift, mirroring the mix-script lesson that transient exposure does not guarantee lasting change.

**No warm-start:** All runs start from fresh policy initialization (`seed` set via `torch.manual_seed` and `np.random.seed` before network construction).

## 4. Seeds

Same four seeds as all prior phases: **42, 123, 7, 999**.

## 5. Checkpoint Naming

Checkpoints are saved at exact step boundaries with the suffix `_expl_ablation`:

```
mappo_academy_3_vs_1_with_keeper_seed{N}_expl_ablation_{K}k.pt
```

where `K` ∈ {5, 10, 15, 20, 25, 30} (in thousands of steps).

The **Phase 2 tail-end checkpoint** is `mappo_academy_3_vs_1_with_keeper_seed{N}_expl_ablation_30k.pt` for each seed. This is the primary evaluation artifact.

## 6. Evaluation Protocol

- **Mode:** pure-π, deterministic, no forcing, no mix-script.
- **Episodes:** 20 episodes per checkpoint per seed.
- **Scenario:** `academy_3_vs_1_with_keeper_onball`.
- **Metrics recorded:** per-action selection counts/rates, event counts (`pass_completed`, `shot`, `goal`), episode reward, episode length.
- **Checkpoints evaluated:** all intermediate (5k, 10k, 15k, 20k, 25k, 30k) plus the fresh-training baseline from prior phase (`mappo_academy_3_vs_1_with_keeper_seed{N}_freshtrain_50176.pt`).

## 7. Success Criteria (Pre-Registered)

### Primary Criterion
On the **Phase 2 tail-end checkpoint** (30k, bonus off), pure-π evaluation shows PASS+SHOT combined selection rate exceeding the fresh-training baseline by **≥ 1.5 percentage points** in **at least 3 of 4 seeds independently**.

Fresh-training baseline (pooled): 1.10% PASS+SHOT.  
Target per seed: ≥ 2.60% PASS+SHOT (1.10% + 1.50%).

### Secondary Criterion
On the Phase 2 tail-end checkpoint, **at least one** non-forced `pass_completed` event and **at least one** non-forced `shot` event under pure π, in **at least 2 of 4 seeds**.

### Guard Criterion (Reversion Check)
Report whether any seed's Phase 2 tail-end PASS+SHOT selection rate falls **below** its own fresh-training baseline. This is not a pass/fail gate but must be reported explicitly per seed.

### Diagnostic-Only Reporting (Not a Gate)
Phase 1 intervention-window π(PASS)/π(SHOT) rates and event counts are reported for context on whether the bonus mechanism is active, but are **explicitly labeled as non-evidentiary** for the primary/secondary criteria. Success is judged solely on the Phase 2 unscripted tail.

## 8. Non-Goals

- No reward function changes.
- No GAE γ/λ changes.
- No mask changes.
- No global/uniform entropy coefficient changes.
- No warm-starting from existing checkpoints.
- No judging success on Phase 1 numbers alone.
- No modification of prior `training/results/` files from earlier phases.

## 9. Artefact Policy

- Committed to git: `training/results/EXPLORATION_ABLATION_PROTOCOL.md` (this file), `training/results/EXPLORATION_ABLATION_FINDINGS.md`, `training/results/exploration_ablation_summary.csv`.
- Large per-frame detail files (>10 MB) are kept as local/CI artefacts only, not committed to normal git history.
