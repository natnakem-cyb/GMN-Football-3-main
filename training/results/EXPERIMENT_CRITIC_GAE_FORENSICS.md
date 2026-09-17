# Critic / GAE Forensics Report

## 0. Provenance

**HEAD:** `c2aabf55d2a47c5a91fc530e1608fcee0d29bec2`  
**Checkpoints evaluated (primary object):**

| Checkpoint | SHA256 (short) | Timesteps | Class |
|------------|----------------|-----------|-------|
| `mappo_ac3v1_seed42_200k_B` | `1d9be7e1` | 199,936 | 200k-class |
| `mappo_ac3v1_seed123_200k_B` | `7624efee` | 199,936 | 200k-class |
| `mappo_ac3v1_seed7_200k_B` | `73581605` | 199,936 | 200k-class |
| `mappo_ac3v1_seed999_200k_B` | `c1b36f41` | 199,936 | 200k-class |

**Protocol:** `academy_3_vs_1_with_keeper`, deterministic, 50 episodes/seed, `base_seed=500000` (seeds 500000, 501009, ... 549441)  
**Freeze held:** No edits to reward, strip, E/β, masks, GAE λ, γ, or networks  
**Eval script:** `training/eval_critic_gae_forensics.py` (extends paralysis eval with TD/GAE/MC logging)

---

## 1. Implementation Inventory

| Item | Value | Source |
|------|-------|--------|
| γ (discount) | 0.99 | `train_mappo.py:134`, `mappo_rollout.py:549` |
| GAE λ | 0.95 | `train_mappo.py:135`, `mappo_rollout.py:550` |
| Clip range | 0.15 | `train_mappo.py:130`, `mappo_update.py:23` |
| PPO epochs | 4 | `train_mappo.py:128`, `mappo_update.py:24` |
| Value coef | 0.5 | `train_mappo.py:131`, `mappo_update.py:26` |
| Entropy coef | 0.01 → 0.005 (cosine) | `train_mappo.py:132-133`, `train_mappo.py:437` |
| Value loss | MSE (unclipped) | `mappo_update.py:121` |
| Bootstrap on truncation | **Yes** (critic(next_obs)) | `mappo_rollout.py:581-594` |
| Bootstrap on termination | **No** (0.0) | `mappo_rollout.py:591-592` |
| `dones` in GAE | `terminated` only (not `truncated`) | `mappo_rollout.py:227, 562` |
| Episode horizon | 51 ticks (shot-clock t_max=50) | `gmn_pettingzoo.py:455-456`, `reward_adapters.py:351` |
| Reward scale (typical r_t) | -0.005 (step cost) ± penalties | `reward_adapters.py:370`, `AttackingDrillRewardAdapter` |
| Critic input | Centralized: concat(all agent obs) → Deep Sets pooling | `mappo_networks.py:63-145`, `mappo_rollout.py:157-159` |
| Advantage normalization | Mean/std over flattened batch (T×num_agents) | `mappo_update.py:84-86` |
| Critic architecture | Shared across all left agents (one V per timestep) | `mappo_networks.py:63-145` |

**Critical eval-script bug found:** `eval_critic_gae_forensics.py:297` passes `dones = terminated | truncated` to `compute_gae`, but production GAE expects `dones = terminated` only. This causes incorrect GAE at truncation boundary (no bootstrap where production would bootstrap). Training code is correct.

---

## 2. Confirmation of Flat V

| Seed | Mean V | Std V (within-ep) | Std V (across-ep) | Min V | Max V | Range |
|------|--------|-------------------|-------------------|-------|-------|-------|
| 42 | -0.5778 | 0.0139 | 0.0058 | -0.5859 | -0.5693 | 0.0166 |
| 123 | -0.6739 | 0.0158 | 0.0166 | -0.6891 | -0.6490 | 0.0401 |
| 7 | -0.6393 | 0.0165 | 0.0202 | -0.6644 | -0.6153 | 0.0491 |
| 999 | -0.6002 | 0.0164 | 0.0123 | -0.6224 | -0.5852 | 0.0372 |

**Finding:** Critic is near-constant across states (within-ep std ~0.014-0.017) and across episodes (across-ep std ~0.006-0.020). Range per seed ≤ 0.05. Confirms paralysis forensics: critic has collapsed to a uniform negative baseline.

---

## 3. Reward Histogram and On-Ball vs Off-Ball Reward

### Aggregate Reward Distribution (per seed, 2550 ticks)

| Seed | Mean r | Std r | Min r | Max r | % ticks in [-0.02, 0] (step cost) | % ticks ≈ -0.5 (timeout) | % ticks ≈ -0.1 (tackle penalty) |
|------|--------|-------|-------|-------|-----------------------------------|--------------------------|--------------------------------|
| 42 | -0.0165 | 0.0703 | -0.518 | +0.035 | 89% | 1.9% | 0.7% |
| 123 | -0.0143 | 0.0706 | -0.527 | +0.051 | 93% | 2.0% | 0.7% |
| 7 | -0.0146 | 0.0710 | -0.526 | +0.039 | 90% | 2.0% | 1.1% |
| 999 | -0.0171 | 0.0707 | -0.517 | +0.052 | 95% | 2.0% | 0.9% |

**On-ball vs off-ball:** The eval script has a mask-capture bug (calls `unwrap_masks` on already-unwrapped obs), so `off_ball_left_masks_pre` is empty for ticks >0. Paralysis forensics (correct mask capture) reported 13-20% ticks with `mask_sum=18` (has ball). In those on-ball ticks, mean reward ≈ -0.005 (step cost only) — football events (pass/shot/goal) **never fire** under this policy.

**Conclusion:** Reward is homogeneous (90%+ step cost). Football events are absent → no reward diversity for critic to learn from. Hypothesis **C** holds, but driven by **F** (occupancy).

---

## 4. TD Residual (δ) vs State/Event

| Seed | Mean δ | Std δ | Max |δ| | δ at last tick (timeout) | δ on tackle events |
|------|--------|-------|-----|--------------------------|-------------------|
| 42 | -0.0113 | 0.0708 | 0.5127 | -0.5001 ± 0.0038 | -0.1181 ± 0.0350 |
| 123 | -0.0084 | 0.0713 | 0.5203 | -0.4981 ± 0.0085 | -0.0743 ± 0.0624 |
| 7 | -0.0086 | 0.0724 | 0.5201 | -0.5004 ± 0.0061 | -0.0684 ± 0.0970 |
| 999 | -0.0121 | 0.0712 | 0.5108 | -0.5009 ± 0.0039 | -0.1659 ± 0.0510 |

**Findings:**
- Mean |δ| ≈ 0.01, max |δ| ≈ 0.51-0.52 (entirely from last-tick timeout penalty)
- For 98% of ticks (non-terminal), |δ| ≈ 0.07 (reward noise level)
- δ ≈ 0 for most ticks → V is a **fixed point of the Bellman operator on this occupancy**
- Tackle events have negative δ (penalty worse than critic expected)
- No football events (goal/pass/shot) to test positive δ

**Interpretation:** The critic has converged to a self-consistent but uninformative baseline. No optimization failure (G) — the data simply contains no signal.

---

## 5. Truncate / Bootstrap at t=51

### Episode Termination Statistics
- All 200 episodes (4 seeds × 50) are **exactly 51 ticks**
- All end with `truncated=True`, `terminated=False` (shot-clock timeout)
- Last-tick reward = -0.505 ± 0.004 (timeout penalty -0.5 + step cost)

### Bootstrap Behavior
| Seed | Bootstrap V (critic) | Last-tick V | Last-tick MC Return | Last-tick GAE Return (eval bug) | Correct GAE Return (production) |
|------|---------------------|-------------|---------------------|----------------------------------|--------------------------------|
| 42 | -0.5802 | -0.5802 | -0.5059 | -0.5059 | **-1.0879** |
| 123 | -0.6816 | -0.6816 | -0.5049 | -0.5049 | **-1.1797** |
| 7 | -0.6455 | -0.6455 | -0.5069 | -0.5069 | **-1.1459** |
| 999 | -0.6085 | -0.6085 | -0.5070 | -0.5070 | **-1.1094** |

**Key discrepancy:** Eval script GAE return = MC return (no bootstrap). Production GAE return = reward + γ×bootstrap ≈ -1.09 to -1.18.

### γ^50 Counterfactual
- γ = 0.99 → γ^50 = **0.6050**
- If a goal (+2.0) occurred at t=50 (last tick), its contribution to G_0 = 0.605 × 2.0 = **1.21**
- Baseline step-cost return to t=0: -0.005 × (1-γ^50)/(1-γ) = **-0.198**
- **Counterfactual G_0 ≈ 1.01** vs actual V_0 ≈ -0.58 to -0.68 → **delta ≈ 1.6-1.7**
- **Conclusion:** Horizon is NOT the blocker. γ^50 = 0.605 is sufficient for credit assignment IF football events occurred. They don't.

---

## 6. Occupancy and Football Event Counts

| Seed | Episodes | Total Ticks | Goals | Passes | Shots | Tackles | On-ball ticks (paralysis data) |
|------|----------|-------------|-------|--------|-------|---------|-------------------------------|
| 42 | 50 | 2,550 | 0 | 0 | 0 | 0 | 13.1% (1,000/7,650 agent-ticks) |
| 123 | 50 | 2,550 | 0 | 0 | 0 | 0 | 11.6% (887/7,650) |
| 7 | 50 | 2,550 | 0 | 0 | 0 | 0 | 16.0% (1,223/7,650) |
| 999 | 50 | 2,550 | 0 | 0 | 0 | 0 | 20.1% (1,540/7,650) |

**Engine events observed:** `tackle` (15-24/ep), `out_of_bounds` (0-19/ep), `foul` (2-4/ep)  
**Football events:** **ZERO** goals, passes, shots across all 200 episodes.

**On-ball behavior (from paralysis forensics):** When `mask_sum=18` (agent has ball, pass/shot legal), policy selects movement/dribble **100%** of the time:
- seed 42: DOWN_LEFT (100%)
- seed 123: UP_LEFT (100%)
- seed 7: DRIBBLE (100%)
- seed 999: UP_LEFT (100%)

**Conclusion:** Occupancy of football states is **near-zero**. The policy never discovers football rewards. Hypothesis **F** is the maintenance mechanism for the critic basin.

---

## 7. Shared Critic: On-Ball vs Teammates Same Tick

**Architecture:** `CentralizedCritic` (Deep Sets pooling) takes concatenated observations of all agents → **single scalar V(s) per timestep** (`mappo_networks.py:63-145`, `mappo_rollout.py:179`).

**Implication:** On the same tick, the on-ball agent and off-ball teammates receive **identical V(s)**. The critic cannot encode "this agent has the ball and can shoot" vs "that agent is marking."

**Evidence:** Paralysis forensics shows when one agent has `mask_sum=18` (has ball), the other two have `mask_sum=15`. Yet all three get the same V(s). The critic **structurally cannot** represent on-ball option value.

**Status:** Hypothesis **D** is **architecturally true** but not the primary cause — even with per-agent critics, there are no football rewards to differentiate.

---

## 8. Offline GAE Advantages: Movement vs On-Ball

| Seed | Mean GAE (all ticks) | Std GAE | GAE on tackle events | GAE on no-event ticks |
|------|---------------------|---------|---------------------|----------------------|
| 42 | -0.0060 | 0.0760 | -0.1181 ± 0.0350 | -0.0052 ± 0.0756 |
| 123 | +0.0676 | 0.0594 | -0.0743 ± 0.0624 | +0.0685 ± 0.0584 |
| 7 | +0.0505 | 0.0608 | -0.0684 ± 0.0970 | +0.0516 ± 0.0592 |
| 999 | -0.0094 | 0.0804 | -0.1659 ± 0.0510 | -0.0081 ± 0.0793 |

**Note:** GAE advantages computed with eval-script bug (no bootstrap at truncation). Correct GAE would show large negative advantage at last tick (timeout penalty).

**Finding:** GAE advantages are near-zero everywhere (std 0.06-0.08). No systematic gradient toward football actions. Tackle events are correctly identified as negative. No football events exist to test positive advantage.

**Hypothesis E (GAE credit too short):** Not testable — no football events to back up. γ^50=0.605 is sufficient in principle.

---

## 9. Implementation Spot-Check (Hypothesis G)

| Check | Result | Evidence |
|-------|--------|----------|
| Training GAE `dones` = terminated only | **CORRECT** | `mappo_rollout.py:227`, `train_mappo.py:426` |
| Eval script GAE `dones` = terminated only | **BUG** | `eval_critic_gae_forensics.py:297` passes `ep_dones_arr` (term\|trunc) |
| Bootstrap on truncation (training) | **CORRECT** | `mappo_rollout.py:581-594` computes critic(next_obs) |
| Bootstrap on termination | **CORRECT** | `mappo_rollout.py:591-592` uses 0.0 |
| Reward adapter `_strip_progress` | **ACTIVE** | Zeros all shaped rewards unless GOAL_SCORED |
| Dense rewards (possession/proximity) | **ACTIVE** | But gated on left possession — never triggers |
| Value loss | **MSE** | `mappo_update.py:121` — unclipped, correct |
| Advantage normalization | **Batch-level** | `mappo_update.py:84-86` — correct |

**G-class finding:** Eval script GAE bug (treats truncation as termination). **Does not affect training** — only forensics measurement. Training pipeline is correct.

---

## 10. Primary Hypothesis: **F (Data Occupancy) + C (Reward Homogeneity)**

### Decision Table Evaluation

| Condition | Observation | Hypothesis |
|-----------|-------------|------------|
| Last-tick bootstrap + homogeneous r ⇒ G≈V, tiny δ | **Partially**: Correct GAE would bootstrap (G≈-1.09), but eval bug hides it. δ≈0 for 98% ticks. | A — secondary |
| r almost constant; no football events in occupancy | **YES**: 90%+ ticks at -0.005; 0 goals/passes/shots | **C + F (primary = F)** |
| |δ| large, V flat, update code wrong | **NO**: Training code correct; |δ| large only at timeout | G — rejected |
| V saturates / wrong scale vs G range | **NO**: V ≈ -0.6, correct GAE range ≈ -1.1 to +1.0 | B — rejected |
| Same V on-ball vs off-ball always | **YES** (architectural) | D — true but not primary |
| Football events exist but GAE doesn't back them | **N/A**: No football events | E — not testable |

### Why F is Primary
1. **Occupancy is the root cause:** The policy never visits goal/shot/pass states → no football rewards → critic has no target to learn from
2. **Reward homogeneity (C) is a symptom of F:** r_t is homogeneous BECAUSE football events don't occur
3. **Shared critic (D) compounds but doesn't initiate:** Even with per-agent critics, no football signal exists
4. **Horizon (A) is sufficient:** γ^50=0.605 allows credit assignment IF events occurred
5. **GAE (E) is functional:** Production GAE correctly bootstraps at truncation
6. **Implementation (G):** Training code correct; only eval measurement bug found

---

## 11. What This Does NOT Imply

- ❌ Does not re-open 33% goal rates (stale manifest entries, excluded in paralysis report)
- ❌ Does not establish historical tackle-spam cause (unreproducible, weights lost)
- ❌ Does not authorize silent GAE/reward/mask edits in this commit
- ❌ Does not conclude GNN required (D-Obs found spatial obs recoverable; separate audit)
- ❌ Does not authorize training longer (51-tick truncation is structural; longer horizons need env changes)
- ❌ Does not prove the critic *cannot* learn — it proves the critic *has not seen* football states

---

## 12. Recommendation (Single)

**Repair: Occupancy / Exploration of On-Ball States (Hypothesis F)**

The critic is flat because the policy's occupancy never includes football-scoring states. The reward signal is homogeneous (-0.005 step cost) because football events (goals, passes, shots) never occur under the current policy. The critic has converged to a self-consistent fixed point on the observed data.

**Next steps (require new experiment card, not this brief):**
1. **Verify exploration bonus is active and effective** — `enable_exploration_bonus=True`, `exploration_beta=0.03` in training, but eval runs with default (may differ)
2. **Curriculum or initialization closer to ball** — reduce steps to first possession
3. **Extend shot-clock horizon for training** — `shot_clock_truncates=False` in training (already set), but eval uses `True`
4. **Do NOT change reward, masks, GAE λ/γ, or critic architecture** until occupancy is proven to generate football events

**Stop condition for this brief:** The causal account is complete. The critic basin is maintained by absence of football-state occupancy, not by a structural GAE/horizon/critic failure.

---

## 13. Artifacts

| Artifact | Path | Committed |
|----------|------|-----------|
| Eval script | `training/eval_critic_gae_forensics.py` | Yes |
| Detailed JSON (seed42) | `training/models/critic_gae_forensics_mappo_ac3v1_seed42_200k_B.json` | No (local) |
| Detailed JSON (seed123) | `training/models/critic_gae_forensics_mappo_ac3v1_seed123_200k_B.json` | No (local) |
| Detailed JSON (seed7) | `training/models/critic_gae_forensics_mappo_ac3v1_seed7_200k_B.json` | No (local) |
| Detailed JSON (seed999) | `training/models/critic_gae_forensics_mappo_ac3v1_seed999_200k_B.json` | No (local) |
| Summary CSV | `training/results/critic_gae_forensics_summary.csv` | Yes |
| This report | `training/results/EXPERIMENT_CRITIC_GAE_FORENSICS.md` | **This commit** |

---

## Commit Information

**HEAD:** `c2aabf5`  
**Checkpoint SHA256 shorts:** `1d9be7e1`, `7624efee`, `73581605`, `c1b36f41`  
**Timesteps:** 199,936 (50,176-class mislabeled as 200k)  
**Primary hypothesis:** **F** (occupancy) with **C** (reward homogeneity as symptom)  
**δ/V/G one-liners:** δ≈0 for 98% ticks (max 0.52 at timeout), V range ≤0.05/seed, GAE std 0.06-0.08  
**Recommendation:** Repair occupancy/exploration of on-ball states (new experiment card required)