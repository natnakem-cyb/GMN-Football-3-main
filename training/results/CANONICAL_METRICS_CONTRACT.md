# CANONICAL METRICS CONTRACT

**Date:** 2026-09-18  
**HEAD:** `a8a1f3189e357ba7eb95e657324321735196cd7f`  
**Phase:** Critic/GAE Horizon Forensics — Metric Lock

---

## 1. Canonical Metric Definition

### 1.1 PASS+SHOT Rate

**Formula:**

```
PASS+SHOT rate = (N_pass + N_shot) / N_decision_ticks
```

Where:
- `N_pass` = total count of PASS actions (action IDs 9, 10, 11) taken by any controlled agent across all evaluated episodes
- `N_shot` = total count of SHOT actions (action ID 12) taken by any controlled agent across all evaluated episodes
- `N_decision_ticks` = total number of agent-action decisions across all controlled agents and all evaluated episodes

**Agent scope:** AGGREGATED across all on-ball-eligible agents in the controlled team.

**Rationale:** In the 3-vs-1 scenario, any of the 3 left-team players may possess the ball and be the appropriate actor for a PASS or SHOT. Restricting the count to a single agent (e.g., agent 0) undercounts events and produces seed-sensitive denominators that vary with the policy's positional preferences. Aggregation across all agents matches the production training signal, which uses shared team reward.

### 1.2 Eligible Decision-Ticks

An "eligible decision-tick" is any tick at which a controlled agent selects an action from the action mask. This includes:
- Ticks where the agent has the ball (on-ball)
- Ticks where the agent does not have the ball (off-ball)

The denominator is **all decision-ticks**, not just on-ball ticks, because:
1. The policy selects actions on every tick, including off-ball movement.
2. Comparing football-action frequency against the full action distribution is the standard RL metric.
3. Restricting to on-ball ticks would require a deterministic ball-ownership oracle that may not align with the policy's belief state.

### 1.3 Football Action Definitions

| Action category | Action IDs | Engine event types |
|-----------------|------------|-------------------|
| PASS | 9 (LONG_PASS), 10 (HIGH_PASS), 11 (SHORT_PASS) | `pass`, `pass_completed` |
| SHOT | 12 (SHOT) | `shot`, `shot_saved`, `shot_missed` |
| MOVE | 0–8 | — |
| TACKLE | 16 | `tackle` |
| DRIBBLE | 17 | — |

### 1.4 Metric Reporting Convention

All rate metrics must be reported with:
- Numerator (count)
- Denominator (N_decision_ticks)
- Rate as a percentage (×100)
- Episode count and total tick count adjacent to the rate

Example:
```
PASS+SHOT: 50 / 2,550 = 1.96%  (50 episodes, 3 agents, 51 ticks/episode)
```

---

## 2. Canonical Evaluation Parameters

### 2.1 Deterministic Evaluation Protocol

| Parameter | Value |
|-----------|-------|
| Script | `training/eval_critic_gae_forensics.py` |
| Scenario | `academy_3_vs_1_with_keeper_onball` |
| Episodes per seed | 50 |
| Base seed | 500,000 |
| Episode seeds | 500000, 501009, 502018, ..., 549441 (base + episode×1009) |
| Policy mode | Deterministic (argmax) |
| Action masks | Production masks from bridge (not modified) |
| GAE computation | `compute_gae` from `training/mappo_rollout.py` with `dones = terminated` only |
| Bootstrap on truncation | Critic(next_obs) — production semantics |
| Bootstrap on termination | 0.0 |
| Gamma | 0.99 |
| Lambda | 0.95 |

### 2.2 Stochastic Rollout Protocol

| Parameter | Value |
|-----------|-------|
| Script | `training/stochastic_rollout_probe.py` (new) |
| Scenario | `academy_3_vs_1_with_keeper_onball` |
| Episodes per seed | 200 |
| Base seed | 600,000 |
| Episode seeds | 600000, 601009, 602018, ..., 619801 (base + episode×1009) |
| Policy mode | Stochastic (sample from categorical distribution) |
| Action masks | Production masks from bridge (not modified) |
| GAE computation | Same as deterministic, with `dones = terminated` only |
| Bootstrap on truncation | Critic(next_obs) |
| Bootstrap on termination | 0.0 |
| Gamma | 0.99 |
| Lambda | 0.95 |

### 2.3 Checkpoint Scope

All evaluations use the **fresh-training 50k checkpoints**:

| Seed | Checkpoint path | SHA-256 |
|------|-----------------|---------|
| 42 | `training/models/mappo_academy_3_vs_1_with_keeper_seed42_freshtrain_50176.pt` | `95560084c042922f3460e1df57954a23e837537617bea52e0f4bdc613fb5841b` |
| 123 | `training/models/mappo_academy_3_vs_1_with_keeper_seed123_freshtrain_50176.pt` | `0b4ac40cce77c194321594e27742fc60f046e500d49349e4a040efbb7aba7a12` |
| 7 | `training/models/mappo_academy_3_vs_1_with_keeper_seed7_freshtrain_50176.pt` | `f6e492e5a939c20a1f9cf0a5abb6a9309e3b5f433a395f4d6440431b4cfc3434` |
| 999 | `training/models/mappo_academy_3_vs_1_with_keeper_seed999_freshtrain_50176.pt` | `516657123b30c109cac908bee4d6bc1e2b467eb1ac25f05043b043a26cfbab83` |

---

## 3. Event Taxonomy

### 3.1 Event Classes

| Event class | Event types | Description |
|-------------|-------------|-------------|
| PASS_attempt | `pass` | PASS action taken, outcome pending |
| PASS_complete | `pass_completed` | PASS action completed successfully |
| PASS_fail | — | PASS action taken but not completed (inferred from `pass` without subsequent `pass_completed` within 5 ticks) |
| SHOT_attempt | `shot` | SHOT action taken |
| SHOT_on_target | `shot_saved` | SHOT on target, saved by keeper |
| SHOT_missed | `shot_missed` | SHOT off target |
| GOAL | `goal` | GOAL scored |
| TACKLE | `tackle` | TACKLE action taken |
| MOVE | — | Any non-football, non-tackle action (includes IDLE, movement, dribble, sprint) |

### 3.2 Event Alignment

Events are aligned to the tick at which the action was taken, not the tick at which the engine event fired. For PASS/SHOT, the action tick and the event tick may differ by 0–2 ticks due to engine processing delay. The forensic probe records both the action tick and the event tick.

---

## 4. Credit Measurement Definitions

### 4.1 TD Residual

```
delta_t = r_t + gamma * V(s_{t+1}) * next_nonterminal - V(s_t)
```

Where:
- `next_nonterminal = 0.0` if `terminated[t] = True`, else `1.0`
- For truncated episodes, `V(s_{t+1})` uses the bootstrap critic value at the final step

### 4.2 GAE Advantage

```
A_t = sum_{k=0}^{T-1-t} (gamma * lambda)^k * delta_{t+k}
```

Computed backwards from the terminal step using production `compute_gae` semantics.

### 4.3 Effective Credit Horizon

The empirical number of ticks backward from an event before the accumulated GAE contribution drops below the noise floor (defined as the interquartile range of MOVE TD residuals for that seed).

---

## 5. Discriminating Judgment Patterns

The following patterns are used to classify the credit probe results:

| Pattern ID | Observation | Interpretation |
|------------|-------------|----------------|
| P1 | Event residuals weak and long-range contribution collapses quickly | Critic/GAE credit problem (H3) becomes plausible |
| P2 | Event residuals strong but decay before reaching earlier decision points | Horizon/attenuation mechanism |
| P3 | Event advantages clearly useful/positive but MOVE remains dominant in actor logits at the same states | Actor/optimization pathway (H2) remains primary |
| P4 | One seed shows strong critic signal while others show weak/absent signal on the rare events they do produce | Trajectory/critic-coupled basin |
| P5 | Results remain ambiguous even with richer stochastic sample | Stay Class D |

Multiple patterns may apply simultaneously.

---

## 6. Prior Corrections Applied

The following corrections from prior forensic phases are in effect for this contract:

1. **GAE truncation bug fixed:** `eval_critic_gae_forensics.py` now uses `dones = terminated` only, matching production semantics. Re-verification shows this changes mean GAE advantages from +0.005 to +0.116 (fresh 50k seeds) to -0.10 to -0.21 under the corrected semantics.
2. **Baseline reconciliation:** Seed 123's 3.63% and 2.45% values are not comparable. Canonical baseline is 1.96% PASS+SHOT under `base_seed=500000`, 50 episodes, deterministic, aggregated across all agents.
3. **No reward/GAE/mask/network changes:** This phase is measurement-only.

---

## 7. Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-09-18 | Initial canonical metric lock for credit probe |
