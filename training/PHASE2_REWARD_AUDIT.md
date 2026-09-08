# Phase 2 — Reward Function Audit
## Scenario: `academy_3_vs_1_with_keeper`

## 2.1 Base Reward Components (ObservationEncoder.computeReward)

| Component | Value | Frequency | Recipient | Intended Behavior | Exploit Risk |
|-----------|------:| --------: | --------- | ----------------- | ------------ |
| goal_scored | +1.0 | terminal | all left agents (shared broadcast) | Score a goal | LOW — terminal, cannot be farmed |
| goal_conceded | -1.0 | terminal | all left agents (shared broadcast) | Penalize conceding | LOW — terminal |
| ball_progress_checkpoint | +0.05 max/step, `min(0.05, deltaX * 0.5)` when `currBallX > newMaxBallProgressX` and `deltaX > 0.005` | per-step dense | all left agents (shared broadcast) | Encourage ball advancement toward opponent goal | **HIGH** — only pays on monotonic high-water mark; passing can temporarily reduce ball X, so dribbling is incentivized over passing |
| shot_quality_bonus | +0.03 on-target, +0.001 off-target/weak | per-shot | all left agents (shared broadcast) | Encourage aiming at goal mouth | MEDIUM — magnitude is small relative to progress reward |
| shot_penalty (none) | 0 | — | — | — | — |

### 2.1.1 Critical Bug: Per-Agent Reward Assignment in `step()`
**File**: `training/gmn_pettingzoo.py`, lines 936–949 (pre-fix)

The `if is_rondo:` block and subsequent assignments to `rewards`, `terminations`, `truncations`, and `infos` were **outside** the `for i, agent in enumerate(self.agents):` loop. This meant only the **last agent** in `self.agents` received a reward entry; all other agents were missing from the returned dictionaries.

**Impact**: Any code accessing `rewards[agent]` for a non-last agent would receive `KeyError`. In `academy_rondo_4v1`, with agents `[left_1, left_2, left_3, left_4, right_1]`, only `right_1` had its reward set. This broke MAPPO's per-agent reward collection and any evaluation that sampled non-last agents.

**Status**: **FIXED** — moved the block inside the `for` loop.

---

## 2.2 Cooperative Reward Shaper Components (CooperativeRewardShaper)

| Component | Value | Frequency | Recipient | Intended Behavior | Exploit Risk |
|-----------|------:| --------: | --------- | ----------------- | ------------ |
| pass_completion_bonus | +0.25 | per completed pass | passing agent only | Encourage successful passing | MEDIUM — rapid pass cycling between nearby teammates could farm this without progressing |
| assisted_goal_bonus | +0.50 | per goal | all active left agents | Reward team goals built from pass chains | LOW — requires actual goal |
| solitary_shot_penalty | -0.30 | per shot | shooting agent only | Discourage unassisted shots | LOW — straightforward penalty |
| ball_hogging_penalty | -0.005/tick | per-tick (dense) | current ball holder only | Discourage indefinite possession by one player | **MEDIUM** — penalty is very small relative to progress reward; after 30 ticks, cumulative penalty is only -0.15 per additional 30 ticks, far less than one progress step (+0.05) |

---

## 2.3 Effective Reward Dominance Analysis

For `academy_3_vs_1_with_keeper`, the **dominant** reward signal is the **ball progress checkpoint** from `ObservationEncoder.computeReward`. The cooperative shaper's bonuses (+0.25 per pass, +0.50 per assisted goal) are event-sparse and only fire on discrete football events. The ball-hogging penalty (-0.005/tick) is too small to counteract the progress reward.

**Consequence**: The agent's optimal policy under current rewards is to maximize ball X-coordinate monotonically, which is achieved by dribbling forward rather than passing (passing risks losing possession or temporarily reducing ball X).

---

## 2.4 Defender Reward

For non-rondo scenarios, the defender (right team) receives **no explicit reward shaping**. The bridge server's `encodeStepBinary` only carries a single `reward` field for non-rondo scenarios. The right-team agents are controlled by rule-based bots, not learned policies, so this is acceptable for the current training setup.

---

## 2.5 MAPPO Per-Agent Reward Pipeline

**File**: `training/mappo_rollout.py`, `collect_rollout()`

The rollout buffer stores:
- `rewards`: shape `(n_steps,)` — shared team reward
- `per_agent_rewards`: shape `(n_steps, n_agents)` — individual per-agent rewards

The `compute_gae()` function accepts `per_agent_rewards` and uses them for per-agent advantage computation when available.

**Status**: Per-agent rewards are collected and passed through GAE. The `ppo_update()` function uses the shared `advantages` and `returns` for all agents (standard MAPPO with shared policy).

---

## 2.6 Summary of Root Causes

1. **Progress reward dominance**: The `+0.05 max/step` ball-progress checkpoint reward dwarfs all other signals. An agent can earn more by dribbling forward than by passing.
2. **Weak ball-hogging penalty**: The `-0.005/tick` penalty after 30 ticks of possession is insufficient to discourage possession farming.
3. **Sparse pass/shoot rewards**: Pass completion bonus (+0.25) and assisted goal bonus (+0.50) only fire on discrete events; the dense progress reward provides a continuous gradient that dominates learning.
4. **Per-agent reward assignment bug (FIXED)**: The `step()` method in `gmn_pettingzoo.py` only populated rewards for the last agent, breaking per-agent learning signals.
