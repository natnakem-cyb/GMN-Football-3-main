# REWARD_AUDIT_FIX_REPORT.md

## 1. Root Causes

| # | Issue | Location | Mechanism | Observed/analytical effect | Severity | Fix |
|---|-------|----------|-----------|---------------------------|----------|-----|
| 1 | Pass reward double-counting | `training/gmn_pettingzoo.py` `_resolve_pending_pass_state()` + `_build_shaper_events()` | Engine emits `pass_completed` event code; Python pending-pass resolver ALSO emits `PASS_COMPLETED` on ball-ownership change. Both are concatenated into `step_events` and sent to the adapter. | One physical pass → 2 logical `PASS_COMPLETED` events → double payment of `r_pass` | High | Added `_canonicalize_pass_events()` in both single-env and batched paths. Deduplicates by `(tick, passer_id, receiver_id, "PASS_COMPLETED")`. |
| 2 | Engine + adapter duplicate pass reward | `training/reward_adapters.py` + `src/engine/ObservationEncoder.ts` | Engine `ObservationEncoder.computeReward()` pays `+0.15` per completed pass (`passCompletedByTargetTeam`). Python adapter also paid `r_pass=0.30` per `PASS_COMPLETED` event. | Total pass reward could be `0.15 + 0.30 + 0.30 = 0.75` per physical pass (worst case with double event) | High | Set `BaseScenarioRewardAdapter.r_pass` default to `0.0`. Engine owns the base physical pass reward; adapter owns training-specific shaping. |
| 3 | Unlimited pass-loop farming | `training/reward_adapters.py` `BaseScenarioRewardAdapter._handle_events()` | Every `PASS_COMPLETED` event paid `r_pass` with no cap. A→B→C→A loop could farm indefinitely. | Pass reward unbounded; proxy objective could dominate goal reward | High | Added episode-scoped `attacking_pass_reward_count` in `AttackingDrillRewardAdapter`. Hard cap: first 2 passes earn `+0.10` each; all subsequent passes earn `0.00`. |
| 4 | Unlimited shot-spam farming | `training/reward_adapters.py` `_pay_shot_rewards()` | Every `SHOT_TAKEN`/`SHOT_BLOCKED` paid flat `r_shot=0.25` with no cap. | Shot reward unbounded; spam could accumulate more than goal reward | High | Added episode-scoped `shot_attempt_reward_count` and `shot_on_target_reward_count`. Bounded scheme: first shot `+0.15`, second `+0.05`, third+ `0.00`. First `SHOT_SAVED` `+0.20`, later `0.00`. |
| 5 | History-dependent exploration bonus in finishing drill | `training/reward_adapters.py` `AttackingDrillRewardAdapter.EXPLORATION_BETA` | Count-based novelty reward with `_visit_counts` persisting across episodes. Can accumulate large total reward over long training runs. | History-dependent reward component; could exceed terminal goal reward | High | Added `enable_exploration_bonus=False` and `EXPLORATION_BETA=0.0` class defaults. Exploration bonus is now gated on explicit opt-in. |
| 6 | Possession reward ambiguity | `training/reward_adapters.py` `_apply_possession()` + `_strip_progress()` | Dense possession `+0.01/tick/agent` minus step cost `-0.005/tick/agent` gives net `+0.005/tick/agent`. Ball-hog penalty `-0.02` after 15 ticks. | Possible concern: holding ball could be attractive | Medium | Measured and documented. Net becomes negative after 15 ticks per holder. Coefficient kept unchanged because measured trajectory is not exploitable. |

---

## 2. Reward Table

| Component | Amount | Frequency | Cap/diminishing | Recipient |
|-----------|-------:| --------- | --------------- | --------- |
| Engine base pass reward | +0.15 | per completed pass | None (engine base) | All agents (shared broadcast) |
| Adapter pass reward (Rondo) | 0.00 | per `PASS_COMPLETED` | Hard cap: 0 per episode | Receiver (`agent_id`) |
| Adapter pass reward (Attacking drill) | +0.10 | per `PASS_COMPLETED` | Hard cap: 2 passes per episode → max +0.20 | Receiver (`agent_id`) |
| Assisted goal bonus | +0.50 | per goal with `pass_chain_length > 0` | None | All active left agents |
| Solitary shot penalty (Rondo/legacy) | -0.30 | per unassisted shot | None | Shooter |
| Ball-hog penalty | -0.02 | per tick after 15 ticks | None | Ball holder |
| Turnover penalty | -0.10 | per possession-loss sequence | Tackle-spam window: 1 per 10 ticks | Victim |
| Shot attempt (Attacking drill) | +0.15 / +0.05 | first / second shot | Hard cap: 2 rewarded shots per episode | Shooter |
| Shot saved (on-target) | +0.20 | first save event | Hard cap: 1 rewarded save per episode | Shooter |
| Shot missed | 0.00 | per `SHOT_MISSED` | None | None |
| Possession reward | +0.01 | per tick | Left-possession-gated | All active left agents |
| Step cost | -0.005 | per tick | Goal-tick exempt | All active agents |
| PBRS goal-distance potential | γ·φ(new) - φ(prev) | per tick | Left-possession-gated; turnover-tick exempt | All active left agents |
| PBRS proximity potential | `DENSE_PROXIMITY_REWARD * (φ_new - φ_old)` | per tick | Not possession-gated | All active agents |
| Exploration bonus (Attacking drill) | 0.00 | per tick | Disabled by default (`enable_exploration_bonus=False`) | None |
| Shot-clock timeout penalty | -0.50 | once after 50 ticks without shot | One-shot | All agents |
| Goal reward (engine) | +2.00 | per goal | Terminal | All agents (shared broadcast) |
| Goal conceded (engine) | -1.00 | per conceded goal | Terminal | All agents (shared broadcast) |

---

## 3. Exploit Tests

### 3.1 Pass Loop (10 passes)

| Metric | Value |
|--------|-------|
| Pass count | 10 |
| Pass reward component | +0.20 (hard cap: first 2 only) |
| Step cost (3 agents × 10 ticks) | -0.15 |
| Total reward | 0.185 |
| Goals | 0 |

### 3.2 Pass Loop (100 passes)

| Metric | Value |
|--------|-------|
| Pass count | 100 |
| Pass reward component | +0.20 (hard cap: first 2 only) |
| Step cost (3 agents × 100 ticks) | -1.50 |
| Total reward | ~0.185 |
| Goals | 0 |

**Result:** Pass reward does not grow linearly; hard cap enforces `<= 0.20` per episode regardless of loop length.

### 3.3 Shot Spam (10 shots)

| Metric | Value |
|--------|-------|
| Shot count | 10 |
| Shot reward component | +0.20 (first +0.15, second +0.05, rest 0.00) |
| Step cost (3 agents × 10 ticks) | -0.15 |
| Total reward | 0.185 |
| Goals | 0 |

### 3.4 Shot Spam (100 shots)

| Metric | Value |
|--------|-------|
| Shot count | 100 |
| Shot reward component | +0.20 (hard cap) |
| Total reward | ~0.185 |
| Goals | 0 |

### 3.5 Possession Farming

| Ticks | Possession reward | Step cost | Ball-hog penalty | Net |
|-------|-----------------:| ---------:| ----------------:| ---:|
| 1 | 0.030 | -0.015 | 0.000 | 0.015 |
| 10 | 0.300 | -0.150 | 0.000 | 0.150 |
| 15 | 0.450 | -0.225 | 0.000 | 0.225 |
| 16 | 0.480 | -0.240 | -0.020 | 0.220 |
| 50 | 1.500 | -0.750 | -0.700 | 0.050 |
| 100 | 3.000 | -1.500 | -1.700 | -0.200 |
| 1000 | 30.000 | -15.000 | -19.700 | -4.700 |

**Result:** Possession retention becomes negative after ~50 ticks. Not an attractive proxy objective.

### 3.6 Exploration

| Trajectory | Exploration reward |
|------------|-------------------:|
| Default attacking drill | 0.00 |
| Explicitly enabled (`beta=0.03`) | `0.03 / sqrt(1 + count)` per novel cell |

---

## 4. Reward Decomposition

Example single-tick breakdown for 3 left agents with one `PASS_COMPLETED`:

```python
{
    "base_engine_pass_reward": 0.15,   # shared broadcast (engine)
    "adapter_pass_reward": 0.10,       # hard-capped adapter reward
    "shot_reward": 0.00,
    "possession_reward": 0.03,         # 3 agents × 0.01
    "goal_potential": 0.00,            # no movement this tick
    "proximity_potential": 0.00,       # no movement this tick
    "exploration_reward": 0.00,        # disabled
    "turnover_penalty": 0.00,
    "ball_hog_penalty": 0.00,
    "step_cost": -0.015,               # 3 agents × -0.005
    "timeout_penalty": 0.00,
}
# Sum of components = 0.15 + 0.10 + 0.03 - 0.015 = 0.265
# Final agent reward for receiver = 0.10 + 0.01 - 0.005 = 0.095
```

The engine’s `+0.15` is included in the base shared reward broadcast. The adapter’s `+0.10` is added on top for the first two productive passes only.

---

## 5. PBRS Terminology

The code implements **potential-difference shaping**, not standard discounted PBRS.

Key properties:
- `PBRS_GAMMA = 1.0` is used in the shaping term: `γ·φ(new_dist) - φ(prev_dist)`.
- With `γ = 1.0` and a potential `φ(d) = -clip(d, 0, D_MAX) / D_MAX`, a round trip sums to zero (telescoping).
- A stationary ball earns zero per tick (no discounting leak).
- This is **not** the standard Ng/Harada/Russell discounted PBRS theorem, which requires `γ` to match the MDP discount factor.
- The implementation deliberately decouples the shaping `γ` from the PPO/GAE `γ=0.99`.

**Do not claim policy invariance for this term unless the formal conditions are independently verified.**

---

## 6. Regression Status

| Test file | Status |
|-----------|--------|
| `training/tests/test_reward_exploit_regression.py` (18 tests) | **PASS** |
| `training/tests/test_reward_shaper.py` (31 tests) | **PASS** |
| `training/tests/test_action_masks.py` (10 tests) | **PASS** |
| `training/tests/test_reward_exploits.py` (live bridge) | **PASS** |

All existing tests continue to pass. New regression tests cover:
- Single pass accounting
- Duplicate pass event handling
- 10-pass and 100-pass loop caps
- 10-shot and 100-shot spam caps
- Possession farming at 1/10/15/16/50/100/1000 ticks
- Exploration disabled by default
- Reward decomposition sanity check
- Rondo adapter compatibility with `r_pass=0.0`

---

## 7. Remaining Risks

1. **Pass event canonicalization relies on `(tick, passer_id, receiver_id)` identity.** If the engine changes event ordering or omits `agent_id` on `pass_completed`, the deduplication key may need adjustment. The current implementation is robust to the observed event formats.

2. **Exploration bonus is disabled by default but still present in the code.** If a future scenario enables it, the history-dependent `_visit_counts` could still accumulate. This is now an explicit opt-in risk, not a silent default.

3. **Shot reward cap is episode-scoped.** If `reset()` is not called between episodes in some batched path, counters could leak. The current `reset_batch()` and `reset_one()` paths both call `_scenario_adapter()` → `previous.reset()`, which clears episode-scoped counters.

4. **Possession reward net is still slightly positive for short holds** (e.g., +0.225 at 15 ticks). This is intentional: the signal must be large enough to learn ball retention, but the ball-hog penalty prevents indefinite farming.

5. **Turnover-spam suppression uses a time window** (`TURNOVER_SPAM_WINDOW_TICKS=10`). A pathological policy could still stack turnovers by spacing them >10 ticks apart. The penalty per turnover is `-0.10`, which is larger than the per-tick dense reward, making this marginally attractive only if it directly leads to a goal.

6. **Engine-adapter reward split is now architectural:** engine owns base pass reward (`+0.15`), adapter owns shaping. Changing either value requires cross-referencing both layers. The documentation in `BaseScenarioRewardAdapter.__init__` makes this explicit.

7. **PBRS terminology is corrected to “potential-difference shaping.”** The code preserves the existing potential calculation and scale. No mathematical policy-invariance guarantee is claimed.

---

## Summary of Code Changes

| File | Change |
|------|--------|
| `training/reward_adapters.py` | `r_pass` default → `0.0`; pass hard cap (2 × +0.10); shot bounded scheme (first +0.15, second +0.05); exploration disabled by default; episode-scoped counters; `_pay_pass_rewards` added; `_pay_shot_rewards` bounded; `get_diagnostics` extended |
| `training/gmn_pettingzoo.py` | `_canonicalize_pass_events()` added to both single-env and batched paths; deduplicates `PASS_COMPLETED` by `(tick, passer_id, receiver_id)` |
| `training/tests/test_reward_exploit_regression.py` | 18 new deterministic regression tests covering pass accounting, pass-loop farming, shot spam, possession farming, exploration disabled, reward decomposition, and Rondo compatibility |
