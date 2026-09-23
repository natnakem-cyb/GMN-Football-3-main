# Experiment D — Tackle Spam Forensics (Current HEAD)

**Date:** 2026-09-23  
**HEAD at investigation:** `007ffc59c9caf50919d1682ead6f761ce8b729cd`  
**Status:** Measurement complete — mechanism not reproduced on current checkpoints  
**Objective:** Re-satisfy D's exit criterion on current code + current checkpoints after reward fixes landed since f2fd157.

---

## A.0 — Mapping bug still present (read-only)

**Quoted from `training/gmn_pettingzoo.py:1553-1555`** (`_build_shaper_events`):
```python
elif ev_type in ("interception", "tackle", "foul"):
    shaper_type = "TURNOVER_CONCEDED"
    shaper_team = "right"
```

**Quoted from `training/gmn_pettingzoo.py:1932-1934`** (single-env `step()` event mapping):
```python
elif ev_type in ("interception", "tackle", "foul"):
    shaper_type = "TURNOVER_CONCEDED"
    shaper_team = "right"
```

**Confirmed:** both sites still hardcode `shaper_team = "right"` for `tackle`, `foul`, and `interception` events. A left-team tackle is mapped to `TURNOVER_CONCEDED` with `team="right"`, which penalizes the left team as if they lost possession when they actually won it. This is a **follow-up item** — out of scope for Task A, not patched here.

---

## A.1 — Instrumenter fix

**File:** `training/eval_tackle_forensics.py`  
**Class:** `InstrumentedAdapter.compute_shaped_rewards`

**Before (double-apply bug):**
```python
def compute_shaped_rewards(self, base_rewards, step_events, info_ground_truth, active_agents, actions=None, tick=0, max_ticks=None):
    shaped = {a: base_rewards.get(a, 0.0) for a in active_agents if not a.startswith("right_")}
    self._current_tick = tick
    base_snapshot = {a: float(base_rewards.get(a, 0.0)) for a in active_agents if not a.startswith("right_")}

    # Manually apply shaping helpers
    self._apply_action_cost(shaped, actions)
    self._apply_possession(shaped, info_ground_truth)
    self._handle_events(shaped, step_events, active_agents, actions)

    # Then super() applies them AGAIN — double-apply
    result = super().compute_shaped_rewards(...)
```

**After (single super() call):**
```python
def compute_shaped_rewards(self, base_rewards, step_events, info_ground_truth, active_agents, actions=None, tick=0, max_ticks=None):
    self._current_tick = tick
    base_snapshot = {a: float(base_rewards.get(a, 0.0)) for a in active_agents if not a.startswith("right_")}

    # Call super() exactly once — parent applies all shaping terms in a single pass
    result = super().compute_shaped_rewards(
        base_rewards, step_events, info_ground_truth, active_agents,
        actions=actions, tick=tick, max_ticks=max_ticks,
    )

    delta = {a: float(result.get(a, 0.0)) - base_snapshot.get(a, 0.0) for a in base_snapshot}

    self.tick_log.append({
        "tick": tick,
        "base_rewards": base_snapshot,
        "shaped_rewards": {a: float(result.get(a, 0.0)) for a in base_snapshot},
        "delta": delta,
        "step_events": [...],
        "info_ground_truth": info_ground_truth,
        "actions": actions,
    })
    return result
```

**Constructor kwargs verified:** `InstrumentedAdapter` is constructed with `step_cost`, `shot_reward`, `on_target_reward`, `t_max`, `timeout_penalty`, `enable_exploration_bonus`, `exploration_beta`. These match the current `AttackingDrillRewardAdapter.__init__` signature (`reward_adapters.py:350-355`).

---

## A.2 — Checkpoint selection and provenance

### Current files on disk

| Candidate | Path | Size (MB) | SHA256 (current) | 9/14 sidecar SHA256 | Match? |
|-----------|------|-----------|------------------|---------------------|--------|
| mappo_ac3v1_seed42_200k_B | `training/models/mappo_ac3v1_seed42_200k_B` | 4.1 | `1d9be7e16f07d9becaa1de364510bee48b15256768fc14327e47532c856bf47b` | `0b9d885...` (E0) / `9b71c94...` (E1) | **MISMATCH** |
| mappo_ac3v1_seed123_200k_B | `training/models/mappo_ac3v1_seed123_200k_B` | 4.1 | `7624efee69555a3e8e8a1c582ef0cdb39b51d31edfd8555e559e63ceaa95e8cc` | `53c1e27...` (E0) / `e0caa7f...` (E1) | **MISMATCH** |
| mappo_ac3v1_seed7_200k_B | `training/models/mappo_ac3v1_seed7_200k_B` | 4.1 | `73581605c65bd7d298be848b23de1d29ced52c4680147957a17825dcd3726eaf` | `46f9765...` (E0) / `66dbfde...` (E1) | **MISMATCH** |
| mappo_ac3v1_seed999_200k_B | `training/models/mappo_ac3v1_seed999_200k_B` | 4.1 | `c1b36f41d74975f4154923bcdec920f9f0f67f58998c59de65394dba9d0f7ebb` | `9e48e56...` (E0) / `dd302da...` (E1) | **MISMATCH** |

**Historical tackle-spam blobs:** The original checkpoint files that exhibited tackle spam on 9/14 are **not present** in the current working tree. The SHA256 hashes recorded in `comprehensive_eval_*_clean.json` sidecars do not match any file currently on disk. These files are **not recoverable** from git history.

**Current files evaluated:** All four `mappo_ac3v1_seed*_200k_B` extensionless files (4.1 MB each, `timesteps=199936`). These are the highest-step checkpoints available for the ac3v1 family.

---

## A.3 — Forensic eval results

### Policy eval (10 episodes each, deterministic, base_seed=500000)

| Seed | Episodes | Goals | Mean tackles/ep | Mean shots/ep | Mean passes/ep | Mean reward |
|------|----------|-------|-----------------|---------------|----------------|-------------|
| 42   | 10       | 0.0%  | 0.0             | 0.0           | 0.0            | -0.174      |
| 123  | 10       | 0.0%  | 0.0             | 0.0           | 0.0            | -0.133      |
| 7    | 10       | 0.0%  | 0.0             | 0.0           | 0.0            | -0.181      |
| 999  | 10       | 0.0%  | 0.0             | 0.0           | 0.0            | -0.168      |

**Tackle-heavy episodes (≥3 tackles): 0 across all seeds.**

**First-tick action selection:**
- seed42: all agents select DRIBBLE (17)
- seed123: all agents select DRIBBLE (17)
- seed7: all agents select DRIBBLE (17)
- seed999: all agents select DRIBBLE (17)

**First-tick mask (off-ball left agents):**
- `tackle_legal: 1`
- `shot_legal: 0`
- `pass_legal: 0`
- `mask_sum: 15`

**Conclusion:** Current weights exhibit **policy paralysis**, not tackle spam. The policy selects DRIBBLE at kickoff and never selects tackle, shot, or pass during the episode.

### Forced-tackle probe (seed42, 3 episodes; seed7, 2 episodes)

Forced SLIDING action on off-ball left agents for the first 20 ticks of each episode.

| Checkpoint | Tackle ticks | Event codes emitted | TURNOVER_CONCEDED (team=right) | Mean adapter delta (tackle ticks) | Mean adapter delta (non-tackle offball ticks) |
|------------|-------------|---------------------|-------------------------------|----------------------------------|----------------------------------------------|
| seed42     | 56          | 0                   | 0                             | -0.0153                          | -0.0151                                      |
| seed7      | 36          | 0                   | 0                             | -0.0152                          | N/A (non-tackle sample empty)                |

**Key observations:**
- Forcing tackle out-of-range produces **no events** (no `event_code`, no `step_events`). The engine does not register a tackle when the agent is too far from the ball owner.
- Forcing tackle produces **no `TURNOVER_CONCEDED` events** with `team="right"`. The event-mapping bug is therefore not triggered by forced out-of-range tackles because the engine never emits a `tackle` event in the first place.
- Adapter delta on forced tackle ticks is slightly more negative than non-tackle offball ticks (-0.0153 vs -0.0151), but the difference is negligible (~0.0002). This is within the variance of the dense proximity reward and action cost.
- No possession changes occur during forced tackle ticks (ball remains unowned or with right team).

---

## A.4 — Hypothesis table (updated against current HEAD)

| Hypothesis | Assessment | Evidence |
|------------|------------|----------|
| **H-mask** | **Partially supported** — tackle is always legal for off-ball agents, but current policy does not select it. Mask is necessary but not sufficient. | 10 episodes × 4 seeds: tackle_legal=1 at every tick, tackle_actions=0 |
| **H-free-action** | **Supported** — failed tackles are free. Forced out-of-range tackles produce no event, no penalty, only step cost (~-0.005) and dense proximity reward. | Forced probe: 92 tackle ticks, 0 events, 0 TURNOVER_CONCEDED |
| **H-downstream** | **Unproven** — dense proximity reward while defending exists. Forced tackle ticks show delta similar to non-tackle offball ticks (-0.0153 vs -0.0151), suggesting proximity reward is not significantly enhanced by tackling. | Forced probe delta comparison |
| **H-engine** | **Partially supported** — tackle success requires proximity to ball owner. Forced tackles out of range produce no engine event. | Forced probe: 0 tackle events emitted |
| **H-event-mapping** | **Confirmed present, not triggered** — the `team="right"` hardcoding exists at both sites (gmn_pettingzoo.py:1553-1555, 1932-1934) but is not exercised by current weights because no tackle events are emitted. | Code inspection |
| **H-paralysis** | **Primary on current weights** — all 4 checkpoints show 0 tackles, 0 shots, 0 passes over 10 episodes each. Policy selects DRIBBLE at kickoff and persists with idle/movement. | Policy eval |

### Primary mechanism on CURRENT weights
**H-paralysis** is the sole observable behavior. The policy has converged to a DRIBBLE/idle attractor and never attempts any productive action.

### Best explanation of HISTORICAL tackle spam
**H-free-action + H-engine + H-event-mapping** (combination). The original tackle-spam checkpoints (SHA256s recorded in 9/14 sidecars) are no longer on disk, so this cannot be directly reproduced. The best-supported narrative from 9/14 data is:
1. Failed tackles are free (no penalty, only step cost).
2. Successful tackles deflect the ball, creating loose-ball scramble.
3. The event-mapping bug penalizes left-team tackles as if they were turnovers.
4. Tackle-spam suppression prevents rapid re-tackle penalties within 10 ticks.
5. The net expected value may have been slightly positive due to proximity reward during scramble.

This explanation is **historical, unreproducible on disk** — the original weights are gone.

---

## A.5 — Deliverables

| File | Description |
|------|-------------|
| `training/eval_tackle_forensics.py` | Instrumenter fix: `super().compute_shaped_rewards` called exactly once |
| `training/eval_tackle_forensics_forced_probe.py` | New forced-tackle probe script |
| `training/models/tackle_forensics_mappo_ac3v1_seed42_200k_B.json` | Policy eval (10 episodes) — NEW file, current HEAD |
| `training/models/tackle_forensics_mappo_ac3v1_seed123_200k_B.json` | Policy eval (10 episodes) — NEW file, current HEAD |
| `training/models/tackle_forensics_mappo_ac3v1_seed7_200k_B.json` | Policy eval (10 episodes) — NEW file, current HEAD |
| `training/models/tackle_forensics_mappo_ac3v1_seed999_200k_B.json` | Policy eval (10 episodes) — NEW file, current HEAD |
| `training/models/tackle_forensics_forced_probe_mappo_ac3v1_seed42_200k_B.json` | Forced probe (3 episodes) — NEW file |
| `training/models/tackle_forensics_forced_probe_mappo_ac3v1_seed7_200k_B.json` | Forced probe (2 episodes) — NEW file |
| `training/results/tackle_forensics_summary.csv` | Aggregated metrics (current HEAD rows only) |
| `training/results/EXPERIMENT_D_TACKLE_FORENSICS_CURRENT_HEAD.md` | This report |

**Not modified:** `training/results/EXPERIMENT_D_TACKLE_SPAM_FORENSICS.md` (9/17 historical record preserved).

---

## TASK A CONFIRMATIONS

| Confirmation | Status |
|-------------|--------|
| No training performed | yes |
| No production reward/mask/engine/physics change | yes |
| Event-mapping bug not silently fixed (quoted, flagged as follow-up) | yes |
| Fresh numbers generated on current HEAD (not recycled 9/17 tables) | yes |
| Checkpoint SHA256 recorded for every file evaluated | yes |
| Instrumenter no longer double-applies compute_shaped_rewards | yes |
