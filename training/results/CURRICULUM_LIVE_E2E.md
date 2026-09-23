# Task A — Curriculum Live End-to-End Report

**Date:** 2026-09-23  
**HEAD before B:** `b738ed6`  
**HEAD after B:** `2939366`  
**Pushed to origin/main:** Yes

---

## L1 — Live set_scenario switches engine stage

**Status:** PASS

Evidence:
- `test_live_env_set_scenario_switches_engine_stage` passes against real bridge.
- `env.scenario` changes from `academy_empty_goal` → `academy_run_to_score` after `set_scenario`.
- Next `env.reset()` and `env.step()` succeed in the new scenario.

---

## L2 — train_mappo.py --curriculum live wiring

**Status:** PASS (wiring confirmation)

Evidence:
- `test_train_mappo_curriculum_live_smoke_promotes_or_records` passes.
- Subprocess runs `train_mappo.py` with `--curriculum`, `timesteps=1024`, `promote_threshold=0.0`, `min_episodes=1`.
- Pre-seeded curriculum state has window of 5 successes.
- After run: `curriculum_state.json` shows `current_idx == 1` and `history[-1]["type"] == "promote"`.
- A `mappo_curriculum_promote_to_academy_run_to_score_*` checkpoint was written.

**Note:** This used `promote_threshold=0.0` and a pre-seeded window. This is a wiring confirmation, not a skill confirmation. The live bridge was available for this test.

---

## L3 — Persistence reload

**Status:** PASS

Evidence:
- `test_save_load_round_trip` passes (unit test, no bridge required).
- `CurriculumScheduler.save/load` round-trip preserves `current_idx`, `window`, `episodes_in_stage`, `history`, `total_episodes`.

---

## L4 — Demotion live

**Status:** PASS

Evidence:
- `test_live_demote_path_with_real_env` passes against real bridge.
- Scheduler starts on stage 1 with 5 failures; `evaluate_and_step()` demotes to `academy_empty_goal`.
- `env.set_scenario("academy_empty_goal")` + `env.reset()` succeeds.

---

## L5 — Adapter identity across finishing-family promotion

**Status:** PASS

Evidence:
- `test_live_adapter_stays_attacking_drill_across_finishing_promotion` passes against real bridge.
- `type(env.reward_adapter).__name__` is `AttackingDrillRewardAdapter` both before and after `academy_empty_goal` → `academy_run_to_score`.

---

## L6 — Real success signal from live engine

**Status:** PASS

Evidence:
- `test_live_success_signal_from_real_terminal_info` passes against real bridge.
- `is_scenario_success("academy_empty_goal", terminal_info)` matches `terminal_info["score"]["left"] > 0`.

---

## Pre-existing failure (not introduced by this work)

| Test | Failure mode | Classification |
|---|---|---|
| `test_live_single_pass_single_event_and_engine_reward` | Bridge timeout / no pass within 600 ticks | Pre-existing (bridge infra) |
| `test_live_scheduler_promotes_on_real_successes_or_records` | Bridge WS timeout mid-episode | Pre-existing (bridge infra, flaky) |

The `test_live_scheduler_promotes_on_real_successes_or_records` failure is a bridge-stability issue during a long-running live episode loop, not a code bug. It failed with `[GMN-PettingZoo WS Timeout] No frame received within 10.0s during 'step'.`

---

## Production code change

**File:** `training/train_mappo.py` line 357  
**Change:** Replaced Unicode arrow `→` with ASCII `->` in curriculum promotion print statement.  
**Reason:** The Unicode character caused `UnicodeEncodeError` on Windows cp1252 console when the live smoke test ran `train_mappo.py` as a subprocess. This was a real bug exposed by the live smoke test.

```python
# Before:
print(f"\n   [Curriculum] {direction.capitalize()} → {new_stage} ...")

# After:
print(f"\n   [Curriculum] {direction.capitalize()} -> {new_stage} ...")
```

No other production code was changed.

---

## Confirmation

- **No production-length curriculum train:** Yes
- **No 5v5/11v11 climb:** Yes
- **No silent mapping-bug fix:** Yes (mapping bug quoted in Task A report, not patched)
- **Fresh numbers generated on current HEAD:** Yes
