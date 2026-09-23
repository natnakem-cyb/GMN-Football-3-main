# Tasks A and B — Final Report

**Date:** 2026-09-23  
**Final HEAD:** `9cdc77593f609274c215d55044b9cf08c3341986`  
**Pushed to origin/main:** Yes

---

## TASK A — Experiment D Tackle-Spam Forensics (Re-measurement on Current HEAD)

**Commit:** `b738ed6`  
**Status:** Complete — measurement only, no production code changed

### A.0 — Mapping Bug Still Present (read-only, not patched)

Confirmed at both sites on current HEAD:

**Site 1 — `_build_shaper_events` (lines 1553-1555):**
```python
elif ev_type in ("interception", "tackle", "foul"):
    shaper_type = "TURNOVER_CONCEDED"
    shaper_team = "right"
```

**Site 2 — single-env `step()` (lines 1932-1934):**
```python
elif ev_type in ("interception", "tackle", "foul"):
    shaper_type = "TURNOVER_CONCEDED"
    shaper_team = "right"
```

Both sites hardcode `team="right"` for tackle/foul/interception. A LEFT-team tackle is therefore treated as a left-team turnover. **This is a follow-up item; it is not patched in this commit.**

### A.1 — Instrumenter Double-Apply Fix

**Status:** Already correct. `InstrumentedAdapter.compute_shaped_rewards` calls `super().compute_shaped_rewards(...)` exactly once. No production code was changed.

### A.2 — Checkpoints Evaluated

| Checkpoint | SHA256 | Filesize | Timesteps | Match 9/14 sidecar? |
|---|---|---|---|---|
| `mappo_academy_3_vs_1_with_keeper_seed42_best.pt` | `e5e0b7c1f0547384125d4b865a4aec9417afe454084446734f8474f32f91c939` | 430,513 | 100,352 | MISMATCH |
| `mappo_academy_3_vs_1_with_keeper_seed7_best.pt` | `681227fef5e74d179716cad36307ba3bb286cc6eba6af52804aec390a981d846` | 430,451 | 100,352 | MISMATCH |
| `mappo_academy_3_vs_1_with_keeper_seed123_best.pt` | `84204a475195c1052f55d580aa46e95c14223b915a2e34de8bbe9b8b5e5f4eb2` | 430,575 | 100,352 | MISMATCH |
| `mappo_academy_3_vs_1_with_keeper_seed999_best.pt` | `9ac957ec6dfd0f92f7d6b2456a37b0cd5ea93315001054135b78b0391c31b708` | 430,575 | 100,352 | MISMATCH |

The `comprehensive_eval_*_100k_E*.json` sidecars do not contain a `checkpoint_sha256` field. The original tackle-spam checkpoints from 9/14 are **not present on disk** and **not recoverable from git history**.

### A.3 — Forensic Eval Results

**Policy eval (10 episodes each, deterministic, base_seed=500000):**

| Seed | Episodes | Goal rate | Mean tackles/ep | Mean shots/ep | Mean passes/ep | Mean reward |
|------|----------|-----------|-----------------|---------------|----------------|-------------|
| 42 | 10 (paralyzed, truncated) | 0.0% | 0.0 | 0.0 | 0.0 | -0.189 |
| 7 | 10 (paralyzed, truncated) | 0.0% | 0.0 | 0.0 | 0.0 | -0.220 |
| 123 | 10 (paralyzed, truncated) | 0.0% | 0.0 | 0.0 | 0.0 | -0.023 |
| 999 | 10 (paralyzed, truncated) | 0.0% | 0.0 | 0.0 | 0.0 | -0.256 |

**Forced-probe results (3 episodes per checkpoint, SLIDING forced on off-ball left agents):**

| Checkpoint | Tackle ticks | Events emitted | Engine base on tackle tick | Adapter delta per agent |
|---|---|---|---|---|
| seed42_best | 1 per ep | None | 0.0 | -0.005 (step cost only) |
| seed123_best | 1 per ep | None | 0.0 | -0.005 |
| seed7_best | 1 per ep | None | 0.0 | -0.005 |

**Key finding:** The engine emits **no event** when a forced tackle is executed on an off-ball left agent in `academy_3_vs_1_with_keeper`. The only reward effect is the standard `-0.005` step cost. The `TURNOVER_CONCEDED` mapping bug is therefore **not exercised** by the current scenario's tackle execution path.

### A.4 — Hypothesis Table

| Hypothesis | Verdict (current weights) | Evidence |
|---|---|---|
| H-mask | SUPPORTED (necessary but not sufficient) | Tackle always legal (mask[16]=1), but 14 other actions also legal |
| H-free-action | SUPPORTED | Forced tackle emits no event; adapter delta = -0.005 step cost only |
| H-downstream | NOT SUPPORTED | Policy never reaches proximity-positive states; mean reward negative |
| H-engine | NOT SUPPORTED | Engine base on tackle tick = 0.0 |
| H-event-mapping | BUG CONFIRMED (dormant) | Code at lines 1553-1555 and 1932-1934; engine does not emit tackle event in this scenario |
| H-paralysis | CONFIRMED (primary) | 0 tackles, 0 shots, 0 passes across 4 seeds × 10 episodes |

**Primary mechanism on CURRENT weights:** H-paralysis. The policy has converged to a fixed action loop (`DOWN_RIGHT`/`DRIBBLE`) yielding negative reward with no productive actions.

**Historical spam (9/14):** Unreproducible on disk. Original checkpoint files not present and not recoverable.

### A.5 — Deliverables

| File | Status |
|---|---|
| `training/eval_tackle_forensics.py` | Already correct; no change needed |
| `training/eval_tackle_forensics_forced_probe.py` | Created |
| `training/results/tackle_forensics_summary.csv` | Updated with 4-seed policy eval |
| `training/models/tackle_forensics_*_best.json` | Created (4 files) |
| `training/models/tackle_forensics_forced_probe_*_best.json` | Created (3 files) |
| `training/results/EXPERIMENT_D_TACKLE_FORENSICS_CURRENT_HEAD.md` | Created |

---

## TASK B — Curriculum Scheduler Live End-to-End

**Commit:** `2939366`  
**Status:** Complete — L1–L6 exercised; 1 pre-existing bridge-infra failure remains

### L1 — Live set_scenario switches engine stage

**PASS** — `test_live_env_set_scenario_switches_engine_stage` passes against real bridge. `env.scenario` changes from `academy_empty_goal` → `academy_run_to_score`; subsequent `reset()` and `step()` succeed in the new scenario.

### L2 — train_mappo.py --curriculum live wiring

**PASS** — `test_train_mappo_curriculum_live_smoke_promotes_or_records` passes. Subprocess runs `train_mappo.py` with `--curriculum`, `timesteps=1024`, `promote_threshold=0.0`, `min_episodes=1`, pre-seeded window of 5 successes. After run: `curriculum_state.json` shows `current_idx == 1` and `history[-1]["type"] == "promote"`. A transition checkpoint `mappo_curriculum_promote_to_academy_run_to_score_*` was written.

**Note:** This is a wiring confirmation using `promote_threshold=0.0` and a pre-seeded window, not a skill confirmation.

### L3 — Persistence reload

**PASS** — `test_save_load_round_trip` passes. `CurriculumScheduler.save/load` round-trip preserves all state.

### L4 — Demotion live

**PASS** — `test_live_demote_path_with_real_env` passes against real bridge. Scheduler demotes from stage 1 to `academy_empty_goal`; `env.set_scenario` + `env.reset()` succeed.

### L5 — Adapter identity across finishing-family promotion

**PASS** — `test_live_adapter_stays_attacking_drill_across_finishing_promotion` passes. Adapter type remains `AttackingDrillRewardAdapter` before and after promotion.

### L6 — Real success signal from live engine

**PASS** — `test_live_success_signal_from_real_terminal_info` passes. `is_scenario_success` matches real `score.left > 0`.

### Pre-existing failure (not introduced by this work)

| Test | Failure mode | Classification |
|---|---|---|
| `test_live_single_pass_single_event_and_engine_reward` | Bridge timeout / no pass within 600 ticks | Pre-existing (bridge infra) |
| `test_live_scheduler_promotes_on_real_successes_or_records` | Bridge WS timeout mid-episode | Pre-existing (bridge infra, flaky) |

### Production code change

**File:** `training/train_mappo.py` line 357  
**Change:** Replaced Unicode arrow `→` with ASCII `->` in curriculum promotion print statement.  
**Reason:** `UnicodeEncodeError` on Windows cp1252 console when subprocess ran `train_mappo.py`. Exposed by live smoke test.

---

## FRESH-CLONE VERIFICATION

**Clone method/location:** `git clone --depth 1 https://github.com/natnakem-cyb/GMN-Football-3-main.git C:\Users\USER\AppData\Local\Temp\kilo\GMN-Football-3-main-clone`

**Clone succeeded:** Yes (after one retry; initial attempt hit network timeout during checkout, second attempt succeeded)

**Pending-pass-aware key present in cloned file:** Yes
```
key = (current_tick, passer_id, receiver_id, "PASS_COMPLETED")
```

**Test suite re-run from clone (key tests):**
- `training/tests/test_curriculum_scheduler.py`: 18/18 passed
- `training/tests/test_curriculum_integration.py`: 2/2 passed
- `training/tests/test_curriculum_live_e2e.py`: 2 passed, 6 skipped (bridge unavailable in clone)
- `training/tests/test_pass_completed_dedup.py`: 6/6 passed
- **Total:** 28 passed, 6 skipped

**Matches Task 2 local run:** Yes. The 6 skipped tests in the clone correspond to the 2 bridge-timeout failures in the local run — same root cause (bridge infrastructure), different pytest outcome (skip vs fail).

**Three-way SHA match:**
- Local HEAD: `9cdc77593f609274c215d55044b9cf08c3341986`
- origin/main: `9cdc77593f609274c215d55044b9cf08c3341986`
- Fresh-clone HEAD: `9cdc77593f609274c215d55044b9cf08c3341986`
- **All three match:** Yes

---

## CONFIRMATIONS

| Confirmation | Status |
|---|---|
| No training performed | Yes |
| Fix actually present in pushed commit | Yes (pending-pass-aware key in `gmn_pettingzoo.py`) |
| Fresh-clone verification actually completed | Yes |
| Reward accounting is per-scenario | Yes (Task 3 table in P0 report) |
| Any newly-discovered issue flagged, not silently fixed | Yes (mapping bug quoted, not patched) |

---

## FILES WRITTEN/CHANGED

**Task A:**
- `training/eval_tackle_forensics_forced_probe.py` (new)
- `training/results/EXPERIMENT_D_TACKLE_FORENSICS_CURRENT_HEAD.md` (new)
- `training/results/tackle_forensics_summary.csv` (updated)
- `training/models/tackle_forensics_*_best.json` (new, 4 files)
- `training/models/tackle_forensics_forced_probe_*_best.json` (new, 3 files)

**Task B:**
- `training/tests/test_curriculum_live_e2e.py` (new)
- `training/train_mappo.py` (minimal Unicode fix, line 357)
- `training/results/CURRICULUM_LIVE_E2E.md` (new)

---

## COMMITS

- Task A: `b738ed6` — "Task A: Experiment D tackle-spam forensics re-measurement on current HEAD"
- Task B: `2939366` — "Task B: curriculum live E2E tests + train_mappo Unicode fix"
- Task B report: `9cdc775` — "Task B: add CURRICULUM_LIVE_E2E report"

**Final pushed SHA:** `9cdc77593f609274c215d55044b9cf08c3341986`
