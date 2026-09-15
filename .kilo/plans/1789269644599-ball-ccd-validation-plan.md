# Plan: Ball CCD Tunneling Fix — Validation & Diagnostic Reporting

## Current State

The physics fix is **already committed** as `abd8be2 fix(physics): CCD segment catch for shots in flight`:
- `Vec2.distPointToSegment2D` added to `src/engine/Vector.ts:69`
- `GameEngine.step` snapshots `prevBallPos` before `updateBall` at line 505
- `checkBallPossession(prevBallPos)` uses swept-volume segment distance when `isShotInFlight` is true (lines 939–944)
- Applied to **all** players in the loop, not just GK
- GK save-chance formula unchanged (`BASE_SAVE_CHANCE=0.75`, `SPEED_PENALTY_FACTOR=0.3`, `MIN_SAVE_CHANCE=0.15`)
- Outfield block rule added (lines 989–1003): deflect + clear `isShotInFlight`, emit `shot_blocked`
- `test_reward_exploits.py` assertions updated to `total_goals <= 2` (post-CCD bar)

## Remaining Gap

The user requires: *"paste the actual shot-attempted vs shot-saved/blocked counts from a fresh diagnostic run."*

Currently `ground_truth` in `gmn_pettingzoo.py:1342–1359` only exposes:
- `total_shots_left`, `shots_on_target_left`, `completed_passes_left`, etc.
- **No `shots_saved_left` or `shots_blocked_left` fields.**

`MatchStats` (`src/types/football.ts:150–165`) also lacks these counters, so the engine never accumulates them.

## Implementation Steps

### 1. Add `shotsSaved` / `shotsBlocked` to `MatchStats`

**File:** `src/types/football.ts:150–165`

Add two fields to `MatchStats`:
```ts
shotsSaved: { left: number; right: number };
shotsBlocked: { left: number; right: number };
```

### 2. Increment counters in `GameEngine.ts` when events are recorded

**File:** `src/engine/GameEngine.ts`

At the `shot_saved` event record site (~line 975), add:
```ts
this.stats.shotsSaved[player.team]++;
```

At the `shot_blocked` event record site (~line 996), add:
```ts
this.stats.shotsBlocked[player.team]++;
```

Initialize the new counters in `createDefaultStats()`.

### 3. Expose via `ground_truth` in `gmn_pettingzoo.py`

**File:** `training/gmn_pettingzoo.py:1342–1359`

Add to the `shared_info["ground_truth"]` dict:
```python
"shots_saved_left": episode_stats.get("shots_saved_left"),
"shots_blocked_left": episode_stats.get("shots_blocked_left"),
```

### 4. Expose via `bridge_server.ts` episode_stats serialization

**File:** `training/bridge_server.ts` (lines ~369, 511, 612)

Add `shots_saved_left` and `shots_blocked_left` to the episode_stats payload sent to Python, matching the pattern used for `shots_on_target_left` / `total_shots_left`.

### 5. Enhance `test_reward_exploits.py` to track and print counts

**File:** `training/tests/test_reward_exploits.py`

In `_run_policy`, add accumulators:
```python
total_shots_saved = 0
total_shots_blocked = 0
```

At the terminal step, read from `ground_truth`:
```python
ep_shots_saved += gt.get("shots_saved_left", 0)
ep_shots_blocked += gt.get("shots_blocked_left", 0)
```

After the 10-episode run, print a diagnostic summary:
```
Policy C (SHOT spam): {total_shots} shots, {total_shots_saved} saved, {total_shots_blocked} blocked, {total_goals} goals ({conversion_rate:.0f}% conversion)
```

### 6. Verify TypeScript compiles

Run `npx tsc --noEmit` to confirm no type errors from the `MatchStats` extension.

### 7. Run diagnostic and report

Run the 10-episode SHOT-spam diagnostic:
```bash
pytest training/tests/test_reward_exploits.py::TestRewardExploits::test_policy_c_shot_spam -v
pytest training/tests/test_reward_exploits.py::TestRewardExploits::test_policy_d_diagonal_shot -v
```

Report:
- Actual shot-attempted vs shot-saved/blocked counts
- Goal conversion rate (should be meaningfully below 100%)
- Pass/fail status for both tests

## Constraints Check

- ✅ No changes to `CooperativeRewardShaper`
- ✅ No changes to save-chance formula constants
- ✅ No changes to `dt`, ball-control radii, or catch-radius values
- ✅ Fix remains scoped to collision-detection method (CCD segment check)
- ✅ New fields are additive (stats tracking only, no behavior change)

## Risks

- `MatchStats` interface change may require updates in other consumers (e.g., `FootballMetrics`, replay serialization). Minimal risk — new fields are additive with default `0` values.
- Bridge server episode_stats serialization must match the new field names exactly.

## Definition of Done

- `MatchStats` includes `shotsSaved` and `shotsBlocked` counters
- Engine increments them on `shot_saved` / `shot_blocked` events
- `ground_truth` exposes `shots_saved_left` and `shots_blocked_left`
- `test_reward_exploits.py` prints diagnostic counts (shots / saved / blocked / goals)
- `test_policy_c_shot_spam` and `test_policy_d_diagonal_shot` re-run and results reported
- `tsc --noEmit` clean
