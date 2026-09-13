# GNN Phase 2 — Task Vector Specification

Stable 8-dimensional deterministic task-feature vector (`z_scenario`).

**Important:** For this phase, `z_scenario` is a deterministic feature vector, **not** a learned neural embedding.

---

## Dimension Table

| Index | Name | Meaning | Formula | Range | Source | Missing Semantics |
|-------|------|---------|---------|-------|--------|-------------------|
| 0 | `terminate_on_turnover` | Scenario terminates when opponent gains possession | `1.0` if `ScenarioConfig.terminateOnOpponentPossession === true` **or** scenario handler has equivalent turnover termination; otherwise `0.0` | `{0.0, 1.0}` | `ScenarioConfig.terminateOnOpponentPossession`, `ScenarioHandler.checkExtraTermination()` | `0.0` |
| 1 | `spatial_bounds_active` | Scenario enforces spatial bounds on ball/players | `1.0` if scenario handler or task spec defines bounds; otherwise `0.0` | `{0.0, 1.0}` | `ScenarioTaskSpec.spatialBounds`, handler logic | `0.0` |
| 2 | `time_remaining_frac` | Normalized fraction of time limit remaining | `max(0.0, totalTicks - tickCount) / totalTicks` | `[0.0, 1.0]` | `GameEngine.tickCount`, `ScenarioConfig.timeLimitSeconds` | `0.0` |
| 3 | `pass_progress` | Progress toward pass completion target | `min(completedPasses / targetPasses, 1.0)`; `0.0` when `targetPasses` is undefined | `[0.0, 1.0]` | `MatchStats.completedPasses.left`, `ScenarioTaskSpec.targetPassesCount` | `0.0` |
| 4 | `goal_progress` | Progress toward goal scoring target | `min(goalsScored / targetGoals, 1.0)`; `0.0` when `targetGoals` is undefined | `[0.0, 1.0]` | `MatchScore.left`, `ScenarioTaskSpec.targetGoalsCount` | `0.0` |
| 5 | `possession_left` | Left team currently possesses the ball | `1.0` if `ball.ownerId` exists and owner's team is `'left'`; otherwise `0.0` | `{0.0, 1.0}` | `Ball.ownerId`, `Player.team` | `0.0` |
| 6 | `reserved_future_1` | Reserved for future tactical/contextual flags | — | — | — | `0.0` |
| 7 | `reserved_future_2` | Reserved for future tactical/contextual flags | — | — | — | `0.0` |

---

## Stability Guarantees

- **Fixed size:** Always 8 floats (`TASK_VECTOR_DIM = 8`).
- **Deterministic:** Same `ScenarioConfig` + `ScenarioDynamicState` always produces identical vector.
- **Missing-safe:** Unavailable inputs produce documented missing values; no exceptions are thrown.
- **Scenario-agnostic:** Vector structure is identical for all scenarios. Individual dimensions may be inactive (0.0) for scenarios where the concept does not apply.

---

## Normalization Notes

- **Binary flags** (`terminate_on_turnover`, `spatial_bounds_active`, `possession_left`): encoded as `0.0` or `1.0`.
- **Fraction features** (`time_remaining_frac`): linearly normalized to `[0.0, 1.0]`.
- **Progress features** (`pass_progress`, `goal_progress`): ratio capped at `1.0`; denominator of `0` or undefined yields `0.0`.

---

## Exclusions

The following are **explicitly excluded** from this phase because the engine does not currently expose them as deterministic, authoritative signals:

| Excluded Concept | Reason |
|---|---|
| Discrete touch count | Engine tracks pass attempts and completions, not per-possession touch events |
| Sequence stage | No sequence state machine exists |
| Allowed/forbidden actions | Engine does not restrict actions by scenario |
| Dribble/shoot permission flags | Not enforced at scenario level |
| Target zone / target player | Not defined in scenario data |
| Exact owner identity in rawVector | Separate from `rawVector`; exposed via `zScenario` only |
