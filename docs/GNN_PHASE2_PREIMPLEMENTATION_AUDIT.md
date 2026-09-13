# GNN Phase 2 — Pre-Implementation Audit

Date: 2026-09-13
Scope: `src/types/`, `src/scenarios/`, `src/engine/`, `training/`

---

## 1. Authoritative Sources Identified

| Concept | Authoritative Source | Notes |
|---|---|---|
| Scenario definitions | `src/scenarios/ScenarioRegistry.ts` | Single source of truth for all academy scenarios |
| Scenario objectives | `ScenarioConfig.objectives` (`ScenarioObjective[]`) | Text + completion flags only |
| Scenario termination | `GameEngine.evaluateScenarioConditions()`, `GameEngine.step()` termination expression, `RondoScenarioHandler.checkExtraTermination()` | Two paths: standard academy goal/possession, and rondo-specific |
| Player possession | `Ball.ownerId` (authoritative), `Player.hasBall` (derived flag) | Exact owner identity is available engine-side |
| Player control | `GameEngine.controlledPlayerId` | Explicit ID, not role-inferred |
| Ball touches | **NOT AVAILABLE** as discrete events | Engine tracks `stats.passes` / `stats.completedPasses` per team, but not per-player touch counts per possession |
| Completed passes | `MatchStats.completedPasses.left|right` | Incremented in `checkBallPossession()` when same-team receiver differs from passer |
| Goals | `MatchScore.left|right`, `MatchStats.goals.left|right` | Incremented in `checkGoalAndBoundaries()` |
| Tick count | `GameEngine.tickCount` | Increments every physics tick, including during goal reset timer |
| Sequence progression | **NOT AVAILABLE** | No sequence stage counter exists |
| Scenario-specific counters | `RondoScenarioHandler` only | `defenderPossessionTime`, `consecutivePossessionTime`, `ballOutOfAreaTime` |
| Scenario handlers | `src/engine/scenarios/ScenarioHandler.ts`, `RondoScenarioHandler.ts` | Hook interface + one implementation |
| Action semantics | `ActionType` enum in `src/types/football.ts` | 19 discrete actions |
| Formation definitions | `src/engine/Rules.ts` (`FORMATIONS`, `getFormationPositions()`) | Static lookup by `FormationType` |
| Zone/bounds definitions | `src/engine/Rules.ts` (`PITCH`) | Pitch bounds, goal mouth, penalty box, center circle |
| Observation contract | `src/engine/Contract.ts`, `ObservationEncoder.ts` | 127-dim `simple115_v3_role` fixed schema |

---

## 2. Field Availability Matrix

| Desired Task Field | Authoritative Source | Currently Available? | Exact Semantics | Deterministic? | Observable by RL? |
|---|---|---|---|---|---|
| `scenario_id` | `ScenarioConfig.id` | Yes | Stable identifier | Yes | No (separate from rawVector) |
| `task_type` | **NOT DEFINED** | No | Would require new taxonomy | — | — |
| `formation` | `TeamConfig.formation` / scenario setup | Yes | Static formation type | Yes | No |
| `allowed_actions` | **NOT DEFINED** | No | Engine does not restrict actions by scenario | — | — |
| `forbidden_actions` | **NOT DEFINED** | No | Engine does not declare forbidden actions | — | — |
| `touch_limit_per_possession` | **NOT DEFINED** | No | No touch counter exists | — | — |
| `dribbling_allowed` | **NOT DEFINED** | No | `ActionType.DRIBBLE` exists but no scenario-level flag | — | — |
| `shooting_allowed` | **NOT DEFINED** | No | `ActionType.SHOT` exists but no scenario-level flag | — | — |
| `passing_required` | **NOT DEFINED** | No | No pass-requirement enforcement | — | — |
| `target_passes_count` | Scenario objectives text only | Partial | `objectives` text says "Complete 10+ passes" but no numeric constraint field | — | — |
| `target_goals_count` | **NOT DEFINED** | No | No numeric goal target in scenario data | — | — |
| `spatial_bounds` | **NOT DEFINED** | No | Rondo uses implicit `0.35` radius in handler, not in scenario config | — | — |
| `step_limit` | `ScenarioConfig.timeLimitSeconds` | Yes | Converted to ticks via `timeLimitSeconds * 60` | Yes | No |
| `terminate_on_turnover` | `terminateOnOpponentPossession` + rondo handler | Partial | Maps to engine termination, but rondo has custom logic | Partial | No |
| `ticks_remaining` | `tickCount` + `timeLimitSeconds` | Yes | `max(0, timeLimitSeconds * 60 - tickCount)` | Yes | No |
| `total_ticks` | `timeLimitSeconds` | Yes | `timeLimitSeconds * 60` | Yes | No |
| `passes_completed` | `stats.completedPasses.left` | Yes | Team-level completed pass counter | Yes | No |
| `goals_completed` | `score.left` / `stats.goals.left` | Yes | Goal counter | Yes | No |
| `current_possession_team` | `ball.ownerId` | Yes | Exact team ownership | Yes | No |
| `touches_in_current_possession` | **NOT AVAILABLE** | No | Would require per-player touch event tracking | — | — |
| `sequence_stage` | **NOT AVAILABLE** | No | No sequence state machine exists | — | — |
| `sequence_length` | **NOT AVAILABLE** | No | No sequence definition exists | — | — |

---

## 3. Key Constraints Discovered

### 3.1 Touch Semantics
The engine does **not** track discrete touch events. The closest available signals are:
- `stats.passes[team]` — pass attempts by team
- `stats.completedPasses[team]` — successful same-team receptions
- `ball.lastKickedBy` / `ball.lastKickedTeam` — last kicker identity
- `currentPassTracking` — in-flight pass tracking

None of these equal "touches in current possession." A player holding the ball for 10 ticks without passing or shooting has made 1 touch (the reception), not 10.

### 3.2 Pass Count vs Sequence Progress
`stats.completedPasses.left` counts total successful passes in the episode. It does **not** track whether a required sequence (e.g., A→B→C→Shot) is being followed. The engine has no sequence state machine.

### 3.3 Turnover Semantics
Engine termination has one authoritative path in `GameEngine.step()`:
```typescript
const isTerminated = this.status === 'fulltime'
  || isAcademyGoal
  || isOpponentPossession
  || extraTermination;
```
- `isAcademyGoal`: academy scenarios terminate on left-team goal (unless handler skips goal check)
- `isOpponentPossession`: `terminateOnOpponentPossession && ball.ownerId && owner.team === 'right'`
- `extraTermination`: handler-specific (rondo: defender possession ≥ 2.0s or ball out of area ≥ 1.5s)

Any task-level `terminateOnTurnover` must be a **semantic projection** of these existing conditions, not an independent system.

### 3.4 Controlled Player Identity
`controlledPlayerId` is an explicit string ID. It changes mid-episode when a left-team player gains possession and `teamLeftConfig.controller === 'human'`. Task-state calculations must use the authoritative ID, not role-based inference.

### 3.5 Ball Ownership
`ball.ownerId` is the authoritative possession signal. The 127-dim `rawVector` exposes team-level ownership only (`ballOwnedTeam`). Exact owner identity is available through `GameEngine.ball.ownerId` but is **not** part of the bridge binary protocol or `rawVector`.

---

## 4. Scenario Handler Capabilities

| Scenario | Handler | Dynamic State Available |
|---|---|---|
| All non-rondo | None (base engine) | `tickCount`, `score`, `stats.completedPasses`, `ball.ownerId` |
| `academy_rondo_4v1` | `RondoScenarioHandler` | `defenderPossessionTime`, `consecutivePossessionTime`, `ballOutOfAreaTime`, `lastPassCompleted`, `lastPassTeam` |

---

## 5. Gap Summary

| Capability | Status | Required Action |
|---|---|---|
| Canonical scenario task spec | Missing | Create `ScenarioTaskSpec` type |
| Constraint validation | Missing | Add schema tests |
| Deterministic task vector | Missing | Create `TaskEncoder` after spec |
| Dynamic task state | Partially missing | Use only available fields; mark others unavailable |
| Touch counting | Not available | Do not fabricate; document limitation |
| Sequence tracking | Not available | Do not fabricate; document limitation |
| Action restriction enforcement | Not available | Represent as advisory only if present |
| Spatial bounds enforcement | Partially available (rondo only) | Document actual geometry |

---

## 6. Decisions

1. **Do not add** `touchesByCurrentPlayer` as a proxy for "ticks holding possession." These are different quantities.
2. **Do not add** `currentConsecutivePasses` unless it maps to a real pass counter. The engine has `stats.completedPasses` which is episode-total, not consecutive.
3. **Do not invent** `allowedActions`, `forbiddenActions`, `dribblingAllowed`, `shootingAllowed` unless the engine actually enforces them.
4. **Do not create** a second termination system. `terminateOnTurnover` in task spec must project existing engine termination.
5. **Preserve** the 127-dim `rawVector` exactly. Add `zScenario` as a separate optional field on `RLObservation`.
6. **Do not send** `zScenario` through the existing binary bridge protocol yet.
