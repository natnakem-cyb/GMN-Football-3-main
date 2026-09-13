# GNN Phase 3 Closure Audit

## 1. Executive Summary

This document records the closure audit of GNN Phase 3 following the completion of P0 fixes to the graph builder. All four P0 work items are closed. The graph builder now derives its core identity signals (ball ownership, controlled player, formation assignment) exclusively from authoritative engine state delivered via the bridge (`info` records), not from scenario setup or heuristic fallbacks. Six new tests cover the fixed behaviors. All 30 graph builder tests pass. TypeScript type-check (`tsc --noEmit`) passes cleanly. Bridge protocol tests pass. Full Vite build hits OOM on the audit machine and was not re-run; this does not affect closure because type-check and targeted tests cover the changed surface.

## 2. P0 Closure Status

| ID | Title | Status | Evidence |
|----|-------|--------|----------|
| P0-A | Exact ball owner from engine | Closed | `build_graph()` reads `info.ground_truth.current_ball_owner.agent_id`; nearest-player heuristic removed; `exact_owner_id` is `None` when unavailable and no `POSSESSES` edge is created. |
| P0-B | Controlled player from engine | Closed | `PLAYER.is_controlled` is set from `info.controlledPlayerId`; scenario-based `is_controlled` logic removed. |
| P0-C | Formation assignment role-based | Closed | `_build_assigned_to_edges` matches players to slots by role with deterministic `(x, y)` tie-breaking; nearest-slot heuristic removed; nearest-slot fallback only when no role match exists. |
| P0-D | Formation and edge direction docs | Closed | `docs/GNN_PHASE3_FORMATION_IDENTITY_SPEC.md` and `docs/GNN_PHASE3_EDGE_DIRECTION_SPEC.md` produced. |

## 3. What Was Fixed

### P0-A: Exact Ball Owner

**Before:** `build_graph()` inferred ball ownership by selecting the nearest player to the ball. This produced incorrect `POSSESSES` edges whenever the nearest player was not the true owner, especially during passes, shots, and tackles.

**After:** `build_graph()` reads `info.ground_truth.current_ball_owner.agent_id`. When the engine reports no owner (`None`), `exact_owner_id` is set to `None` and no `POSSESSES` edge is created. The nearest-player path was removed entirely.

### P0-B: Controlled Player

**Before:** `is_controlled` was determined from scenario setup data, which could lag or disagree with the live engine state.

**After:** `is_controlled` is set per player node from `info.controlledPlayerId` on every step. The scenario-setup path was removed. `PLAYER.is_controlled` now reflects the current engine state.

### P0-C: Formation Assignment

**Before:** Players were assigned to formation slots by nearest-slot heuristic, ignoring declared roles. This could mismatch a player to a slot that did not carry the player's role.

**After:** `_build_assigned_to_edges` matches each player to a slot whose `role` matches the player's role. Tie-breaking within matching slots is deterministic by `(x, y)`. A nearest-slot fallback is used only when no slot matches the player's role. The nearest-slot primary path was removed.

### P0-D: Documentation

`docs/GNN_PHASE3_FORMATION_IDENTITY_SPEC.md` documents the formation identity model, slot role definitions, and assignment rules. `docs/GNN_PHASE3_EDGE_DIRECTION_SPEC.md` documents the undirected edge semantics for `TEAMMATE`, `OPPONENT`, `NEAR`, and `POSSESSES`.

## 4. What Was Deliberately Left Unavailable

- **No role-match fallback enhancement:** When no formation slot matches a player's role, the fallback is the nearest slot. This is intentional and documented in `docs/GNN_PHASE3_FORMATION_IDENTITY_SPEC.md`. Attempting to match by proximity first would reintroduce the old heuristic and obscure role semantics.
- **No owner inference on missing data:** When `info.ground_truth.current_ball_owner.agent_id` is unavailable, no `POSSESSES` edge is created. The graph does not attempt to infer ownership from proximity in this state. Inference can be re-evaluated in a later phase if the engine consistently omits owner data in specific scenarios.
- **No Vite build re-run:** The full Vite build was not re-run on the audit machine due to OOM. Closure relies on `tsc --noEmit` and targeted test suites, which cover the changed surface.

## 5. What Is Authoritative

These inputs are treated as ground truth and are not derived or inferred by the graph builder:

- `info.ground_truth.current_ball_owner.agent_id` — exact ball owner for `POSSESSES` edges.
- `info.controlledPlayerId` — controlled player for `PLAYER.is_controlled`.
- `ScenarioRegistry` slot `role` definitions — used for role-based formation assignment in `_build_assigned_to_edges`.
- `ScenarioRegistry` slot positions and team counts — used for graph topology.

## 6. What Remains Derived

These values are computed by the graph builder from authoritative inputs:

- Player node positions and velocities from `getObservation()`.
- `NEAR` edges from player-to-player and player-to-ball distance thresholds.
- `TEAMMATE` and `OPPONENT` edges from team assignment.
- `ASSIGNED_TO` edges from role matching plus deterministic tie-breaking.
- Edge feature geometry (relative position, distance) from node positions.
- Ball `ownerId` bridge mapping from `info.ground_truth.current_ball_owner.agent_id` to engine ball reference.

## 7. Exact Graph Input Contract for Phase 4

Phase 4 work must assume the following contract from the graph builder output:

- **Nodes:** `PLAYER` nodes carry `team` (0 or 1), `role` (from scenario), `is_controlled` (from `info.controlledPlayerId`), position, velocity. `BALL` node carries position, velocity, and optional owner reference.
- **Edges:**
  - `TEAMMATE`: undirected, between players of the same team.
  - `OPPONENT`: undirected, between players of opposing teams.
  - `NEAR`: undirected, between players and ball, or players and players, within distance threshold.
  - `POSSESSES`: directed from `current_ball_owner.agent_id` to `BALL`. Absent when owner is `None`.
  - `ASSIGNED_TO`: directed from player to formation slot; role-matched with deterministic `(x, y)` tie-break; nearest-slot fallback only when no role match.
- **Determinism:** Node ordering is fixed by team and index. Edges are sorted lexicographically within type. Observation ordering matches node ordering.
- **Feature dimensions:** Node features are 127-dimensional as specified in `docs/GNN_PHASE3_GRAPH_FEATURE_DICTIONARY.md`.

## 8. Exact zScenario Contract for Phase 4

- **Source:** `zScenario` is delivered in `step()` observations via `this.getObservation()` → `ObservationEncoder.encode` on the normal agent transition path.
- **Dimensions:** Fixed per the task vector spec; dynamic values change across ticks.
- **Usage in graph builder:** `zScenario` is included in observation vectors and does not affect graph topology directly.
- **Regression coverage:** `training/test_task_vector.ts` verifies `step()` observation includes `zScenario` with correct dimensions and dynamic changes across ticks.

## 9. Test Results

| Suite | Result |
|-------|--------|
| Graph builder tests (30 tests) | Pass |
| TypeScript type-check (`tsc --noEmit`) | Pass |
| Standard tests | Pass |
| Bridge protocol tests | Pass |
| Full Vite build | Not re-run (OOM on audit machine) |

### Tests Added for P0 Fixes

- `test_exact_owner_overrides_nearest_player` — verifies `POSSESSES` source matches `info` ground truth.
- `test_controlled_player_from_info` — verifies `is_controlled` matches `info.controlledPlayerId`.
- `test_formation_assignment_role_based_not_nearest` — verifies `ASSIGNED_TO` uses matching role.
- `test_teammate_reverse_edge_geometry` — verifies undirected `TEAMMATE` edge geometry is well-formed.
- `test_opponent_reverse_edge_geometry` — verifies undirected `OPPONENT` edge geometry is well-formed.
- `test_possesses_none_creates_no_edge` — verifies no false `POSSESSES` when ownership is `none`.

## 10. Remaining Technical Debt

- **Vite build OOM:** Full production build should be re-run on a machine with sufficient memory to confirm no regressions in bundle output.
- **Role-match fallback edge case:** When no formation slot matches a player's role, the nearest-slot fallback is a known degenerate path. If scenario data ever omits roles for slots, this path will be exercised silently. Monitor in training.
- **Owner-unavailable state:** When `info.ground_truth.current_ball_owner.agent_id` is `None`, the graph has no `POSSESSES` edge. Downstream consumers must handle the missing edge without error. This is intentional but should be validated in Phase 4 GNN training.
- **Bridge protocol coverage:** Bridge protocol tests cover the P0 fields. Additional fields exposed by the bridge (if any) are not covered by this audit.

## 11. Phase 3 Closure Decision

Phase 3 is closed. The graph builder produces a deterministic, engine-authoritative graph with documented node and edge contracts. The four P0 fixes eliminate the principal sources of incorrect identity and assignment in the graph. Phase 4 may proceed against the contracts documented in Sections 7 and 8.
