# D-Obs Stage 1: Pipeline Trace

## 1. Trace path (source-verified)

```
GameEngine state (players, ball, gameMode, score, tickCount)
    ↓
ObservationEncoder.encode(players, ball, viewpointPlayerId, score, stepCount, maxSteps, gameMode)
    ↓
RLObservation {
  leftTeamPositions, leftTeamVelocities,
  rightTeamPositions, rightTeamVelocities,
  ballPosition, ballVelocity,
  ballOwnedTeam, ballOwnedPlayer,
  activePlayerIndex, gameMode, score, stepsLeft,
  rawVector: number[127]
}
    ↓
TaskEncoder.encode(activeScenario, getScenarioDynamicState())
    ↓
zScenario: Float32Array(8)
    ↓
GameEngine.getObservation() returns { ...obs, zScenario }
    ↓
bridge_server.ts: ObservationEncoder.encode(...).rawVector  (127-float array)
    ↓
Binary serialization: encodeStepBinary / encodeMultiStepBinary / encodeBatchedBinary
    ↓
WebSocket → gmn_pettingzoo.py
    ↓
np.frombuffer(..., dtype="<f4", count=OBSERVATION_DIM)  → np.ndarray shape (127,)
    ↓
PettingZoo obs dict: {"observation": np.ndarray, "action_mask": np.ndarray}
    ↓
MAPPO input: local_obs shape (num_steps, num_agents, 127)
```

## 2. Source files

| Stage | File | Key function/line |
|-------|------|-------------------|
| Engine state | `src/engine/GameEngine.ts` | `players`, `ball`, `gameMode`, `score`, `tickCount` |
| Observation encoding | `src/engine/ObservationEncoder.ts` | `ObservationEncoder.encode()` lines 40-192 |
| Task encoding | `src/engine/TaskEncoder.ts` | `TaskEncoder.encode()` lines 6-47 |
| Observation assembly | `src/engine/GameEngine.ts` | `getObservation()` lines 257-271 |
| Bridge serialization | `training/bridge_server.ts` | `encodeStepBinary` line 953, `encodeMultiStepBinary` line 1006, `encodeBatchedBinary` line ~1060 |
| Python deserialization | `training/gmn_pettingzoo.py` | `_decode_step_binary` / `_decode_multi_step_binary` / `_decode_batch_binary` |
| MAPPO consumption | `training/mappo_rollout.py` | `collect_rollout_batched()` |
| Contract constants | `src/engine/Contract.ts` | `OBSERVATION_DIM=127`, `OBSERVATION_SCHEMA_VERSION='simple115_v3_role'` |

## 3. Observation contract

| Property | Value |
|----------|-------|
| Schema version | `simple115_v3_role` |
| Dimension | 127 |
| Base payload | 115 floats (`BASE_OBSERVATION_DIM`) |
| Role slice | 12 floats (indices 115-126) |
| Coordinate system | Pitch units: x ∈ [-1.0, 1.0], y ∈ [-0.42, 0.42] |
| Velocity scaling | `velocity.x * 50`, `velocity.y * 50` (pitch units/tick × 50) |
| Inactive player slots | `-1.0, -1.0` for position; `-1.0, -1.0` for velocity |
| Ownership encoding | One-hot `[no-one, left, right]` at indices 94-96 |
| Active player encoding | One-hot over 11 players at indices 97-107 |
| Game mode encoding | One-hot `[Normal, KickOff, GoalKick, FreeKick, Corner, ThrowIn, Penalty]` at indices 108-114 |
| Role encoding | One-hot over `ROLE_VOCABULARY` at indices 115-126 |
| zScenario | **NOT transmitted to Python** — produced by `TaskEncoder.encode()` but only present in `GameEngine.getObservation()` return value, never serialized in bridge binary frames or JSON responses |
| Normalization | None beyond velocity scaling × 50 |
| Padding | Inactive player slots padded with `-1.0` |
| Derived fields | `stepsLeft = max(0, maxSteps - stepCount)` at index not separately stored (in `RLObservation.stepsLeft` but not in `rawVector`) |
| History / temporal stack | **None** — MAPPO receives only current-state 127-float vector per step |

## 4. Critical finding: zScenario gap

`TaskEncoder.encode()` produces an 8-dimensional task vector (`zScenario`) that is attached to `RLObservation` in `GameEngine.getObservation()` (line 269). However, the bridge serialization (`bridge_server.ts`) only transmits `rawVector` (127 floats). The `zScenario` is **not included** in any binary frame format (`encodeStepBinary`, `encodeMultiStepBinary`, `encodeBatchedBinary`).

This means:
- The Python `gmn_pettingzoo.py` never receives `zScenario`.
- MAPPO policies have no access to task/scenario context beyond what is implicitly encoded in the 127-dim observation.
- The `TaskEncoder` output is effectively dead code from the training perspective.

This is a **known gap** documented in `docs/GNN_PHASE3_CLOSURE_AUDIT.md` line 40 and `docs/GNN_PHASE3_GRAPH_FEATURE_DICTIONARY.md` line 398.

## 5. Observation layout (authoritative)

| Offset | Length | Content | Notes |
|--------|--------|---------|-------|
| 0 | 22 | Left team (x, y) positions, 11 players | Inactive: `-1.0, -1.0` |
| 22 | 22 | Left team (x, y) velocity × 50 | Inactive: `-1.0, -1.0` |
| 44 | 22 | Right team (x, y) positions, 11 players | Inactive: `-1.0, -1.0` |
| 66 | 22 | Right team (x, y) velocity × 50 | Inactive: `-1.0, -1.0` |
| 88 | 3 | Ball (x, y, z) position | Raw pitch coordinates |
| 91 | 3 | Ball (x, y, z) velocity × 50 | Pre-scaled by engine |
| 94 | 3 | Ball ownership one-hot: `[no-one, left, right]` | `ballOwnedTeam` from engine |
| 97 | 11 | Active/controlled player one-hot | 11 players |
| 108 | 7 | Game mode one-hot | `[Normal, KickOff, GoalKick, FreeKick, Corner, ThrowIn, Penalty]` |
| 115 | 12 | Agent role one-hot | `ROLE_VOCABULARY` order |
| **Total** | **127** | | |

## 6. What the observation does NOT contain

| Variable | Status |
|----------|--------|
| `zScenario` task vector | NOT transmitted to Python |
| Exact ball owner player index | NOT in rawVector (only team-level one-hot) |
| Formation/line assignment | NOT in observation |
| Pass/receive intent | NOT in observation |
| Shot clock / time remaining | NOT in rawVector (`stepsLeft` is in `RLObservation` but not serialized) |
| Episode progress | NOT directly (game mode encodes some state) |
| Historical ball position | NOT (no temporal stack) |
| Teammate/opponent roles | NOT (only self role at indices 115-126) |
| Intercepted/pass-target flags | NOT |

## 7. Source-of-truth precedence

1. `src/engine/Contract.ts` — schema version, dimensions, role vocabulary
2. `src/engine/ObservationEncoder.ts` — exact encoding logic and offsets
3. `src/engine/TaskEncoder.ts` — task vector construction
4. `src/engine/GameEngine.ts` — `getObservation()` assembly
5. `training/bridge_server.ts` — serialization to binary/JSON
6. `training/gmn_pettingzoo.py` — deserialization and obs space definition
7. `training/mappo_rollout.py` — buffer construction

Documentation/README is secondary to the above source files.
