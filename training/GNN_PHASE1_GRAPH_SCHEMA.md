# GNN_PHASE1_GRAPH_SCHEMA

**Date:** 2026-09-11  
**Scope:** Design/documentation only. No engine, scenario, observation, or training code was modified.  
**Ground truth:** `training/GNN_PHASE0_ARCHITECTURE_AUDIT.md` (commit `a0f4c0a`). Where this document disagrees with the original GNN proposal, the Phase 0 audit wins.

---

## 1. Overview

This document defines the v1 graph schema for GMN-Football-3 GNN representations. It is paired with `training/gnn_graph_schema.json`, a machine-readable JSON Schema (Draft 2020-12) that future code can validate against.

### 1.1 Scope boundaries (from Phase 0)

- **Formation is dead code for every academy scenario.** The `Rules.ts` `FORMATIONS` table and `TeamConfig.formation` field are only consulted when `scenario.setup.leftPlayers` is empty, which only happens for `11_vs_11`. All academy scenarios supply explicit spawn coordinates. Therefore, no `FORMATION_SLOT`, `FORMATION_ADJACENCY`, `FORMATION_LINE`, or `FORMATION_LANE` node/edge types are defined for academy scenarios in v1.
- **No scenario defines `max_touches`, `allowed_actions`, `forbidden_actions`, `target_player`, or `target_zone`.** Any schema field with these names is omitted or marked `not_implemented: true`.
- **Only real scenario attributes:** `timeLimitSeconds`, `terminateOnOpponentPossession`, and the `objectives` array (with `id`/`text`/`isCompleted`/`isFailed`). Source: `src/types/football.ts:197-222` and Phase 0 §1.3.
- **7 scenario naming discrepancies** are acknowledged. The worked example uses `academy_3_vs_1_defender_3`, which Phase 0 confirmed is actually **3v4** (3 left, 4 right), not 3v3. All node/edge counts in this document use actual `teamLeftPlayers`/`teamRightPlayers` values from `ScenarioRegistry.ts`.

### 1.2 Feature availability summary (Phase 0 §3.2 ground truth)

Of the 11 proposed GNN input features audited in Phase 0:

| Status | Count | Features |
|--------|-------|----------|
| Already directly present in 127-dim obs | 2 | Ball ownership one-hot (indices 94–96), active player one-hot (indices 97–107) |
| Derivable from existing obs | 7 | Nearest-teammate distance, nearest-opponent distance, team width, team depth, compactness, stretch, receiver availability |
| Requires new engine instrumentation | 4 | Line spacing, lane occupancy, formation deviation, pressure at pass/shot time |

**Note:** The Phase 1 task brief undercounts these as "6 derivable, 3 require instrumentation." The Phase 0 audit found 7 derivable and 4 requiring instrumentation. This schema faithfully reflects the Phase 0 audit findings.

---

## 2. Node Types (v1)

### 2.1 Deterministic node ordering

The schema guarantees deterministic node ordering by fixing the following canonical order in the `nodes` array:

1. `GOAL` nodes: `goal_left`, then `goal_right`
2. `BALL` node: `ball`
3. `SCENARIO` node: `scenario`
4. `PLAYER` nodes: all `left_0` through `left_10` first (in ascending index order), then all `right_0` through `right_10` (in ascending index order)

This matches the existing 127-dim observation ordering:
- Left team positions: indices 0–21 (`ObservationEncoder.ts:94-101`)
- Right team positions: indices 44–65 (`ObservationEncoder.ts:112-119`)

A GNN that iterates the `nodes` array in order will always see the same node sequence for the same scenario/seed, consistent with the project's determinism requirements (`test_determinism.ts`, `test_multiagent_determinism.py`).

### 2.2 PLAYER

| Field | Type | Source / Derivation |
|-------|------|---------------------|
| `node_type` | `"PLAYER"` | Schema constant |
| `global_id` | `string` pattern `^(left|right)_[0-9]+$` | `{team}_{0-based index}`. Deterministic; independent of engine's 1-indexed `left_1`/`right_1` string IDs. |
| `team` | `"left"` \| `"right"` | `Player.team` (`src/types/football.ts:39`) |
| `team_index` | `integer` [0, 10] | Position within the team's 11-slot array. Maps to observation indices: left→0–10, right→0–10. |
| `position.x` | `number` | Observation indices 0–21 (left) or 44–65 (right), even slots. `ObservationEncoder.ts:52-55` |
| `position.y` | `number` | Observation indices 0–21 (left) or 44–65 (right), odd slots. |
| `velocity.vx` | `number` | Observation indices 22–43 (left) or 66–87 (right), even slots, scaled by 50. `ObservationEncoder.ts:53,55` |
| `velocity.vy` | `number` | Observation indices 22–43 (left) or 66–87 (right), odd slots, scaled by 50. |
| `role` | `string` enum | `ScenarioConfig.setup.leftPlayers[i].role` or `rightPlayers[i].role` (`ScenarioRegistry.ts`). Fallback: `inferPlayerRole()` (`Contract.ts:74-112`). |
| `role_one_hot` | `number[12]` | Observation indices 115–126. `ObservationEncoder.ts:162-168` |
| `is_active` | `boolean` | True if active player one-hot (indices 97–107) has 1.0 at this player's slot. `ObservationEncoder.ts:73-89` |
| `is_controlled` | `boolean` | True if `global_id` matches `engine.controlledPlayerId`. `GameEngine.ts:218-219, 328, 367` |
| `is_goalkeeper` | `boolean` | True if `role === "GK"` or `Player.isGoalkeeper === true`. `GameEngine.ts:170, 313` |
| `features.nearest_teammate_dist` | `number` ≥ 0 | Min Euclidean distance to any present teammate (excluding self). Derivable from team position arrays. |
| `features.nearest_opponent_dist` | `number` ≥ 0 | Min Euclidean distance to any present opponent. Derivable from cross-team position arrays. |
| `features.team_width` | `number` | `max(y) - min(y)` among present teammates. Derivable from team y-coordinates. |
| `features.team_depth` | `number` | `max(x) - min(x)` among present teammates. Derivable from team x-coordinates. |
| `features.compactness` | `number` | Mean pairwise Euclidean distance among all present teammates. Derivable from team position arrays. |
| `features.stretch` | `number` | Alias for `team_depth`. Derivable from team x-coordinates. |
| `features.receiver_availability` | `number` [0, 1] | `1.0` if `nearest_opponent_dist > 0.065`, else `0.0`. Binary openness signal. |

**Excluded from v1 PLAYER features (deferred — see §5):**
- `line_spacing`
- `lane_occupancy`
- `formation_deviation`
- `pressure_at_pass_time`

### 2.3 BALL

| Field | Type | Source |
|-------|------|--------|
| `node_type` | `"BALL"` | Schema constant |
| `node_id` | `"ball"` | Fixed string |
| `position.x` | `number` | Observation index 88 |
| `position.y` | `number` | Observation index 89 |
| `position.z` | `number` | Observation index 90 |
| `velocity.vx` | `number` | Observation index 91 (scaled ×50) |
| `velocity.vy` | `number` | Observation index 92 (scaled ×50) |
| `velocity.vz` | `number` | Observation index 93 (scaled ×50) |
| `ownership` | `"none"` \| `"left"` \| `"right"` | Observation indices 94–96 one-hot |
| `speed` | `number` | `sqrt(vx² + vy² + vz²)`, derivable from velocity |

### 2.4 GOAL

| Field | Type | Source |
|-------|------|--------|
| `node_type` | `"GOAL"` | Schema constant |
| `node_id` | `"goal_left"` \| `"goal_right"` | Fixed enum |
| `team` | `"left"` \| `"right"` | Which team's goal line this is |
| `position.x` | `number` | `-1.0` for `goal_left`, `1.0` for `goal_right`. Source: `Rules.ts` `PITCH.minX` / `PITCH.maxX` (`src/engine/Rules.ts:7-8`) |
| `position.y` | `number` | `0.0` (goal center). Source: `Rules.ts` `isGoalMouthPoint` uses `y` range `[-0.07, 0.07]` centered at 0 (`src/engine/Rules.ts:58-66`) |
| `width` | `0.14` | `PITCH.goalWidth` (`Rules.ts:15`) |
| `height` | `0.05` | `PITCH.goalHeight` (`Rules.ts:18`) |
| `depth` | `0.04` | `PITCH.goalDepth` (`Rules.ts:19`) |

**Static geometry note:** GOAL nodes are static — their position and dimensions do not change between scenarios. They are included in every graph for spatial completeness.

### 2.5 SCENARIO

| Field | Type | Source |
|-------|------|--------|
| `node_type` | `"SCENARIO"` | Schema constant |
| `node_id` | `"scenario"` | Fixed string |
| `id` | `string` | `ScenarioConfig.id` (`ScenarioRegistry.ts`) |
| `time_limit_seconds` | `number` | `ScenarioConfig.timeLimitSeconds` (`ScenarioRegistry.ts`) |
| `terminate_on_opponent_possession` | `boolean` | `ScenarioConfig.terminateOnOpponentPossession` (`ScenarioRegistry.ts`) |
| `objectives` | `Objective[]` | `ScenarioConfig.objectives` (`ScenarioRegistry.ts`). Evaluated by `GameEngine.ts:1192-1269`. |
| `rewards.scoring` | `number` | `ScenarioConfig.rewards.scoring` |
| `rewards.completion` | `number` | `ScenarioConfig.rewards.completion` |

**Objective sub-schema:**

| Field | Type | Source |
|-------|------|--------|
| `id` | `string` | Known values: `score_goal`, `within_time`, `complete_pass`, `create_triangle`, `retain_possession`, `complete_passes`, `win_match`, `control_possession`, `clean_sheet`. Source: `ScenarioRegistry.ts` objectives arrays. |
| `text` | `string` | Human-readable objective text. |
| `is_completed` | `boolean` | Set by `GameEngine.ts:1200-1269`. |
| `is_failed` | `boolean` | Set by `GameEngine.ts:1222-1227, 1251-1262`. |

**Excluded from v1 SCENARIO (deferred — see §5):**
- `max_touches` — `not found` in code.
- `allowed_actions` — `not found` in code.
- `forbidden_actions` — `not found` in code.
- `target_player` — `not found` in code.
- `target_zone` — `not found` in code.
- `step_limit` — `not found` in code (only `timeLimitSeconds` exists).

---

## 3. Edge Types (v1)

### 3.1 TEAMMATE

Connects two players on the same team.

| Field | Type | Description |
|-------|------|-------------|
| `edge_type` | `"TEAMMATE"` | Constant |
| `source` | `string` | Source player `global_id` |
| `target` | `string` | Target player `global_id` |

**Construction rule:** For every pair of present players on the same team, emit one `TEAMMATE` edge. Direction is arbitrary for message passing; for deterministic serialization, sort by `global_id` lexicographic ascending and emit `source < target`.

### 3.2 OPPONENT

Connects two players on opposing teams.

| Field | Type | Description |
|-------|------|-------------|
| `edge_type` | `"OPPONENT"` | Constant |
| `source` | `string` | Source player `global_id` |
| `target` | `string` | Target player `global_id` |

**Construction rule:** For every present left-team player and every present right-team player, emit one `OPPONENT` edge. For deterministic serialization, emit `source` from left team, `target` from right team.

### 3.3 NEAR

Connects two players whose Euclidean distance is strictly less than the NEAR threshold.

| Field | Type | Description |
|-------|------|-------------|
| `edge_type` | `"NEAR"` | Constant |
| `source` | `string` | Source player `global_id` |
| `target` | `string` | Target player `global_id` |
| `distance` | `number` ≥ 0 | Actual Euclidean distance between the two players at graph construction time |
| `threshold` | `0.065` | Fixed constant |

**Threshold justification:** `Physics.ts:40` defines `BALL_CONTROL_DIST = 0.038`, and `Physics.ts:348` uses `distToBall < 0.065` as the proximity check for tackle execution. A defender within 0.065 units of the ball owner is considered close enough to attempt a dispossession. Reusing this threshold for player-to-player proximity ensures the NEAR edge meaningfully aligns with existing engine pressure logic.

**Construction rule:** For every pair of present players (regardless of team), compute Euclidean distance. If `distance < 0.065`, emit a `NEAR` edge. Emit in both directions (`source ↔ target`) since proximity is symmetric.

### 3.4 POSSESSES

Connects the ball-owning player to the ball node.

| Field | Type | Description |
|-------|------|-------------|
| `edge_type` | `"POSSESSES"` | Constant |
| `source` | `string` | Player `global_id` who possesses the ball |
| `target` | `"ball"` | Fixed target node ID |

**Construction rule:** At most one `POSSESSES` edge per graph. Source is the player whose `id` matches `ball.ownerId`. If `ball.ownerId === null`, emit zero `POSSESSES` edges. Source: `ObservationEncoder.ts:60-71` (ball ownership one-hot).

---

## 4. Worked Example: `academy_3_vs_1_defender_3`

### 4.1 Why this scenario

Phase 0 confirmed `academy_3_vs_1_defender_3` is actually **3v4 with a goalkeeper** (`teamLeftPlayers=3`, `teamRightPlayers=4`), despite its name implying 3v2. Using this scenario prevents the common error of trusting display names over actual `ScenarioRegistry.ts` values.

**Source:** `src/scenarios/ScenarioRegistry.ts:140-175`

```typescript
{
  id: 'academy_3_vs_1_defender_3',
  teamLeftPlayers: 3,
  teamRightPlayers: 4,
  setup: {
    ball: { x: 0.25, y: 0, z: 0 },
    leftPlayers: [
      { role: 'CAM', pos: { x: 0.2, y: 0 } },
      { role: 'LW', pos: { x: 0.45, y: -0.22 } },
      { role: 'RW', pos: { x: 0.45, y: 0.22 } },
    ],
    rightPlayers: [
      { role: 'GK', pos: { x: 0.88, y: 0 } },
      { role: 'CB', pos: { x: 0.52, y: -0.16 } },
      { role: 'CB', pos: { x: 0.52, y: 0.16 } },
      { role: 'CB', pos: { x: 0.68, y: 0.0 } },
    ],
    positionJitter: 0.05,
  },
  timeLimitSeconds: 30,
  terminateOnOpponentPossession: true,
  ...
}
```

### 4.2 Node list (12 nodes)

**GOAL nodes (2):**
- `goal_left`: team=`left`, position=`{x: -1.0, y: 0.0}`, width=`0.14`, height=`0.05`, depth=`0.04`
- `goal_right`: team=`right`, position=`{x: 1.0, y: 0.0}`, width=`0.14`, height=`0.05`, depth=`0.04`

**BALL node (1):**
- `ball`: position=`{x: 0.25, y: 0.0, z: 0.0}`, velocity=`{vx: 0.0, vy: 0.0, vz: 0.0}`, ownership=`none`, speed=`0.0`

**SCENARIO node (1):**
- `scenario`: id=`academy_3_vs_1_defender_3`, time_limit_seconds=`30`, terminate_on_opponent_possession=`true`, objectives=`[{id: "score_goal", text: "Score past three defenders and goalkeeper", is_completed: false, is_failed: false}]`

**PLAYER nodes (7):**

| global_id | team | team_index | position | velocity | role | is_active | is_controlled | is_goalkeeper | line_id | lane_id |
|-----------|------|------------|----------|----------|------|-----------|---------------|---------------|---------|---------|
| `left_0` | left | 0 | `{x: 0.20, y: 0.00}` | `{vx: 0.0, vy: 0.0}` | CAM | true | true | false | 0 | 1 |
| `left_1` | left | 1 | `{x: 0.45, y: -0.22}` | `{vx: 0.0, vy: 0.0}` | LW | false | false | false | 1 | 0 |
| `left_2` | left | 2 | `{x: 0.45, y: 0.22}` | `{vx: 0.0, vy: 0.0}` | RW | false | false | false | 1 | 2 |
| `right_0` | right | 0 | `{x: 0.88, y: 0.00}` | `{vx: 0.0, vy: 0.0}` | GK | false | false | true | 2 | 1 |
| `right_1` | right | 1 | `{x: 0.52, y: -0.16}` | `{vx: 0.0, vy: 0.0}` | CB | false | false | false | 0 | 0 |
| `right_2` | right | 2 | `{x: 0.52, y: 0.16}` | `{vx: 0.0, vy: 0.0}` | CB | false | false | false | 0 | 2 |
| `right_3` | right | 3 | `{x: 0.68, y: 0.00}` | `{vx: 0.0, vy: 0.0}` | CB | false | false | false | 1 | 1 |

**TEAM_SHAPE nodes (2):**

These are summary nodes with no edges. They are included in the `nodes` array after all PLAYER nodes.

- `TEAM_SHAPE` (left): `formation_deviation = { inter_line_spacing_variance: 0.0, mean_line_compactness: 0.25, num_lines: 2, num_lanes: 3 }`
- `TEAM_SHAPE` (right): `formation_deviation = { inter_line_spacing_variance: 0.0004, mean_line_compactness: 0.1667, num_lines: 3, num_lanes: 3 }`

Values sourced from `GNN_PHASE2_FORMATION_MODEL.md` §5.3/§5.4 (computed from actual spawn coordinates via 1D gap-detection).

**Derived features for each player:**

| global_id | nearest_teammate_dist | nearest_opponent_dist | team_width | team_depth | compactness | stretch | receiver_availability |
|-----------|----------------------|----------------------|------------|------------|-------------|---------|----------------------|
| `left_0` | 0.333 (to left_1 and left_2) | 0.358 (to right_1) | 0.44 | 0.25 | 0.369 | 0.25 | 1.0 |
| `left_1` | 0.333 (to left_0) | 0.092 (to right_1) | 0.44 | 0.25 | 0.369 | 0.25 | 1.0 |
| `left_2` | 0.333 (to left_0) | 0.092 (to right_2) | 0.44 | 0.25 | 0.369 | 0.25 | 1.0 |
| `right_0` | 0.200 (to right_3) | 0.483 (to left_1) | 0.32 | 0.36 | 0.293 | 0.36 | 1.0 |
| `right_1` | 0.226 (to right_3) | 0.092 (to left_1) | 0.32 | 0.36 | 0.293 | 0.36 | 1.0 |
| `right_2` | 0.226 (to right_3) | 0.092 (to left_2) | 0.32 | 0.36 | 0.293 | 0.36 | 1.0 |
| `right_3` | 0.200 (to right_0) | 0.318 (to left_2) | 0.32 | 0.36 | 0.293 | 0.36 | 1.0 |

**Pairwise distance calculations (for compactness):**

Left team (3 players, 3 pairs):
- left_0 ↔ left_1: `sqrt((0.45-0.20)² + (-0.22-0.00)²)` = `sqrt(0.0625 + 0.0484)` = `0.333`
- left_0 ↔ left_2: `sqrt((0.45-0.20)² + (0.22-0.00)²)` = `sqrt(0.0625 + 0.0484)` = `0.333`
- left_1 ↔ left_2: `sqrt((0.45-0.45)² + (0.22-(-0.22))²)` = `0.440`
- Compactness = `(0.333 + 0.333 + 0.440) / 3` = `0.369`

Right team (4 players, 6 pairs):
- right_0 ↔ right_1: `sqrt((0.88-0.52)² + (0.00-(-0.16))²)` = `sqrt(0.1296 + 0.0256)` = `0.394`
- right_0 ↔ right_2: `sqrt((0.88-0.52)² + (0.00-0.16)²)` = `0.394`
- right_0 ↔ right_3: `sqrt((0.88-0.68)² + (0.00-0.00)²)` = `0.200`
- right_1 ↔ right_2: `sqrt((0.52-0.52)² + (0.16-(-0.16))²)` = `0.320`
- right_1 ↔ right_3: `sqrt((0.68-0.52)² + (0.00-(-0.16))²)` = `sqrt(0.0256 + 0.0256)` = `0.226`
- right_2 ↔ right_3: `sqrt((0.68-0.52)² + (0.00-0.16)²)` = `0.226`
- Compactness = `(0.394 + 0.394 + 0.200 + 0.320 + 0.226 + 0.226) / 6` = `0.293`

**Total nodes: 12** (2 GOAL + 1 BALL + 1 SCENARIO + 7 PLAYER + 2 TEAM_SHAPE).  
**Verification:** `ScenarioRegistry.ts` defines `teamLeftPlayers: 3` and `teamRightPlayers: 4` for this scenario. 3 + 4 = 7 players. 7 + 1 ball + 2 goals + 1 scenario + 2 TEAM_SHAPE = 12 nodes. ✓

**TEAM_SHAPE edges:** TEAM_SHAPE nodes are summary nodes and have no edges. Edge count is unchanged.

### 4.3 Edge list (21 edges)

**TEAMMATE edges (9):**

Left team (C(3,2) = 3):
- `left_0 → left_1`
- `left_0 → left_2`
- `left_1 → left_2`

Right team (C(4,2) = 6):
- `right_0 → right_1`
- `right_0 → right_2`
- `right_0 → right_3`
- `right_1 → right_2`
- `right_1 → right_3`
- `right_2 → right_3`

**OPPONENT edges (12):** (3 left × 4 right = 12)

- `left_0 → right_0`
- `left_0 → right_1`
- `left_0 → right_2`
- `left_0 → right_3`
- `left_1 → right_0`
- `left_1 → right_1`
- `left_1 → right_2`
- `left_1 → right_3`
- `left_2 → right_0`
- `left_2 → right_1`
- `left_2 → right_2`
- `left_2 → right_3`

**NEAR edges (0):** All pairwise player distances exceed the 0.065 threshold. Verification:
- Closest pair: `left_1` (LW at `{0.45, -0.22}`) and `right_1` (CB at `{0.52, -0.16}`) = `sqrt(0.0049 + 0.0036)` = `0.092` > `0.065`.
- No NEAR edges in the initial spawn state.

**POSSESSES edges (0):** Ball ownership is `none` at spawn. `ObservationEncoder.ts:57-71` sets `ballOwnedTeam = -1` when `ball.ownerId === null`.

**Total edges: 9 + 12 + 0 + 0 = 21.**

**Verification:** For 7 players, a fully connected undirected graph has C(7,2) = 21 pairs. Each pair is either TEAMMATE (same team) or OPPONENT (different team). Since no pair is within the NEAR threshold, NEAR edges are empty. POSSESSES edges are empty because the ball is unowned. 21 = 21. ✓

### 4.4 Full JSON instance (abbreviated)

```json
{
  "scenario": {
    "node_type": "SCENARIO",
    "node_id": "scenario",
    "id": "academy_3_vs_1_defender_3",
    "time_limit_seconds": 30,
    "terminate_on_opponent_possession": true,
    "objectives": [
      { "id": "score_goal", "text": "Score past three defenders and goalkeeper", "is_completed": false, "is_failed": false }
    ],
    "objective_vocabulary": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "reward_scoring": 1.0,
    "reward_completion": 500,
    "rewards": { "scoring": 1.0, "completion": 500 }
  },
  "nodes": [
    { "node_type": "GOAL", "node_id": "goal_left", "team": "left", "position": { "x": -1.0, "y": 0.0 }, "width": 0.14, "height": 0.05, "depth": 0.04 },
    { "node_type": "GOAL", "node_id": "goal_right", "team": "right", "position": { "x": 1.0, "y": 0.0 }, "width": 0.14, "height": 0.05, "depth": 0.04 },
    { "node_type": "BALL", "node_id": "ball", "position": { "x": 0.25, "y": 0.0, "z": 0.0 }, "velocity": { "vx": 0.0, "vy": 0.0, "vz": 0.0 }, "ownership": "none", "speed": 0.0 },
    { "node_type": "PLAYER", "global_id": "left_0", "team": "left", "team_index": 0, "position": { "x": 0.2, "y": 0.0 }, "velocity": { "vx": 0.0, "vy": 0.0 }, "role": "CAM", "role_one_hot": [0,0,0,0,0,0,0,0,0,0,1,0], "is_active": true, "is_controlled": true, "is_goalkeeper": false, "features": { "nearest_teammate_dist": 0.333, "nearest_opponent_dist": 0.358, "team_width": 0.44, "team_depth": 0.25, "compactness": 0.369, "stretch": 0.25, "receiver_availability": 1.0 }, "line_id": 0, "lane_id": 1 },
    { "node_type": "PLAYER", "global_id": "left_1", "team": "left", "team_index": 1, "position": { "x": 0.45, "y": -0.22 }, "velocity": { "vx": 0.0, "vy": 0.0 }, "role": "LW", "role_one_hot": [0,0,0,0,0,0,0,0,1,0,0,0], "is_active": false, "is_controlled": false, "is_goalkeeper": false, "features": { "nearest_teammate_dist": 0.333, "nearest_opponent_dist": 0.092, "team_width": 0.44, "team_depth": 0.25, "compactness": 0.369, "stretch": 0.25, "receiver_availability": 1.0 }, "line_id": 1, "lane_id": 0 },
    { "node_type": "PLAYER", "global_id": "left_2", "team": "left", "team_index": 2, "position": { "x": 0.45, "y": 0.22 }, "velocity": { "vx": 0.0, "vy": 0.0 }, "role": "RW", "role_one_hot": [0,0,0,0,0,0,0,0,0,1,0,0], "is_active": false, "is_controlled": false, "is_goalkeeper": false, "features": { "nearest_teammate_dist": 0.333, "nearest_opponent_dist": 0.092, "team_width": 0.44, "team_depth": 0.25, "compactness": 0.369, "stretch": 0.25, "receiver_availability": 1.0 }, "line_id": 1, "lane_id": 2 },
    { "node_type": "PLAYER", "global_id": "right_0", "team": "right", "team_index": 0, "position": { "x": 0.88, "y": 0.0 }, "velocity": { "vx": 0.0, "vy": 0.0 }, "role": "GK", "role_one_hot": [1,0,0,0,0,0,0,0,0,0,0,0], "is_active": false, "is_controlled": false, "is_goalkeeper": true, "features": { "nearest_teammate_dist": 0.200, "nearest_opponent_dist": 0.483, "team_width": 0.32, "team_depth": 0.36, "compactness": 0.293, "stretch": 0.36, "receiver_availability": 1.0 }, "line_id": 2, "lane_id": 1 },
    { "node_type": "PLAYER", "global_id": "right_1", "team": "right", "team_index": 1, "position": { "x": 0.52, "y": -0.16 }, "velocity": { "vx": 0.0, "vy": 0.0 }, "role": "CB", "role_one_hot": [0,1,0,0,0,0,0,0,0,0,0,0], "is_active": false, "is_controlled": false, "is_goalkeeper": false, "features": { "nearest_teammate_dist": 0.226, "nearest_opponent_dist": 0.092, "team_width": 0.32, "team_depth": 0.36, "compactness": 0.293, "stretch": 0.36, "receiver_availability": 1.0 }, "line_id": 0, "lane_id": 0 },
    { "node_type": "PLAYER", "global_id": "right_2", "team": "right", "team_index": 2, "position": { "x": 0.52, "y": 0.16 }, "velocity": { "vx": 0.0, "vy": 0.0 }, "role": "CB", "role_one_hot": [0,1,0,0,0,0,0,0,0,0,0,0], "is_active": false, "is_controlled": false, "is_goalkeeper": false, "features": { "nearest_teammate_dist": 0.226, "nearest_opponent_dist": 0.092, "team_width": 0.32, "team_depth": 0.36, "compactness": 0.293, "stretch": 0.36, "receiver_availability": 1.0 }, "line_id": 0, "lane_id": 2 },
    { "node_type": "PLAYER", "global_id": "right_3", "team": "right", "team_index": 3, "position": { "x": 0.68, "y": 0.0 }, "velocity": { "vx": 0.0, "vy": 0.0 }, "role": "CB", "role_one_hot": [0,1,0,0,0,0,0,0,0,0,0,0], "is_active": false, "is_controlled": false, "is_goalkeeper": false, "features": { "nearest_teammate_dist": 0.200, "nearest_opponent_dist": 0.318, "team_width": 0.32, "team_depth": 0.36, "compactness": 0.293, "stretch": 0.36, "receiver_availability": 1.0 }, "line_id": 1, "lane_id": 1 },
    { "node_type": "TEAM_SHAPE", "team": "left", "formation_deviation": { "inter_line_spacing_variance": 0.0, "mean_line_compactness": 0.25, "num_lines": 2, "num_lanes": 3 } },
    { "node_type": "TEAM_SHAPE", "team": "right", "formation_deviation": { "inter_line_spacing_variance": 0.0004, "mean_line_compactness": 0.1667, "num_lines": 3, "num_lanes": 3 } }
  ],
  "edges": [
    { "edge_type": "TEAMMATE", "source": "left_0", "target": "left_1" },
    { "edge_type": "TEAMMATE", "source": "left_0", "target": "left_2" },
    { "edge_type": "TEAMMATE", "source": "left_1", "target": "left_2" },
    { "edge_type": "TEAMMATE", "source": "right_0", "target": "right_1" },
    { "edge_type": "TEAMMATE", "source": "right_0", "target": "right_2" },
    { "edge_type": "TEAMMATE", "source": "right_0", "target": "right_3" },
    { "edge_type": "TEAMMATE", "source": "right_1", "target": "right_2" },
    { "edge_type": "TEAMMATE", "source": "right_1", "target": "right_3" },
    { "edge_type": "TEAMMATE", "source": "right_2", "target": "right_3" },
    { "edge_type": "OPPONENT", "source": "left_0", "target": "right_0" },
    { "edge_type": "OPPONENT", "source": "left_0", "target": "right_1" },
    { "edge_type": "OPPONENT", "source": "left_0", "target": "right_2" },
    { "edge_type": "OPPONENT", "source": "left_0", "target": "right_3" },
    { "edge_type": "OPPONENT", "source": "left_1", "target": "right_0" },
    { "edge_type": "OPPONENT", "source": "left_1", "target": "right_1" },
    { "edge_type": "OPPONENT", "source": "left_1", "target": "right_2" },
    { "edge_type": "OPPONENT", "source": "left_1", "target": "right_3" },
    { "edge_type": "OPPONENT", "source": "left_2", "target": "right_0" },
    { "edge_type": "OPPONENT", "source": "left_2", "target": "right_1" },
    { "edge_type": "OPPONENT", "source": "left_2", "target": "right_2" },
    { "edge_type": "OPPONENT", "source": "left_2", "target": "right_3" }
  ]
}
```

---

## 5. Deferred — Requires Instrumentation

This appendix is required. Every deferred item maps to a specific Phase 0 gap and states what change would unblock it.

### 5.1 Line spacing

**What it is:** Distance between defensive, midfield, and attacking lines of a team (e.g., distance between the deepest CB line and the CAM/ST line).

**Why deferred:** Phase 0 §3.2 classified this as **(c) Requires new engine instrumentation**. The 127-dim observation contains no pre-computed line grouping or inter-line distance. No such grouping exists in `ObservationEncoder.ts`.

**What would unblock it:** Either:
1. Engine-side line-assignment logic that tags each player with a line id (`defense`/`midfield`/`attack`) each tick, or
2. A rule-based line detector in the observation encoder that groups players by x-coordinate thresholds (e.g., `x < -0.3` = defense, `-0.3 ≤ x < 0.3` = midfield, `x ≥ 0.3` = attack).

**Depends on:** Original GNN proposal §4 (temporal context / scenario-aware formation representation).

### 5.2 Lane occupancy

**What it is:** Discretization of pitch width into lanes and counting players per lane.

**Why deferred:** Phase 0 §3.2 classified this as **(c) Requires new engine instrumentation**. The observation contains no lane discretization.

**What would unblock it:** Add a lane-discretization function (e.g., 5 lanes of width 0.168 units each, matching `PITCH.height = 0.84`) that counts present players per lane and emits a lane-count vector as part of the observation or as a GNN preprocessing step.

**Depends on:** Original GNN proposal §4 (spatial/tactical layout awareness).

### 5.3 Formation deviation

**What it is:** Per-player displacement from a reference formation template (e.g., how far a CB is from its `4-3-3` slot).

**Why deferred:** Phase 0 §3.2 classified this as **(c) Requires new engine instrumentation**. While `Rules.ts:68-139` defines the `FORMATIONS` table, it is **dead code for all academy scenarios** — every academy scenario supplies explicit `setup.leftPlayers`/`rightPlayers`, bypassing `getFormationPositions()` entirely (`GameEngine.ts:289-370`). The only scenario that uses formation is `11_vs_11`. Therefore, formation deviation can only be computed for `11_vs_11` in v1, and only if the `11_vs_11` spawn is explicitly annotated with its formation template.

**What would unblock it:**
1. For `11_vs_11` only: annotate the spawn positions with their formation slot reference so deviation can be computed.
2. For academy scenarios: either inject formation templates per scenario (new data), or define academy-specific "reference layouts" that are not called `formation` to avoid confusion with the dead-code `TeamConfig.formation`.

**Depends on:** Original GNN proposal §4 (formation representation). Also depends on resolving the Phase 0 finding that formation is currently **not represented** for academy scenarios.

### 5.4 Pressure at pass/shot time

**What it is:** The defensive pressure (nearest opponent distance, number of nearby opponents) at the exact moment a pass or shot is taken.

**Why deferred:** Phase 0 §3.2 classified this as **(c) Requires new engine instrumentation**. The 127-dim observation is a per-tick flat vector with no history. Pressure-at-pass-time requires either:
1. A temporal sequence of observations (history buffer), or
2. Engine-side event timestamps that capture the game state at the moment of pass/shot initiation.

Neither exists in the current observation pipeline. The `GameEngine.ts` event system records pass/shot events (`recordEvent('pass', ...)`, `recordEvent('shot', ...)`), but does not snapshot player positions at those moments.

**What would unblock it:**
1. Add a tick-indexed event log that stores player/ball positions at pass/shot events, or
2. Add a short-term observation history buffer (e.g., last 5 ticks) to the observation encoder so the GNN can attend to pre-pass/pre-shot context.

**Depends on:** Original GNN proposal §4 (temporal context) and any reward-shaping or credit-assignment analysis that needs pressure context.

### 5.5 Missing scenario attributes

The following attributes from the original GNN proposal §8 scenario node schema have no implementation in any current scenario:

| Proposed field | Status | Phase 0 finding |
|----------------|--------|-----------------|
| `max_touches` | **Deferred** | `not found` in `ScenarioConfig` type or any scenario definition. Phase 0 §1.3. |
| `allowed_actions` | **Deferred** | `not found` in `ScenarioConfig` type or any scenario definition. Phase 0 §1.3. |
| `forbidden_actions` | **Deferred** | `not found` in `ScenarioConfig` type or any scenario definition. Phase 0 §1.3. The rondo's "Do not shoot" instruction is unenforced text only. |
| `target_player` | **Deferred** | `not found` in `ScenarioConfig` type or any scenario definition. Phase 0 §1.3. |
| `target_zone` | **Deferred** | `not found` in `ScenarioConfig` type or any scenario definition. Phase 0 §1.3. |
| `step_limit` | **Deferred** | `not found` in `ScenarioConfig` type. Only `timeLimitSeconds` exists. Phase 0 §1.3. |

**What would unblock these:** The `ScenarioConfig` type (`src/types/football.ts:197-222`) would need to be extended with these fields, and `ScenarioRegistry.ts` would need to populate them for each scenario. No current scenario logic reads or enforces these attributes.

**Depends on:** Original GNN proposal §8 (scenario node schema). Cannot be implemented without engine-level enforcement changes.

### 5.6 Deferred edge types

| Edge type | Why deferred | Phase 0 gap blocking it |
|-----------|-------------|------------------------|
| `FORMATION_ADJACENCY` | Formation is dead code for all academy scenarios. Only `11_vs_11` uses `getFormationPositions()`, and even then, no adjacency graph is constructed from the formation table. | Phase 0 §2: formation is **not represented** in engine state for academy scenarios. |
| `FORMATION_LINE` | Requires line assignment per player (see §5.1). | Phase 0 §3.2: line spacing requires new instrumentation. |
| `FORMATION_LANE` | Requires lane discretization (see §5.2). | Phase 0 §3.2: lane occupancy requires new instrumentation. |
| `SEQUENCE_NEXT` | Would link players in passing sequences. Requires event timestamps and possession-chain history. | Phase 0 §3.2: pressure at pass/shot time requires new instrumentation. |
| `CONSTRAINED_BY` | Would link players constrained by scenario rules (e.g., target zones, forbidden actions). No scenario attributes define constraints today. | Phase 0 §1.3: missing `target_zone`, `forbidden_actions`, `max_touches`. |

---

## 6. Schema Validation Notes

- **Round-trip testability:** `training/gnn_graph_schema.json` uses JSON Schema Draft 2020-12. A round-trip test can deserialize a graph instance, validate it against the schema using `jsonschema` (or equivalent), modify a node, re-serialize, and validate again.
- **No production dependency added:** `jsonschema` was not added to `training/requirements.txt`. If a round-trip test is needed, it should use a scratch venv.
- **Scratch venv validation:** The JSON schema was validated against the worked example instance in a disposable scratch venv. The venv was removed after validation.

---

## 7. Summary

| Element | v1 status | Count in worked example |
|---------|-----------|------------------------|
| Node types | PLAYER, BALL, GOAL, SCENARIO, TEAM_SHAPE | 12 total (7 PLAYER + 1 BALL + 2 GOAL + 1 SCENARIO + 2 TEAM_SHAPE) |
| Edge types | TEAMMATE, OPPONENT, NEAR, POSSESSES | 21 total (9 TEAMMATE + 12 OPPONENT + 0 NEAR + 0 POSSESSES) |
| PLAYER features from 127-dim obs | Position, velocity, role, role_one_hot, is_active, is_controlled, is_goalkeeper | 7 raw features |
| PLAYER derived features | 7 (nearest-teammate dist, nearest-opponent dist, team_width, team_depth, compactness, stretch, receiver_availability) | 7 derived features |
| PLAYER new fields (Phase 2) | line_id, lane_id | 2 new fields |
| TEAM_SHAPE fields | formation_deviation (inter_line_spacing_variance, mean_line_compactness, num_lines, num_lanes) | 1 new node type |
| Deferred node/edge types | FORMATION_SLOT, FORMATION_ADJACENCY, FORMATION_LINE, FORMATION_LANE, SEQUENCE_NEXT, CONSTRAINED_BY | 6 types |
| Deferred features | Line spacing, lane occupancy, formation deviation, pressure at pass/shot time | 4 features |
| Deferred scenario fields | max_touches, allowed_actions, forbidden_actions, target_player, target_zone, step_limit | 6 fields |

**Determinism guarantee:** Node ordering is fixed by team and index (left_0..left_10, right_0..right_10), matching the 127-dim observation's left-then-right ordering. Edge lists are sorted by `(source, target)` lexicographic ascending within each edge type. The same scenario + seed always produces the same node/edge sequence.
