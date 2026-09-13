# GNN Phase 3 Graph Feature Dictionary

## 1. Overview

This document defines the complete feature dictionary for the GMN-Football-3 Phase 3 GNN graph representation. The graph is constructed by `training/gnn_graph_builder.py` and validated against `training/gnn_graph_schema.json` (v3).

A single graph instance represents one timestep of a football match and contains:

- **Nodes**: PLAYER, BALL, GOAL, TEAM_SHAPE, FORMATION_SLOT, SCENARIO
- **Edges**: TEAMMATE, OPPONENT, NEAR, POSSESSES, PLAYER_BALL, PLAYER_GOAL, BALL_GOAL, ASSIGNED_TO, BELONGS_TO_SHAPE, SCENARIO_CONTEXT, FORMATION_ADJACENCY, FORMATION_LINE, FORMATION_LANE
- **Graph-level context**: `z_scenario` (Float32Array[8])

Node ordering is deterministic: left team players (indices 0..left_count-1), right team players (indices 0..right_count-1), BALL, GOAL nodes, TEAM_SHAPE nodes, FORMATION_SLOT nodes. Edges are sorted lexicographically by `(edge_type, source, target)`.

---

## 2. Node Feature Dictionary

### 2.1 PLAYER

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `node_type` | string | `"PLAYER"` | — | Constant | None | Never missing |
| `global_id` | string | `"left_{i}"` or `"right_{i}"` | — | Deterministic: `{team}_{0-based index}` | None | Never missing; matches 127-dim observation ordering (left indices 0-10, right indices 0-10) |
| `team` | string | `"left"` \| `"right"` | — | Scenario config + observation ordering | None | Never missing |
| `team_index` | int | `0` .. `10` | — | Index within team array | None | Never missing |
| `position.x` | float | `[-1.0, 1.0]` | Pitch units | ObservationEncoder.ts indices 0-21 (left) or 44-65 (right) | Raw pitch coordinates | Sentinel `-1.0` indicates absent player slot (e.g., 3v1 scenario uses only first 3 of 11 slots) |
| `position.y` | float | `[-0.42, 0.42]` | Pitch units | ObservationEncoder.ts indices 0-21 (left) or 44-65 (right) | Raw pitch coordinates | Sentinel `-1.0` indicates absent player slot |
| `velocity.vx` | float | Scaled by 50 | Pitch units / tick | ObservationEncoder.ts indices 22-43 (left) or 66-87 (right) | Pre-scaled by engine (÷50 to recover raw) | `0.0` for absent slots |
| `velocity.vy` | float | Scaled by 50 | Pitch units / tick | ObservationEncoder.ts indices 22-43 (left) or 66-87 (right) | Pre-scaled by engine (÷50 to recover raw) | `0.0` for absent slots |
| `role` | string | ROLE_VOCABULARY (12 values) | — | ScenarioConfig.setup.leftPlayers/rightPlayers role field; fallback `inferPlayerRole()` | None | `"UNKNOWN"` if role index exceeds scenario setup length |
| `role_one_hot` | float[12] | One-hot over ROLE_VOCABULARY | — | Derived from `role` via ROLE_TO_INDEX mapping | None | Zero vector for `"UNKNOWN"` role |
| `is_active` | bool | `true` \| `false` | — | ObservationEncoder.ts indices 97-107 active player one-hot | None | `false` when active index is ambiguous (both teams have same role) — defaults to left team |
| `is_controlled` | bool | `true` \| `false` | — | `info.controlledPlayerId` from engine bridge | None | `false` for all right-team players and left-team non-controlled players |
| `is_goalkeeper` | bool | `true` \| `false` | — | `role == "GK"` | None | `false` for non-GK roles |
| `features.nearest_teammate_dist` | float | `>= 0` | Pitch units (Euclidean) | Computed from teammate position arrays | Raw Euclidean distance | `0.0` for absent slots or teams with < 2 valid players |
| `features.nearest_opponent_dist` | float | `>= 0` | Pitch units (Euclidean) | Computed from cross-team position arrays | Raw Euclidean distance | `0.0` for absent slots or when no opponents are present |
| `features.team_width` | float | `>= 0` | Pitch units | `max(y) - min(y)` among present teammates | Raw | `0.0` for absent slots or teams with < 2 valid players |
| `features.team_depth` | float | `>= 0` | Pitch units | `max(x) - min(x)` among present teammates | Raw | `0.0` for absent slots or teams with < 2 valid players |
| `features.compactness` | float | `>= 0` | Pitch units (Euclidean) | Mean pairwise Euclidean distance among present teammates | Raw mean | `0.0` for absent slots or teams with < 2 valid players |
| `features.stretch` | float | `>= 0` | Pitch units | Alias for `team_depth` | Raw | Same as `team_depth` |
| `features.receiver_availability` | float | `0.0` \| `1.0` | — | `1.0` if `nearest_opponent_dist > 0.065`, else `0.0` | Binary threshold on `nearest_opponent_dist` | `0.0` for absent slots |
| `line_id` | int | `>= 0` | — | 1D gap detection on teammate x-coordinates with threshold 0.15 | Deterministic group ID from sorted x-values | `0` for absent slots |
| `lane_id` | int | `>= 0` | — | 1D gap detection on teammate y-coordinates with threshold 0.15 | Deterministic group ID from sorted y-values | `0` for absent slots |

**ROLE_VOCABULARY**: `["GK", "CB", "LB", "RB", "CDM", "CM", "LM", "RM", "LW", "RW", "CAM", "ST"]`

**Sentinel convention**: Absent player slots (e.g., when `teamLeftPlayers=3` but observation provides 11 slots) use position `(-1.0, -1.0)`. Derived features for sentinel slots are zeroed. Valid positions are those where `x != -1.0` or `y != -1.0`.

---

### 2.2 BALL

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `node_type` | string | `"BALL"` | — | Constant | None | Never missing |
| `node_id` | string | `"ball"` | — | Constant | None | Never missing |
| `position.x` | float | `[-1.0, 1.0]` | Pitch units | ObservationEncoder.ts index 88 | Raw pitch coordinate | N/A |
| `position.y` | float | `[-0.42, 0.42]` | Pitch units | ObservationEncoder.ts index 89 | Raw pitch coordinate | N/A |
| `position.z` | float | Unbounded | Pitch units (height) | ObservationEncoder.ts index 90 | Raw pitch coordinate | N/A |
| `velocity.vx` | float | Scaled by 50 | Pitch units / tick | ObservationEncoder.ts index 91 | Pre-scaled by engine | N/A |
| `velocity.vy` | float | Scaled by 50 | Pitch units / tick | ObservationEncoder.ts index 92 | Pre-scaled by engine | N/A |
| `velocity.vz` | float | Scaled by 50 | Pitch units / tick | ObservationEncoder.ts index 93 | Pre-scaled by engine | N/A |
| `ownership` | string | `"none"` \| `"left"` \| `"right"` | — | ObservationEncoder.ts indices 94-96 one-hot | None | `"none"` when no team possesses ball |
| `speed` | float | `>= 0` | Pitch units / tick | `math.hypot(vx, vy, vz)` | Raw Euclidean norm | N/A |

---

### 2.3 GOAL

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `node_type` | string | `"GOAL"` | — | Constant | None | Never missing |
| `node_id` | string | `"goal_left"` \| `"goal_right"` | — | Deterministic by team side | None | Never missing |
| `team` | string | `"left"` \| `"right"` | — | `goal_left` → `"left"`, `goal_right` → `"right"` | None | Never missing |
| `position.x` | float | `-1.0` (left) \| `1.0` (right) | Pitch units | Rules.ts PITCH.minX / PITCH.maxX | None | Never missing |
| `position.y` | float | `0.0` | Pitch units | Rules.ts goal geometry centered at y=0 | None | Never missing |
| `width` | float | `0.14` | Pitch units | Rules.ts PITCH.goalWidth | Constant | Never missing |
| `height` | float | `0.05` | Pitch units | Rules.ts PITCH.goalHeight | Constant | Never missing |
| `depth` | float | `0.04` | Pitch units | Rules.ts PITCH.goalDepth | Constant | Never missing |

---

### 2.4 TEAM_SHAPE

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `node_type` | string | `"TEAM_SHAPE"` | — | Constant | None | Never missing |
| `team` | string | `"left"` \| `"right"` | — | One per team | None | Never missing |
| `formation_deviation.inter_line_spacing_variance` | float | `>= 0` | Pitch units² | Variance of consecutive line centroid x-gaps; 0 when < 2 lines | Population variance | `0.0` when team has 0 valid players |
| `formation_deviation.mean_line_compactness` | float | `[0, 1]` | — | Mean per-line compactness: `sigma_y / team_width` per line, averaged across lines; 0 when team_width = 0 | Normalized by team_width | `0.0` when team has 0 valid players |
| `formation_deviation.num_lines` | int | `>= 1` | — | Count of distinct line groups from gap detection on teammate x-coordinates (threshold 0.15) | Deterministic count | `1` when team has 0 valid players |
| `formation_deviation.num_lanes` | int | `>= 1` | — | Count of distinct lane groups from gap detection on teammate y-coordinates (threshold 0.15) | Deterministic count | `1` when team has 0 valid players |

**Note**: TEAM_SHAPE nodes do not have an explicit `node_id` in the schema. They are implicitly identified as `team_shape_{team}` (e.g., `team_shape_left`).

---

### 2.5 FORMATION_SLOT

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `node_type` | string | `"FORMATION_SLOT"` | — | Constant | None | Never missing (empty list if scenario has no formation) |
| `slot_id` | string | `"left_slot{i}"` \| `"right_slot{i}"` | — | Deterministic: `{team}_{0-based index in FORMATIONS array}` | None | Never missing for present slots |
| `team` | string | `"left"` \| `"right"` | — | Derived from formation template | None | Never missing for present slots |
| `role` | string | ROLE_VOCABULARY | — | Rules.ts FormationNode.role | None | Never missing for present slots |
| `xRatio` | float | `[0, 1]` | — | Rules.ts FormationNode.xRatio | Template ratio | Never missing for present slots |
| `yRatio` | float | `[0, 1]` | — | Rules.ts FormationNode.yRatio | Template ratio | Never missing for present slots |
| `x` | float | `[-1.0, 1.0]` | Pitch units | `Rules.ts:141-168` getFormationPositions: `-1.0 + xRatio * 1.2` (left) or `1.0 - xRatio * 1.2` (right) | Absolute pitch coordinate | Never missing for present slots |
| `y` | float | `[-0.42, 0.42]` | Pitch units | `Rules.ts:141-168` getFormationPositions: `-0.42 + yRatio * 0.84` (left) or `0.42 - yRatio * 0.84` (right) | Absolute pitch coordinate | Never missing for present slots |

**Presence**: FORMATION_SLOT nodes are only generated for scenarios that have a `formation` field in `ScenarioRegistry.ts` (e.g., `11_vs_11`). Academy scenarios without a formation return an empty list.

---

### 2.6 SCENARIO (graph-level, not in `nodes` array)

The SCENARIO node is stored at `graph["scenario"]`, not inside the `nodes` array.

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `node_type` | string | `"SCENARIO"` | — | Constant | None | Never missing |
| `node_id` | string | `"scenario"` | — | Constant | None | Never missing |
| `id` | string | Scenario identifier (e.g., `"academy_3_vs_1_defender_3"`) | — | ScenarioConfig.id (ScenarioRegistry.ts) | None | Never missing |
| `time_limit_seconds` | float | `> 0` | Seconds | ScenarioConfig.timeLimitSeconds | Raw | Never missing |
| `terminate_on_opponent_possession` | bool | `true` \| `false` | — | ScenarioConfig.terminateOnOpponentPossession | None | Never missing |
| `objectives` | array of objects | See Objective sub-dictionary below | — | ScenarioConfig.objectives | None | Never missing (empty array if scenario has no objectives) |
| `objective_vocabulary` | float[10] | Multi-hot (0.0 or 1.0) | — | Alphabetical objective IDs present in scenario | Multi-hot over fixed 10-element vocabulary | `0.0` for objectives not present in scenario |
| `reward_scoring` | float | `>= 0` | Reward units | ScenarioConfig.rewards.scoring | Raw | Never missing |
| `reward_completion` | float | `>= 0` | Reward units | ScenarioConfig.rewards.completion | Raw | Never missing |
| `rewards.scoring` | float | `>= 0` | Reward units | ScenarioConfig.rewards.scoring | Raw | Never missing |
| `rewards.completion` | float | `>= 0` | Reward units | ScenarioConfig.rewards.completion | Raw | Never missing |

**Objective sub-dictionary** (each element of `objectives` array):

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `id` | string | Objective identifier | — | ScenarioConfig.objectives[].id | None | Never missing |
| `text` | string | Human-readable description | — | ScenarioConfig.objectives[].text | None | Never missing |
| `is_completed` | bool | `true` \| `false` | — | Set by `GameEngine.ts evaluateScenarioConditions()` | None | `false` at graph construction time (evaluated during episode) |
| `is_failed` | bool | `true` \| `false` | — | Set by `GameEngine.ts evaluateScenarioConditions()` | None | `false` at graph construction time (evaluated during episode) |

**Objective vocabulary ordering** (10 elements, alphabetical):

1. `avoid_dispossess`
2. `clean_sheet`
3. `complete_pass`
4. `complete_passes`
5. `control_possession`
6. `create_triangle`
7. `retain_possession`
8. `score_goal`
9. `within_time`
10. `win_match`

---

## 3. Edge Feature Dictionary

### 3.1 TEAMMATE

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `edge_type` | string | `"TEAMMATE"` | — | Constant | None | Never missing |
| `source` | string | Player `global_id` | — | Left or right team player | None | Never missing |
| `target` | string | Player `global_id` | — | Left or right team player | None | Never missing |
| `distance` | float | `>= 0` | Pitch units (Euclidean) | `math.hypot(dx, dy)` between source and target | Raw Euclidean distance | N/A |
| `relative_x` | float | Unbounded | Pitch units | `target.x - source.x` | Raw coordinate difference | N/A |
| `relative_y` | float | Unbounded | Pitch units | `target.y - source.y` | Raw coordinate difference | N/A |
| `relative_vx` | float | Scaled by 50 | Pitch units / tick | `target.velocity.vx - source.velocity.vx` | Raw velocity difference (pre-scaled) | N/A |
| `relative_vy` | float | Scaled by 50 | Pitch units / tick | `target.velocity.vy - source.velocity.vy` | Raw velocity difference (pre-scaled) | N/A |
| `angle` | float | `[-π, π]` | Radians | `math.atan2(dy, dx)` — bearing from source to target | Radians from +x axis | N/A |
| `closing_speed` | float | Unbounded | Pitch units / tick² | `-(dx*dvx + dy*dvy) / distance` | Negative = approaching; 0.0 when distance < 1e-6 | `0.0` when players are coincident |

**Topology**: Complete graph within each team (all pairs `i < j`). Absent players (sentinel positions) are excluded.

---

### 3.2 OPPONENT

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `edge_type` | string | `"OPPONENT"` | — | Constant | None | Never missing |
| `source` | string | Player `global_id` (left team) | — | Left team player | None | Never missing |
| `target` | string | Player `global_id` (right team) | — | Right team player | None | Never missing |
| `distance` | float | `>= 0` | Pitch units (Euclidean) | `math.hypot(dx, dy)` | Raw Euclidean distance | N/A |
| `relative_x` | float | Unbounded | Pitch units | `target.x - source.x` | Raw | N/A |
| `relative_y` | float | Unbounded | Pitch units | `target.y - source.y` | Raw | N/A |
| `relative_vx` | float | Scaled by 50 | Pitch units / tick | `target.velocity.vx - source.velocity.vx` | Raw | N/A |
| `relative_vy` | float | Scaled by 50 | Pitch units / tick | `target.velocity.vy - source.velocity.vy` | Raw | N/A |
| `angle` | float | `[-π, π]` | Radians | `math.atan2(dy, dx)` | Radians from +x axis | N/A |
| `closing_speed` | float | Unbounded | Pitch units / tick² | `-(dx*dvx + dy*dvy) / distance` | Same as TEAMMATE | `0.0` when distance < 1e-6 |
| `pressure` | float | `0.0` \| `1.0` | — | `1.0` if `distance < 0.065`, else `0.0` | Binary threshold on `distance` | `0.0` when distance >= 0.065 |

**Topology**: Complete bipartite graph between left and right teams. Absent players excluded.

---

### 3.3 NEAR

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `edge_type` | string | `"NEAR"` | — | Constant | None | Never missing |
| `source` | string | Player `global_id` | — | Cross-team player | None | Never missing |
| `target` | string | Player `global_id` | — | Cross-team player | None | Never missing |
| `distance` | float | `>= 0` | Pitch units (Euclidean) | `math.hypot(dx, dy)` at graph construction time | Raw | N/A |
| `threshold` | float | `0.065` (constant) | Pitch units | `NEAR_THRESHOLD` constant | None | Never missing; constant value |

**Topology**: Sparse cross-team edges only when `distance < 0.065`. Justification: `Physics.ts:348` uses `distToBall < 0.065` as the proximity threshold for tackle execution, making this a defensible near/pressure threshold. Edges are undirected (added once per unordered pair).

---

### 3.4 POSSESSES

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `edge_type` | string | `"POSSESSES"` | — | Constant | None | Never missing when owner exists |
| `source` | string | Player `global_id` | — | `info.ground_truth.current_ball_owner.agent_id` | None | Edge omitted when `exact_owner_id` is `None` |
| `target` | string | `"ball"` | — | Constant | None | Never missing |

**Topology**: At most one POSSESSES edge per graph (directed from owner to ball). No geometric features.

---

### 3.5 PLAYER_BALL

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `edge_type` | string | `"PLAYER_BALL"` | — | Constant | None | Never missing for present players |
| `source` | string | Player `global_id` | — | Present player | None | Never missing for present players |
| `target` | string | `"ball"` | — | Constant | None | Never missing |
| `distance` | float | `>= 0` | Pitch units (Euclidean) | `math.hypot(dx, dy)` from player to ball | Raw | N/A |
| `relative_x` | float | Unbounded | Pitch units | `ball.x - player.x` | Raw | N/A |
| `relative_y` | float | Unbounded | Pitch units | `ball.y - player.y` | Raw | N/A |
| `relative_vx` | float | Scaled by 50 | Pitch units / tick | `ball.velocity.vx - player.velocity.vx` | Raw | N/A |
| `relative_vy` | float | Scaled by 50 | Pitch units / tick | `ball.velocity.vy - player.velocity.vy` | Raw | N/A |
| `angle` | float | `[-π, π]` | Radians | `math.atan2(dy, dx)` — bearing from player to ball | Radians from +x axis | N/A |
| `has_possession` | bool | `true` \| `false` | — | `player.global_id == exact_owner_id` | None | `false` for all non-owners |

**Topology**: Directed edges from each present player to the ball.

---

### 3.6 PLAYER_GOAL

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `edge_type` | string | `"PLAYER_GOAL"` | — | Constant | None | Never missing for present players |
| `source` | string | Player `global_id` | — | Present player | None | Never missing for present players |
| `target` | string | `"goal_left"` \| `"goal_right"` | — | Goal node ID | None | Never missing |
| `distance` | float | `>= 0` | Pitch units (Euclidean) | `math.hypot(dx, dy)` from player to goal center | Raw | N/A |
| `relative_x` | float | Unbounded | Pitch units | `goal.x - player.x` | Raw | N/A |
| `relative_y` | float | Unbounded | Pitch units | `goal.y - player.y` | Raw | N/A |
| `angle` | float | `[-π, π]` | Radians | `math.atan2(dy, dx)` — bearing from player to goal | Radians from +x axis | N/A |
| `shot_angle` | float | `>= 0` | Radians | `2 * atan(goalWidth / (2 * distance))`; 0.0 when distance <= 0 | Opening angle to goal mouth | `0.0` when player is at goal center |
| `nearest_opponent_pressure` | float | `>= 0` | Pitch units (Euclidean) | Min distance to any present opponent | Raw | `0.0` when no opponents are present |

**Topology**: Directed edges from each present player to both goals (2 edges per player).

---

### 3.7 BALL_GOAL

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `edge_type` | string | `"BALL_GOAL"` | — | Constant | None | Never missing |
| `source` | string | `"ball"` | — | Constant | None | Never missing |
| `target` | string | `"goal_left"` \| `"goal_right"` | — | Goal node ID | None | Never missing |
| `distance` | float | `>= 0` | Pitch units (Euclidean) | `math.hypot(dx, dy)` from ball to goal center | Raw | N/A |
| `relative_x` | float | Unbounded | Pitch units | `goal.x - ball.x` | Raw | N/A |
| `relative_y` | float | Unbounded | Pitch units | `goal.y - ball.y` | Raw | N/A |
| `angle` | float | `[-π, π]` | Radians | `math.atan2(dy, dx)` — bearing from ball to goal | Radians from +x axis | N/A |
| `shot_angle` | float | `>= 0` | Radians | `2 * atan(goalWidth / (2 * distance))`; 0.0 when distance <= 0 | Opening angle to goal mouth | `0.0` when ball is at goal center |

**Topology**: Directed edges from ball to both goals (2 edges total).

---

### 3.8 ASSIGNED_TO

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `edge_type` | string | `"ASSIGNED_TO"` | — | Constant | None | Never missing when formation slots exist |
| `source` | string | Player `global_id` | — | Present player | None | Never missing for present players |
| `target` | string | FORMATION_SLOT `slot_id` | — | Deterministic role-based matching | None | Never missing when slots exist |
| `role` | string | ROLE_VOCABULARY | — | Formation slot role | None | Never missing when slots exist |
| `nominal_x` | float | `[-1.0, 1.0]` | Pitch units | `Rules.ts:141-168` getFormationPositions absolute x | Absolute pitch coordinate | Never missing when slots exist |
| `nominal_y` | float | `[-0.42, 0.42]` | Pitch units | `Rules.ts:141-168` getFormationPositions absolute y | Absolute pitch coordinate | Never missing when slots exist |
| `deviation_x` | float | Unbounded | Pitch units | `player.x - nominal_x` | Raw coordinate difference | N/A |
| `deviation_y` | float | Unbounded | Pitch units | `player.y - nominal_y` | Raw coordinate difference | N/A |
| `deviation_distance` | float | `>= 0` | Pitch units (Euclidean) | `math.hypot(deviation_x, deviation_y)` | Raw Euclidean distance | N/A |

**Topology**: Directed edges from present players to their matched formation slot. Matching is deterministic: exact role match preferred; if no role match, nearest slot by `(x, y)` is used as fallback. Ties broken by sorting candidates by `(x, y)`.

---

### 3.9 BELONGS_TO_SHAPE

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `edge_type` | string | `"BELONGS_TO_SHAPE"` | — | Constant | None | Never missing for present players |
| `source` | string | Player `global_id` | — | Present player | None | Never missing for present players |
| `target` | string | `"team_shape_{team}"` | — | Implicit TEAM_SHAPE node ID | None | Never missing for present players |
| `team` | string | `"left"` \| `"right"` | — | Player's team | None | Never missing for present players |

**Topology**: Directed edges from each present player to their team's TEAM_SHAPE node.

---

### 3.10 SCENARIO_CONTEXT

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `edge_type` | string | `"SCENARIO_CONTEXT"` | — | Constant | None | Never missing for present players |
| `source` | string | Player `global_id` | — | Present player | None | Never missing for present players |
| `target` | string | `"scenario"` | — | Constant (SCENARIO node_id) | None | Never missing |

**Topology**: Directed edges from each present player to the SCENARIO node. No additional geometric features.

---

### 3.11 FORMATION_ADJACENCY

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `edge_type` | string | `"FORMATION_ADJACENCY"` | — | Constant | None | Never missing when slots exist |
| `source` | string | FORMATION_SLOT `slot_id` | — | Formation slot | None | Never missing when slots exist |
| `target` | string | FORMATION_SLOT `slot_id` | — | Formation slot | None | Never missing when slots exist |

**Topology**: Undirected k=2 nearest neighbors in `(xRatio, yRatio)` space for each team's formation slots. No additional features.

---

### 3.12 FORMATION_LINE

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `edge_type` | string | `"FORMATION_LINE"` | — | Constant | None | Never missing when slots exist |
| `source` | string | FORMATION_SLOT `slot_id` | — | Formation slot | None | Never missing when slots exist |
| `target` | string | FORMATION_SLOT `slot_id` | — | Formation slot | None | Never missing when slots exist |
| `line_id` | int | `0` .. `3` | — | xRatio band: 0=GK (<0.10), 1=Defense (0.10-0.35), 2=Midfield (0.35-0.60), 3=Attack (>=0.60) | Deterministic band assignment | Never missing when slots exist |

**Topology**: Complete graph within each line group (same `line_id`) for each team.

---

### 3.13 FORMATION_LANE

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `edge_type` | string | `"FORMATION_LANE"` | — | Constant | None | Never missing when slots exist |
| `source` | string | FORMATION_SLOT `slot_id` | — | Formation slot | None | Never missing when slots exist |
| `target` | string | FORMATION_SLOT `slot_id` | — | Formation slot | None | Never missing when slots exist |
| `lane_id` | int | `0` .. `2` | — | yRatio band: 0=Left (<0.33), 1=Center (0.33-0.67), 2=Right (>0.67) | Deterministic band assignment | Never missing when slots exist |

**Topology**: Complete graph within each lane group (same `lane_id`) for each team.

---

## 4. Graph-Level Context (z_scenario)

| Name | Type | Range / Values | Units | Source | Normalization | Missing Semantics |
|------|------|----------------|-------|--------|---------------|-------------------|
| `z_scenario` | Float32Array[8] | Unbounded floats | — | Optional task context vector from engine | Attached as `graph["z_scenario"]` when provided | Entire field omitted when `z_scenario` parameter is `None` |

`z_scenario` is **not** a node or edge feature. It is a graph-level attribute attached directly to the graph dict. It is separate from the SCENARIO node's `objective_vocabulary` (which is a 10-element multi-hot over known objective IDs). The 8-element `z_scenario` vector carries richer task context from the engine.

---

## 5. Normalization Summary

### Raw features (no normalization applied)

| Feature | Notes |
|---------|-------|
| `position.x` | Pitch coordinates in `[-1.0, 1.0]` |
| `position.y` | Pitch coordinates in `[-0.42, 0.42]` |
| `position.z` (BALL) | Height in pitch units |
| `velocity.vx`, `velocity.vy`, `velocity.vz` | Pre-scaled by engine (÷50 to recover raw) |
| `relative_x`, `relative_y` | Raw coordinate differences |
| `relative_vx`, `relative_vy` | Raw velocity differences (pre-scaled) |
| `distance` (all edge types) | Raw Euclidean distances in pitch units |
| `shot_angle` | Radians from `2 * atan(goalWidth / (2 * distance))` |
| `deviation_x`, `deviation_y` | Raw coordinate differences from formation slot |
| `deviation_distance` | Raw Euclidean distance |
| `team_width`, `team_depth`, `compactness`, `stretch` | Raw pitch unit distances |
| `nearest_teammate_dist`, `nearest_opponent_dist` | Raw Euclidean distances |
| `inter_line_spacing_variance` | Raw variance in pitch units² |
| `mean_line_compactness` | Normalized internally by `team_width` to `[0, 1]` |
| `time_limit_seconds` | Raw seconds |
| `reward_scoring`, `reward_completion` | Raw reward magnitudes |
| `angle`, `closing_speed` | Raw computed values |

### Pre-scaled features (engine applies scaling before observation)

| Feature | Scale factor | Source |
|---------|--------------|--------|
| `velocity.vx`, `velocity.vy` (PLAYER) | ÷50 | ObservationEncoder.ts |
| `velocity.vx`, `velocity.vy`, `velocity.vz` (BALL) | ÷50 | ObservationEncoder.ts |

### Categorical / boolean features

| Feature | Encoding |
|---------|----------|
| `team` | String enum: `"left"`, `"right"` |
| `role` | String enum: ROLE_VOCABULARY (12 values) |
| `role_one_hot` | 12-float one-hot vector |
| `ownership` | String enum: `"none"`, `"left"`, `"right"` |
| `is_active`, `is_controlled`, `is_goalkeeper`, `is_completed`, `is_failed` | Boolean |
| `has_possession` | Boolean |
| `pressure` | Binary float: `0.0` or `1.0` |
| `receiver_availability` | Binary float: `0.0` or `1.0` |
| `objective_vocabulary` | 10-float multi-hot |
| `line_id`, `lane_id` | Integer group IDs from gap detection |
| `team_index` | Integer 0-10 |
| `line_id` (FORMATION_LINE) | Integer 0-3 from xRatio bands |
| `lane_id` (FORMATION_LANE) | Integer 0-2 from yRatio bands |

### Threshold-derived features

| Feature | Threshold | Condition |
|---------|-----------|-----------|
| `receiver_availability` | `0.065` | `1.0` if `nearest_opponent_dist > 0.065` |
| `pressure` (OPPONENT) | `0.065` | `1.0` if `distance < 0.065` |
| `NEAR` edge inclusion | `0.065` | Edge created only if `distance < 0.065` |

### Constants

| Feature | Value | Source |
|---------|-------|--------|
| `GOAL_WIDTH` | `0.14` | Rules.ts PITCH.goalWidth |
| `GOAL_CENTER_Y` | `0.0` | Rules.ts goal geometry |
| `NEAR_THRESHOLD` | `0.065` | Physics.ts:348 tackle proximity |
| `LINE_LANE_GAP_THRESHOLD` | `0.15` | Gap detection for line/lane assignment |
| `line_bands` (FORMATION_LINE) | `[0.10, 0.35, 0.60]` | xRatio thresholds |
| `lane_bands` (FORMATION_LANE) | `[0.33, 0.67]` | yRatio thresholds |
| `PITCH_WIDTH` | `2.4` (`1.2 * 2`) | Implicit in formation position calculation |
| `PITCH_HEIGHT` | `0.84` (`0.42 * 2`) | Implicit in formation position calculation |

---

## 6. Missing Value Semantics

### Absent player slots (sentinel positions)

When a scenario defines fewer players than the maximum 11 per team (e.g., `teamLeftPlayers=3` in `academy_empty_goal`), the observation still provides 11 position slots per team. Unused slots are filled with sentinel values:

- **Position**: `(-1.0, -1.0)`
- **Velocity**: `(0.0, 0.0)`
- **Derived features**: All zeroed (`nearest_teammate_dist=0.0`, `nearest_opponent_dist=0.0`, `team_width=0.0`, `team_depth=0.0`, `compactness=0.0`, `stretch=0.0`, `receiver_availability=0.0`)
- **line_id / lane_id**: `0`
- **role**: `"UNKNOWN"` if index exceeds scenario setup length
- **role_one_hot**: Zero vector `[0.0, ..., 0.0]`

Sentinel players are **excluded** from edge construction (no TEAMMATE, OPPONENT, PLAYER_BALL, PLAYER_GOAL, or SCENARIO_CONTEXT edges).

### Missing ball ownership

When no team possesses the ball, `ownership` is `"none"` and no POSSESSES edge is created. The `exact_owner_id` from `info.ground_truth.current_ball_owner.agent_id` may be `None` when the bridge does not provide authoritative ownership data.

### Missing active player

When the active player one-hot (indices 97-107) has no value > 0.5, `active_index` is set to `-1` and no player is marked `is_active=true`. When the viewpoint role is ambiguous (present in both teams), `viewpoint_team` defaults to `"left"`.

### Missing formation data

When a scenario has no `formation` field (most academy scenarios), `_build_formation_slot_nodes` returns an empty list. Consequently, no ASSIGNED_TO, BELONGS_TO_SHAPE, FORMATION_ADJACENCY, FORMATION_LINE, or FORMATION_LANE edges are created.

### Missing team shape data

When a team has 0 valid players, `_compute_team_shape` returns defaults: `inter_line_spacing_variance=0.0`, `mean_line_compactness=0.0`, `num_lines=1`, `num_lanes=1`.

### Missing z_scenario

When the `z_scenario` parameter is `None`, the `z_scenario` key is entirely omitted from the graph dict.

---

## 7. Determinism Guarantees

- **Node ordering**: Left team players first (by `team_index`), then right team players (by `team_index`), then BALL, GOAL nodes, TEAM_SHAPE nodes, FORMATION_SLOT nodes.
- **Edge ordering**: Lexicographic sort by `(edge_type, source, target)` after construction.
- **Line/lane assignment**: Gap detection on sorted coordinates with fixed threshold `0.15`.
- **Formation slot matching**: Deterministic role-based matching with `(x, y)` tie-breaking.
- **Global IDs**: `{team}_{index}` where index is 0-based within team, matching the 127-dim observation ordering.

---

## 8. Schema Version

This documentation corresponds to **GNN Graph Schema v3** as defined in `training/gnn_graph_schema.json` and implemented in `training/gnn_graph_builder.py`.
