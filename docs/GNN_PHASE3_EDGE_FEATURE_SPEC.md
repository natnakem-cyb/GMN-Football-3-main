# GNN Phase 3 Edge Feature Specification

## 1. Introduction

This document specifies the complete edge feature vocabulary for the Phase 3 GNN graph produced by `training/gnn_graph_builder.py` (`build_graph`, line 1494). It is the authoritative reference for implementors of the Phase 4 GNN message-passing layers, evaluators, and any downstream consumer of edge features.

The graph is validated against `training/gnn_graph_schema.json` (v3). All 13 edge types are enumerated below with their exact feature names, types, derivation formulas, directionality semantics, and normalization contracts.

Pitch coordinate conventions (source: `src/engine/Rules.ts`):

- The pitch is a normalized rectangle with x in [-1.0, 1.0] and y in [-0.42, 0.42].
- `+x` points toward the right goal (`goal_right` at x = 1.0).
- `+y` points toward the top sideline as viewed from the left team's perspective.
- Positions are in pitch-normalized units, not pixels or meters.
- Velocity components (vx, vy) are in the same pitch-normalized units per tick.

---

## 2. Normalization Contract

All edge features are emitted in **raw pitch-normalized units**. The graph builder (`gnn_graph_builder.py`) performs no per-feature normalization, clipping, or scaling. Normalization, if required, is the responsibility of the consuming GNN model (Phase 4).

For every normalized feature, the contract below records the raw quantity, the function a Phase 4 model should apply, the expected raw range, clipping policy, and missing/unavailable behavior.

| Feature | Raw Quantity | Normalization Function | Expected Raw Range | Clipping | Missing Behavior |
|---------|-------------|----------------------|--------------------|----------|------------------|
| `distance` | Euclidean distance in pitch units | Divide by pitch diagonal (~2.016) or use learned scaling | [0.0, ~2.016] | None applied at emission | 0.0 if either endpoint is sentinel (-1.0, -1.0); edge may be omitted entirely |
| `relative_x` | dx = target.x - source.x | Divide by pitch half-width (1.0) or use per-feature standardization | [-2.0, 2.0] | None | 0.0 if source is sentinel |
| `relative_y` | dy = target.y - source.y | Divide by pitch half-height (0.42) or use per-feature standardization | [-0.84, 0.84] | None | 0.0 if source is sentinel |
| `relative_vx` | dvx = target.vx - source.vx | Per-feature standardization computed from training distribution | Unbounded in theory; typical magnitude < 0.1 per tick | None | 0.0 if either endpoint is sentinel |
| `relative_vy` | dvy = target.vy - source.vy | Per-feature standardization computed from training distribution | Unbounded in theory; typical magnitude < 0.1 per tick | None | 0.0 if either endpoint is sentinel |
| `angle` | atan2(dy, dx) in radians | Divide by pi or use angular encoding (sin/cos) | [-pi, pi] | None | 0.0 if distance < 1e-6 (implementation guard); atan2(0, 0) is undefined, replaced with 0.0 |
| `closing_speed` | -(dx*dvx + dy*dvy) / distance | Per-feature standardization | Unbounded; negative = approaching, positive = separating | None | 0.0 if distance < 1e-6 |
| `pressure` | Binary indicator (1.0 or 0.0) | No normalization required; already in [0, 1] | {0.0, 1.0} | None | 0.0 if distance >= NEAR_THRESHOLD |
| `shot_angle` | 2 * atan(goalWidth / (2 * distance)) | Divide by pi or use learned scaling | [0.0, ~0.44 rad] (max at goal mouth) | None | 0.0 if distance <= 0.0 (implementation guard) |
| `threshold` | Constant NEAR_THRESHOLD | No normalization required; constant | 0.065 | None | Not missing; always present as constant 0.065 |
| `has_possession` | Boolean (true/false) | Map to {0.0, 1.0} or use learned embedding | {false, true} | None | false if exact_ball_owner_id is None or does not match this player |
| `nearest_opponent_pressure` | Euclidean distance to closest opponent | Divide by pitch diagonal (~2.016) or use learned scaling | [0.0, ~2.016] | None | 0.0 if no opponents present |
| `deviation_x` | player.x - nominal_x | Divide by pitch half-width (1.0) or use per-feature standardization | [-1.0, 1.0] in typical play; can exceed at edges | None | 0.0 if player is sentinel or no FORMATION_SLOT nodes exist |
| `deviation_y` | player.y - nominal_y | Divide by pitch half-height (0.42) or use per-feature standardization | [-0.42, 0.42] in typical play; can exceed at edges | None | 0.0 if player is sentinel or no FORMATION_SLOT nodes exist |
| `deviation_distance` | Euclidean(player.pos, nominal.pos) | Divide by pitch diagonal (~2.016) or use learned scaling | [0.0, ~2.016] | None | 0.0 if player is sentinel or no FORMATION_SLOT nodes exist |
| `nominal_x` | Formation slot x from Rules.ts getFormationPositions | Divide by pitch half-width (1.0) | [-1.0, 1.0] | None | Not missing; always present when ASSIGNED_TO edge exists |
| `nominal_y` | Formation slot y from Rules.ts getFormationPositions | Divide by pitch half-height (0.42) | [-0.42, 0.42] | None | Not missing; always present when ASSIGNED_TO edge exists |
| `line_id` | Integer line band identifier | Integer embedding or one-hot (4 classes) | {0, 1, 2, 3} | None | 0 if fewer than 2 valid players (default assignment) |
| `lane_id` | Integer lane band identifier | Integer embedding or one-hot (3 classes) | {0, 1, 2} | None | 0 if fewer than 2 valid players (default assignment) |
| `role` | String role label | Learned role embedding (12-class vocabulary) | One of 12 strings | None | "UNKNOWN" if role index out of range in observation |
| `team` | String team label | Learned team embedding (2-class) | {"left", "right"} | None | Not missing |

---

## 3. Per-Edge-Type Feature Dictionary

### 3.1 TEAMMATE

| Feature | Type | Unit | Description |
|---------|------|------|-------------|
| `distance` | float >= 0 | pitch units | Euclidean distance between source and target player centers. |
| `relative_x` | float | pitch units | target.x - source.x. |
| `relative_y` | float | pitch units | target.y - source.y. |
| `relative_vx` | float | pitch units/tick | target.velocity.vx - source.velocity.vx. |
| `relative_vy` | float | pitch units/tick | target.velocity.vy - source.velocity.vy. |
| `angle` | float | radians | Bearing from source to target, measured from +x axis using atan2(dy, dx). Range: [-pi, pi]. |
| `closing_speed` | float | pitch units/tick | Rate of change of inter-player distance. Negative means players are approaching; positive means separating. |

Source: `gnn_graph_builder.py:1062-1093` (`_build_rich_teammate_edges`).

### 3.2 OPPONENT

All TEAMMATE features plus:

| Feature | Type | Unit | Description |
|---------|------|------|-------------|
| `pressure` | float in {0.0, 1.0} | dimensionless | 1.0 if distance < NEAR_THRESHOLD (0.065), else 0.0. Binary pressure signal aligned with the tackle threshold in `Physics.ts`. |

Source: `gnn_graph_builder.py:1096-1125` (`_build_rich_opponent_edges`).

### 3.3 NEAR

| Feature | Type | Unit | Description |
|---------|------|------|-------------|
| `distance` | float >= 0 | pitch units | Euclidean distance between the two players at graph construction time. |
| `threshold` | const | pitch units | Fixed constant 0.065. The inclusion criterion for this edge type. |

Source: `gnn_graph_builder.py:1367-1383`. Emitted only for cross-team pairs with distance < 0.065.

### 3.4 POSSESSES

| Feature | Type | Description |
|---------|------|-------------|
| *(none)* | -- | No geometric features. Identity is conveyed entirely by source and target node IDs. |

- `source`: `global_id` of the player on the owning team closest to the ball center (heuristic, see Section 7).
- `target`: `"ball"` (constant string).

Source: `gnn_graph_builder.py:1359-1365`.

### 3.5 PLAYER_BALL

| Feature | Type | Unit | Description |
|---------|------|------|-------------|
| `distance` | float >= 0 | pitch units | Euclidean distance from player center to ball center. |
| `relative_x` | float | pitch units | ball.x - player.x. |
| `relative_y` | float | pitch units | ball.y - player.y. |
| `relative_vx` | float | pitch units/tick | ball.velocity.vx - player.velocity.vx. |
| `relative_vy` | float | pitch units/tick | ball.velocity.vy - player.velocity.vy. |
| `angle` | float | radians | Bearing from player to ball, measured from +x axis. |
| `has_possession` | bool | dimensionless | True only for the exact ball owner. False for all other players. |

Source: `gnn_graph_builder.py:1128-1158` (`_build_player_ball_edges`).

### 3.6 PLAYER_GOAL

| Feature | Type | Unit | Description |
|---------|------|------|-------------|
| `distance` | float >= 0 | pitch units | Euclidean distance from player center to goal center. |
| `relative_x` | float | pitch units | goal.x - player.x. |
| `relative_y` | float | pitch units | goal.y - player.y. |
| `angle` | float | radians | Bearing from player to goal, measured from +x axis. |
| `shot_angle` | float >= 0 | radians | Opening angle subtended by the goal mouth at the player's position. |
| `nearest_opponent_pressure` | float >= 0 | pitch units | Euclidean distance from player to the closest opponent. 0.0 if no opponents are present. |

Two edges are emitted per present player: one to `goal_left` and one to `goal_right`.

Source: `gnn_graph_builder.py:1161-1199` (`_build_player_goal_edges`).

### 3.7 BALL_GOAL

| Feature | Type | Unit | Description |
|---------|------|------|-------------|
| `distance` | float >= 0 | pitch units | Euclidean distance from ball center to goal center. |
| `relative_x` | float | pitch units | goal.x - ball.x. |
| `relative_y` | float | pitch units | goal.y - ball.y. |
| `angle` | float | radians | Bearing from ball to goal, measured from +x axis. |
| `shot_angle` | float >= 0 | radians | Opening angle subtended by the goal mouth at the ball's position. |

Two edges are emitted: one to `goal_left` and one to `goal_right`.

Source: `gnn_graph_builder.py:1202-1227` (`_build_ball_goal_edges`).

### 3.8 ASSIGNED_TO

| Feature | Type | Unit | Description |
|---------|------|------|-------------|
| `role` | string | -- | Tactical role of the matched formation slot (e.g. "GK", "CB", "ST"). |
| `nominal_x` | float | pitch units | x position of the matched formation slot from `Rules.ts getFormationPositions`. |
| `nominal_y` | float | pitch units | y position of the matched formation slot. |
| `deviation_x` | float | pitch units | player.x - nominal_x. Positive means player is right of slot. |
| `deviation_y` | float | pitch units | player.y - nominal_y. Positive means player is above slot. |
| `deviation_distance` | float >= 0 | pitch units | Euclidean distance between player position and nominal slot position. |

One edge per present player, targeting the nearest FORMATION_SLOT on the same team (Euclidean distance in pitch space). Omitted if no FORMATION_SLOT nodes exist for the scenario.

Source: `gnn_graph_builder.py:1230-1268` (`_build_assigned_to_edges`).

### 3.9 BELONGS_TO_SHAPE

| Feature | Type | Description |
|---------|------|-------------|
| `team` | string: "left" or "right" | Team side the source player belongs to. Also identifies which TEAM_SHAPE node is the target. |

No geometric features. The target node_id is implicitly `team_shape_{team}`.

Source: `gnn_graph_builder.py:1271-1290` (`_build_belongs_to_shape_edges`).

### 3.10 SCENARIO_CONTEXT

No extra features beyond `edge_type`, `source`, and `target`.

- `source`: player `global_id`.
- `target`: `"scenario"` (constant string).

One edge per present player. The SCENARIO node lives at `graph["scenario"]` and is not included in `graph["nodes"]`.

Source: `gnn_graph_builder.py:1293-1306` (`_build_scenario_context_edges`).

### 3.11 FORMATION_ADJACENCY

No extra features.

- Connects each FORMATION_SLOT to its k=2 nearest neighbors in template (xRatio, yRatio) space (Euclidean distance).
- Undirected by construction: edges are stored with source < target lexicographically by slot_id.

Source: `gnn_graph_builder.py:1468-1483`.

### 3.12 FORMATION_LINE

| Feature | Type | Description |
|---------|------|-------------|
| `line_id` | int: 0, 1, 2, or 3 | Line band identifier. |

Line band assignment from `xRatio` thresholds:

| line_id | Label | xRatio Range |
|---------|-------|-------------|
| 0 | GK | xRatio < 0.10 |
| 1 | Defense | 0.10 <= xRatio < 0.35 |
| 2 | Midfield | 0.35 <= xRatio < 0.60 |
| 3 | Attack | xRatio >= 0.60 |

All pairs of FORMATION_SLOT nodes within the same line band on the same team are connected.

Source: `gnn_graph_builder.py:1409-1437`.

### 3.13 FORMATION_LANE

| Feature | Type | Description |
|---------|------|-------------|
| `lane_id` | int: 0, 1, or 2 | Lane band identifier. |

Lane band assignment from `yRatio` thresholds:

| lane_id | Label | yRatio Range |
|---------|-------|-------------|
| 0 | Left | yRatio < 0.33 |
| 1 | Center | 0.33 <= yRatio <= 0.67 |
| 2 | Right | yRatio > 0.67 |

All pairs of FORMATION_SLOT nodes within the same lane band on the same team are connected.

Source: `gnn_graph_builder.py:1439-1466`.

---

## 4. Directionality Rules

### 4.1 Explicit Directionality Per Edge Type

| Edge Type | Directionality | Rule |
|-----------|---------------|------|
| **TEAMMATE** | Conceptually bidirectional; stored **undirected** | Each unordered teammate pair yields exactly one edge. Source/target ordering is deterministic: sorted lexicographically by `global_id` (i.e., `i < j` where `i` and `j` are team-relative indices, yielding `left_0 < left_1 < ... < right_0 < ...`). Reverse geometry would have negated `relative_x`, `relative_y`, `relative_vx`, `relative_vy` and `angle` shifted by pi. |
| **OPPONENT** | **Directed** | Always left -> right. For every present left player `lp` and present right player `rp`, exactly one OPPONENT edge is emitted with `source = lp.global_id` and `target = rp.global_id`. No reverse (right -> left) edges are emitted. |
| **NEAR** | **Undirected** | Emitted only for cross-team pairs with distance < 0.065. Stored with `i < j` lexicographic ordering on `global_id`. |
| **POSSESSES** | **Directed** | Player -> ball. Source is the owning team's closest player to the ball center. Omitted entirely when `ball_ownership == "none"`. |
| **PLAYER_BALL** | **Directed** | Player -> ball. One edge per present player. |
| **PLAYER_GOAL** | **Directed** | Player -> goal. Two edges per present player (one per goal). |
| **BALL_GOAL** | **Directed** | Ball -> goal. Exactly two edges. |
| **ASSIGNED_TO** | **Directed** | Player -> formation_slot. One edge per present player when FORMATION_SLOT nodes exist. |
| **BELONGS_TO_SHAPE** | **Directed** | Player -> team_shape. One edge per present player. |
| **SCENARIO_CONTEXT** | **Directed** | Player -> scenario. One edge per present player. |
| **FORMATION_ADJACENCY** | **Undirected** | Slot -> slot. Stored with lexicographic slot_id ordering. |
| **FORMATION_LINE** | **Undirected** | Slot -> slot. Stored with lexicographic slot_id ordering. |
| **FORMATION_LANE** | **Undirected** | Slot -> slot. Stored with lexicographic slot_id ordering. |

### 4.2 Reverse Edge Geometry for TEAMMATE

Although TEAMMATE edges are stored undirected, the geometry is directional in the source->target sense. If a reverse edge were needed, the transformations would be:

- `reverse_relative_x = -relative_x`
- `reverse_relative_y = -relative_y`
- `reverse_relative_vx = -relative_vx`
- `reverse_relative_vy = -relative_vy`
- `reverse_angle = angle + pi` (modulo 2pi to keep in [-pi, pi])
- `reverse_closing_speed = closing_speed` (symmetric scalar)

The Phase 4 GNN should treat TEAMMATE edges as undirected in message passing, or if using directional attention, should not rely on the reverse geometry being present in the edge list.

---

## 5. Edge Construction Order

Edges are sorted by the following deterministic key applied at `gnn_graph_builder.py:1386`:

```python
edges.sort(key=lambda e: (e["edge_type"], e["source"], e.get("target", "")))
```

Sort key components in priority order:

1. **`edge_type`**: Lexicographic string comparison. Order: `BELONGS_TO_SHAPE`, `BALL_GOAL`, `FORMATION_ADJACENCY`, `FORMATION_LANE`, `FORMATION_LINE`, `NEAR`, `OPPONENT`, `POSSESSES`, `PLAYER_BALL`, `PLAYER_GOAL`, `SCENARIO_CONTEXT`, `TEAMMATE`, `ASSIGNED_TO`.
2. **`source`**: Lexicographic string comparison of the source node ID.
3. **`target`**: Lexicographic string comparison of the target node ID, with empty string as fallback for edges that omit `target` (none currently do, but the `get("target", "")` guard preserves determinism).

Within each edge type, the order of emission from the builder functions is:

- **TEAMMATE**: Left team first (left_0<->left_1, left_0<->left_2, ...), then right team. Within each team, pairs are emitted with `i < j` in team-relative index order.
- **OPPONENT**: Nested loops over `left_players` (outer) and `right_players` (inner). All left players are iterated before any right player as source.
- **NEAR**: Nested loops over `all_players` (concatenated left + right) with `i < j`, skipping same-team pairs. Emitted after POSSESSES in the sort order due to edge_type string comparison.
- **POSSESSES**: Emitted at most once, after all geometric edges but before NEAR in the sort order.
- **PLAYER_BALL**: Emitted in player node list order (left_0, left_1, ..., right_0, right_1, ...).
- **PLAYER_GOAL**: Emitted in player node list order; for each player, `goal_left` precedes `goal_right`.
- **BALL_GOAL**: `goal_left` precedes `goal_right`.
- **ASSIGNED_TO**: Emitted in player node list order.
- **BELONGS_TO_SHAPE**: Emitted in player node list order.
- **SCENARIO_CONTEXT**: Emitted in player node list order.
- **FORMATION_ADJACENCY, FORMATION_LINE, FORMATION_LANE**: Emitted within `_build_formation_edges` after all player-centric edges, with their own internal deterministic ordering by slot_id.

---

## 6. Derived Quantity Formulas

All formulas below are implemented verbatim in `training/gnn_graph_builder.py`.

### 6.1 Euclidean Distance

```python
def _euclidean(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)
```

Used for: `distance` in TEAMMATE, OPPONENT, NEAR, PLAYER_BALL, PLAYER_GOAL, BALL_GOAL, ASSIGNED_TO, and `deviation_distance`.

### 6.2 Angle (Bearing)

```python
def _angle_rad(dx, dy):
    return math.atan2(dy, dx)
```

Returns radians in [-pi, pi]. Used for: `angle` in TEAMMATE, OPPONENT, PLAYER_BALL, PLAYER_GOAL, BALL_GOAL.

- TEAMMATE/OPPONENT: `dx = target.x - source.x`, `dy = target.y - source.y`.
- PLAYER_BALL: `dx = ball.x - player.x`, `dy = ball.y - player.y`.
- PLAYER_GOAL: `dx = goal.x - player.x`, `dy = goal.y - player.y`.
- BALL_GOAL: `dx = goal.x - ball.x`, `dy = goal.y - ball.y`.

### 6.3 Closing Speed

```python
closing_speed = -(dx * dvx + dy * dvy) / dist if dist > 1e-6 else 0.0
```

Where `dx, dy` are the relative position components and `dvx, dvy` are the relative velocity components between source and target.

Interpretation:
- `closing_speed < 0`: The distance between the two players is decreasing (they are approaching).
- `closing_speed > 0`: The distance is increasing (they are separating).
- `closing_speed == 0`: Distance is neither increasing nor decreasing at this instant.

This is the negative time-derivative of distance: `-d(distance)/dt = -(dx*dvx + dy*dvy) / distance`.

Used in: TEAMMATE and OPPONENT edges.

### 6.4 Pressure

```python
pressure = 1.0 if dist < NEAR_THRESHOLD else 0.0
```

Where `NEAR_THRESHOLD = 0.065` (pitch units).

This is a binary indicator aligned with the tackle proximity threshold in `Physics.ts` (distToBall < 0.065). It is NOT a smooth sigmoid or distance-based falloff; it is a hard step function.

Used in: OPPONENT edges.

### 6.5 Shot Angle

```python
def _shot_angle(distance, goal_width=0.14):
    if distance <= 0:
        return 0.0
    return 2.0 * math.atan(goal_width / (2.0 * distance))
```

- `goal_width = 0.14` (pitch units, from `Rules.ts` `GOAL_WIDTH`).
- Returns the opening angle subtended by the goal mouth at the observer's position.
- At `distance = 0`, the function returns 0.0 (guard against division by zero; the limit as distance approaches 0 is pi/2, but at the exact center of the goal line the angle is undefined, so 0.0 is the chosen sentinel).
- Maximum value approaches `2 * atan(0.14 / 0) = pi/2` as distance approaches 0 from the front.

Used in: PLAYER_GOAL and BALL_GOAL edges.

### 6.6 Nearest Opponent Pressure (PLAYER_GOAL)

```python
nearest_opponent_pressure = min(
    _euclidean(player.x, player.y, opponent.x, opponent.y)
    for opponent in opposing_team_players
    if opponent is not sentinel
) if opposing_team_players else 0.0
```

This is the Euclidean distance to the single closest opponent, not a binary indicator. It is 0.0 only when the opposing team has no present (non-sentinel) players.

Used in: PLAYER_GOAL edges only.

### 6.7 Deviation Distance (ASSIGNED_TO)

```python
deviation_distance = _euclidean(player.x, player.y, nominal_x, nominal_y)
```

Where `(nominal_x, nominal_y)` is the position of the nearest FORMATION_SLOT on the same team, computed from `Rules.ts getFormationPositions`.

---

## 7. Missing/Unavailable Feature Policy

### 7.1 Sentinel Positions

Players not present in the current scenario are encoded at position `(-1.0, -1.0)` in the 127-dim observation vector. The graph builder applies the following policy:

- **PLAYER nodes**: Sentinel-positioned players are included in the node list with `line_id = 0`, `lane_id = 0`, and all `features` fields set to 0.0. This preserves deterministic node ordering (left_0..left_N, right_0..right_M) regardless of scenario.
- **Proximity edges (TEAMMATE, OPPONENT, NEAR)**: Sentinel players participate in edge construction because their positions are read from the node dict. However, if both endpoints of a TEAMMATE/OPPONENT pair are sentinels, the distance will be 0.0 (both at (-1.0, -1.0)). In practice, scenarios never have both teammates as sentinels simultaneously because the observation vector is sized to the scenario's `teamLeftPlayers` / `teamRightPlayers`. The `_count_active_players` function (line 638) filters by `x != -1.0 or y != -1.0` for team-shape computation, but the edge loops iterate over all nodes in the team lists.
- **PLAYER_BALL, PLAYER_GOAL, ASSIGNED_TO, BELONGS_TO_SHAPE, SCENARIO_CONTEXT**: These edges are **skipped** for sentinel-positioned players (checked via `x == -1.0 and y == -1.0` guard in each builder function).
- **POSSESSES**: The `exact_owner_id` heuristic only considers non-sentinel players on the owning team.

### 7.2 Exact Ball Ownership Unavailability

The 127-dim observation vector exposes ball ownership only at the team level (indices 94-96: `ball_owned_none`, `ball_owned_left`, `ball_owned_right`). The exact player index (`ball.ownerId` in `GameEngine.ts`) is not available in the Python observation path.

As a result:

- `has_possession` in PLAYER_BALL edges is `True` only for the player on the owning team with the minimum Euclidean distance to the ball center. This heuristic can disagree with engine ground truth in edge cases (e.g., two players equidistant, or a player diving to block).
- The `POSSESSES` edge `source` uses the same heuristic.
- The JSON schema documents `has_possession` source as `GameEngine.ts ball.ownerId`, but the Python path cannot access it.

### 7.3 Formation Data Unavailability

FORMATION_SLOT nodes, ASSIGNED_TO edges, and all formation-structural edges (FORMATION_ADJACENCY, FORMATION_LINE, FORMATION_LANE) are emitted only when the scenario has a `formation` field in the `SCENARIOS` dict. Academy scenarios without a formation field produce zero formation-related nodes and edges. This is by design: academy scenarios do not use formations.

When no FORMATION_SLOT nodes exist:
- `ASSIGNED_TO` edges are omitted entirely.
- `deviation_x`, `deviation_y`, `deviation_distance`, `nominal_x`, `nominal_y`, and `role` features do not appear in any edge.

### 7.4 Opponent Absence

When a team has no present players (e.g., `academy_empty_goal` has `teamRightPlayers = 0`):

- No OPPONENT edges are emitted (empty right team list).
- `nearest_opponent_pressure` in PLAYER_GOAL edges is 0.0 for all players.
- `nearest_opponent_dist` in PLAYER node features is 0.0.

### 7.5 Distance Zero Guard

The `angle` and `closing_speed` features have a numerical guard against division by zero when `distance < 1e-6`:

- `angle` is set to 0.0.
- `closing_speed` is set to 0.0.
- `shot_angle` is set to 0.0 when `distance <= 0.0`.

These are implementation-defined sentinel values. A consuming GNN should not attribute semantic meaning to `angle = 0.0` or `closing_speed = 0.0` when the players are coincident; the zero indicates an undefined or degenerate geometry, not a meaningful bearing or closing rate.

### 7.6 Missing Feature Policy Summary

| Condition | Affected Features | Value |
|-----------|-------------------|-------|
| Player is sentinel (-1.0, -1.0) | All PLAYER-centric edge features | Edge omitted entirely |
| No FORMATION_SLOT nodes for scenario | ASSIGNED_TO features | Edge omitted entirely |
| No opposing team players | `nearest_opponent_pressure` | 0.0 |
| No opposing team players | `pressure` (OPPONENT edges) | No OPPONENT edges emitted |
| `ball_ownership == "none"` | POSSESSES edge, `has_possession` | POSSESSES edge omitted; `has_possession` = false for all PLAYER_BALL edges |
| Distance < 1e-6 (coincident endpoints) | `angle`, `closing_speed`, `shot_angle` | 0.0 (implementation guard) |
