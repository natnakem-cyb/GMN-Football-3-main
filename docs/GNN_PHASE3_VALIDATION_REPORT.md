# GNN Phase 3 Validation Report

## 1. Summary

Phase 3 semantic integration is complete. The graph builder (`training/gnn_graph_builder.py`) now emits a connected, semantically rich graph representation from the 127-dim observation vector. The schema has been updated to v3 in `training/gnn_graph_schema.json`. All 24 tests in `training/tests/test_gnn_graph_builder.py` pass, including 10 new Phase 3 tests. The TypeScript project (`npm run lint`, `npm run build`) and the 15-scenario regression suite (`npm run test`) continue to pass unchanged. The bridge-protocol validation also passes.

The answer to the Phase 3 question — **"Can a future GNN now receive a connected representation of players, ball, goals, formation, team shape, and scenario context?"** — is **yes**. All five entity classes are present as nodes, and cross-entity edges (PLAYER_BALL, PLAYER_GOAL, BALL_GOAL, ASSIGNED_TO, BELONGS_TO_SHAPE, SCENARIO_CONTEXT, FORMATION_*) connect them into a single graph.

---

## 2. Files Changed

| File | Change |
|------|--------|
| `training/gnn_graph_schema.json` | Updated to **v3**. Added new node types (`TEAM_SHAPE`, `FORMATION_SLOT`, `SCENARIO`), new edge types, and new edge feature definitions. |
| `training/gnn_graph_builder.py` | Added Phase 3 semantic integration: ball/goal geometry, formation slot nodes, team-shape nodes, scenario-context edges, possesses-edge ownership heuristic, and `z_scenario` pass-through. |
| `training/tests/test_gnn_graph_builder.py` | Added 10 new tests covering all Phase 3 edge types and the `z_scenario` parameter. |

No files in `src/`, `training/train_mappo.py`, `training/gmn_pettingzoo.py`, or the reward/bridge pipeline were modified.

---

## 3. Exact Graph Schema Changes (v2 → v3)

### New Node Types
- `TEAM_SHAPE`
- `FORMATION_SLOT`
- `SCENARIO` (lives at `graph["scenario"]`, not in `graph["nodes"]`)

### New Edge Types
- `PLAYER_BALL`
- `PLAYER_GOAL`
- `BALL_GOAL`
- `ASSIGNED_TO`
- `BELONGS_TO_SHAPE`
- `SCENARIO_CONTEXT`
- `FORMATION_ADJACENCY`
- `FORMATION_LINE`
- `FORMATION_LANE`

### New Edge Features
- `distance`, `relative_x`, `relative_y`, `relative_vx`, `relative_vy`, `angle`, `closing_speed` (TEAMMATE)
- `pressure` (OPPONENT)
- `has_possession` (PLAYER_BALL)
- `shot_angle`, `nearest_opponent_pressure` (PLAYER_GOAL)
- `shot_angle` (BALL_GOAL)
- `role`, `nominal_x`, `nominal_y`, `deviation_x`, `deviation_y`, `deviation_distance` (ASSIGNED_TO)
- `team` (BELONGS_TO_SHAPE)
- `line_id` (FORMATION_LINE)
- `lane_id` (FORMATION_LANE)

---

## 4. New Node Types

### TEAM_SHAPE
- `node_id`: `team_shape_{team}` where `team` is `left` or `right`.
- Features: `team`, `formation_deviation` (`inter_line_spacing_variance`, `mean_line_compactness`, `num_lines`, `num_lanes`).
- Computed from live PLAYER positions and `line_id` assignments via `_compute_team_shape`.
- Dynamic: changes every tick.

### FORMATION_SLOT
- `node_id`: `{team}_slot{i}` where `team` is `left` or `right` and `i` is the 0-based template index.
- Features: `role` (12-way enum), `xRatio`, `yRatio` (template ratios), `x`, `y` (absolute pitch positions from `Rules.ts getFormationPositions`).
- Static for a given scenario.
- Omitted entirely for academy scenarios without a `formation` field.

### SCENARIO
- `node_id`: `scenario` (constant string).
- Lives at `graph["scenario"]` (top-level key), not in `graph["nodes"]`.
- Features: `id`, `time_limit_seconds`, `terminate_on_opponent_possession`, `objectives`, `rewards`, `objective_vocabulary`, `reward_scoring`, `reward_completion`.
- Static for the scenario definition.

---

## 5. New Edge Types

| Edge Type | Source | Target | Directionality | Features |
|-----------|--------|--------|----------------|----------|
| `PLAYER_BALL` | present PLAYER | `ball` | Directed | `distance`, `relative_x`, `relative_y`, `relative_vx`, `relative_vy`, `angle`, `has_possession` |
| `PLAYER_GOAL` | present PLAYER | `goal_left` / `goal_right` | Directed | `distance`, `relative_x`, `relative_y`, `angle`, `shot_angle`, `nearest_opponent_pressure` |
| `BALL_GOAL` | `ball` | `goal_left` / `goal_right` | Directed | `distance`, `relative_x`, `relative_y`, `angle`, `shot_angle` |
| `ASSIGNED_TO` | present PLAYER | nearest FORMATION_SLOT (same team) | Directed | `role`, `nominal_x`, `nominal_y`, `deviation_x`, `deviation_y`, `deviation_distance` |
| `BELONGS_TO_SHAPE` | present PLAYER | `team_shape_{team}` | Directed | `team` |
| `SCENARIO_CONTEXT` | present PLAYER | `scenario` | Directed | *(none)* |
| `FORMATION_ADJACENCY` | FORMATION_SLOT | FORMATION_SLOT (k=2 nearest in template space) | Undirected | *(none)* |
| `FORMATION_LINE` | FORMATION_SLOT | FORMATION_SLOT (same line band) | Undirected | `line_id` |
| `FORMATION_LANE` | FORMATION_SLOT | FORMATION_SLOT (same lane band) | Undirected | `lane_id` |

---

## 6. New Edge Features

### Passing Geometry (`PLAYER_BALL`, `TEAMMATE`, `OPPONENT`)
- `distance`: Euclidean distance in pitch units.
- `relative_x`, `relative_y`: target.x - source.x, target.y - source.y.
- `relative_vx`, `relative_vy`: target velocity - source velocity.
- `angle`: `atan2(dy, dx)` in radians, bearing from source to target.
- `closing_speed`: `-(dx*dvx + dy*dvy) / distance`. Negative = approaching.

### Shooting Geometry (`PLAYER_GOAL`, `BALL_GOAL`)
- `shot_angle`: `2 * atan(goalWidth / (2 * distance))`. Opening angle subtended by the goal mouth.
- `nearest_opponent_pressure` (PLAYER_GOAL only): Euclidean distance to the closest opponent, or 0.0 if none present.

### Pressure (`OPPONENT`)
- `pressure`: `1.0` if `distance < NEAR_THRESHOLD` (0.065), else `0.0`. Binary indicator aligned with the tackle threshold in `Physics.ts`.

### Formation Deviation (`ASSIGNED_TO`)
- `deviation_x`, `deviation_y`, `deviation_distance`: player position minus nearest FORMATION_SLOT nominal position.
- `nominal_x`, `nominal_y`: absolute pitch position of the matched slot from `Rules.ts getFormationPositions`.
- `role`: tactical role label of the matched slot.

### Team Membership (`BELONGS_TO_SHAPE`)
- `team`: `"left"` or `"right"`. Identifies which TEAM_SHAPE node is the target.

### Formation Structure (`FORMATION_LINE`, `FORMATION_LANE`)
- `line_id`: 0=GK, 1=Defense, 2=Midfield, 3=Attack. Derived from `xRatio` thresholds.
- `lane_id`: 0=Left, 1=Center, 2=Right. Derived from `yRatio` thresholds.

---

## 7. Formation Integration

FORMATION_SLOT nodes and all formation-structural edges (`ASSIGNED_TO`, `FORMATION_ADJACENCY`, `FORMATION_LINE`, `FORMATION_LANE`) are emitted only when the scenario has a `formation` field in the `SCENARIOS` dict (currently `11_vs_11` only). Academy scenarios without a formation field produce zero formation-related nodes and edges.

- `ASSIGNED_TO`: One edge per present PLAYER to the nearest FORMATION_SLOT on the same team (Euclidean distance in pitch space). Omitted if no FORMATION_SLOT nodes exist.
- `FORMATION_ADJACENCY`: k=2 nearest neighbors in template (xRatio, yRatio) space.
- `FORMATION_LINE`: All pairs of slots within the same line band on the same team.
- `FORMATION_LANE`: All pairs of slots within the same lane band on the same team.

All formation-structural edges are undirected by construction and stored with lexicographic slot_id ordering.

---

## 8. Team Shape Integration

TEAM_SHAPE nodes (`team_shape_left`, `team_shape_right`) are always present for any scenario with at least one player on the team. `BELONGS_TO_SHAPE` edges connect every present PLAYER to its team's TEAM_SHAPE node.

TEAM_SHAPE features (`formation_deviation`) are computed from live PLAYER positions and `line_id` assignments via `_compute_team_shape` (`gnn_graph_builder.py:440`). These metrics change every tick as players move, giving the GNN a summary of team shape dynamics without manual feature engineering.

---

## 9. Ball/Goal Integration

- `PLAYER_BALL` edges: One directed edge from every present PLAYER to the `ball` node. Includes `has_possession` (boolean, true only for the exact owner).
- `POSSESSES` edges: Directed edge from the owning team's closest PLAYER to `ball`. Omitted when `ball_ownership == "none"`.
- `PLAYER_GOAL` edges: Two directed edges per present PLAYER (one to `goal_left`, one to `goal_right`). Includes `shot_angle` and `nearest_opponent_pressure`.
- `BALL_GOAL` edges: Two directed edges from `ball` to each goal. Includes `shot_angle`.

Goals are static landmarks at `goal_left` (x = -1.0) and `goal_right` (x = 1.0). Their node features include constant pitch geometry (`width` = 0.14, `height` = 0.05, `depth` = 0.04).

---

## 10. Passing Geometry

All player-to-player edges (`TEAMMATE`, `OPPONENT`) now carry continuous relational geometry:

- `distance`: Euclidean distance.
- `relative_x`, `relative_y`: position delta.
- `relative_vx`, `relative_vy`: velocity delta.
- `angle`: bearing from source to target in radians.
- `closing_speed`: rate of change of inter-player distance. Negative = approaching.

OPPONENT edges additionally carry `pressure` (1.0 if distance < 0.065, else 0.0).

---

## 11. Shooting Geometry

All ball/goal and player/goal edges carry `shot_angle`:

```python
shot_angle = 2.0 * math.atan(goal_width / (2.0 * distance))
```

where `goal_width = 0.14` (pitch units). Returns 0.0 when `distance <= 0.0` (guard against division by zero).

PLAYER_GOAL edges additionally carry `nearest_opponent_pressure`: the Euclidean distance from the player to the closest opponent, or 0.0 if no opponents are present.

---

## 12. Scenario Context Integration

`SCENARIO_CONTEXT` edges connect every present PLAYER to the `scenario` node (`graph["scenario"]`). This gives the GNN a single entry point to the global scenario metadata (objectives, rewards, time limit, etc.) from any player's perspective.

The `z_scenario` parameter is attached verbatim as `graph["z_scenario"]` (list of floats) without entering the node/edge schema. It is intended as a Phase 4 task-context vector consumed alongside the graph.

---

## 13. Tests Added

10 new tests were added to `training/tests/test_gnn_graph_builder.py`:

1. **`test_rich_teammate_edges_have_geometry`** (line 476) — TEAMMATE edges carry continuous relational geometry (`distance`, `relative_x`, `relative_y`, `relative_vx`, `relative_vy`, `angle`, `closing_speed`).
2. **`test_rich_opponent_edges_have_geometry_and_pressure`** (line 493) — OPPONENT edges carry continuous geometry plus binary `pressure` signal (must be 0.0 or 1.0).
3. **`test_player_ball_edges_exist_and_have_geometry`** (line 510) — PLAYER_BALL edges exist for every present player to `ball`, with geometry and `has_possession`.
4. **`test_player_goal_edges_exist_and_have_geometry`** (line 530) — PLAYER_GOAL edges exist for every present player to both goals, with `shot_angle` and `nearest_opponent_pressure`.
5. **`test_ball_goal_edges_exist`** (line 552) — BALL_GOAL edges exist from `ball` to both goals, with `shot_angle`.
6. **`test_formation_slot_connectivity_all_scenarios`** (line 570) — ASSIGNED_TO edges exist for any scenario with FORMATION_SLOT nodes; absent when no formation is defined.
7. **`test_belongs_to_shape_edges_exist`** (line 596) — BELONGS_TO_SHAPE edges connect every present player to the correct TEAM_SHAPE node.
8. **`test_scenario_context_edges_exist`** (line 614) — SCENARIO_CONTEXT edges connect every present player to the `scenario` node.
9. **`test_z_scenario_support`** (line 630) — `z_scenario` is attached verbatim to the graph when provided.
10. **`test_possesses_edge_matches_ownership`** (line 643) — POSSESSES edge source belongs to the owning team.

**Total tests: 24 passed** (14 existing + 10 new).

---

## 14. Existing Tests Passed

| Command | Result |
|---------|--------|
| `python -m pytest training/tests/test_gnn_graph_builder.py -v` | **24 passed** |
| `npm run lint` | **passed** |
| `npm run build` | **passed** |
| `npm run test` | **15/15 scenarios passed**, determinism passed |
| `npm run test:bridge-protocol` | **all checks passed** |

No existing tests were broken by the Phase 3 changes.

---

## 15. Known Unavailable Information

The following data is **not available** in the Python observation path and is handled via heuristics or omitted:

- **Exact ball owner ID**: The 127-dim observation vector exposes ball ownership only at the team level (`ball_owned_none`, `ball_owned_left`, `ball_owned_right`). The exact player index (`ball.ownerId` in `GameEngine.ts`) is not available. The graph builder uses a heuristic: the player on the owning team with the minimum Euclidean distance to the ball center.
- **Formation data for academy scenarios**: Academy scenarios without a `formation` field produce zero formation-related nodes and edges. This is by design.
- **Opponent absence**: When a team has no present players, `nearest_opponent_pressure` is 0.0 and no OPPONENT edges are emitted.

---

## 16. Performance Measurements

Not measured in this phase. The graph builder processes a single 127-dim observation vector per agent per tick. No profiling or timing benchmarks were collected during Phase 3.

---

## 17. Remaining Limitations

1. **Heuristic ball ownership**: `has_possession` in PLAYER_BALL edges and the POSSESSES edge source use a distance heuristic, not engine ground truth. This can disagree with `GameEngine.ts ball.ownerId` in edge cases (e.g., two players equidistant, diving to block).
2. **No normalization**: All edge features are emitted in raw pitch-normalized units. Normalization, if required, is the responsibility of the consuming Phase 4 GNN model.
3. **Sentinel players**: Players not present in the current scenario are encoded at position `(-1.0, -1.0)`. PLAYER-centric edges (PLAYER_BALL, PLAYER_GOAL, ASSIGNED_TO, BELONGS_TO_SHAPE, SCENARIO_CONTEXT) are skipped for sentinel players, but TEAMMATE/OPPONENT/NEAR edges are still constructed with sentinel endpoints (distance will be 0.0 for sentinel-sentinel pairs).
4. **Academy scenarios lack formation/tactical structure**: Only scenarios with a `formation` field emit FORMATION_SLOT nodes and formation-structural edges.
5. **Scenario node excluded from `graph["nodes"]`**: The `scenario` node lives at `graph["scenario"]` and is not included in the `nodes` array. Phase 4 code must account for this special placement.

---

## 18. Exact Interface That Phase 4 GNN Encoder Will Consume

The Phase 4 GNN encoder will consume the following Python dict structure produced by `build_graph(observation, info, scenario_id, z_scenario=None)`:

```python
graph = {
    "scenario": {
        "id": str,
        "time_limit_seconds": float,
        "terminate_on_opponent_possession": bool,
        "objectives": list[dict],
        "rewards": {"scoring": float, "completion": float},
        "objective_vocabulary": list[float],   # 10-dim multi-hot
        "reward_scoring": float,
        "reward_completion": float,
    },
    "z_scenario": list[float] | None,         # optional task context vector
    "nodes": [
        {
            "node_type": "PLAYER" | "BALL" | "GOAL" | "TEAM_SHAPE" | "FORMATION_SLOT",
            "global_id" | "node_id" | "slot_id": str,
            "position": {"x": float, "y": float, "z?": float},
            "velocity": {"vx": float, "vy": float, "vz?": float},
            # PLAYER-specific:
            "role": str,
            "role_one_hot": list[float],
            "is_active": bool,
            "is_controlled": bool,
            "is_goalkeeper": bool,
            "line_id": int,
            "lane_id": int,
            "features": {
                "nearest_teammate_dist": float,
                "nearest_opponent_dist": float,
                "team_width": float,
                "team_depth": float,
                "compactness": float,
                "stretch": float,
                "receiver_availability": float,
            },
            # BALL-specific:
            "ownership": str,   # "none" | "left" | "right"
            "speed": float,
            # GOAL-specific:
            "width": float,
            "height": float,
            "depth": float,
            # TEAM_SHAPE-specific:
            "team": str,
            "formation_deviation": dict,
            # FORMATION_SLOT-specific:
            "xRatio": float,
            "yRatio": float,
        },
    ],
    "edges": [
        {
            "edge_type": str,
            "source": str,
            "target": str,
            # TEAMMATE/OPPONENT:
            "distance": float,
            "relative_x": float,
            "relative_y": float,
            "relative_vx": float,
            "relative_vy": float,
            "angle": float,
            "closing_speed": float,
            # OPPONENT additionally:
            "pressure": float,
            # PLAYER_BALL:
            "has_possession": bool,
            # PLAYER_GOAL / BALL_GOAL:
            "shot_angle": float,
            # PLAYER_GOAL additionally:
            "nearest_opponent_pressure": float,
            # ASSIGNED_TO:
            "role": str,
            "nominal_x": float,
            "nominal_y": float,
            "deviation_x": float,
            "deviation_y": float,
            "deviation_distance": float,
            # BELONGS_TO_SHAPE:
            "team": str,
            # FORMATION_LINE:
            "line_id": int,
            # FORMATION_LANE:
            "lane_id": int,
        },
    ],
}
```

**Normalization contract**: All geometric features are in **raw pitch-normalized units** (x in [-1.0, 1.0], y in [-0.42, 0.42]). No normalization is applied by the graph builder. The Phase 4 GNN is responsible for any per-feature scaling, standardization, or learned normalization.

**Determinism contract**: Node ordering is fixed by team + index matching the 127-dim observation ordering. Edges are sorted lexicographically by `(edge_type, source, target)` at `gnn_graph_builder.py:1386`.

---

## 19. Final Answer to Phase 3 Question

**Can a future GNN now receive a connected representation of players, ball, goals, formation, team shape, and scenario context?**

**Yes.**

All five entity classes are present as graph nodes:
- **Players** → `PLAYER` nodes with position, velocity, role, and team-shape features.
- **Ball** → `BALL` node with position, velocity, ownership, and speed.
- **Goals** → `GOAL` nodes (static landmarks).
- **Formation** → `FORMATION_SLOT` nodes with template positions and roles.
- **Team shape** → `TEAM_SHAPE` nodes with live formation-deviation metrics.
- **Scenario context** → `SCENARIO` node with objectives, rewards, and time limit.

Cross-entity edges connect these nodes into a single, schema-validated graph:
- `PLAYER_BALL`, `PLAYER_GOAL`, `BALL_GOAL` connect the ball and goals to every player.
- `ASSIGNED_TO` connects players to their formation slots.
- `BELONGS_TO_SHAPE` connects players to team-shape summaries.
- `SCENARIO_CONTEXT` connects players to global scenario metadata.
- `FORMATION_ADJACENCY`, `FORMATION_LINE`, `FORMATION_LANE` provide structural connectivity within formation templates.

The graph is deterministic, schema-validated, and exercises the full Phase 3 test suite. Phase 4 can now implement a GNN encoder that consumes this graph directly.
