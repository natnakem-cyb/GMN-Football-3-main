# GNN Phase 3 Graph Semantic Specification

## 1. Overview

This document defines the semantics of the Phase 3 GNN graph produced by `training/gnn_graph_builder.py:1494` (`build_graph`). The graph is validated against `training/gnn_graph_schema.json` (v3) and exercised by `training/tests/test_gnn_graph_builder.py`.

The graph is constructed from a single 127-dimensional observation vector plus an optional per-agent `info` dict and a `scenario_id` string. No engine state beyond what is derivable from the 127-dim observation is used. The bridge binary protocol, MAPPO code, reward code, and raw vector encoding are all untouched.

The graph has the top-level shape:

```
graph = {
    "scenario": <SCENARIO node>,   # top-level key, not in graph["nodes"]
    "z_scenario": list[float] | None,  # optional task context vector
    "nodes": [<PLAYER>, <BALL>, <GOAL>, <TEAM_SHAPE>, <FORMATION_SLOT>],
    "edges": [<TEAMMATE>, <OPPONENT>, <NEAR>, <POSSESSES>, ...]
}
```

## 2. Node Dictionary

| Node Type | Identity | Features | Source | Dynamic? |
|-----------|----------|----------|--------|----------|
| **PLAYER** | `global_id` = `left_0`..`left_10`, `right_0`..`right_10`. Determined by team + 0-based index within the team array. | `position` {x, y}, `velocity` {vx, vy}, `role` (12-way enum), `role_one_hot` (12 floats), `is_active` (bool), `is_controlled` (bool), `is_goalkeeper` (bool), `line_id` (int), `lane_id` (int), `features` (object): `nearest_teammate_dist`, `nearest_opponent_dist`, `team_width`, `team_depth`, `compactness`, `stretch`, `receiver_availability`. | 127-dim observation vector (positions at indices 0-21 / 44-65, velocities at 22-43 / 66-87, active one-hot at 97-107, role one-hot at 115-126). Scenario config for role/controlled assignment. | Yes. Positions, velocities, ownership, line_id, lane_id, and all derived features change every tick. |
| **BALL** | `node_id` = `ball` (constant string). | `position` {x, y, z}, `velocity` {vx, vy, vz}, `ownership` (`none` / `left` / `right`), `speed` (float = euclidean norm of velocity). | Observation indices 88-93. Ownership from one-hot at 94-96. | Yes. Position, velocity, ownership, and speed change every tick. |
| **GOAL** | `node_id` = `goal_left` (x = -1.0, team = left) or `goal_right` (x = 1.0, team = right). | `position` {x, y} (constant), `width` = 0.14, `height` = 0.05, `depth` = 0.04. | `Rules.ts` pitch geometry (constants). | No. Goals are static landmarks. |
| **FORMATION_SLOT** | `slot_id` = `left_slot0`..`left_slotN-1`, `right_slot0`..`right_slotM-1`. Deterministic index in the formation template array. | `role` (12-way enum), `xRatio`, `yRatio` (template ratios), `x`, `y` (absolute pitch positions derived from `getFormationPositions`). | Embedded `FORMATIONS` table (mirrors `Rules.ts`), scenario `formation` field in `SCENARIOS` dict. | No. Template slots are static for a given scenario. |
| **TEAM_SHAPE** | Implicit `node_id` = `team_shape_{team}` where team is `left` or `right`. No explicit `node_id` field in the dict; identity is the target string used in edges. | `team` (`left` / `right`), `formation_deviation` (object): `inter_line_spacing_variance` (float >= 0), `mean_line_compactness` (float [0, 1]), `num_lines` (int >= 1), `num_lanes` (int >= 1). | Computed from live PLAYER positions and `line_id` assignments via `_compute_team_shape` (`gnn_graph_builder.py:440`). | Yes. Formation deviation metrics change every tick as players move. |
| **SCENARIO** | `node_id` = `scenario` (constant string). Lives at `graph["scenario"]`, not inside `graph["nodes"]`. | `id` (scenario string), `time_limit_seconds` (float), `terminate_on_opponent_possession` (bool), `objectives` (array of `{id, text, is_completed, is_failed}`), `rewards` (`{scoring, completion}`), `objective_vocabulary` (10-dim multi-hot), `reward_scoring` (float), `reward_completion` (float). | `SCENARIOS` dict (embedded `ScenarioRegistry.ts` data), `info["score"]` from engine. | Static for the scenario definition. `is_completed` / `is_failed` are always `False` at graph construction time. |

## 3. Edge Dictionary

| Edge Type | Source | Target | Meaning | Features | Directionality |
|-----------|--------|--------|---------|----------|----------------|
| **TEAMMATE** | `global_id` of a PLAYER on team A | `global_id` of a different PLAYER on the same team A | Teammate spatial/velocity relation. | `distance` (float >= 0), `relative_x`, `relative_y` (target - source), `relative_vx`, `relative_vy` (velocity delta), `angle` (bearing from source to target in radians), `closing_speed` (rate of change of distance). | Undirected by construction: each unordered teammate pair yields exactly one edge. Source/target ordering is deterministic (sorted by `global_id` lexicographically). |
| **OPPONENT** | `global_id` of a PLAYER on team A | `global_id` of a PLAYER on the opposite team B | Opponent spatial/velocity relation plus pressure signal. | Same as TEAMMATE plus `pressure` (1.0 if distance < `NEAR_THRESHOLD` = 0.065, else 0.0). | Directed (left -> right only). Every cross-team pair present in the observation yields exactly one OPPONENT edge. |
| **NEAR** | `global_id` of PLAYER on team A | `global_id` of PLAYER on team B | Cross-team proximity indicator. | `distance` (float >= 0), `threshold` = 0.065 (constant). | Undirected by construction; emitted only when distance < 0.065. |
| **POSSESSES** | `global_id` of the PLAYER on the owning team closest to the ball | `ball` | Indicates which player currently possesses the ball. | No extra fields beyond `edge_type`, `source`, `target`. | Directed (player -> ball). Omitted entirely when `ownership == "none"`. |
| **PLAYER_BALL** | `global_id` of a present PLAYER | `ball` | Per-player ball proximity and possession state. | `distance`, `relative_x`, `relative_y`, `relative_vx`, `relative_vy`, `angle`, `has_possession` (bool, true only for the exact owner). | Directed (player -> ball). One edge per present player. |
| **PLAYER_GOAL** | `global_id` of a present PLAYER | `goal_left` or `goal_right` | Per-player shooting geometry to each goal. | `distance`, `relative_x`, `relative_y`, `angle`, `shot_angle` (opening angle to goal mouth in radians), `nearest_opponent_pressure` (distance to closest opponent, 0 if none present). | Directed (player -> goal). Two edges per present player (one per goal). |
| **BALL_GOAL** | `ball` | `goal_left` or `goal_right` | Ball shooting geometry to each goal. | `distance`, `relative_x`, `relative_y`, `angle`, `shot_angle`. | Directed (ball -> goal). Exactly two edges (one per goal). |
| **ASSIGNED_TO** | `global_id` of a present PLAYER | `slot_id` of nearest FORMATION_SLOT on same team | Player-to-template assignment deviation. | `role` (slot role), `nominal_x`, `nominal_y` (slot position), `deviation_x`, `deviation_y` (actual - nominal), `deviation_distance` (euclidean). | Directed (player -> slot). One edge per present player when FORMATION_SLOT nodes exist for the scenario. |
| **BELONGS_TO_SHAPE** | `global_id` of a present PLAYER | `team_shape_{team}` | Player membership in team shape summary node. | `team` (`left` / `right`). | Directed (player -> team shape). One edge per present player. |
| **SCENARIO_CONTEXT** | `global_id` of a present PLAYER | `scenario` | Attaches each player to the global scenario context node. | No extra fields. | Directed (player -> scenario). One edge per present player. |
| **FORMATION_ADJACENCY** | `slot_id` of a FORMATION_SLOT | `slot_id` of another FORMATION_SLOT on the same team | k=2 nearest neighbors in template (xRatio, yRatio) space. | No extra fields. | Undirected by construction. Each slot connects to its 2 nearest neighbors. |
| **FORMATION_LINE** | `slot_id` of a FORMATION_SLOT | `slot_id` of another FORMATION_SLOT on the same team within the same line band | Intra-line structural relation. | `line_id` (0 = GK, 1 = Defense, 2 = Midfield, 3 = Attack). Derived from xRatio thresholds: GK < 0.10, Defense < 0.35, Midfield < 0.60, else Attack. | Undirected by construction. All pairs within the same line band are connected. |
| **FORMATION_LANE** | `slot_id` of a FORMATION_SLOT | `slot_id` of another FORMATION_SLOT on the same team within the same lane band | Intra-lane structural relation. | `lane_id` (0 = Left, 1 = Center, 2 = Right). Derived from yRatio thresholds: Left < 0.33, Center <= 0.67, else Right. | Undirected by construction. All pairs within the same lane band are connected. |

## 4. Topology / Connectivity Diagram

The graph connectivity can be described textually as follows:

- **PLAYER cluster:** Every present PLAYER is connected to every other present PLAYER on the same team via exactly one TEAMMATE edge, and to every present PLAYER on the opposite team via exactly one OPPONENT edge. Cross-team pairs with Euclidean distance < 0.065 additionally receive a NEAR edge. TEAMMATE and OPPONENT edges are always present; NEAR edges are conditional.

- **PLAYER <-> BALL:** Every present PLAYER receives one PLAYER_BALL edge pointing to the `ball` node. If the ball is owned (ownership != "none"), the owning team's closest PLAYER to the ball receives one additional POSSESSES edge pointing to `ball`.

- **PLAYER <-> GOAL:** Every present PLAYER receives two PLAYER_GOAL edges, one to `goal_left` and one to `goal_right`.

- **BALL <-> GOAL:** The `ball` node receives two BALL_GOAL edges, one to each goal.

- **PLAYER <-> FORMATION_SLOT:** When FORMATION_SLOT nodes exist for the scenario, every present PLAYER receives one ASSIGNED_TO edge to the nearest FORMATION_SLOT on the same team.

- **PLAYER <-> TEAM_SHAPE:** Every present PLAYER receives one BELONGS_TO_SHAPE edge to `team_shape_{team}`.

- **PLAYER <-> SCENARIO:** Every present PLAYER receives one SCENARIO_CONTEXT edge to the `scenario` node.

- **FORMATION_SLOT <-> FORMATION_SLOT:** FORMATION_SLOT nodes on the same team are connected by three types of structural edges: FORMATION_ADJACENCY (k=2 nearest in template space), FORMATION_LINE (same xRatio line band), and FORMATION_LANE (same yRatio lane band).

- **SCENARIO node:** The `scenario` node has no parent edge from outside the PLAYER cluster; it is reachable only through SCENARIO_CONTEXT edges from present players. It is stored at `graph["scenario"]` and is not part of `graph["nodes"]`.

## 5. Data Flow (ScenarioRegistry -> Graph Builder -> Graph)

The data flow follows this pipeline:

1. **ScenarioRegistry.ts** defines scenario metadata: `timeLimitSeconds`, `terminateOnOpponentPossession`, `objectives`, `rewards`, `teamLeftPlayers`, `teamRightPlayers`, `setup` (player roles and controlled flags), and optionally `formation`.

2. **Embedded SCENARIOS dict** (`training/gnn_graph_builder.py:48`): This Python-side mirror of `ScenarioRegistry.ts` is used by the graph builder. It contains the authoritative scenario definitions for graph construction. The `formation` field is present only for scenarios that use formation templates (currently `11_vs_11`).

3. **127-dim observation vector** arrives from `GMNMultiAgentEnv.step()` / `reset()` as `observation["observation"]` (shape `(127,)`). The vector is parsed by `_parse_observation` (`gnn_graph_builder.py:555`) into structured fields: positions, velocities, ball ownership, active index, game mode, and viewpoint role. The raw 127-float array is never modified.

4. **Optional z_scenario vector** (`list[float]`) is passed directly through to `graph["z_scenario"]` without entering the node/edge schema.

5. **Node construction** proceeds in deterministic order:
   - PLAYER nodes (`_build_player_nodes`, line 720)
   - BALL node (`_build_ball_node`, line 916)
   - GOAL nodes (`_build_goal_nodes`, line 937)
   - TEAM_SHAPE nodes (`_build_team_shape_nodes`, line 1014)
   - FORMATION_SLOT nodes (`_build_formation_slot_nodes`, line 1035)

6. **Edge construction** (`_build_edges`, line 1314) uses the assembled nodes and parsed observation to emit all edge types. Edges are sorted lexicographically by `(edge_type, source, target)` for determinism (`gnn_graph_builder.py:1386`).

7. **Schema validation** (`jsonschema.validate`) runs at the end of `build_graph` (`gnn_graph_builder.py:1585`). Invalid graphs raise immediately.

8. **Graph assembly** (`build_graph`, line 1574): `scenario` node is placed at the top-level `graph["scenario"]` key; `nodes` and `edges` are placed in their respective arrays. Optional `z_scenario` is attached if provided.

## 6. Separation of Concerns

- **rawVector (127 floats):** The observation vector passed to `build_graph` is never modified. It is read-only input. Its structure is fixed by `ObservationEncoder.ts` and documented in `gnn_graph_schema.json` per-field descriptions. The graph builder does not write back to the observation.

- **zScenario:** When provided as the `z_scenario` parameter, it is attached verbatim as `graph["z_scenario"]` (list of floats). It is not incorporated into any node or edge feature, nor is it validated by the JSON schema. It is intended as a Phase 4 task-context vector consumed alongside the graph by the GNN. It is entirely separate from the 127-dim observation.

- **Graph nodes and edges:** These are the only schema-validated artifacts. They are derived exclusively from the parsed 127-dim observation, the embedded `SCENARIOS` dict, and (for formation) the embedded `FORMATIONS` table. No bridge binary data, no MAPPO state, and no reward shaping state touches the graph.

- **Bridge binary protocol:** Unchanged. The graph builder operates on the Python-side `observation` dict produced after bridge decoding. Binary frame parsing is unaffected.

## 7. Known Limitations

- **Exact ball ownership unavailable in Python path:** The 127-dim observation vector exposes ball ownership only at the team level (`ball_owned_left`, `ball_owned_right`, `ball_owned_none` at indices 94-96). The exact player index (`ball.ownerId` in the engine) is not present in the Python observation. Therefore, the `POSSESSES` edge source is computed by a closest-player heuristic: the player on the owning team with the minimum Euclidean distance to the ball center is selected as the possessor (`gnn_graph_builder.py:1517-1535`). This means `has_possession` in PLAYER_BALL edges and the `POSSESSES` edge source can disagree with the engine ground truth in edge cases (e.g., two players equidistant, or a player diving to block). The schema documents the true source as `GameEngine.ts ball.ownerId`, but the Python path cannot access it.

- **Formation nodes only where template exists:** FORMATION_SLOT, ASSIGNED_TO, and all formation-structural edges (FORMATION_ADJACENCY, FORMATION_LINE, FORMATION_LANE) are only emitted when the scenario has a `formation` field in the `SCENARIOS` dict. Academy scenarios with empty `setup` arrays and no `formation` field produce zero FORMATION_SLOT nodes and zero formation edges. This is by design: academy scenarios do not use formations, so template-based semantics would be meaningless.

- **No fake touch counts or sequence states:** The graph does not contain any synthetic touch counters, pass sequences, or historical state. All features are computed from the current tick's observation only.

- **Sentinel positions:** Players not present in the scenario are encoded at position `(-1.0, -1.0)` in the observation. The graph builder skips sentinel-positioned players for proximity edges (NEAR, ASSIGNED_TO, BELONGS_TO_SHAPE, SCENARIO_CONTEXT) and populates their PLAYER features with zeros. However, PLAYER nodes for absent slots are still included in the node list with `line_id = 0` and `lane_id = 0` to preserve deterministic ordering.

- **Objective completion flags:** `is_completed` and `is_failed` in the SCENARIO node's objectives are always `False` at graph construction time. They are set by `GameEngine.ts evaluateScenarioConditions()` during live play. The graph builder does not evaluate objectives.

- **Determinism depends on stable sort keys:** Node ordering is PLAYER (left_0..left_N, right_0..right_M), BALL, GOAL, TEAM_SHAPE, FORMATION_SLOT. Edge ordering is `(edge_type, source, target)` lexicographic. Any change to `global_id` format or `edge_type` strings would break determinism guarantees verified by `test_determinism` and `test_edge_ordering_deterministic` in `training/tests/test_gnn_graph_builder.py`.

## 8. What Phase 4 GNN Will Consume

The Phase 4 GNN model will consume the graph as follows:

- **Input nodes:** The `nodes` array provides typed node embeddings. PLAYER nodes carry the richest feature set (position, velocity, role, tactical flags, and 7 derived spatial features). BALL and GOAL nodes provide fixed geometry. TEAM_SHAPE nodes provide team-level formation-deviation summary statistics. FORMATION_SLOT nodes provide template anchors.

- **Input edges:** The `edges` array provides typed edges with continuous relational features. The GNN message-passing layers will use edge features such as `distance`, `relative_x/y`, `angle`, `closing_speed`, `pressure`, `shot_angle`, and `deviation_distance` to compute spatially-informed node updates.

- **Graph-level context:** The `graph["scenario"]` node provides task metadata (objectives, rewards, time limit, termination condition) that the GNN can use for conditioning. The `graph["z_scenario"]` vector (when present) provides an additional task-context signal that is orthogonal to the scenario metadata and is not part of the JSON schema.

- **What will NOT change:** The raw 127-dim observation vector is untouched. The bridge binary protocol is untouched. MAPPO policy and value networks are untouched. Reward shaping code is untouched. No new synthetic features (touch counts, pass sequences, etc.) will be added in Phase 4 without a corresponding spec revision.

- **Schema stability:** The v3 schema is the contract. Any Phase 4 GNN implementation must validate its consumed graphs against `training/gnn_graph_schema.json`. Adding new node types, edge types, or features requires a schema version bump and a corresponding update to this document.
