# GNN Phase 3 Formation Integration

## 1. Overview

Phase 3 of the GNN graph builder introduces formation-aware structure into the graph. Two new node types (`FORMATION_SLOT`, `TEAM_SHAPE`) and four new edge types (`ASSIGNED_TO`, `BELONGS_TO_SHAPE`, `FORMATION_ADJACENCY`, `FORMATION_LINE`, `FORMATION_LANE`) extend the schema beyond pure geometric observation encoding.

Formation integration serves two distinct purposes:
1. **Nominal shape**: `FORMATION_SLOT` nodes encode the *intended* tactical positions defined by the scenario's formation template. These are reference points, not observed state.
2. **Actual shape**: `TEAM_SHAPE` nodes and `BELONGS_TO_SHAPE` edges capture the *realized* geometric layout of present players via gap-detection line/lane assignments and derived compactness metrics.

The design intentionally separates nominal from actual shape so that a model can learn to compare intended vs. realized team structure.

## 2. Formation Template Source

The canonical formation definitions live in the TypeScript engine and are mirrored into Python at graph-build time.

**TypeScript source**: `src/engine/Rules.ts`

- `FORMATIONS` constant: lines 68–139. A `Record<FormationType, FormationNode[]>` mapping formation strings (e.g. `"4-3-3"`, `"4-4-2"`, `"3-5-2"`, `"5-3-2"`, `"1-2-1"`, `"1-1-1"`, `"1-0"`) to ordered arrays of `{ role, xRatio, yRatio }` objects.
- `getFormationPositions(formation, team, numPlayers)`: lines 141–168. Converts the ratio template into absolute pitch coordinates using the GMN standard coordinate system.

**Python mirror**: `training/gnn_graph_builder.py`

- `FORMATIONS` dict: lines 328–397. Embedded directly in the Python module to avoid runtime TS→Python coupling during graph construction. Values are verbatim copies of the `src/engine/Rules.ts` arrays.
- `_get_formation_slot_positions(formation, team, num_players)`: lines 691–712. Python reimplementation of `Rules.ts getFormationPositions()`.

Coordinate mapping in `_get_formation_slot_positions` (lines 699–704):

```python
if team == "left":
    x = -1.0 + x_ratio * 1.2
    y = -0.42 + y_ratio * 0.84
else:
    x = 1.0 - x_ratio * 1.2
    y = 0.42 - y_ratio * 0.84
```

This is equivalent to the TypeScript:
```typescript
x = PITCH.minX + node.xRatio * (PITCH.width * 0.6);   // -1.0 + xRatio * 1.2
y = PITCH.minY + node.yRatio * PITCH.height;           // -0.42 + yRatio * 0.84
```

## 3. Slot Generation Logic

Entry point: `_build_formation_slot_nodes(scenario_id)` at `training/gnn_graph_builder.py:1035`.

The function delegates to `_get_formation_positions_for_scenario(scenario_id)` (lines 520–547), which determines whether the scenario supports formation slots:

1. Look up the scenario in the embedded `SCENARIOS` dict (lines 48–322).
2. Read `scenario.get("formation")`. If missing or falsy, return `None`.
3. Read `teamLeftPlayers` and `teamRightPlayers` (defaults to 11).
4. Call `_get_formation_slot_positions(formation, "left", num_left)` and `_get_formation_slot_positions(formation, "right", num_right)`.
5. Stamp each slot with a deterministic `slot_id` (`left_slot0` … `left_slotN-1`, `right_slot0` … `right_slotM-1`) and a `team` field.
6. Return the concatenated list.

Each slot node (lines 1042–1052) contains:
- `node_type`: `"FORMATION_SLOT"`
- `slot_id`: e.g. `"left_slot0"`
- `team`: `"left"` or `"right"`
- `role`: from the `FORMATIONS` template
- `xRatio`, `yRatio`: template ratios (0–1)
- `x`, `y`: absolute pitch coordinates computed by `_get_formation_slot_positions`

The implementation is generic: **any** scenario with a `formation` key in `SCENARIOS` will receive formation slots, regardless of player count or formation name.

## 4. Player → Slot Assignment (ASSIGNED_TO Edge)

Builder: `_build_assigned_to_edges(player_nodes, formation_slot_nodes)` at `training/gnn_graph_builder.py:1230`.

This edge connects every *present* player to their nearest formation slot using a pure Euclidean nearest-neighbor heuristic.

Algorithm (lines 1239–1266):
1. Group `formation_slot_nodes` by `team` into `slots_by_team`.
2. For each player node:
   - Skip players with sentinel positions `(-1.0, -1.0)`.
   - Restrict candidate slots to the player's own team.
   - Find the slot minimizing `_euclidean(player.x, player.y, slot.x, slot.y)`.
   - Emit an `ASSIGNED_TO` edge with:
     - `source`: player `global_id` (e.g. `"left_2"`)
     - `target`: `slot_id` of the nearest slot
     - `role`: the slot's tactical role
     - `nominal_x`, `nominal_y`: slot's absolute position
     - `deviation_x`, `deviation_y`: `player.x - nominal_x`, `player.y - nominal_y`
     - `deviation_distance`: Euclidean distance between actual and nominal position

The edge is emitted in `_build_edges` at line 1346–1347, after all geometric player-to-player/ball/goal edges but before `BELONGS_TO_SHAPE` and formation-internal edges.

Schema reference: `training/gnn_graph_schema.json` lines 827–878 (`assigned_to_edge`).

## 5. Formation Internal Edges (FORMATION_ADJACENCY, FORMATION_LINE, FORMATION_LANE)

Builder: `_build_formation_edges(formation_slot_nodes)` at `training/gnn_graph_builder.py:1391`.

This function constructs three edge types within each team's set of formation slots:

### 5.1 FORMATION_LINE (lines 1409–1437)

Groups slots by `xRatio` bands:
- Line 0 (GK): `xRatio < 0.10`
- Line 1 (Defense): `0.10 <= xRatio < 0.35`
- Line 2 (Midfield): `0.35 <= xRatio < 0.60`
- Line 3 (Attack): `xRatio >= 0.60`

All unordered pairs within each line group receive a `FORMATION_LINE` edge carrying `line_id`.

### 5.2 FORMATION_LANE (lines 1439–1466)

Groups slots by `yRatio` bands:
- Lane 0 (Left): `yRatio < 0.33`
- Lane 1 (Center): `0.33 <= yRatio <= 0.67`
- Lane 2 (Right): `yRatio > 0.67`

All unordered pairs within each lane group receive a `FORMATION_LANE` edge carrying `lane_id`.

### 5.3 FORMATION_ADJACENCY (lines 1468–1483)

For each slot, compute Euclidean distance to every other slot in `(xRatio, yRatio)` space (not absolute pitch space). Connect each slot to its `k=2` nearest neighbors. This yields a sparse 2-nearest-neighbor graph over the template.

All three edge types are sorted deterministically at line 1485:
```python
edges.sort(key=lambda e: (e["edge_type"], e["source"], e.get("target", "")))
```

Schema references:
- `FORMATION_ADJACENCY`: `training/gnn_graph_schema.json` lines 930–949
- `FORMATION_LINE`: lines 951–975
- `FORMATION_LANE`: lines 977–1001

## 6. Current Coverage

The `SCENARIOS` dict is defined at `training/gnn_graph_builder.py:48–322`. Only the following entry has a `formation` field:

- `11_vs_11` (line 320): `"formation": "4-3-3"`

All other academy and small-sided scenarios lack the `formation` key:
- `academy_empty_goal`
- `academy_run_to_score`
- `academy_pass_and_shoot_with_keeper`
- `academy_3_vs_1_with_keeper`
- `academy_3_vs_1_defender_2`
- `academy_3_vs_1_defender_3`
- `academy_3_vs_1_keeper_aggressive`
- `academy_3_vs_1_shifted`
- `academy_3_vs_1_randomized`
- `academy_rondo_4v1`
- `5_vs_5`

Because `_get_formation_positions_for_scenario` returns `None` when the `formation` key is absent (line 528–530), `_build_formation_slot_nodes` returns an empty list (lines 1038–1039), and consequently `_build_assigned_to_edges` and `_build_formation_edges` produce no edges.

**Why academy scenarios lack formation slots:**
Academy scenarios use small, irregular rosters (1–5 players per side) with fixed starting positions and specialized roles that do not map cleanly to full 11v11 formation templates. The `_get_scenario_roles` function (lines 652–664) already infers roles from the scenario config `setup` arrays for these cases, so a formation template would be redundant or misleading.

**Why `5_vs_5` lacks formation slots:**
`5_vs_5` (lines 275–303) has a `setup` array with explicit roles (`GK`, `CB`, `LM`, `RM`, `ST`) but no `formation` field. Although `src/engine/Rules.ts` defines a `"1-2-1"` formation intended for 5v5, the Python `SCENARIOS` entry does not reference it.

## 7. Nominal vs. Actual Shape Distinction

The graph maintains two complementary representations of team structure:

### 7.1 Nominal Shape: FORMATION_SLOT Nodes

`FORMATION_SLOT` nodes are **static reference anchors** derived from the formation template. They represent *where players should be* according to tactical doctrine, not where they actually are at the current tick.

Properties:
- Positions are deterministic for a given `(scenario_id, tick)` because they depend only on the template and player counts.
- They are **not** updated by observation parsing; they are rebuilt fresh from the template on every `build_graph` call (line 1553).
- They carry the `role` from the template, which may differ from a player's actual `role` if the player has drifted or if the scenario config overrides the template role.

### 7.2 Actual Shape: TEAM_SHAPE Nodes and PLAYER line_id/lane_id

`TEAM_SHAPE` nodes (built at `training/gnn_graph_builder.py:1014`) summarize the *realized* geometry of present players:

- `line_id` and `lane_id` on each `PLAYER` node (lines 833–839) are computed via 1D gap-detection on the player's actual `x` and `y` coordinates using `LINE_LANE_GAP_THRESHOLD = 0.15` (line 38).
- `formation_deviation` on the `TEAM_SHAPE` node (lines 1026–1030) is computed by `_compute_team_shape` (lines 440–500) and includes:
  - `inter_line_spacing_variance`: variance of gaps between line centroids
  - `mean_line_compactness`: normalized width of each line
  - `num_lines`: number of detected lines from gap-detection
  - `num_lanes`: number of detected lanes from gap-detection

`BELONGS_TO_SHAPE` edges (lines 1271–1290) connect every present player to their team's `TEAM_SHAPE` node, making the team-level summary accessible to GNN message passing.

### 7.3 Relationship

The `ASSIGNED_TO` edge bridges nominal and actual by recording per-player deviation from the nearest nominal slot. A GNN can therefore:
- Read nominal role/tactical intent from the `FORMATION_SLOT` node and `ASSIGNED_TO.role`.
- Read actual spatial behavior from the `PLAYER` node's `position`, `line_id`, `lane_id`, and `features`.
- Read team-level summary from `TEAM_SHAPE.formation_deviation`.
- Compare actual vs. nominal via `ASSIGNED_TO.deviation_distance`, `deviation_x`, `deviation_y`.

## 8. Limitations and Future Work

### 8.1 Coverage is limited to 11_vs_11

Only `11_vs_11` carries a `formation` field. Academy scenarios and `5_vs_5` produce zero `FORMATION_SLOT` nodes. This means:
- No `ASSIGNED_TO` edges exist for ~90% of registered scenarios.
- `_build_formation_edges` produces no output for those scenarios.
- The `FORMATION_SLOT` node/edge types are effectively untrained on most of the scenario distribution.

**Future work:** Add `formation` fields to academy scenarios that have stable tactical structures, or define mini-formations (e.g. `"1-0"` for `academy_empty_goal`) so that the formation infrastructure is exercised across the full curriculum.

### 8.2 No dynamic formation switching

Formation is read once from the static `SCENARIOS` dict. There is no mechanism for in-episode formation changes (e.g. a team shifting from `"4-3-3"` to `"5-3-2"` at half-time).

**Future work:** If the engine exposes dynamic formation state, `build_graph` would need a `formation` parameter or a `_get_current_formation(scenario_id, tick)` hook.

### 8.3 Nearest-slot assignment is role-agnostic

`_build_assigned_to_edges` (line 1250–1253) assigns each player to the geometrically nearest slot, ignoring the player's actual `role` and the slot's `role`. A left winger assigned to a center-forward slot is valid if it is closer.

**Future work:** Add a role-aware fallback: prefer nearest slot with matching role, then fall back to geometric nearest if no match is within a radius threshold.

### 8.4 Formation edges are template-only, not observation-derived

`FORMATION_ADJACENCY`, `FORMATION_LINE`, and `FORMATION_LANE` are computed purely from the `xRatio`/`yRatio` values in the `FORMATIONS` template. They do not reflect the actual spacing or alignment of players during the episode.

**Future work:** Consider adding `ACTUAL_ADJACENCY`, `ACTUAL_LINE`, and `ACTUAL_LANE` edge types derived from the gap-detection line/lane IDs already computed on `PLAYER` nodes (lines 833–839), enabling direct comparison of nominal vs. actual connectivity.

### 8.5 Template mirroring is manual

The Python `FORMATIONS` dict (lines 328–397) is a hand-copied mirror of `src/engine/Rules.ts` lines 68–139. Drift between the two sources is not automatically detected.

**Future work:** Generate the Python dict from TypeScript at build time, or add a contract test (`npm run check:contracts`) that asserts byte-level equality of formation definitions.

### 8.6 No validation of formation-to-player-count consistency

`_get_formation_slot_positions` slices `template[:num_players]` (line 694). If `num_players` exceeds the template length (e.g. a 12-player scenario with an 11-slot template), the extra players receive no slot and will not get `ASSIGNED_TO` edges. No warning is emitted.

**Future work:** Add an assertion or schema validation that `len(template) >= num_players` and surface a clear error when a scenario overruns its formation template.
