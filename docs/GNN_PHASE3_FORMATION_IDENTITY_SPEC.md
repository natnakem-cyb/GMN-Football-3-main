# GNN Phase 3: Formation and Player Identity Specification

## 1. Overview

This document specifies the identity fields emitted by the GNN graph builder in `training/gnn_graph_builder.py`. It covers `PLAYER` node identity, `FORMATION_SLOT` node identity, and the `ASSIGNED_TO` edge that links players to their nominal formation slots. The spec is tied to the actual implementation and uses exact line references so future changes to the builder can be audited against this contract.

**Scope:** This is documentation only. No code changes are implied.

**Source file:** `training/gnn_graph_builder.py`

---

## 2. Player Identity Fields

Player nodes are built in `_build_player_nodes()` (`training/gnn_graph_builder.py:720-814`). The function receives a parsed 127-dim observation, a `scenario_id`, and an optional `controlled_player_id` derived from `info.get("controlledPlayerId")` (`training/gnn_graph_builder.py:1531`).

| Field | Type | Source | Static / Dynamic | Notes |
|---|---|---|---|---|
| `global_id` | `str` | `f"left_{i}"` or `f"right_{i}"` where `i` is the 0-based loop index within the team (`training/gnn_graph_builder.py:766, 794`). | **Static** per episode | Deterministic and does not change when the engine's `controlledPlayerId` switches. It is a roster slot identifier, not a player identity token. |
| `team` | `"left"` / `"right"` | Hardcoded per loop branch (`training/gnn_graph_builder.py:773, 801`). | **Static** | Derived from the observation layout: left team occupies indices 0-10 (positions) and 22-32 (velocities); right team occupies 44-54 (positions) and 66-76 (velocities). |
| `team_index` | `int` (0..10) | Loop index `i` (`training/gnn_graph_builder.py:774, 802`). | **Static** per episode | 0-based position within the team roster slice. |
| `position` | `{x, y}` | Parsed from `left_positions[i]` / `right_positions[i]` (`training/gnn_graph_builder.py:764, 792`). | **Dynamic** | Normalized pitch coordinates in `[-1.0, 1.0] x [-0.42, 0.42]`. The observation parsing is at `training/gnn_graph_builder.py:555-635`. |
| `velocity` | `{vx, vy}` | Parsed from `left_velocities[i]` / `right_velocities[i]` (`training/gnn_graph_builder.py:765, 793`). | **Dynamic** | Scaled by 50 relative to raw engine units. Velocity indices in the 127-dim vector: left 22-43, right 66-87. |
| `role` | `str` | `left_roles[i]` / `right_roles[i]`, sourced from `_get_scenario_roles()` (`training/gnn_graph_builder.py:763, 791`). For `11_vs_11`, roles come from the formation template. For academy scenarios, from `scenario["setup"]`. | **Static** per scenario setup | Values are drawn from `ROLE_VOCABULARY` (12 entries). |
| `role_one_hot` | `list[float]` (12-dim) | `_role_one_hot(role)` (`training/gnn_graph_builder.py:683-688`). | **Static** per scenario setup | One-hot over `ROLE_VOCABULARY`. Dim 0..11. Unknown roles map to all zeros. |
| `is_active` | `bool` | `(viewpoint_team == team and active_idx == i)` (`training/gnn_graph_builder.py:767, 795`). `active_idx` is from observation indices 97-107 (`training/gnn_graph_builder.py:602-604`). | **Dynamic** | Indicates which player is the current viewpoint agent. The active index is within the viewpoint team's 0-based roster slot. |
| `is_controlled` | `bool` | `(global_id == controlled_player_id)` (`training/gnn_graph_builder.py:768, 807`). `controlled_player_id` comes from `info.get("controlledPlayerId")` (`training/gnn_graph_builder.py:1531`). | **Dynamic** | The engine-controlled player for the controlled team. For the non-controlled team, this is always `False` (`training/gnn_graph_builder.py:807`). |
| `is_goalkeeper` | `bool` | `_is_gk_from_role(role)` (`training/gnn_graph_builder.py:679-680, 781, 808`). | **Static** per scenario setup | True iff `role == "GK"`. |

**Additional derived fields (not identity, but present on the node):**

- `features`: Dict of derived geometric features (nearest teammate/opponent distances, team width/depth/compactness/stretch, receiver availability). Computed at `training/gnn_graph_builder.py:837-859`.
- `line_id`, `lane_id`: Integers from gap-detection on x/y axes. Computed at `training/gnn_graph_builder.py:829-834`.

---

## 3. Formation Slot Identity Fields

Formation slots are built in `_get_formation_positions_for_scenario()` (`training/gnn_graph_builder.py:520-547`). The function requires that the scenario has a `formation` key in `SCENARIOS`; if absent or falsy, it returns `None` and no `FORMATION_SLOT` nodes are emitted.

The absolute positions are computed by `_get_formation_slot_positions()` (`training/gnn_graph_builder.py:691-712`). This mirrors `Rules.ts getFormationPositions` and uses the embedded `FORMATIONS` table.

| Field | Type | Source | Static / Dynamic | Notes |
|---|---|---|---|---|
| `slot_id` | `str` | `f"left_slot{i}"` or `f"right_slot{i}"` where `i` is the enumeration index over the returned slot list (`training/gnn_graph_builder.py:539, 543`). | **Static** | Deterministic: `{team}_{index in FORMATIONS array}`. |
| `team` | `"left"` / `"right"` | Hardcoded per loop branch (`training/gnn_graph_builder.py:538, 542`). | **Static** | |
| `role` | `str` | `node["role"]` from the `FORMATION` template (`training/gnn_graph_builder.py:706`). | **Static** | Matches `ROLE_VOCABULARY`. |
| `xRatio` | `float` (0..1) | `node["xRatio"]` from the `FORMATION` template (`training/gnn_graph_builder.py:697, 707`). | **Static** | Normalized horizontal position within the team's half. |
| `yRatio` | `float` (0..1) | `node["yRatio"]` from the `FORMATION` template (`training/gnn_graph_builder.py:698, 708`). | **Static** | Normalized vertical position. |
| `x` | `float` | Computed at `training/gnn_graph_builder.py:700-703`. For `left`: `-1.0 + xRatio * 1.2`; for `right`: `1.0 - xRatio * 1.2`. | **Static** | Absolute pitch x coordinate. Maps `xRatio \in [0,1]` to `[-1.0, 1.0]`. |
| `y` | `float` | Computed at `training/gnn_graph_builder.py:701, 704`. For `left`: `-0.42 + yRatio * 0.84`; for `right`: `0.42 - yRatio * 0.84`. | **Static** | Absolute pitch y coordinate. Maps `yRatio \in [0,1]` to `[-0.42, 0.42]`. |

**Node type label:** All formation slot nodes have `node_type: "FORMATION_SLOT"`.

---

## 4. ASSIGNED_TO Edge Semantics

`ASSIGNED_TO` edges are built by `_build_assigned_to_edges()` (`training/gnn_graph_builder.py:1225-1277`). Only present players (those whose position is not the sentinel `{-1.0, -1.0}`) are assigned.

### Assignment Algorithm

1. **Group by team:** Slots are partitioned into `left` and `right` lists (`training/gnn_graph_builder.py:1238-1240`).
2. **Group by role:** Within the player's team, slots are grouped by `role` (`training/gnn_graph_builder.py:1249-1252`).
3. **Role match:** Candidate slots are those matching the player's `role` (`training/gnn_graph_builder.py:1254`).
4. **Fallback:** If no slot has the player's role, all team slots are used as candidates (`training/gnn_graph_builder.py:1255-1258`).
5. **Deterministic tie-breaking:** Candidates are sorted by `(x, y)` ascending, and the first slot is selected (`training/gnn_graph_builder.py:1260-1262`).
6. **Deviation:** The edge stores the difference between actual player position and nominal slot position (`training/gnn_graph_builder.py:1263-1275`).

### Edge Fields

| Field | Type | Notes |
|---|---|---|
| `edge_type` | `"ASSIGNED_TO"` | |
| `source` | `str` | Player `global_id`. |
| `target` | `str` | Formation slot `slot_id`. |
| `role` | `str` | Slot role (should match player role for role-based assignment). |
| `nominal_x` | `float` | Slot's static x position. |
| `nominal_y` | `float` | Slot's static y position. |
| `deviation_x` | `float` | `player.x - slot.x`. |
| `deviation_y` | `float` | `player.y - slot.y`. |
| `deviation_distance` | `float` | Euclidean distance between player position and slot position (`training/gnn_graph_builder.py:1274`). |

### Known Limitations

- **Academy scenarios (except `11_vs_11`):** These scenarios do not have a `formation` field in `SCENARIOS`, so `_get_formation_positions_for_scenario()` returns `None` and no `ASSIGNED_TO` edges are produced (`training/gnn_graph_builder.py:528-530`).
- **Duplicate roles:** When multiple slots share the same role (e.g., two CBs), the assignment is deterministic by `(x, y)` sort order, but the mapping between a specific player and a specific same-role slot is a derived convention, not an authoritative engine signal.
- **No explicit engine signal:** The game engine does not expose a "player X is assigned to slot Y" mapping. The assignment is inferred from role matching and deterministic geometry.

---

## 5. Nominal vs Actual Distinction

The GNN graph maintains two layers of positional truth:

- **Nominal (formation slot):** Static positions derived from the `FORMATION` template. These represent the team's intended shape and do not change during an episode. Fields: `xRatio`, `yRatio`, `x`, `y` on `FORMATION_SLOT` nodes; `nominal_x`, `nominal_y` on `ASSIGNED_TO` edges.
- **Actual (player):** Dynamic positions parsed from the 127-dim observation vector each tick. Fields: `position.{x,y}` on `PLAYER` nodes; `deviation_x`, `deviation_y`, `deviation_distance` on `ASSIGNED_TO` edges.

The `ASSIGNED_TO` edge bridges these layers: its `nominal_*` fields anchor the player to the team's planned structure, while its `deviation_*` fields measure how far the player has drifted from that plan.

---

## 6. Duplicate Role Handling

When a formation template contains multiple players with the same role (e.g., `["CB", "CB", "CM", "CM", "ST"]`), the assignment logic at `training/gnn_graph_builder.py:1249-1262` proceeds as follows:

1. All slots with the matching role are collected into `candidate_slots` (`training/gnn_graph_builder.py:1254`).
2. If the player's role has no matching slots, the fallback is all team slots (`training/gnn_graph_builder.py:1255-1258`).
3. Candidates are sorted by `(slot.x, slot.y)` ascending (`training/gnn_graph_builder.py:1261`).
4. The first candidate after sort is selected (`training/gnn_graph_builder.py:1262`).

**Implication:** Two players with the same role (e.g., two CBs) will be assigned to the two same-role slots ordered by x then y. Because the player loop processes players in `global_id` order (left_0, left_1, ..., right_0, ...), and the slot sort is deterministic, the overall mapping is deterministic across episodes for the same scenario. However, there is no guarantee that `left_0` maps to the "leftmost" CB slot versus the "rightmost" CB slot unless the role ordering in the formation template is aligned with the roster index ordering.

**No authoritative mapping:** The engine does not expose which specific player occupies which specific same-role slot. The mapping is a deterministic convention, not a contract.

---

## 7. Unavailable Information

The following data points are **not** available from the engine observation or the current graph builder:

| Missing Data | Reason |
|---|---|
| Explicit player-to-slot assignment from the engine | The engine exposes `controlledPlayerId` and the 127-dim observation, but no direct "slot assignment" signal. |
| Player fatigue / stamina | Not present in the 127-dim observation vector. |
| Player height / physical attributes | Not present. |
| Ball owner player global_id | `exact_ball_owner_id` is set to `None` in parsed output (`training/gnn_graph_builder.py:634`). Ball ownership is only available as a team-level one-hot (indices 94-96). |
| Formation slot occupancy confidence | The `ASSIGNED_TO` edge is always created for present players; there is no uncertainty score. |
| Substitution / roster changes within an episode | Player identities are static per episode; the graph builder does not handle mid-episode roster mutations. |
| Formation slot timeout / abandonment | Formation slots are static template nodes; they are never removed even if all players drift far from them. |

---

## 8. Phase 4 Contract (GNN Encoder Input)

The GNN encoder will receive the following identity contract from the graph builder:

### Nodes

**PLAYER nodes (per team):**
- `node_type`: `"PLAYER"`
- `global_id`: `"left_0"` .. `"left_{N-1}"`, `"right_0"` .. `"right_{M-1}"`
- `team`: `"left"` / `"right"`
- `team_index`: `0` .. `10`
- `position`: `{x, y}` normalized `[-1.0, 1.0] x [-0.42, 0.42]`
- `velocity`: `{vx, vy}` scaled by 50
- `role`: string from `ROLE_VOCABULARY`
- `role_one_hot`: 12-float vector
- `is_active`: bool
- `is_controlled`: bool
- `is_goalkeeper`: bool
- `features`: dict of derived geometric features
- `line_id`, `lane_id`: integers from gap detection

**FORMATION_SLOT nodes (only for scenarios with a `formation` field, e.g., `11_vs_11`):**
- `node_type`: `"FORMATION_SLOT"`
- `slot_id`: `"left_slot0"` .. `"left_slot{N-1}"`, `"right_slot0"` .. `"right_slot{M-1}"`
- `team`: `"left"` / `"right"`
- `role`: string from formation template
- `xRatio`, `yRatio`: floats in `[0, 1]`
- `x`, `y`: absolute pitch coordinates

### Edges

**ASSIGNED_TO edges (only for scenarios with formation slots):**
- `edge_type`: `"ASSIGNED_TO"`
- `source`: player `global_id`
- `target`: slot `slot_id`
- `role`: slot role
- `nominal_x`, `nominal_y`: slot position
- `deviation_x`, `deviation_y`: `player.pos - slot.pos`
- `deviation_distance`: Euclidean distance

**Other edges present in the graph (for context):**
- `TEAMMATE` (rich)
- `OPPONENT` (rich)
- `PLAYER_BALL`
- `BELONGS_TO_SHAPE`
- `SCENARIO_CONTEXT`

### Determinism Guarantees

- Node ordering within a type is fixed: left team first, then right team; within each team, by `team_index` ascending.
- Edge ordering within a type is fixed by source `global_id` lexicographic order.
- `slot_id` and `global_id` are stable across ticks for the same episode.
- `ASSIGNED_TO` assignment is deterministic by role match + `(x, y)` sort; it does not change arbitrarily between ticks unless the player's position changes such that a different slot becomes the first-match after sort (which can only happen if role fallback is used and positions cross).

---

*Generated from implementation at `training/gnn_graph_builder.py` (rev 9bc2516).*
