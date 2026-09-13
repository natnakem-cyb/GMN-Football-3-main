# GNN Phase 4 — Tensor Feature Specification

> **Scope:** Documentation-only. No code changes.  
> **Source of truth:** `training/gnn_graph_to_tensor.py` (lines 1–496).  
> **Schema version:** v3 graph dict → `GraphTensor` dataclass.

---

## 1. Overview

The Phase 4 tensor adapter converts a Phase 3 graph dict (produced by `training.gnn_graph_builder.build_graph()`) into a `GraphTensor` dataclass ready for PyTorch GNN consumption.  
All continuous quantities are packed into fixed-width tensors with deterministic ordering.

| Tensor field | Shape | Dtype | Notes |
|---|---|---|---|
| `node_features` | `(num_nodes, 32)` | `float32` | Fixed-width per-node feature matrix |
| `node_type` | `(num_nodes,)` | `int64` | Node-type index |
| `edge_index` | `(2, num_edges)` | `int64` | COO format |
| `edge_type` | `(num_edges,)` | `int64` | Edge-type index |
| `edge_features` | `(num_edges, 10)` | `float32` | Fixed-width per-edge feature matrix |
| `node_mask` | `(num_nodes,)` | `float32` | Valid-node mask (1.0 = active) |
| `graph_context` | `(8,)` or `None` | `float32` | Phase 2 `zScenario` embedding |
| `agent_node_indices` | `List[int]` | — | Indices of controllable PLAYER nodes |

**Key constants** (`training/gnn_graph_to_tensor.py`):
- `NODE_FEATURE_DIM = 32` (line 49)
- `EDGE_FEATURE_DIM = 10` (line 52)
- `Z_SCENARIO_DIM = 8` (line 55)
- `NODE_TYPE_TO_INDEX` mapping (lines 24–30)
- `EDGE_TYPE_TO_INDEX` mapping (lines 32–46)

---

## 2. Node Feature Dictionary

Each node is encoded into a 32-dimensional float vector. The semantic meaning of each index depends on the node’s type.

### 2.1 PLAYER Nodes

Encoder: `_encode_player_node` (lines 101–143).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0 | `position.x` | World-space x | Already normalized | `[-1.0, 1.0]` | Pitch half-length = 1.0 |
| 1 | `position.y` | World-space y | Already normalized | `[-0.42, 0.42]` | Pitch half-width ≈ 0.42 |
| 2 | `velocity.vx` | X velocity (m/s) | `/ 50.0` | `[-1.0, 1.0]` typical | |
| 3 | `velocity.vy` | Y velocity (m/s) | `/ 50.0` | `[-1.0, 1.0]` typical | |
| 4–15 | `role_one_hot` | 12-dim one-hot | None | `{0, 1}` | Vocabulary from `ROLE_VOCABULARY` (line 113) |
| 16 | `is_active` | Boolean | `1.0 if True else 0.0` | `{0, 1}` | Missing key → 0.0 (line 117) |
| 17 | `is_controlled` | Boolean | `1.0 if True else 0.0` | `{0, 1}` | Missing key → 0.0 (line 118) |
| 18 | `is_goalkeeper` | Boolean | `1.0 if True else 0.0` | `{0, 1}` | Missing key → 0.0 (line 119) |
| 19 | `nearest_teammate_dist` | Distance (m) | `/ 1.414` | `[0.0, 1.0]` | Max diagonal of normalized pitch |
| 20 | `nearest_opponent_dist` | Distance (m) | `/ 1.414` | `[0.0, 1.0]` | |
| 21 | `team_width` | Team width (m) | `/ 0.84` | `[0.0, 1.0]` | Full pitch width |
| 22 | `team_depth` | Team depth (m) | `/ 2.0` | `[0.0, 1.0]` | Full pitch length |
| 23 | `compactness` | Compactness (m) | `/ 1.414` | `[0.0, 1.0]` | |
| 24 | `stretch` | Stretch (m) | `/ 2.0` | `[0.0, 1.0]` | |
| 25 | `receiver_availability` | Boolean | `0.0` or `1.0` | `{0, 1}` | Missing key → 0.0 (line 129) |
| 26 | `line_id` | Formation line ID | Float cast | `int` | Missing key → 0.0 (line 132) |
| 27 | `lane_id` | Formation lane ID | Float cast | `int` | Missing key → 0.0 (line 133) |
| 28 | `team_index` | Team index | Float cast | `int` | Missing key → 0.0 (line 136) |
| 29 | `sentinel` | Sentinel flag | `1.0 if (x==-1.0 and y==-1.0) else 0.0` | `{0, 1}` | Detects uninitialized positions (line 141) |
| 30–31 | *unused* | — | — | `0.0` | Reserved |

### 2.2 BALL Nodes

Encoder: `_encode_ball_node` (lines 146–172).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0 | `position.x` | World-space x | Already normalized | `[-1.0, 1.0]` | |
| 1 | `position.y` | World-space y | Already normalized | `[-0.42, 0.42]` | |
| 2 | `position.z` | Height (m) | None | `[0.0, ~5.0]` | Raw height |
| 3 | `velocity.vx` | X velocity (m/s) | `/ 50.0` | `[-1.0, 1.0]` | |
| 4 | `velocity.vy` | Y velocity (m/s) | `/ 50.0` | `[-1.0, 1.0]` | |
| 5 | `velocity.vz` | Z velocity (m/s) | `/ 50.0` | `[-1.0, 1.0]` | |
| 6 | `speed` | Scalar speed (m/s) | `/ 50.0` | `[0.0, ~1.0]` | Missing key → 0.0 (line 160) |
| 7–9 | `ownership` | One-hot (none, left, right) | None | `{0, 1}` | `ownership="none"` → index 7; `"left"` → 8; `"right"` → 9 (lines 163–169) |
| 10–31 | *unused* | — | — | `0.0` | |

### 2.3 GOAL Nodes

Encoder: `_encode_goal_node` (lines 175–198).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0 | `position.x` | World-space x | Already normalized | `[-1.0, 1.0]` | |
| 1 | `position.y` | World-space y | Already normalized | `[-0.42, 0.42]` | |
| 2 | `width` | Goal width (m) | None | `≈ 0.14` | Default = 0.14 (line 183) |
| 3 | `height` | Goal height (m) | None | `≈ 0.05` | Default = 0.05 (line 184) |
| 4 | `depth` | Goal depth (m) | None | `≈ 0.04` | Default = 0.04 (line 185) |
| 5–6 | `team` | One-hot (left, right) | None | `{0, 1}` | `"left"` → index 5; else → 6 (lines 189–192) |
| 7 | `node_id` | Goal identity | `0.0` if `"goal_left"` else `1.0` | `{0, 1}` | Default = `"goal_left"` (lines 195–196) |
| 8–31 | *unused* | — | — | `0.0` | |

### 2.4 FORMATION_SLOT Nodes

Encoder: `_encode_formation_slot_node` (lines 201–224).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0 | `x` | Nominal x (normalized) | None | `[-1.0, 1.0]` | Default = 0.0 (line 205) |
| 1 | `y` | Nominal y (normalized) | None | `[-0.42, 0.42]` | Default = 0.0 (line 206) |
| 2 | `xRatio` | Relative x ratio | None | `[0.0, 1.0]` | Default = 0.0 (line 207) |
| 3 | `yRatio` | Relative y ratio | None | `[0.0, 1.0]` | Default = 0.0 (line 208) |
| 4–15 | `role` | 12-dim one-hot | None | `{0, 1}` | `ROLE_VOCABULARY` lookup; missing/unknown → all zeros (lines 211–215) |
| 16–17 | `team` | One-hot (left, right) | None | `{0, 1}` | `"left"` → 16; else → 17 (lines 219–222) |
| 18–31 | *unused* | — | — | `0.0` | |

### 2.5 TEAM_SHAPE Nodes

Encoder: `_encode_team_shape_node` (lines 227–244).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0 | `inter_line_spacing_variance` | Variance value | None | `≥ 0.0` | Default = 0.0 (line 232) |
| 1 | `mean_line_compactness` | Compactness value | None | `≥ 0.0` | Default = 0.0 (line 233) |
| 2 | `num_lines` | Number of lines | None | `int ≥ 1` | Default = 1 (line 234) |
| 3 | `num_lanes` | Number of lanes | None | `int ≥ 1` | Default = 1 (line 235) |
| 4–5 | `team` | One-hot (left, right) | None | `{0, 1}` | `"left"` → 4; else → 5 (lines 239–242) |
| 6–31 | *unused* | — | — | `0.0` | |

---

## 3. Edge Feature Dictionary

Each edge is encoded into a 10-dimensional float vector. The semantic meaning depends on the edge type.

Dispatch table: `_EDGE_ENCODERS` (lines 369–383).  
`SCENARIO_CONTEXT` edges are **skipped** during tensorization (lines 443–445); the scenario is represented as `graph_context` instead.

### 3.1 TEAMMATE

Encoder: `_encode_teammate_edge` (lines 271–280).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0 | `distance` | Euclidean distance (m) | `/ 1.414` | `[0.0, 1.0]` | Max pitch diagonal |
| 1 | `relative_x` | X offset (m) | `/ 2.0` | `[-1.0, 1.0]` | Half pitch length |
| 2 | `relative_y` | Y offset (m) | `/ 0.84` | `[-1.0, 1.0]` | Half pitch width |
| 3 | `relative_vx` | X velocity diff (m/s) | `/ 50.0` | `[-1.0, 1.0]` | |
| 4 | `relative_vy` | Y velocity diff (m/s) | `/ 50.0` | `[-1.0, 1.0]` | |
| 5 | `angle` | Angle (rad) | `/ π` | `[-1.0, 1.0]` | π ≈ 3.14159 (line 278) |
| 6 | `closing_speed` | Closing speed (m/s) | None | `≥ 0.0` | Missing key → 0.0 (line 279) |
| 7–9 | *unused* | — | — | `0.0` | |

### 3.2 OPPONENT

Encoder: `_encode_opponent_edge` (lines 283–286).  
Inherits all TEAMMATE fields; adds pressure at index 7.

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0–6 | Same as TEAMMATE | — | — | — | See §3.1 |
| 7 | `pressure` | Pressure indicator | None | `{0, 1}` | Missing key → 0.0 (line 285) |
| 8–9 | *unused* | — | — | `0.0` | |

### 3.3 NEAR

Encoder: `_encode_near_edge` (lines 289–293).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0 | `distance` | Euclidean distance (m) | `/ 1.414` | `[0.0, 1.0]` | |
| 1 | `threshold` | Proximity threshold | None | `≈ 0.065` | Fixed constant (line 292) |
| 2–9 | *unused* | — | — | `0.0` | |

### 3.4 POSSESSES

Encoder: `_encode_possesses_edge` (lines 296–297).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0–9 | *all zeros* | — | — | `0.0` | Semantic handled by PLAYER_BALL ownership one-hot |

### 3.5 PLAYER_BALL

Encoder: `_encode_player_ball_edge` (lines 300–309).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0 | `distance` | Euclidean distance (m) | `/ 1.414` | `[0.0, 1.0]` | |
| 1 | `relative_x` | X offset (m) | `/ 2.0` | `[-1.0, 1.0]` | |
| 2 | `relative_y` | Y offset (m) | `/ 0.84` | `[-1.0, 1.0]` | |
| 3 | `relative_vx` | X velocity diff (m/s) | `/ 50.0` | `[-1.0, 1.0]` | |
| 4 | `relative_vy` | Y velocity diff (m/s) | `/ 50.0` | `[-1.0, 1.0]` | |
| 5 | `angle` | Angle (rad) | `/ π` | `[-1.0, 1.0]` | |
| 6 | `has_possession` | Possession flag | `1.0 if True else 0.0` | `{0, 1}` | Missing key → 0.0 (line 308) |
| 7–9 | *unused* | — | — | `0.0` | |

### 3.6 PLAYER_GOAL

Encoder: `_encode_player_goal_edge` (lines 312–320).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0 | `distance` | Euclidean distance (m) | `/ 1.414` | `[0.0, 1.0]` | |
| 1 | `relative_x` | X offset (m) | `/ 2.0` | `[-1.0, 1.0]` | |
| 2 | `relative_y` | Y offset (m) | `/ 0.84` | `[-1.0, 1.0]` | |
| 3 | `angle` | Angle (rad) | `/ π` | `[-1.0, 1.0]` | |
| 4 | `shot_angle` | Shot angle (rad) | None | `[0.0, π]` | Missing key → 0.0 (line 318) |
| 5 | `nearest_opponent_pressure` | Pressure distance (m) | `/ 1.414` | `[0.0, 1.0]` | Missing key → 0.0 (line 319) |
| 6–9 | *unused* | — | — | `0.0` | |

### 3.7 BALL_GOAL

Encoder: `_encode_ball_goal_edge` (lines 323–330).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0 | `distance` | Euclidean distance (m) | `/ 1.414` | `[0.0, 1.0]` | |
| 1 | `relative_x` | X offset (m) | `/ 2.0` | `[-1.0, 1.0]` | |
| 2 | `relative_y` | Y offset (m) | `/ 0.84` | `[-1.0, 1.0]` | |
| 3 | `angle` | Angle (rad) | `/ π` | `[-1.0, 1.0]` | |
| 4 | `shot_angle` | Shot angle (rad) | None | `[0.0, π]` | Missing key → 0.0 (line 329) |
| 5–9 | *unused* | — | — | `0.0` | |

### 3.8 ASSIGNED_TO

Encoder: `_encode_assigned_to_edge` (lines 333–338).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0 | `deviation_distance` | Slot deviation (m) | `/ 1.414` | `[0.0, 1.0]` | |
| 1 | `deviation_x` | X deviation (m) | `/ 2.0` | `[-1.0, 1.0]` | |
| 2 | `deviation_y` | Y deviation (m) | `/ 0.84` | `[-1.0, 1.0]` | |
| 3–9 | *unused* | — | — | `0.0` | |

### 3.9 BELONGS_TO_SHAPE

Encoder: `_encode_belongs_to_shape_edge` (lines 341–346).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0 | `team_left` | Team flag | `1.0` if `team == "left"` else `0.0` | `{0, 1}` | Default team = `"left"` (line 343) |
| 1 | `team_right` | Team flag | `1.0` if `team == "right"` else `0.0` | `{0, 1}` | |
| 2–9 | *unused* | — | — | `0.0` | |

### 3.10 SCENARIO_CONTEXT

Encoder: `_encode_scenario_context_edge` (lines 349–350).  
**This edge type is excluded from the tensor adapter.** The scenario is encoded as the graph-level `zScenario` tensor instead (lines 443–445).

### 3.11 FORMATION_ADJACENCY

Encoder: `_encode_formation_adjacency_edge` (lines 353–354).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0–9 | *all zeros* | — | — | `0.0` | Structural adjacency only; features unused |

### 3.12 FORMATION_LINE

Encoder: `_encode_formation_line_edge` (lines 357–360).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0 | `line_id` | Line identifier | None | `int` | Missing key → 0.0 (line 359) |
| 1–9 | *unused* | — | — | `0.0` | |

### 3.13 FORMATION_LANE

Encoder: `_encode_formation_lane_edge` (lines 363–366).

| Index | Name | Raw Quantity | Normalization | Effective Range | Notes |
|---|---|---|---|---|---|
| 0 | `lane_id` | Lane identifier | None | `int` | Missing key → 0.0 (line 365) |
| 1–9 | *unused* | — | — | `0.0` | |

---

## 4. Graph Context (zScenario)

The scenario is encoded as a global graph-level vector, not as a node or edge.

- **Source:** `graph.get("z_scenario")` (line 402)
- **Dimension:** `Z_SCENARIO_DIM = 8` (line 55)
- **Dtype:** `float32` (line 479)
- **Construction:** Phase 2 `TaskEncoder` output (8-dim `Float32Array`)
- **Handling:**
  - If `z_scenario` is `None`, `graph_context` is set to `None` (lines 477–478).
  - If present, it is converted via `torch.tensor(z_scenario, dtype=torch.float32)` and unsqueezed to 1-D if scalar (lines 479–481).

`SCENARIO_CONTEXT` edges are filtered out before tensorization (lines 443–445), so the scenario is **never** duplicated as an edge feature.

---

## 5. Categorical Encodings

### 5.1 Node Types

Defined in `NODE_TYPE_TO_INDEX` (`training/gnn_graph_to_tensor.py`, lines 24–30).

| Node Type | Index |
|---|---|
| `PLAYER` | 0 |
| `BALL` | 1 |
| `GOAL` | 2 |
| `FORMATION_SLOT` | 3 |
| `TEAM_SHAPE` | 4 |

Unknown node types fall back to `0` (`PLAYER`) at line 249.

### 5.2 Edge Types

Defined in `EDGE_TYPE_TO_INDEX` (`training/gnn_graph_to_tensor.py`, lines 32–46).

| Edge Type | Index |
|---|---|
| `TEAMMATE` | 0 |
| `OPPONENT` | 1 |
| `NEAR` | 2 |
| `POSSESSES` | 3 |
| `PLAYER_BALL` | 4 |
| `PLAYER_GOAL` | 5 |
| `BALL_GOAL` | 6 |
| `ASSIGNED_TO` | 7 |
| `BELONGS_TO_SHAPE` | 8 |
| `SCENARIO_CONTEXT` | 9 |
| `FORMATION_ADJACENCY` | 10 |
| `FORMATION_LINE` | 11 |
| `FORMATION_LANE` | 12 |

Unknown edge types fall back to `0` (`TEAMMATE`) at line 438.

### 5.3 Role Vocabulary

- **Used in:** PLAYER nodes (indices 4–15, line 113) and FORMATION_SLOT nodes (indices 4–15, lines 211–215).
- **Source:** `ROLE_VOCABULARY` imported from `training.gnn_graph_builder` (line 18).
- **Size:** 12 categories.
- **Encoding:** Dense one-hot. Index of role in vocabulary → 1.0; all others → 0.0.
- **Fallback:** If the role string is missing from `ROLE_VOCABULARY`, the entire 12-dim slice is zero (lines 213–215).

---

## 6. Normalization Constants

All divisors are chosen to bound inputs to approximately `[-1.0, 1.0]` or `[0.0, 1.0]`.

| Divisor / Constant | Applied To | Location in Source | Purpose |
|---|---|---|---|
| `1.0` / `1.0` | `position.x`, `position.y` | Lines 105–106, 150–152, 179–180 | Already normalized by upstream builder |
| `50.0` | Velocities (`vx`, `vy`, `vz`), `speed` | Lines 109–110, 155–157, 160, 274–277, 303–307 | Typical max speed ~50 m/s |
| `1.414` | Distances, compactness, stretch, relative offsets, pressure distance | Lines 123–127, 273, 291, 302, 314–315, 319, 335 | √2 ≈ max diagonal of normalized pitch |
| `2.0` | `team_depth`, stretch, relative_x, deviation_x | Lines 122, 126, 274, 304, 336 | Full pitch length / 2 |
| `0.84` | `team_width`, relative_y, deviation_y | Lines 125, 275, 305, 337 | Full pitch width / 2 |
| `π` (≈ `3.14159`) | `angle` fields | Lines 278, 307, 317, 328 | Normalize radians to `[-1.0, 1.0]` |
| `0.065` | `threshold` (NEAR edge) | Line 292 | Fixed proximity threshold |
| `1.0` / `0.0` | Boolean flags (`is_active`, `is_controlled`, `is_goalkeeper`, `receiver_availability`, `has_possession`, pressure, team flags) | Various | Standardized to `{0.0, 1.0}` |

---

## 7. Missing Value Handling

The adapter is defensive: every raw quantity is accessed with `.get(key, default)` or wrapped in `float(...)`, ensuring the tensor is always fully populated with zeros for absent data.

| Scenario | Behavior | Source Line |
|---|---|---|
| Missing `role_one_hot` in PLAYER | 12-dim zeros | 113 |
| Missing boolean flags (`is_active`, `is_controlled`, `is_goalkeeper`) | `False` → `0.0` | 117–119 |
| Missing derived team features (`nearest_teammate_dist`, etc.) | `0.0` fallback | 122–129 |
| Missing `line_id`, `lane_id`, `team_index` | `0.0` fallback | 132–136 |
| Missing `speed` in BALL | `0.0` fallback | 160 |
| Missing `ownership` in BALL | Defaults to `"none"` → index 7 = 1.0 | 163 |
| Missing `width`, `height`, `depth` in GOAL | Fixed geometry defaults (0.14, 0.05, 0.04) | 183–185 |
| Missing `node_id` in GOAL | Defaults to `"goal_left"` | 195 |
| Missing `role` in FORMATION_SLOT | All 12-dim one-hot zeros | 211–215 |
| Missing formation deviation in TEAM_SHAPE | `0.0` fallback | 231–235 |
| Missing edge attributes (`distance`, `relative_x`, etc.) | `0.0` fallback via `.get(..., 0.0)` | 273, 291, 302, 314, 325, 335 |
| Missing `z_scenario` | `graph_context = None` | 477–478 |
| Dangling edge references | `ValueError` raised | 450–453 |

**Important:** Missing or malformed edge references are **not** silently dropped; they raise a `ValueError` to surface schema violations early (lines 450–453).

---

## 8. Stability Guarantees

1. **Deterministic Node Ordering**  
   Nodes are processed in the order they appear in the graph dict’s `nodes` list (line 422). The `node_id_to_index` mapping preserves this order (lines 405–417), so `edge_index` references are stable across identical graph inputs.

2. **Fixed Tensor Dimensions**  
   - `NODE_FEATURE_DIM = 32` (line 49)
   - `EDGE_FEATURE_DIM = 10` (line 52)
   - `Z_SCENARIO_DIM = 8` (line 55)  
   These are compile-time constants; tensors are allocated with fixed second dimensions (`training/gnn_graph_to_tensor.py`).

3. **Deterministic Categorical Mappings**  
   `NODE_TYPE_TO_INDEX` and `EDGE_TYPE_TO_INDEX` are static dictionaries (lines 24–46). No random or hash-based assignment occurs.

4. **Graph Context Isolation**  
   `zScenario` is stored as a separate `graph_context` field (line 72), not embedded into node or edge features. This prevents shape-conditional branching from corrupting tensor layouts.

5. **Edge Skipping for SCENARIO_CONTEXT**  
   `SCENARIO_CONTEXT` edges are explicitly skipped (lines 443–445) so that scenario data flows only through `graph_context`, avoiding duplicate or conflicting representations.

---

*End of specification.*
