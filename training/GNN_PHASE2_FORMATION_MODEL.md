# GNN_PHASE2_FORMATION_MODEL

**Date:** 2026-09-12  
**Scope:** Design/documentation only. No engine, scenario, observation, or training code was modified.  
**Ground truth:** `training/GNN_PHASE0_ARCHITECTURE_AUDIT.md` (commit `a0f4c0a`) and `training/GNN_PHASE1_GRAPH_SCHEMA.md`. Where this document disagrees with Phase 0/1, the earlier phase wins.

---

## Changelog

| Version | Phase | Changes |
|---------|-------|--------|
| v1 | Phase 1 | Initial schema: PLAYER, BALL, GOAL, SCENARIO nodes; TEAMMATE, OPPONENT, NEAR, POSSESSES edges. No formation structure. |
| v2 | Phase 2 | Adds Track A formation graph structure (11_vs_11 only): FORMATION_SLOT node, FORMATION_ADJACENCY/LINE/LANE edges. Adds Track B continuous shape descriptors for academy scenarios: `line_id`/`lane_id` on PLAYER, TEAM_SHAPE node with template-free deviation measures. Extends worked example. |

---

## 1. Explicit Scope Statement

**Formation identity is available and used as ground truth for `11_vs_11` only.** For all academy scenarios, this phase defines continuous shape descriptors only -- no formation label is assigned, inferred, or displayed as fact.

This is not a limitation to be worked around. It is a structural property of the environment:

- `GameEngine.ts:289` gates formation lookup behind `scenario.setup.leftPlayers.length === 0`, which is true only for `11_vs_11` (`ScenarioRegistry.ts:409`: `leftPlayers: []`).
- Every academy scenario supplies explicit spawn coordinates (`ScenarioRegistry.ts:19-420`). None consult `TeamConfig.formation` or `Rules.ts` `FORMATIONS` table.
- Therefore, conditioning a GNN on a categorical formation label for any academy scenario would require inventing a label by fitting positions to the nearest `FORMATIONS` template. A best-fit match against a 4-4-2 template for a 3-vs-4 academy drill is a modeling choice, not a fact recovered from the environment. Track B represents it as such -- continuous descriptors with no categorical identity.

The two tracks in this phase are deliberately separate and never blended:

- **Track A** uses the real `FormationType` label and `FORMATIONS` template for `11_vs_11` only. This is ground truth, treated with full confidence.
- **Track B** defines template-free continuous shape descriptors (line grouping, lane grouping, formation deviation) for all academy scenarios. No formation label appears anywhere in Track B output.

---

## 2. Track A -- 11_vs_11 Formation Graph Structure

### 2.1 Formation inventory

`Rules.ts:68-139` defines `FORMATIONS: Record<FormationType, FormationNode[]>` with 7 formation keys (`types/football.ts:87`):

| Formation | Role count | Role entries (in array order) |
|-----------|-----------|-------------------------------|
| `4-3-3` | 11 | GK, LB, CB, CB, RB, CM, CM, CM, LW, ST, RW |
| `4-4-2` | 11 | GK, LB, CB, CB, RB, LM, CM, CM, RM, ST, ST |
| `3-5-2` | 11 | GK, CB, CB, CB, LM, CM, CAM, CM, RM, ST, ST |
| `5-3-2` | 11 | GK, LB, CB, CB, CB, RB, CM, CM, CM, ST, ST |
| `1-2-1` | 5 | GK, CB, LM, RM, ST |
| `1-1-1` | 4 | GK, CB, CM, ST |
| `1-0` | 1 | GK |

The four full-pitch formations (`4-3-3`, `4-4-2`, `3-5-2`, `5-3-2`) each have exactly 11 entries: 1 GK + 10 outfield players. This is a clean 1:1 mapping to `11_vs_11` 11 players per side (`ScenarioRegistry.ts:402-403`: `teamLeftPlayers: 11`, `teamRightPlayers: 11`).

### 2.2 Default formation for 11_vs_11

`GameEngine.ts:30` and `GameEngine.ts:43` set both `teamLeftConfig.formation` and `teamRightConfig.formation` to `4-3-3`. When `setup.leftPlayers` is empty (`ScenarioRegistry.ts:409`), `GameEngine.ts:289-290` calls `getFormationPositions(this.teamLeftConfig.formation, "left", scenario.teamLeftPlayers)`, which resolves to `FORMATIONS["4-3-3"]` sliced to 11 entries (`Rules.ts:146-147`). All 11 slots are populated.

`getFormationPositions` (`Rules.ts:141-168`) maps each `FormationNode` to absolute pitch coordinates:

- Left team: `x = -1.0 + xRatio * 1.2`, `y = -0.42 + yRatio * 0.84` (occupies own half, stretching to midfield)

- Right team: `x = 1.0 - xRatio * 1.2`, `y = 0.42 - yRatio * 0.84` (mirror image)

For the 4-3-3, this produces the following absolute positions (left team, no jitter):

| Slot | Role | xRatio | yRatio | x (abs) | y (abs) |
|------|------|--------|--------|---------|---------|
| 0 | GK | 0.05 | 0.50 | -0.940 | 0.000 |
| 1 | LB | 0.25 | 0.15 | -0.700 | -0.315 |
| 2 | CB | 0.22 | 0.38 | -0.736 | -0.098 |
| 3 | CB | 0.22 | 0.62 | -0.736 | 0.098 |
| 4 | RB | 0.25 | 0.85 | -0.700 | 0.315 |
| 5 | CM | 0.48 | 0.30 | -0.424 | -0.168 |
| 6 | CM | 0.44 | 0.50 | -0.472 | 0.000 |
| 7 | CM | 0.48 | 0.70 | -0.424 | 0.168 |
| 8 | LW | 0.75 | 0.15 | -0.100 | -0.315 |
| 9 | ST | 0.82 | 0.50 | -0.016 | 0.000 |
| 10 | RW | 0.75 | 0.85 | -0.100 | 0.315 |

### 2.3 FORMATION_SLOT nodes

For `11_vs_11` only, the graph includes 11 `FORMATION_SLOT` nodes per team (22 total). Each slot corresponds to one entry in the `FORMATIONS` template array.

| Field | Type | Description |
|-------|------|-------------|
| `node_type` | `"FORMATION_SLOT"` | Schema constant |
| `slot_id` | `string` pattern `^(left|right)_slot[0-9]+$` | `{team}_{index in FORMATIONS array}` |
| `team` | `"left"` or `"right"` | Team side |
| `role` | `PlayerRole` enum | Role from `FormationNode.role` (`Rules.ts:90`) |
| `xRatio` | `number` [0,1] | Template x-ratio from `FormationNode.xRatio` |
| `yRatio` | `number` [0,1] | Template y-ratio from `FormationNode.yRatio` |
| `x` | `number` | Absolute pitch x from `getFormationPositions` |
| `y` | `number` | Absolute pitch y from `getFormationPositions` |

The `slot_id` uses the array index (0-based), not the player number. For the 4-3-3 left team: `left_slot0` (GK) through `left_slot10` (RW).

### 2.4 FORMATION_LINE edges

`FORMATION_LINE` edges connect all pairs of `FORMATION_SLOT` nodes that belong to the same tactical line. Lines are determined by xRatio bands in the formation template:

| Line | xRatio band | 4-3-3 members |
|------|-------------|----------------|
| GK | xRatio < 0.10 | {GK} |
| Defense | 0.10 <= xRatio < 0.35 | {LB, CB, CB, RB} |
| Midfield | 0.35 <= xRatio < 0.60 | {CM, CM, CM} |
| Attack | xRatio >= 0.60 | {LW, ST, RW} |

These bands are derived from the 4-3-3 template actual xRatio values (Rules.ts:69-81): GK at 0.05, defenders at 0.22-0.25, midfielders at 0.44-0.48, attackers at 0.75-0.82. The bands are chosen to cleanly separate these clusters with margin.

Each `FORMATION_LINE` edge carries a `line_id` field (0=GK, 1=Defense, 2=Midfield, 3=Attack) to identify which line it belongs to. Edges are undirected: for a line with n slots, all n*(n-1)/2 pairs are emitted.

For the 4-3-3 left team, this produces:
- GK line: 0 edges (single slot)
- Defense line: 6 edges (LB-CB, LB-CB, LB-RB, CB-CB, CB-RB, CB-RB)
- Midfield line: 3 edges (CM-CM, CM-CM, CM-CM)
- Attack line: 3 edges (LW-ST, LW-RW, ST-RW)
- **Total: 12 FORMATION_LINE edges per team**

### 2.5 FORMATION_LANE edges

`FORMATION_LANE` edges connect all pairs of `FORMATION_SLOT` nodes that belong to the same lateral lane. Lanes are determined by yRatio bands:

| Lane | yRatio band | 4-3-3 members |
|------|-------------|----------------|
| Left | yRatio < 0.33 | {LB, LW} |
| Center | 0.33 <= yRatio <= 0.67 | {CB, CB, CM, CM, CM, ST, GK} |
| Right | yRatio > 0.67 | {RB, RW} |

Each `FORMATION_LANE` edge carries a `lane_id` field (0=Left, 1=Center, 2=Right). For the 4-3-3 left team:
- Left lane: 1 edge (LB-LW)
- Center lane: 21 edges (7 choose 2)
- Right lane: 1 edge (RB-RW)
- **Total: 23 FORMATION_LANE edges per team**

### 2.6 FORMATION_ADJACENCY edges

`FORMATION_ADJACENCY` edges connect each `FORMATION_SLOT` to its k=2 nearest neighbors in the formation template, measured by Euclidean distance in (xRatio, yRatio) space. This captures the intuitive support structure: who is expected to support whom.

For the 4-3-3 left team, the adjacency pairs (by slot index) are:
- Slot 0 (GK): nearest = slot 2 (CB, dist=0.208), slot 3 (CB, dist=0.208)
- Slot 1 (LB): nearest = slot 2 (CB, dist=0.232), slot 5 (CM, dist=0.275)
- Slot 2 (CB): nearest = slot 0 (GK, dist=0.208), slot 3 (CB, dist=0.240)
- Slot 3 (CB): nearest = slot 0 (GK, dist=0.208), slot 2 (CB, dist=0.240)
- Slot 4 (RB): nearest = slot 3 (CB, dist=0.232), slot 7 (CM, dist=0.275)
- Slot 5 (CM): nearest = slot 2 (CB, dist=0.208), slot 6 (CM, dist=0.200)
- Slot 6 (CM): nearest = slot 5 (CM, dist=0.200), slot 7 (CM, dist=0.200)
- Slot 7 (CM): nearest = slot 6 (CM, dist=0.200), slot 3 (CB, dist=0.208)
- Slot 8 (LW): nearest = slot 1 (LB, dist=0.500), slot 9 (ST, dist=0.354)
- Slot 9 (ST): nearest = slot 6 (CM, dist=0.354), slot 8 (LW, dist=0.354)
- Slot 10 (RW): nearest = slot 4 (RB, dist=0.500), slot 9 (ST, dist=0.354)

**Total: 22 FORMATION_ADJACENCY edges per team** (11 slots x 2 neighbors, directed).

### 2.7 Track A summary

For `11_vs_11` with 4-3-3, the formation graph adds per team:
- 11 FORMATION_SLOT nodes
- 12 FORMATION_LINE edges
- 23 FORMATION_LANE edges
- 22 FORMATION_ADJACENCY edges

Across both teams: 22 FORMATION_SLOT nodes and 91 formation edges total. These are in addition to the standard PLAYER/BALL/GOAL/SCENARIO nodes and TEAMMATE/OPPONENT/NEAR/POSSESSES edges.

---
## 3. Track B -- Line/Lane/Deviation for Academy Scenarios (the real new work)

This section defines template-free continuous shape descriptors for all academy scenarios. No formation label is assigned, inferred, or displayed. The descriptors are computed directly from raw spawn coordinates.

### 3.1 Line grouping algorithm

**Method: 1D gap-detection on x-coordinate (depth).**

Algorithm (deterministic, no formation template dependency):

1. Collect all teammates of the target player (same `team` field).
2. Sort by `position.x` ascending (for left team, this is defensive-to-attacking; for right team, the sort order is reversed in pitch terms but the algorithm is applied identically on raw x values).
3. Compute consecutive gaps: `gap_i = x_{i+1} - x_i` for i = 0..n-2.
4. Split at ALL gaps exceeding threshold `T = 0.15`. Each contiguous block becomes one line.
5. Assign `line_id` sequentially from 0 (deepest/most defensive line) upward.

**Threshold justification:**

- `T = 0.15` in normalized pitch coordinates (x in [-1.0, 1.0], total range 2.0) equals 7.5% of pitch length.
- For reference: penalty box depth is 0.33 (`Rules.ts:22`), so T is approximately half a penalty box width -- a meaningful tactical gap.
- This is a fixed, deterministic threshold (not data-adaptive like k-means k-selection), ensuring reproducibility across runs.
- Gap-detection was chosen over fixed x-bands (e.g., defensive = x < -0.2) because fixed bands are arbitrary and fail for asymmetric scenarios. It was chosen over k-means because k-means requires choosing k and has non-deterministic initialization.

**Edge cases:**
- 1 player: 1 line (line_id=0).
- All players within T of each other: 1 line.
- GK at extreme x: naturally isolated into its own line (gap to nearest outfield player typically > 0.15).

### 3.2 Lane grouping algorithm

**Method: 1D gap-detection on y-coordinate (width).** Identical algorithm to line grouping but on `position.y` instead of `position.x`, with the same threshold `T = 0.15`.

1. Sort teammates by `position.y` ascending.
2. Compute consecutive gaps in y.
3. Split at all gaps > 0.15.
4. Assign `lane_id` sequentially from 0 (lowest y, top of pitch from left teams view).

**Note on y-range:** Pitch y is in [-0.42, 0.42] (range 0.84, `Rules.ts:9-10`). T = 0.15 equals 17.9% of pitch width. This is intentionally larger than the x-threshold in relative terms because lateral positioning in academy scenarios tends to be more compact.

### 3.3 Formation deviation (template-free)

Instead of measuring distance from a formation template (which would require assuming a formation identity), we define two self-referential measures:

#### 3.3.1 Inter-line spacing variance

Given k lines L_1, ..., L_k ordered by mean x-position:

- c_i = mean x of players in line L_i
- s_i = c_{i+1} - c_i for i = 1..k-1 (inter-line spacings)
- s_bar = (1/(k-1)) * sum(s_i)
- variance = (1/(k-1)) * sum((s_i - s_bar)^2)
If k < 2: variance = 0.

This measures how evenly spaced the lines are. A team with evenly-spaced lines (e.g., GK at 0.95, CB at 0.65, CM at 0.35, ST at 0.05) has low variance. A team with two lines bunched together and one far away has high variance.

#### 3.3.2 Line compactness (per-line)

For each line L_i with n_i players:

- If n_i < 2: compactness_i = 0 (trivially compact)
- Else:
  - y_bar_i = mean y of players in L_i
  - sigma_i = sqrt((1/n_i) * sum((y_j - y_bar_i)^2))  [population std dev]
  - team_width = max(y across all teammates) - min(y across all teammates)
  - compactness_i = (team_width > 0) ? sigma_i / team_width : 0

mean_line_compactness = (1/k) * sum(compactness_i)

This measures how tightly grouped each line is laterally, normalized by the teams own total width. A line spread across the full team width scores 0.5 (two players at opposite extremes); a line with all players at the same y scores 0.

### 3.4 Verification against real academy scenarios

#### Verification 1: `academy_3_vs_1_with_keeper` (`ScenarioRegistry.ts:105-139`)

Left team spawn coordinates (lines 120-122):
- CAM: (0.2, 0.0)
- LW: (0.45, -0.22)
- RW: (0.45, 0.22)

Line grouping (x-sorted: 0.2, 0.45, 0.45; gaps: 0.25, 0.00):
- Gap 0.25 > 0.15: split after CAM
- Line 0: {CAM}, Line 1: {LW, RW}
- Result: 2 lines. The 3 attackers are NOT all in one line -- CAM is separated from the wingers by the x-gap. This confirms the algorithm produces non-degenerate groupings.

Lane grouping (y-sorted: -0.22, 0.0, 0.22; gaps: 0.22, 0.22):
- Both gaps > 0.15: split at both
- Lane 0: {LW}, Lane 1: {CAM}, Lane 2: {RW}
- Result: 3 lanes (left wing, center, right wing). Correct.

Right team spawn coordinates (lines 125-126):
- GK: (0.88, 0.0)
- CB: (0.55, 0.0)

Line grouping (x-sorted: 0.55, 0.88; gap: 0.33):
- Gap 0.33 > 0.15: split
- Line 0: {CB}, Line 1: {GK}
- Result: 2 lines. GK correctly isolated from defender.

Lane grouping (y-sorted: 0.0, 0.0; gap: 0.0):
- No gap > 0.15: 1 lane
- Lane 0: {GK, CB}
- Result: 1 lane. Both players centrally positioned. Correct.

#### Verification 2: `academy_rondo_4v1` (`ScenarioRegistry.ts:319-353`)

Left team spawn coordinates (lines 334-337):
- CM: (0.25, 0.0)
- CM: (-0.25, 0.0)
- LW: (0.0, 0.25)
- RW: (0.0, -0.25)

Line grouping (x-sorted: -0.25, 0.0, 0.0, 0.25; gaps: 0.25, 0.0, 0.25):
- Gaps at indices 0 and 2 both > 0.15: split at both
- Line 0: {CM at x=-0.25}, Line 1: {LW, RW at x=0.0}, Line 2: {CM at x=0.25}
- Result: 3 lines. Deep CM, middle wingers, advanced CM. Correct for a diamond rondo shape.

Lane grouping (y-sorted: -0.25, 0.0, 0.0, 0.25; gaps: 0.25, 0.0, 0.25):
- Gaps at indices 0 and 2 both > 0.15: split at both
- Lane 0: {RW}, Lane 1: {CM, CM}, Lane 2: {LW}
- Result: 3 lanes. Right wing, center, left wing. Correct.

### 3.5 Track B node/field placement

- `line_id` and `lane_id` are new fields on the **PLAYER** node (per-player assignment).
- `formation_deviation` (inter_line_spacing_variance, mean_line_compactness, num_lines, num_lanes) goes on a new **TEAM_SHAPE** summary node (one per team).

This separation keeps per-player assignments on the player node and team-level aggregates on a dedicated summary node, consistent with Phase 1s pattern of SCENARIO node for scenario-level data.

---
## 4. Schema Changes

### 4.1 PLAYER node additions

Two new required fields added to `player_node`:

- `line_id`: integer (>= 0). Deterministic line assignment from 1D gap-detection on teammate x-coordinates. 0 = deepest (most defensive) line.
- `lane_id`: integer (>= 0). Deterministic lane assignment from 1D gap-detection on teammate y-coordinates. 0 = lowest y.

These are added to the `required` array of `player_node`.

### 4.2 TEAM_SHAPE node (new node type)

A new `team_shape_node` type is added to `$defs` with fields:
- `node_type`: const `"TEAM_SHAPE"`
- `team`: enum `["left", "right"]`
- `formation_deviation`: object with fields:
  - `inter_line_spacing_variance`: number (>= 0)
  - `mean_line_compactness`: number (0 to 1)
  - `num_lines`: integer (>= 1)
  - `num_lanes`: integer (>= 1)

### 4.3 FORMATION_SLOT node (new node type, Track A only)

A new `formation_slot_node` type is added to `$defs` with fields:
- `node_type`: const `"FORMATION_SLOT"`
- `slot_id`: string pattern `^(left|right)_slot[0-9]+$`
- `team`: enum `["left", "right"]`
- `role`: PlayerRole enum
- `xRatio`: number (0 to 1)
- `yRatio`: number (0 to 1)
- `x`: number (absolute pitch x)
- `y`: number (absolute pitch y)

### 4.4 New edge types (Track A only)

Three new edge types are added to `$defs`:
- `formation_adjacency_edge`: edge_type=`"FORMATION_ADJACENCY"`, source, target
- `formation_line_edge`: edge_type=`"FORMATION_LINE"`, source, target, line_id
- `formation_lane_edge`: edge_type=`"FORMATION_LANE"`, source, target, lane_id

### 4.5 nodes and edges array updates

The `nodes.items.oneOf` array is extended with `$ref` to `formation_slot_node` and `team_shape_node`. The `edges.items.oneOf` array is extended with `$ref` to all three formation edge types.

### 4.6 Schema description update

The top-level `description` field is updated to: `v2 extends v1 with Track A formation structure (11_vs_11 only) and Track B continuous shape descriptors (academy scenarios).`

---
## 5. Worked Example Extension

Extending the existing `academy_3_vs_1_defender_3` worked example from Phase 1 with the new line/lane/deviation fields computed from actual spawn coordinates.

### 5.1 Spawn coordinates (source: `ScenarioRegistry.ts:189-202`)

Left team (`teamLeftPlayers: 3`, lines 191-195):
| Player | Role | x | y |
|--------|------|---|---|
| left_0 (controlled) | CAM | 0.20 | 0.00 |
| left_1 | LW | 0.45 | -0.22 |
| left_2 | RW | 0.45 | 0.22 |

Right team (`teamRightPlayers: 4`, lines 196-201):
| Player | Role | x | y |
|--------|------|---|---|
| right_0 | GK | 0.88 | 0.00 |
| right_1 | CB | 0.52 | -0.16 |
| right_2 | CB | 0.52 | 0.16 |
| right_3 | CB | 0.68 | 0.00 |

### 5.2 Computed line_id and lane_id values

**Left team line assignment** (x-sorted: 0.20, 0.45, 0.45; gaps: 0.25, 0.00):
| Player | x | line_id | Rationale |
|--------|---|---------|-----------|
| left_0 (CAM) | 0.20 | 0 | Deepest player; gap to next is 0.25 > 0.15 |
| left_1 (LW) | 0.45 | 1 | Advanced line with RW |
| left_2 (RW) | 0.45 | 1 | Same line as LW (gap 0.00) |

**Left team lane assignment** (y-sorted: -0.22, 0.00, 0.22; gaps: 0.22, 0.22):
| Player | y | lane_id | Rationale |
|--------|---|---------|-----------|
| left_1 (LW) | -0.22 | 0 | Lowest y; gap to next is 0.22 > 0.15 |
| left_0 (CAM) | 0.00 | 1 | Center; gaps on both sides > 0.15 |
| left_2 (RW) | 0.22 | 2 | Highest y; gap to previous is 0.22 > 0.15 |

**Right team line assignment** (x-sorted: 0.52, 0.52, 0.68, 0.88; gaps: 0.00, 0.16, 0.20):
| Player | x | line_id | Rationale |
|--------|---|---------|-----------|
| right_1 (CB) | 0.52 | 0 | Deepest defenders (2 CBs at same x) |
| right_2 (CB) | 0.52 | 0 | Same line as right_1 (gap 0.00) |
| right_3 (CB) | 0.68 | 1 | Gap 0.16 > 0.15 from line 0 |
| right_0 (GK) | 0.88 | 2 | Gap 0.20 > 0.15 from line 1 |

**Right team lane assignment** (y-sorted: -0.16, 0.00, 0.00, 0.16; gaps: 0.16, 0.00, 0.16):
| Player | y | lane_id | Rationale |
|--------|---|---------|-----------|
| right_1 (CB) | -0.16 | 0 | Lowest y; gap 0.16 > 0.15 |
| right_3 (CB) | 0.00 | 1 | Center cluster; gap 0.16 > 0.15 from lane 0 |
| right_0 (GK) | 0.00 | 1 | Same lane as right_3 (gap 0.00) |
| right_2 (CB) | 0.16 | 2 | Gap 0.16 > 0.15 from lane 1 |

### 5.3 Computed TEAM_SHAPE values

**Left team:**
- Lines: L0={CAM, x_mean=0.20}, L1={LW,RW, x_mean=0.45}
- Inter-line spacing: s1 = 0.45 - 0.20 = 0.25. k=2, so 1 spacing. Variance of 1 value = 0.
- `inter_line_spacing_variance = 0.0`
- Line compactness: L0 has 1 player (compactness=0.0); L1 has LW(y=-0.22) and RW(y=0.22):
  - y_bar = 0.0, sigma = sqrt((0.22^2 + 0.22^2)/2) = sqrt(0.0484) = 0.22
  - team_width = 0.22 - (-0.22) = 0.44
  - compactness_L1 = 0.22 / 0.44 = 0.5
- `mean_line_compactness = (0.0 + 0.5) / 2 = 0.25`
- `num_lines = 2`
- `num_lanes = 3`

**Right team:**
- Lines: L0={CB,CB, x_mean=0.52}, L1={CB, x_mean=0.68}, L2={GK, x_mean=0.88}
- Inter-line spacings: s1 = 0.68 - 0.52 = 0.16, s2 = 0.88 - 0.68 = 0.20
- s_bar = (0.16 + 0.20) / 2 = 0.18
- variance = ((0.16 - 0.18)^2 + (0.20 - 0.18)^2) / 2 = (0.0004 + 0.0004) / 2 = 0.0004
- `inter_line_spacing_variance = 0.0004`
- Line compactness: L0={CB(y=-0.16), CB(y=0.16)}:
  - y_bar = 0.0, sigma = sqrt((0.16^2 + 0.16^2)/2) = sqrt(0.0256) = 0.16
  - team_width = 0.16 - (-0.16) = 0.32
  - compactness_L0 = 0.16 / 0.32 = 0.5
  - L1 has 1 player (compactness=0.0), L2 has 1 player (compactness=0.0)
- `mean_line_compactness = (0.5 + 0.0 + 0.0) / 3 = 0.1667`
- `num_lines = 3`
- `num_lanes = 3`

### 5.4 Extended worked example snippet

The PLAYER node for `left_0` (CAM) in the extended example now includes these two new fields (all other fields unchanged from Phase 1):

```json
  "line_id": 0,
  "lane_id": 1
```

The new TEAM_SHAPE node for the left team (inside `formation_deviation`):

```json
  "inter_line_spacing_variance": 0.0,
  "mean_line_compactness": 0.25,
  "num_lines": 2,
  "num_lanes": 3
```

The TEAM_SHAPE node for the right team:

```json
  "inter_line_spacing_variance": 0.0004,
  "mean_line_compactness": 0.1667,
  "num_lines": 3,
  "num_lanes": 3
```

### 5.5 Validation: worked example still passes

The Phase 1 worked example (`academy_3_vs_1_defender_3`) had 10 nodes (7 PLAYER + 1 BALL + 2 GOAL + 1 SCENARIO) and 21 edges (9 TEAMMATE + 12 OPPONENT). With Phase 2 additions:

- All 7 PLAYER nodes now have `line_id` and `lane_id` (required fields with computed values -- see 5.2).
- 2 new TEAM_SHAPE nodes are added (one per team).
- No FORMATION_SLOT nodes or formation edges are added (this is an academy scenario, not 11_vs_11).
- BALL, GOAL, SCENARIO, and all existing edge types are unchanged.

The updated graph has 12 nodes (7 PLAYER + 1 BALL + 2 GOAL + 1 SCENARIO + 2 TEAM_SHAPE) and 21 edges (unchanged). This validates against the updated schema because:

1. All previously required fields are still present.
2. New required fields (`line_id`, `lane_id`) have computed values for every PLAYER.
3. TEAM_SHAPE nodes are valid instances of the new `team_shape_node` definition.
4. No formation-related nodes/edges appear for an academy scenario (correct per Track A/B split).

---
## 6. What Remains Genuinely Deferred

### 6.1 FORMATION_ADJACENCY-style relational edges for academy scenarios

The original GNN proposal Section 4 described `FORMATION_ADJACENCY` edges as capturing who is expected to support whom. This relational structure genuinely requires an assumed formation identity: adjacency is defined relative to a template that says player A at slot 5 is expected to support player B at slot 6.

Track Bs template-free approach cannot recover this. Gap-detection on x and y gives us lines and lanes (which players are at similar depth / similar width), but it does not tell us which specific player is expected to support which other player. Two players in the same line and lane are co-located but not necessarily in a support relationship.

This remains deferred. It would require either:
1. Assigning a formation identity to academy scenarios (which Track B explicitly avoids), or
2. Defining a new notion of support based on actual passing data (which requires engine instrumentation -- see Phase 1 5.6).

### 6.2 Updated Phase 1 5.6 deferred appendix

Phase 1 5.6 listed these edge types as deferred:

| Edge type | Phase 1 status | Phase 2 status | Reason |
|-----------|---------------|---------------|--------|
| `FORMATION_ADJACENCY` | Deferred | **Defined for 11_vs_11 only** (Track A). Still deferred for academy scenarios. | Academy scenarios lack formation identity. |
| `FORMATION_LINE` | Deferred | **Defined for 11_vs_11 only** (Track A). Track B provides `line_id` on PLAYER as a continuous alternative. | For academy scenarios, line assignment is data-driven (gap-detection), not template-driven. |
| `FORMATION_LANE` | Deferred | **Defined for 11_vs_11 only** (Track A). Track B provides `lane_id` on PLAYER as a continuous alternative. | For academy scenarios, lane assignment is data-driven (gap-detection), not template-driven. |
| `SEQUENCE_NEXT` | Deferred | **Still deferred** | Requires event timestamps and possession-chain history. |
| `CONSTRAINED_BY` | Deferred | **Still deferred** | No scenario attributes define constraints. |

Phase 1 5.6 listed these features as deferred:

| Feature | Phase 1 status | Phase 2 status | Reason |
|---------|---------------|---------------|--------|
| Line spacing | Deferred | **Partially addressed** by Track B `inter_line_spacing_variance` | Template-free measure provided. Template-based line spacing (distance from formation template) remains deferred. |
| Lane occupancy | Deferred | **Partially addressed** by Track B `num_lanes` and `lane_id` | Template-free measure provided. Template-based lane occupancy remains deferred. |
| Formation deviation | Deferred | **Addressed** by Track B `formation_deviation` (template-free) | Self-referential compactness/symmetry measure provided. Template-based deviation (distance from nearest formation) remains deferred for academy scenarios. |
| Pressure at pass/shot time | Deferred | **Still deferred** | Requires engine instrumentation. |

### 6.3 What would unblock the remaining deferred items

- `SEQUENCE_NEXT`: Requires event timestamps and possession-chain history in the graph. Phase 0 3.2 identified this as requiring new engine instrumentation.
- `CONSTRAINED_BY`: Requires scenario attributes (`target_zone`, `forbidden_actions`, `max_touches`) that do not exist in `ScenarioConfig` (`types/football.ts:197-222`).
- Template-based formation measures for academy scenarios: Requires either (a) assigning formation labels to academy scenarios (rejected by Track B design), or (b) a future phase that explicitly models formation inference as a probabilistic hypothesis rather than a categorical fact.

---

## 7. Summary

| Element | v2 status | Count in worked example (academy_3_vs_1_defender_3) |
|---------|-----------|---------------------------------------------------|
| Node types | PLAYER, BALL, GOAL, SCENARIO, TEAM_SHAPE, FORMATION_SLOT | 12 total (7 PLAYER + 1 BALL + 2 GOAL + 1 SCENARIO + 2 TEAM_SHAPE + 0 FORMATION_SLOT) |
| Edge types | TEAMMATE, OPPONENT, NEAR, POSSESSES, FORMATION_ADJACENCY, FORMATION_LINE, FORMATION_LANE | 21 total (9 TEAMMATE + 12 OPPONENT + 0 NEAR + 0 POSSESSES + 0 formation edges) |
| PLAYER fields from 127-dim obs | Position, velocity, role, role_one_hot, is_active, is_controlled, is_goalkeeper | 7 raw features |
| PLAYER derived features | 7 (nearest-teammate dist, nearest-opponent dist, team_width, team_depth, compactness, stretch, receiver_availability) | 7 derived features |
| PLAYER new fields (Phase 2) | line_id, lane_id | 2 new fields |
| TEAM_SHAPE fields | formation_deviation (inter_line_spacing_variance, mean_line_compactness, num_lines, num_lanes) | 1 new node type with 4 sub-fields |
| FORMATION_SLOT fields | slot_id, team, role, xRatio, yRatio, x, y | 1 new node type (111 only) |
| Deferred edge types | SEQUENCE_NEXT, CONSTRAINED_BY | 2 types |
| Deferred features | Pressure at pass/shot time | 1 feature |
| Deferred scenario fields | max_touches, allowed_actions, forbidden_actions, target_player, target_zone, step_limit | 6 fields |

**Determinism guarantee:** Node ordering is fixed by team and index (left_0..left_10, right_0..right_10), matching the 127-dim observations left-then-right ordering. TEAM_SHAPE nodes are appended after all PLAYER nodes (left first, then right). FORMATION_SLOT nodes (11_vs_11 only) are appended after TEAM_SHAPE nodes. Edge lists are sorted by (source, target) lexicographic ascending within each edge type. The same scenario + seed always produces the same node/edge sequence.

---

*End of Phase 2 formation model document. No source files were modified.*