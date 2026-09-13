# GNN Phase 3 Edge Directionality Specification

## 1. Overview

This document specifies the directionality contract for every edge type emitted by
`training/gnn_graph_builder.py`. It is the single source of truth for how edges are
stored, how reverse geometry is derived, and how the final edge list is ordered for
deterministic tensorization into `edge_index` / `edge_attr`.

### Scope

- v1 node types: `PLAYER`, `BALL`, `GOAL`, `SCENARIO`
- v1 edge types: `TEAMMATE`, `OPPONENT`, `NEAR`, `POSSESSES`, `PLAYER_BALL`,
  `PLAYER_GOAL`, `BALL_GOAL`, `ASSIGNED_TO`, `BELONGS_TO_SHAPE`,
  `SCENARIO_CONTEXT`, `FORMATION_ADJACENCY`, `FORMATION_LINE`, `FORMATION_LANE`

---

## 2. Directionality Contract Table

| Edge Type | Directed | Reverse Edge Stored | Geometry Symmetric | Notes |
|-----------|----------|---------------------|--------------------|-------|
| `TEAMMATE` | No | No | Partial | Single stored edge per unordered pair `i < j`; features from source perspective; closing speed is symmetric, all other spatial features require negation + angle offset |
| `OPPONENT` | No | No | Partial | Single stored edge per cross-team pair; features from left-player perspective; pressure is distance-only and symmetric |
| `NEAR` | No | No | Yes | Single stored edge per unordered pair `i < j`, cross-team only; distance is symmetric; threshold is constant |
| `POSSESSES` | Yes | No | N/A | `player -> ball`; semantic is inherently directed; no reverse edge |
| `PLAYER_BALL` | Yes | No | N/A | `player -> ball`, one edge per player; features from player perspective |
| `PLAYER_GOAL` | Yes | No | N/A | `player -> goal`, one edge per player per goal (2 goals); features from player perspective |
| `BALL_GOAL` | Yes | No | N/A | `ball -> goal`, one edge per goal; features from ball perspective |
| `ASSIGNED_TO` | Yes | No | N/A | `player -> formation_slot`, one edge per player when slots exist |
| `BELONGS_TO_SHAPE` | Yes | No | N/A | `player -> team_shape`, one edge per player |
| `SCENARIO_CONTEXT` | Yes | No | N/A | `player -> scenario`, one edge per player |
| `FORMATION_ADJACENCY` | No | No | Yes | Undirected between slots, `k=2` nearest neighbors in template space; no features |
| `FORMATION_LINE` | No | No | Yes | Undirected between slots in same line band; `line_id` feature |
| `FORMATION_LANE` | No | No | Yes | Undirected between slots in same lane band; `lane_id` feature |

---

## 3. Per-Type Directionality Rules

### 3.1 TEAMMATE

- **Builder**: `_build_rich_teammate_edges`
- **Loop**: `i < j` over same-team players
- **Storage**: Exactly one edge per unordered teammate pair
- **Source perspective**: Features computed as `target - source` (e.g., `relative_x = target.x - source.x`)
- **Reverse derivation**: Negate `relative_x`, `relative_y`, `relative_vx`, `relative_vy`; add `pi` to `angle`
- **Symmetric component**: `closing_speed` is identical for both directions because it depends on the distance derivative, which is antisymmetric and cancels in magnitude

### 3.2 OPPONENT

- **Builder**: `_build_rich_opponent_edges`
- **Loop**: `left_players x right_players`
- **Storage**: Exactly one edge per cross-team pair
- **Source perspective**: Features from left-player perspective
- **Symmetric component**: `pressure` depends only on distance and is therefore symmetric

### 3.3 NEAR

- **Builder**: `_build_edges`
- **Loop**: `i < j` over all players, cross-team only
- **Storage**: Exactly one edge per unordered cross-team pair within proximity threshold
- **Symmetric component**: `distance` is symmetric; threshold is a constant

### 3.4 POSSESSES

- **Direction**: `player -> ball`
- **Storage**: One directed edge per possessing player; no reverse edge
- **Semantics**: "Player possesses ball" is inherently directed; a ball does not possess a player

### 3.5 PLAYER_BALL

- **Direction**: `player -> ball`
- **Storage**: One edge per player
- **Features**: From player perspective (`relative_x = ball.x - player.x`, etc.)
- **Derived flag**: `has_possession` is player-specific boolean

### 3.6 PLAYER_GOAL

- **Direction**: `player -> goal`
- **Storage**: One edge per player per goal (2 goals total)
- **Features**: From player perspective

### 3.7 BALL_GOAL

- **Direction**: `ball -> goal`
- **Storage**: One edge per goal (2 goals total)
- **Features**: From ball perspective

### 3.8 ASSIGNED_TO

- **Direction**: `player -> formation_slot`
- **Storage**: One edge per player (when formation slots exist)
- **Semantics**: "Player is assigned to this slot"

### 3.9 BELONGS_TO_SHAPE

- **Direction**: `player -> team_shape`
- **Storage**: One edge per player
- **Semantics**: "Player belongs to this team shape"

### 3.10 SCENARIO_CONTEXT

- **Direction**: `player -> scenario`
- **Storage**: One edge per player
- **Semantics**: "Player is operating under this scenario"

### 3.11 FORMATION_ADJACENCY

- **Direction**: Undirected between slots
- **Construction**: `k=2` nearest neighbors in template space
- **Features**: None
- **Storage**: One edge per slot adjacency pair

### 3.12 FORMATION_LINE

- **Direction**: Undirected between slots
- **Condition**: Slots in the same line band
- **Features**: `line_id`

### 3.13 FORMATION_LANE

- **Direction**: Undirected between slots
- **Condition**: Slots in the same lane band
- **Features**: `lane_id`

---

## 4. Geometry Reversal Formulas

For undirected edges where a single edge is stored with source-relative features,
the reverse edge features are derived as follows:

| Feature | Reverse Formula |
|---------|-----------------|
| `relative_x` | `-relative_x` |
| `relative_y` | `-relative_y` |
| `relative_vx` | `-relative_vx` |
| `relative_vy` | `-relative_vy` |
| `angle` | `angle + pi` (normalize to `(-pi, pi]` or `[0, 2pi)`) |
| `closing_speed` | `closing_speed` (unchanged; symmetric) |
| `pressure` | `pressure` (unchanged; symmetric, distance-only) |
| `distance` | `distance` (unchanged; symmetric) |
| `line_id` | `line_id` (unchanged; symmetric) |
| `lane_id` | `lane_id` (unchanged; symmetric) |

**Applicability**: These formulas apply to `TEAMMATE`, `OPPONENT`, `NEAR`,
`FORMATION_LINE`, and `FORMATION_LANE`. Directed edge types (`POSSESSES`,
`PLAYER_BALL`, `PLAYER_GOAL`, `BALL_GOAL`, `ASSIGNED_TO`, `BELONGS_TO_SHAPE`,
`SCENARIO_CONTEXT`) do not have reverse edges and these formulas do not apply.
`FORMATION_ADJACENCY` has no features.

---

## 5. Sorting and Determinism

All edges MUST be sorted by the composite key:

```
(edge_type, source_node_id, target_node_id)
```

### Ordering Rules

1. **edge_type**: Lexicographic order of the edge type string
2. **source**: Numeric or lexicographic order of the source node ID
3. **target**: Numeric or lexicographic order of the target node ID

### Requirements

- Node ordering within each type is fixed by `(team, index)` matching the
  127-dim observation vector ordering
- Edges within each type are sorted lexicographically by `(source, target)`
- The sort key must be computed from stable node identifiers, not from
  runtime-assigned indices that may vary across runs

### Rationale

Deterministic ordering ensures that:

- `edge_index` and `edge_attr` tensors are identical across runs with the same
  input state
- Checkpoint comparisons and regression tests are stable
- GNN message passing is reproducible given identical graph inputs

---

## 6. Implications for GNN Tensorization

### 6.1 Edge Index Construction

The sorted edge list is converted to PyTorch Geometric-style `edge_index`:

```
edge_index = torch.tensor([[source_0, source_1, ...],
                           [target_0, target_1, ...]], dtype=torch.long)
```

- `edge_index[0]` contains source node indices
- `edge_index[1]` contains target node indices
- Row `i` of `edge_index` corresponds to row `i` of `edge_attr`

### 6.2 Edge Attribute Construction

`edge_attr` is a 2D tensor of shape `(num_edges, feature_dim)` where:

- Each row corresponds to one edge in sorted order
- Feature dimensions are consistent within an edge type but may vary across types
- Missing features for a type are zero-padded to the maximum feature dimension
  across all types, or the tensor is constructed as a list of type-specific
  tensors concatenated along the edge dimension

### 6.3 Directed vs Undirected Handling

- **Directed edges** (`POSSESSES`, `PLAYER_BALL`, `PLAYER_GOAL`, `BALL_GOAL`,
  `ASSIGNED_TO`, `BELONGS_TO_SHAPE`, `SCENARIO_CONTEXT`): Stored as-is; no
  reverse edges are generated
- **Undirected edges** (`TEAMMATE`, `OPPONENT`, `NEAR`, `FORMATION_ADJACENCY`,
  `FORMATION_LINE`, `FORMATION_LANE`): Stored as a single directed edge per
  unordered pair. If a GNN layer requires symmetric message passing, the
  implementation must either:
  - Use the reversal formulas in Section 4 to materialize reverse edges at
    tensorization time, or
  - Use an undirected GNN layer that internally symmetrizes messages

### 6.4 Batch Construction

When batching multiple graphs:

- Edge indices are offset by the cumulative node count of preceding graphs
- The sort key `(edge_type, source, target)` is applied per-graph before
  batching, then global re-sorting is NOT required if node IDs are unique
  across the batch (e.g., prefixed with graph ID)

### 6.5 Type Embeddings

If the GNN uses type-conditioned message passing:

- `edge_type` integer IDs are assigned in lexicographic order
- The mapping from string type to integer ID is fixed and deterministic
- `edge_type` can be stored as a separate `edge_type` tensor of shape
  `(num_edges,)` or encoded into `edge_attr` as a one-hot or learned embedding
  lookup
