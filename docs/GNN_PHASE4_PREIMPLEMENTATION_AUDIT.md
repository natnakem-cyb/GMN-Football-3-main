# GNN Phase 4 — Pre-Implementation Audit

Date: 2026-09-13
Scope: Phase 4 GNN encoder implementation

---

## 1. Frozen Graph Input Contract

Source: Phase 3 frozen graph (`ac693b5`)

```text
graph = {
    "scenario": {...},       # SCENARIO node (graph-level context)
    "nodes": [...],          # All nodes in deterministic order
    "edges": [...],          # All edges in deterministic order
    "z_scenario": [...]      # Optional 8-dim task context vector
}
```

Node types (6): `PLAYER`, `BALL`, `GOAL`, `FORMATION_SLOT`, `TEAM_SHAPE`, `SCENARIO`

Edge types (13): `TEAMMATE`, `OPPONENT`, `NEAR`, `POSSESSES`, `PLAYER_BALL`, `PLAYER_GOAL`, `BALL_GOAL`, `ASSIGNED_TO`, `BELONGS_TO_SHAPE`, `SCENARIO_CONTEXT`, `FORMATION_ADJACENCY`, `FORMATION_LINE`, `FORMATION_LANE`

The graph is schema-validated against `training/gnn_graph_schema.json` v3 before entering the adapter.

---

## 2. Available ML Framework

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.14.5 | |
| PyTorch | 2.14.0+cpu | CPU only, no CUDA |
| NumPy | available | |
| Gymnasium | available | |
| PettingZoo | available | |
| Stable-Baselines3 | available | |
| ONNX | available | |
| TensorBoard | available | |

### Graph libraries

| Library | Available | Decision |
|---------|-----------|----------|
| PyTorch Geometric | NO | Cannot use |
| DGL | NO | Cannot use |
| torch_scatter | NO | Cannot use |
| torch_sparse | NO | Cannot use |

**Decision:** Implement GNN message passing using **vanilla PyTorch only**. No external graph libraries. This keeps the dependency footprint minimal and ensures the code runs in the existing environment.

---

## 3. Dependency Decision

**No new dependencies required.**

All Phase 4 code uses only:
- `torch` (already required)
- `numpy` (already required)

GNN operations (message passing, attention, pooling) will be implemented as custom PyTorch modules.

---

## 4. Expected Tensor Shapes

### Single graph (unbatched)

```text
node_features:  (num_nodes, node_feat_dim)
node_type:     (num_nodes,)          # int64 categorical
edge_index:    (2, num_edges)        # int64, source/target indices
edge_type:     (num_edges,)          # int64 categorical
edge_features: (num_edges, edge_feat_dim)
node_mask:     (num_nodes,)          # float32, 1.0 for present nodes
agent_node_indices: List[int]        # indices of controllable player nodes
graph_context: (z_dim,)              # Float32Array[8] from Phase 2
```

### Batched graphs

```text
node_features:  (batch_num_nodes, node_feat_dim)
node_type:     (batch_num_nodes,)
edge_index:    (2, batch_num_edges)
edge_type:     (batch_num_edges,)
edge_features: (batch_num_edges, edge_feat_dim)
node_mask:     (batch_num_nodes,)
batch:         (batch_num_nodes,)    # int64, graph index per node
agent_node_indices: List[List[int]]  # per-graph agent indices
graph_context: (batch_size, z_dim)
```

---

## 5. Batching Strategy

**No automatic batching in Phase 4.**

Each graph is processed individually through the encoder. This avoids:
- Variable node count padding complexity
- Edge index offset computation
- Masking complexity

Batch processing can be added in Phase 5 when integrating with MAPPO.

For diagnostic training, graphs are processed one at a time with batch_size=1.

---

## 6. Agent-Embedding Strategy

Agent embeddings are extracted from the final node embeddings:

```python
agent_embeddings = node_embeddings[agent_node_indices]  # (num_agents, embed_dim)
```

For single-agent scenarios, this is a single vector.
For multi-agent scenarios, this is a set of vectors.

The mapping from `global_id` to `node_index` is maintained in the adapter output.

---

## 7. Global-Embedding Strategy

Global embedding is computed by masked mean pooling over all node embeddings:

```python
global_embedding = (node_embeddings * node_mask.unsqueeze(-1)).sum(dim=0) / node_mask.sum()
```

This is permutation-invariant and handles variable node counts.

---

## 8. Normalization Strategy

### Continuous features

| Feature | Raw range | Normalization | Clipping |
|---------|-----------|---------------|----------|
| position.x | [-1.0, 1.0] | None (already normalized) | [-1.0, 1.0] |
| position.y | [-0.42, 0.42] | None (already normalized) | [-0.42, 0.42] |
| velocity.vx/vy | scaled by 50 | Divide by 50 | [-1.0, 1.0] |
| distance | [0, ~2.83] | Divide by sqrt(2) (max pitch diagonal) | [0, 1] |
| relative_x | [-2.0, 2.0] | Divide by 2.0 | [-1.0, 1.0] |
| relative_y | [-0.84, 0.84] | Divide by 0.84 | [-1.0, 1.0] |
| angle | [-pi, pi] | Divide by pi | [-1.0, 1.0] |
| closing_speed | unbounded | None | Clip to [-1.0, 1.0] |
| shot_angle | [0, ~0.7] | None | [0, 1] |
| deviation_distance | [0, ~2.83] | Divide by sqrt(2) | [0, 1] |

### Categorical features

- `node_type`: learned embedding (6 types)
- `edge_type`: learned embedding (13 types)
- `role`: learned embedding (12 roles)

---

## 9. Diagnostic Protocol

Diagnostics are lightweight supervised probes attached to frozen GNN embeddings:

```python
class DiagnosticProbe(nn.Module):
    def forward(self, embeddings, targets):
        ...
```

Probes:
1. **SpatialGeometryProbe**: Predict player-player distance, player-ball distance, player-goal distance
2. **DirectionProbe**: Predict bearing to goal, goal-mouth angle
3. **PressureProbe**: Predict nearest-opponent distance
4. **FormationProbe**: Predict formation identity, slot role, line, lane
5. **GoalGeometryProbe**: Predict shot distance, shot angle
6. **PassingProbe**: Predict teammate distance, direction, relative velocity
7. **ScenarioProbe**: Predict scenario ID from graph embedding + zScenario

Each probe is trained independently on frozen encoder outputs.

---

## 10. Resource/Performance Constraints

- **CPU-only training**: No GPU available on this machine
- **Model size**: Keep encoder < 100K parameters for lightweight experimentation
- **Inference time**: < 10ms per graph on CPU
- **Memory**: < 100MB per graph batch
- **Batch size**: 1 (no batching in Phase 4)

---

## 11. Key Decisions

1. **Vanilla PyTorch only** — No PyTorch Geometric, DGL, or other graph libraries
2. **No automatic batching** — Process one graph at a time
3. **No GNN training for policy** — Diagnostics only, no MAPPO integration
4. **Frozen Phase 3 graph** — Only bug fixes for tensor consumption, no schema changes
5. **Lightweight encoders** — 2-3 message-passing layers, modest hidden dimensions

---

## 12. Risk Notes

- **CPU-only training** will be slow for larger experiments. Keep models small.
- **Vanilla PyTorch GNN** requires manual message-passing implementation. This is acceptable for Phase 4's lightweight scope.
- **No batching** means diagnostic training will be slow. Acceptable for Phase 4.
- **Formation data limited** — Only 11_vs_11 has formation slots currently. Formation probes will have limited data.
