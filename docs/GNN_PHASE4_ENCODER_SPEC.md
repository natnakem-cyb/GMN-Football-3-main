# GNN Phase 4: Encoder Specification

## 1. Overview

This document specifies the three encoder architectures implemented in `training/gnn_encoders.py` for the GMN Football project. Each encoder consumes a graph-structured observation (`GraphTensor`) and produces two embedding vectors:

- **agent embeddings** — per-controlled-player representations used by the policy head.
- **global embedding** — a single graph-level representation fused with scenario context (`zScenario`), used by the value head and other global conditioning paths.

All encoders share the same input/output contract and are instantiated via a factory function. None of the encoders perform batching; they operate on a single graph at a time.

## 2. Encoder Comparison Table

| Encoder | Graph Usage | Message Passing | Coordinate Update | Estimated Parameters |
|---|---|---|---|---|
| `MLPBaselineEncoder` | Flat node set | None | None | ~200 K |
| `GATEncoder` | Fully connected with typed edges | Multi-head graph attention (4 heads) | None | ~500 K |
| `GeometryAwareEncoder` | Spatial graph with k-nn / typed edges | 3 EGNN-style message passing steps | Equivariant coordinate MLP | ~700 K |

## 3. MLPBaselineEncoder Specification

### Architecture

```
node_features (num_nodes, 32)
       │
       ▼
  node_encoder (shared across all nodes)
       │
       ├─ Linear(32 → 128)
       ├─ LayerNorm
       ├─ ReLU
       ├─ Linear(128 → 128)
       ├─ LayerNorm
       ├─ ReLU
       │    (3-layer MLP block repeated twice inside the encoder)
       ▼
  masked mean pool (over valid nodes)
       │
       ▼
  context_fusion
       ├─ Linear(128 + 8 → 128)
       ├─ LayerNorm
       ├─ ReLU
       │
       ├──► agent_head: Linear(128 → 128)
       │    → agent_embeddings (num_agents, 128)
       │
       └──► global_head: Linear(128 → 128)
            → global_embedding (128,)
```

### Forward Pass

1. Encode every node independently through the shared `node_encoder` MLP.
2. Apply `node_mask` to zero out padded/invalid nodes.
3. Compute a masked mean over the node dimension to obtain a single graph vector.
4. Concatenate the graph vector with `graph_context` (`zScenario`, 8-dim).
5. Fuse via `context_fusion`.
6. Split into `agent_head` and `global_head` branches.

### Interface

- **Input:** `GraphTensor`
  - `node_features`: `(num_nodes, 32)`
  - `node_mask`: `(num_nodes,)` — binary mask; `1` for valid nodes, `0` for padding.
  - `edge_index`: not used (ignored).
  - `edge_features`: not used (ignored).
  - `graph_context`: `(8,)`
- **Output:** `(agent_embeddings, global_embedding)`
  - `agent_embeddings`: `(num_agents, 128)`
  - `global_embedding`: `(128,)`

## 4. GATEncoder Specification

### Architecture

```
node_features (num_nodes, 32)
       │
       ▼
  input_proj: Linear(32 → 128)
       │
       ▼
  GraphAttentionLayer × 2–3
       │
       ├─ query_proj: Linear(128 → 128)
       ├─ key_proj:   Linear(128 → 128)
       ├─ value_proj: Linear(128 → 128)
       ├─ edge_proj:  Linear(10 → 128)
       │
       ├─ Multi-head scaled dot-product attention (4 heads, head_dim = 32)
       │   Attention scores are biased by projected edge features.
       │
       ├─ Residual connection (pre-attention features)
       ├─ LayerNorm
       ▼
  masked mean pool
       │
       ▼
  context_fusion
       ├─ Linear(128 + 8 → 128)
       ├─ LayerNorm
       ├─ ReLU
       │
       ├──► agent_head: Linear(128 → 128)
       │    → agent_embeddings (num_agents, 128)
       │
       └──► global_head: Linear(128 → 128)
            → global_embedding (128,)
```

### Attention Mechanism

For each attention layer:

1. Project node features into query, key, and value spaces (128-dim).
2. Project edge features (10-dim) into the same hidden dimension.
3. Compute scaled dot-product attention scores: `softmax((Q K^T) / sqrt(d_k) + edge_bias)`.
4. Aggregate values using the attention weights.
5. Apply residual connection and LayerNorm.

Edge features act as additive bias terms in the attention score matrix, allowing the model to condition inter-node interactions on relationship type, distance, and possession state.

### Forward Pass

1. Project raw node features to 128-dim via `input_proj`.
2. Stack 2–3 `GraphAttentionLayer` modules, each consuming the full edge structure (`edge_index`, `edge_features`).
3. Mask invalid nodes before pooling.
4. Masked mean pool → graph vector.
5. Fuse with `graph_context`.
6. Split into agent and global heads.

### Interface

- **Input:** `GraphTensor`
  - `node_features`: `(num_nodes, 32)`
  - `node_mask`: `(num_nodes,)`
  - `edge_index`: `(2, num_edges)` — required.
  - `edge_features`: `(num_edges, 10)` — required.
  - `graph_context`: `(8,)`
- **Output:** `(agent_embeddings, global_embedding)`
  - `agent_embeddings`: `(num_agents, 128)`
  - `global_embedding`: `(128,)`

## 5. GeometryAwareEncoder Specification

### Architecture

```
Initialize node positions x from node_features (first 2 dims or learned)
       │
       ▼
  Initialize node features h from node_features (32-dim → 128-dim via Linear)
       │
       ▼
  3 × GeometryAwareMessagePassingStep
       │
       ├─ edge_mlp:  Linear(2*128 + 10 + 1 → 128) → LayerNorm → ReLU
       │              → Linear(128 → 128) → LayerNorm → ReLU
       │
       ├─ feature_mlp: Linear(32 + 128 → 128) → LayerNorm → ReLU
       │                 → Linear(128 → 128) → LayerNorm → ReLU
       │
       ├─ coord_mlp:  Linear(128 → 64) → ReLU
       │               → Linear(64 → 2)
       │
       ├─ Message: m_ij = edge_mlp([h_i, h_j, edge_features_ij, ||x_i - x_j||])
       ├─ Aggregate: sum over incoming edges
       ├─ Update h: h_i ← feature_mlp([node_features_i, h_i + aggregated_message])
       └─ Update x: x_i ← x_i + coord_mlp(h_i)   (equivariant translation)
       │
       ▼
  masked mean pool over final h
       │
       ▼
  context_fusion
       ├─ Linear(128 + 8 → 128)
       ├─ LayerNorm
       ├─ ReLU
       │
       ├──► agent_head: Linear(128 → 128)
       │    → agent_embeddings (num_agents, 128)
       │
       └──► global_head: Linear(128 → 128)
            → global_embedding (128,)
```

### Equivariant Update

- **Feature update:** Node features are updated via an MLP conditioned on the original 32-dim node features and the aggregated edge message. This preserves permutation equivariance over nodes.
- **Coordinate update:** Coordinates are updated by a separate MLP (`coord_mlp`) applied to the node feature. The update is added to the current coordinates, producing an equivariant translation under rotation and translation of the input geometry.

### Forward Pass

1. Extract initial 2D coordinates from `node_features` (or initialize as zeros if not present).
2. Initialize hidden features `h` by projecting `node_features` to 128-dim.
3. Run 3 message-passing steps:
   - Compute pairwise messages for all edges using `edge_mlp`.
   - Aggregate messages to target nodes.
   - Update `h` via `feature_mlp`.
   - Update `x` via `coord_mlp` (equivariant).
4. Masked mean pool over final `h`.
5. Fuse with `graph_context`.
6. Split into agent and global heads.

### Interface

- **Input:** `GraphTensor`
  - `node_features`: `(num_nodes, 32)` — first 2 dimensions are interpreted as (x, y) coordinates.
  - `node_mask`: `(num_nodes,)`
  - `edge_index`: `(2, num_edges)` — required.
  - `edge_features`: `(num_edges, 10)` — required.
  - `graph_context`: `(8,)`
- **Output:** `(agent_embeddings, global_embedding)`
  - `agent_embeddings`: `(num_agents, 128)`
  - `global_embedding`: `(128,)`

## 6. Common Interface

### GraphTensor

All encoders accept a single `GraphTensor` object with the following fields:

| Field | Shape | Description |
|---|---|---|
| `node_features` | `(num_nodes, 32)` | Raw per-node observation vector. |
| `node_mask` | `(num_nodes,)` | Binary mask; `1` = valid node, `0` = padding / absent. |
| `edge_index` | `(2, num_edges)` | Source/target node indices for each edge. |
| `edge_features` | `(num_edges, 10)` | Per-edge feature vector (type, distance, possession, etc.). |
| `graph_context` | `(8,)` | Scenario-level context vector (`zScenario`). |

### Output Contract

All encoders return a tuple:

```python
agent_embeddings: (num_agents, output_dim)
global_embedding: (output_dim,)
```

- `num_agents` is derived from `node_mask` and the agent type bit in `node_features`.
- `output_dim` is 128 for all three encoders.
- `global_embedding` is the only output that incorporates `graph_context`; `agent_embeddings` do not.

### Context Fusion

`graph_context` is concatenated to the pooled graph vector and passed through a shared `context_fusion` MLP (`Linear(128+8 → 128) → LayerNorm → ReLU`) before splitting into the agent and global heads. This ensures scenario information influences both the per-agent policy and the value estimate.

## 7. Factory Pattern

Encoders are instantiated via the `create_encoder` factory function:

```python
from training.gnn_encoders import create_encoder

encoder = create_encoder(
    encoder_type="mlp_baseline",  # or "gat", "geometry_aware"
    **kwargs
)
```

The factory accepts arbitrary keyword arguments and forwards them to the selected encoder constructor. This allows centralized configuration of hidden dimensions, number of attention heads, message-passing steps, and other hyperparameters without changing calling code.

## 8. Determinism Guarantees

All three encoders are deterministic given identical inputs and weights:

- **No stochastic layers:** No dropout, no random edge subsampling.
- **Deterministic operations:** All matrix multiplications, LayerNorm, and ReLU are deterministic in PyTorch.
- **Masked mean pool:** Summation and division are performed with fixed masks; no dynamic graph structure changes during the forward pass.
- **Edge ordering:** Edges are sorted lexicographically within type before being passed to the encoder, ensuring consistent edge feature ordering across runs.

## 9. Extension Points

To add a new encoder:

1. **Implement the interface.** Subclass `nn.Module` and implement `forward(self, graph: GraphTensor) -> Tuple[Tensor, Tensor]`.
2. **Respect the input contract.** Accept `node_features`, `node_mask`, `edge_index`, `edge_features`, and `graph_context` from the `GraphTensor`.
3. **Return the standard shape.** Output must be `(num_agents, output_dim)` and `(output_dim,)`.
4. **Fuse context.** Use the same `context_fusion` pattern (or equivalent) to inject `graph_context` into the global embedding.
5. **Register in the factory.** Add a new branch to `create_encoder` mapping a string key (e.g., `"transformer"`) to the new class.
6. **Document.** Add a section in this spec following the structure of Sections 3–5.

New encoders should preserve determinism and masked mean pooling unless there is a specific reason to deviate, and they should not introduce batching semantics that conflict with the single-graph-at-a-time contract.
