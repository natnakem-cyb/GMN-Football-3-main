# GNN Phase 4 Diagnostic Protocol

## 1. Overview

Phase 4 validates that the GNN encoder learns semantically meaningful representations by training lightweight probe heads on frozen encoder outputs. Each probe targets a specific football-specific property: spatial geometry, directional awareness, pressure, formation identity, goal geometry, passing lanes, or scenario class. Probes are trained independently without backpropagation into the encoder, allowing measurement of how much task-relevant information is already captured in the fixed embeddings.

## 2. Probe Catalog

| Probe | Task Type | Targets | Input Dim | Output Dim | Architecture |
|---|---|---|---|---|---|
| SpatialGeometryProbe | Regression | player-player distance, player-ball distance, player-goal distance | `input_dim` | 3 | Linear -> ReLU -> Linear |
| DirectionProbe | Regression | bearing to goal, goal-mouth opening angle | `input_dim` | 2 | Linear -> ReLU -> Linear |
| PressureProbe | Regression | nearest-opponent distance | `input_dim` | 1 | Linear -> ReLU -> Linear |
| FormationProbe | Classification | formation identity + slot role + line + lane | `input_dim` | 20 (1+12+4+3) | Linear -> ReLU -> Linear |
| GoalGeometryProbe | Regression | shot distance, shot angle | `input_dim` | 2 | Linear -> ReLU -> Linear |
| PassingProbe | Regression | teammate distance, direction, relative velocity | `input_dim` | 4 | Linear -> ReLU -> Linear |
| ScenarioProbe | Classification | scenario identity | `input_dim` | `num_scenarios` | Linear -> ReLU -> Linear |

## 3. Training Protocol

- **Frozen encoder**: The GNN encoder is held fixed for the entire Phase 4 diagnostic run. No gradients flow back into the encoder parameters.
- **Independent probe training**: Each probe is trained as a standalone MLP head on top of the frozen embeddings. Probes do not share parameters or influence each other.
- **Embedding source**: For agent-specific probes, extract the per-agent embedding from the encoder. For global probes, extract the graph-level embedding.
- **Loss functions**:
  - Regression probes: Mean Squared Error (MSE)
  - Classification probes: Cross-entropy loss
- **Optimization**: Train each probe with its own optimizer on the frozen embedding dataset until convergence on a held-out validation split.

## 4. Evaluation Metrics

### Regression Probes
- **Mean Absolute Error (MAE)**: average magnitude of errors across all target dimensions
- **Root Mean Squared Error (RMSE)**: penalizes large errors more heavily
- **R-squared (R2)**: proportion of variance explained by the probe

### Classification Probes
- **Accuracy**: fraction of correctly classified samples
- **F1 score**: harmonic mean of precision and recall, reported macro-averaged across classes

## 5. Data Splitting

- **Episode-level split**: All snapshots from the same episode must remain within the same split. Randomly assign whole episodes to train, validation, or test to prevent data leakage.
- **In-distribution split**: Standard train/validation/test split drawn from the same scenario distribution seen during encoder training.
- **Held-out scenarios**: Reserve a subset of scenario configurations that are never seen during encoder pretraining. Use these exclusively for out-of-distribution evaluation to measure generalization.

## 6. Scenario-Conditioning Test

- Collect matched graph states where the only difference is the `zScenario` field while spatial configuration, player positions, and game state are held constant.
- Train a probe on one scenario's embeddings and evaluate on the matched state from a different scenario.
- If the probe generalizes poorly, the encoder is encoding scenario-specific artifacts rather than invariant spatial features.

## 7. Ablation Protocol

- Train each probe on embeddings from the **full graph** (all node and edge types present) to establish a baseline.
- Retrain the same probe on embeddings from ablated graphs with one component removed at a time:
  - Remove `TEAMMATE` edges
  - Remove `OPPONENT` edges
  - Remove `NEAR` edges
  - Remove `POSSESSES` edges
  - Remove non-`PLAYER` node types (`BALL`, `GOAL`, `SCENARIO`)
- Compare probe performance across ablations. Large drops indicate that the ablated component carries critical signal for that probe's target.

## 8. Expected Outcomes

- Probes should achieve above-random performance on all targets, confirming the encoder captures football-relevant geometry.
- Regression probes on core spatial targets (player-player distance, player-ball distance, bearing to goal) should yield low MAE/RMSE and high R2 on in-distribution test episodes.
- Classification probes for formation and scenario should exceed majority-class accuracy.
- Scenario-conditioning tests should show meaningful degradation, revealing scenario bias in the encoder.
- Ablation results should highlight which edge and node types are most informative for each semantic property, guiding future architecture refinements.
