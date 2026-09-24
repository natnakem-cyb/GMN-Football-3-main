# GNN Implementation Progress

Updated: 2026-09-24

This document tracks implementation of the GNN gaps one at a time. “Implemented
in code” does not imply tests, training, or behavioral performance have been
verified. No tests or training runs have been executed for these changes yet.

## Gap 1 — Build graph observations from environment reset and step

**Status: Implemented in code; verification pending.**

`GMNMultiAgentEnv` accepts `include_graph_observations=False`. When enabled,
reset and step attach one graph per controlled agent to
`info[agent_id]["graph_observation"]`. The direct batched helpers attach a
`graph_observations` mapping to their shared info dictionary. `reset_one` and
the standard reset/step wrapper paths are also covered. The default remains off,
and the flat observation plus action-mask contract is unchanged.

The environment passes each agent's own observation to the graph builder and
sets that agent as controlled. Bridge player IDs are one-based (`left_1`), while
the graph schema uses zero-based roster IDs (`left_0`); the adapter translates
between those conventions. Scenarios with an `_onball` suffix use the matching
base scenario definition when the graph schema has no separate entry.

The GNN builder is imported lazily, so users of the existing flat-observation
environment do not load GNN dependencies. Unsupported graph scenarios fail
explicitly when graph observations are requested.

## Gap 2 — GNN policy and MAPPO rollout/update path

**Status: Implemented in code for single-environment training; verification pending.**

Added `GNNMAPPOActor` and `GNNMAPPOCritic`, which tensorize graph dictionaries,
run a selected Phase 4 encoder (`gat`, `geometry`, or `mlp`), and produce masked
categorical actions and centralized scalar values. `collect_rollout` retains
per-agent graphs, samples actions from them, and uses the first controlled
agent's graph for the shared critic value/bootstrap. `ppo_update` replays the
stored graphs through both models so gradients flow through the encoders.

`train_mappo.py` exposes `--policy-architecture` with `flat` as the unchanged
default and the three GNN options. GNN training currently requires `--n-envs 1`;
the batched MAPPO collector is not graph-aware. Progress evaluation can load the
GNN actor and requests graph info from the environment. Checkpoints and the run
manifest record the architecture string.

**Not yet verified:** No tests, compilation check, live bridge run, or training
run was performed. The GNN policy path has no performance claim. `eval_f_act.py`,
ONNX export, and browser inference are not integrated.

## Remaining implementation sequence

| Gap | Status | Scope |
|---|---|---|
| 1. Environment graph construction | Implemented in code; verification pending | Graph per agent on reset/step, including batched helpers |
| 2. GNN actor/critic in rollout and PPO update | Implemented in code for `n_envs=1`; verification pending | Validate gradients and end-to-end updates; graph-aware batched rollout remains open |
| 3. Checkpoint/evaluator contract | Partially implemented | Architecture string is saved and `eval_progress.py` loads it; complete canonical evaluator/checkpoint-contract coverage |
| 4. Deployment export | Open; needed only for browser deployment | Export and run the selected GNN architecture in target runtime |
| 5. Quantitative probe validation | Open research work | Collect data and train probes with held-out splits |

