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
the canonical three-agent measurement, and `eval_progress.py` now load the
architecture declared by the checkpoint. ONNX export validates the checkpoint
and fails with an explicit Gap 4 message for GNN policies; GNN export and browser
inference are not implemented.

## Gap 3 — Checkpoint contract and evaluator compatibility

**Status: Implemented in code; verification pending.**

`checkpoint_contract.py` now defines versioned flat/GNN policy metadata and
validates the environment, flat observation, action, graph schema and feature
dimensions, expected encoder state-dict signatures, and action/value head
shapes. New MAPPO checkpoints and experiment manifests store this contract.
Legacy flat checkpoints without architecture metadata continue to load as
`flat`; GNN checkpoints without a matching contract are rejected.

`load_mappo_actor` and `load_mappo_critic` centralize architecture-aware loading.
`eval_progress.py`, `eval_f_act.py`, and
`eval_canonical_three_agent_measurement.py` now request graph observations and
use the checkpoint's encoder when the checkpoint is GNN-based. ONNX export
validates flat checkpoints and explicitly reports that GNN export is Gap 4 work.

**Not yet verified:** No test, checkpoint load, evaluator run, or training run
was performed for this update. Existing flat-checkpoint compatibility and new
GNN checkpoint rejection/load behavior still need targeted verification.

## Verification results — 2026-09-24

- Focused command
  `python -m pytest training/tests/test_gnn_phase4.py training/tests/test_mappo_rollout_regression.py training/tests/test_action_masks.py -q -p no:cacheprovider`:
  **42 passed**. These cover the existing Phase 4 encoder, rollout regression,
  and action-mask suites; they do not directly exercise the new Gap 1–3
  integration paths.
- `npm test`: **passed**. Scenario validation passed 16/16 scenarios and
  regressions; determinism checks passed for all four configured scenarios.
- Full training Python suite, run with elevated filesystem/process access:
  `python -m pytest training/tests/ -x -q -p no:cacheprovider` reached
  **36 passed, 1 failed**. The failure was
  `test_curriculum_live_e2e.py::TestCurriculumLiveSchedulerPromotion::test_live_scheduler_promotes_on_real_successes_or_records`:
  the test's bridge-backed environment timed out waiting for its WebSocket
  step frame, then could not reconnect to the bridge. A non-fail-fast rerun
  was interrupted after about three minutes without further output; because
  quiet mode did not identify the in-progress test, that run has no aggregate
  result and its stall is not attributed to a specific test.
- An initial sandboxed run stopped earlier because a test could not create its
  temporary `models` directory. The elevated rerun passed that point; it is not
  counted as a code failure.

## Next tasks from verification

1. Diagnose bridge startup/readiness and WebSocket responsiveness for the live
   curriculum integration tests; rerun those tests and then complete the Python
   training suite without interruption.
2. Add focused tests for Gap 1 graph attachment on reset/step (including the
   unchanged flat-observation default), Gap 2 GNN rollout/PPO update and critic
   gradients, and Gap 3 checkpoint contract validation plus flat/GNN evaluator
   loading.
3. Run a small single-environment GNN training/evaluation smoke run after the
   focused integration tests pass. Do not make a policy-quality claim from a
   smoke run.

## Remaining implementation sequence

| Gap | Status | Scope |
|---|---|---|
| 1. Environment graph construction | Implemented; direct integration coverage pending | Test graph per agent on reset/step and the unchanged default observation contract |
| 2. GNN actor/critic in rollout and PPO update | Implemented in code for `n_envs=1`; direct GNN update verification pending | Test gradients and end-to-end updates; graph-aware batched rollout remains open |
| 3. Checkpoint/evaluator contract | Implemented; direct integration coverage pending | Test versioned graph contract, legacy flat loading, and progress/F_act/canonical measurement loaders |
| 4. Deployment export | Open; needed only for browser deployment | Export and run the selected GNN architecture in target runtime |
| 5. Quantitative probe validation | Open research work | Collect data and train probes with held-out splits |
