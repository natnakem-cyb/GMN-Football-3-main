# GNN Implementation Progress

Updated: 2026-09-24

This document tracks implementation and verification of the GNN gaps one at a
time. A smoke run verifies that the requested path executes; it does not establish
policy quality or production readiness.

## Gap 1 — Build graph observations from environment reset and step

**Status: Implemented; single-environment reset/step and flat-default behavior verified.**

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

Focused tests exercise graph attachment after reset and step, verify that the
default flat observation/action-mask response is unchanged, and verify the
`_onball` scenario mapping and controlled-player ID translation. Batched graph
helper coverage remains open.

## Gap 2 — GNN policy and MAPPO rollout/update path

**Status: Implemented and verified for single-environment `gnn:mlp`; batched GNN training remains open.**

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

Focused tests verify graph rollout storage and nonzero actor and critic encoder
updates. A 256-step live-bridge `gnn:mlp` smoke run completed one PPO update and
reloaded/forwarded the saved checkpoint. A real deterministic two-episode
evaluation also completed. This is execution evidence only; it is not a policy
quality result. `eval_f_act.py`,
the canonical three-agent measurement, and `eval_progress.py` now load the
architecture declared by the checkpoint. ONNX export validates the checkpoint
and fails with an explicit Gap 4 message for GNN policies; GNN export and browser
inference are not implemented.

## Gap 3 — Checkpoint contract and evaluator compatibility

**Status: Implemented and focused contract/evaluator loading tests passed.**

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

Tests cover legacy flat checkpoint loading, matching GNN contract loading for
all three encoder families, rejection of missing or incompatible GNN contract
metadata, and flat/GNN actor loading in F_act and canonical measurement
evaluators. The live smoke also exercised the GNN `eval_progress.py` path.

## Verification results — 2026-09-24 (latest run)

- Focused command
  `python -m pytest training/tests/test_gnn_phase4.py training/tests/test_mappo_rollout_regression.py training/tests/test_action_masks.py -q -p no:cacheprovider`:
  **42 passed**. These cover the existing Phase 4 encoder, rollout regression,
  and action-mask suites; they do not directly exercise the new Gap 1–3
  integration paths.
- `python -m pytest training/tests/test_gnn_policy_integration.py -q -p no:cacheprovider`:
  **8 passed**. This includes reset/step graph attachment, the flat default,
  GNN rollout/PPO actor and critic updates, checkpoint contracts, evaluator
  loading, and the Windows listener-PID parser regression.
- `npm test`: **passed**. Scenario validation passed 16/16 scenarios and
  regressions; determinism checks passed for all four configured scenarios.
- The isolated live curriculum test no longer timed out. It **skipped** its
  promotion assertion because no goal was scored in ten episodes, as designed
  by its skip path.
- Full Python suite `python -m pytest training/tests/ -q -p no:cacheprovider`:
  **318 passed, 1 skipped, 4 failed** in 1493.98 seconds. The remaining failures:
  `test_curriculum_live_e2e.py::TestTrainMappoCurriculumLiveSmoke::test_train_mappo_curriculum_live_smoke_promotes_or_records` (expected the
  seeded curriculum state to promote to index 1, but it remained at index 0);
  `test_reward_exploits.py::TestRewardExploits::test_policy_a_pass_spam_long`;
  `test_reward_exploits.py::TestRewardExploits::test_policy_b_pass_spam_short`
  (both expect pass-spam policies to complete zero passes); and
  `test_reward_whole_pipeline.py::TestLiveOnePassPipeline::test_live_single_pass_single_event_and_engine_reward`
  (no completed pass observed within 600 ticks).
- Bridge diagnosis and correction: the Python wrapper started the bridge with
  unread stdout/stderr pipes while the TypeScript server logged on every
  single-agent WebSocket step. Once the pipe filled, Node could block and the
  environment timed out waiting for a frame. The wrapper now inherits output
  streams, and the per-step debug line was removed. Windows process cleanup also
  parsed the wrong `netstat` column and killed only the `npx` parent; it now
  extracts listeners from the `LISTENING` column and terminates the process tree
  for that exact bridge port. A startup log exposed the resulting stale-listener
  `EADDRINUSE` collision.
- GNN live smoke: `gnn:mlp`, `academy_empty_goal`, seed `93024`, one environment,
  **256 training steps / one PPO update**, followed by a real deterministic
  **2-episode** evaluation. Training and checkpoint reload/forward pass
  completed; `validate_policy_checkpoint` returned valid. Evaluation reported
  0% goal rate and 0 passes/shots in this tiny sample. These numbers are smoke
  diagnostics only and support no policy-quality conclusion. Local checkpoint
  artifacts are under `runs/gnn_gap3_smoke_artifacts/`.

## Next tasks from verification

1. Fix the curriculum-training smoke: establish why the pre-seeded scheduler
   state is not advanced by the tiny live trainer run.
2. Investigate the two pass-spam regressions and the 600-tick live pass-pipeline
   failure; confirm whether the engine behavior or the tests' zero-pass
   expectations are incorrect, then update code/tests with measured evidence.
3. Add coverage for Gap 1 batched graph helpers and exercise GNN `gat` and
   `geometry` beyond checkpoint construction/loading. Graph-aware batched MAPPO
   remains unimplemented.
4. Keep ONNX/browser deployment and held-out diagnostic-probe validation open
   until those capabilities are explicitly needed and measured.

## Remaining implementation sequence

| Gap | Status | Scope |
|---|---|---|
| 1. Environment graph construction | Single-env reset/step and flat default verified | Add batched helper coverage |
| 2. GNN actor/critic in rollout and PPO update | `gnn:mlp` rollout/update tests and 256-step live smoke passed | Exercise `gat`/`geometry`; graph-aware batched rollout remains open |
| 3. Checkpoint/evaluator contract | Contract rejection/loading tests and evaluator loaders passed | Continue compatibility checks as formats evolve |
| 4. Deployment export | Open; needed only for browser deployment | Export and run the selected GNN architecture in target runtime |
| 5. Quantitative probe validation | Open research work | Collect data and train probes with held-out splits |
