# GMN-Football-3 — Training Directory

This directory contains the full training pipeline: environment wrappers, trainers,
evaluation harnesses, network definitions, debugging/trace utilities, and tests.

## Quick start

```bash
# Install Python dependencies (from repo root)
pip install -r training/requirements.txt

# Ensure the contract is in sync before any training run
npm run check:contracts

# Train MAPPO on the canonical 3v1 scenario (default)
python training/train_mappo.py

# Train PPO on the same scenario with parallel stepping (4 bridges)
python training/train_ppo.py --n-envs 4 --timesteps 200000

# Train PPO on a full 11v11 match (proof-of-pipeline baseline)
python training/train_ppo_full.py --timesteps 100000 --n-envs 4

# Evaluate a trained checkpoint
python training/eval_mappo.py --checkpoint training/models/mappo_..._best.pt
```

> **Note:** Python scripts must be run from the repository root so that `sys.path`
> insertion resolves the `training/` package correctly.

---

## Environment

The training pipeline does not ship a standalone Python game engine. Every
environment (`GMNFootballEnv` / `GMNMultiAgentEnv`) communicates with the
authoritative TypeScript `GameEngine` through a long-running Node bridge
(`training/bridge_server.ts`) over HTTP + WebSocket.

| Wrapper | Protocol | Module | Notes |
|---------|----------|--------|-------|
| Single-agent (Gym) | `gymnasium.Env` | `gmn_gym.py` | Used by `train_ppo.py`, `train_stage2_ppo.py`. |
| Multi-agent (PettingZoo) | `ParallelEnv` | `gmn_pettingzoo.py` | Used by `train_mappo.py`, `train_mappo_shaped.py`, `train_ppo_full.py`. |
| Bridge server | HTTP + WS | `bridge_server.ts` | Boot with `npx tsx training/bridge_server.ts`. |

### Parallel stepping (optional)


---

## Trainer scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `train_mappo.py` | Canonical MAPPO trainer (parameter-shared actor + centralized critic). Plain reward path. | **Stable** |
| `train_mappo_shaped.py` | MAPPO variant using shaped cooperative rewards (`CooperativeRewardShaper`). | **Stable** |
| `train_ppo.py` | Stable-Baseline3 PPO trainer wrapping `GMNFootballEnv`. Supports `--n-envs` for parallel stepping. | **Stable** |
| `train_ppo_full.py` | PPO on the full 11v11 match scenario. Re-exports `run_ppo_training` from `train_ppo.py`. | **Stable** (proof-of-pipeline) |
| `train_stage2_ppo.py` | Two-stage curriculum: pre-train on 3v1, then fine-tune on a harder scenario. | **Stable** |
| `train_ippo.py` | Independent PPO (one agent per player). Not the primary path — see `ippo_credit_assignment_report.md`. | **Reference** |

### Common trainer flags

| Flag | Default | Description |
|------|---------|-------------|
| `--timesteps` | scenario-dependent | Total environment steps to train. |
| `--n-envs` | `1` | Parallel bridge instances (`train_ppo.py` / `train_ppo_full.py`). |
| `--resume` | `None` | Path to a checkpoint to resume from. |
| `--checkpoint` | `None` | Output checkpoint filename. |
| `--lr` / `--lr-schedule` | `3e-4` / `linear` | Learning rate and schedule. |
| `--opponent-difficulty` | `medium` | Right-team difficulty for self-play pool. |
| `--eval-episodes` | `10` | Evaluation episodes at milestones. |

---

## Evaluation scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `eval_mappo.py` | Deterministic evaluation of a MAPPO checkpoint against the multi-agent env. | **Stable** |
| `eval_generalization.py` | Tests whether a policy generalizes to varied opponent formations / keeper arrangements. | **Stable** |
| `eval_mappo_comprehensive.py` | Comprehensive MAPPO evaluation (win rates, goals, trajectory metrics). | **Stable** |
| `eval_checkpoint.py` | Evaluates a single checkpoint file and prints metrics. | **Stable** |
| `eval_ippo_baseline.py` | Baseline evaluation against the IPPO reference policy. | **Stable** |
| `eval_progress.py` | Tracks training progress across checkpoints (generates trend CSVs). | **Stable** |
| `validate_learned_policy.py` | Validates a learned policy end-to-end (reward, episode length, goal rate). | **Stable** |
| `export_onnx.py` | Exports a MAPPO actor checkpoint to ONNX for browser inference. | **Stable** |

---


---

## Contract sync

Contract constants (observation size, action space, role vocabulary, etc.) are
defined canonically in `src/engine/Contract.ts` and replicated into Python by
`scripts/sync_contracts.ts`. The contract version is **3.1.0**.

| Command | Effect |
|---------|--------|
| `npm run check:contracts` | Verifies Python constants match `Contract.ts`; CI fails on drift. |
| `npm run sync:contracts` | Rewrites Python files from `Contract.ts`. |

`check:contracts` runs automatically in CI (`ci.yml`) and should be run before
any training session to catch drift early.

---

## Debug / trace / development scripts

These are stable development aids. They are not part of the automated pipeline.

| Script | Purpose |
|--------|---------|
| `debug_actions.py` | Inspects action-space coverage and action distributions. |
| `debug_physics.py` | Sanity-checks physics engine behavior (ball movement, collisions). |
| `debug_smoke.py` | Quick smoke test that the bridge responds to a few steps. |
| `trace_10steps.py`, `trace_short.py`, `trace_goal_brief.py` | Short trace runs for debugging. |
| `trace_goal_episodes.py`, `trace_goal_episodes_corrected.py` | Trace goal-scoring episodes. |
| `trace_goal_episodes_500k.py` | Long-running trace (500k steps). |
| `trace_to_frames.py` | Converts a trace into rendered frames. |
| `full_trace_episode.py` | Records one full episode trace. |
| `episode_recorder.py` | Reusable episode recorder used by eval scripts. |
| `binary_event_decoder.py` | Decodes binary match-event streams. |
| `checkpoint_contract.py` | Checkpoint integrity validation. |
| `onnx_proto_builder.py` | Builds ONNX protos for export testing. |
| `benchmark_bridge.py`, `benchmark_bridge_ws.py` | Bridge throughput benchmarks (HTTP vs WebSocket). |
| `analysis_200k_goal_episodes.py` | Post-hoc analysis of 200k-step goal-scoring episodes. |
| `reeval_goal_episodes.py` | Re-evaluates previously traced goal episodes. |
| `generate_comparison_table.py` | Generates comparison tables across checkpoints. |
| `rl_validation_suite.py` | RL-specific validation (advantages, returns, clipping). |
| `stage2_full_validation.py` | Full validation for stage-2 curriculum. |
| `verify_checkpoints.py` | Verifies checkpoint integrity across the model directory. |
| `verify_goal_trace.py` | Cross-checks goal traces against the authoritative engine. |
| `sync_contracts.py` | Python-side helper that calls the TS sync (legacy wrapper). |

---

## Test scripts

Run with `pytest training/` (or individually). These are **stable** and part of
the regression suite.

| Script | Purpose |
|--------|---------|
| `test_env.py` | Environment smoke tests (reset, step, spaces). |
| `test_pettingzoo_wrapper.py` | PettingZoo API conformance. |
| `test_mappo_pipeline.py` | End-to-end MAPPO training mini-pipeline. |

---

## Results & artifacts

| Path | Description |
|------|-------------|
| `training/models/` | Saved PyTorch checkpoints (`.pt`) and exported ONNX policies. |
| `training/logs/` | TensorBoard logs per run. |
| `training/results/` | CSV trend data and comparison tables. |
| `training/RESULTS_INDEX.md` | Evidence lineage: maps each reported result to a checkpoint SHA + `git describe`. |

---

## Dependency & environment notes

- Python dependencies: `training/requirements.txt` (the single canonical file).
- The root `requirements.txt` has been removed; always reference `training/requirements.txt`.
- New Python dependencies should be added to `training/requirements.txt` with a
  pinned version range and noted in the pull request description.
- The TypeScript bridge requires Node >= 20 and is started via `npx tsx training/bridge_server.ts`.

See the [root README](../README.md) for the architecture overview and the
[contributing guide](../CONTRIBUTING.md) for development conventions.
| `test_episode_recorder.py` | Episode recorder correctness. |
| `test_gae_bootstrap.py` | GAE / advantage computation. |
| `test_eval_cache_identity.py` | Evaluation cache identity. |
| `test_critic_scaling.py` | Centralized critic scaling. |
| `test_ippo_shared_reward.py` | IPPO shared-reward attribution. |
| `test_checkpoint_selection.py` | Checkpoint selection logic. |
| `test_e2e_determinism.py` | End-to-end determinism (TS engine ↔ Python). |
| `test_multiagent_determinism.py` | Multi-agent trajectory determinism. |
| `test_transport_parity.py` | HTTP vs WebSocket transport parity. |
| `test_browser_inference_parity.py` | Browser ONNX inference parity. |
| `test_gym_safety.py` | Gym API safety invariants. |
| `test_reward_shape_e2e.py` | End-to-end shaped-vs-unshaped reward wire test. |
| `smoke_test_1ep.py` | Single-episode smoke test. |
## Network & rollout modules

| Module | Purpose |
|--------|---------|
| `mappo_networks.py` | Shared actor + centralized critic network definitions. |
| `modular_networks.py` | Modular / ablatable network variants (experimental). |
| `mappo_rollout.py` | Rollout buffer, GAE computation, and mini-batch iterator for MAPPO. |
| `mappo_update.py` | PPO-clip update step (actor + critic loss). |
| `modular_encoder.ts` | TypeScript-side modular encoder (used by bridge + ONNX export). |
| `opponent_pool.py` | Opponent pool + self-play snapshotting (selection strategies: uniform / cyclic / elo). |
| `football_metrics.py` | Football-specific metrics (expected goals, possession, passing accuracy). |
Single-environment stepping blocks on a round-trip to the Node bridge. For
throughput, pass `--n-envs N` to spawn `N` parallel bridge instances on ports
`5050..5050+N-1`. Each sub-environment runs in its own `DummyVecEnv` worker so
SB3 sees one logical vector environment. `n_envs=1` keeps the legacy single-env
path unchanged.

```bash
python training/train_ppo.py --n-envs 4 --timesteps 200000
```

### Self-play / opponent pools (optional)

The right (non-learning) team is selected by an `OpponentPool`
(`training/opponent_pool.py`). Selection strategies: `uniform` (default),
`cyclic`, `elo`. Rule-based difficulties (`easy` / `medium` / `hard` / `master`)
are executable today via the bridge `/opponent` endpoint. Learned-policy
snapshots are periodically saved by the pool but require future bridge support
to be executed. Enable via `--opponent-difficulty` on the trainers.