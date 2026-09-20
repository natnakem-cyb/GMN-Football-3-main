# GMN-Football-3 — Training Directory

This directory contains the full training pipeline: environment wrappers, trainers,
evaluation harnesses, network definitions, debugging/trace utilities, and tests.

## Curriculum vs scenario registry (audit note)

The MAPPO curriculum ladder (`CURRICULUM_STAGES` in `training/curriculum_scheduler.py`)
is a **strict subset** of the 12 scenarios registered in
`src/scenarios/ScenarioRegistry.ts`:

- 8 stages are on the ladder (promotion/demotion driven by
  `is_scenario_success` win-rate thresholds).
- `academy_rondo_4v1` is a **parallel track** — trained directly with
  `python training/train_mappo.py --scenario academy_rondo_4v1`, never promoted
  through the ladder (its success rule is rondo-specific: no goals conceded +
  `scenario_complete`).
- `academy_3_vs_1_keeper_aggressive`, `academy_3_vs_1_shifted`, and
  `academy_3_vs_1_randomized` are **held-out generalization variants**. They are
  intentionally not part of the ladder; adding them is an explicit design
  decision, not an automatic extension.

Topology naming: `academy_3_vs_1_with_keeper` means **3 attackers vs 1 defender
plus a goalkeeper** (teamLeftPlayers: 3, teamRightPlayers: 2 with a GK role,
`ScenarioRegistry.ts:110-126`). The id counts outfield opponents only. Scenario
ids are wire keys used by checkpoints and the bridge and must not be renamed.

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
| `eval_post_reweight_logits.py` | Actor logit/π snapshot on on-ball frames (historical; its retention used the POST-step owner while reporting pre-step state — corrected by the pre-step collector below). | **Superseded (measurement)** |
| `eval_post_reweight_logits_prestep.py` | Corrected PRE-STEP `obs[95]` on-ball measurement: retention, mask, logits, π and deterministic action all come from one pre-step state; verifies checkpoint SHA-256 against the inventory. | **Measurement-only** |
| `verify_prestep_measurement.py` | Independent consistency checker for the pre-step measurement artifacts (re-derives every aggregate from raw frame detail). | **Measurement-only** |

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
| `test_binary_frame_parser.py` | Byte-level regression tests for standard and rondo binary frame parsing. |
| `test_prestep_onball_temporal_alignment.py` | Synthetic gate for the pre-step `obs[95]` on-ball retention rule (Cases A–D, mask/obs/logit alignment, π-vs-frequency separation). |
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
---

## Training documentation & results registry

This table registers every `.md` file under `training/` for issue tracking and research reference.

| # | File | Title | Short description | Date | Time | Issues |
|---|------|-------|-------------------|------|------|--------|
| 1 | `training\REPRODUCTION_LOG.md` | GMN-Football-3 — Training Reproduction Log | \| Field \| Value \| | 2026-09-05 | — | — |
| 2 | `training\INVESTIGATION_REPORT.md` | GMN-Football-3 — Post-Rondo-Fix RL Maturity Investigation Report | **Date:** 2026-09-08 | 2026-09-08 | — | bug, fix |
| 3 | `training\GNN_PHASE0_ARCHITECTURE_AUDIT.md` | GNN_PHASE0_ARCHITECTURE_AUDIT | **Date:** 2026-09-11 | 2026-09-11 | — | — |
| 4 | `training\GNN_PHASE1_GRAPH_SCHEMA.md` | GNN_PHASE1_GRAPH_SCHEMA | **Date:** 2026-09-11 | 2026-09-11 | — | fix |
| 5 | `training\GNN_PHASE2_FORMATION_MODEL.md` | GNN_PHASE2_FORMATION_MODEL | **Date:** 2026-09-12 | 2026-09-12 | — | — |
| 6 | `training\GNN_PHASE3_SCENARIO_MODEL.md` | GNN_PHASE3_SCENARIO_MODEL | **Date:** 2026-09-12 | 2026-09-12 | — | fix |
| 7 | `training\RESULTS_INDEX.md` | GMN-Football-3 — Canonical Results & Evidence Index | Single source of truth mapping each *reported* result to its checkpoint, evaluator, | 2026-09-12 | — | failure, fix, issue |
| 8 | `training\GNN_PHASE4_GRAPH_DIAGNOSTICS.md` | GNN_PHASE4_GRAPH_DIAGNOSTICS | **Date:** 2026-09-13 | 2026-09-13 | — | fix |
| 9 | `training\results\BASELINE_100k_POST_STRIP_4SEED.md` | Baseline Report: 4-Seed 100k MAPPO — Post-Strip Reward Freeze | - Git commit: `aaaec6338593ecbd8a131a73ce31601156d10289` | 2026-09-16 | — | fix, incident |
| 10 | `training\results\EXPERIMENT_C_EXPLORATION_ABLATION_100k.md` | Experiment C: Exploration Ablation (E=0 vs E=1) at 100k | - HEAD hash used for all runs: `046c378b9d3a45df849f4cad55a0c7eb251c498a` | 2026-09-16 | — | bug, fix, issue |
| 11 | `training\results\EXPERIMENT_D_PASS_PATH_FORENSICS.md` | Experiment D: Pass-Path Forensics | - HEAD: `2230fdc5b578cd135eddb89a9a77e16f2c278479` | 2026-09-16 | — | bug, failure, fix, issue |
| 12 | `training\results\EXPERIMENT_D_QUALIFICATION.md` | Experiment D: Scenario Qualification (Canonical) | - HEAD hash: `2230fdc5b578cd135eddb89a9a77e16f2c278479` | 2026-09-16 | — | crash, failure, fix |
| 13 | `training\results\REWARD_WHOLE_PIPELINE_STRESS_TEST.md` | GMN-Football-3 Whole-Pipeline Reward Stress Test Report | **Report Date:** 2026-09-16 | 2026-09-16 | — | failure, fix |
| 14 | `training\results\D_EXPERIMENT_TACKLE_FORENSICS.md` | Experiment D — Tackle Exploitation Forensic Investigation | **Date:** 2026-09-17 | 2026-09-17 | — | failure, fix |
| 15 | `training\results\EXPERIMENT_B_HORIZON_200k.md` | Experiment B Horizon 200k — Best-vs-Final Gap Investigation | **Date:** 2026-09-17 | 2026-09-17 | — | bug, failure, fix |
| 16 | `training\results\EXPERIMENT_B_PROVENANCE.md` | Experiment B Provenance | **Date:** 2026-09-17 | 2026-09-17 | — | bug, crash, error, fix |
| 17 | `training\results\EXPERIMENT_D_TACKLE_SPAM_FORENSICS.md` | Experiment D — Tackle Spam Forensic Investigation | **Date:** 2026-09-17 | 2026-09-17 | — | bug, failure, fix |
| 18 | `training\results\EXPERIMENT_F_ACT.md` | F_act: Forced One-Tick PASS/SHOT | **Date:** 2026-09-17 | 2026-09-17 | — | failure |
| 19 | `training\results\EXPERIMENT_OCCUPANCY_F.md` | Experiment F — Occupancy Intervention Gate | **Date:** 2026-09-17 | 2026-09-17 | — | fix, issue |
| 20 | `training\results\EXPERIMENT_OCCUPANCY_F_PROVENANCE.md` | Experiment F Provenance | **Date:** 2026-09-17 | 2026-09-17 | — | fix, issue |
| 21 | `training\results\EXPERIMENT_PARALYSIS_FORENSICS.md` | Current-policy paralysis forensics | **Date:** 2026-09-17 | 2026-09-17 | — | fix |
| 22 | `training\results\EXPLORATION_ABLATION_FINDINGS.md` | Exploration Ablation Findings — Form A: Targeted On-Ball Entropy Bonus | HEAD: dc2718d7fbf50e4a17effd9553023b0bdad4d812 | 2026-09-18 | — | — |
| 23 | `training\results\EXPLORATION_ABLATION_PROTOCOL.md` | Exploration Ablation Protocol — Form A: Targeted On-Ball Entropy Bonus | Status: Pre-registered (locked before training begins) | 2026-09-18 | — | fix |
| 24 | `training\results\FRESH_TRAINING_FINDINGS.md` | Fresh Training Findings — µ-onball, Repaired PASS Path | **Date:** 2026-09-18 | 2026-09-18 | — | — |
| 25 | `training\results\FRESH_TRAINING_PROTOCOL.md` | Fresh Training Protocol — µ-onball, Repaired PASS Path | **Pre-registered:** 2026-09-18 | 2026-09-18 | — | failure |
| 26 | `training\results\FROZEN_PI_POST_PASS_FIX.md` | Frozen-π Re-evaluation Post PASS Fix | - **HEAD:** `5a9dfdda449af14f2bed8b5c94161bf5dfe48acf` | 2026-09-18 | — | defect, fix, problem |
| 27 | `training\results\MIXSCRIPT_FINDINGS.md` | Mix-Script Findings — µ-onball, Bounded Scripted-Assistance Intervention | **Date:** 2026-09-18 | 2026-09-18 | — | — |
| 28 | `training\results\MIXSCRIPT_PROTOCOL.md` | Mix-Script Protocol (Option B) — µ-onball, Repaired PASS Path | **Date:** 2026-09-18 | 2026-09-18 | — | failure, issue, problem |
| 29 | `training\results\PASS_DIAGNOSTIC_FINDINGS.md` | PASS Diagnostic Findings  F_act Extension | **Date:** 2026-09-18 07:25 UTC | 2026-09-18 | — | error, fix |
| 30 | `training\results\REWARD_ADVANTAGE_AUDIT.md` | Reward / Advantage Audit — Post-Mix-Script Diagnosis | Audit script: `training/eval_reward_advantage_audit.py` | 2026-09-18 | — | issue |
| 31 | `training\BUGFIX_CORRECTION.md` | Bug Fix Correction: step_batch() Terminal Reward Drop | The commit message claimed: | — | — | bug, fix |
| 32 | `training\PHASE10_RETRAIN.md` | Phase 10 — Retrain Configuration | All pipeline verification must pass before retraining: | — | — | bug, failure, fix |
| 33 | `training\PHASE11_BEHAVIORAL_EVAL.md` | Phase 11 — Behavioral Evaluation | - pass_attempts: number of pass actions taken | — | — | — |
| 34 | `training\PHASE12_SUCCESS_CRITERIA.md` | Phase 12 — Success Criteria | Based on baseline measurements: | — | — | — |
| 35 | `training\PHASE13_MULTI_SEED.md` | Phase 13 — Multi-Seed Validation Protocol | For each of 3 seeds (42, 123, 999): | — | — | — |
| 36 | `training\PHASE14_GENERALIZATION.md` | Phase 14 — Generalization Protocol | Vary: | — | — | — |
| 37 | `training\PHASE15_RL_MATURITY.md` | Phase 15 — RL Maturity Update | - [x] Infrastructure works (bridge, environment, training loop) | — | — | bug, fix |
| 38 | `training\PHASE16_RONDO_PRESERVATION.md` | Phase 16 — Rondo Architecture Preservation | - File: `src/engine/scenarios/RondoScenarioHandler.ts` | — | — | bug, fix |
| 39 | `training\PHASE17_FINAL_REPORT.md` | Phase 17 — Final Report | \| Item \| Value \| | — | — | bug, fix, issue |
| 40 | `training\PHASE2_REWARD_AUDIT.md` | Phase 2 — Reward Function Audit | \| Component \| Value \| Frequency \| Recipient \| Intended Behavior \| Exploit Risk \| | — | — | bug, error, fix |
| 41 | `training\PHASE5_REWARD_REDESIGN.md` | Phase 5 — Reward Redesign | The original reward function in `src/engine/ObservationEncoder.ts` produced a scalar where: | — | — | — |
| 42 | `training\PHASE8_CHECKPOINT_BUG.md` | Phase 8 — Checkpoint Promotion Bug Investigation | Seeds 43, 44, and 137 produced byte-identical terminal 500k checkpoints: | — | — | bug, fix |
| 43 | `training\README.md` | GMN-Football-3 — Training Directory | This directory contains the full training pipeline: environment wrappers, trainers, | — | — | bug |
| 44 | `training\REWARD_AUDIT.md` | Phase 2: Reward Audit for academy_3_vs_1_with_keeper | \| Component \| Value \| Frequency \| Recipient \| Conditions \| Exploit Risk \| | — | — | — |
| 45 | `training\TELEMETRY_PROTOCOL.md` | GMN-Football-3 — Canonical Telemetry Protocol (Stabilization Release) | Single source of truth for ALL telemetry messages exchanged between the bridge | — | — | error, fix |
| 46 | `training\experimental_hierarchy.md` | Experimental Hierarchy | \| Label \| Name \| Freeze / settings \| Scope \| Status \| | — | — | failure, fix, issue |
| 47 | `training\ippo_credit_assignment_report.md` | IPPO vs. MAPPO: Credit-Assignment Architecture Comparison | This report compares Independent PPO (IPPO) and Multi-Agent PPO with a Centralized Critic (MAPPO) in cooperative multi-a | — | — | error |
| 48 | `training\results\ACTOR_ACTION_FORENSICS.md` | — | ACTOR ACTION FORENSICS REPORT | — | — | defect |
| 49 | `training\results\BASELINE.md` | GMN-Football-3 — FROZEN PRE-RETRAINING BASELINE (PRE-STABILIZATION → STABILIZATION BOUNDARY) | **Status:** environment stabilization complete. NO new training has been run after | — | — | fix |
| 50 | `training\results\BASELINE_200k_M1b_report.md` | Baseline Report: 3-Seed 200k Retrain Post-M1b | **Commit:** `8b788cd` | — | — | crash, failure, fix |
| 51 | `training\results\BASELINE_200k_MASKING_FIXED.md` | Baseline Report: 4-Seed 200k Masking-Fix Retrain (stale checkpoints) | **Note:** This document evaluates pre-fix checkpoints from commit `e2509b0` using the new mask-aware eval scripts. For t | — | — | failure, fix |
| 52 | `training\results\BASELINE_200k_MASKING_FIXED_RETRAIN.md` | Baseline Report: 4-Seed 200k Masking-Fix Retrain | **Supersedes:** `training/results/BASELINE_200k_MASKING_FIXED.md` — that report reflects a stale re-eval of pre-fix chec | — | — | fix |
| 53 | `training\results\BASELINE_200k_PBRS_EXPLORATION.md` | §1 — Training Run Provenance | ﻿Baseline Report: 4-Seed 200k PBRS + Exploration-Bonus Retrain | — | — | error, failure, fix |
| 54 | `training\results\BASELINE_200k_POST_CCD_FIX.md` | §1 — Training Run Provenance | ﻿# Baseline Report: 4-Seed 200k Post-CCD-Fix Retrain | — | — | bug, error, failure, fix |
| 55 | `training\results\BASELINE_200k_VERIFIED.md` | Baseline Report: 3-Seed 200k Shaperfix Retrain — INDEPENDENTLY VERIFIED | > **Note (post-CCD-fix):** This baseline was trained and evaluated **before** the ball-tunneling CCD fix (`abd8be2`/`d5c | — | — | error, fix, problem |
| 56 | `training\results\DIAGNOSIS_100k_POST_STRIP_4SEED.md` | Diagnosis Note: 4-Seed 100k MAPPO Post-Strip Baseline | - Baseline: `training/results/BASELINE_100k_POST_STRIP_4SEED.md` | — | — | bug, failure, fix, issue |
| 57 | `training\results\D_OBSERVATION_SUFFICIENCY_AUDIT.md` | D-Obs: Observation Sufficiency Audit | **Primary scenario:** `academy_3_vs_1_with_keeper` | — | — | — |
| 58 | `training\results\EXPERIMENT_CRITIC_GAE_FORENSICS.md` | Critic / GAE Forensics Report | **HEAD:** `c2aabf55d2a47c5a91fc530e1608fcee0d29bec2` | — | — | bug, failure, fix |
| 59 | `training\results\EXPERIMENT_D_SCENARIO_QUALIFICATION.md` | Experiment D: Scenario Qualification | > **Deprecated:** This report has been superseded by the canonical document: | — | — | — |
| 60 | `training\results\EXPERIMENT_D_WHOLE_REGISTRY_SCENARIO_QUALIFICATION.md` | Experiment D: Whole-Registry Scenario Qualification | > **Deprecated:** This report has been superseded by the canonical document: | — | — | — |
| 61 | `training\results\PASS_FIX_VERIFICATION.md` | PASS Fix Verification — Nearest-Teammate Aiming (Option A) | \| Metric \| Value \| | — | — | bug, error, failure, fix |
| 62 | `training\results\README.md` | GMN-Football-3 Training Results & Benchmarks | This directory contains persistent evaluation metrics, training trend traces, and algorithm comparison tables for reinfo | — | — | fix |
| 63 | `training\results\comparison_table.md` | GMN-Football-3 — Reinforcement Learning Benchmark Comparison Table | > **Methodology**: Win-Rate is defined strictly as Goal Conversion Rate (% of evaluation episodes where left team scores | — | — | fix |
| 64 | `training\results\d_obs_pipeline_trace.md` | D-Obs Stage 1: Pipeline Trace | ``` | — | — | — |
| 65 | `training\validation_report.md` | GMN-Football-3 Validation Report | \| Checkpoint \| SHA-256 (full) \| Timesteps \| Notes \| | — | — | bug |
| 66 | `training\results\ONBALL_OCCUPANCY_PRESTEP_CHECK.md` | ONBALL_OCCUPANCY_PRESTEP_CHECK — Corrected Pre-Step On-Ball Measurement (50k actor-reweight checkpoints) | Measurement-only correction of the pre-step `obs[95]` on-ball gate; occupancy, legality, π and deterministic PASS+SHOT frequency with explicit `n`. | 2026-09-20 | — | defect, fix, measurement |

**Total:** 66 `.md` files registered.
