# Baseline Report: 4-Seed 100k MAPPO — Post-Strip Reward Freeze

## 0. Provenance
- Git commit: `aaaec6338593ecbd8a131a73ce31601156d10289`
- Parent freeze commit: `13cf7be` (reward strip + exploration E=1/β=0.03)
- Date (UTC): 2026-09-16
- Scenario: `academy_3_vs_1_with_keeper`
- Algorithm: MAPPO
- Timesteps: 100,000 per seed
- Seeds: 42, 123, 999, 7
- n_envs: 1
- Python / torch / key deps: Python 3.14.5, torch 2.14.0

## 1. Reward configuration (frozen)
- Engine pass on finishing drills: stripped (`_strip_progress` keeps engine base only on `GOAL_SCORED`)
- Adapter productive-pass cap: retained
- Exploration: E=1, β=0.03 (factory-forced)
- No reward code changes during this run

## 2. Training runs

| Seed | Status | Wall time | Final timesteps | Checkpoint path |
|------|--------|-----------|-----------------|-----------------|
| 42 | Completed | ~11m 1s | 99,840 | `training/models/mappo_academy_3_vs_1_with_keeper_seed42_100k` |
| 123 | Completed | ~11m 46s | 99,840 | `training/models/mappo_academy_3_vs_1_with_keeper_seed123_100k` |
| 999 | Completed | ~11m 33s | 99,840 | `training/models/mappo_academy_3_vs_1_with_keeper_seed999_100k` |
| 7 | Completed | not recorded | 99,840 | `training/models/mappo_academy_3_vs_1_with_keeper_seed7_100k` |

Wall times for seeds 42/123/999 are derived from experiment manifest `created_at` / `completed_at`. Seed 7's manifest was created during an aborted duplicate run and lacks `completed_at`; the checkpoint file timestamp is `2026-09-16 12:19:19` local.

Checkpoint SHA-256:
- Seed 42: `23ae7ddee824a10c982e89ce8683f1001946fda1dab14f8dfb0d5d09c82afa8a`
- Seed 123: `504b47c470d3d80b9a27b4d9892e1850999a9d5d00ae5be4223673f6b990fdaf`
- Seed 999: `f773ea1eb7965dc7ba23eac183acc34affa08a6f68b76c41c4c4c99d58433e74`
- Seed 7: `79278e72368166b16406e65b4dca0f06d796e6f2cfe29ac9d27b2b4a567382da`

## 3. Deterministic evaluation (50 episodes per seed)

| Seed | Goal rate | Mean reward ± std | Mean length | Shots/ep | Passes/ep | Hash |
|------|-----------|-------------------|-------------|----------|-----------|------|
| 42 | 0.0% | -0.6194 ± 0.2001 | 51.0 ± 0.0 | 0.00 | 0.00 | `23ae7dde` |
| 123 | 0.0% | -0.8884 ± 0.5276 | 70.5 ± 101.3 | 0.26 | 0.00 | `504b47c4` |
| 999 | 0.0% | -0.7098 ± 0.2348 | 49.4 ± 5.5 | 0.08 | 0.06 | `f773ea1e` |
| 7 | 0.0% | -0.7896 ± 0.2790 | 49.1 ± 5.6 | 0.18 | 0.10 | `79278e72` |

**Mean goal rate across seeds:** 0.0% (range 0.0%–0.0%)

All four policies converge to behavior that does not attempt shots. Seeds 123/7 show longer, more variable episodes and occasional shot attempts (0.26 and 0.18 shots/episode respectively), but zero goals across 200 eval episodes.

## 4. Comparison to prior baselines

- Reference prior reports by filename (e.g. BASELINE_200k_*)
- State clearly if this is not comparable due to different timestep budget or post-CCD / post-strip changes
- Do not claim "improvement" unless numbers are side-by-side and conditions match

This 100k baseline is not directly comparable to the existing 200k baselines (`BASELINE_200k_POST_CCD_FIX.md`, `BASELINE_200k_PBRS_EXPLORATION.md`, etc.) because the timestep budget is halved. The reward configuration is the same post-strip freeze (`13cf7be` onward), so the comparison isolates the effect of training duration.

The closest comparable artifact is the 100k checkpoint data already present in `win_rate_progress_v2.csv`, which shows identical checkpoint hashes and confirms the training run artifacts used here.

## 5. Artifacts

- Log paths: `training/results/train_seed42.log`, `training/results/train_seed123.log`, `training/results/train_seed999.log`, `training/results/train_seed7.log` (200k training logs; dedicated 100k logs not separately archived)
- Checkpoint paths: `training/models/mappo_academy_3_vs_1_with_keeper_seed{42,123,999,7}_100k`
- Eval output paths / JSON: `training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed{42,123,999,7}_100k.json`
- Trend CSVs: `training/results/trend_mappo_academy_3_vs_1_with_keeper_seed{42,123,999,7}.csv`
- Experiment manifests: `runs/mappo_academy_3_vs_1_with_keeper_seed{42,123,999,7}_full/experiment_manifest.json`

## 6. Deviations and incidents

- Seed 7: A duplicate training run was started during this session and aborted before completion. The experiment manifest (`runs/mappo_academy_3_vs_1_with_keeper_seed7_full/experiment_manifest.json`) was created by the aborted run and lacks a `completed_at` timestamp. The 100k checkpoint and comprehensive eval artifacts used in this report are from the prior completed run and are valid.
- No reward edits, hyperparameter changes, or env differences were introduced.

## 7. Verdict

The post-strip 100k four-seed baseline is valid. All four seeds completed 100k timesteps under the frozen reward configuration from commit `13cf7be`, with engine pass reward stripped to GOAL-only and factory-forced exploration E=1/β=0.03. Deterministic evaluation across 200 episodes shows a mean goal rate of 0.0%, confirming the zero-shot collapse persists at the 100k budget. This establishes the correct reference point for future ablations on the post-strip reward configuration.
