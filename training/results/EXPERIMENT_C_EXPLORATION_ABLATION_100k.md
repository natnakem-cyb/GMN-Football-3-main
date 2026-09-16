# Experiment C: Exploration Ablation (E=0 vs E=1) at 100k

## 0. Provenance
- HEAD hash used for all runs: `046c378b9d3a45df849f4cad55a0c7eb251c498a`
- Verified freeze / strip commit: `13cf7be` (reward strip + exploration E=1/β=0.03 factory default)
- Baseline A report: `training/results/BASELINE_100k_POST_STRIP_4SEED.md`
- Diagnosis report: `training/results/DIAGNOSIS_100k_POST_STRIP_4SEED.md`
- Date (UTC): 2026-09-16
- Freeze verification: `_strip_progress` keeps engine base only on `GOAL_SCORED`; factory forces `enable_exploration_bonus=True, exploration_beta=0.03` for finishing scenarios; no reward edits during this experiment

## 1. Design
- Independent variable: `(E, β)` ∈ `{(0, 0), (1, 0.03)}`
- Fixed: `academy_3_vs_1_with_keeper`, MAPPO, 100,000 timesteps, seeds `{42, 123, 999, 7}`, post-strip reward implementation
- Injection: added `--no-exploration` / `--enable-exploration` and `--exploration-beta` CLI flags in `training/train_mappo.py`, forwarded through `run_mappo_training()` into `GMNMultiAgentEnv(enable_exploration_bonus=..., exploration_beta=...)`, then into `get_reward_adapter(scenario, enable_exploration_bonus=..., exploration_beta=...)`
- Confirmation: every training run printed `[RewardAdapter] active=AttackingDrillRewardAdapter scenario=academy_3_vs_1_with_keeper enable_exploration_bonus={True|False} exploration_beta={0.03|0.0}` at startup

## 2. Training inventory

| Arm | Seed | Status | Wall time | Final timesteps | Checkpoint path |
|-----|------|--------|-----------|-----------------|-----------------|
| OFF | 42 | Completed | ~12m | 99,840 | `training/models/mappo_ac3v1_seed42_100k_E0` |
| OFF | 123 | Completed | ~12m | 99,840 | `training/models/mappo_ac3v1_seed123_100k_E0` |
| OFF | 999 | Completed | ~12m | 99,840 | `training/models/mappo_ac3v1_seed999_100k_E0` |
| OFF | 7 | Completed | ~12m | 99,840 | `training/models/mappo_ac3v1_seed7_100k_E0` |
| ON | 42 | Completed | ~12m | 99,840 | `training/models/mappo_ac3v1_seed42_100k_E1` |
| ON | 123 | Completed | ~12m | 99,840 | `training/models/mappo_ac3v1_seed123_100k_E1` |
| ON | 999 | Completed | ~12m | 99,840 | `training/models/mappo_ac3v1_seed999_100k_E1` |
| ON | 7 | Completed | ~12m | 99,840 | `training/models/mappo_ac3v1_seed7_100k_E1` |

Checkpoint SHA-256:
- OFF 42: `0b9d885253dd1435f4fa5a2a5f22de9979fe7755bc87b242d69fdbe28bff1696`
- OFF 123: `53c1e27176636bbe93e4832be919b031fc364c77c34b0b312c3316608692146a`
- OFF 999: `9e48e56f0b18a3a0783aa4e104c2bd07e94351de8f53464b6aab94d209990878`
- OFF 7: `46f9765d2f9583043f7ec45730b1ec73acaa6eaa81e1d2ab05ab80aafecc45a5`
- ON 42: `9b71c949c4a67c3eba1f809ae331e2a23a99ab9b27519296f0d4ec194ce8c406`
- ON 123: `e0caa7f3d6099d9c1fe8ef7e57aa728b8700a8feb727934359a32ed6afe4b2a6`
- ON 999: `dd302da9db580ff3a8483c75d05bf3352c60b8516ec1e542799cec150bc9f503`
- ON 7: `66dbfded8ab1583db2413b0f466de58183880c6379129818292731629507a044`

## 3. Evaluation protocol
- 50 deterministic episodes per checkpoint
- Eval seed = training seed
- Goals counted from JSON episode flags, not console summary percentages
- Eval JSON paths:
  - `training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed42_100k_E0.json`
  - `training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed123_100k_E0.json`
  - `training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed999_100k_E0.json`
  - `training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed7_100k_E0.json`
  - `training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed42_100k_E1.json`
  - `training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed123_100k_E1.json`
  - `training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed999_100k_E1.json`
  - `training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed7_100k_E1.json`

## 4. Results — per seed × arm

| Seed | Arm | Goal rate | Mean reward ± std | Mean length | Shots/ep | Passes/ep | Possession % | Turnovers/ep | Forward progress/ep | Min dist to goal | Tackles/ep |
|------|-----|-----------|-------------------|-------------|----------|-----------|--------------|--------------|---------------------|------------------|------------|
| 42 | E0 | 0.0% | -0.6096 ± 0.3198 | 50.52 ± 2.385 | 0.08 | 0.00 | 0.0 | 1.16 | -0.2938 | 0.5959 | 0.00 |
| 42 | E1 | 0.0% | -0.5471 ± 0.2766 | 50.56 ± 3.080 | 0.04 | 0.02 | 0.0 | 1.26 | -0.2235 | 0.5465 | 0.04 |
| 123 | E0 | 0.0% | -0.6359 ± 0.2958 | 49.48 ± 5.551 | 0.12 | 0.18 | 7.5 | 1.28 | -0.4324 | 0.5631 | 1.10 |
| 123 | E1 | 0.0% | -0.5670 ± 0.2882 | 47.62 ± 9.503 | 0.16 | 0.08 | 0.0 | 1.10 | -0.3623 | 0.5527 | 5.14 |
| 999 | E0 | 0.0% | -0.5607 ± 0.2487 | 49.54 ± 6.194 | 0.06 | 0.34 | 0.0 | 1.28 | -0.2378 | 0.5547 | 0.30 |
| 999 | E1 | 0.0% | -0.6659 ± 0.3312 | 49.94 ± 4.296 | 0.16 | 0.34 | 0.0 | 1.32 | -0.5362 | 0.5717 | 0.04 |
| 7 | E0 | 0.0% | -0.6020 ± 0.3335 | 48.94 ± 6.987 | 0.12 | 0.20 | 0.0 | 1.16 | -0.4252 | 0.5538 | 0.20 |
| 7 | E1 | 0.0% | -0.6120 ± 0.2572 | 48.46 ± 9.117 | 0.10 | 0.92 | 0.0 | 1.30 | -0.4819 | 0.5692 | 0.00 |

## 5. Results — aggregate OFF vs ON

| Metric | OFF (E=0) mean ± range | ON (E=1) mean ± range | Delta (ON − OFF) |
|--------|------------------------|-----------------------|------------------|
| Goal rate | 0.0% (0.0–0.0%) | 0.0% (0.0–0.0%) | 0.0% |
| Mean reward | -0.6271 ± 0.0521 | -0.5980 ± 0.0503 | +0.0291 |
| Mean length | 49.62 ± 1.46 | 49.15 ± 1.31 | -0.47 |
| Shots/ep | 0.095 ± 0.027 | 0.115 ± 0.050 | +0.020 |
| Passes/ep | 0.180 ± 0.146 | 0.340 ± 0.380 | +0.160 |
| Possession % | 1.88% ± 3.38% | 0.0% ± 0.0% | -1.88% |
| Turnovers/ep | 1.22 ± 0.06 | 1.25 ± 0.10 | +0.03 |
| Forward progress/ep | -0.3473 ± 0.0909 | -0.4010 ± 0.1278 | -0.0537 |
| Min dist to goal | 0.5669 ± 0.0195 | 0.5600 ± 0.0126 | -0.0069 |
| Tackles/ep | 0.40 ± 0.48 | 1.30 ± 2.25 | +0.90 |

## 6. Pathology comparison

| Axis | OFF (E=0) | ON (E=1) | Assessment |
|------|-----------|----------|------------|
| Shot volume | 0.095/ep | 0.115/ep | Slight increase, still negligible |
| Pass volume | 0.180/ep | 0.340/ep | Increase driven mainly by seed 7 (0.92/ep); others flat |
| Possession | 1.88% aggregate | 0.0% aggregate | OFF actually better; ON loses possession |
| Forward progress | -0.347 | -0.401 | ON slightly worse; ball regresses more |
| Tackle spam | 0.40/ep | 1.30/ep | **ON worse**, driven by seed 123 pathological 5.14/ep |
| Episode length | 49.62 | 49.15 | Similar; no truncation effect |
| Reward | -0.627 | -0.598 | ON slightly better, but not productive |

**Dominant pathology:** Mode 1 (no exploration of finishing actions) + Mode 3 (possession/turnover collapse). Exploration does not fix the underlying issue; in seed 123 it introduces pathological tackle spam.

## 7. Answer to the scientific question

Exploration at `(E=1, β=0.03)` does **not** improve productive football behavior relative to `(E=0, β=0)`. The ON arm shows slightly higher mean reward only because of seed-to-seed variance, not because of meaningful progress toward goals. Shot volume increases marginally, pass volume increases only in seed 7, possession drops to 0%, forward progress worsens slightly, and seed 123 develops pathological tackle spam (5.14 tackles/episode vs 1.10 in OFF). The policies remain in early-training dynamics regardless of exploration setting.

## 8. Decision

**Primary next step: D — targeted investigation**

Rationale:
- Exploration is not the main lever at 100k: ON vs OFF differences are small, inconsistent across seeds, and accompanied by new pathology in seed 123
- All arms still show 0% goal rate, confirming the issue is structural, not just insufficient horizon
- The variance in results across seeds suggests bugs or masking issues rather than a clean exploration effect
- Before any further training, investigate:
  1. Action masking / discrete action space correctness
  2. Reward attribution and whether step cost dominates
  3. Possession collapse mechanics
  4. Seed 123 tackle spam root cause

If B is eventually chosen, it should be after D identifies and fixes the underlying structural issue, not as a blind horizon extension.

## 9. Deviations and incidents

- OFF seed 123: `torch.save` I/O error at step 50,176 during first attempt. Process exited with code 1. Retried identically after verifying save path/permissions; second attempt completed successfully at step 99,840. No reward, exploration, or hyperparameter changes were made.
- No other crashes, retries, or flag mismatches.

## 10. Artifacts

- Training logs: `training/results/logs/expC_E{0,1}_seed{42,123,999,7}_100k_*.log`
- Checkpoints: `training/models/mappo_ac3v1_seed{42,123,999,7}_100k_E{0,1}`
- Eval JSONs: `training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed{42,123,999,7}_100k_E{0,1}.json`
- Progress CSV: `training/results/win_rate_progress_v2.csv`
- Experiment manifests: `runs/mappo_academy_3_vs_1_with_keeper_seed{42,123,999,7}_full/experiment_manifest.json`
