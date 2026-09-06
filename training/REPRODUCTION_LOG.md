# GMN-Football-3 — Training Reproduction Log

## Host Environment

| Field | Value |
|---|---|
| OS | Windows (host: DESKTOP-JFT7577) |
| Python | 3.14.5 |
| PyTorch | 2.14.0+cpu |
| CUDA | Not available (CPU-only training) |
| Node.js | 22.23.2 |
| Date | 2026-09-05 to 2026-09-06 |

## Reproduced Runs

### MAPPO — `academy_3_vs_1_with_keeper`

| Checkpoint | Timesteps | Seed | Goal Rate | Mean Reward | Std Reward | Notes |
|---|---|---|---|---|---|---|
| `mappo_academy_3_vs_1_with_keeper_trained.pt` | 199,936 | N/A | 0.00% (eval) | 0.1859 | 0.1805 | Best evaluated MAPPO 3v1 checkpoint |
| `mappo_academy_3_vs_1_with_keeper_500k.pt` | 50,176 | N/A | 6.67% (eval) | 0.2937 | 0.3358 | Resumed run, highest observed goal rate |
| `mappo_academy_3_vs_1_with_keeper_100352.pt` | 100,352 | N/A | 0.00% (eval) | 0.2245 | 0.1713 | Mid-run milestone |
| `mappo_academy_3_vs_1_with_keeper_150528.pt` | 150,528 | N/A | 0.00% (eval) | 0.2155 | 0.1645 | Late-run milestone |

**Checkpoint SHA-256 (`mappo_academy_3_vs_1_with_keeper_trained.pt`):**
`03fdd778ffd01b659d246e0b36af18010972bc75cfccd41c75d9959d315e2edd`

**Exported ONNX:**
- `public/models/mappo_policy.onnx` (55487 bytes)
- ONNX sidecar SHA-256: `2be576e71b62a6258ef5bf1f343a57ac7c2dac199eed12d444f080c1f63f8233`
- Browser weights SHA-256: `e6ab531ff0fa...` (see `src/agents/mappo_weights.ts`)

### IPPO — `academy_3_vs_1_with_keeper`

| Checkpoint | Timesteps | Goal Rate | Mean Reward | Std Reward |
|---|---|---|---|---|
| `ippo_academy_3_vs_1_with_keeper_trained.zip` | 200,000 | 68.0% | 3.75 | 0.41 |

### PPO — `academy_empty_goal`

| Checkpoint | Timesteps | Goal Rate | Mean Reward | Std Reward |
|---|---|---|---|---|
| `ppo_academy_empty_goal_trained.zip` | 100,000 | 96.0% | 5.40 | 0.28 |

### MAPPO — `5_vs_5` (experimental)

| Checkpoint | Timesteps | Goal Rate | Mean Reward |
|---|---|---|---|
| `mappo_5_vs_5_20000.pt` | 19,968 | 2.00% | -2.4285 |

## Procedure Used

```powershell
# MAPPO 3v1 keeper
python training/train_mappo.py --timesteps 200000 --scenario academy_3_vs_1_with_keeper

# IPPO 3v1 keeper
python training/train_ippo.py 200000 --checkpoint ippo_academy_3_vs_1_with_keeper_trained.zip

# PPO empty goal
python training/train_ppo.py 100000 --scenario academy_empty_goal
```

## Evaluation

```powershell
python training/eval_progress.py
python training/eval_checkpoint.py
python training/test_reward_equivalence.ts
npm test
```

## Deviations from README

- Training ran on CPU only (no GPU available on this host).
- Python version is 3.14.5, not 3.10.11 as previously documented.
- MAPPO 3v1 training did not reach the task-brief target of `success_rate >= 40%` over 100 eval episodes; highest observed goal rate was 6.67% at 50k steps.
- 5v5 MAPPO training was run experimentally but has not converged.
