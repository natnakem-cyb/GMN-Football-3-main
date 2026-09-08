# Phase 10 — Retrain Configuration

## Prerequisites

All pipeline verification must pass before retraining:
- [x] TypeScript compilation: PASS
- [x] Python tests: 20 passed (1 pre-existing failure in test_onnx_opponent.py)
- [x] Per-agent reward assignment bug: FIXED
- [x] Reward redesign: IMPLEMENTED
- [x] Regression tests: 9 passed
- [x] MAPPO buffer shapes: VERIFIED

## Training Configuration

### Canonical Plain Trainer (`train_mappo.py`)

```bash
python training/train_mappo.py \
  --scenario academy_3_vs_1_with_keeper \
  --seed 42 \
  --timesteps 200000 \
  --checkpoint mappo_academy_3_vs_1_with_keeper_seed42_retrain.pt
```

### Canonical Shaped Trainer (`train_mappo_shaped.py`)

```bash
python training/train_mappo_shaped.py \
  --scenario academy_3_vs_1_with_keeper \
  --seed 42 \
  --total-steps 500000 \
  --enable-reward-shaping
```

### Hyperparameters (both trainers)

| Parameter | Value |
|-----------|-------|
| Timesteps | 200,000 (plain), 500,000 (shaped) |
| Rollout length (n_steps) | 256 |
| Mini-batch size | 256 |
| PPO Epochs | 4 |
| Learning rate | 3e-4 (Adam, cosine anneal to 3e-5) |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.15 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0.01 -> 0.005 (linear schedule) |
| Max grad norm | 0.5 |
| Architecture | SharedActor (Mlp 64x64) + CentralizedCritic (Deep Sets) |

## Seeds

Train 3 independent seeds:
- Seed 42 (default)
- Seed 123
- Seed 999

Each seed produces:
- `mappo_{scenario}_seed{seed}_retrain.pt` — terminal checkpoint
- `mappo_{scenario}_seed{seed}_retrain_best.pt` — best deterministic checkpoint
- `mappo_{scenario}_seed{seed}_retrain_rolling_best.pt` — best rolling checkpoint

## Metrics to Record

For each seed, record:
- seed
- steps
- episode reward (mean, std)
- pass accuracy
- pass attempts
- successful passes
- turnovers
- shots
- goals
- possession %
- episode length
- policy entropy
- KL divergence

## Commands

```bash
# Seed 42
python training/train_mappo.py --seed 42 --timesteps 200000 --checkpoint mappo_academy_3_vs_1_with_keeper_seed42_retrain.pt

# Seed 123
python training/train_mappo.py --seed 123 --timesteps 200000 --checkpoint mappo_academy_3_vs_1_with_keeper_seed123_retrain.pt

# Seed 999
python training/train_mappo.py --seed 999 --timesteps 200000 --checkpoint mappo_academy_3_vs_1_with_keeper_seed999_retrain.pt
```
