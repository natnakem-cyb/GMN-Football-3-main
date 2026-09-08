# Phase 8 — Checkpoint Promotion Bug Investigation

## Observation

Seeds 43, 44, and 137 produced byte-identical terminal 500k checkpoints:

```
SHA-256: 939e6ceefb95c880d2021067bee0103cf4a770258b1ecf01ea892a756d0afe50
```

Affected files:
- `mappo_academy_3_vs_1_with_keeper_best.pt`
- `mappo_academy_3_vs_1_with_keeper_seed43.pt`
- `mappo_academy_3_vs_1_with_keeper_seed44.pt`
- `mappo_academy_3_vs_1_with_keeper_seed137.pt`

## Root Cause

The checkpoint promotion logic in `train_mappo.py` and `train_mappo_shaped.py` overwrites the **terminal checkpoint** (`{scenario}_seed{seed}.pt`) with the best deterministic checkpoint at the end of training:

```python
# train_mappo.py lines 458-469
if best_deterministic_checkpoint_path and os.path.exists(best_deterministic_checkpoint_path):
    import shutil
    shutil.copy2(best_deterministic_checkpoint_path, checkpoint_path)
```

The `checkpoint_path` is the terminal checkpoint (e.g., `mappo_academy_3_vs_1_with_keeper_seed43.pt`). When training completes, the best checkpoint found during training is copied over the terminal checkpoint.

If multiple seeds share the same `best_deterministic_checkpoint_path` (e.g., because they were trained in the same process or shared a common best), they will all end up with identical terminal checkpoints.

Additionally, the `_best.pt` naming is shared across seeds:
- `mappo_academy_3_vs_1_with_keeper_best.pt` is the global best (unseeded)
- `mappo_academy_3_vs_1_with_keeper_seed43_best.pt` is seed43-specific best

The bug occurs because `mappo_academy_3_vs_1_with_keeper_best.pt` and `mappo_academy_3_vs_1_with_keeper_seed43.pt` ended up with the same SHA-256, indicating that the global best was copied over the seed-specific terminal checkpoint.

## Evidence

| File | SHA-256 |
|------|---------|
| `mappo_academy_3_vs_1_with_keeper_best.pt` | `939e6ceefb95c880d2021067bee0103cf4a770258b1ecf01ea892a756d0afe50` |
| `mappo_academy_3_vs_1_with_keeper_seed43.pt` | `939e6ceefb95c880d2021067bee0103cf4a770258b1ecf01ea892a756d0afe50` |
| `mappo_academy_3_vs_1_with_keeper_seed44.pt` | `939e6ceefb95c880d2021067bee0103cf4a770258b1ecf01ea892a756d0afe50` |
| `mappo_academy_3_vs_1_with_keeper_seed137.pt` | `939e6ceefb95c880d2021067bee0103cf4a770258b1ecf01ea892a756d0afe50` |

The per-seed intermediate checkpoints (e.g., `seed42_50176.pt`, `seed43_50176.pt`) have **unique** SHA-256s, confirming that training did run independently. The duplication is caused by the final `shutil.copy2` step.

## Impact

- **Historical checkpoints are corrupted**: The terminal checkpoints for seeds 43, 44, and 137 no longer represent the actual end-of-training state for those seeds.
- **Evaluation artifacts may be misleading**: Any evaluation that used these terminal checkpoints evaluated the best checkpoint, not the seed-specific terminal checkpoint.
- **Multi-seed comparison is invalid**: Cannot compare seed 43 vs seed 42 terminal checkpoints because seed 43's terminal checkpoint was overwritten.

## Fix

The fix should ensure that:
1. Terminal checkpoints (`{scenario}_seed{seed}.pt`) are **never overwritten** by promotion logic
2. Best checkpoints are stored in separate files (`{scenario}_seed{seed}_best.pt`)
3. The end-of-run preservation copies the best checkpoint to a **durable artifact** path, not the terminal path

Current code already uses `_best.pt` for the best checkpoint, but the final `shutil.copy2` overwrites the terminal checkpoint with the best. This is the source of the bug.

**Recommended fix**: Remove or guard the final `shutil.copy2` so it never overwrites a seed-specific terminal checkpoint. The best checkpoint should be preserved as `_best.pt` only.

## Regression Test

A test should verify that:
1. After training, the terminal checkpoint SHA-256 is unique per seed
2. The terminal checkpoint does not equal the best checkpoint unless training never improved
