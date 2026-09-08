# Phase 13 — Multi-Seed Validation Protocol

## Procedure

For each of 3 seeds (42, 123, 999):
1. Train fresh model from scratch for 200k timesteps
2. Evaluate deterministically for 50 episodes
3. Record all metrics

## Reporting

For each metric, report:
- mean
- standard deviation
- minimum
- maximum
- coefficient of variation (std/mean)

## Stability Classification

| Classification | Criterion |
|----------------|-----------|
| Stable | CV < 0.3 for all primary metrics |
| Moderately variable | CV 0.3–0.6 for some metrics |
| Highly variable | CV > 0.6 for any primary metric |
| Failed | Does not meet success criteria for any seed |

## Primary Metrics

- goal_rate_pct
- pass_completion_rate_pct
- shots_per_ep
- mean_reward
- reward_std

## Example Output Format

```
Metric: goal_rate_pct
  Seed 42: 25.3%
  Seed 123: 22.1%
  Seed 999: 28.7%
  Mean: 25.4%
  Std: 3.3%
  Min: 22.1%
  Max: 28.7%
  CV: 0.13
  Classification: Stable
```
