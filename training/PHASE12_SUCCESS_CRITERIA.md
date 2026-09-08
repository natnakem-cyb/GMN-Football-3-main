# Phase 12 — Success Criteria

## Quantitative Thresholds

Based on baseline measurements:

| Metric | Historical Baseline | Success Threshold |
|--------|-------------------|-------------------|
| pass_accuracy | 0.0% | > 5.0% |
| passing_frequency (passes/episode) | 0.00–0.44 | > 2.0 |
| shots_per_episode | 0.00 | > 0.5 |
| goal_rate | 13.2%–37.4% | > 20.0% (stable) |
| reward_variance | high | < 0.5 (std/mean ratio) |
| behavioral_diversity | low | entropy > 1.5 |
| dribbling_dominance | 100% progress | progress < 70% of total reward |

## Judgment Criteria

A model is successful if:
1. Pass accuracy > 5% (measured from engine ground truth)
2. Pass attempts per episode > 2.0
3. Shots per episode > 0.5
4. Goal rate is stable (std < 10% across seeds)
5. Reward variance decreases compared to baseline
6. Behavior differs from trivial dribbling policy (action entropy > 1.5)
7. Progress reward is < 70% of total reward (events contribute meaningfully)

## Threshold Derivation

- Pass accuracy > 5%: baseline was 0%, so any meaningful passing is improvement
- Pass attempts > 2.0/episode: baseline was 0.00–0.44, so 2.0 is ~5x improvement
- Shots > 0.5/episode: baseline was 0.00, so any shooting is improvement
- Goal rate > 20%: baseline varied 13–37%, so 20% is achievable and stable
- Reward variance < 0.5: high variance indicates unstable learning
- Action entropy > 1.5: indicates policy diversity, not collapse to single action
