# Phase 11 — Behavioral Evaluation

## Metrics

### Attack
- pass_attempts: number of pass actions taken
- successful_passes: passes completed (from engine stats)
- pass_accuracy: successful_passes / pass_attempts
- turnovers: possessions lost to right team
- possession_pct: left team possession percentage
- progress: average ball X advancement per step
- shots: number of shot actions
- goals: number of goals scored

### Defense
- pressure: average defender-to-ball distance
- interceptions: number of interceptions
- turnovers_forced: possessions won from left team
- possession_disruption: tackles + interceptions
- distance_to_ball: mean defender distance

### Policy Quality
- action_diversity: entropy of action distribution
- repetitive_action_frequency: % of steps using same action
- behavioral_collapse: std of action distribution
- reward_concentration: Gini coefficient of reward distribution

## Baseline Comparison

Compare against:
1. Random baseline (uniform random actions)
2. Simple heuristic baseline (always move right)
3. Historical checkpoint (mappo_academy_3_vs_1_with_keeper_best.pt)
4. New trained checkpoint

## Evaluation Protocol

For each checkpoint:
1. Run 50 deterministic episodes
2. Compute ground-truth metrics from engine stats
3. Compare action distributions
4. Detect behavioral collapse (high entropy -> low entropy transition)
