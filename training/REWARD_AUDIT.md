# Phase 2: Reward Audit for academy_3_vs_1_with_keeper

## Base Reward: ObservationEncoder.computeReward()

| Component | Value | Frequency | Recipient | Conditions | Exploit Risk |
|-----------|------:|----------:|-----------|------------|--------------|
| Goal scored | +1.0 | Terminal | All agents (shared) | Left team scores | Low — terminal event |
| Goal conceded | -1.0 | Terminal | All agents (shared) | Right team scores | Low — terminal event |
| Checkpoint (progress) | +0.01 to +0.05 | Per tick | All agents (shared) | Ball advances >0.005 toward opponent goal | **HIGH** — can dribble forward without passing/shooting |
| Shot quality (on-target) | +0.03 | Per shot | All agents (shared) | Shot projects onto goal mouth | Medium — can take low-quality shots |
| Shot quality (off-target) | +0.001 | Per shot | All agents (shared) | Any shot attempt | Medium — spam shots for tiny reward |

## Scenario-Specific Reward: None

`academy_3_vs_1_with_keeper` has NO `ScenarioHandler`, so the base reward from `ObservationEncoder.computeReward()` is used directly. No rondo-style override applies.

## CooperativeRewardShaper

| Component | Value | Frequency | Recipient | Conditions | Exploit Risk |
|-----------|------:|----------:|-----------|------------|--------------|
| Pass completion | +0.25 | Per pass | Passing agent | Left-team pass completed | Low — requires actual pass |
| Assisted goal | +0.50 | Per goal | All left-team agents | Pass chain >0 before goal | Low — requires goal |
| Solitary shot | -0.30 | Per shot | Shooter | Pass chain ==0 before shot | Low — discourages shooting without passing |
| Ball-hogging | -0.005 per tick | Per tick | Ball holder | Single agent holds >30 ticks | Low — can avoid by passing every 29 ticks |

## Reward Flow Summary

```
GameEngine.step()
    ↓
ObservationEncoder.computeReward()
    - Goal: +1.0 / -1.0 (terminal)
    - Checkpoint: +0.01–0.05 per tick (dense)
    - Shot quality: +0.03 / +0.001 per shot
    ↓
scenarioHandler?.computeReward() → NOOP for academy_3_vs_1_with_keeper
    ↓
Bridge sends reward to Python client
    ↓
GMNMultiAgentEnv.step()
    - Rondo: per-team assignment (not applicable here)
    - Non-rondo: shared_reward to all agents
    ↓
CooperativeRewardShaper (if enabled)
    - Adds pass bonus, assisted goal bonus
    - Subtracts solitary shot penalty, ball-hogging penalty
    ↓
Final per-agent reward
```

## Key Finding: Reward Dominance

The **checkpoint/progress reward dominates** the episode return for academy_3_vs_1_with_keeper:

- **Dense reward:** +0.01–0.05 per tick for ball advancement
- **Sparse reward:** +1.0 for goal (rare), +0.25 for pass (requires action selection), -0.30 for solitary shot

**The easiest high-return strategy is dribbling forward, not passing or shooting.**

## Exploitation Mechanisms

1. **Dribbling reward farming:** Agents can dribble the ball forward (X direction) to accumulate +0.01–0.05 per tick without ever passing or shooting.
2. **Progress without intent:** The checkpoint reward triggers purely on X advancement, not on tactical intent.
3. **Shot spamming:** Agents can take low-quality shots (+0.001 each) without penalty if they avoid the solitary-shot penalty by passing first.
4. **Pass avoidance:** Passing is not required for reward maximization. The base reward has no pass component.

## Root Cause of Reward/Behavior Divergence

The reward function incentivizes **ball progression** (dribbling) far more than **ball distribution** (passing) or **shot accuracy**. The CooperativeRewardShaper (+0.25 per pass, -0.30 per solitary shot) is insufficient to overcome the dense progress reward.

**Numerical evidence from validation_report.md:**
- seed42_best: mean reward +0.5428, 0.02 passes/ep, 0.00 shots/ep
- seed43_best: mean reward +0.2801, 0.00 passes/ep, 0.00 shots/ep
- seed44_best: mean reward +0.3081, 0.19 passes/ep, 0.00 shots/ep
- seed137_best: mean reward +0.2594, 0.44 passes/ep, 0.00 shots/ep

Even the best checkpoint (seed42_best) with 37.4% goal rate achieves this through direct dribbling, not passing.
