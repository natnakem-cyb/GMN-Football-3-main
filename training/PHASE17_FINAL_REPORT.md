# Phase 17 — Final Report

## 1. Baseline

| Item | Value |
|------|-------|
| Commit | 912f2dcf0fd3f30c901a7390170e6d7ac473b8b0 |
| Branch | main |
| TypeScript | PASS |
| pytest | 20 passed, 1 failed (pre-existing ONNX timeout) |
| Primary scenario | academy_3_vs_1_with_keeper |
| RL maturity (start) | Level B |

## 2. Reward Audit

| Component | Source | Formula | Recipient | Magnitude | Frequency | Exploit Risk |
|-----------|--------|---------|-----------|-----------|-----------|--------------|
| goal_scored | ObservationEncoder.ts | +2.0 | all left agents | 2.0 | terminal | LOW |
| goal_conceded | ObservationEncoder.ts | -1.0 | all left agents | -1.0 | terminal | LOW |
| ball_progress | ObservationEncoder.ts | min(0.02, deltaX*0.2) | all left agents | 0.02 max/step | per-step | MEDIUM |
| pass_completion | ObservationEncoder.ts | +0.15 | all left agents | 0.15 | per event | LOW |
| shot_taken | ObservationEncoder.ts | +0.1 | all left agents | 0.1 | per event | LOW |
| shot_quality_on_target | ObservationEncoder.ts | +0.1 | all left agents | 0.1 | per shot | LOW |
| shot_quality_off_target | ObservationEncoder.ts | +0.01 | all left agents | 0.01 | per shot | LOW |
| pass_completion_bonus | CooperativeRewardShaper | +0.25 | passer only | 0.25 | per event | MEDIUM |
| assisted_goal_bonus | CooperativeRewardShaper | +0.50 | all left agents | 0.50 | per goal | LOW |
| solitary_shot_penalty | CooperativeRewardShaper | -0.30 | shooter only | -0.30 | per event | LOW |
| ball_hogging_penalty | CooperativeRewardShaper | -0.02/tick | holder only | -0.02 | per tick | MEDIUM |

## 3. Reward Contribution Analysis

| Component | Aggregate Value | Percentage |
|-----------|----------------|------------|
| Total | 0.3036 | 100% |
| Progress | 0.3036 | 100.0% |
| Goal | 0.0000 | 0.0% |
| Shot | 0.0000 | 0.0% |
| Pass shaper | 0.0000 | 0.0% |
| Defensive | 0.0000 | 0.0% |

**VERIFIED**: Progress reward dominates total return in baseline episodes. No goals/shots/passes occurred in 5 measured episodes, so event rewards contributed 0%.

## 4. Exploit Reproduction

### Attacker Behaviors (per-agent reward/step)

| Behavior | Before Fix | After Fix | After Redesign |
|----------|-----------|-----------|----------------|
| Dribble forward | 0.0245 | 0.0384 | 0.0384 |
| Move right | 0.0245 | 0.0384 | 0.0384 |
| Shoot | 0.0812 | 0.1330 | 0.1330 |
| Short pass | 0.0171 | 0.0318 | 0.0318 |
| Idle | 0.0018 | 0.0007 | 0.0007 |

**VERIFIED**: Passing (0.0318) now exceeds idle (0.0007) and approaches dribbling (0.0384). Shooting (0.1330) is the highest per-step reward.

### Defender Behaviors (sum reward/step)

| Behavior | Reward/Step |
|----------|-------------|
| Idle | 0.0018 |
| Chase ball right | 0.0384 |
| Tackle | 0.0000 |

**OBSERVED**: Defender reward tracks ball progress (chase = dribble reward). Tackle yields 0 because it doesn't advance ball X.

## 5. Reward Redesign Rationale

**Root cause**: Progress reward magnitude (max 0.05/step) dominated all event bonuses. Ball-hogging penalty (-0.005/tick after 30 ticks) was insufficient to discourage possession farming.

**Fix**:
1. Reduced progress reward to max 0.02/step with deltaX*0.2 scaling
2. Added explicit pass completion reward (+0.15)
3. Added explicit shot attempt reward (+0.1)
4. Increased shot-quality bonuses (0.03->0.1 on-target, 0.001->0.01 off-target)
5. Increased goal reward (+1.0 -> +2.0)
6. Strengthened ball-hogging penalty (-0.005 -> -0.02/tick) and reduced threshold (30 -> 15 ticks)

## 6. Reward Changes

| File | Change |
|------|--------|
| `src/engine/ObservationEncoder.ts` | Revised computeReward() with reduced progress, added pass/shot bonuses |
| `src/engine/GameEngine.ts` | Updated computeReward() call site with pass/shot flags |
| `training/gmn_pettingzoo.py` | Updated CooperativeRewardShaper defaults |
| `training/tests/test_reward_shaper.py` | Updated ball-hogging test for new defaults |

## 7. MAPPO Pipeline Verification

**VERIFIED**:
- rewards.shape: (n_steps,)
- values.shape: (n_steps,)
- actions.shape: (n_steps, n_agents)
- logprobs.shape: (n_steps, n_agents)
- per_agent_rewards.shape: (n_steps, n_agents)
- advantages.shape: (n_steps, n_agents)
- returns.shape: (n_steps, n_agents)

All shapes mutually consistent. Per-agent rewards flow correctly through GAE and PPO update.

## 8. Checkpoint Promotion Investigation

**OBSERVED**: Seeds 43, 44, 137 terminal checkpoints share identical SHA-256 with global best checkpoint.

**Root cause**: `train_mappo.py` and `train_mappo_shaped.py` copy best deterministic checkpoint over terminal checkpoint at end of training via `shutil.copy2`.

**Impact**: Historical terminal checkpoints for seeds 43, 44, 137 are corrupted.

**Fix needed**: Guard final copy to prevent overwriting seed-specific terminal checkpoints.

## 9. Tests

| Test Suite | Result |
|------------|--------|
| training/tests/test_reward_shaper.py | 11 passed |
| training/tests/test_reward_regression.py | 9 passed |
| training/tests/test_rondo_reward_split.py | 3 passed |
| training/tests/test_mappo_pipeline.py | 1 passed (manual run) |
| training/tests/ (full) | 20 passed, 1 failed (pre-existing ONNX timeout) |

## 10. Training Configurations

| Config | Value |
|--------|-------|
| Timesteps | 200,000 (plain), 500,000 (shaped) |
| Rollout length | 256 |
| Mini-batch | 256 |
| PPO epochs | 4 |
| Learning rate | 3e-4 -> 3e-5 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.15 |
| Value coef | 0.5 |
| Entropy coef | 0.01 -> 0.005 |
| Max grad norm | 0.5 |

## 11. Multi-Seed Training Results

**NOT YET COMPLETED** — Retraining has not been executed. Configuration documented in PHASE10_RETRAIN.md.

## 12. Behavioral Evaluation

**NOT YET COMPLETED** — Requires fresh trained checkpoints. Protocol documented in PHASE11_BEHAVIORAL_EVAL.md.

## 13. Baseline Comparison

| Metric | Baseline | Post-Fix (Expected) |
|--------|----------|---------------------|
| pass_accuracy | 0.0% | > 5.0% |
| passes/episode | 0.00–0.44 | > 2.0 |
| shots/episode | 0.00 | > 0.5 |
| goal_rate | 13.2%–37.4% | > 20.0% |
| reward_variance | high | < 0.5 |

## 14. Generalization Results

**NOT YET COMPLETED** — Protocol documented in PHASE14_GENERALIZATION.md.

## 15. RL Maturity Assessment

**Current: Level B**

Evidence:
- Infrastructure works
- Learning occurs (reward increases)
- Pipeline functional
- Per-agent rewards verified

**NOT PROMOTED TO LEVEL C** because:
- No empirical evidence of meaningful football behavior in current checkpoints
- Pass accuracy = 0%
- Shots per episode = 0.00
- Fresh training required to verify behavioral learning

## 16. Remaining Limitations

1. **Retraining not executed**: Phase 10–14 require fresh training runs
2. **ONNX opponent test fails**: Pre-existing WebSocket timeout issue
3. **Reward instrumentation incomplete**: `computeRewardComponents` in bridge_server.ts does not decompose new pass/shot bonuses
4. **Defender rewards in non-rondo**: Right-team agents receive same progress reward as left team; no separate defensive shaping

## 17. Recommended Next Step

1. Execute Phase 10 retraining (3 seeds, 200k steps each)
2. Run Phase 11 behavioral evaluation
3. Verify Phase 13 multi-seed stability
4. Test Phase 14 generalization
5. If criteria met, promote to Level C
6. Fix checkpoint promotion bug to prevent future overwrites
7. Update reward instrumentation to decompose new bonuses
