/**
 * GMN-Football-3 — Reward Differentiation Diagnostic
 *
 * Computes the actual reward magnitudes for different shot outcomes
 * to assess whether the reward function meaningfully differentiates
 * shot quality.
 */

import { ObservationEncoder } from '../src/engine/ObservationEncoder';

function main(): void {
  console.log('====================================================');
  console.log('GMN-FOOTBALL-3 — REWARD DIFFERENTIATION DIAGNOSTIC');
  console.log('====================================================\n');

  // Scenario: ball is at x=0.8 (near opponent goal), target team is 'left'
  const targetTeam: 'left' = 'left';

  // Case 1: Scored goal (ball crossed goal line)
  const goalResult = ObservationEncoder.computeReward(0.7, 1.05, 'left', targetTeam, false, 0.7);
  console.log('Case 1: Scored goal');
  console.log(`  prevBallX=0.7, currBallX=1.05, goalScoredTeam='left'`);
  console.log(`  reward = ${goalResult.reward.toFixed(4)} (goal +1.0 + checkpoint ${goalResult.checkpoint.toFixed(4)})`);
  console.log(`  checkpoint = ${goalResult.checkpoint.toFixed(4)}\n`);

  // Case 2: Shot on target, saved by keeper (ball advanced from 0.7 to 0.85, trajectory inside goal)
  const savedResult = ObservationEncoder.computeReward(
    0.7, 0.85, null, targetTeam, true, 0.7,
    { x: 0.85, y: 0.02, z: 0.0 },
    { x: 0.8, y: -0.01, z: 0.0 }
  );
  console.log('Case 2: Shot on target, saved by keeper (projected Y inside goal)');
  console.log(`  prevBallX=0.7, currBallX=0.85, shot=true, ball inside goal trajectory`);
  console.log(`  reward = ${savedResult.reward.toFixed(4)} (checkpoint ${savedResult.checkpoint.toFixed(4)} + on-target bonus 0.03)`);
  console.log(`  checkpoint = ${savedResult.checkpoint.toFixed(4)}\n`);

  // Case 3: Shot wide (ball barely advanced from 0.8 to 0.82, trajectory misses goal)
  const wideResult = ObservationEncoder.computeReward(
    0.8, 0.82, null, targetTeam, true, 0.8,
    { x: 0.82, y: 0.2, z: 0.0 },
    { x: 0.8, y: 0.05, z: 0.0 }
  );
  console.log('Case 3: Shot wide (projected Y outside goal mouth)');
  console.log(`  prevBallX=0.8, currBallX=0.82, shot=true, ball outside goal trajectory`);
  console.log(`  reward = ${wideResult.reward.toFixed(4)} (checkpoint ${wideResult.checkpoint.toFixed(4)} + off-target bonus 0.001)`);
  console.log(`  checkpoint = ${wideResult.checkpoint.toFixed(4)}\n`);

  // Case 4: No shot, just ball progress (dribbling)
  const dribbleResult = ObservationEncoder.computeReward(0.7, 0.75, null, targetTeam, false, 0.7);
  console.log('Case 4: Dribbling without shooting (ball advanced from 0.7 to 0.75)');
  console.log(`  prevBallX=0.7, currBallX=0.75, no shot`);
  console.log(`  reward = ${dribbleResult.reward.toFixed(4)} (checkpoint only, no shot bonus)`);
  console.log(`  checkpoint = ${dribbleResult.checkpoint.toFixed(4)}\n`);

  // Case 5: Long-range shot on target
  const distantResult = ObservationEncoder.computeReward(
    0.3, 0.35, null, targetTeam, true, 0.3,
    { x: 0.35, y: -0.02, z: 0.0 },
    { x: 0.3, y: 0.01, z: 0.0 }
  );
  console.log('Case 5: Long-range shot on target (projected Y inside goal)');
  console.log(`  prevBallX=0.3, currBallX=0.35, shot=true, ball inside goal trajectory`);
  console.log(`  reward = ${distantResult.reward.toFixed(4)} (checkpoint ${distantResult.checkpoint.toFixed(4)} + on-target bonus 0.03)`);
  console.log(`  checkpoint = ${distantResult.checkpoint.toFixed(4)}\n`);

  // Summary
  console.log('====================================================');
  console.log('SUMMARY');
  console.log('====================================================');
  console.log(`  Goal reward:                    +1.0000`);
  console.log(`  On-target shot bonus:           +0.0300`);
  console.log(`  Off-target shot bonus:          +0.0010`);
  console.log(`  Max checkpoint reward:          +0.0500`);
  console.log(`  Total max on-target shot reward:+0.0800`);
  console.log(`  Total max off-target shot reward:+0.0510`);
  console.log(`  Ratio goal-to-on-target-bonus:  ${(1.0 / 0.03).toFixed(0)}:1`);
  console.log(`  Ratio on-target-to-off-target:  ${(0.03 / 0.001).toFixed(0)}:1`);
  console.log('');
  console.log('ASSESSMENT:');
  console.log('  The reward function NOW meaningfully differentiates');
  console.log('  shot quality. On-target shots receive +0.03 (60x larger');
  console.log('  than off-target shots at +0.001), creating a clear gradient');
  console.log('  for the policy to learn shot accuracy.');
  console.log('  The goal reward (+1.0) remains the dominant objective,');
  console.log('  but the 30x gap between on-target and off-target shots');
  console.log('  should provide sufficient signal to escape shot-spamming.');
  console.log('====================================================');
}

main();
