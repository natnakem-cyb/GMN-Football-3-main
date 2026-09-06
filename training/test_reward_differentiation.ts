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
  const basePrevX = 0.7;
  const baseCurrX = 0.8;
  const targetTeam: 'left' = 'left';

  // Case 1: Scored goal (ball crossed goal line)
  const goalResult = ObservationEncoder.computeReward(0.7, 1.05, 'left', targetTeam, false, 0.7);
  console.log('Case 1: Scored goal');
  console.log(`  prevBallX=0.7, currBallX=1.05, goalScoredTeam='left'`);
  console.log(`  reward = ${goalResult.reward.toFixed(4)} (goal +1.0, no checkpoint since goal overrides)`);
  console.log(`  checkpoint = ${goalResult.checkpoint.toFixed(4)}\n`);

  // Case 2: Shot on target but saved (ball advanced but no goal)
  const savedResult = ObservationEncoder.computeReward(0.7, 0.85, null, targetTeam, true, 0.7);
  console.log('Case 2: Shot on target, saved by keeper (ball advanced from 0.7 to 0.85)');
  console.log(`  prevBallX=0.7, currBallX=0.85, goalScoredTeam=null, shotTakenByTargetTeam=true`);
  console.log(`  reward = ${savedResult.reward.toFixed(4)} (checkpoint ${savedResult.checkpoint.toFixed(4)} + shot bonus 0.005)`);
  console.log(`  checkpoint = ${savedResult.checkpoint.toFixed(4)}\n`);

  // Case 3: Shot wide (ball didn't advance much)
  const wideResult = ObservationEncoder.computeReward(0.8, 0.82, null, targetTeam, true, 0.8);
  console.log('Case 3: Shot wide (ball barely advanced from 0.8 to 0.82)');
  console.log(`  prevBallX=0.8, currBallX=0.82, goalScoredTeam=null, shotTakenByTargetTeam=true`);
  console.log(`  reward = ${wideResult.reward.toFixed(4)} (checkpoint ${wideResult.checkpoint.toFixed(4)} + shot bonus 0.005)`);
  console.log(`  checkpoint = ${wideResult.checkpoint.toFixed(4)}\n`);

  // Case 4: No shot, just ball progress (dribbling)
  const dribbleResult = ObservationEncoder.computeReward(0.7, 0.75, null, targetTeam, false, 0.7);
  console.log('Case 4: Dribbling without shooting (ball advanced from 0.7 to 0.75)');
  console.log(`  prevBallX=0.7, currBallX=0.75, goalScoredTeam=null, shotTakenByTargetTeam=false`);
  console.log(`  reward = ${dribbleResult.reward.toFixed(4)} (checkpoint only, no shot bonus)`);
  console.log(`  checkpoint = ${dribbleResult.checkpoint.toFixed(4)}\n`);

  // Case 5: Shot from distance (low quality)
  const distantResult = ObservationEncoder.computeReward(0.3, 0.35, null, targetTeam, true, 0.3);
  console.log('Case 5: Long-range shot (ball advanced from 0.3 to 0.35)');
  console.log(`  prevBallX=0.3, currBallX=0.35, goalScoredTeam=null, shotTakenByTargetTeam=true`);
  console.log(`  reward = ${distantResult.reward.toFixed(4)} (checkpoint ${distantResult.checkpoint.toFixed(4)} + shot bonus 0.005)`);
  console.log(`  checkpoint = ${distantResult.checkpoint.toFixed(4)}\n`);

  // Summary
  console.log('====================================================');
  console.log('SUMMARY');
  console.log('====================================================');
  console.log(`  Goal reward:                    +1.0000`);
  console.log(`  Shot bonus (any shot):          +0.0050`);
  console.log(`  Max checkpoint reward:          +0.0500`);
  console.log(`  Total max non-goal shot reward: +0.0550`);
  console.log(`  Ratio goal-to-shot-bonus:       ${(1.0 / 0.005).toFixed(0)}:1`);
  console.log('');
  console.log('ASSESSMENT:');
  console.log('  The reward function does NOT meaningfully differentiate');
  console.log('  shot quality. All shots receive the same +0.005 bonus');
  console.log('  regardless of whether they are on-target, wide, or saved.');
  console.log('  The only quality signal is the +1.0 goal reward, which is');
  console.log('  200x larger than the shot bonus. This creates a sparse');
  console.log('  reward landscape where the policy cannot learn shot');
  console.log('  accuracy through intermediate rewards.');
  console.log('');
  console.log('  This likely explains the 500k run plateau: the policy');
  console.log('  converges to shot-spamming (getting +0.005 per shot) but');
  console.log('  cannot improve shot quality because there is no reward');
  console.log('  gradient for on-target shots vs off-target shots.');
  console.log('====================================================');
}

main();
