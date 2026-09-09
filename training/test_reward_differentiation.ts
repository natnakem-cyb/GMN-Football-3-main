/**
 * GMN-Football-3 — Reward Differentiation Diagnostic
 *
 * Computes the actual reward magnitudes for different outcomes
 * to assess whether the reward function meaningfully differentiates
 * verified completions from attempts.
 */

import { ObservationEncoder } from '../src/engine/ObservationEncoder';

function main(): void {
  console.log('====================================================');
  console.log('GMN-FOOTBALL-3 — REWARD DIFFERENTIATION DIAGNOSTIC (Phase 11)');
  console.log('====================================================\n');

  const targetTeam: 'left' = 'left';

  // Case 1: Goal scored
  const goalResult = ObservationEncoder.computeReward(0.7, 1.05, 'left', targetTeam, 0.7);
  console.log('Case 1: Scored goal');
  console.log(`  prevBallX=0.7, currBallX=1.05, goalScoredTeam='left'`);
  console.log(`  reward = ${goalResult.reward.toFixed(4)} (goal +2.0 + checkpoint ${goalResult.checkpoint.toFixed(4)})`);
  console.log(`  checkpoint = ${goalResult.checkpoint.toFixed(4)}\n`);

  // Case 2: Ball progress (no shot reward in Phase 11)
  const progressResult = ObservationEncoder.computeReward(0.7, 0.85, null, targetTeam, 0.7);
  console.log('Case 2: Ball progress without shot');
  console.log(`  prevBallX=0.7, currBallX=0.85, no shot`);
  console.log(`  reward = ${progressResult.reward.toFixed(4)} (checkpoint only)`);
  console.log(`  checkpoint = ${progressResult.checkpoint.toFixed(4)}\n`);

  // Case 3: No progress
  const noProgressResult = ObservationEncoder.computeReward(0.8, 0.82, null, targetTeam, 0.8);
  console.log('Case 3: Minimal ball progress');
  console.log(`  prevBallX=0.8, currBallX=0.82`);
  console.log(`  reward = ${noProgressResult.reward.toFixed(4)} (checkpoint only)`);
  console.log(`  checkpoint = ${noProgressResult.checkpoint.toFixed(4)}\n`);

  // Case 4: Dribbling without shooting
  const dribbleResult = ObservationEncoder.computeReward(0.7, 0.75, null, targetTeam, 0.7);
  console.log('Case 4: Dribbling without shooting (ball advanced from 0.7 to 0.75)');
  console.log(`  prevBallX=0.7, currBallX=0.75, no shot`);
  console.log(`  reward = ${dribbleResult.reward.toFixed(4)} (checkpoint only, no shot bonus)`);
  console.log(`  checkpoint = ${dribbleResult.checkpoint.toFixed(4)}\n`);

  // Case 5: Pass completion verified
  const passResult = ObservationEncoder.computeReward(0.5, 0.55, null, targetTeam, 0.5, true);
  console.log('Case 5: Verified pass completion');
  console.log(`  prevBallX=0.5, currBallX=0.55, passCompleted=true`);
  console.log(`  reward = ${passResult.reward.toFixed(4)} (checkpoint + pass completion bonus)`);
  console.log(`  checkpoint = ${passResult.checkpoint.toFixed(4)}\n`);

  // Summary
  console.log('====================================================');
  console.log('SUMMARY');
  console.log('====================================================');
  console.log(`  Goal reward:                    +2.0000`);
  console.log(`  Verified pass completion:       +0.1500`);
  console.log(`  Max checkpoint reward:          +0.0500`);
  console.log(`  Shot attempt rewards:           REMOVED (Phase 11)`);
  console.log(`  Action cost (per ball action):  -0.0100 (applied in Python shaper)`);
  console.log('');
  console.log('ASSESSMENT:');
  console.log('  Shot-attempt rewards have been removed. Only verified');
  console.log('  completions (pass_completed, goal) and explicit penalties');
  console.log('  (missed shot -0.05, action cost -0.01) remain.');
  console.log('====================================================');
}

main();
