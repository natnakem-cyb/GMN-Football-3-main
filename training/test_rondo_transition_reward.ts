/**
 * GMN-Football-3 — Rondo Defender Transition Reward Test (Audit P0 fix)
 *
 * Verifies that ObservationEncoder.computeRondoReward pays the defender +0.2
 * ONCE on a possession transition to right, not +0.2 per tick.
 *
 * Regression: the pre-fix code added +0.2 every tick while the right team held
 * the ball, letting a defender accumulate ~12.0 over 60 ticks of continuous
 * possession. Post-fix, 60 ticks with one interception must total ≈ 0.2
 * (plus optional distance-closing shaping, which is 0 when distances are equal).
 */

import { ObservationEncoder } from '../src/engine/ObservationEncoder';

function assertApprox(actual: number, expected: number, tolerance = 1e-6, msg = ''): void {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`Rondo mismatch${msg ? ': ' + msg : ''}: expected ${expected}, got ${actual}`);
  }
}

function main(): void {
  console.log('====================================================');
  console.log('GMN-FOOTBALL-3 — RONDO DEFENDER TRANSITION REWARD TEST (Audit P0)');
  console.log('====================================================');

  // 60 ticks of continuous right possession, ONE interception on tick 0.
  // Distances are held equal so no distance-closing shaping is paid.
  let defenderTotal = 0.0;
  for (let tick = 0; tick < 60; tick++) {
    const justWon = tick === 0; // only the first tick is the transition
    const { defenderReward } = ObservationEncoder.computeRondoReward({
      prevBallX: 0.0,
      currBallX: 0.0,
      currBallY: 0.0,
      ballOwnerTeam: 'right',
      lastPassTeam: null,
      lastPassCompleted: false,
      defenderDistToBall: 0.3,
      prevDefenderDistToBall: 0.3,
      drillRadius: 0.35,
      consecutivePossessionTime: tick,
      defenderJustWonPossession: justWon,
    });
    defenderTotal += defenderReward;
  }
  assertApprox(defenderTotal, 0.2, 1e-6, '60 ticks continuous right possession after one interception');
  console.log(`   ✓ Defender total over 60 ticks of continuous possession = ${defenderTotal.toFixed(4)} (transition-only, not ~12.0)`);

  // No transition (right already had the ball from a previous tick): 0 per tick.
  const { defenderReward: noTransitionReward } = ObservationEncoder.computeRondoReward({
    prevBallX: 0.0,
    currBallX: 0.0,
    currBallY: 0.0,
    ballOwnerTeam: 'right',
    lastPassTeam: null,
    lastPassCompleted: false,
    defenderDistToBall: 0.3,
    prevDefenderDistToBall: 0.3,
    drillRadius: 0.35,
    consecutivePossessionTime: 5,
    defenderJustWonPossession: false,
  });
  assertApprox(noTransitionReward, 0.0, 1e-6, 'no transition, right already holding -> 0');
  console.log(`   ✓ No transition tick: defender reward = ${noTransitionReward.toFixed(4)} (0)`);

  // Transition from LEFT: defender wins possession -> +0.2 exactly once.
  const transition = ObservationEncoder.computeRondoReward({
    prevBallX: 0.0,
    currBallX: 0.2,
    currBallY: 0.0,
    ballOwnerTeam: 'right',
    lastPassTeam: null,
    lastPassCompleted: false,
    defenderDistToBall: 0.3,
    prevDefenderDistToBall: 0.3,
    drillRadius: 0.35,
    consecutivePossessionTime: 0,
    defenderJustWonPossession: true,
  });
  assertApprox(transition.defenderReward, 0.2, 1e-6, 'transition to right -> +0.2 once');
  console.log(`   ✓ Transition (left -> right) defender reward = ${transition.defenderReward.toFixed(4)} (+0.2 once)`);

  // Attacker +0.01 possession and +0.1 pass bonuses remain unchanged.
  const attackerHolding = ObservationEncoder.computeRondoReward({
    prevBallX: 0.0,
    currBallX: 0.1,
    currBallY: 0.1,
    ballOwnerTeam: 'left',
    lastPassTeam: null,
    lastPassCompleted: false,
    defenderDistToBall: 0.5,
    prevDefenderDistToBall: 0.5,
    drillRadius: 0.35,
    consecutivePossessionTime: 0,
    defenderJustWonPossession: false,
  });
  assertApprox(attackerHolding.attackerReward, 0.01, 1e-6, 'attacker possession in drill area +0.01');
  const attackerPass = ObservationEncoder.computeRondoReward({
    prevBallX: 0.0,
    currBallX: 0.1,
    currBallY: 0.1,
    ballOwnerTeam: 'left',
    lastPassTeam: 'left',
    lastPassCompleted: true,
    defenderDistToBall: 0.5,
    prevDefenderDistToBall: 0.5,
    drillRadius: 0.35,
    consecutivePossessionTime: 0,
    defenderJustWonPossession: false,
  });
  assertApprox(attackerPass.attackerReward, 0.11, 1e-6, 'attacker pass +0.1 on top of +0.01 possession');
  console.log('   ✓ Attacker +0.01 possession and +0.1 pass bonuses unchanged');

  console.log('\n====================================================');
  console.log('✓ ALL RONDO TRANSITION REWARD CHECKS PASSED');
  console.log('====================================================');
}

try {
  main();
  console.log('\n[OK] Rondo transition reward test completed successfully.');
} catch (err: any) {
  console.error('✗ RONDO TRANSITION REWARD TEST FAILED:', err instanceof Error ? err.message : err);
  process.exit(1);
}