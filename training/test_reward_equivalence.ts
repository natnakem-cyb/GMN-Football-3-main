/**
 * GMN-Football-3 — Reward Shaping Equivalence Unit Test
 *
 * Verifies ObservationEncoder.computeReward() produces exact expected values
 * for known inputs, ensuring no drift in the TS reward implementation.
 */

import { ObservationEncoder } from '../src/engine/ObservationEncoder';

function assertApprox(actual: number, expected: number, tolerance = 1e-9, msg = ''): void {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`Reward mismatch${msg ? ': ' + msg : ''}: expected ${expected}, got ${actual}`);
  }
}

function main(): void {
  console.log('====================================================');
  console.log('GMN-FOOTBALL-3 — REWARD SHAPING EQUIVALENCE TEST (Phase 11)');
  console.log('====================================================');

  // Test 1: Goal scored by target team
  let result = ObservationEncoder.computeReward(0.5, 1.05, 'left', 'left', 0.5);
  assertApprox(result.reward, 2.0, 1e-9, 'goal for target team');
  assertApprox(result.checkpoint, 0.02, 1e-9, 'goal checkpoint');
  console.log('   ✓ Goal scored by target team: reward=+2.0');

  // Test 2: Goal conceded by target team
  result = ObservationEncoder.computeReward(0.5, 0.5, 'right', 'left', 0.5);
  assertApprox(result.reward, -1.0, 1e-9, 'goal conceded');
  assertApprox(result.checkpoint, 0.0, 1e-9, 'concede checkpoint');
  console.log('   ✓ Goal conceded by target team: reward=-1.0');

  // Test 3: No goal, ball progress
  result = ObservationEncoder.computeReward(0.3, 0.35, null, 'left', 0.3);
  assertApprox(result.reward, 0.025, 1e-9, 'checkpoint advance');
  assertApprox(result.checkpoint, 0.025, 1e-9, 'checkpoint value');
  console.log('   ✓ Checkpoint advance (0.05 * 0.5): reward=+0.025');

  // Test 4: No shot bonus in Phase 11 — only checkpoint
  result = ObservationEncoder.computeReward(0.3, 0.3, null, 'left', 0.3);
  assertApprox(result.reward, 0.0, 1e-9, 'no progress no reward');
  assertApprox(result.checkpoint, 0.0, 1e-9, 'no progress checkpoint');
  console.log('   ✓ No progress / no shot: reward=0.0 (shot bonus removed)');

  // Test 5: Monotonic checkpoint — should not pay for regressing ball position
  result = ObservationEncoder.computeReward(0.8, 0.7, null, 'left', 0.8);
  assertApprox(result.reward, 0.0, 1e-9, 'regression no reward');
  assertApprox(result.checkpoint, 0.0, 1e-9, 'regression checkpoint');
  console.log('   ✓ Ball regression: reward=0.0 (no checkpoint paid)');

  // Test 6: Controlled-team invariant — right-team target must throw
  let invariantThrew = false;
  try {
    ObservationEncoder.computeReward(-0.3, -0.35, null, 'right', -0.3);
  } catch (err: any) {
    invariantThrew = String(err.message).includes('GMN Reward Invariant Violation');
  }
  if (!invariantThrew) {
    throw new Error('Reward invariant: computeReward(targetTeam="right") must throw [GMN Reward Invariant Violation]');
  }
  console.log('   ✓ Controlled-team invariant enforced (right-team target throws)');

  // Test 7: Small movement below threshold
  result = ObservationEncoder.computeReward(0.5, 0.502, null, 'left', 0.5);
  assertApprox(result.reward, 0.0, 1e-9, 'below threshold');
  console.log('   ✓ Small movement below threshold: reward=0.0');

  // Test 8: Checkpoint capped at 0.05
  result = ObservationEncoder.computeReward(0.0, 0.2, null, 'left', 0.0);
  assertApprox(result.reward, 0.05, 1e-9, 'checkpoint cap');
  assertApprox(result.checkpoint, 0.05, 1e-9, 'checkpoint cap value');
  console.log('   ✓ Checkpoint reward capped at +0.05');

  // Test 9: Verified pass completion (+0.15)
  result = ObservationEncoder.computeReward(0.5, 0.55, null, 'left', 0.5, true);
  assertApprox(result.checkpoint, 0.025, 1e-9, 'pass completion checkpoint');
  assertApprox(result.reward, 0.175, 1e-9, 'pass completion total');
  console.log('   ✓ Verified pass completion: reward=+0.175 (checkpoint + 0.15)');

  // Test 10: Ball regression despite pass completion flag (checkpoint only pays on progress)
  result = ObservationEncoder.computeReward(0.5, 0.45, null, 'left', 0.5, true);
  assertApprox(result.reward, 0.0, 1e-9, 'regression with pass flag');
  assertApprox(result.checkpoint, 0.0, 1e-9, 'regression checkpoint with pass flag');
  console.log('   ✓ Ball regression with pass flag: reward=0.0 (no checkpoint paid)');

  console.log('\n====================================================');
  console.log('✓ ALL REWARD SHAPING CHECKS PASSED');
  console.log('====================================================');
}

try {
  main();
  console.log('\n[OK] Reward equivalence test completed successfully.');
} catch (err: any) {
  console.error('✗ REWARD EQUIVALENCE TEST FAILED:', err instanceof Error ? err.message : err);
  process.exit(1);
}
