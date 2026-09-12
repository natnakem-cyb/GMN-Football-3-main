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
  // Reward = +2.0 goal + 0.02 checkpoint (ball advanced past high-water mark on
  // the goal tick). The checkpoint is added to the total; the old assertion of
  // exactly 2.0 was stale and failed before the audit fixes.
  assertApprox(result.reward, 2.02, 1e-9, 'goal for target team + checkpoint');
  assertApprox(result.checkpoint, 0.02, 1e-9, 'goal checkpoint');
  console.log('   ✓ Goal scored by target team: reward=+2.02 (2.0 goal + 0.02 checkpoint)');

  // Test 2: Goal conceded by target team
  result = ObservationEncoder.computeReward(0.5, 0.5, 'right', 'left', 0.5);
  assertApprox(result.reward, -1.0, 1e-9, 'goal conceded');
  assertApprox(result.checkpoint, 0.0, 1e-9, 'concede checkpoint');
  console.log('   ✓ Goal conceded by target team: reward=-1.0');

  // Test 3: No goal, ball progress
  result = ObservationEncoder.computeReward(0.3, 0.35, null, 'left', 0.3);
  assertApprox(result.reward, 0.01, 1e-9, 'checkpoint advance');
  assertApprox(result.checkpoint, 0.01, 1e-9, 'checkpoint value');
  console.log('   ✓ Checkpoint advance (0.05 * 0.2 = 0.01): reward=+0.01');

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

  // Test 8: Checkpoint capped at 0.02 (formula: min(0.02, deltaX * 0.2))
  result = ObservationEncoder.computeReward(0.0, 0.2, null, 'left', 0.0);
  assertApprox(result.reward, 0.02, 1e-9, 'checkpoint cap');
  assertApprox(result.checkpoint, 0.02, 1e-9, 'checkpoint cap value');
  console.log('   ✓ Checkpoint reward capped at +0.02');

  // Test 9: Verified pass completion (+0.15)
  result = ObservationEncoder.computeReward(0.5, 0.55, null, 'left', 0.5, true);
  assertApprox(result.checkpoint, 0.01, 1e-9, 'pass completion checkpoint');
  assertApprox(result.reward, 0.16, 1e-9, 'pass completion total');
  console.log('   ✓ Verified pass completion: reward=+0.16 (0.01 checkpoint + 0.15 pass)');

  // Test 10: Ball regression despite pass completion flag (checkpoint only pays on progress,
  // but the verified-pass bonus +0.15 is independent of ball direction).
  result = ObservationEncoder.computeReward(0.5, 0.45, null, 'left', 0.5, true);
  assertApprox(result.reward, 0.15, 1e-9, 'regression with pass flag pays only pass bonus');
  assertApprox(result.checkpoint, 0.0, 1e-9, 'regression checkpoint with pass flag');
  console.log('   ✓ Ball regression with pass flag: reward=+0.15 (pass bonus only, no checkpoint)');

  // Test 11: Possession-gated progress — left owns ball, X advances -> checkpoint paid.
  result = ObservationEncoder.computeReward(0.3, 0.4, null, 'left', 0.3, false, 'left');
  assertApprox(result.reward, 0.02, 1e-9, 'left-owned progress (delta 0.1 * 0.2 capped)');
  assertApprox(result.checkpoint, 0.02, 1e-9, 'left-owned checkpoint');
  assertApprox(result.newMaxBallProgressX, 0.4, 1e-9, 'left-owned high-water mark advances');
  console.log('   ✓ Possession-gated: left owns + X advance -> checkpoint paid');

  // Test 12: Possession-gated — NO owner (null), X advances -> checkpoint == 0 and high-water mark UNCHANGED.
  result = ObservationEncoder.computeReward(0.3, 0.4, null, 'left', 0.3, false, null);
  assertApprox(result.reward, 0.0, 1e-9, 'loose-ball progress gated');
  assertApprox(result.checkpoint, 0.0, 1e-9, 'loose-ball checkpoint gated');
  assertApprox(result.newMaxBallProgressX, 0.3, 1e-9, 'loose-ball high-water mark NOT advanced by opponent movement');
  console.log('   ✓ Possession-gated: loose ball (no owner) + X advance -> no checkpoint, mark not raised');

  // Test 13: Possession-gated — right owns ball, X advances -> checkpoint == 0 and high-water mark UNCHANGED.
  result = ObservationEncoder.computeReward(0.3, 0.4, null, 'left', 0.3, false, 'right');
  assertApprox(result.reward, 0.0, 1e-9, 'opponent-owned progress gated');
  assertApprox(result.checkpoint, 0.0, 1e-9, 'opponent-owned checkpoint gated');
  assertApprox(result.newMaxBallProgressX, 0.3, 1e-9, 'opponent-owned high-water mark NOT advanced');
  console.log('   ✓ Possession-gated: right owns + X advance -> no checkpoint, mark not raised');

  // Test 14: Legacy (undefined owner) retains ungated behavior for backward compat.
  result = ObservationEncoder.computeReward(0.3, 0.4, null, 'left', 0.3, false);
  assertApprox(result.reward, 0.02, 1e-9, 'legacy ungated progress still paid');
  assertApprox(result.checkpoint, 0.02, 1e-9, 'legacy ungated checkpoint');
  console.log('   ✓ Legacy (undefined owner): ungated behavior preserved');

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
