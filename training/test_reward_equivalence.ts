/**
 * GMN-Football-3 — Reward Shaping Equivalence Unit Test
 *
 * Verifies ObservationEncoder.computeReward() produces exact expected values
 * for known inputs, ensuring no drift in the TS reward implementation.
 */

import { ObservationEncoder } from '../src/engine/ObservationEncoder';
import { TeamSide } from '../src/types/football';

function assertApprox(actual: number, expected: number, tolerance = 1e-9, msg = '' ): void {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`Reward mismatch${msg ? ': ' + msg : ''}: expected ${expected}, got ${actual}`);
  }
}

function main(): void {
  console.log('====================================================');
  console.log('GMN-FOOTBALL-3 — REWARD SHAPING EQUIVALENCE TEST');
  console.log('====================================================');

  // Test 1: Goal scored by target team
  let result = ObservationEncoder.computeReward(0.5, 0.5, 'left', 'left', false, 0.5);
  assertApprox(result.reward, 1.0, 1e-9, 'goal for target team');
  assertApprox(result.checkpoint, 0.0, 1e-9, 'goal checkpoint');
  console.log('   ✓ Goal scored by target team: reward=+1.0');

  // Test 2: Goal conceded by target team
  result = ObservationEncoder.computeReward(0.5, 0.5, 'right', 'left', false, 0.5);
  assertApprox(result.reward, -1.0, 1e-9, 'goal conceded');
  assertApprox(result.checkpoint, 0.0, 1e-9, 'concede checkpoint');
  console.log('   ✓ Goal conceded by target team: reward=-1.0');

  // Test 3: No goal, no shot
  result = ObservationEncoder.computeReward(0.3, 0.35, null, 'left', false, 0.3);
  assertApprox(result.reward, 0.025, 1e-9, 'checkpoint advance');
  assertApprox(result.checkpoint, 0.025, 1e-9, 'checkpoint value');
  console.log('   ✓ Checkpoint advance (0.05 * 0.5): reward=+0.025');

  // Test 4: Off-target shot without ball state — falls back to minimal bonus
  result = ObservationEncoder.computeReward(0.3, 0.3, null, 'left', true, 0.3);
  assertApprox(result.reward, 0.001, 1e-9, 'off-target shot fallback');
  console.log('   ✓ Off-target shot (no ball state): reward=+0.001');

  // Test 5: Monotonic checkpoint — should not pay for regressing ball position
  result = ObservationEncoder.computeReward(0.8, 0.7, null, 'left', false, 0.8);
  assertApprox(result.reward, 0.0, 1e-9, 'regression no reward');
  assertApprox(result.checkpoint, 0.0, 1e-9, 'regression checkpoint');
  console.log('   ✓ Ball regression: reward=0.0 (no checkpoint paid)');

  // Test 6: Controlled-team invariant (stabilization release #7) — the reward
  // shaping is only valid for the left-controlled team; right-team target must
  // fail LOUDLY instead of silently producing meaningless rewards.
  let invariantThrew = false;
  try {
    ObservationEncoder.computeReward(-0.3, -0.35, null, 'right', false, -0.3);
  } catch (err: any) {
    invariantThrew = String(err.message).includes('GMN Reward Invariant Violation');
  }
  if (!invariantThrew) {
    throw new Error('Reward invariant: computeReward(targetTeam="right") must throw [GMN Reward Invariant Violation]');
  }
  console.log('   ✓ Controlled-team invariant enforced (right-team target throws)');

  // Test 7: Small movement below threshold
  result = ObservationEncoder.computeReward(0.5, 0.502, null, 'left', false, 0.5);
  assertApprox(result.reward, 0.0, 1e-9, 'below threshold');
  console.log('   ✓ Small movement below threshold: reward=0.0');

  // Test 8: Checkpoint capped at 0.05
  result = ObservationEncoder.computeReward(0.0, 0.2, null, 'left', false, 0.0);
  assertApprox(result.reward, 0.05, 1e-9, 'checkpoint cap');
  assertApprox(result.checkpoint, 0.05, 1e-9, 'checkpoint cap value');
  console.log('   ✓ Checkpoint reward capped at +0.05');

  // Test 9: On-target shot — ball moving toward right goal, projected Y inside goal mouth
  // Ball at x=0.8, y=0.02, velocity toward goal with slight inward curve
  result = ObservationEncoder.computeReward(
    0.75, 0.8, null, 'left', true, 0.75,
    { x: 0.8, y: 0.02, z: 0.0 },
    { x: 0.8, y: -0.01, z: 0.0 }
  );
  // deltaX = 0.05 -> checkpoint = 0.025
  // shot on target -> +0.03
  // total = 0.025 + 0.03 = 0.055
  assertApprox(result.checkpoint, 0.025, 1e-9, 'on-target checkpoint');
  assertApprox(result.reward, 0.055, 1e-9, 'on-target shot total');
  console.log('   ✓ On-target shot (projected Y inside goal): reward=+0.055');

  // Test 10: Off-target shot — ball moving toward right goal but projected Y misses goal mouth
  // Ball at x=0.8, y=0.2 (wide), velocity toward goal but Y drift stays wide
  result = ObservationEncoder.computeReward(
    0.75, 0.8, null, 'left', true, 0.75,
    { x: 0.8, y: 0.2, z: 0.0 },
    { x: 0.8, y: 0.05, z: 0.0 }
  );
  // deltaX = 0.05 -> checkpoint = 0.025
  // off-target -> +0.001
  // total = 0.025 + 0.001 = 0.026
  assertApprox(result.checkpoint, 0.025, 1e-9, 'off-target checkpoint');
  assertApprox(result.reward, 0.026, 1e-9, 'off-target shot total');
  console.log('   ✓ Off-target shot (projected Y outside goal): reward=+0.026');

  // Test 11: (removed — right-team rewards are now guarded by the
  // controlled-team invariant; see Test 6).

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
