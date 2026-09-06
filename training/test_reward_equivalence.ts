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

  // Test 4: Shot bonus
  result = ObservationEncoder.computeReward(0.3, 0.3, null, 'left', true, 0.3);
  assertApprox(result.reward, 0.005, 1e-9, 'shot bonus');
  console.log('   ✓ Shot-attempt bonus: reward=+0.005');

  // Test 5: Monotonic checkpoint — should not pay for regressing ball position
  result = ObservationEncoder.computeReward(0.8, 0.7, null, 'left', false, 0.8);
  assertApprox(result.reward, 0.0, 1e-9, 'regression no reward');
  assertApprox(result.checkpoint, 0.0, 1e-9, 'regression checkpoint');
  console.log('   ✓ Ball regression: reward=0.0 (no checkpoint paid)');

  // Test 6: Right team checkpoint (negative X progress)
  result = ObservationEncoder.computeReward(-0.3, -0.35, null, 'right', false, -0.3);
  assertApprox(result.reward, 0.025, 1e-9, 'right team checkpoint');
  assertApprox(result.checkpoint, 0.025, 1e-9, 'right team checkpoint value');
  console.log('   ✓ Right-team checkpoint advance: reward=+0.025');

  // Test 7: Small movement below threshold
  result = ObservationEncoder.computeReward(0.5, 0.502, null, 'left', false, 0.5);
  assertApprox(result.reward, 0.0, 1e-9, 'below threshold');
  console.log('   ✓ Small movement below threshold: reward=0.0');

  // Test 8: Checkpoint capped at 0.05
  result = ObservationEncoder.computeReward(0.0, 0.2, null, 'left', false, 0.0);
  assertApprox(result.reward, 0.05, 1e-9, 'checkpoint cap');
  assertApprox(result.checkpoint, 0.05, 1e-9, 'checkpoint cap value');
  console.log('   ✓ Checkpoint reward capped at +0.05');

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
