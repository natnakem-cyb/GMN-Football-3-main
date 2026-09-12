/**
 * GMN-Football-3 — Rondo Defender Transition Reward Integration Test
 *
 * Drives the real GameEngine + RondoScenarioHandler directly (no HTTP bridge,
 * no hand-fed flags) to verify that defenderJustWonPossession is computed
 * from the pre-mutation snapshot of lastPossessionTeam inside onStep(), so
 * the +0.2 transition bonus fires exactly once on a genuine possession change.
 *
 * This test catches the class of bug where unit tests pass on hand-fed values
 * but the caller-side state-tracking logic is broken.
 */

import { GameEngine } from '../src/engine/GameEngine';
import { RondoScenarioHandler } from '../src/engine/scenarios/RondoScenarioHandler';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';

function assertApprox(actual: number, expected: number, tolerance = 1e-6, msg = ''): void {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`Rondo mismatch${msg ? ': ' + msg : ''}: expected ${expected}, got ${actual}`);
  }
}

function main(): void {
  console.log('====================================================');
  console.log('GMN-FOOTBALL-3 — RONDO DEFENDER TRANSITION INTEGRATION TEST');
  console.log('====================================================');

  const rondoScenario = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_rondo_4v1');
  if (!rondoScenario) {
    throw new Error('academy_rondo_4v1 scenario not found in ACADEMY_SCENARIOS');
  }

  const engine = new GameEngine();
  engine.setSeed(0);
  engine.loadScenario(rondoScenario);

  const handler = engine.getActiveScenarioHandler();
  if (!(handler instanceof RondoScenarioHandler)) {
    throw new Error(`Expected RondoScenarioHandler, got ${handler?.constructor.name}`);
  }

  const leftPlayers = engine.players.filter((p) => p.team === 'left');
  const rightPlayers = engine.players.filter((p) => p.team === 'right');
  const leftOwner = leftPlayers[0].id;
  const rightOwner = rightPlayers[0].id;

  // --- Phase 1: several ticks of left possession — defender reward must stay 0 ---
  const leftPossessionTicks = 5;
  let phase1DefenderReward = 0;
  for (let t = 0; t < leftPossessionTicks; t++) {
    engine.ball.ownerId = leftOwner;
    engine.ball.position.x = leftPlayers[0].position.x;
    engine.ball.position.y = leftPlayers[0].position.y;
    engine.ball.velocity.x = 0;
    engine.ball.velocity.y = 0;
    engine.step(new Map(), 1 / 60);
    phase1DefenderReward += handler.getLastDefenderReward();
  }
  console.log(`   Phase 1: ${leftPossessionTicks} ticks left possession, defender reward = ${phase1DefenderReward.toFixed(4)}`);
  assertApprox(phase1DefenderReward, 0, 1e-6, 'left possession should yield 0 defender reward');

  // --- Phase 2: one transition tick (left → right) — defender should get +0.2 ---
  engine.ball.ownerId = rightOwner;
  engine.ball.position.x = rightPlayers[0].position.x;
  engine.ball.position.y = rightPlayers[0].position.y;
  engine.ball.velocity.x = 0;
  engine.ball.velocity.y = 0;
  engine.step(new Map(), 1 / 60);
  const transitionReward = handler.getLastDefenderReward();
  console.log(`   Phase 2: transition tick (left→right), defender reward = ${transitionReward.toFixed(4)}`);
  if (transitionReward < 0.2) {
    throw new Error(`Transition tick defender reward ${transitionReward} is below 0.2 — transition bonus did not fire`);
  }

  // --- Phase 3: 60 more ticks of continuous right possession ---
  // Additional reward should be small (distance-closing shaping only), not ~12.0
  const continuedRightTicks = 60;
  let phase3AdditionalReward = 0;
  for (let t = 0; t < continuedRightTicks; t++) {
    engine.ball.ownerId = rightOwner;
    engine.ball.position.x = rightPlayers[0].position.x;
    engine.ball.position.y = rightPlayers[0].position.y;
    engine.ball.velocity.x = 0;
    engine.ball.velocity.y = 0;
    engine.step(new Map(), 1 / 60);
    phase3AdditionalReward += handler.getLastDefenderReward();
  }
  console.log(`   Phase 3: ${continuedRightTicks} ticks continuous right possession, additional defender reward = ${phase3AdditionalReward.toFixed(4)}`);

  // Total defender reward across all 66 ticks must equal the single transition bonus
  // plus a small distance-closing shaping term. It must NOT be ~12.0 (per-tick bug)
  // and must NOT be 0 (transition bonus must have fired).
  const totalDefenderReward = phase1DefenderReward + transitionReward + phase3AdditionalReward;
  console.log(`   Total defender reward over all ticks = ${totalDefenderReward.toFixed(4)}`);
  // Total = +0.2 transition bonus + small distance-closing shaping on transition tick
  // (ball moved from left player to right player, so distDelta > 0 on that tick).
  // Must NOT be ~12.0 (per-tick bug) and must NOT be 0 (transition must have fired).
  if (totalDefenderReward < 0.2) {
    throw new Error(`Total defender reward ${totalDefenderReward} < 0.2 — transition bonus did not fire`);
  }
  if (totalDefenderReward > 0.5) {
    throw new Error(`Total defender reward ${totalDefenderReward} > 0.5 — likely per-tick exploit (expected ~0.25)`);
  }
  assertApprox(phase3AdditionalReward, 0, 0.01, '60 ticks of continued possession should not accrue another +0.2');

  console.log('\n====================================================');
  console.log('✓ ALL RONDO DEFENDER TRANSITION INTEGRATION CHECKS PASSED');
  console.log('====================================================');
}

try {
  main();
  console.log('\n[OK] Rondo defender transition integration test completed successfully.');
} catch (err: any) {
  console.error('✗ RONDO DEFENDER TRANSITION INTEGRATION TEST FAILED:', err instanceof Error ? err.message : err);
  process.exit(1);
}
