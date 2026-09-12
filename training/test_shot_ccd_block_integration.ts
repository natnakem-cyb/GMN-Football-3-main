/**
 * GMN-Football-3 — Shot CCD / Block Integration Test
 *
 * Verifies that fast shots in flight use swept-volume (CCD) contact against
 * players, so a collinear default SHOT from the CAM in academy_3_vs_1_with_keeper
 * does not tunnel through the CB+GK and score on every attempt.
 *
 * Also verifies the outfield block rule (no 75% save roll required) and that
 * a real goal still works when the GK is off the line.
 */

import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { Vec2 } from '../src/engine/Vector';
import { ActionType, AgentAction } from '../src/types/football';

function assert(condition: boolean, msg: string): void {
  if (!condition) throw new Error(msg);
}

function runShotTrial(engine: GameEngine, camId: string, gkId: string, moveGkOffLine = false): {
  goal: boolean;
  hadBlockOrSave: boolean;
  ticks: number;
} {
  const scenario = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_3_vs_1_with_keeper')!;

  engine.setSeed(Math.floor(Math.random() * 100000));
  engine.loadScenario(scenario);

  const cam = engine.players.find((p) => p.id === camId)!;
  const gk = engine.players.find((p) => p.id === gkId)!;

  if (moveGkOffLine) {
    gk.position.y = 1.0;
    gk.targetPosition.y = 1.0;
  }

  cam.hasBall = true;
  engine.ball.ownerId = cam.id;
  engine.ball.position.x = cam.position.x;
  engine.ball.position.y = cam.position.y;
  engine.ball.position.z = 0;
  engine.ball.velocity.x = 0;
  engine.ball.velocity.y = 0;
  engine.ball.velocity.z = 0;

  const actionMap = new Map<string, AgentAction>();
  actionMap.set(cam.id, { type: ActionType.SHOT });

  let goal = false;
  let hadBlockOrSave = false;
  let ticks = 0;
  const maxTicks = 300;

  while (ticks < maxTicks) {
    const eventsBefore = engine.events.length;
    engine.step(actionMap, 1 / 60);
    ticks++;

    const newEvents = engine.events.slice(eventsBefore);
    for (const ev of newEvents) {
      if (ev.type === 'goal') goal = true;
      if (ev.type === 'shot_saved' || ev.type === 'shot_blocked') hadBlockOrSave = true;
    }

    if (goal) break;
    if (engine.status !== 'playing' && engine.status !== 'goal') break;
    if (engine.ball.ownerId && engine.players.find((p) => p.id === engine.ball.ownerId)?.team === 'right') break;
  }

  return { goal, hadBlockOrSave, ticks };
}

function main(): void {
  console.log('====================================================');
  console.log('GMN-FOOTBALL-3 — SHOT CCD / BLOCK INTEGRATION TEST');
  console.log('====================================================');

  const scenario = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_3_vs_1_with_keeper');
  if (!scenario) throw new Error('academy_3_vs_1_with_keeper not found');

  const engine = new GameEngine();
  engine.setSeed(0);
  engine.loadScenario(scenario);

  const leftPlayers = engine.players.filter((p) => p.team === 'left');
  const rightPlayers = engine.players.filter((p) => p.team === 'right');
  const camId = leftPlayers[0].id;
  const gkId = rightPlayers[0].id;

  // --- Unit test for Vec2.distPointToSegment2D ---
  console.log('   [unit] Vec2.distPointToSegment2D');
  const A = { x: 0, y: 0 };
  const B = { x: 1, y: 0 };

  // Point on segment
  let d = Vec2.distPointToSegment2D({ x: 0.5, y: 0 }, A, B);
  assert(Math.abs(d) < 1e-9, `point on segment: expected 0, got ${d}`);

  // Point past endpoint B
  d = Vec2.distPointToSegment2D({ x: 2, y: 0 }, A, B);
  assert(Math.abs(d - 1) < 1e-9, `past endpoint: expected 1, got ${d}`);

  // Perpendicular miss at midpoint
  d = Vec2.distPointToSegment2D({ x: 0.5, y: 0.2 }, A, B);
  assert(Math.abs(d - 0.2) < 1e-9, `perpendicular miss: expected 0.2, got ${d}`);

  // Zero-length segment
  d = Vec2.distPointToSegment2D({ x: 0.3, y: 0.4 }, A, A);
  assert(Math.abs(d - Math.hypot(0.3, 0.4)) < 1e-9, `zero-length segment: expected 0.5, got ${d}`);

  console.log('      ✓ distPointToSegment2D unit tests passed');

  // --- Integration trials: collinear shot with GK on line ---
  const NUM_TRIALS = 20;
  let goals = 0;
  let blocksOrSaves = 0;
  const results: { goal: boolean; hadBlockOrSave: boolean; ticks: number }[] = [];

  for (let i = 0; i < NUM_TRIALS; i++) {
    const result = runShotTrial(engine, camId, gkId, false);
    results.push(result);
    if (result.goal) goals++;
    if (result.hadBlockOrSave) blocksOrSaves++;
  }

  const goalRate = goals / NUM_TRIALS;
  console.log(`   ${NUM_TRIALS} trials collinear GK-on-line: ${goals} goals (${(goalRate * 100).toFixed(0)}%), ${blocksOrSaves} blocks/saves`);

  assert(goalRate <= 0.4, `Goal rate ${(goalRate * 100).toFixed(0)}% exceeds 40% — CCD may not be active`);
  assert(blocksOrSaves > 0, 'No blocks or saves recorded — CCD contact may not be firing');

  // --- Control trial: GK off the line, shot should still be able to score ---
  let controlGoals = 0;
  const NUM_CONTROL = 5;
  for (let i = 0; i < NUM_CONTROL; i++) {
    const result = runShotTrial(engine, camId, gkId, true);
    if (result.goal) controlGoals++;
  }

  const controlRate = controlGoals / NUM_CONTROL;
  console.log(`   ${NUM_CONTROL} control trials (GK off-line): ${controlGoals} goals (${(controlRate * 100).toFixed(0)}%)`);
  assert(controlGoals > 0, 'Control: no goals with GK off the line — scoring may be broken globally');

  console.log('\n====================================================');
  console.log('✓ ALL SHOT CCD / BLOCK CHECKS PASSED');
  console.log('====================================================');
}

try {
  main();
  console.log('\n[OK] Shot CCD/block integration test completed successfully.');
} catch (err: any) {
  console.error('✗ SHOT CCD/BLOCK INTEGRATION TEST FAILED:', err instanceof Error ? err.message : err);
  process.exit(1);
}
