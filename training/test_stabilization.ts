/**
 * GMN-Football-3 — Stabilization Release Regression Suite (P0/P1)
 *
 * Covers:
 *   - P0 #2/#3: goal-mouth geometry (crossbar), goal detection, shot consistency
 *   - P0 #4: corner-kick event loop prevention + restart position validity
 *   - P1 #14: role orientation symmetry
 *   - P1 #15/#17: offside + shot_missed event contract codes
 *   - Contract: 127-dim observation, Discrete(19), event code stability
 *   - Determinism: identical seed + action sequence => identical trajectory
 */

import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { PITCH, isGoalMouthPoint } from '../src/engine/Rules';
import { inferPlayerRole, getEventCode, OBSERVATION_DIM, ACTION_SPACE_SIZE, validateObservationVector } from '../src/engine/Contract';
import { mapDiscreteAction } from '../src/engine/ActionMapping';
import { ActionType, AgentAction } from '../src/types/football';
import { ObservationEncoder } from '../src/engine/ObservationEncoder';

let failures = 0;
function check(cond: boolean, label: string): void {
  if (cond) {
    console.log(`   ✓ ${label}`);
  } else {
    failures++;
    console.error(`   ✗ ${label}`);
  }
}

function approx(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) <= tol;
}

// ---------------------------------------------------------------- P0 #2/#3
function testGoalMouthGeometry(): void {
  console.log('\n[P0] Goal-mouth geometry (isGoalMouthPoint)');
  check(isGoalMouthPoint(0, 0) === true, 'ground-level center shot inside mouth');
  check(isGoalMouthPoint(PITCH.goalMinY, 0.02) === true, 'y == goalMinY boundary inclusive');
  check(isGoalMouthPoint(PITCH.goalMaxY, 0.02) === true, 'y == goalMaxY boundary inclusive');
  check(isGoalMouthPoint(PITCH.goalMinY - 0.001) === false, 'y just outside goal mouth rejected');
  check(isGoalMouthPoint(PITCH.goalMaxY + 0.001) === false, 'y just outside goal mouth (top) rejected');
  // Crossbar boundary semantics: z <= goalHeight counts as goal (inclusive, tol 1e-9).
  check(isGoalMouthPoint(0, PITCH.goalHeight - 0.001) === true, 'z clearly below crossbar -> goal');
  check(isGoalMouthPoint(0, PITCH.goalHeight) === true, 'z == goalHeight boundary -> goal (inclusive)');
  check(isGoalMouthPoint(0, PITCH.goalHeight + 0.001) === false, 'z clearly above crossbar -> NOT a goal');
  check(isGoalMouthPoint(0, PITCH.goalHeight + 5e-10) === true, 'z within GOAL_MOUTH_EPSILON (1e-9) tolerance -> goal');
}

function testEngineGoalDetection(): void {
  console.log('\n[P0] Engine goal detection (crossbar enforced)');
  const sc = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_empty_goal')!;
  const mk = (x: number, y: number, z: number, lastOwner: 'left' | 'right') => {
    const e = new GameEngine();
    e.loadScenario(sc, 7);
    e.ball.position = { x, y, z };
    e.ball.velocity = { x: 0, y: 0, z: 0 };
    e.ball.ownerId = null;
    e.ball.lastOwnerTeam = lastOwner;
    e.step(new Map(), 1 / 60);
    return e;
  };

  const below = mk(PITCH.maxX, 0.0, 0.02, 'left');
  check(below.score.left === 1, 'ball below crossbar crossing right goal line counts as goal');
  const above = mk(PITCH.maxX, 0.0, 0.3, 'left');
  check(above.score.left === 0, 'ball ABOVE crossbar crossing right goal line is NOT a goal');
  const wide = mk(PITCH.maxX, 0.2, 0.0, 'left');
  check(wide.score.left === 0, 'ball outside goal mouth (y wide) is NOT a goal');
  const mirrored = mk(PITCH.minX, 0.0, 0.3, 'right');
  check(mirrored.score.right === 0, 'mirrored goal line: ball above crossbar is NOT a goal');
}

function testShotOnTargetConsistency(): void {
  console.log('\n[P0] Shot on-target consistency (shared goal geometry)');
  const sc = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_empty_goal')!;

  const shoot = (y: number, dirX: number, dirY: number) => {
    const e = new GameEngine();
    e.loadScenario(sc, 7);
    const p = e.players.find((pl) => pl.team === 'left' && !pl.isGoalkeeper)!;
    e.players.forEach((pl) => (pl.hasBall = false));
    p.position = { x: 0.8, y };
    p.hasBall = true;
    e.ball.ownerId = p.id;
    const action: AgentAction = { type: ActionType.SHOT, power: 0.95, direction: { x: dirX, y: dirY } };
    e.step(new Map([[p.id, action]]), 1 / 60);
    return e;
  };

  const onTarget = shoot(0.0, 1.0, 0.0);
  check(
    onTarget.stats.shots.left === 1 && onTarget.stats.shotsOnTarget.left === 1,
    'straight shot at goal center is counted on-target'
  );
  const wide = shoot(0.3, 0.7071067811865476, 0.7071067811865476);
  check(
    wide.stats.shots.left === 1 && wide.stats.shotsOnTarget.left === 0,
    'shot projected wide of the goal mouth is NOT on-target'
  );

  // Reward consistency: an on-target shot earns the ON_TARGET bonus.
  const r = ObservationEncoder.computeReward(
    0.75, 0.8, null, 'left', true, 0.75,
    { x: 0.8, y: 0.02, z: 0.0 },
    { x: 0.8, y: -0.01, z: 0.0 }
  );
  check(approx(r.reward, 0.055), 'shot-quality reward: on-target trajectory earns +0.03 bonus');

  // A trajectory that would cross ABOVE the crossbar must NOT earn the bonus.
  // (Ball starts high at z=0.2 moving flat and fast (vx=2.0); it reaches the
  // goal line in ~6 ticks while still at z≈0.14, far above the 0.05 crossbar.)
  const high = ObservationEncoder.computeReward(
    0.75, 0.8, null, 'left', true, 0.75,
    { x: 0.8, y: 0.0, z: 0.2 },
    { x: 2.0, y: 0.0, z: 0.0 }
  );
  check(approx(high.reward, 0.025 + 0.001), 'shot-quality reward: trajectory above crossbar earns only off-target bonus');
}

// ---------------------------------------------------------------- P0 #4
function testCornerEventLoop(): void {
  console.log('\n[P0] Corner-kick event loop prevention');
  const sc = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_empty_goal')!;
  const e = new GameEngine();
  e.loadScenario(sc, 7);
  // Force a corner: ball beyond the right goal line, outside the mouth,
  // last touched by the defending (right) team.
  e.ball.position = { x: PITCH.maxX + 0.01, y: 0.2, z: 0.0 };
  e.ball.velocity = { x: 0, y: 0, z: 0 };
  e.ball.ownerId = null;
  e.ball.lastOwnerTeam = 'right';
  e.ball.isShotInFlight = false;

  e.step(new Map(), 1 / 60);
  const cornerEventsAfterFirstTick = e.events.filter((ev) => ev.description.includes('Corner Kick awarded')).length;
  check(cornerEventsAfterFirstTick === 1, 'exactly one corner event awarded on the boundary crossing');

  for (let i = 0; i < 20; i++) {
    e.step(new Map(), 1 / 60);
  }
  const cornerEventsAfter20Ticks = e.events.filter((ev) => ev.description.includes('Corner Kick awarded')).length;
  check(cornerEventsAfter20Ticks === 1, `corner event count remains 1 after 20 idle ticks (got ${cornerEventsAfter20Ticks})`);

  const b = e.ball.position;
  const positionValid =
    b.x > PITCH.minX && b.x < PITCH.maxX && b.y >= PITCH.minY && b.y <= PITCH.maxY && b.z === 0;
  check(positionValid, 'post-corner ball position is strictly inside the pitch bounds');
}

// ---------------------------------------------------------------- P1 #14
function testRoleOrientationSymmetry(): void {
  console.log('\n[P1] Role orientation symmetry (fallback inference)');
  const mk = (team: 'left' | 'right', x: number, y: number) => ({ team, position: { x, y } });
  // Spec mapping: left LB <-> right RB, left RB <-> right LB, left LW <-> right RW.
  check(inferPlayerRole(mk('left', -0.85, -0.3)) === 'LB', 'left deep + (-y) -> LB');
  check(inferPlayerRole(mk('right', 0.85, -0.3)) === 'RB', 'mirrored right deep + (-y) -> RB (left LB ↔ right RB)');
  check(inferPlayerRole(mk('left', -0.85, 0.3)) === 'RB', 'left deep + (+y) -> RB');
  check(inferPlayerRole(mk('right', 0.85, 0.3)) === 'LB', 'mirrored right deep + (+y) -> LB (left RB ↔ right LB)');
  check(inferPlayerRole(mk('left', 0.85, -0.3)) === 'LW', 'left attacker + (-y) -> LW');
  check(inferPlayerRole(mk('right', -0.85, -0.3)) === 'RW', 'mirrored right attacker + (-y) -> RW (left LW ↔ right RW)');
  check(inferPlayerRole(mk('left', 0.85, 0.3)) === 'RW', 'left attacker + (+y) -> RW');
  check(inferPlayerRole(mk('right', -0.85, 0.3)) === 'LW', 'mirrored right attacker + (+y) -> LW (left RW ↔ right LW)');
  check(inferPlayerRole(mk('left', -0.85, 0)) === 'CB', 'central defender -> CB (both teams)');
  check(inferPlayerRole(mk('right', 0.85, 0)) === 'CB', 'central defender -> CB (right team)');
  // Explicit roles are never overridden:
  check(inferPlayerRole({ role: 'ST', team: 'left', position: { x: -0.9, y: 0 } }) === 'ST', 'explicit role always wins');
  check(inferPlayerRole({ isGoalkeeper: true, team: 'right', position: { x: 0.95, y: 0 } }) === 'GK', 'goalkeeper fallback -> GK');
}

// ------------------------------------------------- P1 #15 / #17 + contract
function testEventContract(): void {
  console.log('\n[P1] Event contract codes (additive stability)');
  check(getEventCode('goal') === 1, 'code 1 = goal (unchanged)');
  check(getEventCode('shot') === 2, 'code 2 = shot (unchanged)');
  check(getEventCode('shot_saved') === 3, 'code 3 = shot_saved (unchanged)');
  check(getEventCode('shot_missed') === 4, 'code 4 = shot_missed (unchanged)');
  check(getEventCode('pass') === 5, 'code 5 = pass (unchanged)');
  check(getEventCode('interception') === 6, 'code 6 = interception (unchanged)');
  check(getEventCode('tackle') === 7, 'code 7 = tackle (unchanged)');
  check(getEventCode('foul') === 8, 'code 8 = foul (unchanged)');
  check(getEventCode('kickoff') === 9, 'code 9 = kickoff (unchanged)');
  check(getEventCode('out_of_bounds') === 10, 'code 10 = out_of_bounds (unchanged)');
  check(getEventCode('scenario_complete') === 11, 'code 11 = scenario_complete (unchanged)');
  check(getEventCode('scenario_failed') === 12, 'code 12 = scenario_failed (unchanged)');
  check(getEventCode('offside') === 13, 'code 13 = offside (ADDITIVE, appended)');
  check(getEventCode(undefined) === 0 && getEventCode('nonexistent') === 0, 'unknown/absent events map to 0');
}

function testContractDims(): void {
  console.log('\n[Contract] observation / action space');
  const sc = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_empty_goal')!;
  const e = new GameEngine();
  e.loadScenario(sc, 42);
  const obs = e.getObservation();
  check(obs.rawVector.length === OBSERVATION_DIM && OBSERVATION_DIM === 127, 'observation dimension = 127');
  const { valid } = validateObservationVector(obs.rawVector);
  check(valid, 'observation passes 127-dim schema validation (role one-hot populated)');
  check(ACTION_SPACE_SIZE === 19, 'action space size = 19');
}

// ---------------------------------------------------------------- #22
function testDeterminism(): void {
  console.log('\n[Determinism] identical seed + action sequence');
  const sc = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_empty_goal')!;
  const ACTIONS: number[] = [5, 5, 5, 13, 0, 11, 0, 5, 5, 12, 0, 0, 2, 0, 8, 0, 11, 0];

  const run = () => {
    const e = new GameEngine();
    e.loadScenario(sc, 424242);
    const trace: number[][] = [];
    for (let t = 0; t < ACTIONS.length; t++) {
      const leftIds = e.players.filter((p) => p.team === 'left').map((p) => p.id);
      const map = new Map<string, AgentAction>();
      map.set(leftIds[0], mapDiscreteAction(ACTIONS[t]));
      e.step(map, 1 / 60);
      trace.push(e.getObservation().rawVector.slice());
    }
    return { trace, score: { ...e.score }, events: e.events.map((ev) => `${ev.type}:${ev.timeSeconds}`) };
  };

  const a = run();
  const b = run();
  let identical = a.score.left === b.score.left && a.score.right === b.score.right;
  identical = identical && JSON.stringify(a.events) === JSON.stringify(b.events);
  for (let t = 0; t < a.trace.length && identical; t++) {
    for (let i = 0; i < OBSERVATION_DIM; i++) {
      if (a.trace[t][i] !== b.trace[t][i]) {
        identical = false;
        break;
      }
    }
  }
  check(identical, 'same seed + same action sequence => bitwise-identical observations, score and events');
}

function main(): void {
  console.log('====================================================');
  console.log('GMN-FOOTBALL-3 — STABILIZATION REGRESSION SUITE');
  console.log('====================================================');
  testGoalMouthGeometry();
  testEngineGoalDetection();
  testShotOnTargetConsistency();
  testCornerEventLoop();
  testRoleOrientationSymmetry();
  testEventContract();
  testContractDims();
  testDeterminism();
  console.log('\n====================================================');
  if (failures === 0) {
    console.log('✓ ALL STABILIZATION REGRESSION CHECKS PASSED');
  } else {
    console.error(`✗ ${failures} STABILIZATION CHECK(S) FAILED`);
    process.exit(1);
  }
  console.log('====================================================');
}

main();
