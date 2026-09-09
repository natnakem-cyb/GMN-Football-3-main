/**
 * GMN-Football-3 — Event-Code Transmission Regression Test
 *
 * Root-cause regression for the "goals happen but event_code = 0" bug:
 * GameEngine.step() used to surface the human-readable event DESCRIPTION in
 * stepResult.info.event (e.g. "GOAL! Team Left scored! (1 - 0)"), and the three
 * bridge encoders (encodeStepBinary / encodeMultiStepBinary /
 * encodeBatchedStepBinary) mapped that straight into the binary EVENT_CODE_MAP
 * byte via an exact indexOf lookup. Description strings never equal the
 * canonical type token ('goal' | 'pass' | 'shot' | ...), so eventCode was ALWAYS
 * 0 on every tick where any event legitimately occurred.
 *
 * This test asserts:
 *   1. ENGINE contract: stepResult.info.event is the canonical TYPE token on
 *      every tick where an event occurs, and the human text is preserved in
 *      info.eventDescription.
 *   2. ENCODE contract: all three binary encoders write getEventCode(info.event)
 *      at the documented offset-16 event byte for every event type, and 0 on
 *      ticks with no events.
 *
 * Run: npx tsx training/test_event_code_transmission.ts
 */

import { GameEngine } from '../src/engine/GameEngine';
import { ActionType, AgentAction, ScenarioConfig } from '../src/types/football';
import { OBSERVATION_DIM, getEventCode } from '../src/engine/Contract';
import {
  encodeStepBinary,
  encodeMultiStepBinary,
  encodeBatchedStepBinary,
} from './bridge_server';

const ALL_EVENT_TYPES = [
  'goal',
  'shot',
  'shot_saved',
  'shot_missed',
  'pass',
  'interception',
  'tackle',
  'foul',
  'kickoff',
  'out_of_bounds',
  'scenario_complete',
  'scenario_failed',
  'offside',
  'pass_completed',
  'pass_intercepted',
] as const;

let totalTests = 0;
let passedTests = 0;

function check(cond: boolean, label: string): void {
  totalTests++;
  if (cond) {
    passedTests++;
    console.log('  [PASS] ' + label);
  } else {
    console.log('  [FAIL] ' + label);
  }
}

function makeTestScenario(): ScenarioConfig {
  return {
    id: 'test_event_code',
    name: 'Event Code Test',
    codeName: 'test_event_code',
    description: 'event code transmission regression',
    instructions: 'n/a',
    difficulty: 'Beginner',
    stage: 1,
    teamLeftPlayers: 2,
    teamRightPlayers: 0,
    hasGoalkeeperLeft: false,
    hasGoalkeeperRight: false,
    timeLimitSeconds: 3600,
    setup: {
      ball: { x: 0, y: 0, z: 0 },
      leftPlayers: [
        { role: 'ST', pos: { x: 0.25, y: 0 } },
        { role: 'ST', pos: { x: -0.25, y: 0 } },
      ],
      rightPlayers: [],
      positionJitter: 0,
    },
    objectives: [],
    terminateOnOpponentPossession: false,
    rewards: { scoring: 1.0, completion: 0 },
  };
}

const zeros = (n: number): number[] => new Array<number>(n).fill(0);
const ones = (n: number): number[] => new Array<number>(n).fill(1);

function idleMap(...players: any[]): Map<string, AgentAction> {
  const m = new Map<string, AgentAction>();
  players.forEach((p) => m.set(p.id, { type: ActionType.IDLE }));
  return m;
}

function givePossession(engine: GameEngine, player: any): void {
  engine.players.forEach((p) => (p.hasBall = false));
  player.hasBall = true;
  engine.ball.ownerId = player.id;
  engine.ball.lastOwnerId = player.id;
  engine.ball.lastOwnerTeam = player.team;
  engine.ball.position = { x: player.position.x, y: player.position.y, z: 0 };
  engine.ball.velocity = { x: 0, y: 0, z: 0 };
  engine.ball.isInAir = false;
  engine.ball.isShotInFlight = false;
  engine.ball.lastKickedBy = null;
  engine.ball.lastKickedTeam = null;
  engine.currentPassTracking = null;
}

// ---------------------------------------------------------------------------
// Part 1 — ENGINE contract: info.event carries the canonical type token
// ---------------------------------------------------------------------------
function engineContractTest(): void {
  console.log('Validating ENGINE contract: info.event is the canonical type token...');

  const engine = new GameEngine();
  const scenario = makeTestScenario();
  engine.loadScenario(scenario, 42);
  let player1 = engine.players.find((p) => p.team === 'left')!;
  let player2 = engine.players.find((p) => p.team === 'left' && p.id !== player1.id)!;
  engine.controlledPlayerId = player1.id;

  // loadScenario replaces the players array; re-fetch fresh references after
  // each load so possession staging targets the CURRENT player objects.
  const refetch = () => {
    player1 = engine.players.find((p) => p.team === 'left')!;
    player2 = engine.players.find((p) => p.team === 'left' && p.id !== player1.id)!;
  };

  // 1. Tick with no events -> undefined event
  {
    const res = engine.step(idleMap(player1, player2), 1 / 60);
    check(res.info.event === undefined, 'no-event tick: info.event undefined (got ' + JSON.stringify(res.info.event) + ')');
    check(res.info.eventDescription === undefined, 'no-event tick: info.eventDescription undefined');
  }

  // 2. Goal tick
  {
    engine.loadScenario(scenario, 42);
    refetch();
    engine.controlledPlayerId = player1.id;
    engine.ball.position = { x: 1.05, y: 0, z: 0 };
    engine.ball.velocity = { x: 0, y: 0, z: 0 };
    engine.ball.ownerId = null;
    engine.ball.isShotInFlight = false;
    const res = engine.step(idleMap(player1, player2), 1 / 60);
    check(res.info.event === 'goal', "goal tick: info.event='goal' (got " + JSON.stringify(res.info.event) + ')');
    check(typeof res.info.eventDescription === 'string' && res.info.eventDescription.includes('GOAL'),
      "goal tick: eventDescription='" + res.info.eventDescription + "'");
    check(getEventCode(res.info.event) === 1, 'goal tick: event code 1 (got ' + getEventCode(res.info.event) + ')');
    check(res.info.score.left === 1, 'goal tick: score.left incremented (got ' + res.info.score.left + ')');
  }

  // 3. Shot tick
  {
    engine.loadScenario(scenario, 42);
    refetch();
    engine.controlledPlayerId = player1.id;
    givePossession(engine, player1);
    const actions = idleMap(player1, player2);
    actions.set(player1.id, { type: ActionType.SHOT, direction: { x: 1, y: 0 }, power: 0.5 });
    const res = engine.step(actions, 1 / 60);
    check(res.info.event === 'shot', "shot tick: info.event='shot' (got " + JSON.stringify(res.info.event) + ')');
    check(engine.stats.shots.left === 1, 'shot tick: stats.shots.left === 1 (got ' + engine.stats.shots.left + ')');
    check(getEventCode(res.info.event) === 2, 'shot tick: event code 2 (got ' + getEventCode(res.info.event) + ')');
  }

  // 4. Pass-initiation tick
  {
    engine.loadScenario(scenario, 42);
    refetch();
    engine.controlledPlayerId = player1.id;
    givePossession(engine, player1);
    const actions = idleMap(player1, player2);
    // Backward short pass (direction -x from x=0.25) so no offside and no
    // same-tick completion against the receiver at x=-0.25.
    actions.set(player1.id, { type: ActionType.SHORT_PASS, direction: { x: -1, y: 0 }, power: 0.7 });
    const res = engine.step(actions, 1 / 60);
    check(res.info.event === 'pass', "pass tick: info.event='pass' (got " + JSON.stringify(res.info.event) + ')');
    check(engine.stats.passes.left === 1, 'pass tick: stats.passes.left === 1 (got ' + engine.stats.passes.left + ')');
    check(getEventCode(res.info.event) === 5, 'pass tick: event code 5 (got ' + getEventCode(res.info.event) + ')');
  }

  // 5. Pass-completion tick (receiver gains possession on the NEXT tick)
  {
    engine.loadScenario(scenario, 42);
    refetch();
    engine.controlledPlayerId = player1.id;
    givePossession(engine, player1);
    const actions = idleMap(player1, player2);
    actions.set(player1.id, { type: ActionType.SHORT_PASS, direction: { x: -1, y: 0 }, power: 0.7 });
    engine.step(actions, 1 / 60);
    // currentPassTracking is now set (passerId=player1). Place the ball exactly
    // on player2. The next tick's checkBallPossession gives player2 the ball ->
    // pass_completed + completedPasses increment.
    engine.players.forEach((p) => (p.hasBall = false));
    engine.ball.position = { x: player2.position.x, y: player2.position.y, z: 0 };
    engine.ball.velocity = { x: 0, y: 0, z: 0 };
    engine.ball.ownerId = null;
    const res = engine.step(idleMap(player1, player2), 1 / 60);
    check(res.info.event === 'pass_completed',
      "pass-completion tick: info.event='pass_completed' (got " + JSON.stringify(res.info.event) + ')');
    check(engine.stats.completedPasses.left === 1,
      'pass-completion tick: stats.completedPasses.left === 1 (got ' + engine.stats.completedPasses.left + ')');
    check(getEventCode(res.info.event) === 14,
      'pass-completion tick: event code 14 (got ' + getEventCode(res.info.event) + ')');
  }
}
// ---------------------------------------------------------------------------
// Part 2 — ENCODE contract: all three binary encoders write the event byte
// ---------------------------------------------------------------------------
function singleResult(eventType?: string): any {
  return {
    reward: 1.0,
    terminated: false,
    truncated: false,
    info: {
      score: { left: eventType === 'goal' ? 1 : 0, right: 0 },
      event: eventType,
      checkpointReward: 0,
      ballDistanceToGoal: 0.1,
      ground_truth: { current_ball_owner: { agent_id: 'left_1' } },
    },
    observation: zeros(OBSERVATION_DIM),
    action_mask: ones(19),
  };
}

function multiResult(eventType?: string): any {
  return {
    observations: [zeros(OBSERVATION_DIM), zeros(OBSERVATION_DIM)],
    action_masks: [ones(19), ones(19)],
    controllableIds: ['left_1', 'left_2'],
    reward: eventType === 'goal' ? 1.0 : 0.0,
    terminated: false,
    truncated: false,
    info: {
      score: { left: eventType === 'goal' ? 1 : 0, right: 0 },
      event: eventType,
      checkpointReward: 0,
      ballDistanceToGoal: 0.1,
      ground_truth: { current_ball_owner: { agent_id: 'left_1' } },
    },
  };
}

function encodeContractTest(): void {
  console.log('Validating ENCODE contract: offset-16 event byte across all 3 encoders...');

  for (const eventType of ALL_EVENT_TYPES) {
    const expected = getEventCode(eventType);
    const single = encodeStepBinary(singleResult(eventType), 0);
    const multi = encodeMultiStepBinary(multiResult(eventType));
    const batched = encodeBatchedStepBinary([multiResult(eventType)]);
    check(single.readUInt8(16) === expected,
      "encodeStepBinary('" + eventType + "') -> " + single.readUInt8(16) + ' (expected ' + expected + ')');
    check(multi.readUInt8(16) === expected,
      "encodeMultiStepBinary('" + eventType + "') -> " + multi.readUInt8(16) + ' (expected ' + expected + ')');
    check(batched.readUInt8(2 + 16) === expected,
      "encodeBatchedStepBinary('" + eventType + "') -> " + batched.readUInt8(2 + 16) + ' (expected ' + expected + ')');
  }

  // No event -> code 0 on all encoders
  const single = encodeStepBinary(singleResult(undefined), 0);
  const multi = encodeMultiStepBinary(multiResult(undefined));
  const batched = encodeBatchedStepBinary([multiResult(undefined)]);
  check(single.readUInt8(16) === 0, 'encodeStepBinary(no event) -> 0');
  check(multi.readUInt8(16) === 0, 'encodeMultiStepBinary(no event) -> 0');
  check(batched.readUInt8(2 + 16) === 0, 'encodeBatchedStepBinary(no event) -> 0');

  // Rondo variant: event byte at offset 16 stays intact.
  const rondo = encodeStepBinary(singleResult('goal'), 0, true, 0.25);
  check(rondo.readUInt8(16) === 1, 'encodeStepBinary rondo goal -> offset 16 = 1 (got ' + rondo.readUInt8(16) + ')');
}

function main(): void {
  console.log('====================================================');
  console.log('GMN-FOOTBALL-3 — EVENT-CODE TRANSMISSION REGRESSION');
  console.log('====================================================');

  try {
    engineContractTest();
    encodeContractTest();
  } catch (err: any) {
    console.error('  [FAIL] UNHANDLED EXCEPTION: ' + err.message);
    if (err.stack) console.error(err.stack);
  }

  console.log('====================================================');
  console.log('  RESULT: ' + passedTests + '/' + totalTests + ' checks passed');
  console.log('====================================================');

  if (passedTests !== totalTests) {
    process.exit(1);
  }
}

main();