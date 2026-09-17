import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ActionType, AgentAction, Player } from '../src/types/football';
import { RuleBasedAgent } from '../src/agents/RuleBasedAgent';
import { mapDiscreteAction } from '../src/engine/ActionMapping';
import { Vec2 } from '../src/engine/Vector';
import { PITCH } from '../src/engine/Rules';
import { SeededRNG } from '../src/engine/SeededRNG';
import { ObservationEncoder } from '../src/engine/ObservationEncoder';
import * as fs from 'fs';

// ---------------------------------------------------------------------------
// Experiment D: Whole-Registry Scenario Qualification (Continuation)
// Primary forensic case: academy_3_vs_1_with_keeper
// ---------------------------------------------------------------------------

export interface ScenarioCard {
  scenarioId: string;
  stage: number;
  name: string;
  difficulty: string;
  teamSize: string;
  timeLimitSeconds: number;
  objectives: string[];
  taskType?: string;
  expectedPassCount?: number;
  expectedGoalCount?: number;
  legalActions: string[];
  maskNotes: string;
  knownFailureModes: string[];
}

export interface ProbeResult {
  scenarioId: string;
  probeName: string;
  episodes: number;
  goals: number;
  passActions: number;       // discrete pass actions taken by left team
  passCompleted: number;     // engine events type === 'pass_completed' && team === 'left'
  shotActions: number;       // discrete shot actions taken by left team
  shotEvents: number;        // engine events type === 'shot' && team === 'left'
  turnovers: number;
  timeouts: number;
  crashes: number;
  avgSteps: number;
  avgReward: number;
  notes: string;
}

const PRIMARY_SCENARIO = 'academy_3_vs_1_with_keeper';
const ACADEMY_DRILLS = ACADEMY_SCENARIOS.filter((s) => s.id !== '5_vs_5' && s.id !== '11_vs_11');
const DETERMINISTIC_SEED = 12345;
const PROBE_EPISODES = 100;

// Exact event type strings from src/types/football.ts MatchEvent['type']
const EVENT_PASS_COMPLETED = 'pass_completed';
const EVENT_SHOT = 'shot';
const EVENT_GOAL = 'goal';
const EVENT_SHOT_SAVED = 'shot_saved';
const EVENT_SHOT_BLOCKED = 'shot_blocked';
const EVENT_SHOT_MISSED = 'shot_missed';

// Action indices from src/engine/ActionMapping.ts
const ACTION_SHORT_PASS = 11;
const ACTION_SHOT = 12;
const ACTION_TACKLE = 16;

const SHOT_EVENT_TYPES = new Set([EVENT_SHOT, EVENT_SHOT_SAVED, EVENT_SHOT_BLOCKED, EVENT_SHOT_MISSED]);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function isPassAction(action: AgentAction): boolean {
  return action.type === ActionType.SHORT_PASS || action.type === ActionType.LONG_PASS || action.type === ActionType.HIGH_PASS;
}

function isShotAction(action: AgentAction): boolean {
  return action.type === ActionType.SHOT;
}

function leftPlayers(engine: GameEngine) {
  return engine.players.filter((p) => p.team === 'left');
}

// ---------------------------------------------------------------------------
// Step 1: Scenario Cards
// ---------------------------------------------------------------------------
export function generateScenarioCards(): ScenarioCard[] {
  const cards: ScenarioCard[] = [];

  for (const sc of ACADEMY_SCENARIOS) {
    const legalActions = [
      'IDLE', 'MOVE (8 dirs)', 'SPRINT', 'RELEASE_SPRINT',
      'SHORT_PASS', 'LONG_PASS', 'HIGH_PASS', 'SHOT',
      'TACKLE', 'DRIBBLE', 'RELEASE_DRIBBLE', 'RELEASE_DIRECTION',
    ];

    const maskNotes = [
      'Movement + IDLE always valid.',
      'PASS/SHOT require possession.',
      'TACKLE requires NO possession.',
      'DRIBBLE is off-ball valid (sticky direction).',
    ].join(' ');

    const knownFailureModes: string[] = [];
    if (sc.id === PRIMARY_SCENARIO) {
      knownFailureModes.push(
        'Opponent CB can intercept if pass lane is straight.',
        'Aggressive keeper rush can force early shot under pressure.',
        'Turnover on opponent possession ends episode.'
      );
    } else if (sc.id === 'academy_empty_goal') {
      knownFailureModes.push('Ball out of bounds if dribbled too far wide.');
    } else if (sc.id === 'academy_run_to_score') {
      knownFailureModes.push('Chasing CB tackle can dispossess before shot.');
    } else if (sc.id.startsWith('academy_3_vs_1')) {
      knownFailureModes.push('Dual/triple CB block can crowd passing lanes.');
    } else if (sc.id === 'academy_rondo_4v1') {
      knownFailureModes.push('Lone defender interception if pass is slow.');
    }

    cards.push({
      scenarioId: sc.id,
      stage: sc.stage,
      name: sc.name,
      difficulty: sc.difficulty,
      teamSize: `${sc.teamLeftPlayers}v${sc.teamRightPlayers}`,
      timeLimitSeconds: sc.timeLimitSeconds,
      objectives: sc.objectives.map((o) => o.text),
      taskType: sc.taskSpec?.taskType,
      expectedPassCount: sc.taskSpec?.targetPassesCount,
      expectedGoalCount: sc.taskSpec?.targetGoalsCount,
      legalActions,
      maskNotes,
      knownFailureModes,
    });
  }

  return cards;
}

function printScenarioCards(): void {
  const cards = generateScenarioCards();
  console.log('\n==================================================');
  console.log('STEP 1: SCENARIO CARDS (WHOLE REGISTRY)');
  console.log('==================================================\n');
  for (const card of cards) {
    console.log(`[${card.scenarioId}] ${card.name}`);
    console.log(`  Stage: ${card.stage} | Difficulty: ${card.difficulty} | Size: ${card.teamSize}`);
    console.log(`  Time Limit: ${card.timeLimitSeconds}s`);
    console.log(`  Objectives: ${card.objectives.join('; ')}`);
    if (card.taskType) console.log(`  Task Type: ${card.taskType}`);
    if (card.expectedPassCount) console.log(`  Target Passes: ${card.expectedPassCount}`);
    if (card.expectedGoalCount) console.log(`  Target Goals: ${card.expectedGoalCount}`);
    console.log(`  Legal Actions: ${card.legalActions.join(', ')}`);
    console.log(`  Mask Notes: ${card.maskNotes}`);
    console.log(`  Known Failure Modes: ${card.knownFailureModes.join('; ')}`);
    console.log('');
  }
}

// ---------------------------------------------------------------------------
// Step 2: Deterministic Scripted Feasibility (Primary Case)
// ---------------------------------------------------------------------------
class DeterministicFeasibilityController {
  private controlledId: string | null = null;

  reset(engine: GameEngine): void {
    this.controlledId = engine.controlledPlayerId;
  }

  decide(engine: GameEngine): Map<string, AgentAction> {
    const actionMap = new Map<string, AgentAction>();
    const controlled = engine.players.find((p) => p.id === this.controlledId);
    if (!controlled) return actionMap;

    // Non-controlled left players: simple support run
    for (const p of engine.players) {
      if (p.team === 'left' && p.id !== this.controlledId) {
        actionMap.set(p.id, this.supportRun(p));
      }
    }

    // Right team: use rule-based AI with fixed seed for determinism
    const rightAgent = new RuleBasedAgent('right_scripted', 'Right AI', 'medium');
    for (const p of engine.players) {
      if (p.team === 'right') {
        const teammates = engine.players.filter((pl) => pl.team === 'right');
        const opponents = engine.players.filter((pl) => pl.team === 'left');
        const ctx = {
          player: p,
          teammates,
          opponents,
          ball: engine.ball,
          allPlayers: engine.players,
          teamSide: 'right' as const,
          controlledPlayerId: p.id,
          matchTime: engine.matchTimeSeconds,
        };
        actionMap.set(p.id, rightAgent.decide(ctx));
      }
    }

    // Controlled player: hardcoded triangle logic
    actionMap.set(controlled.id, this.controlledLogic(controlled, engine));
    return actionMap;
  }

  private supportRun(player: Player): AgentAction {
    const forwardX = player.team === 'left' ? 0.15 : -0.15;
    const targetX = Math.max(-0.9, Math.min(0.9, player.position.x + forwardX));
    const targetY = Math.max(PITCH.minY + 0.05, Math.min(PITCH.maxY - 0.05, player.position.y * 0.8));
    const dir = Vec2.normalize({ x: targetX - player.position.x, y: targetY - player.position.y });
    return { type: ActionType.MOVE, direction: dir };
  }

  private controlledLogic(player: Player, engine: GameEngine): AgentAction {
    const ball = engine.ball;
    const hasPossession = player.hasBall || ball.ownerId === player.id;

    if (hasPossession) {
      const candidates = engine.players.filter(
        (p) => p.team === 'left' && p.id !== player.id && !p.isGoalkeeper
      );
      let bestTarget: Player | null = null;
      let bestScore = -Infinity;
      for (const mate of candidates) {
        const dist = Vec2.distance(player.position, mate.position);
        const forwardProgress = mate.position.x - player.position.x;
        const score = forwardProgress * 2.0 - dist * 0.5;
        if (score > bestScore && dist > 0.05 && dist < 0.8) {
          bestScore = score;
          bestTarget = mate;
        }
      }

      const distToGoal = Vec2.distance(player.position, { x: 1.0, y: 0 });
      if (distToGoal < 0.35 || !bestTarget) {
        const goalDir = Vec2.normalize({ x: 1.0 - player.position.x, y: 0 - player.position.y });
        return { type: ActionType.SHOT, direction: goalDir, power: 0.9 };
      }

      const passDir = clampPassDirection(
        Vec2.normalize({
          x: bestTarget.position.x - player.position.x,
          y: bestTarget.position.y - player.position.y,
        })
      );
      const passType = Vec2.distance(player.position, bestTarget.position) > 0.45
        ? ActionType.HIGH_PASS
        : ActionType.SHORT_PASS;
      return { type: passType, direction: passDir, power: 0.8, targetPlayerId: bestTarget.id };
    } else {
      const ballPos = { x: ball.position.x, y: ball.position.y };
      const dist = Vec2.distance(player.position, ballPos);
      if (dist > 0.05) {
        const dir = Vec2.normalize({ x: ballPos.x - player.position.x, y: ballPos.y - player.position.y });
        return { type: ActionType.SPRINT, direction: dir };
      }
      if (ball.ownerId && engine.players.find((p) => p.id === ball.ownerId)?.team === 'right') {
        return { type: ActionType.TACKLE };
      }
      return { type: ActionType.IDLE };
    }
  }
}

function runDeterministicFeasibilityProbe(): ProbeResult[] {
  console.log('\n==================================================');
  console.log(`STEP 2: DETERMINISTIC FEASIBILITY PROBE — ${PRIMARY_SCENARIO}`);
  console.log('==================================================\n');

  const scenario = ACADEMY_SCENARIOS.find((s) => s.id === PRIMARY_SCENARIO)!;
  const controller = new DeterministicFeasibilityController();
  let goals = 0;
  let totalPassActions = 0;
  let totalPassCompleted = 0;
  let totalShotActions = 0;
  let totalShotEvents = 0;
  let totalTurnovers = 0;
  let totalTimeouts = 0;
  let totalCrashes = 0;
  let totalSteps = 0;
  let totalReward = 0;

  for (let ep = 0; ep < PROBE_EPISODES; ep++) {
    const engine = new GameEngine();
    engine.loadScenario(scenario, DETERMINISTIC_SEED + ep);
    controller.reset(engine);

    let epPassActions = 0;
    let epPassCompleted = 0;
    let epShotActions = 0;
    let epShotEvents = 0;
    let epReward = 0;
    let done = false;
    let steps = 0;
    const maxSteps = scenario.timeLimitSeconds * 60 + 120;

    while (!done && steps < maxSteps) {
      steps++;
      const actionMap = controller.decide(engine);
      const res = engine.step(actionMap, 1 / 60);
      epReward += res.reward;

      // Split metrics: actions vs completed events
      for (const player of leftPlayers(engine)) {
        const action = actionMap.get(player.id);
        if (action && isPassAction(action)) epPassActions++;
        if (action && isShotAction(action)) epShotActions++;
      }

      const newEvents = engine.events;
      for (const ev of newEvents) {
        if (ev.type === EVENT_PASS_COMPLETED && ev.team === 'left') epPassCompleted++;
        if (ev.type === EVENT_SHOT && ev.team === 'left') epShotEvents++;
      }

      if (res.terminated) {
        done = true;
        if (engine.score.left > 0) {
          goals++;
        } else if (engine.ball.ownerId && engine.players.find((p) => p.id === engine.ball.ownerId)?.team === 'right') {
          totalTurnovers++;
        }
      }
      if (res.truncated) {
        done = true;
        totalTimeouts++;
      }
    }

    totalPassActions += epPassActions;
    totalPassCompleted += epPassCompleted;
    totalShotActions += epShotActions;
    totalShotEvents += epShotEvents;
    totalSteps += steps;
    totalReward += epReward;
  }

  const results: ProbeResult[] = [{
    scenarioId: PRIMARY_SCENARIO,
    probeName: 'DeterministicScriptedFeasibility',
    episodes: PROBE_EPISODES,
    goals,
    passActions: totalPassActions,
    passCompleted: totalPassCompleted,
    shotActions: totalShotActions,
    shotEvents: totalShotEvents,
    turnovers: totalTurnovers,
    timeouts: totalTimeouts,
    crashes: totalCrashes,
    avgSteps: Math.round((totalSteps / PROBE_EPISODES) * 100) / 100,
    avgReward: Math.round((totalReward / PROBE_EPISODES) * 10000) / 10000,
    notes: `Goal rate: ${((goals / PROBE_EPISODES) * 100).toFixed(1)}%. Feasibility demonstrated if goals > 0 or reward/step > 0.`,
  }];

  console.log(`Episodes:        ${PROBE_EPISODES}`);
  console.log(`Goals:           ${goals} (${((goals / PROBE_EPISODES) * 100).toFixed(1)}%)`);
  console.log(`Pass Actions:    ${totalPassActions} (${(totalPassActions / PROBE_EPISODES).toFixed(1)}/ep)`);
  console.log(`Pass Completed:  ${totalPassCompleted} (${(totalPassCompleted / PROBE_EPISODES).toFixed(1)}/ep)`);
  console.log(`Shot Actions:    ${totalShotActions} (${(totalShotActions / PROBE_EPISODES).toFixed(1)}/ep)`);
  console.log(`Shot Events:     ${totalShotEvents} (${(totalShotEvents / PROBE_EPISODES).toFixed(1)}/ep)`);
  console.log(`Turnovers:       ${totalTurnovers}`);
  console.log(`Timeouts:        ${totalTimeouts}`);
  console.log(`Crashes:         ${totalCrashes}`);
  console.log(`Avg Steps:       ${Math.round((totalSteps / PROBE_EPISODES) * 100) / 100}`);
  console.log(`Avg Reward:      ${Math.round((totalReward / PROBE_EPISODES) * 10000) / 10000}`);
  console.log(`Notes:           ${results[0].notes}`);

  return results;
}

// ---------------------------------------------------------------------------
// Step 3: Random Legal Baseline (reconciled split metrics)
// ---------------------------------------------------------------------------
function runRandomLegalBaseline(): ProbeResult[] {
  console.log('\n==================================================');
  console.log('STEP 3: RANDOM LEGAL BASELINE (ACADEMY DRILLS)');
  console.log('==================================================\n');

  const results: ProbeResult[] = [];

  for (const scenario of ACADEMY_DRILLS) {
    let goals = 0;
    let totalPassActions = 0;
    let totalPassCompleted = 0;
    let totalShotActions = 0;
    let totalShotEvents = 0;
    let totalTurnovers = 0;
    let totalTimeouts = 0;
    let totalCrashes = 0;
    let totalSteps = 0;
    let totalReward = 0;

    for (let ep = 0; ep < PROBE_EPISODES; ep++) {
      const engine = new GameEngine();
      engine.loadScenario(scenario, DETERMINISTIC_SEED + ep);

      let epPassActions = 0;
      let epPassCompleted = 0;
      let epShotActions = 0;
      let epShotEvents = 0;
      let epReward = 0;
      let done = false;
      let steps = 0;
      const maxSteps = scenario.timeLimitSeconds * 60 + 120;
      const stepRng = new SeededRNG(DETERMINISTIC_SEED + ep);

      while (!done && steps < maxSteps) {
        steps++;
        const actionMap = new Map<string, AgentAction>();
        for (const player of engine.players) {
          const actionIdx = stepRng.nextInt(0, 18);
          const action = mapDiscreteAction(actionIdx);
          actionMap.set(player.id, action);
          if (player.team === 'left') {
            if (isPassAction(action)) epPassActions++;
            if (isShotAction(action)) epShotActions++;
          }
        }
        const res = engine.step(actionMap, 1 / 60);
        epReward += res.reward;

        for (const ev of engine.events) {
          if (ev.type === EVENT_PASS_COMPLETED && ev.team === 'left') epPassCompleted++;
          if (ev.type === EVENT_SHOT && ev.team === 'left') epShotEvents++;
        }

        if (res.terminated) {
          done = true;
          if (engine.score.left > 0) goals++;
          else if (engine.ball.ownerId && engine.players.find((p) => p.id === engine.ball.ownerId)?.team === 'right') {
            totalTurnovers++;
          }
        }
        if (res.truncated) {
          done = true;
          totalTimeouts++;
        }
      }

      totalPassActions += epPassActions;
      totalPassCompleted += epPassCompleted;
      totalShotActions += epShotActions;
      totalShotEvents += epShotEvents;
      totalSteps += steps;
      totalReward += epReward;
    }

    const result: ProbeResult = {
      scenarioId: scenario.id,
      probeName: 'RandomLegalBaseline',
      episodes: PROBE_EPISODES,
      goals,
      passActions: totalPassActions,
      passCompleted: totalPassCompleted,
      shotActions: totalShotActions,
      shotEvents: totalShotEvents,
      turnovers: totalTurnovers,
      timeouts: totalTimeouts,
      crashes: totalCrashes,
      avgSteps: Math.round((totalSteps / PROBE_EPISODES) * 100) / 100,
      avgReward: Math.round((totalReward / PROBE_EPISODES) * 10000) / 10000,
      notes: `Goal rate: ${((goals / PROBE_EPISODES) * 100).toFixed(1)}%. Lower bound on feasibility.`,
    };

    results.push(result);
    console.log(`[${scenario.id}] Goals: ${goals}/${PROBE_EPISODES} | PassActions: ${totalPassActions} | PassCompleted: ${totalPassCompleted} | ShotActions: ${totalShotActions} | Turnovers: ${totalTurnovers} | Timeouts: ${totalTimeouts}`);
  }

  return results;
}

// ---------------------------------------------------------------------------
// Step 4a: Possession / Control Consistency
// ---------------------------------------------------------------------------
function runPossessionControlConsistencyProbe(): ProbeResult[] {
  console.log('\n==================================================');
  console.log('STEP 4a: POSSESSION / CONTROL CONSISTENCY PROBE');
  console.log('==================================================\n');

  const results: ProbeResult[] = [];

  for (const scenario of ACADEMY_DRILLS) {
    let inconsistencyCount = 0;
    let totalTicks = 0;
    let crashes = 0;

    for (let ep = 0; ep < Math.min(PROBE_EPISODES, 10); ep++) {
      const engine = new GameEngine();
      engine.loadScenario(scenario, DETERMINISTIC_SEED + ep);

      const maxSteps = scenario.timeLimitSeconds * 60 + 60;
      const stepRng = new SeededRNG(DETERMINISTIC_SEED + ep);
      for (let step = 0; step < maxSteps; step++) {
        const actionMap = new Map<string, AgentAction>();
        for (const player of engine.players) {
          const actionIdx = stepRng.nextInt(0, 18);
          actionMap.set(player.id, mapDiscreteAction(actionIdx));
        }
        const res = engine.step(actionMap, 1 / 60);
        if (res.terminated || res.truncated) break;

        totalTicks++;
        const ownerId = engine.ball.ownerId;
        if (ownerId) {
          const owner = engine.players.find((p) => p.id === ownerId);
          if (!owner) {
            inconsistencyCount++;
            continue;
          }
          const hasBallCount = engine.players.filter((p) => p.hasBall).length;
          if (hasBallCount !== 1) {
            inconsistencyCount++;
          }
          if (!owner.hasBall) {
            inconsistencyCount++;
          }
        } else {
          const hasBallCount = engine.players.filter((p) => p.hasBall).length;
          if (hasBallCount !== 0) {
            inconsistencyCount++;
          }
        }
      }
    }

    const result: ProbeResult = {
      scenarioId: scenario.id,
      probeName: 'PossessionControlConsistency',
      episodes: Math.min(PROBE_EPISODES, 10),
      goals: 0,
      passActions: 0,
      passCompleted: 0,
      shotActions: 0,
      shotEvents: 0,
      turnovers: 0,
      timeouts: 0,
      crashes,
      avgSteps: 0,
      avgReward: 0,
      notes: `Inconsistencies: ${inconsistencyCount} / ${totalTicks} ticks (${totalTicks > 0 ? ((inconsistencyCount / totalTicks) * 100).toFixed(4) : 0}%)`,
    };

    results.push(result);
    console.log(`[${scenario.id}] Inconsistencies: ${inconsistencyCount}/${totalTicks} ticks`);
  }

  return results;
}

function clampPassDirection(dir: { x: number; y: number }, maxY = 0.5): { x: number; y: number } {
  const clampedY = Math.max(-maxY, Math.min(maxY, dir.y));
  const x = dir.x;
  const y = clampedY;
  const len = Math.sqrt(x * x + y * y);
  if (len === 0) return { x: 1.0, y: 0.0 };
  return { x: x / len, y: y / len };
}

// ---------------------------------------------------------------------------
// Step 4b: Forced PASS / SHOT under hard guarantees
// ---------------------------------------------------------------------------
interface ForcedProbeStats {
  passCommanded: number;
  passCompleted: number;
  shotCommanded: number;
  shotFired: number;
  ownershipChangesAfterPass: number;
  ballDeltaXAfterShot: number[];
  tickLagToGoalAfterShot: number[];
  maskSamples: number[];
  failReasons: Record<string, number>;
}

function runForcedPassShotProbe(): ProbeResult[] {
  console.log('\n==================================================');
  console.log('STEP 4b: FORCED PASS/SHOT PROBES (PRIMARY + ACADEMY DRILLS)');
  console.log('==================================================\n');

  const results: ProbeResult[] = [];

  for (const scenario of ACADEMY_DRILLS) {
    const stats: ForcedProbeStats = {
      passCommanded: 0,
      passCompleted: 0,
      shotCommanded: 0,
      shotFired: 0,
      ownershipChangesAfterPass: 0,
      ballDeltaXAfterShot: [],
      tickLagToGoalAfterShot: [],
      maskSamples: [],
      failReasons: {},
    };
    const episodes = PROBE_EPISODES;

    for (let ep = 0; ep < episodes; ep++) {
      const engine = new GameEngine();
      engine.loadScenario(scenario, DETERMINISTIC_SEED + ep);

      // Roll forward tick-by-tick until we find a valid left-possession +
      // mask-legal state, or the episode ends.
      const maxSearchSteps = scenario.timeLimitSeconds * 60 + 120;
      const stepRng = new SeededRNG(DETERMINISTIC_SEED + ep);
      let foundState = false;

      for (let step = 0; step < maxSearchSteps; step++) {
        const actionMap = new Map<string, AgentAction>();
        for (const player of engine.players) {
          const actionIdx = stepRng.nextInt(0, 18);
          actionMap.set(player.id, mapDiscreteAction(actionIdx));
        }
        const res = engine.step(actionMap, 1 / 60);
        if (res.terminated || res.truncated) break;

        // Check all left controllable players for valid state
        const candidates = leftPlayers(engine).filter((p) => !p.isGoalkeeper);
        for (const player of candidates) {
          const hasPossession = player.hasBall || engine.ball.ownerId === player.id;
          if (!hasPossession) continue;

          const mask = ObservationEncoder.getActionMask(player, engine);
          const shortPassLegal = mask[ACTION_SHORT_PASS] === 1;
          const shotLegal = mask[ACTION_SHOT] === 1;

          if (!shortPassLegal && !shotLegal) {
            const reason = 'mask_illegal';
            stats.failReasons[reason] = (stats.failReasons[reason] || 0) + 1;
            continue;
          }

          stats.maskSamples.push(mask[ACTION_SHORT_PASS]);
          foundState = true;

          // Record ball position before forced action
          const ballXBefore = engine.ball.position.x;
          const ownerIdBefore = engine.ball.ownerId;

          // Forced SHORT_PASS — use forward-biased direction and reduced power
          // to keep the ball in bounds and allow a teammate to receive it.
          if (shortPassLegal) {
            stats.passCommanded++;
            const passTargets = engine.players.filter(
              (p) => p.team === 'left' && p.id !== player.id && !p.isGoalkeeper
            );
            const targetId = passTargets[0]?.id || player.id;
            const passDir = clampPassDirection(
              passTargets[0]
                ? Vec2.normalize({ x: passTargets[0].position.x - player.position.x, y: passTargets[0].position.y - player.position.y })
                : { x: 1.0, y: 0.0 }
            );
            const passPower = 0.75;

            const passAction: AgentAction = {
              type: ActionType.SHORT_PASS,
              direction: passDir,
              power: passPower,
              targetPlayerId: targetId,
            };
            const forcedMap = new Map<string, AgentAction>();
            forcedMap.set(player.id, passAction);
            for (const p of engine.players) {
              if (p.id !== player.id) {
                // Make non-passing players sprint toward the ball to increase
                // the chance of receiving the pass (matches deterministic
                // controller support-run behavior).
                const ballPos = engine.ball.position;
                const dir = Vec2.normalize({ x: ballPos.x - p.position.x, y: ballPos.y - p.position.y });
                forcedMap.set(p.id, { type: ActionType.SPRINT, direction: dir });
              }
            }

            const passRes = engine.step(forcedMap, 1 / 60);

            // Observe for pass completion within next 50 ticks (pass may take
            // multiple physics steps to reach the target).
            const observeWindow = 50;
            let passCompleted = false;
            let eventCursor = engine.events.length;
            for (let obs = 1; obs <= observeWindow; obs++) {
              if (passRes.terminated || passRes.truncated) break;
              const obsMap = new Map<string, AgentAction>();
              for (const p of engine.players) {
                obsMap.set(p.id, { type: ActionType.IDLE });
              }
              const obsRes = engine.step(obsMap, 1 / 60);
              const newEvents = engine.events.slice(eventCursor);
              eventCursor = engine.events.length;

              if (newEvents.some((e) => e.type === EVENT_PASS_COMPLETED && e.team === 'left')) {
                passCompleted = true;
                break;
              }
              if (newEvents.some((e) => e.type === 'pass_intercepted' || e.type === 'out_of_bounds')) {
                break;
              }
              if (obsRes.terminated || obsRes.truncated) break;
            }
            if (passCompleted) stats.passCompleted++;

            // Ownership change?
            if (engine.ball.ownerId && engine.ball.ownerId !== ownerIdBefore) {
              stats.ownershipChangesAfterPass++;
            }

            // If episode ended, stop observing
            if (passRes.terminated || passRes.truncated) break;
          }

          // --- Forced SHOT ---
          if (shotLegal) {
            stats.shotCommanded++;
            const goalX = player.team === 'left' ? 1.0 : -1.0;
            const shotDir = Vec2.normalize({ x: goalX - player.position.x, y: 0 - player.position.y });
            const shotAction: AgentAction = {
              type: ActionType.SHOT,
              direction: shotDir,
              power: 0.95,
            };
            const forcedMap = new Map<string, AgentAction>();
            forcedMap.set(player.id, shotAction);
            for (const p of engine.players) {
              if (p.id !== player.id) forcedMap.set(p.id, { type: ActionType.IDLE });
            }

            const beforeEvents = engine.events.length;
            const shotRes = engine.step(forcedMap, 1 / 60);
            const newEvents = engine.events.slice(beforeEvents);

            if (newEvents.some((e) => SHOT_EVENT_TYPES.has(e.type) && e.team === 'left')) {
              stats.shotFired++;
            }

            // Ball delta X after shot
            const ballXAfter = engine.ball.position.x;
            stats.ballDeltaXAfterShot.push(ballXAfter - ballXBefore);

            // Look for goal within next 50 ticks
            let goalTickLag: number | null = null;
            const observeWindow = 50;
            for (let obs = 1; obs <= observeWindow; obs++) {
              if (shotRes.terminated || shotRes.truncated) break;
              const obsMap = new Map<string, AgentAction>();
              for (const p of engine.players) {
                obsMap.set(p.id, { type: ActionType.IDLE });
              }
              const obsRes = engine.step(obsMap, 1 / 60);
              if (engine.score.left > (scenario.id === PRIMARY_SCENARIO ? 0 : 0)) {
                // Check if this was a left goal
                const goalEvents = engine.events.filter(
                  (e) => e.type === EVENT_GOAL && e.team === 'left' && e.timeSeconds > (shotRes.info?.ballDistanceToGoal || 0)
                );
                if (goalEvents.length > 0) {
                  goalTickLag = obs;
                  break;
                }
              }
              if (obsRes.terminated || obsRes.truncated) break;
            }
            if (goalTickLag !== null) {
              stats.tickLagToGoalAfterShot.push(goalTickLag);
            }

            if (shotRes.terminated || shotRes.truncated) break;
          }

          // Only test one candidate per episode to keep it simple
          break;
        }
      }

      if (!foundState) {
        const reason = 'no_valid_state';
        stats.failReasons[reason] = (stats.failReasons[reason] || 0) + 1;
      }
    }

    const result: ProbeResult = {
      scenarioId: scenario.id,
      probeName: 'ForcedPassShot',
      episodes,
      goals: 0,
      passActions: stats.passCommanded,
      passCompleted: stats.passCompleted,
      shotActions: stats.shotCommanded,
      shotEvents: stats.shotFired,
      turnovers: 0,
      timeouts: 0,
      crashes: 0,
      avgSteps: 0,
      avgReward: 0,
      notes: `P(PASS_COMPLETED|poss+mask+cmd)=${stats.passCommanded > 0 ? ((stats.passCompleted / stats.passCommanded) * 100).toFixed(1) : 'N/A'}% (${stats.passCompleted}/${stats.passCommanded}). P(SHOT|poss+mask+cmd)=${stats.shotCommanded > 0 ? ((stats.shotFired / stats.shotCommanded) * 100).toFixed(1) : 'N/A'}% (${stats.shotFired}/${stats.shotCommanded}). Fail reasons: ${JSON.stringify(stats.failReasons)}`,
    };

    results.push(result);
    console.log(`[${scenario.id}] PASS: ${stats.passCompleted}/${stats.passCommanded} | SHOT: ${stats.shotFired}/${stats.shotCommanded} | FailReasons: ${JSON.stringify(stats.failReasons)}`);
  }

  return results;
}

// ---------------------------------------------------------------------------
// Step 4c: Mask Opportunity + Transition Integrity (200-tick lookback) + Reward Ordering
// ---------------------------------------------------------------------------
function runMaskOpportunityAnalysis(): void {
  console.log('\n==================================================');
  console.log('STEP 4c: MASK OPPORTUNITY ANALYSIS');
  console.log('==================================================\n');

  for (const scenario of ACADEMY_DRILLS) {
    const engine = new GameEngine();
    engine.loadScenario(scenario, DETERMINISTIC_SEED);

    const masks: number[][] = [];
    const maxSteps = Math.min(60, scenario.timeLimitSeconds * 60);
    const stepRng = new SeededRNG(DETERMINISTIC_SEED);
    for (let step = 0; step < maxSteps; step++) {
      const actionMap = new Map<string, AgentAction>();
      for (const player of engine.players) {
        const actionIdx = stepRng.nextInt(0, 18);
        actionMap.set(player.id, mapDiscreteAction(actionIdx));
      }
      const res = engine.step(actionMap, 1 / 60);
      if (res.terminated || res.truncated) break;

      for (const player of leftPlayers(engine)) {
        const mask = ObservationEncoder.getActionMask(player, engine);
        masks.push(mask);
      }
    }

    if (masks.length === 0) continue;

    const validCounts = new Array(19).fill(0);
    for (const mask of masks) {
      for (let i = 0; i < 19; i++) {
        if (mask[i] === 1) validCounts[i]++;
      }
    }
    const validPct = validCounts.map((c) => ((c / masks.length) * 100).toFixed(1) + '%').join(', ');

    console.log(`[${scenario.id}] Mask validity across ${masks.length} samples:`);
    console.log(`  IDLE..MOVE(1-8): always valid`);
    console.log(`  LONG_PASS(9): ${validPct.split(',')[9]}`);
    console.log(`  HIGH_PASS(10): ${validPct.split(',')[10]}`);
    console.log(`  SHORT_PASS(11): ${validPct.split(',')[11]}`);
    console.log(`  SHOT(12): ${validPct.split(',')[12]}`);
    console.log(`  TACKLE(16): ${validPct.split(',')[16]}`);
    console.log('');
  }
}

function runTransitionIntegrityCheck(): void {
  console.log('\n==================================================');
  console.log('STEP 4c: TRANSITION INTEGRITY CHECK (200-tick lookback)');
  console.log('==================================================\n');

  for (const scenario of ACADEMY_DRILLS) {
    let totalGoals = 0;
    let precededByShotOrPass = 0;
    let noPriorShotPass = 0;
    const episodes = Math.min(PROBE_EPISODES, 10);

    for (let ep = 0; ep < episodes; ep++) {
      const engine = new GameEngine();
      engine.loadScenario(scenario, DETERMINISTIC_SEED + ep);

      let prevScoreLeft = 0;
      const maxSteps = scenario.timeLimitSeconds * 60 + 60;
      const stepRng = new SeededRNG(DETERMINISTIC_SEED + ep);
      for (let step = 0; step < maxSteps; step++) {
        const actionMap = new Map<string, AgentAction>();
        for (const player of engine.players) {
          const actionIdx = stepRng.nextInt(0, 18);
          actionMap.set(player.id, mapDiscreteAction(actionIdx));
        }
        const res = engine.step(actionMap, 1 / 60);

        // Detect goal by score delta
        if (engine.score.left > prevScoreLeft) {
          totalGoals++;

          // Search backward up to 200 ticks for the most recent shot or pass_completed
          const lookbackLimit = Math.min(200, engine.events.length);
          let foundPrior = false;
          for (let i = engine.events.length - 1; i >= Math.max(0, engine.events.length - lookbackLimit); i--) {
            const ev = engine.events[i];
            if (ev.type === EVENT_SHOT || ev.type === EVENT_PASS_COMPLETED) {
              precededByShotOrPass++;
              foundPrior = true;
              break;
            }
          }
          if (!foundPrior) {
            noPriorShotPass++;
          }
        }
        prevScoreLeft = engine.score.left;

        if (res.terminated || res.truncated) break;
      }
    }

    console.log(`[${scenario.id}] Goals: ${totalGoals} | Preceded by shot/pass within 200 ticks: ${precededByShotOrPass} | No prior shot/pass: ${noPriorShotPass}`);
  }
}

function runOfflineRewardOrderingCheck(): void {
  console.log('\n==================================================');
  console.log('STEP 4c: OFFLINE REWARD ORDERING CHECK');
  console.log('==================================================\n');

  for (const scenario of ACADEMY_DRILLS) {
    const engine = new GameEngine();
    engine.loadScenario(scenario, DETERMINISTIC_SEED);

    const rewards: number[] = [];
    let totalReward = 0;
    const maxSteps = Math.min(120, scenario.timeLimitSeconds * 60);
    const stepRng = new SeededRNG(DETERMINISTIC_SEED);
    for (let step = 0; step < maxSteps; step++) {
      const actionMap = new Map<string, AgentAction>();
      for (const player of engine.players) {
        const actionIdx = stepRng.nextInt(0, 18);
        actionMap.set(player.id, mapDiscreteAction(actionIdx));
      }
      const res = engine.step(actionMap, 1 / 60);
      totalReward += res.reward;
      rewards.push(res.reward);
      if (res.terminated || res.truncated) break;
    }

    const monotonicIncreasing = rewards.filter((r, i) => i === 0 || r >= rewards[i - 1]).length;
    const monotonicDecreasing = rewards.filter((r, i) => i === 0 || r <= rewards[i - 1]).length;
    const isMonotonic = monotonicIncreasing === rewards.length || monotonicDecreasing === rewards.length;

    console.log(`[${scenario.id}] Total reward: ${totalReward.toFixed(4)} | Monotonic: ${isMonotonic} (${Math.max(monotonicIncreasing, monotonicDecreasing)}/${rewards.length} steps)`);
  }
}

// ---------------------------------------------------------------------------
// Step 4d: Spam-move / Spam-tackle / Scripted-best return ordering (I)
// ---------------------------------------------------------------------------
interface ReturnOrderResult {
  scenarioId: string;
  policy: string;
  episodes: number;
  meanReturn: number;
  stdReturn: number;
  minReturn: number;
  maxReturn: number;
  goals: number;
}

function runSpamMovePolicy(scenario: any, episodes: number, seedBase: number): ReturnOrderResult {
  let totalReturn = 0;
  const returns: number[] = [];
  let goals = 0;

  for (let ep = 0; ep < episodes; ep++) {
    const engine = new GameEngine();
    engine.loadScenario(scenario, seedBase + ep);
    let epReturn = 0;
    let done = false;
    let steps = 0;
    const maxSteps = scenario.timeLimitSeconds * 60 + 120;
    const stepRng = new SeededRNG(seedBase + ep);

    while (!done && steps < maxSteps) {
      steps++;
      const actionMap = new Map<string, AgentAction>();
      for (const player of engine.players) {
        if (player.team === 'left') {
          // Spam-move: always move forward or idle
          const actionIdx = stepRng.nextInt(1, 8); // MOVE actions
          actionMap.set(player.id, mapDiscreteAction(actionIdx));
        } else {
          const actionIdx = stepRng.nextInt(0, 18);
          actionMap.set(player.id, mapDiscreteAction(actionIdx));
        }
      }
      const res = engine.step(actionMap, 1 / 60);
      epReturn += res.reward;

      if (res.terminated) {
        if (engine.score.left > 0) goals++;
        done = true;
      }
      if (res.truncated) done = true;
    }

    totalReturn += epReturn;
    returns.push(epReturn);
  }

  const mean = totalReturn / episodes;
  const variance = returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / episodes;
  const std = Math.sqrt(variance);

  return {
    scenarioId: scenario.id,
    policy: 'spam-move',
    episodes,
    meanReturn: Math.round(mean * 10000) / 10000,
    stdReturn: Math.round(std * 10000) / 10000,
    minReturn: Math.round(Math.min(...returns) * 10000) / 10000,
    maxReturn: Math.round(Math.max(...returns) * 10000) / 10000,
    goals,
  };
}

function runSpamTacklePolicy(scenario: any, episodes: number, seedBase: number): ReturnOrderResult {
  let totalReturn = 0;
  const returns: number[] = [];
  let goals = 0;

  for (let ep = 0; ep < episodes; ep++) {
    const engine = new GameEngine();
    engine.loadScenario(scenario, seedBase + ep);
    let epReturn = 0;
    let done = false;
    let steps = 0;
    const maxSteps = scenario.timeLimitSeconds * 60 + 120;
    const stepRng = new SeededRNG(seedBase + ep);

    while (!done && steps < maxSteps) {
      steps++;
      const actionMap = new Map<string, AgentAction>();
      for (const player of engine.players) {
        if (player.team === 'left') {
          // Spam-tackle: tackle when legal, else move
          const mask = ObservationEncoder.getActionMask(player, engine);
          if (mask[ACTION_TACKLE] === 1) {
            actionMap.set(player.id, mapDiscreteAction(ACTION_TACKLE));
          } else {
            const actionIdx = stepRng.nextInt(1, 8);
            actionMap.set(player.id, mapDiscreteAction(actionIdx));
          }
        } else {
          const actionIdx = stepRng.nextInt(0, 18);
          actionMap.set(player.id, mapDiscreteAction(actionIdx));
        }
      }
      const res = engine.step(actionMap, 1 / 60);
      epReturn += res.reward;

      if (res.terminated) {
        if (engine.score.left > 0) goals++;
        done = true;
      }
      if (res.truncated) done = true;
    }

    totalReturn += epReturn;
    returns.push(epReturn);
  }

  const mean = totalReturn / episodes;
  const variance = returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / episodes;
  const std = Math.sqrt(variance);

  return {
    scenarioId: scenario.id,
    policy: 'spam-tackle',
    episodes,
    meanReturn: Math.round(mean * 10000) / 10000,
    stdReturn: Math.round(std * 10000) / 10000,
    minReturn: Math.round(Math.min(...returns) * 10000) / 10000,
    maxReturn: Math.round(Math.max(...returns) * 10000) / 10000,
    goals,
  };
}

function runScriptedBestPolicy(scenario: any, episodes: number, seedBase: number): ReturnOrderResult {
  const controller = new DeterministicFeasibilityController();
  let totalReturn = 0;
  const returns: number[] = [];
  let goals = 0;

  for (let ep = 0; ep < episodes; ep++) {
    const engine = new GameEngine();
    engine.loadScenario(scenario, seedBase + ep);
    controller.reset(engine);

    let epReturn = 0;
    let done = false;
    let steps = 0;
    const maxSteps = scenario.timeLimitSeconds * 60 + 120;

    while (!done && steps < maxSteps) {
      steps++;
      const actionMap = controller.decide(engine);
      const res = engine.step(actionMap, 1 / 60);
      epReturn += res.reward;

      if (res.terminated) {
        if (engine.score.left > 0) goals++;
        done = true;
      }
      if (res.truncated) done = true;
    }

    totalReturn += epReturn;
    returns.push(epReturn);
  }

  const mean = totalReturn / episodes;
  const variance = returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / episodes;
  const std = Math.sqrt(variance);

  return {
    scenarioId: scenario.id,
    policy: 'scripted-best',
    episodes,
    meanReturn: Math.round(mean * 10000) / 10000,
    stdReturn: Math.round(std * 10000) / 10000,
    minReturn: Math.round(Math.min(...returns) * 10000) / 10000,
    maxReturn: Math.round(Math.max(...returns) * 10000) / 10000,
    goals,
  };
}

function runReturnOrderingCheck(): ReturnOrderResult[] {
  console.log('\n==================================================');
  console.log('STEP 4d: RETURN ORDERING CHECK (I) — PRIMARY CASE');
  console.log('==================================================\n');

  const scenario = ACADEMY_SCENARIOS.find((s) => s.id === PRIMARY_SCENARIO)!;
  const episodes = 20;

  const spamMove = runSpamMovePolicy(scenario, episodes, DETERMINISTIC_SEED);
  const spamTackle = runSpamTacklePolicy(scenario, episodes, DETERMINISTIC_SEED);
  const scriptedBest = runScriptedBestPolicy(scenario, episodes, DETERMINISTIC_SEED);

  console.log(`[${PRIMARY_SCENARIO}]`);
  console.log(`  Spam-move:      mean=${spamMove.meanReturn} std=${spamMove.stdReturn} goals=${spamMove.goals}`);
  console.log(`  Spam-tackle:    mean=${spamTackle.meanReturn} std=${spamTackle.stdReturn} goals=${spamTackle.goals}`);
  console.log(`  Scripted-best:  mean=${scriptedBest.meanReturn} std=${scriptedBest.stdReturn} goals=${scriptedBest.goals}`);

  const scriptedBetterThanMove = scriptedBest.meanReturn > spamMove.meanReturn;
  const scriptedBetterThanTackle = scriptedBest.meanReturn > spamTackle.meanReturn;
  console.log(`  Scripted > Spam-move: ${scriptedBetterThanMove}`);
  console.log(`  Scripted > Spam-tackle: ${scriptedBetterThanTackle}`);
  console.log(`  I pass: ${scriptedBetterThanMove && scriptedBetterThanTackle}`);

  return [spamMove, spamTackle, scriptedBest];
}

// ---------------------------------------------------------------------------
// Step 5: Write Outputs
// ---------------------------------------------------------------------------
function writeResults(results: ProbeResult[], _filename: string): void {
  const path = `experiment_d_results_${Date.now()}.json`;
  fs.writeFileSync(path, JSON.stringify(results, null, 2));
  console.log(`\n[OUTPUT] Wrote ${path}`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
function main(): void {
  console.log('==================================================');
  console.log('EXPERIMENT D: WHOLE-REGISTRY SCENARIO QUALIFICATION (CONTINUATION)');
  console.log('Primary Forensic Case: academy_3_vs_1_with_keeper');
  console.log('==================================================');

  // Step 1
  printScenarioCards();

  // Step 2
  const feasibilityResults = runDeterministicFeasibilityProbe();

  // Step 3
  const baselineResults = runRandomLegalBaseline();

  // Step 4a
  const consistencyResults = runPossessionControlConsistencyProbe();

  // Step 4b
  const forcedResults = runForcedPassShotProbe();

  // Step 4c
  runMaskOpportunityAnalysis();
  runTransitionIntegrityCheck();
  runOfflineRewardOrderingCheck();

  // Step 4d
  const returnOrderResults = runReturnOrderingCheck();

  // Summary
  const allResults = [
    ...feasibilityResults,
    ...baselineResults,
    ...consistencyResults,
    ...forcedResults,
    ...returnOrderResults,
  ] as ProbeResult[];
  writeResults(allResults, 'experiment_d_results.json');

  console.log('\n==================================================');
  console.log('EXPERIMENT D CONTINUATION COMPLETE');
  console.log('==================================================');
}

if (import.meta.url.endsWith(process.argv[1]) || process.argv[1]?.includes('experiment_d_scenario_probes')) {
  main();
}
