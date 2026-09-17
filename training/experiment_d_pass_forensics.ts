import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ActionType, AgentAction } from '../src/types/football';
import { mapDiscreteAction } from '../src/engine/ActionMapping';
import { Vec2 } from '../src/engine/Vector';
import { SeededRNG } from '../src/engine/SeededRNG';
import { ObservationEncoder } from '../src/engine/ObservationEncoder';
import * as fs from 'fs';

const PRIMARY = 'academy_3_vs_1_with_keeper';
const SEED_BASE = 12345;
const ACTION_SHORT_PASS = 11;
const ACTION_SHOT = 12;

interface TickLog {
  tick: number;
  phase: 'before' | 'after_command' | 'observe';
  event?: string;
  ballOwnerId?: string | null;
  ballX?: number;
  ballY?: number;
  ballZ?: number;
  ballVx?: number;
  ballVy?: number;
  ballVz?: number;
  nearestTeammateId?: string;
  nearestTeammateDist?: number;
  ballToNearestTeammateDist?: number;
  teammatePositions?: Record<string, { x: number; y: number }>;
  actingPlayerId?: string;
  actingPlayerX?: number;
  actingPlayerY?: number;
  maskPass?: number;
  maskShot?: number;
  maskTackle?: number;
  actionTaken?: string;
  passCompleted?: boolean;
  shotFired?: boolean;
  ownershipChanged?: boolean;
  notes?: string;
}

function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function nearestTeammate(playerId: string, engine: GameEngine) {
  const player = engine.players.find((p) => p.id === playerId);
  if (!player) return { id: null, dist: Infinity };
  const mates = engine.players.filter((p) => p.team === player.team && p.id !== player.id && !p.isGoalkeeper);
  let best = { id: null, dist: Infinity };
  for (const m of mates) {
    const d = dist(player.position, m.position);
    if (d < best.dist) best = { id: m.id, dist: d };
  }
  return best;
}

function ballToNearestTeammate(engine: GameEngine) {
  const ballPos = { x: engine.ball.position.x, y: engine.ball.position.y };
  const mates = engine.players.filter((p) => p.team === 'left' && !p.isGoalkeeper);
  let best = { id: null, dist: Infinity };
  for (const m of mates) {
    const d = dist(ballPos, m.position);
    if (d < best.dist) best = { id: m.id, dist: d };
  }
  return best;
}

function teammatePositions(engine: GameEngine): Record<string, { x: number; y: number }> {
  const positions: Record<string, { x: number; y: number }> = {};
  for (const p of engine.players.filter((p) => p.team === 'left' && !p.isGoalkeeper)) {
    positions[p.id] = { x: p.position.x, y: p.position.y };
  }
  return positions;
}

function clampPassDirection(dir: { x: number; y: number }, maxY = 0.5): { x: number; y: number } {
  const clampedY = Math.max(-maxY, Math.min(maxY, dir.y));
  const x = dir.x;
  const y = clampedY;
  const len = Math.sqrt(x * x + y * y);
  if (len === 0) return { x: 1.0, y: 0.0 };
  return { x: x / len, y: y / len };
}

function runForensics(episodes = 20): void {
  const scenario = ACADEMY_SCENARIOS.find((s) => s.id === PRIMARY)!;
  const allLogs: TickLog[] = [];
  let validStatesFound = 0;
  let passCommanded = 0;
  let passCompleted = 0;
  let shotCommanded = 0;
  let shotFired = 0;
  const failReasons: Record<string, number> = {};

  for (let ep = 0; ep < episodes; ep++) {
    const engine = new GameEngine();
    engine.loadScenario(scenario, SEED_BASE + ep);
    const rng = new SeededRNG(SEED_BASE + ep);
    const maxSearch = scenario.timeLimitSeconds * 60 + 120;
    let found = false;

    for (let step = 0; step < maxSearch; step++) {
      const actionMap = new Map<string, AgentAction>();
      for (const player of engine.players) {
        const idx = rng.nextInt(0, 18);
        actionMap.set(player.id, mapDiscreteAction(idx));
      }
      const res = engine.step(actionMap, 1 / 60);
      if (res.terminated || res.truncated) break;

      const candidates = engine.players.filter((p) => p.team === 'left' && !p.isGoalkeeper);
      for (const player of candidates) {
        const hasPossession = player.hasBall || engine.ball.ownerId === player.id;
        if (!hasPossession) continue;

        const mask = ObservationEncoder.getActionMask(player, engine);
        const passLegal = mask[ACTION_SHORT_PASS] === 1;
        const shotLegal = mask[ACTION_SHOT] === 1;
        if (!passLegal && !shotLegal) {
          failReasons['mask_illegal'] = (failReasons['mask_illegal'] || 0) + 1;
          continue;
        }

        validStatesFound++;
        found = true;
        const nearest = nearestTeammate(player.id, engine);
        const ballToNearest = ballToNearestTeammate(engine);

        // BEFORE log
        allLogs.push({
          tick: engine.tickCount,
          phase: 'before',
          ballOwnerId: engine.ball.ownerId,
          ballX: engine.ball.position.x,
          ballY: engine.ball.position.y,
          ballZ: engine.ball.position.z,
          ballVx: engine.ball.velocity.x,
          ballVy: engine.ball.velocity.y,
          ballVz: engine.ball.velocity.z,
          nearestTeammateId: nearest.id,
          nearestTeammateDist: nearest.dist,
          ballToNearestTeammateDist: ballToNearest.dist,
          teammatePositions: teammatePositions(engine),
          actingPlayerId: player.id,
          actingPlayerX: player.position.x,
          actingPlayerY: player.position.y,
          maskPass: mask[ACTION_SHORT_PASS],
          maskShot: mask[ACTION_SHOT],
          maskTackle: mask[16],
          notes: `owner=${engine.ball.ownerId} passLegal=${passLegal} shotLegal=${shotLegal}`,
        });

        // Command SHORT_PASS
        if (passLegal) {
          passCommanded++;
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
            if (p.id !== player.id) forcedMap.set(p.id, { type: ActionType.IDLE });
          }

          const beforeEvents = engine.events.length;
          const passRes = engine.step(forcedMap, 1 / 60);
          const newEvents = engine.events.slice(beforeEvents);
          const completed = newEvents.some((e) => e.type === 'pass_completed' && e.team === 'left');
          if (completed) passCompleted++;

          const ballToNearestAfter = ballToNearestTeammate(engine);
          allLogs.push({
            tick: engine.tickCount,
            phase: 'after_command',
            event: completed ? 'pass_completed' : 'none',
            ballOwnerId: engine.ball.ownerId,
            ballX: engine.ball.position.x,
            ballY: engine.ball.position.y,
            ballZ: engine.ball.position.z,
            ballVx: engine.ball.velocity.x,
            ballVy: engine.ball.velocity.y,
            ballVz: engine.ball.velocity.z,
            nearestTeammateId: nearest.id,
            nearestTeammateDist: nearest.dist,
            ballToNearestTeammateDist: ballToNearestAfter.dist,
            teammatePositions: teammatePositions(engine),
            actingPlayerId: player.id,
            actingPlayerX: player.position.x,
            actingPlayerY: player.position.y,
            passCompleted: completed,
            ownershipChanged: engine.ball.ownerId !== player.id,
            notes: `command=SHORT_PASS target=${targetId} dir=${passDir.x.toFixed(2)},${passDir.y.toFixed(2)} power=0.75 events=${newEvents.map((e) => e.type).join(',')}`,
          });

          // Observe next 50 ticks
          const observeRng = new SeededRNG(SEED_BASE + ep);
          for (let obs = 1; obs <= 50; obs++) {
            const obsActionMap = new Map<string, AgentAction>();
            for (const p of engine.players) {
              const idx = observeRng.nextInt(0, 18);
              obsActionMap.set(p.id, mapDiscreteAction(idx));
            }
            const obsRes = engine.step(obsActionMap, 1 / 60);
            if (obsRes.terminated || obsRes.truncated) break;

            const nearestNow = nearestTeammate(player.id, engine);
            const ballToNearestNow = ballToNearestTeammate(engine);
            const ownerNow = engine.ball.ownerId;
            const newEvts = engine.events.slice(engine.events.length > 0 ? engine.events.length - 1 : 0);
            const evtType = newEvts.length > 0 ? newEvts[newEvts.length - 1].type : undefined;

            allLogs.push({
              tick: engine.tickCount,
              phase: 'observe',
              event: evtType,
              ballOwnerId: ownerNow,
              ballX: engine.ball.position.x,
              ballY: engine.ball.position.y,
              ballZ: engine.ball.position.z,
              ballVx: engine.ball.velocity.x,
              ballVy: engine.ball.velocity.y,
              ballVz: engine.ball.velocity.z,
              nearestTeammateId: nearestNow.id,
              nearestTeammateDist: nearestNow.dist,
              ballToNearestTeammateDist: ballToNearestNow.dist,
              teammatePositions: teammatePositions(engine),
              actingPlayerId: player.id,
              notes: `obs_tick=${obs}`,
            });
          }
        }

        // Also test SHOT if legal
        if (shotLegal) {
          shotCommanded++;
          const goalDir = Vec2.normalize({ x: 1.0 - player.position.x, y: 0 - player.position.y });
          const shotAction: AgentAction = { type: ActionType.SHOT, direction: goalDir, power: 0.95 };
          const shotMap = new Map<string, AgentAction>();
          shotMap.set(player.id, shotAction);
          for (const p of engine.players) {
            if (p.id !== player.id) shotMap.set(p.id, { type: ActionType.IDLE });
          }
          const beforeShotEvents = engine.events.length;
          engine.step(shotMap, 1 / 60);
          const newShotEvents = engine.events.slice(beforeShotEvents);
          if (newShotEvents.some((e) => e.type === 'shot' && e.team === 'left')) shotFired++;
        }

        break; // one candidate per episode
      }
    }

    if (!found) {
      failReasons['no_valid_state'] = (failReasons['no_valid_state'] || 0) + 1;
    }
  }

  const out = {
    meta: {
      scenario: PRIMARY,
      episodes,
      head: '2230fdc5b578cd135eddb89a9a77e16f2c278479',
      date: new Date().toISOString(),
    },
    summary: {
      validStatesFound,
      passCommanded,
      passCompleted,
      shotCommanded,
      shotFired,
      failReasons,
    },
    logs: allLogs,
  };

  const path = `training/results/forensics/experiment_d_pass_forensics_${Date.now()}.json`;
  fs.mkdirSync('training/results/forensics', { recursive: true });
  fs.writeFileSync(path, JSON.stringify(out, null, 2));
  console.log(`[OUTPUT] Wrote ${path}`);
  console.log(`Valid states: ${validStatesFound} | PASS: ${passCompleted}/${passCommanded} | SHOT: ${shotFired}/${shotCommanded}`);
  console.log(`Fail reasons: ${JSON.stringify(failReasons)}`);
}

if (import.meta.url.endsWith(process.argv[1]) || process.argv[1]?.includes('experiment_d_pass_forensics')) {
  runForensics(20);
}
