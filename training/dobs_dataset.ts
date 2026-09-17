import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ObservationEncoder } from '../src/engine/ObservationEncoder';
import { ActionType, AgentAction } from '../src/types/football';
import { mapDiscreteAction } from '../src/engine/ActionMapping';
import { Vec2 } from '../src/engine/Vector';
import { SeededRNG } from '../src/engine/SeededRNG';
import * as fs from 'fs';

const PRIMARY_SCENARIO = 'academy_3_vs_1_with_keeper';
const SEED_BASE = 1000;
const MAX_TICKS_PER_EPISODE = 600;

interface StateSample {
  scenario_id: string;
  seed: number;
  timestep: number;
  controlled_player_id: string;
  controlled_player_role: string;
  ball_owner_id: string | null;
  ball_position: { x: number; y: number; z: number };
  ball_velocity: { x: number; y: number; z: number };
  self_position: { x: number; y: number };
  self_role: string;
  teammates: Array<{ id: string; role: string; position: { x: number; y: number } }>;
  opponents: Array<{ id: string; role: string; position: { x: number; y: number } }>;
  goal_position: { x: number; y: number };
  raw_observation: number[];
  action_mask: number[];
  events: string[];
  d_self_ball: number;
  d_teammate_ball: number;
  d_opponent_ball: number;
  d_self_goal: number;
  d_teammate_goal: number;
  ball_delta_x: number;
  ball_delta_y: number;
  split: 'train' | 'validation' | 'test';
  source: 'random' | 'forced_left' | 'forced_right' | 'perturb';
}

function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function nearestTeammateBallDist(playerId: string, engine: GameEngine): number {
  const ballPos = engine.ball.position;
  const player = engine.players.find((p) => p.id === playerId);
  if (!player) return Infinity;
  const mates = engine.players.filter((p) => p.team === player.team && p.id !== player.id && !p.isGoalkeeper);
  let best = Infinity;
  for (const m of mates) {
    const d = dist(ballPos, m.position);
    if (d < best) best = d;
  }
  return best;
}

function nearestOpponentBallDist(playerId: string, engine: GameEngine): number {
  const ballPos = engine.ball.position;
  const player = engine.players.find((p) => p.id === playerId);
  if (!player) return Infinity;
  const opponents = engine.players.filter((p) => p.team !== player.team);
  let best = Infinity;
  for (const o of opponents) {
    const d = dist(ballPos, o.position);
    if (d < best) best = d;
  }
  return best;
}

function nearestTeammateGoalDist(playerId: string, engine: GameEngine): number {
  const goalPos = { x: 1.0, y: 0.0 };
  const player = engine.players.find((p) => p.id === playerId);
  if (!player) return Infinity;
  const mates = engine.players.filter((p) => p.team === player.team && p.id !== player.id && !p.isGoalkeeper);
  let best = Infinity;
  for (const m of mates) {
    const d = dist(goalPos, m.position);
    if (d < best) best = d;
  }
  return best;
}

function recordState(engine: GameEngine, split: 'train' | 'validation' | 'test', source: StateSample['source'], prevBall: { x: number; y: number }): StateSample | null {
  const controlledId = engine.controlledPlayerId;
  if (!controlledId) return null;
  const controlledPlayer = engine.players.find((p) => p.id === controlledId);
  if (!controlledPlayer) return null;

  const obs = ObservationEncoder.encode(
    engine.players,
    engine.ball,
    controlledId,
    engine.score,
    engine.tickCount,
    engine.activeScenario ? engine.activeScenario.timeLimitSeconds * 60 : 3600,
    engine.gameMode
  );

  const mask = ObservationEncoder.getActionMask(controlledPlayer, engine);

  const teammates = engine.players
    .filter((p) => p.team === 'left' && p.id !== controlledId && !p.isGoalkeeper)
    .map((p) => ({
      id: p.id,
      role: p.role || 'unknown',
      position: { x: p.position.x, y: p.position.y },
    }));

  const opponents = engine.players
    .filter((p) => p.team !== 'left')
    .map((p) => ({
      id: p.id,
      role: p.role || 'unknown',
      position: { x: p.position.x, y: p.position.y },
    }));

  return {
    scenario_id: engine.activeScenario?.id || PRIMARY_SCENARIO,
    seed: engine.tickCount,
    timestep: engine.tickCount,
    controlled_player_id: controlledId,
    controlled_player_role: controlledPlayer.role || 'unknown',
    ball_owner_id: engine.ball.ownerId,
    ball_position: { x: engine.ball.position.x, y: engine.ball.position.y, z: engine.ball.position.z },
    ball_velocity: { x: engine.ball.velocity.x, y: engine.ball.velocity.y, z: engine.ball.velocity.z },
    self_position: { x: controlledPlayer.position.x, y: controlledPlayer.position.y },
    self_role: controlledPlayer.role || 'unknown',
    teammates,
    opponents,
    goal_position: { x: 1.0, y: 0.0 },
    raw_observation: obs.rawVector,
    action_mask: mask,
    events: engine.events.map((e) => e.type),
    d_self_ball: dist(controlledPlayer.position, engine.ball.position),
    d_teammate_ball: nearestTeammateBallDist(controlledId, engine),
    d_opponent_ball: nearestOpponentBallDist(controlledId, engine),
    d_self_goal: dist(controlledPlayer.position, { x: 1.0, y: 0.0 }),
    d_teammate_goal: nearestTeammateGoalDist(controlledId, engine),
    ball_delta_x: engine.ball.position.x - prevBall.x,
    ball_delta_y: engine.ball.position.y - prevBall.y,
    split,
    source,
  };
}

function randomWalkEpisodes(seeds: number[], split: 'train' | 'validation' | 'test'): StateSample[] {
  const samples: StateSample[] = [];
  for (const seed of seeds) {
    const engine = new GameEngine();
    const scenario = ACADEMY_SCENARIOS.find((s) => s.id === PRIMARY_SCENARIO)!;
    engine.loadScenario(scenario, seed);
    const rng = new SeededRNG(seed);
    let prevBall = { x: engine.ball.position.x, y: engine.ball.position.y };

    for (let tick = 0; tick < MAX_TICKS_PER_EPISODE; tick++) {
      const actionMap = new Map<string, AgentAction>();
      for (const player of engine.players) {
        const idx = rng.nextInt(0, 18);
        actionMap.set(player.id, mapDiscreteAction(idx));
      }
      const res = engine.step(actionMap, 1 / 60);
      const sample = recordState(engine, split, 'random', prevBall);
      if (sample) samples.push(sample);
      prevBall = { x: engine.ball.position.x, y: engine.ball.position.y };
      if (res.terminated || res.truncated) break;
    }
  }
  return samples;
}

function forcedPossessionEpisodes(seeds: number[], split: 'train' | 'validation' | 'test'): StateSample[] {
  const samples: StateSample[] = [];
  for (const seed of seeds) {
    const engine = new GameEngine();
    const scenario = ACADEMY_SCENARIOS.find((s) => s.id === PRIMARY_SCENARIO)!;
    engine.loadScenario(scenario, seed);
    const rng = new SeededRNG(seed + 1000);

    // Stabilize for 30 ticks
    for (let i = 0; i < 30; i++) {
      const actionMap = new Map<string, AgentAction>();
      for (const player of engine.players) {
        const idx = rng.nextInt(0, 18);
        actionMap.set(player.id, mapDiscreteAction(idx));
      }
      engine.step(actionMap, 1 / 60);
    }

    let prevBall = { x: engine.ball.position.x, y: engine.ball.position.y };

    // Force left team possession for 60 ticks
    for (let tick = 0; tick < 60; tick++) {
      const leftPlayers = engine.players.filter((p) => p.team === 'left' && !p.isGoalkeeper);
      if (leftPlayers.length === 0) break;
      const ballCarrier = leftPlayers[0];
      const forcedMap = new Map<string, AgentAction>();
      for (const p of engine.players) {
        if (p.id === ballCarrier.id) {
          forcedMap.set(p.id, { type: ActionType.SPRINT, direction: { x: 1.0, y: 0.0 } });
        } else if (p.team === 'left') {
          const dir = Vec2.normalize({ x: ballCarrier.position.x - p.position.x, y: ballCarrier.position.y - p.position.y });
          forcedMap.set(p.id, { type: ActionType.SPRINT, direction: dir });
        } else {
          const idx = rng.nextInt(0, 18);
          forcedMap.set(p.id, mapDiscreteAction(idx));
        }
      }
      engine.step(forcedMap, 1 / 60);
      const sample = recordState(engine, split, 'forced_left', prevBall);
      if (sample) samples.push(sample);
      prevBall = { x: engine.ball.position.x, y: engine.ball.position.y };
    }

    // Force right team possession for 60 ticks
    for (let tick = 0; tick < 60; tick++) {
      const rightPlayers = engine.players.filter((p) => p.team === 'right' && !p.isGoalkeeper);
      if (rightPlayers.length === 0) break;
      const ballCarrier = rightPlayers[0];
      const forcedMap = new Map<string, AgentAction>();
      for (const p of engine.players) {
        if (p.id === ballCarrier.id) {
          forcedMap.set(p.id, { type: ActionType.SPRINT, direction: { x: -1.0, y: 0.0 } });
        } else if (p.team === 'right') {
          const dir = Vec2.normalize({ x: ballCarrier.position.x - p.position.x, y: ballCarrier.position.y - p.position.y });
          forcedMap.set(p.id, { type: ActionType.SPRINT, direction: dir });
        } else {
          const idx = rng.nextInt(0, 18);
          forcedMap.set(p.id, mapDiscreteAction(idx));
        }
      }
      engine.step(forcedMap, 1 / 60);
      const sample = recordState(engine, split, 'forced_right', prevBall);
      if (sample) samples.push(sample);
      prevBall = { x: engine.ball.position.x, y: engine.ball.position.y };
    }
  }
  return samples;
}

function perturbEpisodes(seeds: number[], split: 'train' | 'validation' | 'test'): StateSample[] {
  const samples: StateSample[] = [];
  for (const seed of seeds) {
    const engine = new GameEngine();
    const scenario = ACADEMY_SCENARIOS.find((s) => s.id === PRIMARY_SCENARIO)!;
    engine.loadScenario(scenario, seed);
    const rng = new SeededRNG(seed + 2000);

    // Stabilize for 30 ticks
    for (let i = 0; i < 30; i++) {
      const actionMap = new Map<string, AgentAction>();
      for (const player of engine.players) {
        const idx = rng.nextInt(0, 18);
        actionMap.set(player.id, mapDiscreteAction(idx));
      }
      engine.step(actionMap, 1 / 60);
    }

    let prevBall = { x: engine.ball.position.x, y: engine.ball.position.y };

    // Perturb ball position and record state
    for (let tick = 0; tick < 60; tick++) {
      const controlledId = engine.controlledPlayerId;
      if (!controlledId) continue;
      const controlledPlayer = engine.players.find((p) => p.id === controlledId);
      if (!controlledPlayer) continue;

      // Move controlled player toward ball with some randomness
      const ballPos = engine.ball.position;
      const dir = Vec2.normalize({ x: ballPos.x - controlledPlayer.position.x, y: ballPos.y - controlledPlayer.position.y });
      const perturbedDir = { x: dir.x + (rng.nextFloat() - 0.5) * 0.5, y: dir.y + (rng.nextFloat() - 0.5) * 0.5 };
      const normalizedDir = Vec2.normalize(perturbedDir);

      const actionMap = new Map<string, AgentAction>();
      actionMap.set(controlledId, { type: ActionType.SPRINT, direction: normalizedDir });
      for (const p of engine.players) {
        if (p.id !== controlledId) {
          const idx = rng.nextInt(0, 18);
          actionMap.set(p.id, mapDiscreteAction(idx));
        }
      }
      engine.step(actionMap, 1 / 60);
      const sample = recordState(engine, split, 'perturb', prevBall);
      if (sample) samples.push(sample);
      prevBall = { x: engine.ball.position.x, y: engine.ball.position.y };
    }
  }
  return samples;
}

function assignSplits(samples: StateSample[]): StateSample[] {
  // Group by source to ensure balanced coverage
  const bySource: Record<string, StateSample[]> = {};
  for (const s of samples) {
    if (!bySource[s.source]) bySource[s.source] = [];
    bySource[s.source].push(s);
  }

  const result: StateSample[] = [];
  for (const source of Object.keys(bySource)) {
    const group = bySource[source];
    const shuffled = [...group].sort(() => Math.random() - 0.5);
    const n = shuffled.length;
    const trainEnd = Math.floor(n * 0.7);
    const valEnd = Math.floor(n * 0.85);
    shuffled.forEach((s, i) => {
      if (i < trainEnd) s.split = 'train';
      else if (i < valEnd) s.split = 'validation';
      else s.split = 'test';
    });
    result.push(...shuffled);
  }
  return result;
}

function main(): void {
  console.log('Building ground-truth dataset...');

  const trainSeeds = [SEED_BASE, SEED_BASE + 10, SEED_BASE + 20];
  const valSeeds = [SEED_BASE + 30];
  const testSeeds = [SEED_BASE + 40];

  const trainSamples = [
    ...randomWalkEpisodes(trainSeeds, 'train'),
    ...forcedPossessionEpisodes(trainSeeds, 'train'),
    ...perturbEpisodes(trainSeeds, 'train'),
  ];

  const valSamples = [
    ...randomWalkEpisodes(valSeeds, 'validation'),
    ...forcedPossessionEpisodes(valSeeds, 'validation'),
    ...perturbEpisodes(valSeeds, 'validation'),
  ];

  const testSamples = [
    ...randomWalkEpisodes(testSeeds, 'test'),
    ...forcedPossessionEpisodes(testSeeds, 'test'),
    ...perturbEpisodes(testSeeds, 'test'),
  ];

  const allSamples = [...trainSamples, ...valSamples, ...testSamples];
  const splitSamples = assignSplits(allSamples);

  const train = splitSamples.filter((s) => s.split === 'train');
  const val = splitSamples.filter((s) => s.split === 'validation');
  const test = splitSamples.filter((s) => s.split === 'test');

  console.log(`Total samples: ${splitSamples.length}`);
  console.log(`  Train: ${train.length}`);
  console.log(`  Validation: ${val.length}`);
  console.log(`  Test: ${test.length}`);

  // Summary statistics
  const ownershipCounts = { none: 0, left: 0, right: 0 };
  const sourceCounts: Record<string, number> = {};
  for (const s of splitSamples) {
    if (!s.ball_owner_id) ownershipCounts.none++;
    else if (s.ball_owner_id.startsWith('left')) ownershipCounts.left++;
    else ownershipCounts.right++;
    sourceCounts[s.source] = (sourceCounts[s.source] || 0) + 1;
  }
  console.log(`\nOwnership distribution:`);
  console.log(`  none: ${ownershipCounts.none} (${(ownershipCounts.none / splitSamples.length * 100).toFixed(1)}%)`);
  console.log(`  left: ${ownershipCounts.left} (${(ownershipCounts.left / splitSamples.length * 100).toFixed(1)}%)`);
  console.log(`  right: ${ownershipCounts.right} (${(ownershipCounts.right / splitSamples.length * 100).toFixed(1)}%)`);
  console.log(`\nSource distribution:`);
  for (const [src, count] of Object.entries(sourceCounts)) {
    console.log(`  ${src}: ${count} (${(count / splitSamples.length * 100).toFixed(1)}%)`);
  }

  const outDir = 'training/results/dobs';
  fs.mkdirSync(outDir, { recursive: true });
  const timestamp = Date.now();
  const path = `${outDir}/dobs_dataset_${timestamp}.json`;
  fs.writeFileSync(path, JSON.stringify(splitSamples, null, 2));
  console.log(`\n[OUTPUT] Wrote ${path}`);
}

if (import.meta.url.endsWith(process.argv[1]) || process.argv[1]?.includes('dobs_dataset')) {
  main();
}
