/**
 * Investigation script: analyze goal-scoring episodes from a trained checkpoint
 * to determine whether goals come from dribbling past defenders, shooting, or
 * other mechanisms.
 *
 * This is a permanent, committed artifact — not a throwaway script.
 * It loads the smoke-test ONNX model, runs deterministic episodes through
 * GameEngine with active rule-based defenders, and logs per-step actions,
 * ball ownership, player positions, and events.
 */

import * as ort from 'onnxruntime-web';
import { GameEngine } from '../src/engine/GameEngine';
import { ObservationEncoder } from '../src/engine/ObservationEncoder';
import { mapDiscreteAction } from '../src/engine/ActionMapping';
import { RuleBasedAgent } from '../src/agents/RuleBasedAgent';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';

const CHECKPOINT_PATH = 'public/models/mappo_policy_smoke_embedded.onnx';
const NUM_EPISODES = 30;
const MAX_TICKS = 1800; // 30 seconds at 60fps

async function main() {
  console.log(`Loading ONNX model from: ${CHECKPOINT_PATH}`);
  const session = await ort.InferenceSession.create(CHECKPOINT_PATH, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  console.log(`✓ Session created. Inputs: ${session.inputNames}, Outputs: ${session.outputNames}`);

  const inputName = session.inputNames[0] || 'obs';
  const outputName = session.outputNames[0] || 'action_logits';

  const scenario = ACADEMY_SCENARIOS.find(s => s.id === 'academy_3_vs_1_with_keeper');
  if (!scenario) {
    throw new Error('Scenario academy_3_vs_1_with_keeper not found');
  }

  const goalEpisodes: Array<{
    seed: number;
    ticks: number;
    finalBallOwner: string | null;
    hadShot: boolean;
    shotTicks: number[];
    goalTick: number;
    preGoalFrames: any[];
    defenderPositionsAtGoal: any[];
    keeperPositionAtGoal: any;
    ballTrajectory: any[];
  }> = [];

  for (let ep = 0; ep < NUM_EPISODES; ep++) {
    const seed = 500000 + ep * 1009;
    const engine = new GameEngine();
    engine.loadScenario(scenario, seed);

    // Create rule-based agents for the right team (defenders)
    const rightAgents: RuleBasedAgent[] = [];
    for (const p of engine.players) {
      if (p.team === 'right') {
        const agent = new RuleBasedAgent(p.id, p.name, 'medium', seed);
        rightAgents.push(agent);
      }
    }

    const controllableAgents = ['left_1', 'left_2', 'left_3'];
    let goalScored = false;
    let goalTick = -1;
    const shotTicks: number[] = [];
    let finalBallOwner: string | null = null;
    const ballTrajectory: any[] = [];

    for (let tick = 0; tick < MAX_TICKS; tick++) {
      const actionMap = new Map<string, any>();

      // Compute actions for ALL players
      for (const player of engine.players) {
        const teammates = engine.players.filter((p) => p.team === player.team);
        const opponents = engine.players.filter((p) => p.team !== player.team);

        let action: any;
        if (player.team === 'left' && controllableAgents.includes(player.id)) {
          // Neural agent for left team controllable players
          const obs = ObservationEncoder.encode(
            engine.players,
            engine.ball,
            engine.controlledPlayerId,
            engine.score,
            engine.tickCount,
            engine.activeScenario ? engine.activeScenario.timeLimitSeconds * 60 : 3600,
            engine.gameMode
          );

          const obsArray = Array.from(obs.rawVector);
          if (obsArray.some(v => isNaN(v))) {
            // Skip this tick for the neural agent if observation contains NaN
            action = { type: 0 }; // IDLE
          } else {
            const tensor = new ort.Tensor('float32', Float32Array.from(obs.rawVector), [1, 127]);
            const results = await session.run({ [inputName]: tensor });
            const logits = results[outputName].data as Float32Array;
            const logitsArray = Array.from(logits);
            const maxLogit = Math.max(...logitsArray);
            const actionIdx = logitsArray.indexOf(maxLogit);
            if (actionIdx < 0 || actionIdx > 18) {
              console.error(`[warn] Invalid action index ${actionIdx} at tick ${tick}, seed ${seed} — using IDLE`);
              action = { type: 0 };
            } else {
              action = mapDiscreteAction(actionIdx);
            }
          }
        } else {
          // Rule-based agent for right team and non-controllable left players
          const ruleAgent = rightAgents.find((a) => a.id === player.id);
          if (ruleAgent) {
            const context: any = {
              player,
              teammates,
              opponents,
              ball: engine.ball,
              allPlayers: engine.players,
              teamSide: player.team,
              controlledPlayerId: engine.controlledPlayerId,
              matchTime: engine.matchTimeSeconds,
              gameMode: engine.gameMode,
              rng: { next: () => engine.rng.next.bind(engine.rng) },
            };
            action = ruleAgent.decide(context);
          } else {
            action = { type: 0 }; // IDLE for non-controllable left players
          }
        }

        actionMap.set(player.id, action);
      }

      const prevBallOwner = engine.ball.ownerId;
      engine.step(actionMap, 1 / 60);

      const currentBallOwner = engine.ball.ownerId;
      finalBallOwner = currentBallOwner;

      ballTrajectory.push({
        tick,
        x: engine.ball.position.x,
        y: engine.ball.position.y,
        ownerId: currentBallOwner,
        velocity: { ...engine.ball.velocity },
      });

      // Track shots from left team controllable agents
      const leftAction = actionMap.get('left_1');
      if (leftAction && leftAction.type === 12) { // ActionType.SHOT = 12
        shotTicks.push(tick);
      }

      // Check for goal event
      const lastEvent = engine.events[engine.events.length - 1];
      if (lastEvent && lastEvent.type === 'goal') {
        goalScored = true;
        goalTick = tick;
        break;
      }

      if (engine.status === 'fulltime') {
        break;
      }
    }

    if (goalScored) {
      // Capture defender and keeper positions at goal moment
      const defenders = engine.players.filter(p => p.team === 'right' && !p.isGoalkeeper);
      const keeper = engine.players.find(p => p.team === 'right' && p.isGoalkeeper);

      const preGoalFrames = engine.replayBuffer.slice(-10);

      goalEpisodes.push({
        seed,
        ticks: goalTick,
        finalBallOwner,
        hadShot: shotTicks.length > 0,
        shotTicks,
        goalTick,
        preGoalFrames,
        defenderPositionsAtGoal: defenders.map(p => ({
          id: p.id,
          position: { ...p.position },
          velocity: { ...p.velocity },
          hasBall: p.hasBall,
        })),
        keeperPositionAtGoal: keeper ? {
          id: keeper.id,
          position: { ...keeper.position },
          velocity: { ...keeper.velocity },
          hasBall: keeper.hasBall,
        } : null,
        ballTrajectory: ballTrajectory.slice(-10),
      });
    }
  }

  console.log(`\n=== GOAL EPISODE INVESTIGATION ===`);
  console.log(`Total episodes: ${NUM_EPISODES}`);
  console.log(`Goals scored: ${goalEpisodes.length}`);
  console.log(`Goal rate: ${(goalEpisodes.length / NUM_EPISODES * 100).toFixed(1)}%`);

  console.log(`\n=== PER-EPISODE BREAKDOWN ===`);
  for (let i = 0; i < goalEpisodes.length; i++) {
    const ep = goalEpisodes[i];
    console.log(`\n--- Goal Episode ${i + 1} (seed ${ep.seed}) ---`);
    console.log(`  Goal at tick: ${ep.goalTick}`);
    console.log(`  Had shot action: ${ep.hadShot} (shot ticks: ${ep.shotTicks.join(', ') || 'none'})`);
    console.log(`  Ball owner at goal: ${ep.finalBallOwner}`);

    console.log(`  Ball trajectory (last 5 ticks before goal):`);
    for (const frame of ep.ballTrajectory.slice(-5)) {
      console.log(`    tick ${frame.tick}: ball=(${frame.x.toFixed(3)}, ${frame.y.toFixed(3)}) owner=${frame.ownerId} vel=(${frame.velocity.x.toFixed(3)}, ${frame.velocity.y.toFixed(3)})`);
    }

    console.log(`  Defenders at goal:`);
    for (const d of ep.defenderPositionsAtGoal) {
      console.log(`    ${d.id}: pos=(${d.position.x.toFixed(3)}, ${d.position.y.toFixed(3)}) vel=(${d.velocity.x.toFixed(3)}, ${d.velocity.y.toFixed(3)}) hasBall=${d.hasBall}`);
    }

    console.log(`  Keeper at goal:`);
    if (ep.keeperPositionAtGoal) {
      const k = ep.keeperPositionAtGoal;
      console.log(`    ${k.id}: pos=(${k.position.x.toFixed(3)}, ${k.position.y.toFixed(3)}) vel=(${k.velocity.x.toFixed(3)}, ${k.velocity.y.toFixed(3)}) hasBall=${k.hasBall}`);
    } else {
      console.log(`    (no keeper found)`);
    }

    // Check if ball was carried into goal
    const goalFrame = ep.preGoalFrames[ep.preGoalFrames.length - 1];
    if (goalFrame && goalFrame.ball) {
      const wasCarried = goalFrame.ball.ownerId !== null;
      console.log(`  Ball carried into goal: ${wasCarried}`);
      if (wasCarried) {
        console.log(`  Ball owner ID: ${goalFrame.ball.ownerId}`);
      }
    }
  }

  // Summary statistics
  const goalsWithShot = goalEpisodes.filter(ep => ep.hadShot).length;
  const goalsWithoutShot = goalEpisodes.filter(ep => !ep.hadShot).length;
  const carriedGoals = goalEpisodes.filter(ep => {
    const goalFrame = ep.preGoalFrames[ep.preGoalFrames.length - 1];
    return goalFrame && goalFrame.ball && goalFrame.ball.ownerId !== null;
  }).length;

  console.log(`\n=== SUMMARY ===`);
  console.log(`Goals with SHOT action: ${goalsWithShot}/${goalEpisodes.length}`);
  console.log(`Goals without SHOT action (dribble/carry): ${goalsWithoutShot}/${goalEpisodes.length}`);
  console.log(`Goals where ball was carried into goal: ${carriedGoals}/${goalEpisodes.length}`);
}

main().catch((err) => {
  console.error('Investigation failed:', err);
  process.exit(1);
});
