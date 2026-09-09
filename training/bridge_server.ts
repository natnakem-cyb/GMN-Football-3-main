import http from 'http';
import path from 'path';
import { pathToFileURL } from 'url';
import { WebSocketServer, WebSocket } from 'ws';
import * as ort from 'onnxruntime-node';
import fs from 'fs';
import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { AgentAction, Player, ScenarioConfig } from '../src/types/football';
import { mapDiscreteAction } from './action_mapping';
import {
  GMN_ENV_VERSION,
  OBSERVATION_SCHEMA_VERSION,
  ACTION_SCHEMA_VERSION,
  OBSERVATION_DIM,
  ACTION_SPACE_SIZE,
  getEventCode,
} from '../src/engine/Contract';
import { Vec2 } from '../src/engine/Vector';
import { RuleBasedAgent } from '../src/agents/RuleBasedAgent';
import { ObservationEncoder } from '../src/engine/ObservationEncoder';

import { CheckpointService } from './CheckpointService';
import { TrainingJobService } from './TrainingJobService';
import { MetricsBroadcaster } from './MetricsBroadcaster';

const PORT = parseInt(process.env.GMN_BRIDGE_PORT || '5050', 10);
const HOST = process.env.GMN_BRIDGE_HOST || '0.0.0.0';

// Singleton broadcaster - shared between bridge and training jobs
export const metricsBroadcaster = new MetricsBroadcaster();
let hardwareStop: (() => void) | null = null;

export class GMNBridgeService {
  public engine: GameEngine;
  private botAgents: Map<string, RuleBasedAgent>;
  /** Per-pool-engine bot agent maps, so each sub-env has independent RNG streams. */
  private poolBotAgents: Map<GameEngine, Map<string, RuleBasedAgent>>;
  /** Per-pool-engine cached ONNX sessions for learned-policy opponents. */
  private poolSnapshotSessions: Map<GameEngine, { session: any; path: string }>;
  /** Current opponent spec: difficulty or snapshot ONNX path. */
  public opponentSpec: { kind: 'difficulty'; difficulty: string } | { kind: 'snapshot'; path: string } | null = null;
  /** Difficulty level used when creating right-team / non-controlled bot agents. */
  public botDifficulty: 'easy' | 'medium' | 'hard' | 'master' = 'medium';
  private scenarioMap: Map<string, ScenarioConfig>;
  public currentScenarioName = 'academy_empty_goal';

  // Batched IPC pool: additional engines for vectorized multi-env stepping.
  private pool: GameEngine[] = [];
  public poolSize = 0;

  // Live RL Training & Telemetry Tracking
  public totalSteps = 0;
  public episodeCount = 0;
  public currentEpisodeReward = 0;
  public currentEpisodeSteps = 0;
  public policyLoss = 0.038;
  public valueLoss = 0.114;
  public entropy = 2.82;
  public approxKl = 0.007;

  constructor() {
    this.engine = new GameEngine();
    this.botAgents = new Map();
    this.poolBotAgents = new Map();
    this.poolSnapshotSessions = new Map();
    this.scenarioMap = new Map();

    ACADEMY_SCENARIOS.forEach((sc) => {
      this.scenarioMap.set(sc.id, sc);
      this.scenarioMap.set(sc.codeName, sc);
    });

    // Default to academy_empty_goal
    const defaultScenario = this.scenarioMap.get('academy_empty_goal');
    if (defaultScenario) {
      this.engine.loadScenario(defaultScenario);
    }
  }

  public ensurePool(size: number) {
    if (this.poolSize >= size) return;
    while (this.pool.length < size) {
      const newEngine = new GameEngine();
      const sc = this.scenarioMap.get(this.currentScenarioName) || this.scenarioMap.get('academy_empty_goal');
      if (sc) {
        newEngine.loadScenario(sc);
      } else {
        newEngine.resetToKickoff(false);
      }
      this.pool.push(newEngine);
      this.poolBotAgents.set(newEngine, new Map());
    }
    this.poolSize = size;
  }

  /** Load (and cache) an ONNX session for the given engine/path. */
  public async getOrCreateSnapshotSession(engine: GameEngine, modelPath: string): Promise<any> {
    const cached = this.poolSnapshotSessions.get(engine);
    if (cached && cached.path === modelPath) {
      return cached.session;
    }
    const resolved = path.isAbsolute(modelPath)
      ? modelPath
      : path.join(process.cwd(), modelPath.replace(/^\//, ''));
    if (!fs.existsSync(resolved)) {
      throw new Error(`[GMN Snapshot] ONNX model not found at: ${resolved}`);
    }
    const modelBuffer = fs.readFileSync(resolved);
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['cpu'],
      graphOptimizationLevel: 'all',
    });
    this.poolSnapshotSessions.set(engine, { session, path: modelPath });
    return session;
  }

  /** Synchronous cache lookup for already-loaded sessions. */
  public getSnapshotSession(engine: GameEngine, modelPath: string): any {
    const cached = this.poolSnapshotSessions.get(engine);
    if (cached && cached.path === modelPath) {
      return cached.session;
    }
    return null;
  }

  /** Mirror a 127-dim left-team observation so a right-team player can use the same model. */
  private static mirrorObservationForRightTeam(obs: number[]): number[] {
    if (obs.length !== OBSERVATION_DIM) return obs;
    const mirrored = new Array(OBSERVATION_DIM);
    // 0..21 <-> 44..65 (positions)
    for (let i = 0; i < 22; i++) {
      mirrored[i] = obs[44 + i];
      mirrored[44 + i] = obs[i];
    }
    // 22..43 <-> 66..87 (velocities)
    for (let i = 0; i < 22; i++) {
      mirrored[22 + i] = obs[66 + i];
      mirrored[66 + i] = obs[22 + i];
    }
    // 88..90 ball position: negate x
    mirrored[88] = -obs[88];
    mirrored[89] = obs[89];
    mirrored[90] = obs[90];
    // 91..93 ball velocity: negate x
    mirrored[91] = -obs[91];
    mirrored[92] = obs[92];
    mirrored[93] = obs[93];
    // 94..96 ball ownership one-hot: flip left/right
    mirrored[94] = obs[94]; // no-one stays
    mirrored[95] = obs[96]; // left <-> right
    mirrored[96] = obs[95]; // right <-> left
    // 97..107 active player: keep as-is (set by caller)
    for (let i = 97; i <= 107; i++) {
      mirrored[i] = obs[i];
    }
    // 108..114 game mode: keep as-is
    for (let i = 108; i <= 114; i++) {
      mirrored[i] = obs[i];
    }
    // 115..126 role: keep as-is (inferPlayerRole already handles right-team mirroring)
    for (let i = 115; i <= 126; i++) {
      mirrored[i] = obs[i];
    }
    return mirrored;
  }

  /** Run ONNX inference for a right-team player and return the discrete action index. */
  private async runOnnxInference(session: any, engine: GameEngine, player: Player): Promise<number> {
    const obs = ObservationEncoder.encode(
      engine.players,
      engine.ball,
      player.id,
      engine.score,
      engine.tickCount,
      engine.activeScenario ? engine.activeScenario.timeLimitSeconds * 60 : 3600,
      engine.gameMode
    );
    const inputObs = OBSERVATION_DIM === obs.rawVector.length
      ? GMNBridgeService.mirrorObservationForRightTeam(obs.rawVector)
      : obs.rawVector;
    const tensor = new ort.Tensor('float32', Float32Array.from(inputObs), [1, OBSERVATION_DIM]);
    const feeds: Record<string, ort.Tensor> = { [session.inputNames[0] || 'obs']: tensor };
    const results = await session.run(feeds);
    const outputName = session.outputNames[0] || 'action_logits';
    const outputTensor = results[outputName] || Object.values(results)[0];
    const logits = (outputTensor.data || outputTensor.cpuData) as Float32Array;
    let bestIdx = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > bestVal) {
        bestVal = logits[i];
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  /** Fallback: apply rule-based bot action for a player. */
  private _applyRuleBasedAction(player: Player, actionMap: Map<string, AgentAction>, engine = this.engine): void {
    if (!this.botAgents.has(player.id)) {
      this.botAgents.set(
        player.id,
        new RuleBasedAgent(`bot_${player.id}`, player.name, this.botDifficulty)
      );
    }
    const bot = this.botAgents.get(player.id)!;
    actionMap.set(
      player.id,
      bot.decide({
        player,
        teammates: engine.players.filter((p) => p.team === player.team),
        opponents: engine.players.filter((p) => p.team !== player.team),
        ball: engine.ball,
        allPlayers: engine.players,
        teamSide: player.team,
        controlledPlayerId: engine.controlledPlayerId,
        matchTime: engine.matchTimeSeconds,
        rng: engine.rng,
      })
    );
  }

  public resetBatch(requests: Array<{ scenario: string; seed?: number }>) {
    this.ensurePool(requests.length);
    const results: any[] = [];
    for (let i = 0; i < requests.length; i++) {
      const req = requests[i];
      const engine = i === 0 ? this.engine : this.pool[i - 1];
      engine.loadScenario(this.scenarioMap.get(req.scenario) || this.scenarioMap.get('academy_empty_goal')!, req.seed);
      const isRondo = engine.activeScenario?.id === 'academy_rondo_4v1';
      const controllableIds = isRondo
        ? engine.players.map((p) => p.id)
        : engine.players.filter((p) => p.team === 'left').map((p) => p.id);
      const perAgentObservations = controllableIds.map((id) =>
        ObservationEncoder.encode(
          engine.players,
          engine.ball,
          id,
          engine.score,
          engine.tickCount,
          engine.activeScenario ? engine.activeScenario.timeLimitSeconds * 60 : 3600,
          engine.gameMode
        ).rawVector
      );
      results.push({
        observation: engine.getObservation().rawVector,
        observations: perAgentObservations,
        info: {
          score: { ...engine.score },
          controllableAgentIds: controllableIds,
          controlledPlayerId: this.engine.controlledPlayerId,
        },
      });
    }
    return results;
  }

  private computeActionMasks(engine: GameEngine, controllableIds: string[]): number[][] {
    return controllableIds.map((id) => {
      const player = engine.players.find((p) => p.id === id);
      if (!player) {
        return new Array(19).fill(0);
      }
      return ObservationEncoder.getActionMask(player, engine);
    });
  }

  public async stepBatch(actionSets: Array<{ actions: number[]; controllableIds: string[] }>): Promise<any> {
    this.ensurePool(actionSets.length);
    const results: any[] = [];
    for (let i = 0; i < actionSets.length; i++) {
      const { actions, controllableIds } = actionSets[i];
      const engine = i === 0 ? this.engine : this.pool[i - 1];
      const isRondoScenario = engine.activeScenario?.id === 'academy_rondo_4v1';
      const expectedControllableIds = isRondoScenario
        ? engine.players.map((p) => p.id)
        : engine.players.filter((p) => p.team === 'left').map((p) => p.id);
      if (actions.length !== expectedControllableIds.length) {
        throw new Error(`[GMN Batch] Env ${i}: expected ${expectedControllableIds.length} actions, got ${actions.length}`);
      }
      const actionMap = new Map<string, AgentAction>();
      controllableIds.forEach((id, idx) => {
        actionMap.set(id, mapDiscreteAction(actions[idx]));
      });
      if (!isRondoScenario) {
        await Promise.all(
          engine.players.map(async (player) => {
            if (controllableIds.includes(player.id)) return;

            if (this.opponentSpec?.kind === 'snapshot' && player.team === 'right') {
              let session = this.getSnapshotSession(engine, this.opponentSpec.path);
              try {
              } catch (e) {}
              if (!session) {
                this._applyRuleBasedAction(player, actionMap, engine);
                return;
              }
            try {
              const actionIdx = await this.runOnnxInference(session, this.engine, player);
              actionMap.set(player.id, mapDiscreteAction(actionIdx));
              try {
              } catch (e) {}
            } catch (err: any) {
              try {
              } catch (e) {}
              console.warn(`[GMN Snapshot] ONNX inference failed for ${player.id}, falling back to rule-based: ${err.message}`);
              this._applyRuleBasedAction(player, actionMap);
            }
              return;
            }

            this._applyRuleBasedAction(player, actionMap, engine);
          })
        );
      }
      const stepResult = engine.step(actionMap, 1 / 60);
      const observations = controllableIds.map((id) =>
        ObservationEncoder.encode(
          engine.players,
          engine.ball,
          id,
          engine.score,
          engine.tickCount,
          engine.activeScenario ? engine.activeScenario.timeLimitSeconds * 60 : 3600,
          engine.gameMode
        ).rawVector
      );
      const actionMasks = this.computeActionMasks(engine, controllableIds);
      results.push({
        observations,
        action_masks: actionMasks,
        reward: stepResult.reward,
        terminated: stepResult.terminated,
        truncated: stepResult.truncated,
        info: {
          score: stepResult.info.score,
          event: stepResult.info.event,
          checkpointReward: stepResult.info.checkpointReward,
          ballDistanceToGoal: stepResult.info.ballDistanceToGoal,
          ground_truth: {
            possession_left_pct: engine.stats.possession.left,
            completed_passes_left: engine.stats.completedPasses.left,
            attempted_passes_left: engine.stats.passes.left,
            shots_on_target_left: engine.stats.shotsOnTarget.left,
            total_shots_left: engine.stats.shots.left,
            current_ball_owner: engine.ball.ownerId != null
              ? (() => {
                  const owner = engine.players.find((p) => p.id === engine.ball.ownerId);
                  return owner ? { agent_id: owner.id, team: owner.team } : null;
                })()
              : null,
          },
        },
        controllableIds,
        defenderReward: engine.getActiveScenarioHandler()?.getLastDefenderReward?.() ?? 0,
        isRondo: engine.activeScenario?.id === 'academy_rondo_4v1',
      });
    }
    return results;
  }

  public reset(scenarioName = 'academy_empty_goal', seed?: number) {
    this.currentScenarioName = scenarioName;
    this.currentEpisodeReward = 0;
    this.currentEpisodeSteps = 0;
    this.episodeCount++;

    const sc = this.scenarioMap.get(scenarioName) || this.scenarioMap.get('academy_empty_goal');
    if (sc) {
      this.engine.loadScenario(sc, seed);
    } else {
      this.engine.resetToKickoff(false, seed);
    }

    // Reset bot states
    this.botAgents.clear();

    // Pure initial observation without stepping physics
    const initialObs = this.engine.getObservation();

    const isRondoScenario = this.engine.activeScenario?.id === 'academy_rondo_4v1';
    const controllableAgentIds = isRondoScenario
      ? this.engine.players.map((p) => p.id)
      : this.engine.players
          .filter((p) => p.team === 'left')
          .map((p) => p.id);

    const perAgentObservations = controllableAgentIds.map((id) =>
      ObservationEncoder.encode(
        this.engine.players,
        this.engine.ball,
        id,
        this.engine.score,
        this.engine.tickCount,
        this.engine.activeScenario ? this.engine.activeScenario.timeLimitSeconds * 60 : 3600,
        this.engine.gameMode
      ).rawVector
    );

    return {
      observation: initialObs.rawVector,
      observations: perAgentObservations,
      info: {
        score: { ...this.engine.score },
        ballDistanceToGoal: Vec2.distance(
          { x: this.engine.ball.position.x, y: this.engine.ball.position.y },
          { x: 1.0, y: 0 }
        ),
        scenario: sc?.codeName || 'free_play',
        controlledPlayerId: this.engine.controlledPlayerId,
        controllableAgentIds,
      },
    };
  }

  public async step(actionIdx: number): Promise<any> {
    const actionMap = new Map<string, AgentAction>();

    // 1. Controlled player action from RL agent
    const controlledPlayer = this.engine.players.find(
      (p) => p.id === this.engine.controlledPlayerId
    ) || this.engine.players.find((p) => p.team === 'left');

    if (controlledPlayer) {
      const mappedAction = mapDiscreteAction(actionIdx);
      actionMap.set(controlledPlayer.id, mappedAction);
    }

    // 2. Automated bots for other players (if any)
    await Promise.all(
      this.engine.players.map(async (player) => {
        if (player.id === controlledPlayer?.id) return;

        if (this.opponentSpec?.kind === 'snapshot') {
          if (player.team === 'right') {
            let session = this.getSnapshotSession(this.engine, this.opponentSpec.path);
            try {
            } catch (e) {}
            if (!session) {
              this._applyRuleBasedAction(player, actionMap);
              return;
            }
            try {
              const actionIdx = await this.runOnnxInference(session, this.engine, player);
              actionMap.set(player.id, mapDiscreteAction(actionIdx));
              try {
              } catch (e) {}
            } catch (err: any) {
              try {
              } catch (e) {}
              console.warn(`[GMN Snapshot] ONNX inference failed for ${player.id}, falling back to rule-based: ${err.message}`);
              this._applyRuleBasedAction(player, actionMap);
            }
            return;
          }
        }

        this._applyRuleBasedAction(player, actionMap);
      })
    );

    // 3. Execute deterministic physics tick (1/60s)
    const result = this.engine.step(actionMap, 1 / 60);

    // 4. Compute action masks for the next decision step (post-possession state).
    const controllableIds = [this.engine.controlledPlayerId].filter((id): id is string => id != null);
    const actionMasks = this.computeActionMasks(this.engine, controllableIds);
    const actionMask = actionMasks[0] ?? new Array(19).fill(0);

    return {
      observation: result.observation.rawVector,
      action_mask: actionMask,
      reward: result.reward,
      terminated: result.terminated,
      truncated: result.truncated,
      info: {
        score: result.info.score,
        event: result.info.event,
        checkpointReward: result.info.checkpointReward,
        ballDistanceToGoal: result.info.ballDistanceToGoal,
        ground_truth: {
          possession_left_pct: this.engine.stats.possession.left,
          completed_passes_left: this.engine.stats.completedPasses.left,
          attempted_passes_left: this.engine.stats.passes.left,
          shots_on_target_left: this.engine.stats.shotsOnTarget.left,
          total_shots_left: this.engine.stats.shots.left,
          current_ball_owner: this.engine.ball.ownerId != null
            ? (() => {
                const owner = this.engine.players.find((p) => p.id === this.engine.ball.ownerId);
                return owner
                  ? { agent_id: owner.id, team: owner.team }
                  : null;
              })()
            : null,
        },
      },
    };
  }

  public async stepMulti(actionIndices: number[]): Promise<any> {
    const isRondoScenario = this.engine.activeScenario?.id === 'academy_rondo_4v1';
    const controllableIds = isRondoScenario
      ? this.engine.players.map((p) => p.id)
      : this.engine.players
          .filter((p) => p.team === 'left')
          .map((p) => p.id);

    if (actionIndices.length !== controllableIds.length) {
      throw new Error(
        `[GMN Multi-Agent] Expected ${controllableIds.length} actions, got ${actionIndices.length}`
      );
    }

    const actionMap = new Map<string, AgentAction>();

    // 1. Controlled agents (all players for rondo, left team otherwise), in fixed order
    controllableIds.forEach((id, i) => {
      actionMap.set(id, mapDiscreteAction(actionIndices[i]));
    });

    // 2. Automated bots for other players (if any) — skipped for rondo
    if (!isRondoScenario) {
      await Promise.all(
        this.engine.players.map(async (player) => {
          if (controllableIds.includes(player.id)) return;

          if (this.opponentSpec?.kind === 'snapshot' && player.team === 'right') {
            let session = this.getSnapshotSession(this.engine, this.opponentSpec.path);
            try {
            } catch (e) {}
            if (!session) {
              // Session not yet loaded; fall back to rule-based for this tick.
              // The /opponent endpoint pre-loads the session, so this path is
              // only hit if the bridge was restarted or the spec changed without
              // going through /opponent.
              this._applyRuleBasedAction(player, actionMap);
              return;
            }
            try {
              const actionIdx = await this.runOnnxInference(session, this.engine, player);
              actionMap.set(player.id, mapDiscreteAction(actionIdx));
            } catch (err: any) {
              console.warn(`[GMN Snapshot] ONNX inference failed for ${player.id}, falling back to rule-based: ${err.message}`);
              this._applyRuleBasedAction(player, actionMap);
            }
            return;
          }

          this._applyRuleBasedAction(player, actionMap);
        })
      );
    }

    // 3. Execute deterministic physics tick (1/60s)
    const result = this.engine.step(actionMap, 1 / 60);

    // 4. Re-encode one observation per controlled agent from the
    // already-updated post-step state — do not step the engine again
    const observations = controllableIds.map((id) =>
      ObservationEncoder.encode(
        this.engine.players,
        this.engine.ball,
        id,
        this.engine.score,
        this.engine.tickCount,
        this.engine.activeScenario ? this.engine.activeScenario.timeLimitSeconds * 60 : 3600,
        this.engine.gameMode
      ).rawVector
    );

    const actionMasks = this.computeActionMasks(this.engine, controllableIds);

    return {
      reward: result.reward,
      terminated: result.terminated,
      truncated: result.truncated,
      info: {
        score: result.info.score,
        event: result.info.event,
        checkpointReward: result.info.checkpointReward,
        ballDistanceToGoal: result.info.ballDistanceToGoal,
        ground_truth: {
          possession_left_pct: this.engine.stats.possession.left,
          completed_passes_left: this.engine.stats.completedPasses.left,
          attempted_passes_left: this.engine.stats.passes.left,
          shots_on_target_left: this.engine.stats.shotsOnTarget.left,
          total_shots_left: this.engine.stats.shots.left,
          current_ball_owner: this.engine.ball.ownerId != null
            ? (() => {
                const owner = this.engine.players.find((p) => p.id === this.engine.ball.ownerId);
                return owner
                  ? { agent_id: owner.id, team: owner.team }
                  : null;
              })()
            : null,
        },
      },
      observations, // array, same order as controllableIds
      action_masks: actionMasks,
      controllableIds,
    };
  }

  public getInfo() {
    return {
      status: 'ok',
      environment: 'GMN-Football-3',
      environment_version: GMN_ENV_VERSION,
      observation_dim: OBSERVATION_DIM,
      observation_schema_version: OBSERVATION_SCHEMA_VERSION,
      action_space_size: ACTION_SPACE_SIZE,
      action_schema_version: ACTION_SCHEMA_VERSION,
      scenario: this.engine.activeScenario?.codeName || 'none',
      controlledPlayerId: this.engine.controlledPlayerId,
      scenarios: Array.from(new Set(ACADEMY_SCENARIOS.map((s) => s.codeName))),
    };
  }
}

// Instantiate Service
const bridge = new GMNBridgeService();

// Create Lightweight HTTP Server
const server = http.createServer((req, res) => {
  // CORS & JSON Headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Content-Type', 'application/json');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  let body = '';
  req.on('data', (chunk) => {
    body += chunk;
  });

  req.on('end', async () => {
    try {
      const parsedBody = body ? JSON.parse(body) : {};

      const urlPath = (req.url || '').split('?')[0];

      if (req.method === 'GET' && (urlPath === '/' || urlPath === '/info' || urlPath === '/health')) {
        res.writeHead(200);
        res.end(JSON.stringify(bridge.getInfo()));
        return;
      }

      // --- TRAINING JOB MANAGEMENT API ---
      if (req.method === 'GET' && urlPath === '/api/training/status') {
        res.writeHead(200);
        res.end(JSON.stringify(TrainingJobService.getStatus()));
        return;
      }

      if (req.method === 'POST' && urlPath === '/api/training/start') {
        try {
          const job = TrainingJobService.startJob({
            algorithm: parsedBody.algorithm || 'mappo',
            scenario: parsedBody.scenario || 'academy_3_vs_1_with_keeper',
            timesteps: parseInt(parsedBody.timesteps, 10) || 1000,
            resumeFrom: parsedBody.resumeFrom || undefined,
          });
          res.writeHead(200);
          res.end(JSON.stringify({ success: true, message: 'Training job started', job }));
        } catch (err: any) {
          res.writeHead(400);
          res.end(JSON.stringify({ success: false, error: err.message || String(err) }));
        }
        return;
      }

      if (req.method === 'POST' && urlPath === '/api/training/stop') {
        const stopped = TrainingJobService.stopJob();
        res.writeHead(200);
        res.end(JSON.stringify({ success: stopped, message: stopped ? 'Training stopped' : 'No job was running' }));
        return;
      }

      if (req.method === 'POST' && urlPath === '/api/training/export') {
        const scenario = parsedBody.scenario || 'academy_3_vs_1_with_keeper';
        const algorithm = parsedBody.algorithm || 'MAPPO';

        TrainingJobService.handleAutomaticExport({
          id: `export_${Date.now()}`,
          config: { algorithm: algorithm.toLowerCase() as any, scenario, timesteps: 0 },
          status: 'running',
          startTime: Date.now(),
          currentStep: 0,
          totalSteps: 0,
          latestMetrics: null,
          recentLogs: [],
        })
          .then((result) => {
            res.writeHead(200);
            res.end(JSON.stringify({ success: true, result }));
          })
          .catch((err) => {
            res.writeHead(500);
            res.end(JSON.stringify({ success: false, error: err.message || String(err) }));
          });
        return;
      }

      // --- CHECKPOINT FILE MANAGEMENT API ---
      if (req.method === 'GET' && urlPath === '/api/checkpoints') {
        const checkpoints = CheckpointService.listCheckpoints();
        res.writeHead(200);
        res.end(JSON.stringify({ checkpoints }));
        return;
      }

      if ((req.method === 'POST' && urlPath === '/api/checkpoints/delete') || (req.method === 'DELETE' && urlPath.startsWith('/api/checkpoints/'))) {
        const filename = parsedBody.filename || path.basename(urlPath);
        const deleteSourcePt = !!parsedBody.deleteSourcePt;
        try {
          const result = CheckpointService.deleteCheckpoint(filename, deleteSourcePt);
          res.writeHead(200);
          res.end(JSON.stringify(result));
        } catch (err: any) {
          res.writeHead(400);
          res.end(JSON.stringify({ success: false, error: err.message || String(err) }));
        }
        return;
      }

      if (req.method === 'POST' && urlPath === '/api/checkpoints/upload') {
        try {
          const filename = parsedBody.filename || `custom_model_${Date.now()}.onnx`;
          const base64 = parsedBody.base64Data || '';
          if (!base64) {
            res.writeHead(400);
            res.end(JSON.stringify({ success: false, error: 'No base64Data provided' }));
            return;
          }
          const buf = Buffer.from(base64, 'base64');
          const saved = CheckpointService.saveUploadedCheckpoint(filename, buf, {
            scenario: parsedBody.scenario,
            algorithm: parsedBody.algorithm,
            timesteps: parsedBody.timesteps,
          });
          res.writeHead(200);
          res.end(JSON.stringify({ success: true, checkpoint: saved }));
        } catch (err: any) {
          res.writeHead(400);
          res.end(JSON.stringify({ success: false, error: err.message || String(err) }));
        }
        return;
      }

      if (req.method === 'POST' && req.url === '/reset') {
        const resetResult = bridge.reset(parsedBody.scenario, parsedBody.seed);
        res.writeHead(200);
        res.end(JSON.stringify(resetResult));
        return;
      }

      if (req.method === 'POST' && req.url === '/step') {
        if (typeof parsedBody.action !== 'number' || !Number.isInteger(parsedBody.action)) {
          res.writeHead(400);
          res.end(JSON.stringify({ error: `Invalid action: ${parsedBody.action}. Must be integer in [0, ${ACTION_SPACE_SIZE - 1}].` }));
          return;
        }
        const stepResult = await bridge.step(parsedBody.action);
        res.writeHead(200);
        res.end(JSON.stringify(stepResult));
        return;
      }

      if (req.method === 'POST' && req.url === '/step_multi') {
        if (!Array.isArray(parsedBody.actions)) {
          res.writeHead(400);
          res.end(JSON.stringify({ error: 'actions must be an array of integers' }));
          return;
        }
        const multiResult = await bridge.stepMulti(parsedBody.actions);
        res.writeHead(200);
        res.end(JSON.stringify(multiResult));
        return;
      }

      if (req.method === 'POST' && req.url === '/opponent') {
        // Opponent-pool integration: switch the difficulty of rule-based bot
        // agents or load a learned ONNX snapshot for the right team.
        const kind = parsedBody.kind || 'difficulty';
        if (kind === 'snapshot') {
          const modelPath = parsedBody.path;
          if (!modelPath || typeof modelPath !== 'string') {
            res.writeHead(400);
            res.end(JSON.stringify({ error: "snapshot opponent requires 'path' to ONNX model" }));
            return;
          }
          try {
            bridge.opponentSpec = { kind: 'snapshot', path: modelPath };
            // Pre-load and cache the session for the main engine so the first
            // step does not pay the file-read + session-create cost.
            await bridge.getOrCreateSnapshotSession(bridge.engine, modelPath);
            res.writeHead(200);
            res.end(JSON.stringify({ status: 'ok', kind: 'snapshot', path: modelPath }));
          } catch (err: any) {
            res.writeHead(400);
            res.end(JSON.stringify({ success: false, error: `Failed to load ONNX snapshot: ${err.message}` }));
          }
          return;
        }

        // Default: rule-based difficulty
        const d = parsedBody.difficulty || parsedBody.botDifficulty;
        if (!['easy', 'medium', 'hard', 'master'].includes(d)) {
          res.writeHead(400);
          res.end(JSON.stringify({ error: "difficulty must be one of 'easy'|'medium'|'hard'|'master'" }));
          return;
        }
        bridge.botDifficulty = d;
        bridge.opponentSpec = { kind: 'difficulty', difficulty: d };
        // Clear existing bots so the new difficulty takes effect immediately.
        bridge.reset(bridge.currentScenarioName);
        res.writeHead(200);
        res.end(JSON.stringify({ status: 'ok', kind: 'difficulty', difficulty: d }));
        return;
      }

      if (req.method === 'POST' && req.url === '/close') {
        res.writeHead(200);
        res.end(JSON.stringify({ status: 'closing' }));
        server.close(() => {
          process.exit(0);
        });
        return;
      }

      res.writeHead(404);
      res.end(JSON.stringify({ error: 'Endpoint not found' }));
    } catch (err: any) {
      res.writeHead(500);
      res.end(JSON.stringify({ error: err.message || 'Internal error' }));
    }
  });
});

// Binary step-response layout: (18 + OBSERVATION_DIM * 4) bytes total (530B for 127-float obs), all little-endian
// For academy_rondo_4v1, the header is extended to 22 bytes to carry a second float32
// (defenderReward) at offset 20 so the Python client can assign team-specific rewards.
// Offset 0 (4B float32): reward (attacker reward for rondo; single shared reward otherwise)
// Offset 4 (1B uint8): terminated (0/1)
// Offset 5 (1B uint8): truncated (0/1)
// Offset 6 (1B uint8): scoreLeft
// Offset 7 (1B uint8): scoreRight
// Offset 8 (4B float32): checkpointReward
// Offset 12 (4B float32): ballDistanceToGoal
// Offset 16 (1B uint8): eventCode
// Offset 17 (1B uint8): ballOwnerAgentId (0 = controlled player owns ball, 255 = no controllable owner)
// Offset 18-19 (rondo only): unused padding (0)
// Offset 20 (4B float32, rondo only): defenderReward
// Offset 18/22 ((OBSERVATION_DIM * 4) B): OBSERVATION_DIM * float32 observation
// Offset 18/22 + OBSERVATION_DIM*4 (19 B): action mask (one uint8 per discrete action, 1=valid 0=invalid)
const MASK_BYTES = 19;
export function encodeStepBinary(stepResult: any, ballOwnerAgentIdx: number, isRondo = false, defenderReward = 0, actionMask?: number[]): Buffer {
  const obsBytes = OBSERVATION_DIM * 4;
  const headerSize = isRondo ? 22 : 18;
  const buf = Buffer.allocUnsafe(headerSize + obsBytes + MASK_BYTES);
  buf.writeFloatLE(stepResult.reward || 0.0, 0);
  buf.writeUInt8(stepResult.terminated ? 1 : 0, 4);
  buf.writeUInt8(stepResult.truncated ? 1 : 0, 5);
  buf.writeUInt8(Math.max(0, Math.min(255, stepResult.info.score?.left ?? 0)), 6);
  buf.writeUInt8(Math.max(0, Math.min(255, stepResult.info.score?.right ?? 0)), 7);
  buf.writeFloatLE(stepResult.info.checkpointReward ?? 0.0, 8);
  buf.writeFloatLE(stepResult.info.ballDistanceToGoal ?? 0.0, 12);

  const eventType = typeof stepResult.info.event === 'string' ? stepResult.info.event : (stepResult.info.event as any)?.type;
  const eventCode = getEventCode(eventType);
  buf.writeUInt8(eventCode, 16);

  buf.writeUInt8(ballOwnerAgentIdx, 17);

  if (isRondo) {
    buf.writeFloatLE(defenderReward, 20);
  }

  const obs = stepResult.observation;
  for (let i = 0; i < OBSERVATION_DIM; i++) {
    buf.writeFloatLE(obs[i] ?? 0.0, headerSize + i * 4);
  }

  const mask = actionMask ?? stepResult.action_mask;
  if (mask && mask.length === 19) {
    const maskOffset = headerSize + obsBytes;
    for (let i = 0; i < 19; i++) {
      buf.writeUInt8(mask[i] ? 1 : 0, maskOffset + i);
    }
  } else {
    const maskOffset = headerSize + obsBytes;
    for (let i = 0; i < 19; i++) {
      buf.writeUInt8(1, maskOffset + i);
    }
  }

  return buf;
}

// Multi-Agent Binary step-response layout: 18 + (OBSERVATION_DIM * 4) * N bytes total, all little-endian
//   Offset 0 (4B float32): reward (shared team reward; attacker reward for rondo)
//   Offset 4 (1B uint8): terminated (0/1)
//   Offset 5 (1B uint8): truncated (0/1)
//   Offset 6 (1B uint8): scoreLeft
//   Offset 7 (1B uint8): scoreRight
//   Offset 8 (4B float32): checkpointReward
//   Offset 12 (4B float32): ballDistanceToGoal
//   Offset 16 (1B uint8): eventCode
//   Offset 17 (1B uint8): ballOwnerAgentId (0..N-1 index into controllableAgentIds, 255 = no controllable owner)
//   Offset 18-19 (rondo only): unused padding (0)
//   Offset 20 (4B float32, rondo only): defenderReward
//   Offset 18/22 ((OBSERVATION_DIM * 4) * N B): N observations, OBSERVATION_DIM * float32 each, in controllableAgentIds order
//   Offset 18/22 + N*OBSERVATION_DIM*4 (N * 19 B): N action masks, 19 uint8s each, in controllableAgentIds order
export function encodeMultiStepBinary(multiResult: any, isRondo = false, defenderReward = 0): Buffer {
  const N = multiResult.observations.length;
  const obsBytes = OBSERVATION_DIM * 4;
  const headerSize = isRondo ? 22 : 18;
  const maskBytes = 19;
  const buf = Buffer.allocUnsafe(headerSize + obsBytes * N + maskBytes * N);
  buf.writeFloatLE(multiResult.reward || 0.0, 0);
  buf.writeUInt8(multiResult.terminated ? 1 : 0, 4);
  buf.writeUInt8(multiResult.truncated ? 1 : 0, 5);
  buf.writeUInt8(Math.max(0, Math.min(255, multiResult.info.score?.left ?? 0)), 6);
  buf.writeUInt8(Math.max(0, Math.min(255, multiResult.info.score?.right ?? 0)), 7);
  buf.writeFloatLE(multiResult.info.checkpointReward ?? 0.0, 8);
  buf.writeFloatLE(multiResult.info.ballDistanceToGoal ?? 0.0, 12);

  const eventType = typeof multiResult.info.event === 'string' ? multiResult.info.event : (multiResult.info.event as any)?.type;
  const eventCode = getEventCode(eventType);
  buf.writeUInt8(eventCode, 16);

  const ownerId = multiResult.info.ground_truth?.current_ball_owner?.agent_id;
  const controllableIds = multiResult.controllableIds || [];
  const ballOwnerAgentIdx = ownerId ? controllableIds.indexOf(ownerId) : 255;
  buf.writeUInt8(ballOwnerAgentIdx >= 0 ? ballOwnerAgentIdx : 255, 17);

  if (isRondo) {
    buf.writeFloatLE(defenderReward, 20);
  }

  for (let agentIdx = 0; agentIdx < N; agentIdx++) {
    const obs = multiResult.observations[agentIdx];
    const obsOffset = headerSize + agentIdx * obsBytes;
    for (let i = 0; i < OBSERVATION_DIM; i++) {
      buf.writeFloatLE(obs[i] ?? 0.0, obsOffset + i * 4);
    }
    const mask = multiResult.action_masks?.[agentIdx];
    const maskOffset = headerSize + N * obsBytes + agentIdx * maskBytes;
    if (mask && mask.length === 19) {
      for (let i = 0; i < 19; i++) {
        buf.writeUInt8(mask[i] ? 1 : 0, maskOffset + i);
      }
    } else {
      for (let i = 0; i < 19; i++) {
        buf.writeUInt8(1, maskOffset + i);
      }
    }
  }

  return buf;
}

/**
 * Batched Binary step-response layout:
 *   Header: [B (1 byte)] [N (1 byte)] where B = env count, N = agents per env
 *   Body:   B concatenated frames. For non-rondo each frame is 18 + N*OBSERVATION_DIM*4 bytes.
 *           For academy_rondo_4v1 each frame is 22 + N*OBSERVATION_DIM*4 bytes
 *           (extra 4B float32 at offset 20 carries the defender-specific reward).
 *           Frame i starts at offset header_size + i * frame_size.
 *           Each frame matches encodeMultiStepBinary layout.
 */
export function encodeBatchedStepBinary(results: any[]): Buffer {
  const B = results.length;
  if (B === 0) return Buffer.allocUnsafe(0);
  const N = results[0].observations.length;
  const obsBytes = OBSERVATION_DIM * 4;
  const maskBytes = 19;
  const anyRondo = results.some((r) => r.isRondo);
  const headerSize = anyRondo ? 22 : 18;
  const frameSize = headerSize + obsBytes * N + maskBytes * N;
  const buf = Buffer.allocUnsafe(2 + B * frameSize);
  buf.writeUInt8(B, 0);
  buf.writeUInt8(N, 1);

  for (let envIdx = 0; envIdx < B; envIdx++) {
    const multiResult = results[envIdx];
    const baseOffset = 2 + envIdx * frameSize;
    const defenderReward = multiResult.defenderReward || 0;
    const isRondo = multiResult.isRondo || false;
    buf.writeFloatLE(multiResult.reward || 0.0, baseOffset + 0);
    buf.writeUInt8(multiResult.terminated ? 1 : 0, baseOffset + 4);
    buf.writeUInt8(multiResult.truncated ? 1 : 0, baseOffset + 5);
    buf.writeUInt8(Math.max(0, Math.min(255, multiResult.info.score?.left ?? 0)), baseOffset + 6);
    buf.writeUInt8(Math.max(0, Math.min(255, multiResult.info.score?.right ?? 0)), baseOffset + 7);
    buf.writeFloatLE(multiResult.info.checkpointReward ?? 0.0, baseOffset + 8);
    buf.writeFloatLE(multiResult.info.ballDistanceToGoal ?? 0.0, baseOffset + 12);

    const eventType = typeof multiResult.info.event === 'string' ? multiResult.info.event : (multiResult.info.event as any)?.type;
    const eventCode = getEventCode(eventType);
    buf.writeUInt8(eventCode, baseOffset + 16);

    const ownerId = multiResult.info.ground_truth?.current_ball_owner?.agent_id;
    const controllableIds = multiResult.controllableIds || [];
    const ballOwnerAgentIdx = ownerId ? controllableIds.indexOf(ownerId) : 255;
    buf.writeUInt8(ballOwnerAgentIdx >= 0 ? ballOwnerAgentIdx : 255, baseOffset + 17);

    if (isRondo) {
      buf.writeFloatLE(defenderReward, baseOffset + 20);
    }

    for (let agentIdx = 0; agentIdx < N; agentIdx++) {
      const obs = multiResult.observations[agentIdx];
      const obsOffset = baseOffset + headerSize + agentIdx * obsBytes;
      for (let i = 0; i < OBSERVATION_DIM; i++) {
        buf.writeFloatLE(obs[i] ?? 0.0, obsOffset + i * 4);
      }
      const mask = multiResult.action_masks?.[agentIdx];
      const maskOffset = baseOffset + headerSize + N * obsBytes + agentIdx * maskBytes;
      if (mask && mask.length === 19) {
        for (let i = 0; i < 19; i++) {
          buf.writeUInt8(mask[i] ? 1 : 0, maskOffset + i);
        }
      } else {
        for (let i = 0; i < 19; i++) {
          buf.writeUInt8(1, maskOffset + i);
        }
      }
    }
  }

  return buf;
}

/**
 * Deterministic binary error frame (P0 #5): same layout/length as a normal
 * step frame so Python clients can decode it without hanging. Carries a
 * sentinel reward of -999.0 and terminated=true.
 *   nAgents = 1 -> single-agent frame length (18 + 127*4 + 19) non-rondo / (22 + 127*4 + 19) rondo
 *   nAgents = N -> multi-agent frame length (18 + N*127*4 + N*19) non-rondo / (22 + N*127*4 + N*19) rondo
 */
export function encodeErrorStepBinary(nAgents = 1, isRondo = false): Buffer {
  const obsBytes = OBSERVATION_DIM * 4;
  const maskBytes = 19;
  const headerSize = isRondo ? 22 : 18;
  const buf = Buffer.allocUnsafe(headerSize + obsBytes * Math.max(1, nAgents) + maskBytes * Math.max(1, nAgents));
  buf.fill(0);
  buf.writeFloatLE(-999.0, 0); // sentinel reward: error indicator
  buf.writeUInt8(1, 4);        // terminated = true
  buf.writeUInt8(0, 17);       // ballOwnerAgentId = 0 (no owner on error)
  if (isRondo) {
    buf.writeFloatLE(0.0, 20); // defenderReward = 0 on error
  }
  // Observations and masks are already zero-filled by buf.fill(0), which is
  // a safe all-invalid mask for the error frame.
  return buf;
}

/**
 * Decompose a step result's scalar reward into components for debug/evaluation.
 * This is an instrumentation-only helper; it does not affect production behavior.
 */
function computeRewardComponents(stepResult: any): any {
  const components: any = {
    total: stepResult.reward || 0,
    goal: 0,
    progress: stepResult.info?.checkpointReward || 0,
    shot: 0,
    pass_shaper: 0,
    defensive: 0,
    other: 0,
  };

  const event = typeof stepResult.info?.event === 'string'
    ? stepResult.info.event
    : stepResult.info?.event?.type;

  if (event === 'goal') {
    components.goal = components.total - components.progress;
  } else if (event === 'shot') {
    components.shot = components.total - components.progress;
  } else if (event === 'pass') {
    components.pass_shaper = components.total - components.progress;
  } else if (event === 'interception' || event === 'tackle') {
    components.defensive = components.total - components.progress;
  }

  return components;
}

// Attach WebSocket Server to the same HTTP Server instance
const wss = new WebSocketServer({ server });

wss.on('connection', (ws: WebSocket, req) => {
  // Start hardware polling on first connection
  if (!hardwareStop) {
    hardwareStop = metricsBroadcaster.startHardwarePolling(3000);
  }

  // Detect dashboard metrics subscribers by URL query parameter
  const url = req.url || '';
  if (url.includes('type=metrics') || url.includes('type=Metrics')) {
    metricsBroadcaster.subscribe(ws);
    // Send initial status (canonical telemetry schema: { type, data })
    const status = TrainingJobService.getStatus();
    ws.send(JSON.stringify({
      type: 'training_status',
      data: {
        isRunning: status.isRunning,
        currentJob: status.currentJob,
        latestMetrics: status.latestMetrics,
      },
    }));
    return;
  }

  // Detect debug subscribers by URL query parameter
  const debugRewards = url.includes('debug=rewards');
  const debugState = debugRewards ? { components: { goal: 0, progress: 0, shot: 0, pass_shaper: 0, defensive: 0 } } : null;

  TrainingJobService.registerWebSocket(ws);
  ws.on('message', async (data: Buffer | ArrayBuffer | Buffer[], isBinary: boolean) => {
    try {
      if (isBinary) {
        const buf = Buffer.isBuffer(data) ? data : Buffer.from(data as any);
        if (buf.length === 1) {
          console.log('[WS DEBUG] Single-agent path');
          // existing single-agent path — unchanged
          const actionIdx = buf.readUInt8(0);
          if (actionIdx >= ACTION_SPACE_SIZE) {
            // P0 #5: NEVER silently drop — send a deterministic binary error
            // frame so Python clients cannot hang waiting for a response.
            ws.send(encodeErrorStepBinary(1, bridge['engine'].activeScenario?.id === 'academy_rondo_4v1'), { binary: true });
            return;
          }
          const stepResult = await bridge.step(actionIdx);

          // Send ground-truth episode stats as JSON BEFORE the binary frame
          if (stepResult.terminated || stepResult.truncated) {
            const episodeStats = {
              type: 'EPISODE_STATS',
              ...stepResult.info.ground_truth,
            };
            ws.send(JSON.stringify(episodeStats));
          }

          // Debug reward breakdown
          if (debugState) {
            const components = computeRewardComponents(stepResult);
            ws.send(JSON.stringify({ type: 'REWARD_COMPONENTS', data: components }));
          }

          const ownerId = bridge['engine'].ball.ownerId as string | null;
          const controlledPlayerId = bridge['engine'].controlledPlayerId as string;
          const ballOwnerAgentIdx = (ownerId && ownerId === controlledPlayerId) ? 0 : 255;
          const isRondo = bridge['engine'].activeScenario?.id === 'academy_rondo_4v1';
          const defenderReward = isRondo ? (bridge['engine'].getActiveScenarioHandler() as any)?.getLastDefenderReward?.() ?? 0 : 0;
          ws.send(encodeStepBinary(stepResult, ballOwnerAgentIdx, isRondo, defenderReward), { binary: true })
        } else if (buf.length >= 2) {
          // batched vectorized path: [B (1B)] [N (1B)] [B*N action bytes]
          const B = buf.readUInt8(0);
          const N = buf.readUInt8(1);
          const expectedLen = 2 + B * N;
          if (B > 0 && N > 0 && buf.length === expectedLen) {
            const actionSets: Array<{ actions: number[]; controllableIds: string[] }> = [];
            for (let envIdx = 0; envIdx < B; envIdx++) {
              const engine = envIdx === 0 ? bridge['engine'] : bridge['pool'][envIdx - 1];
              const isRondo = engine.activeScenario?.id === 'academy_rondo_4v1';
              const envActions: number[] = [];
              const baseOffset = 2 + envIdx * N;
              for (let a = 0; a < N; a++) {
                const act = buf.readUInt8(baseOffset + a);
                if (act >= ACTION_SPACE_SIZE) {
                  ws.send(encodeErrorStepBinary(N, isRondo), { binary: true });
                  return;
                }
                envActions.push(act);
              }
              const isRondoScenario = engine.activeScenario?.id === 'academy_rondo_4v1';
              const controllableIds = isRondoScenario
                ? engine.players.map((p) => p.id)
                : engine.players.filter((p) => p.team === 'left').map((p) => p.id);
              actionSets.push({ actions: envActions, controllableIds });
            }
            const batchResults = await bridge.stepBatch(actionSets);
            const episodeStatsList: any[] = [];
            for (const r of batchResults) {
              if (r.terminated || r.truncated) {
                episodeStatsList.push({ type: 'EPISODE_STATS', ...r.info.ground_truth });
              }
            }
            for (const stats of episodeStatsList) {
              ws.send(JSON.stringify(stats));
            }
            if (debugState) {
              const components = computeRewardComponents(batchResults[0]);
              ws.send(JSON.stringify({ type: 'REWARD_COMPONENTS', data: components }));
            }
            ws.send(encodeBatchedStepBinary(batchResults), { binary: true });
            } else if (buf.length > 1) {
              // existing multi-agent path — unchanged
            const actionIndices = Array.from(buf); // one uint8 per controlled agent, in controllableAgentIds order
            const invalidIdx = actionIndices.findIndex((a) => a >= ACTION_SPACE_SIZE);
            if (invalidIdx >= 0) {
              // P0 #5: deterministic multi-agent error frame (same length as a
              // normal multi-agent response for this agent count).
              const isRondo = bridge['engine'].activeScenario?.id === 'academy_rondo_4v1';
              ws.send(encodeErrorStepBinary(actionIndices.length, isRondo), { binary: true });
              return;
            }
            const multiResult = await bridge.stepMulti(actionIndices);

            // Send ground-truth episode stats as JSON BEFORE the binary frame,
            // so the Python client can capture it in _recv_step_response.
            if (multiResult.terminated || multiResult.truncated) {
              const episodeStats = {
                type: 'EPISODE_STATS',
                ...multiResult.info.ground_truth,
              };
             ws.send(JSON.stringify(episodeStats));
             }
 
             if (debugState) {
               const components = computeRewardComponents(multiResult);
               ws.send(JSON.stringify({ type: 'REWARD_COMPONENTS', data: components }));
             }

             const defenderReward = (bridge['engine'].getActiveScenarioHandler() as any)?.getLastDefenderReward?.() ?? 0;
             const isRondo = bridge['engine'].activeScenario?.id === 'academy_rondo_4v1';
             ws.send(encodeMultiStepBinary(multiResult, isRondo, defenderReward), { binary: true });
          }
        }
      } else {
        const text = data.toString('utf8');
        const parsed = JSON.parse(text);
        if (parsed.type === 'reset') {
          const resetResult = bridge.reset(parsed.scenario, parsed.seed);
          ws.send(JSON.stringify(resetResult));
        } else if (parsed.type === 'reset_batch') {
          const batchResult = bridge.resetBatch(parsed.environments);
          ws.send(JSON.stringify({ type: 'reset_batch_result', results: batchResult }));
        } else if (parsed.type === 'close') {
          ws.close();
        } else if (parsed.type === 'info') {
          ws.send(JSON.stringify(bridge.getInfo()));
        } else if (parsed.type === 'step') {
          const stepResult = await bridge.step(parsed.action);
          ws.send(JSON.stringify(stepResult));
        } else if (parsed.type === 'step_multi') {
          const multiResult = await bridge.stepMulti(parsed.actions);
          ws.send(JSON.stringify(multiResult));
        } else if (parsed.type === 'telemetry_metrics') {
          // Deprecated relay (no Python producers remain). Normalizes the
          // forwarded payload to the canonical training_metrics frame.
          const canonical = JSON.stringify({ type: 'training_metrics', data: parsed.data ?? parsed });
          wss.clients.forEach((client) => {
            if (client !== ws && client.readyState === WebSocket.OPEN) {
              client.send(canonical);
            }
          });
          ws.send(JSON.stringify({ status: 'broadcast_ok' }));
        } else if (parsed.type === 'subscribe_metrics') {
          // Runtime subscription — dashboard can subscribe after connecting
          metricsBroadcaster.subscribe(ws);
          ws.send(JSON.stringify({ status: 'subscribed_metrics' }));
        } else if (parsed.type === 'unsubscribe_metrics') {
          metricsBroadcaster.unsubscribe(ws);
          ws.send(JSON.stringify({ status: 'unsubscribed_metrics' }));
        } else if (parsed.type === 'subscribe_training') {
          // P1 #9: explicit dashboard subscription (previously silently ignored
          // and only worked via accidental registration). Subscribes this
          // socket to the MetricsBroadcaster stream (training_status,
          // hardware_stats, training_metrics, checkpoint_update,
          // training_output) AND keeps it in the TrainingJobService RL-client
          // set (already registered at connection time). Replies with the
          // canonical initial training_status.
          metricsBroadcaster.subscribe(ws);
          const status = TrainingJobService.getStatus();
          ws.send(JSON.stringify({ type: 'training_status', data: status }));
          ws.send(JSON.stringify({ type: 'checkpoint_update', data: { checkpoints: CheckpointService.listCheckpoints() } }));
        } else if (parsed.type === 'unsubscribe_training') {
          metricsBroadcaster.unsubscribe(ws);
          ws.send(JSON.stringify({ status: 'unsubscribed_training' }));
        }
      }
    } catch (err: any) {
      console.error('[WS Error]', err);
      try {
        if (isBinary) {
          const nAgents = Buffer.isBuffer(data) ? Math.max(1, data.length) : 1;
          const isRondo = bridge['engine'].activeScenario?.id === 'academy_rondo_4v1';
          ws.send(encodeErrorStepBinary(nAgents, isRondo), { binary: true });
        } else {
          ws.send(JSON.stringify({ type: 'error', error: err.message || 'Internal bridge error', data: { message: err.message || 'Internal bridge error' } }));
        }
      } catch (sendErr) {
        console.error('[WS Error] Failed to send error response:', sendErr);
      }
    }
  });
});

server.keepAliveTimeout = 120000;
server.headersTimeout = 125000;
server.on('error', (err) => {
  console.error('[GMN Bridge Server] Socket error:', err);
});

// Only start listening when this file is executed directly (npm run bridge /
// spawned by the Python harnesses). When the module is imported by a test or by
// TrainingJobService for `metricsBroadcaster`, we must NOT bind a port.
const isMainModule =
  typeof process.argv[1] === 'string' &&
  import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMainModule) {
  server.listen(PORT, HOST, () => {
    console.log(`[GMN Headless Bridge] Server listening on http://${HOST}:${PORT} (HTTP + Binary WebSocket)`);
  });

  process.on('SIGINT', () => {
    server.close(() => process.exit(0));
  });

  process.on('SIGTERM', () => {
    server.close(() => process.exit(0));
  });
}
