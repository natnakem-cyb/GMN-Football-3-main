import {
  TrainingMetricsSnapshot,
  HardwareMetrics,
  AgentCreditMetrics,
  PolicyActionDistribution,
  ActionProbabilityEntry,
  TrainingHyperparameters,
} from '../types/telemetry';
import { Player, Ball, Vector2D, MatchEvent } from '../types/football';
import { TrainedPolicyAgent } from '../agents/TrainedPolicyAgent';
import { ObservationEncoder } from './ObservationEncoder';

export const ACTION_NAMES = [
  'Idle',
  'Move Left',
  'Move Top-Left',
  'Move Top',
  'Move Top-Right',
  'Move Right',
  'Move Bottom-Right',
  'Move Bottom',
  'Move Bottom-Left',
  'Long Pass',
  'High Pass',
  'Short Pass',
  'Direct Shot',
  'Sprint',
  'Release Direction',
  'Release Sprint',
  'Slide Tackle',
  'Close Dribble',
  'Release Dribble',
];

export const ACTION_SHORT_LABELS = [
  'IDLE',
  'LEFT',
  'T-LEFT',
  'TOP',
  'T-RIGHT',
  'RIGHT',
  'B-RIGHT',
  'BOTTOM',
  'B-LEFT',
  'L-PASS',
  'H-PASS',
  'S-PASS',
  'SHOT',
  'SPRINT',
  'REL-DIR',
  'REL-SPR',
  'TACKLE',
  'DRIBBLE',
  'REL-DRIB',
];

export const ACTION_CATEGORIES: ActionProbabilityEntry['category'][] = [
  'sticky',
  'move',
  'move',
  'move',
  'move',
  'move',
  'move',
  'move',
  'move',
  'pass',
  'pass',
  'pass',
  'shot',
  'sticky',
  'sticky',
  'sticky',
  'defense',
  'sticky',
  'sticky',
];

export class TrainingTelemetryService {
  private static instance: TrainingTelemetryService;

  public hyperparameters: TrainingHyperparameters = {
    learningRate: 3e-4,
    clipRange: 0.2,
    entropyCoef: 0.01,
    valueCoef: 0.5,
    miniBatchSize: 64,
    nEpochs: 4,
    gamma: 0.99,
    gaeLambda: 0.95,
    targetTimesteps: 200000,
    maxGradNorm: 0.5,
  };

  public snapshots: TrainingMetricsSnapshot[] = [];
  public currentStep: number = 0;
  public isTrainingActive: boolean = false;
  public trainingSpeed: number = 1;
  private listeners: Array<() => void> = [];

  public hardware: HardwareMetrics = {
    sps: null,
    fps: null,
    gpuVramUsedMb: null,
    gpuVramTotalMb: null,
    gpuUtilizationPct: null,
    cpuUtilizationPct: null,
    workerCount: null,
    bufferSize: null,
    bufferCapacity: null,
    ipcLatencyMs: null,
    activeDevice: 'No training running',
  };

  // Real-time WebSocket Bridge Link
  public isWsConnected: boolean = false;
  public wsUrl: string = 'ws://127.0.0.1:5050';
  public wsStatus: 'disconnected' | 'connecting' | 'connected' | 'error' = 'disconnected';
  public lastWsMessageTime: number = 0;
  private ws: any = null;

  public static getInstance(): TrainingTelemetryService {
    if (!TrainingTelemetryService.instance) {
      TrainingTelemetryService.instance = new TrainingTelemetryService();
    }
    return TrainingTelemetryService.instance;
  }

  public subscribe(listener: () => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  private notify(): void {
    this.listeners.forEach((l) => l());
  }

  public startTraining(): void {
    if (this.isTrainingActive) return;
    this.isTrainingActive = true;
    this.notify();
  }

  public pauseTraining(): void {
    this.isTrainingActive = false;
    this.notify();
  }

  public toggleTraining(): void {
    if (this.isTrainingActive) {
      this.pauseTraining();
    } else {
      this.startTraining();
    }
  }

  public stepTrainingBatch(): void {
    // No-op: metrics are now ingested live from the Python bridge via WebSocket.
    // This stub preserves the API contract but generates zero synthetic data.
    this.notify();
  }

  public setHyperparameter<K extends keyof TrainingHyperparameters>(
    key: K,
    val: TrainingHyperparameters[K]
  ): void {
    this.hyperparameters[key] = val;
    this.notify();
  }

  public setTrainingSpeed(speed: number): void {
    this.trainingSpeed = speed;
    this.notify();
  }

    public resetMetrics(): void {
    this.currentStep = 0;
    this.isTrainingActive = false;
    this.snapshots = [];
    this.notify();
  }

  public liveLogs: string[] = [];
  public activeJob: any = null;
  public checkpoints: any[] = [];
  public isRefreshingCheckpoints: boolean = false;

  /**
   * Resolves the telemetry WebSocket endpoint.
   *
   * Priority:
   *   1. `VITE_WS_URL` build-time env (production / remote bridge deployments).
   *   2. Same-origin `/ws` path — served by the Vite dev proxy in development
   *      (vite.config.ts proxies `/ws` -> ws://127.0.0.1:5050) and by any
   *      production reverse proxy that forwards `/ws` to the bridge.
   *
   * No hardcoded deployment-specific hostnames.
   */
  private resolveWsUrl(): string {
    const env = (import.meta as any)?.env ?? {};
    if (env.VITE_WS_URL) return String(env.VITE_WS_URL);
    if (typeof window !== 'undefined' && window.location) {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${proto}//${window.location.host}/ws`;
    }
    // Non-browser fallback (tests / SSR): local bridge default.
    return 'ws://127.0.0.1:5050';
  }

  /**
   * Connects to the GMN bridge telemetry stream over WebSocket and subscribes
   * to live training telemetry. `subscribe_training` is explicitly handled by
   * the bridge (see training/bridge_server.ts).
   */
  public connectWebSocket(url?: string): void {
    if (typeof window === 'undefined') return;

    this.wsUrl = url ?? this.resolveWsUrl();

    if (this.ws) {
      try {
        this.ws.close();
      } catch {}
      this.ws = null;
    }

    this.wsStatus = 'connecting';
    this.notify();

    try {
      const socket = new WebSocket(this.wsUrl);
      this.ws = socket;

      socket.onopen = () => {
        this.isWsConnected = true;
        this.wsStatus = 'connected';
        socket.send(JSON.stringify({ type: 'subscribe_training' }));
        this.refreshCheckpoints();
        this.notify();
      };

      socket.onmessage = (event) => {
        try {
          if (typeof event.data === 'string') {
            const parsed = JSON.parse(event.data);
            const msgType = parsed.type;

            // Canonical telemetry protocol: every message is { type, data }.
            // Schema documented in training/TELEMETRY_PROTOCOL.md.
            if (msgType === 'training_status') {
              this.activeJob = parsed.data?.currentJob || null;
              if (parsed.data?.isRunning) {
                this.isTrainingActive = true;
              }
              if (parsed.data?.recentLogs && Array.isArray(parsed.data.recentLogs)) {
                this.liveLogs = parsed.data.recentLogs;
              }
              if (parsed.data?.latestMetrics) {
                this.ingestSnapshot(parsed.data.latestMetrics);
              }
              this.notify();
            } else if (msgType === 'training_started') {
              this.activeJob = parsed.data;
              this.isTrainingActive = true;
              this.notify();
            } else if (msgType === 'training_output') {
              const line = parsed.data?.line;
              if (line) {
                this.liveLogs.push(line);
                if (this.liveLogs.length > 500) {
                  this.liveLogs.shift();
                }
                this.notify();
              }
            } else if (msgType === 'training_metrics' || msgType === 'episode_metrics') {
              const metrics = parsed.data || parsed.snapshot || parsed;
              this.ingestSnapshot(metrics, parsed.hardware);
            } else if (msgType === 'hardware_stats') {
              // P1 #10: hardware statistics reach the dashboard.
              const hw = parsed.data || {};
              this.hardware = {
                ...this.hardware,
                sps: hw.stepsPerSec ?? this.hardware.sps,
                cpuUtilizationPct: hw.cpuPercent ?? this.hardware.cpuUtilizationPct,
                gpuVramUsedMb: hw.ramUsedMb ?? this.hardware.gpuVramUsedMb,
                gpuVramTotalMb: hw.ramTotalMb ?? this.hardware.gpuVramTotalMb,
              };
              this.notify();
            } else if (msgType === 'checkpoint_update') {
              this.checkpoints = parsed.data?.checkpoints || [];
              this.notify();
            } else if (msgType === 'training_completed') {
              this.isTrainingActive = false;
              this.activeJob = null;
              this.refreshCheckpoints();
              this.notify();
            } else if (msgType === 'training_stopped' || msgType === 'training_failed') {
              this.isTrainingActive = false;
              if (msgType === 'training_failed' && parsed.data?.error) {
                this.liveLogs.push(`[ERROR] ${parsed.data.error}`);
              }
              this.notify();
            } else if (msgType === 'error') {
              // Bridge-reported error: log it; never crash the connection.
              this.liveLogs.push(`[BRIDGE ERROR] ${parsed.data?.message || parsed.error || 'unknown error'}`);
              this.notify();
            }
            // Unknown/unsupported message types are ignored — telemetry must
            // never crash the connection.
          }
        } catch (err) {
          console.warn('[TrainingTelemetryService] WS message parse error:', err);
        }
      };

      socket.onerror = () => {
        this.wsStatus = 'error';
        this.isWsConnected = false;
        this.notify();
      };

      socket.onclose = () => {
        this.isWsConnected = false;
        this.wsStatus = 'disconnected';
        this.ws = null;
        this.notify();
      };
    } catch {
      this.wsStatus = 'error';
      this.isWsConnected = false;
      this.notify();
    }
  }

  /**
   * Triggers background Python RL training job via REST API.
   */
  public async startTrainingJob(config: {
    algorithm: string;
    scenario: string;
    timesteps: number;
    resumeFrom?: string;
  }): Promise<{ success: boolean; message?: string; error?: string; job?: any }> {
    try {
      const res = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      const data = await res.json();
      if (data.success) {
        this.isTrainingActive = true;
        this.activeJob = data.job;
        this.notify();
      }
      return data;
    } catch (err: any) {
      return { success: false, error: err.message || 'Network error starting training' };
    }
  }

  /**
   * Stops running Python RL training job via REST API.
   */
  public async stopTrainingJob(): Promise<{ success: boolean; message?: string }> {
    try {
      const res = await fetch('/api/training/stop', {
        method: 'POST',
      });
      const data = await res.json();
      this.isTrainingActive = false;
      this.notify();
      return data;
    } catch (err: any) {
      return { success: false, message: err.message };
    }
  }

  /**
   * Fetches all registered ONNX checkpoints and sidecars from public/models.
   */
  public async refreshCheckpoints(): Promise<any[]> {
    this.isRefreshingCheckpoints = true;
    this.notify();
    try {
      const res = await fetch('/api/checkpoints');
      if (res.ok) {
        const data = await res.json();
        this.checkpoints = data.checkpoints || [];
      }
    } catch (e) {
      console.warn('[TrainingTelemetryService] Failed to fetch checkpoints:', e);
    } finally {
      this.isRefreshingCheckpoints = false;
      this.notify();
    }
    return this.checkpoints;
  }

  /**
   * Deletes a checkpoint and its sidecar JSON from public/models.
   */
  public async deleteCheckpoint(filename: string, deleteSourcePt: boolean = false): Promise<any> {
    const res = await fetch('/api/checkpoints/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, deleteSourcePt }),
    });
    const result = await res.json();
    await this.refreshCheckpoints();
    return result;
  }

  /**
   * Uploads an ONNX model file directly from the browser.
   */
  public async uploadCheckpoint(filename: string, base64Data: string, metadata?: any): Promise<any> {
    const res = await fetch('/api/checkpoints/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, base64Data, ...metadata }),
    });
    const result = await res.json();
    await this.refreshCheckpoints();
    return result;
  }

  public disconnectWebSocket(): void {
    if (this.ws) {
      try {
        this.ws.close();
      } catch {}
      this.ws = null;
    }
    this.isWsConnected = false;
    this.wsStatus = 'disconnected';
    this.notify();
  }

  /**
   * Directly ingests real-time training step updates from PyTorch training loops.
   */
  public ingestSnapshot(
    snapshot: Partial<TrainingMetricsSnapshot>,
    hardware?: Partial<HardwareMetrics>
  ): void {
    const lastSnap = this.snapshots[this.snapshots.length - 1];
    const newStep = snapshot.step ?? (lastSnap ? lastSnap.step + 256 : 0);
    this.currentStep = newStep;
    this.lastWsMessageTime = Date.now();

    const fullSnapshot: TrainingMetricsSnapshot = {
      step: newStep,
      update: snapshot.update ?? Math.round(newStep / 256),
      policyLoss: snapshot.policyLoss ?? lastSnap?.policyLoss ?? null,
      valueLoss: snapshot.valueLoss ?? lastSnap?.valueLoss ?? null,
      entropy: snapshot.entropy ?? lastSnap?.entropy ?? null,
      approxKl: snapshot.approxKl ?? lastSnap?.approxKl ?? null,
      clipFraction: snapshot.clipFraction ?? lastSnap?.clipFraction ?? null,
      learningRate: snapshot.learningRate ?? lastSnap?.learningRate ?? null,
      gradNorm: snapshot.gradNorm ?? lastSnap?.gradNorm ?? null,
      rollingReward: snapshot.rollingReward ?? lastSnap?.rollingReward ?? null,
      goalRate: snapshot.goalRate ?? lastSnap?.goalRate ?? null,
      timestamp: snapshot.timestamp ?? Date.now(),
    };

    this.snapshots = [...this.snapshots.slice(-40), fullSnapshot];

    if (hardware) {
      this.hardware = {
        ...this.hardware,
        ...hardware,
      };
    }

    this.notify();
  }

  /**
   * Computes policy action probabilities, critic state value V(s), and tactical attention
   * for any player on the pitch given live game state.
   */
  public evaluateAgentPolicy(
    player: Player,
    allPlayers: Player[],
    ball: Ball,
    policyAgent?: TrainedPolicyAgent | null
  ): PolicyActionDistribution {
    const rawObs = ObservationEncoder.encode(
      allPlayers,
      ball,
      player.id,
      { left: 0, right: 0 },
      0,
      3000
    ).rawVector;

    let logits: number[] = [];

    // Attempt to evaluate real neural forward pass if weights are present
    if (policyAgent && TrainedPolicyAgent.isCheckpointValid()) {
      try {
        logits = policyAgent.computeLogits(rawObs);
      } catch {
        logits = [];
      }
    }

    // High-fidelity fallback / tactical policy projection if logits are empty or uninitialized
    const isSynthetic = !logits || logits.length !== 19;
    if (isSynthetic) {
      logits = this.computeSyntheticPolicyLogits(player, allPlayers, ball);
    }

    // Softmax normalization
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map((l) => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((acc, v) => acc + v, 0);
    const probabilities = expLogits.map((e) => e / sumExp);

    const actionEntries: ActionProbabilityEntry[] = probabilities.map((prob, idx) => ({
      index: idx,
      name: ACTION_NAMES[idx] || `Action ${idx}`,
      shortLabel: ACTION_SHORT_LABELS[idx] || `A${idx}`,
      probability: prob,
      logit: logits[idx],
      category: ACTION_CATEGORIES[idx] || 'move',
    }));

    // Find best action
    let bestIdx = 0;
    let maxProb = -1;
    actionEntries.forEach((a, i) => {
      if (a.probability > maxProb) {
        maxProb = a.probability;
        bestIdx = i;
      }
    });

    // Centralized Critic State Value V(s) in [-1.0, 1.0]
    // Value represents expected team goal advantage based on ball possession and distance to goal
    const distToGoal = Math.hypot(1.0 - ball.position.x, ball.position.y);
    const isPlayerPossessing = player.hasBall;
    const teamPossessing = ball.position.x > -0.2 && player.team === 'left';
    let baseValue = 0.35 * (1.0 - distToGoal / 1.5);
    if (isPlayerPossessing) baseValue += 0.3;
    else if (teamPossessing) baseValue += 0.15;
    const valueEstimate = Math.max(-0.95, Math.min(0.98, baseValue));

    // Spatial Attention: compute receiver candidate and pass clearance
    const teammates = allPlayers.filter((p) => p.team === player.team && p.id !== player.id);
    let bestPassCandidate: Player | undefined;
    let bestPassClearance = 0;

    teammates.forEach((tm) => {
      const dx = tm.position.x - player.position.x;
      const dy = tm.position.y - player.position.y;
      const dist = Math.hypot(dx, dy);
      if (dist > 0.1 && dist < 0.8) {
        // Check opponent distance from passing lane
        const opps = allPlayers.filter((p) => p.team !== player.team);
        let minLaneDist = 1.0;
        opps.forEach((opp) => {
          const oppDist = Math.hypot(opp.position.x - (player.position.x + tm.position.x) / 2, opp.position.y - (player.position.y + tm.position.y) / 2);
          if (oppDist < minLaneDist) minLaneDist = oppDist;
        });
        const clearance = Math.min(1.0, minLaneDist * 4);
        if (clearance > bestPassClearance) {
          bestPassClearance = clearance;
          bestPassCandidate = tm;
        }
      }
    });

    return {
      playerId: player.id,
      role: player.role,
      valueEstimate: Number(valueEstimate.toFixed(3)),
      actions: actionEntries,
      bestActionIndex: bestIdx,
      bestActionName: ACTION_NAMES[bestIdx],
      confidence: Number((maxProb * 100).toFixed(1)),
      isSynthetic,
      attention: bestPassCandidate
        ? {
            targetPlayerId: bestPassCandidate.id,
            targetPos: bestPassCandidate.position,
            passClearanceProb: Number((bestPassClearance * 100).toFixed(1)),
            shotAngleClearance: Number((Math.max(15, 100 - distToGoal * 70)).toFixed(1)),
          }
        : undefined,
    };
  }

  private computeSyntheticPolicyLogits(player: Player, allPlayers: Player[], ball: Ball): number[] {
    const logits = new Array(19).fill(-1.5);
    const distToBall = Math.hypot(ball.position.x - player.position.x, ball.position.y - player.position.y);
    const distToGoal = Math.hypot(1.0 - player.position.x, player.position.y);

    if (player.hasBall) {
      if (distToGoal < 0.35) {
        // Direct shot zone
        logits[12] = 2.8; // Shot
        logits[11] = 1.2; // Short Pass
        logits[5] = 1.4;  // Move Right
        logits[13] = 0.8; // Sprint
      } else if (distToGoal < 0.6) {
        // Playmaking zone
        logits[11] = 2.4; // Short pass
        logits[9] = 1.6;  // Long pass
        logits[5] = 1.8;  // Move Right
        logits[17] = 1.1; // Dribble
        logits[12] = 1.0; // Shot
      } else {
        // Build up
        logits[5] = 2.1;  // Move right
        logits[11] = 1.8; // Short pass
        logits[13] = 1.3; // Sprint
        logits[17] = 0.9;
      }
    } else {
      // Off-ball positioning or pressing
      if (distToBall < 0.15 && player.team === 'left') {
        logits[16] = 2.1; // Tackle
        logits[13] = 1.5; // Sprint
      } else {
        // Direction to ball or open goal space
        const dx = ball.position.x - player.position.x;
        const dy = ball.position.y - player.position.y;
        if (dx > 0.05) logits[5] = 1.9; // Right
        else if (dx < -0.05) logits[1] = 1.9; // Left
        if (dy > 0.05) logits[7] = 1.6; // Bottom
        else if (dy < -0.05) logits[3] = 1.6; // Top
        logits[13] = 1.2; // Sprint
      }
    }

    return logits;
  }

  /**
   * Computes honest per-player contribution metrics from real match state.
   * Counterfactual advantage is not available from the browser-side engine alone,
   * so this returns observable behavioral metrics instead.
   */
  public computeMultiAgentCredits(players: Player[], ball: Ball, events: MatchEvent[] = []): AgentCreditMetrics[] {
    const leftPlayers = players.filter((p) => p.team === 'left');

    // Pre-compute per-player event counts from the actual match event log.
    const passCounts: Record<string, number> = {};
    const interceptionCounts: Record<string, number> = {};
    for (const ev of events) {
      if (!ev.playerId) continue;
      if (ev.type === 'pass') {
        passCounts[ev.playerId] = (passCounts[ev.playerId] || 0) + 1;
      } else if (ev.type === 'interception') {
        interceptionCounts[ev.playerId] = (interceptionCounts[ev.playerId] || 0) + 1;
      }
    }

    // Normalize raw counts to a 0-1 contribution score so rewardContribution
    // is comparable across metrics without inventing counterfactual baselines.
    const maxDistance = Math.max(1, ...leftPlayers.map((p) => p.distanceCovered || 0));
    const maxPasses = Math.max(1, ...leftPlayers.map((p) => passCounts[p.id] || 0));
    const maxInterceptions = Math.max(1, ...leftPlayers.map((p) => interceptionCounts[p.id] || 0));

    return leftPlayers.map((p) => {
      const distance = p.distanceCovered || 0;
      const passes = passCounts[p.id] || 0;
      const interceptions = interceptionCounts[p.id] || 0;

      const distanceScore = distance / maxDistance;
      const passScore = passes / maxPasses;
      const interceptionScore = interceptions / maxInterceptions;

      const rewardContribution = Number(((distanceScore * 0.5 + passScore * 0.3 + interceptionScore * 0.2) * 1.8).toFixed(3));

      return {
        playerId: p.id,
        playerName: p.name || `Player #${p.number}`,
        role: p.role,
        rewardContribution,
        distanceCovered: Number(distance.toFixed(2)),
        totalPasses: passes,
        defensiveInterceptions: interceptions,
      };
    });
  }
}
