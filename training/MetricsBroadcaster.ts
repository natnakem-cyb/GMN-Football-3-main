import { WebSocket } from 'ws';
import os from 'os';

// Snapshot shapes mirrored from src/types/telemetry.ts (no runtime dep on src/)
export interface TrainingMetricsSnapshot {
  step: number;
  update: number;
  policyLoss: number;
  valueLoss: number;
  entropy: number;
  approxKl: number;
  clipFraction: number;
  learningRate: number;
  gradNorm: number;
  rollingReward: number;
  goalRate: number;
  timestamp: number;
}

export interface HardwareStats {
  cpuPercent: number;
  ramPercent: number;
  ramUsedMb: number;
  ramTotalMb: number;
  stepsPerSec: number;
}

export interface CheckpointInfo {
  id: string;
  filename: string;
  path: string;
  sizeBytes: number;
  scenario: string;
  algorithm: string;
  timesteps: number;
  createdAt: string;
  hasSourcePt: boolean;
}

/**
 * MetricsBroadcaster — pub/sub hub that pushes live training telemetry to
 * connected dashboard WebSocket clients. The dashboard opens a WS with
 * ?type=Metrics and receives frames: training_metrics, hardware_stats,
 * training_output, checkpoints_updated, training_status.
 */
export class MetricsBroadcaster {
  private subscribers: Set<WebSocket> = new Set();
  private _stepsPerSec = 0;

  get stepsPerSec(): number {
    return this._stepsPerSec;
  }

  set stepsPerSec(value: number) {
    this._stepsPerSec = value;
  }

  subscribe(ws: WebSocket) {
    this.subscribers.add(ws);
    ws.on('close', () => this.subscribers.delete(ws));
    ws.on('error', () => this.subscribers.delete(ws));
  }

  unsubscribe(ws: WebSocket) {
    this.subscribers.delete(ws);
  }

  get subscriberCount(): number {
    return this.subscribers.size;
  }

  private send(frame: object) {
    const payload = JSON.stringify(frame);
    for (const ws of this.subscribers) {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(payload);
        } catch {
          this.subscribers.delete(ws);
        }
      } else {
        this.subscribers.delete(ws);
      }
    }
  }

  broadcastMetrics(snapshot: TrainingMetricsSnapshot) {
    this.send({ type: 'training_metrics', payload: snapshot });
  }

  broadcastHardware(stats: HardwareStats) {
    this.send({ type: 'hardware_stats', payload: stats });
  }

  broadcastOutput(text: string) {
    this.send({ type: 'training_output', payload: { text, ts: Date.now() } });
  }

  broadcastCheckpoints(list: CheckpointInfo[]) {
    this.send({ type: 'checkpoints_updated', payload: list });
  }

  broadcastStatus(status: {
    isRunning: boolean;
    currentJob: any;
    latestMetrics: TrainingMetricsSnapshot | { step: number; timestamp: number } | null;
    recentLogs?: string[];
  }) {
    this.send({ type: 'training_status', payload: status });
  }

  /**
   * Compute steps-per-second from a new step sample. Call this whenever
   * a metrics frame arrives so hardware frames carry an accurate rate.
   */
  trackStepRate(step: number, timestampMs: number): void {
    const last = this._lastStepSample;
    if (last !== null) {
      const dt = (timestampMs - last.ts) / 1000;
      if (dt > 0) {
        this._stepsPerSec = Math.round(((step - last.step) / dt) * 10) / 10;
      }
    }
    this._lastStepSample = { step, ts: timestampMs };
  }

  private _lastStepSample: { step: number; ts: number } | null = null;

  /**
   * Start a polling loop that reads system hardware stats at the given
   * interval and broadcasts to all metrics subscribers. Returns a stop fn.
   */
  startHardwarePolling(intervalMs: number = 3000): () => void {
    let lastIdle = 0;
    let lastTotal = 0;
    const cpus = os.cpus();
    if (cpus.length > 0) {
      lastIdle = cpus.reduce((a, c) => a + c.times.idle, 0);
      lastTotal = cpus.reduce(
        (a, c) => a + c.times.user + c.times.nice + c.times.sys + c.times.idle + c.times.irq,
        0
      );
    }

    const timer = setInterval(() => {
      if (this.subscribers.size === 0) return;

      // CPU delta
      const curCpus = os.cpus();
      let curIdle = 0;
      let curTotal = 0;
      if (curCpus.length > 0) {
        curIdle = curCpus.reduce((a, c) => a + c.times.idle, 0);
        curTotal = curCpus.reduce(
          (a, c) => a + c.times.user + c.times.nice + c.times.sys + c.times.idle + c.times.irq,
          0
        );
      }
      const idleDelta = curIdle - lastIdle;
      const totalDelta = curTotal - lastTotal;
      const cpuPercent = totalDelta > 0 ? Math.round((1 - idleDelta / totalDelta) * 100) : 0;
      lastIdle = curIdle;
      lastTotal = curTotal;

      // RAM
      const totalMem = os.totalmem();
      const freeMem = os.freemem();
      const usedMem = totalMem - freeMem;
      const ramPercent = Math.round((usedMem / totalMem) * 100);

      this.broadcastHardware({
        cpuPercent: Math.max(0, Math.min(100, cpuPercent)),
        ramPercent,
        ramUsedMb: Math.round(usedMem / 1024 / 1024),
        ramTotalMb: Math.round(totalMem / 1024 / 1024),
        stepsPerSec: this._stepsPerSec,
      });
    }, intervalMs);

    timer.unref?.();
    return () => clearInterval(timer);
  }
}
