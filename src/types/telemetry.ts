import { Vector2D } from './football';

export interface TrainingMetricsSnapshot {
  step: number;
  update: number;
  policyLoss: number | null;
  valueLoss: number | null;
  entropy: number | null;
  approxKl: number | null;
  clipFraction: number | null;
  learningRate: number | null;
  gradNorm: number | null;
  rollingReward: number | null;
  goalRate: number | null;
  timestamp: number;
}

export interface HardwareMetrics {
  sps: number | null; // steps per second
  fps: number | null; // visual frame rate
  gpuVramUsedMb: number | null;
  gpuVramTotalMb: number | null;
  gpuUtilizationPct: number | null;
  cpuUtilizationPct: number | null;
  workerCount: number | null;
  bufferSize: number | null;
  bufferCapacity: number | null;
  ipcLatencyMs: number | null;
  activeDevice: 'CUDA (RTX 4090 / Cloud T4)' | 'Apple MPS' | 'WebGPU / WASM Vectorized' | 'No training running';
}

export interface AgentCreditMetrics {
  playerId: string;
  playerName: string;
  role: string;
  rewardContribution: number;
  distanceCovered: number;
  totalPasses: number;
  defensiveInterceptions: number;
}

export interface ActionProbabilityEntry {
  index: number;
  name: string;
  shortLabel: string;
  probability: number;
  logit: number;
  category: 'move' | 'pass' | 'shot' | 'defense' | 'sticky';
}

export interface PolicyActionDistribution {
  playerId: string;
  role: string;
  valueEstimate: number; // V(s) baseline in [-1, +1]
  actions: ActionProbabilityEntry[];
  bestActionIndex: number;
  bestActionName: string;
  confidence: number;
  isSynthetic?: boolean;
  attention?: {
    targetPlayerId?: string;
    targetPos?: Vector2D;
    passClearanceProb?: number;
    shotAngleClearance?: number;
  };
}

export interface TrainingHyperparameters {
  learningRate: number;
  clipRange: number;
  entropyCoef: number;
  valueCoef: number;
  miniBatchSize: number;
  nEpochs: number;
  gamma: number;
  gaeLambda: number;
  targetTimesteps: number;
  maxGradNorm: number;
}
