import React, { useState } from 'react';
import { ActionType, AgentAction, RLObservation, RLStepResult } from '../types/football';
import { Cpu, Terminal, Play, RotateCcw, Activity, Award, CheckSquare, Layers } from 'lucide-react';
import { OBSERVATION_DIM, BASE_OBSERVATION_DIM, ROLE_DIM } from '../engine/Contract';

interface RLGymnasiumPanelProps {
  lastStepResult: RLStepResult | null;
  onEnvReset: () => void;
  onEnvStepAction: (action: AgentAction) => void;
  stepCount: number;
}

export const RLGymnasiumPanel: React.FC<RLGymnasiumPanelProps> = ({
  lastStepResult,
  onEnvReset,
  onEnvStepAction,
  stepCount,
}) => {
  const [vectorViewFilter, setVectorViewFilter] = useState<'all' | 'players' | 'ball' | 'match'>('all');

  const obs = lastStepResult?.observation;
  const rawVector = obs?.rawVector || [];

  return (
    <div id="rl-gymnasium-panel" className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-5 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-purple-400" /> Gymnasium RL Environment & SMM Vector Inspector
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
                        Standard {OBSERVATION_DIM}-float Google Research Football observation tensor (simple115_v3_role, base {BASE_OBSERVATION_DIM} + role {ROLE_DIM}), reward signals, and step interface.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onEnvReset}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700"
          >
            <RotateCcw className="w-3.5 h-3.5" /> env.reset()
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Step Reward & Telemetry */}
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
            <h4 className="text-xs font-bold text-slate-300 flex items-center gap-1.5 mb-3">
              <Award className="w-4 h-4 text-amber-400" /> Reward & Episode State
            </h4>

            <div className="space-y-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Step Count:</span>
                <span className="font-mono font-bold text-slate-100">{stepCount}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Immediate Reward:</span>
                <span
                  className={`font-mono font-bold ${
                    (lastStepResult?.reward || 0) > 0
                      ? 'text-emerald-400'
                      : (lastStepResult?.reward || 0) < 0
                      ? 'text-red-400'
                      : 'text-slate-300'
                  }`}
                >
                  {(lastStepResult?.reward || 0).toFixed(4)}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Checkpoint Reward:</span>
                <span
                  className={`font-mono font-semibold ${
                    (lastStepResult?.info.checkpointReward || 0) > 0
                      ? 'text-emerald-400'
                      : (lastStepResult?.info.checkpointReward || 0) < 0
                      ? 'text-red-400'
                      : 'text-slate-300'
                  }`}
                >
                  {(lastStepResult?.info.checkpointReward || 0).toFixed(4)}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Terminated / Truncated:</span>
                <span className="font-mono text-slate-300">
                  {lastStepResult?.terminated ? 'True (Terminated)' : lastStepResult?.truncated ? 'True (Truncated)' : 'False'}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Ball Dist to Goal:</span>
                <span className="font-mono text-blue-400">
                  {(lastStepResult?.info.ballDistanceToGoal || 0).toFixed(3)}
                </span>
              </div>
            </div>
          </div>

          {/* Action Step Trigger Panel */}
                    <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
            <h4 className="text-xs font-bold text-slate-300 flex items-center gap-1.5 mb-2.5">
              <Terminal className="w-4 h-4 text-purple-400" /> Full 19-Action Discrete Action Trigger
            </h4>
            <div className="grid grid-cols-5 gap-1.5">
              {[
                { label: 'MOVE-N', action: () => ({ type: ActionType.MOVE, direction: { x: 0, y: -1 } }), color: 'text-blue-300' },
                { label: 'MOVE-NE', action: () => ({ type: ActionType.MOVE, direction: { x: 0.707, y: -0.707 } }), color: 'text-blue-300' },
                { label: 'MOVE-E', action: () => ({ type: ActionType.MOVE, direction: { x: 1, y: 0 } }), color: 'text-blue-300' },
                { label: 'MOVE-SE', action: () => ({ type: ActionType.MOVE, direction: { x: 0.707, y: 0.707 } }), color: 'text-blue-300' },
                { label: 'MOVE-S', action: () => ({ type: ActionType.MOVE, direction: { x: 0, y: 1 } }), color: 'text-blue-300' },
                { label: 'MOVE-SW', action: () => ({ type: ActionType.MOVE, direction: { x: -0.707, y: 0.707 } }), color: 'text-blue-300' },
                { label: 'MOVE-W', action: () => ({ type: ActionType.MOVE, direction: { x: -1, y: 0 } }), color: 'text-blue-300' },
                { label: 'MOVE-NW', action: () => ({ type: ActionType.MOVE, direction: { x: -0.707, y: -0.707 } }), color: 'text-blue-300' },
                { label: 'SPRINT', action: () => ({ type: ActionType.SPRINT }), color: 'text-cyan-300' },
                { label: 'SHORT P', action: () => ({ type: ActionType.SHORT_PASS }), color: 'text-green-300' },
                { label: 'LONG P', action: () => ({ type: ActionType.LONG_PASS }), color: 'text-green-300' },
                { label: 'HIGH P', action: () => ({ type: ActionType.HIGH_PASS }), color: 'text-green-300' },
                { label: 'SHOT', action: () => ({ type: ActionType.SHOT }), color: 'text-amber-300' },
                { label: 'TACKLE', action: () => ({ type: ActionType.TACKLE }), color: 'text-red-300' },
                { label: 'SLIDE', action: () => ({ type: ActionType.SLIDING }), color: 'text-orange-300' },
                { label: 'DRIBBLE', action: () => ({ type: ActionType.DRIBBLE }), color: 'text-purple-300' },
                { label: 'REL DIR', action: () => ({ type: ActionType.RELEASE_DIRECTION }), color: 'text-slate-300' },
                { label: 'REL SPRINT', action: () => ({ type: ActionType.RELEASE_SPRINT }), color: 'text-slate-300' },
                { label: 'IDLE', action: () => ({ type: ActionType.IDLE }), color: 'text-slate-400' },
              ].map((btn) => (
                <button
                  key={btn.label}
                  onClick={() => {
                    const act = btn.action();
                    onEnvStepAction(act);
                  }}
                  className={`px-1.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold ${btn.color} border border-slate-700 text-center`}
                  title={btn.label}
                >
                  {btn.label}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-slate-500 mt-2">
              Full 19-action coverage: MOVE exposes all 8 cardinal/intercardinal directions. All actions map to the discrete19_v1 action contract.
            </p>
          </div>
        </div>

        {/* 127-Float SMM Observation Vector Tensor Matrix */}
        <div className="lg:col-span-2 p-4 rounded-xl bg-slate-950/60 border border-slate-800 flex flex-col">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
              <Layers className="w-4 h-4 text-emerald-400" /> SMM Vector Inspector ({OBSERVATION_DIM} Dimensions)
            </h4>
            <div className="text-[10px] text-slate-400 font-mono">
              Shape: ({OBSERVATION_DIM},) Float32 | Base: {BASE_OBSERVATION_DIM} + Role: {ROLE_DIM}
            </div>
          </div>

          <p className="text-[11px] text-slate-400 mb-2">
            Normalized coordinates: [-1..1, -0.42..0.42], velocities, ball z-axis, ownership flags.
          </p>

          <div className="flex-1 bg-slate-900 border border-slate-800 rounded-lg p-3 overflow-y-auto max-h-56 font-mono text-[11px] leading-relaxed">
            <div className="grid grid-cols-5 sm:grid-cols-8 gap-1.5 text-center">
              {rawVector.map((val, idx) => (
                <div
                  key={idx}
                  title={`Dim [${idx}] = ${val.toFixed(4)}`}
                  className={`px-1 py-0.5 rounded text-[10px] truncate border ${
                    Math.abs(val) > 0.001
                      ? val > 0
                        ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800/60'
                        : 'bg-blue-950/80 text-blue-300 border-blue-800/60'
                      : 'bg-slate-950 text-slate-600 border-slate-800'
                  }`}
                >
                  <span className="text-[8px] opacity-40 block">{idx}</span>
                  {val.toFixed(2)}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
