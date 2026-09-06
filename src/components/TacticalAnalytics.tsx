import React from 'react';
import { MatchStats, TeamConfig } from '../types/football';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  Legend,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ZAxis,
} from 'recharts';
import { BarChart3, TrendingUp, Target, Activity, Zap } from 'lucide-react';

interface TacticalAnalyticsProps {
  stats: MatchStats;
  teamLeft: TeamConfig;
  teamRight: TeamConfig;
}

export const TacticalAnalytics: React.FC<TacticalAnalyticsProps> = ({
  stats,
  teamLeft,
  teamRight,
}) => {
  const leftPassAcc = stats.passes.left > 0
    ? Math.round((stats.completedPasses.left / stats.passes.left) * 100)
    : 0;

  const rightPassAcc = stats.passes.right > 0
    ? Math.round((stats.completedPasses.right / stats.passes.right) * 100)
    : 0;

  const comparisonData = [
    { metric: 'Shots', [teamLeft.name]: stats.shots.left, [teamRight.name]: stats.shots.right },
    { metric: 'On Target', [teamLeft.name]: stats.shotsOnTarget.left, [teamRight.name]: stats.shotsOnTarget.right },
    { metric: 'Passes', [teamLeft.name]: stats.passes.left, [teamRight.name]: stats.passes.right },
    { metric: 'Tackles', [teamLeft.name]: stats.tackles.left, [teamRight.name]: stats.tackles.right },
    { metric: 'Interceptions', [teamLeft.name]: stats.interceptions.left, [teamRight.name]: stats.interceptions.right },
  ];

  return (
    <div id="tactical-analytics-panel" className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex items-center justify-between mb-5 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-emerald-400" /> Tactical Match Analytics & Telemetry
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time possession dynamics, passing networks, and offensive conversion metrics.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Possession Dynamic Chart */}
        <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
              <TrendingUp className="w-4 h-4 text-blue-400" /> Possession History (% Over Time)
            </h4>
            <span className="text-xs text-slate-400">
              <strong className="text-blue-400">{stats.possession.left}%</strong> vs{' '}
              <strong className="text-red-400">{stats.possession.right}%</strong>
            </span>
          </div>

          <div className="h-48 w-full">
            {stats.possessionHistory.length > 1 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={stats.possessionHistory}>
                  <defs>
                    <linearGradient id="colorLeft" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1} />
                    </linearGradient>
                    <linearGradient id="colorRight" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" stroke="#64748b" tickFormatter={(v) => `${v}s`} />
                  <YAxis domain={[0, 100]} stroke="#64748b" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                  />
                                  <Area
                    type="monotone"
                    dataKey="left"
                    name={teamLeft.name}
                    stroke="#3b82f6"
                    fillOpacity={1}
                    fill="url(#colorLeft)"
                  />
                  <Area
                    type="monotone"
                    dataKey="right"
                    name={teamRight.name}
                    stroke="#ef4444"
                    fillOpacity={1}
                    fill="url(#colorRight)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-500">
                Collecting telemetry... Run match to plot possession.
              </div>
            )}
          </div>
        </div>

        {/* Head-to-Head Stats Comparison Bar Chart */}
        <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
              <Target className="w-4 h-4 text-emerald-400" /> Action Telemetry Comparison
            </h4>
            <div className="flex items-center gap-3 text-xs">
              <span className="text-blue-400 font-semibold">{leftPassAcc}% Pass Acc.</span>
              <span className="text-red-400 font-semibold">{rightPassAcc}% Pass Acc.</span>
            </div>
          </div>

          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData}>
                <XAxis dataKey="metric" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '4px' }} />
                <Bar dataKey={teamLeft.name} fill={teamLeft.color} radius={[4, 4, 0, 0]} />
                <Bar dataKey={teamRight.name} fill={teamRight.color} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Shot Location Scatter Plot */}
      <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
            <Target className="w-4 h-4 text-amber-400" /> Shot Location Heatmap
          </h4>
          <span className="text-xs text-slate-400">
            <strong className="text-blue-400">{stats.shotLocations.filter((s) => s.team === 'left').length}</strong> left /{' '}
            <strong className="text-red-400">{stats.shotLocations.filter((s) => s.team === 'right').length}</strong> right shots
          </span>
        </div>

        <div className="h-56 w-full">
          {stats.shotLocations.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  type="number"
                  dataKey="x"
                  domain={[-1.1, 1.1]}
                  stroke="#64748b"
                  tick={{ fontSize: 10 }}
                  label={{ value: 'Pitch X', position: 'insideBottom', offset: -10, fontSize: 10, fill: '#94a3b8' }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  domain={[-0.5, 0.5]}
                  stroke="#64748b"
                  tick={{ fontSize: 10 }}
                  label={{ value: 'Pitch Y', angle: -90, position: 'insideLeft', offset: 10, fontSize: 10, fill: '#94a3b8' }}
                />
                <Tooltip
                  formatter={(value: any, name: any, props: any) => {
                    const payload = props.payload;
                    if (!payload) return [value, name];
                    return [`x: ${payload.x.toFixed(2)}, y: ${payload.y.toFixed(2)}`, payload.isGoal ? 'Goal' : 'Shot'];
                  }}
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                  cursor={{ strokeDasharray: '3 3' }}
                />
                <Scatter
                  name="Left Shots"
                  data={stats.shotLocations.filter((s) => s.team === 'left')}
                  fill="#3b82f6"
                  shape={(props: any) => {
                    const { cx, cy, payload } = props;
                    if (cx == null || cy == null || !payload) {
                      return <circle cx={0} cy={0} r={0} />;
                    }
                    return (
                      <circle
                        cx={cx}
                        cy={cy}
                        r={payload.isGoal ? 7 : 5}
                        fill={payload.isGoal ? '#facc15' : '#3b82f6'}
                        fillOpacity={0.85}
                        stroke={payload.isGoal ? '#f59e0b' : '#60a5fa'}
                        strokeWidth={1.5}
                      />
                    );
                  }}
                />
                <Scatter
                  name="Right Shots"
                  data={stats.shotLocations.filter((s) => s.team === 'right')}
                  fill="#ef4444"
                  shape={(props: any) => {
                    const { cx, cy, payload } = props;
                    if (cx == null || cy == null || !payload) {
                      return <circle cx={0} cy={0} r={0} />;
                    }
                    return (
                      <circle
                        cx={cx}
                        cy={cy}
                        r={payload.isGoal ? 7 : 5}
                        fill={payload.isGoal ? '#facc15' : '#ef4444'}
                        fillOpacity={0.85}
                        stroke={payload.isGoal ? '#f59e0b' : '#f87171'}
                        strokeWidth={1.5}
                      />
                    );
                  }}
                />
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-slate-500">
              No shots recorded yet. Play a match to plot shot locations.
            </div>
          )}
        </div>
       </div>
     </div>
  );
};
