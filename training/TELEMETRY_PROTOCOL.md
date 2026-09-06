# GMN-Football-3 — Canonical Telemetry Protocol (Stabilization Release)

Single source of truth for ALL telemetry messages exchanged between the bridge
(`training/bridge_server.ts`), `MetricsBroadcaster`, `TrainingJobService`, and
the frontend (`src/engine/TrainingTelemetryService.ts`).

## Envelope

Every telemetry message is JSON with exactly two fields:

```json
{ "type": "<message_type>", "data": { ... } }
```

Lowercase message names. No `payload` / `snapshot` field names.

## Message types

| type                 | producer                        | data shape |
|----------------------|---------------------------------|------------|
| `training_status`    | TrainingJobService, bridge      | `{ isRunning, currentJob, latestMetrics, recentLogs }` |
| `training_started`   | TrainingJobService              | job info object |
| `training_output`    | TrainingJobService, MetricsBroadcaster | `{ line, ts }` |
| `training_metrics`   | TrainingJobService, MetricsBroadcaster | `TrainingMetricsSnapshot` |
| `episode_metrics`    | (reserved) same shape as training_metrics, per-episode granularity | snapshot |
| `hardware_stats`     | MetricsBroadcaster              | `{ cpuPercent, ramPercent, ramUsedMb, ramTotalMb, stepsPerSec }` |
| `checkpoint_update`  | MetricsBroadcaster, bridge      | `{ checkpoints: CheckpointInfo[] }` |
| `training_completed` | TrainingJobService              | `{ jobId, exportResult? }` |
| `training_failed`    | TrainingJobService              | `{ jobId, error }` |
| `training_stopped`   | TrainingJobService              | `{ jobId }` |
| `error`              | bridge                          | `{ message }` (legacy `error` string field also set for compat) |

## Subscription

- Frontend connects and sends `{ "type": "subscribe_training" }` — explicitly
  handled by the bridge (subscribes the socket to the MetricsBroadcaster stream
  AND keeps it in the TrainingJobService RL-client set, then replies with the
  canonical initial `training_status` + `checkpoint_update`).
- `{ "type": "subscribe_metrics" }` / `{ "type": "unsubscribe_metrics" }`
  remain supported for lightweight metrics-only subscribers.
- WS URL query `?type=metrics` auto-subscribes (legacy, still supported).

## Endpoint configuration (frontend)

- `VITE_WS_URL` build-time env var wins (production / remote bridge).
- Otherwise same-origin `/ws` (Vite dev proxy → ws://127.0.0.1:5050; any
  production reverse proxy must forward `/ws` and `/api/*` to the bridge).
- REST endpoints (`/api/...`) follow the same same-origin proxy model.

## Deprecated

- `telemetry_metrics` relay input is still accepted by the bridge but is
  normalized to a canonical `training_metrics` frame (no Python producers remain).
- Uppercase dialect (`TRAINING_STATUS`, `payload`, ...) has been removed.

## Simulation timestep (section 16 note)

The physics is a fixed-timestep simulation at **60 Hz** (`PhysicsEngine.FIXED_DT
= 1/60`). The training bridge always steps with `dt = 1/60`. The `dt` parameter
on `GameEngine.step` / `PhysicsEngine.updateBall` is NOT generalized: drag,
friction and smoothing are per-tick factors. Callers must use the fixed
timestep; unsupported values are documented, not silently supported.
