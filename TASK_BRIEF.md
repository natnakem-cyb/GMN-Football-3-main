# GMN-Football-3 — Task Brief

> Current as of 2026-09-04. Project root: `C:\Users\USER\Documents\Project\GMN-Football-3-main\` (flattened — `package.json` sits directly here). Runtime requires two concurrent processes: **`npm run bridge`** (port 5050) + **`npm run dev`** (port 3000). Open the app at http://localhost:3000.

---

## Current verified state

- [x] **TS typecheck clean** (`npm run lint` / `tsc --noEmit` -> exit 0).
- [x] **127-dim role-aware observation** — `ObservationEncoder` emits `BASE_OBSERVATION_DIM (115) + ROLE_DIM (12) = 127` floats, contract-validated on every encode.
- [x] **19-action discrete space**, 11 seeded academy scenarios (empty-goal to 11v11).
- [x] **Possession chart** rendered in `TacticalAnalytics.tsx` from `stats.possessionHistory` (sampled every 60 ticks, rolling 30-sample window).
- [x] **Checkpoint bias verified** — `src/agents/mappo_weights.ts` `b2` matches the exported `net.4.bias` of the trained checkpoint exactly: `[-0.0500, -0.0118, 0.1311, 0.1118, -0.0677, -0.0249, 0.0402, 0.0971, -0.0332, -0.2066, -0.0611, 0.0052, -0.0673, -0.0820, -0.0142, 0.1224, -0.1343, 0.0129, 0.0497]`.
- [x] **Deterministic PRNG** — `SeededRNG` (Mulberry32) in place.
- [x] **Reward shaping**: goal +/-1.0, monotonic checkpoint <= +0.05, shot bonus +0.03 — implemented in `ObservationEncoder.computeReward`.

---

## Tasks

### Task 0 — Environment and observation-engine verification COMPLETE

Verified working. The shared TS engine (`GameEngine`, `Physics`, `Rules`, `ObservationEncoder`) drives both the browser match and the Python headless RL bridge. No action required.

**Exit criteria (all met):**
1. TypeScript compiles cleanly under `strict` mode.
2. `ObservationEncoder.encode()` produces exactly 127 floats, schema-version tagged, with per-encode validation.
3. `computeReward` delivers goal/concede +1.0/-1.0, monotonic ball-progress checkpoint, shot bonus.
4. `stats.possession` + `stats.possessionHistory` computed in engine and surfaced to `TacticalAnalytics.tsx`.
5. `mappo_weights.ts` bias vector matches the trained checkpoint export byte-for-byte.
6. Deterministic seeded RNG (Mulberry32) used throughout.

---

### Task 1 — Reproduce, understand, and improve the RL training pipeline

The Python `training/` pipeline trains MAPPO/PPO/IPPO policies against the bridge. Goal: a reproducible, documented training run producing a trained `.pt`/`.zip` artifact plus evaluation telemetry.

#### 1.1 Verify checkpoint bias against the in-browser weights DONE
```powershell
python -c "import torch; ckpt=torch.load('training/models/mappo_academy_3_vs_1_with_keeper_smoke.pt', map_location='cpu'); print(ckpt['actor']['net.4.bias'].tolist())"
```
**Result:** PASS — the in-browser policy head `w2`/`b2` faithfully reproduces the trained checkpoint. (Note: the in-browser weights come from `mappo_academy_3_vs_1_with_keeper_trained.pt` at 49,920 steps; the bundled `smoke.pt` is a 4,864-step scratch checkpoint with different weights.)

#### 1.2 Run a short deterministic training smoke test
```powershell
npm run bridge
python training/train_mappo.py --timesteps 5000 --checkpoint-name training/models/mappo_smoke_test.pt
python training/train_ppo.py --timesteps 1000 --scenario academy_empty_goal
python training/train_ippo.py 3072
```
**Exit criteria:** each run completes with "SUCCESSFUL" / "COMPLETED SUCCESSFULLY", writes a checkpoint to `training/models/`, appends eval rows to `training/results/win_rate_progress.csv`, no Python exceptions.

#### 1.3 Run a full MAPPO training run (Stage 2 — the real target)
```powershell
python training/train_stage2_ppo.py
# OR raw MAPPO from scratch:
python training/train_mappo.py --timesteps 200000 --scenario academy_3_vs_1_with_keeper
```
**Exit criteria:** produces `models/mappo_academy_3_vs_1_with_keeper_trained.pt`, prints progressive eval metrics (success rate climbing from ~0% toward >=50%), final `success_rate >= 40%` over 100 eval episodes.

#### 1.4 Export the freshly trained policy to ONNX for in-browser inference
```powershell
python training/export_onnx.py --checkpoint training/models/mappo_academy_3_vs_1_with_keeper_trained.pt --out public/models/mappo_policy.onnx
```
**Exit criteria:** valid `public/models/mappo_policy.onnx` produced; `onnxruntime-web` loads it in the browser via `TrainedPolicyAgent`.

#### 1.5 Understand and document the architecture
- `training/modular_networks.py` — modular entity encoders (Ego / Ball / Teammates / Opponents / Match) + GRU/LSTM recurrent core + actor/critic heads.
- `training/mappo_networks.py` — simple MLP shared actor (64x64) + permutation-invariant Deep-Sets critic.
- `training/mappo_rollout.py` / `mappo_update.py` — GAE + PPO clipped surrogate.
- `training/eval_generalization.py` — multi-formation generalization probe.
- `training/gmn_gym.py` — Gymnasium wrapper over the bridge WS.
- `training/gmn_pettingzoo.py` — PettingZoo multi-agent wrapper.

**Exit criteria:** a `training/ARCHITECTURE.md` (or README addition) that traces a single observation to action at inference time through both the simple (Mlp64) and modular (TiZero) paths.

#### 1.6 Run the headless determinism check
```powershell
python training/test_env.py
python training/test_pettingzoo_wrapper.py
python training/test_multiagent_determinism.py
```
**Exit criteria:** all three tests exit 0; multi-agent determinism produces identical trajectories under the same seed.

---

### Task 2 — Export the trained model to ONNX and integrate it into the agent UI

The app ships a pre-exported `public/models/mappo_policy.onnx` (loaded by `src/agents/TrainedPolicyAgent.ts` via `onnxruntime-web`). Task 2 makes this a living pipeline: train, export, drop, play.

#### 2.1 Export from a freshly trained checkpoint (covers 1.4)
Already covered in Task 1.4.

#### 2.2 Verify the browser loads and runs the new ONNX checkpoint
1. Start the app: `npm run dev`.
2. Pick a scenario (e.g. Academy 3 vs 1 with Keeper).
3. Set the left controller to Neural (`controller: 'neural'`).
4. Confirm in the `RLGymnasiumPanel` overlay that `onnxruntime-web` reports `modelLoaded=true` and shows a 127-dim obs to 19-action logit stream.
5. Verify the on-pitch agent exhibits non-trivial behavior (moves toward ball, attempts shots).

**Exit criteria:** no browser console errors; agent produces shot attempts within 200 steps in >=3 of 5 rollouts.

#### 2.3 Add a Hot-swap ONNX UI control
Currently the ONNX path is hardcoded (`public/models/mappo_policy.onnx`). Add a file-picker / drag-drop on the `RLGymnasiumPanel` that:
- accepts a local `.onnx` file,
- writes it to `public/models/mappo_policy.onnx` (or hot-swaps the model URL),
- reloads the session in `TrainedPolicyAgent`,
- shows the new models SHA-256 and timestamp.

**Exit criteria:** dropping a freshly exported `.onnx` into the panel swaps the live policy without restarting the app.

---

### Task 3 — Visualization of training telemetry, agent behavior, and match analysis

The app already has a rich component library. Task 3 closes specific gaps.

#### 3.1 Verify possession chart DONE
`TacticalAnalytics.tsx:57-103` renders the recharts `<AreaChart>` of `stats.possessionHistory`. Confirmed wired to `engine.stats.possession` (computed from `possessionTicks` in `GameEngine.ts:1126-1142`).

#### 3.2 Add a formation overlay toggle to the pitch view
`PitchCanvas.tsx` draws players + ball. Add a toggle that overlays:
- the `FormationNode` grid (x/y ratios) from `engine.teamLeftConfig.formation` / `teamRightConfig.formation`,
- the role label per player (GK / CB / LB / RB / CDM / CM / LM / RM / LW / RW / CAM / ST),
- the current offside line (already tracked in `Rules.ts`).

**Exit criteria:** toggling the overlay in `MatchControls.tsx` paints the formation skeleton + role letters on the pitch without affecting match rendering.

#### 3.3 Add shot-location heatmaps to `MatchStats.heatmapData`
`GameEngine.ts` initializes `heatmapData: { left: [], right: [], ball: [] }` in `MatchStats` but never populates it. Extend the shot event handler to push `{x, y, team, isGoal}` into `heatmapData` and render a `recharts` scatter/heatmap on `TacticalAnalytics.tsx`.

**Exit criteria:** after a match with shots, the heatmap tab shows a 2D pitch plot colored by shot density.

#### 3.4 TrainingTelemetryDashboard — live reward/curve chart
`TrainingTelemetryDashboard.tsx` exists but currently reads from a static sample stream. Wire it to the bridge WebSocket telemetry feed so it plots:
- rolling episode reward (left axis),
- goal rate % (right axis),
- current timestep / step count,
- live loss metrics from `training/logs/`.

**Exit criteria:** during a `train_mappo.py` run with the dashboard open, reward/goal curves update in real time (<5 s latency).

#### 3.5 Scenario-objective progress bar
`ScenarioSelector.tsx` shows objective text with checkmarks/X but no progress percentage. Add a progress bar computing `objectivesCompleted / objectivesTotal` for the active scenario.

**Exit criteria:** the active drill shows a live progress % badge that updates as objectives are met.

---

### Task 4 — Headless-vs-browser match parity and determinism guarantees

The project's core claim is that the same `GameEngine` drives both the human-playable match and the headless RL rollouts. Task 4 hardens this.

#### 4.1 Run the existing parity tests
```powershell
npm run test:transport-parity
npm run test:env
npm run test:determinism
```
**Exit criteria:** all exit 0.

#### 4.2 Re-run the headless determinism check (from 1.6)
Covered in Task 1.6.

#### 4.3 Verify reward-shaping equivalence
Confirm that the reward reported by `bridge_server.ts` after a goal matches `ObservationEncoder.computeReward(+1.0)` exactly (no drift between TS and Python float encoding). Read `bridge_server.ts:159,233,461,495` — `checkpointReward` is `result.info.checkpointReward` passed through unchanged.

**Exit criteria:** a unit test or manual diff shows zero reward discrepancy across 1000 steps of a scripted scenario.

#### 4.4 Stress test: long-run stability
Run a 100k-step headless MAPPO training run and confirm:
- no WebSocket frame desync (the `gmn_gym.py` / `gmn_pettingzoo.py` broadcast-skip fix is in place),
- no memory leak (RSS stable over run),
- no TypeScript contract violations (the `validateObservationVector` guard never trips).

**Exit criteria:** 100k steps complete without a frame-decode error or contract exception.

---

### Task 5 — Evaluation, documentation, and final packaging

#### 5.1 Final smoke-test sweep
```powershell
npm run test
python training/test_env.py
python training/test_pettingzoo_wrapper.py
python training/test_multiagent_determinism.py
```
**Exit criteria:** every test script exits 0.

#### 5.2 Training reproduction log
Produce a `training/REPRODUCTION_LOG.md` capturing:
- host OS, Python version, GPU vs CPU,
- seed used,
- total wall-clock time,
- final success_rate, mean_reward, mean_steps_to_goal,
- checkpoint SHA-256,
- any deviations from the README procedure.

#### 5.3 Update README sections that lag the code
The README's "Known Caveats" section flags that only `academy_3_vs_1_with_keeper` has a trained checkpoint and that training is single-sided. After a successful Stage-2 run, update:
- the trained-checkpoint table,
- the curriculum section,
- the generalization results (from `eval_generalization.py`).

#### 5.4 Package the final trained artifacts
- `training/models/mappo_academy_3_vs_1_with_keeper_trained.pt`
- `public/models/mappo_policy.onnx` (exported from the above)
- `training/results/win_rate_progress.csv` (full training curve)
- `training/REPRODUCTION_LOG.md`

**Exit criteria:** the four artifacts exist, the ONNX loads in-browser, the `.pt` reloads in Python and produces equivalent actions (Pearson r > 0.99 on logits).

---

## Suggested execution order

| Order | Task | Depends on | Approx effort |
|---|---|---|---|
| 1 | 0 — verify | — | done |
| 2 | 1.1 — checkpoint bias | — | done |
| 3 | 1.2 — smoke training runs | bridge running | 15 min |
| 4 | 1.6 + 4.1 + 4.2 — test sweep | 1.2 | 30 min |
| 5 | 1.3 — full MAPPO run | 1.2 green | hours (GPU) / overnight (CPU) |
| 6 | 1.4 / 2.1 — ONNX export | 1.3 | 5 min |
| 7 | 2.2 — in-browser inference | 6 | 30 min |
| 8 | 2.3 — hot-swap UI | 7 | 1-2 hr |
| 9 | 3.2 — formation overlay | — | 1 hr |
| 10 | 3.3 — shot heatmap | — | 1-2 hr |
| 11 | 3.4 — live telemetry dashboard | bridge WS | 2 hr |
| 12 | 3.5 — scenario progress bar | — | 30 min |
| 13 | 4.3 — reward equivalence | 1.2 | 30 min |
| 14 | 4.4 — 100k stress test | 1.3 | hours |
| 15 | 5.1 — final test sweep | all above | 30 min |
| 16 | 5.2 + 5.3 + 5.4 — docs + package | 15 | 1 hr |

Total: roughly 1-2 days of focused work, dominated by the full training run (step 5) and the 100k stress test (step 14).

---

## Environment notes (this machine)

- **Python** — the project's training deps (gymnasium, SB3, torch, supersuit, pettingzoo, onnx) are installed on the **globally-registered Python 3.10.11** at `C:\Users\USER\AppData\Local\Programs\Python\Python310`. The bundled `.venv` lacks the RL deps — do NOT use it for training scripts.
- **PATH fix** — `python` / `python3` resolve to 3.10.11 via USER PATH. In a fresh terminal this just works; in the current session it was injected per-command.
- **Windows spawn quirk** — `vite.config.ts` was patched (`shell: true` on Windows) so `npm run dev` auto-launches the bridge on port 5050. No separate `npm run bridge` step needed unless you prefer it.
- **Checkpoint bias** — `mappo_weights.ts` line 22 (`b2`) matches the trained checkpoint exactly — this is the single source of truth for the in-browser policy head.