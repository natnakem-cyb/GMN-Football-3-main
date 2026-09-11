# GNN_PHASE0_ARCHITECTURE_AUDIT

**Date:** 2026-09-11  
**Scope:** Read-only source audit. No code was modified. No training was run.  
**Rule:** Every factual claim cites an exact file and line range. Claims without a citation are written as "unverified" or "not found".

---

## 1. Scenario Ground-Truth Dump

### 1.1 Registered Scenarios

All scenarios are defined in `src/scenarios/ScenarioRegistry.ts:3-421` as the `ACADEMY_SCENARIOS` array.

| # | Scenario ID | Name | Left Players | Right Players | GK Left | GK Right | Time Limit (s) | ScenarioHandler | Actual Implemented Attributes |
|---|-------------|------|--------------|---------------|---------|----------|----------------|-----------------|------------------------------|
| 1 | `academy_empty_goal` | Academy: Empty Goal | 1 | 0 | false | false | 15 | None (default) | `setup.ball`, `setup.leftPlayers` (1×ST), `setup.rightPlayers` (empty), `setup.positionJitter=0.03`, `objectives=[score_goal, within_time]`, `terminateOnOpponentPossession=true`, `rewards.scoring=1.0`, `rewards.completion=100` |
| 2 | `academy_run_to_score` | Academy: Run to Score | 1 | 2 | false | true | 20 | None (default) | `setup.ball`, `setup.leftPlayers` (1×ST), `setup.rightPlayers` (GK, CB), `setup.positionJitter=0.04`, `objectives=[score_goal, avoid_dispossess]`, `terminateOnOpponentPossession=true`, `rewards.scoring=1.0`, `rewards.completion=200` |
| 3 | `academy_pass_and_shoot_with_keeper` | Academy: Pass & Shoot with Keeper | 2 | 2 | false | true | 25 | None (default) | `setup.ball`, `setup.leftPlayers` (LW, ST), `setup.rightPlayers` (GK, CB), `setup.positionJitter=0.04`, `objectives=[complete_pass, score_goal]`, `terminateOnOpponentPossession=true`, `rewards.scoring=1.0`, `rewards.completion=350` |
| 4 | `academy_3_vs_1_with_keeper` | Academy: 3 vs 1 with Keeper | 3 | 2 | false | true | 30 | None (default) | `setup.ball`, `setup.leftPlayers` (CAM, LW, RW), `setup.rightPlayers` (GK, CB), `setup.positionJitter=0.05`, `objectives=[create_triangle, score_goal]`, `terminateOnOpponentPossession=true`, `rewards.scoring=1.0`, `rewards.completion=500` |
| 5 | `academy_3_vs_1_defender_2` | Academy: 3 vs 2 with Keeper (Variation) | 3 | 3 | false | true | 30 | None (default) | `setup.ball`, `setup.leftPlayers` (CAM, LW, RW), `setup.rightPlayers` (GK, CB, CB), `setup.positionJitter=0.05`, `objectives=[score_goal]`, `terminateOnOpponentPossession=true`, `rewards.scoring=1.0`, `rewards.completion=500` |
| 6 | `academy_3_vs_1_defender_3` | Academy: 3 vs 3 with Keeper (Variation) | 3 | 4 | false | true | 30 | None (default) | `setup.ball`, `setup.leftPlayers` (CAM, LW, RW), `setup.rightPlayers` (GK, CB, CB, CB), `setup.positionJitter=0.05`, `objectives=[score_goal]`, `terminateOnOpponentPossession=true`, `rewards.scoring=1.0`, `rewards.completion=500` |
| 7 | `academy_3_vs_1_keeper_aggressive` | Academy: 3 vs 1 Aggressive Keeper (Variation) | 3 | 2 | false | true | 30 | None (default) | `setup.ball`, `setup.leftPlayers` (CAM, LW, RW), `setup.rightPlayers` (GK at x=0.82, CB), `setup.positionJitter=0.05`, `objectives=[score_goal]`, `terminateOnOpponentPossession=true`, `rewards.scoring=1.0`, `rewards.completion=500` |
| 8 | `academy_3_vs_1_shifted` | Academy: 3 vs 1 Shifted Arrangement (Variation) | 3 | 2 | false | true | 30 | None (default) | `setup.ball`, `setup.leftPlayers` (CAM, LW, RW), `setup.rightPlayers` (GK, CB), `setup.positionJitter=0.05`, `objectives=[score_goal]`, `terminateOnOpponentPossession=true`, `rewards.scoring=1.0`, `rewards.completion=500` |
| 9 | `academy_3_vs_1_randomized` | Academy: 3 vs 1 Randomized (Variation) | 3 | 2 | false | true | 30 | None (default) | `setup.ball`, `setup.leftPlayers` (CAM, LW, RW), `setup.rightPlayers` (GK, CB), `setup.positionJitter=0.12`, `objectives=[score_goal]`, `terminateOnOpponentPossession=true`, `rewards.scoring=1.0`, `rewards.completion=500` |
| 10 | `academy_rondo_4v1` | Academy: Rondo 4v1 (Keep-Ball Drill) | 4 | 1 | false | false | 20 | `RondoScenarioHandler` | `setup.ball`, `setup.leftPlayers` (CM, CM, LW, RW), `setup.rightPlayers` (CB), `setup.positionJitter=0.02`, `objectives=[retain_possession, complete_passes]`, `terminateOnOpponentPossession=false`, `rewards.scoring=0`, `rewards.completion=0` |
| 11 | `5_vs_5` | 5 vs 5 Arena Match | 5 | 5 | true | true | 90 | None (default) | `setup.ball`, `setup.leftPlayers` (GK, CB, LM, RM, ST), `setup.rightPlayers` (GK, CB, LM, RM, ST), no `positionJitter`, `objectives=[win_match, control_possession]`, `terminateOnOpponentPossession=false`, `rewards.scoring=1.0`, `rewards.completion=800` |
| 12 | `11_vs_11` | 11 vs 11 Full Pitch Match | 11 | 11 | true | true | 180 | None (default) | `setup.ball`, `setup.leftPlayers=[]`, `setup.rightPlayers=[]`, no `positionJitter`, `objectives=[win_match, clean_sheet]`, `terminateOnOpponentPossession=false`, `rewards.scoring=1.0`, `rewards.completion=1200` |

**Handler registry:** `GameEngine.ts:100-102` defines `SCENARIO_HANDLER_REGISTRY` with only `RondoScenarioHandler.SCENARIO_ID` mapped. `GameEngine.ts:270` instantiates the handler via `GameEngine.SCENARIO_HANDLER_REGISTRY[scenario.id]?.() ?? null`. All non-rondo scenarios receive `null` and fall through to default engine logic.

**Default engine logic for non-handler scenarios:** `GameEngine.ts:511` calls `this.scenarioHandler?.onStep(this, dt, prevBallX)` (no-op when null). `GameEngine.ts:514` calls `this.scenarioHandler?.skipStandardGoalCheck()` (returns false when null, so standard goal check runs). `GameEngine.ts:563` calls `this.scenarioHandler?.computeReward(reward, this)` (returns input reward when null). `GameEngine.ts:575` calls `this.scenarioHandler?.checkExtraTermination(this)` (returns false when null). `GameEngine.ts:1192-1235` `evaluateScenarioConditions()` handles objective resolution for all scenarios.

### 1.2 Discrepancies

**DISCREPANCY 1 — `academy_3_vs_1_defender_2` name vs. actual player count:**  
The human-readable `name` is `"Academy: 3 vs 2 with Keeper (Variation)"` (`src/scenarios/ScenarioRegistry.ts:143`), but `teamRightPlayers` is `3` (lines 149), and the `rightPlayers` array contains three entries: `GK`, `CB`, `CB` (lines 160-164). The actual match is 3v3 with a goalkeeper, not 3v2.

**DISCREPANCY 2 — `academy_3_vs_1_defender_3` name vs. actual player count:**  
The `name` is `"Academy: 3 vs 3 with Keeper (Variation)"` (`src/scenarios/ScenarioRegistry.ts:179`), but `teamRightPlayers` is `4` (line 185), and the `rightPlayers` array contains four entries: `GK`, `CB`, `CB`, `CB` (lines 196-201). The actual match is 3v4 with a goalkeeper, not 3v3.

**DISCREPANCY 3 — `academy_3_vs_1_keeper_aggressive` name vs. actual player count:**  
The `name` is `"Academy: 3 vs 1 Aggressive Keeper (Variation)"` (`src/scenarios/ScenarioRegistry.ts:216`), but `teamRightPlayers` is `2` (line 222), and the `rightPlayers` array contains two entries: `GK` (at `x=0.82`) and `CB` (lines 233-236). The actual match is 3v2 with a goalkeeper, not 3v1.

**DISCREPANCY 4 — `academy_3_vs_1_shifted` name vs. actual player count:**  
The `name` is `"Academy: 3 vs 1 Shifted Arrangement (Variation)"` (`src/scenarios/ScenarioRegistry.ts:251`), but `teamRightPlayers` is `2` (line 257), and the `rightPlayers` array contains two entries: `GK` and `CB` (lines 268-271). The actual match is 3v2 with a goalkeeper, not 3v1.

**DISCREPANCY 5 — `academy_3_vs_1_randomized` name vs. actual player count:**  
The `name` is `"Academy: 3 vs 1 Randomized (Variation)"` (`src/scenarios/ScenarioRegistry.ts:286`), but `teamRightPlayers` is `2` (line 292), and the `rightPlayers` array contains two entries: `GK` and `CB` (lines 303-306). The actual match is 3v2 with a goalkeeper, not 3v1.

**DISCREPANCY 6 — Rondo "Do not shoot" instruction vs. no enforcement:**  
The rondo `description` states `"Do not shoot."` (`src/scenarios/ScenarioRegistry.ts:323`), but the `ScenarioConfig` type has no `forbidden_actions` attribute (`src/types/football.ts:197-222`), and `GameEngine.ts:711-759` `applyPlayerAction` processes `ActionType.SHOT` without any scenario-specific guard. The `RondoScenarioHandler` also does not override or block shot actions in `onStep` (`src/engine/scenarios/RondoScenarioHandler.ts:42-107`). Shooting is mechanically possible in the rondo scenario.

**DISCREPANCY 7 — `academy_3_vs_1_with_keeper` name vs. actual player count:**  
The `name` is `"Academy: 3 vs 1 with Keeper"` (`src/scenarios/ScenarioRegistry.ts:70`), and `teamRightPlayers` is `2` (line 78), with `rightPlayers` containing `GK` and `CB` (lines 89-91). The actual match is 3v2 with a goalkeeper, not 3v1.

**Unverified — One-touch enforcement:** No scenario in the registry enforces a maximum touch count. The rondo scenario's name implies keep-ball but the implementation does not limit consecutive touches per player. `not found` in code.

### 1.3 Scenario Attribute Summary

No scenario in the registry defines the following attributes:
- `max_touches` — `not found`
- `allowed_actions` — `not found`
- `forbidden_actions` — `not found`
- `target_player` — `not found`
- `target_zone` — `not found`
- `step_limit` — `not found` (only `timeLimitSeconds` exists)

The only behavioral constraints present in the `ScenarioConfig` type are:
- `timeLimitSeconds` (`src/types/football.ts:209`)
- `terminateOnOpponentPossession` (`src/types/football.ts:217`)
- `objectives` array with `id`/`text`/`isCompleted`/`isFailed` (`src/types/football.ts:190-195`)
- `rewards.scoring` and `rewards.completion` (`src/types/football.ts:218-221`)

Objective evaluation logic lives in `GameEngine.ts:1192-1269`. The `evaluateScenarioConditions()` method evaluates:
- `score_goal` when `this.score.left > 0` (line 1200)
- `within_time` when `this.matchTimeSeconds <= this.activeScenario.timeLimitSeconds` (line 1206)
- `complete_pass` when `this.stats.completedPasses.left >= 1` (line 1212)
- `create_triangle` when `this.stats.completedPasses.left >= 2` (line 1216)
- `win_match`, `control_possession`, `clean_sheet` only at time limit via `resolveMatchObjectives()` (lines 1232-1269)

---

## 2. Formation Representation Check

### 2.1 Formation Is Represented in Engine State

Formation **is** currently represented in engine state, but only as a declarative string on `TeamConfig` and a static lookup table in `Rules.ts`.

**Evidence:**

1. `TeamConfig.formation` is typed as `FormationType` (`src/types/football.ts:102`), where `FormationType = '4-3-3' | '4-4-2' | '3-5-2' | '5-3-2' | '1-2-1' | '1-1-1' | '1-0'` (`src/types/football.ts:87`).

2. `GameEngine.ts:30` sets `teamLeftConfig.formation = '4-3-3'`. `GameEngine.ts:43` sets `teamRightConfig.formation = '4-3-3'`.

3. `Rules.ts:68-139` defines `FORMATIONS: Record<FormationType, FormationNode[]>`, mapping each formation string to an array of `{ role, xRatio, yRatio }` nodes.

4. `Rules.ts:141-168` defines `getFormationPositions(formation, team, numPlayers)`, which converts formation ratios to absolute pitch coordinates using `PITCH.minX`, `PITCH.maxX`, `PITCH.height`, etc.

5. `GameEngine.ts:150-151` calls `getFormationPositions(this.teamLeftConfig.formation, 'left', numPlayers)` and `getFormationPositions(this.teamRightConfig.formation, 'right', numPlayers)` inside `initDefaultMatch()`.

### 2.2 When Formation Is Used vs. When Spawn Is Scenario-Specific

**Formation-based spawn (fallback):**  
`GameEngine.ts:289` checks `if (scenario.setup.leftPlayers.length === 0 && scenario.teamLeftPlayers > 0)`. If true, it calls `getFormationPositions(this.teamLeftConfig.formation, 'left', scenario.teamLeftPlayers)` (line 290). The same logic applies for right players at line 373.

**Scenario-specific spawn (explicit):**  
When `scenario.setup.leftPlayers.length > 0`, the engine iterates `scenario.setup.leftPlayers` directly (lines 330-370) and ignores `teamLeftConfig.formation`. The same applies for right players at lines 415-454.

**Which scenarios use formation fallback:**  
Only `11_vs_11` has `setup.leftPlayers: []` and `setup.rightPlayers: []` (`src/scenarios/ScenarioRegistry.ts:408-411`). It therefore falls back to `getFormationPositions(this.teamLeftConfig.formation, 'left', 11)` using the default `'4-3-3'` formation. All other scenarios define explicit `leftPlayers` and `rightPlayers` arrays and bypass the formation system entirely.

**Can a formation label be derived from scenario spawn positions?**  
For `11_vs_11`: yes, because it uses the `4-3-3` formation table directly.  
For all other scenarios: no, because positions are hardcoded one-offs in the `setup` object with no reusable formation concept. For example, `academy_3_vs_1_with_keeper` places players at `{x: 0.2, y: 0}`, `{x: 0.45, y: -0.22}`, `{x: 0.45, y: 0.22}` (`src/scenarios/ScenarioRegistry.ts:120-123`), which do not match any entry in `Rules.ts:68-139` `FORMATIONS`.

**Conclusion:** Formation is currently **represented** in engine state as a `FormationType` string on `TeamConfig` with a static position lookup in `Rules.ts`, but it is only used for the `11_vs_11` scenario. All other scenarios use scenario-specific hardcoded spawn coordinates. Any GNN proposal that requires formation-aware relational structure for academy scenarios must treat formation as **new-build** for those scenarios, not extraction from existing data.

---

## 3. Observation Encoder Audit

### 3.1 Exact 127-Dimension Layout

Source: `src/engine/ObservationEncoder.ts:23-192` and `src/engine/Contract.ts:1-156`.

The schema version is `simple115_v3_role` (`src/engine/Contract.ts:7`). The name `simple115` refers to the base 115-float payload; the `_v3_role` suffix indicates a 12-float role one-hot appended at the end, yielding 127 total floats (`src/engine/Contract.ts:26-27`).

| Offset | Length | Indices | Content | Source Lines |
|--------|--------|---------|---------|--------------|
| 0 | 22 | 0–21 | Left team player (x, y) positions, 11 players. Inactive slots: `-1.0, -1.0` | `ObservationEncoder.ts:94-101` |
| 22 | 22 | 22–43 | Left team player (x, y) movement direction (velocity × 50), 11 players. Inactive slots: `-1.0, -1.0` | `ObservationEncoder.ts:103-110` |
| 44 | 22 | 44–65 | Right team player (x, y) positions, 11 players. Inactive slots: `-1.0, -1.0` | `ObservationEncoder.ts:112-119` |
| 66 | 22 | 66–87 | Right team player (x, y) movement direction (velocity × 50), 11 players. Inactive slots: `-1.0, -1.0` | `ObservationEncoder.ts:121-128` |
| 88 | 3 | 88–90 | Ball (x, y, z) position | `ObservationEncoder.ts:130-131` |
| 91 | 3 | 91–93 | Ball (x, y, z) movement direction (velocity × 50) | `ObservationEncoder.ts:133-134` |
| 94 | 3 | 94–96 | Ball ownership one-hot: `[no-one, left, right]` | `ObservationEncoder.ts:136-141` |
| 97 | 11 | 97–107 | Active / controlled player one-hot over 11 players. Bug-6 fix: searches both left and right teams (lines 73–89). | `ObservationEncoder.ts:73-89, 143-146` |
| 108 | 7 | 108–114 | `game_mode` one-hot: `[Normal, KickOff, GoalKick, FreeKick, Corner, ThrowIn, Penalty]` | `ObservationEncoder.ts:148-160` |
| 115 | 12 | 115–126 | Agent's assigned role one-hot over `ROLE_VOCABULARY` (`GK, CB, LB, RB, CDM, CM, LM, RM, LW, RW, CAM, ST`). Computed by `inferPlayerRole()` with right-team x-mirroring. | `ObservationEncoder.ts:162-168`, `Contract.ts:10-23` |

**Reconciliation of `115` vs `127`:**  
`BASE_OBSERVATION_DIM = 115` (`src/engine/Contract.ts:26`) covers indices 0–114. `ROLE_DIM = 12` (`src/engine/Contract.ts:25`) covers indices 115–126. `OBSERVATION_DIM = BASE_OBSERVATION_DIM + ROLE_DIM = 127` (`src/engine/Contract.ts:27`). The `/health` endpoint reports `observation_dim: OBSERVATION_DIM` (127) and `observation_schema_version: OBSERVATION_SCHEMA_VERSION` (`simple115_v3_role`) (`src/training/bridge_server.ts:634-638`).

**Validation:** `validateObservationVector()` enforces `rawVector.length === OBSERVATION_DIM` (127) and that indices 115–126 sum to 1.0 (`src/engine/Contract.ts:118-146`).

### 3.2 GNN Input Feature Derivation

For each proposed GNN input feature, the table states whether it is (a) already directly present, (b) derivable with a formula, or (c) requires new instrumentation.

| Proposed Feature | Status | Derivation / Source |
|------------------|--------|---------------------|
| Nearest-teammate distance | **(b) Derivable** | From left-team positions (indices 0–21) and the active player index (97–107). For each present teammate `i` (where `leftPos[i].x > -0.999`), compute ` hypot( leftPos[i].x - egoPos.x, leftPos[i].y - egoPos.y )`. Ego position is extracted via the active one-hot weighted sum of `leftPos` (see `modular_encoder.ts:116-120`). |
| Nearest-opponent distance | **(b) Derivable** | From right-team positions (indices 44–65) and ego position. Same Euclidean formula against each present opponent (`rightPos[j].x > -0.999`). |
| Team width | **(b) Derivable** | From left-team y-coordinates (indices 1, 3, 5, …, 21). Compute `max(present_y) - min(present_y)` among the 11 left-team slots. |
| Team depth | **(b) Derivable** | From left-team x-coordinates (indices 0, 2, 4, …, 20). Compute `max(present_x) - min(present_x)` among present teammates. |
| Line spacing | **(c) Requires new instrumentation** | Not directly present. Would require grouping present teammates by x-depth thresholds (e.g., defensive line, midfield line, attacking line) and computing inter-line distances. No such grouping exists in the encoder. |
| Lane occupancy | **(c) Requires new instrumentation** | Not directly present. Would require discretizing pitch width into lanes and counting players per lane. No lane discretization exists in the encoder. |
| Formation deviation | **(c) Requires new instrumentation** | Not directly present. Would require comparing current player positions against a reference formation template. The `FORMATIONS` table exists in `Rules.ts:68-139`, but it is not referenced by `ObservationEncoder.ts`. |
| Compactness | **(b) Derivable** | From pairwise Euclidean distances among all present teammates. Can be computed from left-team positions (0–21) as `mean(pairwise_dist)` or `std(pairwise_dist)`. |
| Stretch | **(b) Derivable** | Equivalent to team depth (max x - min x among present teammates). |
| Receiver availability / openness | **(b) Derivable** | From left-team and right-team positions. A teammate is "open" if the nearest opponent is beyond a distance threshold. Requires computing opponent-to-teammate distances from indices 44–65 vs 0–21. |
| Pressure at pass/shot time | **(c) Requires new instrumentation** | Not directly present as a temporal feature. The observation is a per-tick flat vector with no history. Pressure would require either (a) a temporal sequence of observations, or (b) engine-side event timestamps for the last pass/shot and the defender positions at that moment. Neither exists in the current 127-float vector. |

---

## 4. Runtime / Library Feasibility Check

### 4.1 Scratch Venv pip install Results

A scratch virtual environment was created at `C:\Users\USER\AppData\Local\Temp\kilo\gnn_scratch_venv` using `python -m venv`. Both libraries were installed with `pip install` from PyPI.

**torch-geometric:**  
`pip install torch-geometric` completed successfully, installing `torch-geometric-2.8.0.post1` and its dependencies (`aiohttp`, `fsspec`, `jinja2`, `numpy`, `psutil`, `pyparsing`, `requests`, `tqdm`, `xxhash`). No native-binary compilation errors were encountered. The wheel `torch_geometric-2.8.0.post1-py3-none-any.whl` is a pure-Python package that delegates to `torch` and `torch-scatter`/`torch-sparse` at runtime.

**dgl:**  
`pip install dgl` completed successfully, installing `dgl-0.1.3`, `networkx-3.6.1`, and `scipy-1.18.1`. The wheel `dgl-0.1.3-py3-none-win_amd64.whl` installed without native-binary errors. **Note:** `dgl-0.1.3` is a legacy release from approximately 2020 and is unlikely to be compatible with modern PyTorch 2.x APIs. The install succeeded, but runtime compatibility is unverified.

**Scratch venv cleanup:** The scratch venv was removed after testing.

### 4.2 Sandbox vs. Production Environment

**Sandbox result:** Both `torch-geometric` and `dgl` install cleanly on this Windows machine with Python 3.14 (as evidenced by `cp314` wheel tags in the pip output).

**Production environment:** The actual training environment is defined by `training/requirements.txt:1-10`, which pins `torch>=2.0.0` but does **not** include `torch-geometric` or `dgl`. The production environment would need these dependencies added to `requirements.txt` and any native backend libraries (e.g., `torch-scatter`, `libtorch` bindings) would need to be verified on the target OS/architecture. Whether a typed/heterogeneous graph library is usable in production is **unverified** because the production Python version, OS, and CUDA availability were not tested.

---

## 5. Baseline Preservation Check

### 5.1 Where a `use_gnn` Flag Would Need to be Threaded

To allow `use_gnn=false` to run the existing observation pipeline completely unchanged, the flag must be threaded through the following exact locations:

**`training/train_mappo.py`:**
- CLI argument parser: `argparse` section at lines 871–887. A new `--use-gnn` flag would be added here.
- `run_mappo_training()` signature: lines 43–59. A `use_gnn: bool = False` parameter would be added.
- Environment construction: lines 170–191. The flag would be passed to `GMNMultiAgentEnv(..., use_gnn=use_gnn)`.
- Network construction: lines 210–211. `SharedActor(obs_dim=obs_dim, ...)` and `CentralizedCritic(obs_dim=obs_dim, ...)` would be replaced with conditional construction: when `use_gnn=False`, the exact same `SharedActor`/`CentralizedCritic` classes are instantiated with `obs_dim=OBSERVATION_DIM` (127). When `use_gnn=True`, a GNN-based actor/critic is instantiated instead.

**`training/gmn_pettingzoo.py`:**
- `GMNMultiAgentEnv.__init__()`: lines 359–448. A `use_gnn: bool = False` parameter would be added. When `False`, the environment uses the existing binary WebSocket protocol. When `True`, it would request graph-structured observations from the bridge.
- Observation space definition: lines 414–428. The `spaces.Dict` with `spaces.Box(shape=(OBSERVATION_DIM,))` must remain the default path. A graph observation space would only be added when `use_gnn=True`.

**`training/mappo_networks.py`:**
- `SharedActor.__init__()`: line 17. Takes `obs_dim`, `action_dim`, `hidden`. A GNN actor would be a separate class or a conditional branch here.
- `CentralizedCritic.__init__()`: lines 44–81. Same conditional pattern.
- New GNN classes would be added alongside the existing `SharedActor` and `CentralizedCritic`.

**`src/engine/ObservationEncoder.ts`** (if graph features are added):
- The existing `encode()` static method at lines 40–192 must remain unchanged for `use_gnn=False`. A new method (e.g., `encodeGraph()`) would be added for the GNN path.

**`src/training/bridge_server.ts`** (if graph serialization is needed):
- The existing `encodeStepBinary()` (lines 888–935), `encodeMultiStepBinary()` (lines 950–1000), and `encodeBatchedStepBinary()` (lines 1011–1075) must remain unchanged for `use_gnn=False`. A new binary layout or JSON graph payload would be added for the GNN path.

### 5.2 Overlap with Verified Systems

The three verified systems and their file locations:

1. **Bridge IPC vectorization:**  
   `src/training/bridge_server.ts:888-1075` (`encodeStepBinary`, `encodeMultiStepBinary`, `encodeBatchedStepBinary`) and `training/gmn_pettingzoo.py:612-763` (`step_batch`, `_recv_batch_response`, `_recv_step_response`).

2. **Opponent-pool ONNX inference:**  
   `src/training/bridge_server.ts:98-125` (`getOrCreateSnapshotSession`, `getSnapshotSession`) and `src/training/bridge_server.ts:169-197` (`runOnnxInference`).

3. **Rondo / GameEngine decoupling:**  
   `src/engine/GameEngine.ts:94-102` (`scenarioHandler` field and `SCENARIO_HANDLER_REGISTRY`) and `src/engine/scenarios/RondoScenarioHandler.ts` (full file).

**Overlap assessment:**

- **Bridge IPC vectorization:** A `use_gnn=false` path does **not** touch `encodeStepBinary`, `encodeMultiStepBinary`, `encodeBatchedStepBinary`, or any of the binary frame parsing in `gmn_pettingzoo.py`. The existing 18-byte header + 127×4-byte observation + 19-byte mask layout is preserved byte-for-byte. **No re-verification required.**

- **Opponent-pool ONNX inference:** A `use_gnn=false` path does **not** touch `runOnnxInference` or `getOrCreateSnapshotSession`. The ONNX opponent path still reads the 127-float `mirrorObservationForRightTeam` observation from `ObservationEncoder.encode()` (`src/training/bridge_server.ts:170-181`). **No re-verification required.**

- **Rondo / GameEngine decoupling:** A `use_gnn=false` path does **not** touch `GameEngine.ts` scenario handler logic or `RondoScenarioHandler.ts`. The `ScenarioHandler` interface and its null-fallback behavior are unchanged. **No re-verification required.**

**Conclusion:** A `use_gnn=false` default preserves all three verified systems exactly. The GNN toggle only activates new code paths in `train_mappo.py`, `gmn_pettingzoo.py`, `mappo_networks.py`, and (optionally) `ObservationEncoder.ts` and `bridge_server.ts`. None of these files overlap with the bridge IPC vectorization, opponent-pool ONNX inference, or rondo/GameEngine decoupling code paths in a way that would require re-verification of those systems.

---

## 6. Additional Findings

### 6.1 Schema Version Reconciliation

The `/health` endpoint reports `observation_schema_version: "simple115_v3_role"` and `observation_dim: 127` (`src/training/bridge_server.ts:634-638`). The `OBSERVATION_SCHEMA_VERSION` constant is `"simple115_v3_role"` and `OBSERVATION_DIM` is `127` (`src/engine/Contract.ts:7, 27`). The Python client validates `bridge_obs_dim != OBSERVATION_DIM` at connection time (`training/gmn_pettingzoo.py:795-799`). The name `simple115` refers to the 115-float base vector (indices 0–114); the `_v3_role` suffix denotes the appended 12-float role one-hot (indices 115–126).

### 6.2 Scenario Objectives Are Not Enforced as Hard Constraints

Scenario objectives (`score_goal`, `within_time`, `complete_pass`, etc.) are evaluated post-hoc in `GameEngine.ts:1192-1269` and stored as `isCompleted`/`isFailed` flags on the `ScenarioConfig.objectives` array. They do **not** gate the physics engine or block actions. For example, the rondo objective `retain_possession` is evaluated only by the time-limit check (`GameEngine.ts:1222-1227`) and the `resolveMatchObjectives()` call (which is skipped for non-match scenarios). There is no mechanism to force a reset or apply a penalty when an objective is failed mid-episode.

### 6.3 No Formation Data in Observation Vector

The 127-float observation contains no formation id, line assignment, or role-to-position mapping. The only tactical signal is the 12-float role one-hot at indices 115–126 (`ObservationEncoder.ts:162-168`). Formation geometry is available in `Rules.ts:68-139` but is never injected into the observation.

---

*End of audit. No source files were modified.*
