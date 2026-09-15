# README Audit & Correction Plan

## Goal
Audit `README.md` against the actual codebase and fix architectural description and diagram errors while preserving the existing look, feel, icons, emojis, and SEO structure.

## Identified Mistakes & Required Corrections

### 1. `mappo_weights.ts` is falsely claimed as `@deprecated` and unused in the live path
- **Location:** Technology Stack table, line ~175-176
- **Current text:** "The older hand-rolled MLP path (`src/agents/mappo_weights.ts`) is explicitly `@deprecated` in the file itself and retained only for offline reference / test parity, not used in the live decision path."
- **Fact:** `mappo_weights.ts` contains no `@deprecated` annotation. `TrainedPolicyAgent.ts` imports `MAPPO_WEIGHTS` and uses it in `predictDiscreteAction()` and `assertMappoWeightsValid()`. The `act()` method dispatches to `actOnnx()` for `Float32Array` and `predictDiscreteAction()` for `number[]`, meaning embedded weights are part of the live decision path.
- **Correction:** State that `mappo_weights.ts` provides an embedded-weight fallback path. `TrainedPolicyAgent` uses ONNX Runtime Web when available and falls back to embedded weights when the session is unavailable or `number[]` observations are received. Remove the false `@deprecated` claim.

### 2. Architecture Diagram 1 (Overview) omits the bridge
- **Location:** "Overview" section, first diagram
- **Current:** Shows direct flow: HUMAN → Browser UI / AI → RL Environment → SB3 PPO / IPPO / MAPPO → PyTorch
- **Fact:** The TypeScript `GameEngine` is authoritative but is accessed by the Python training pipeline exclusively through the Node.js bridge (`training/bridge_server.ts`) over HTTP/WebSocket. There is no direct connection from AI to RL Environment.
- **Correction:** Update the first diagram to include the bridge between the authoritative engine and the Python RL pipeline.

### 3. Architecture Diagram 2 misrepresents transport and environment structure
- **Location:** "Architecture Diagram" section, second diagram
- **Current:** Shows HTTP bridge and Binary WebSocket as parallel branches after the bridge, then sequentially shows Gymnasium env followed by PettingZoo env.
- **Fact:** 
  - HTTP and binary WebSocket are two alternative transports exposed by the same `bridge_server.ts`, not sequential branches.
  - `gmn_gym.py` and `gmn_pettingzoo.py` are two separate environment wrappers, not sequential steps.
- **Correction:** Redraw diagram so HTTP and WebSocket are shown as alternative transports from the bridge. Show Gymnasium and PettingZoo as parallel wrappers around the bridge, not in sequence.

### 4. `npm run lint` description is incomplete
- **Location:** Quick Start section
- **Current:** "`npm run lint` (`tsc --noEmit`)"
- **Fact:** `package.json` shows `"lint": "tsc --noEmit && npm run check:contracts"`
- **Correction:** Update to `tsc --noEmit && npm run check:contracts`.

### 5. `npm run build` omits `sync-contracts`
- **Location:** Quick Start section
- **Current:** "`npm run build` — tsc (src/ and training/) + vite build"
- **Fact:** `package.json` shows `"build": "npm run sync-contracts && tsc && vite build"`
- **Correction:** Update to note that `sync-contracts` runs before `tsc`.

### 6. Hot-swap is attributed to `RLGymnasiumPanel` but logic lives in `App.tsx`
- **Location:** Technology Stack table
- **Current:** "The `RLGymnasiumPanel` supports hot-swapping `.onnx` models at runtime without restarting the app."
- **Fact:** `RLGymnasiumPanel.tsx` provides the file-picker UI. The actual hot-swap is triggered via `App.tsx` calling `trainedAgentRef.current.switchModel()`. `TrainedPolicyAgent.switchModel()` performs the session swap.
- **Correction:** Attribute hot-swap to `App.tsx` / `TrainedPolicyAgent.switchModel()`, with `RLGymnasiumPanel` providing the UI.

### 7. PPO parallel stepping description conflates PPO and MAPPO implementations
- **Location:** Technology Stack table and "Running the RL Training Pipeline"
- **Current:** "PPO and MAPPO now support optional parallel stepping via `--n-envs N` (spawns N bridge instances on ports 5050..5050+N-1)"
- **Fact:**
  - `train_ppo.py` uses `DummyVecEnv` with per-env bridge instances on distinct ports (same-process vectorization).
  - `train_mappo.py` uses a single batched bridge with pooled engines (`collect_rollout_batched`) when `n_envs > 1`.
- **Correction:** Separate the descriptions: PPO uses `DummyVecEnv` with per-env bridge instances; MAPPO uses a single batched bridge with pooled engines.

### 8. `gmn_pettingzoo.py` docstring says `Box(115,)` but code uses 127
- **Location:** Environment Contract section (if propagated) and Technology Stack
- **Fact:** `gmn_pettingzoo.py` docstring line 7 says `Box(115,)`, but `OBSERVATION_DIM` is 127 throughout the actual code.
- **Correction:** Do not propagate the stale `Box(115,)` docstring claim.

### 9. Canonical action mapping source misattributed
- **Location:** Environment Contract section
- **Current:** "see `training/action_mapping.ts` for the canonical mapping"
- **Fact:** `training/action_mapping.ts` is a thin re-export of `src/engine/ActionMapping.ts`. The canonical source is `src/engine/ActionMapping.ts`.
- **Correction:** Reference `src/engine/ActionMapping.ts` as canonical; note `training/action_mapping.ts` re-exports it.

### 10. Engine tree omits several files and subdirectories
- **Location:** Repository Structure section, `src/engine/` tree
- **Current:** Lists `GameEngine.ts`, `Physics.ts`, `Rules.ts`, `ObservationEncoder.ts`, `SeededRNG.ts`, `Contract.ts`, `EventEncoder.ts`, `Vector.ts`
- **Fact:** `src/engine/` also contains `TrainingTelemetryService.ts`, `TaskEncoder.ts`, `FootballMetrics.ts`, `ActionMapping.ts`, and `scenarios/` subdirectory (`ScenarioHandler.ts`, `RondoScenarioHandler.ts`).
- **Correction:** Expand the tree to include these files.

### 11. `/health` cross-check phrasing is inaccurate
- **Location:** Environment Contract section
- **Current:** "the bridge's `/health` endpoint cross-checks `observation_dim`/`action_space_size` at connection time and raises if they disagree"
- **Fact:** The bridge's `/health` endpoint reports dimensions via `getInfo()`. The cross-check is performed by the Python env wrappers (`gmn_gym.py` `_ensure_bridge_running`, `gmn_pettingzoo.py`) against the `/health` response, not by the bridge itself.
- **Correction:** Clarify that Python env wrappers cross-check their local constants against the bridge's `/health` response.

### 12. `TrainedPolicyAgent` ONNX claim understates embedded-weight fallback
- **Location:** Technology Stack table and Current Status
- **Current:** "runs real ONNX inference for the in-browser 'Neural' controller"
- **Fact:** `TrainedPolicyAgent` also uses embedded weights via `predictDiscreteAction()` when ONNX is unavailable.
- **Correction:** Acknowledge the dual path (ONNX primary, embedded weights fallback).

### 13. `README.md` technology stack mentions GNN Phase 4 but omits integration detail
- **Location:** Technology Stack table
- **Current:** "Phase 4 GNN encoder complete"
- **Fact:** GNN files exist and are validated. This is accurate but could note they are wired into `train_mappo.py` via the `gnn_*` modules.
- **Correction:** Keep the claim but ensure it doesn't overstate adoption. Verified files exist.

## Validation Steps
1. Re-run `npm run lint` to confirm `tsc --noEmit && npm run check:contracts` passes.
2. Verify `src/agents/mappo_weights.ts` contains no `@deprecated` annotation.
3. Verify `TrainedPolicyAgent.ts` imports and uses `MAPPO_WEIGHTS` in `predictDiscreteAction()` and `assertMappoWeightsValid()`.
4. Verify `src/engine/ActionMapping.ts` is the canonical source and `training/action_mapping.ts` is a re-export.
5. Verify `src/engine/` contains `TrainingTelemetryService.ts`, `TaskEncoder.ts`, `FootballMetrics.ts`, `ActionMapping.ts`, and `scenarios/`.
6. Verify `RLGymnasiumPanel.tsx` delegates to `onSwitchModel` callback and `App.tsx` calls `TrainedPolicyAgent.switchModel()`.
7. Verify `train_ppo.py` uses `DummyVecEnv` and `train_mappo.py` uses `collect_rollout_batched` for `n_envs > 1`.
8. Verify `gmn_pettingzoo.py` docstring still says `Box(115,)` at line 7 (stale) while code uses `OBSERVATION_DIM` (127).

## Out of Scope
- Do not modify source code files.
- Do not change the visual layout, icons, emojis, or markdown formatting style.
- Do not add or remove sections from the README.
