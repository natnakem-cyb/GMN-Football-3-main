# GNN Phase 3 Pre-Implementation Audit

## 1. Authoritative Sources

| Concept | Authoritative Source | Key Lines |
|---------|---------------------|-----------|
| Ball ownership / controlled player | `src/engine/GameEngine.ts` | `ball.ownerId` (line 131), `controlledPlayerId` (line 22) |
| 127-dim observation layout | `src/engine/ObservationEncoder.ts` | `encode()` lines 40–192; offsets documented lines 27–38 |
| zScenario task vector | `src/engine/TaskEncoder.ts` | `encode()` lines 6–47; `TASK_VECTOR_DIM = 8` (line 3) |
| Scenario definitions | `src/scenarios/ScenarioRegistry.ts` | `ACADEMY_SCENARIOS` lines 3–435 |
| Formation templates / pitch geometry | `src/engine/Rules.ts` | `FORMATIONS` lines 68–139; `PITCH` lines 6–40; `getFormationPositions()` lines 141–168 |
| Player / Ball / Scenario types | `src/types/football.ts` | `Ball` lines 60–78; `ScenarioConfig` lines 233–259; `ScenarioDynamicState` lines 222–231 |
| Rondo-specific semantics | `src/engine/scenarios/RondoScenarioHandler.ts` | state fields lines 16–27; `onStep()` lines 44–117 |
| Bridge binary frame layout | `training/bridge_server.ts` | `encodeStepBinary()` lines 894–941; `encodeMultiStepBinary()` lines 956–1006 |
| PettingZoo info dict shape | `training/gmn_pettingzoo.py` | `stepBatch()` lines 613–763; `_recv_step_response()` lines 897–968 |
| Graph schema (v3) | `training/gnn_graph_schema.json` | title "GNN Graph Schema v3" (line 4); `$id` "gnn_graph_schema_v1.json" (line 3) |
| Graph construction | `training/gnn_graph_builder.py` | `build_graph()` lines 1198–1328 |

**Note:** `training/gnn_graph_schema.json` is versioned as **v3** in its `title` but its `$id` field still reads `gnn_graph_schema_v1.json` (line 3). This is a metadata inconsistency that does not affect validation but should be resolved before Phase 3 implementation.

---

## 2. Concept Availability Matrix

| Concept | Available in Engine | Exposed Through Bridge / Info | Currently in Graph | Needs Implementation |
|---------|--------------------|-------------------------------|--------------------|----------------------|
| **ball owner** | `GameEngine.ts:131` — `ball.ownerId: string \| null`; updated in `checkBallPossession()` lines 1127–1138 | `gmn_pettingzoo.py:373–378` — `ground_truth.current_ball_owner` with `agent_id` + `team`; bridge encodes `ballOwnerAgentIdx` at binary offset 17 (`bridge_server.ts:914`) | Yes — BALL node `ownership` field (`gnn_graph_builder.py:880`), derived from 127-dim observation one-hot indices 94–96 (`ObservationEncoder.ts:136–141`). POSSESSES edge uses closest player of owning team (`gnn_graph_builder.py:1291–1310`). | No — graph uses team-level ownership from observation, not exact `ownerId`. Exact player-id ownership is available in `ground_truth` but not consumed by `build_graph()`. |
| **controlled player** | `GameEngine.ts:22` — `controlledPlayerId: string \| null`; changes on `SWITCH_PLAYER` (line 926) and possession change (line 1137) | `bridge_server.ts:437` — `controlledPlayerId` in reset info; `gmn_pettingzoo.py:1028–1032` — `controlledPlayerId` in `info_data` | Partial — PLAYER node `is_controlled` flag is set from static scenario config `setup.leftPlayers[i].isControlled` (`gnn_graph_builder.py:715–718`) and a hard-coded `11_vs_11` fallback (line 720–721). | Yes — dynamic `controlledPlayerId` changes during episodes (e.g., `SWITCH_PLAYER`) are not propagated into the graph. |
| **formation slot** | `Rules.ts:68–139` — `FORMATIONS` template array; `getFormationPositions()` lines 141–168 computes pitch positions | Not directly exposed in step/reset info dicts. Bridge `getInfo()` (`bridge_server.ts:647`) lists scenario names only. | Yes — `_build_formation_slot_nodes()` (`gnn_graph_builder.py:983–1011`) creates `FORMATION_SLOT` nodes for `11_vs_11` only, using an embedded copy of `FORMATIONS` (`gnn_graph_builder.py:324–393`). | No — limited to `11_vs_11`; academy scenarios with empty `setup.leftPlayers` that fall back to formation layouts do not get FORMATION_SLOT nodes. Embedded `FORMATIONS` dict can drift from `Rules.ts`. |
| **goal geometry** | `Rules.ts:6–40` — `PITCH.goalWidth` (0.14), `goalMinY` (-0.07), `goalMaxY` (0.07), `goalHeight` (0.05), `goalDepth` (0.04); `isGoalMouthPoint()` lines 58–66 | Not exposed in info dict. | Yes — GOAL nodes at `x=-1.0` and `x=1.0`, `y=0.0`, with hardcoded `width=0.14`, `height=0.05`, `depth=0.04` (`gnn_graph_builder.py:885–906`). Values match `Rules.ts`. | No — hardcoded rather than imported from authoritative `PITCH` / `Rules.ts`. |
| **passing geometry** | `GameEngine.ts:742–747` — `currentPassTracking` tracks `passerId`, `team`, `targetId`, `offsideReceiverIds`; `ball.lastKickedBy` / `lastKickedTeam` (lines 736–737) | `gmn_pettingzoo.py:365–378` — `ground_truth` exposes `completed_passes_left` and `attempted_passes_left` only; no passer→receiver pairs or pass vectors. | No — there are no PASS edges or pass-geometry features. | Yes — passer/receiver identities and pass vectors are available in engine state but are not surfaced to the graph. |
| **formation realization** | `Rules.ts:141–168` — `getFormationPositions()` maps `xRatio`/`yRatio` to absolute pitch `(x,y)` using `PITCH.width * 0.6` and `PITCH.height`; used in `GameEngine.ts:166–167` and `GameEngine.ts:326–363` | Not directly exposed in info dict. | Partial — `_get_formation_slot_positions()` (`gnn_graph_builder.py:639–660`) mirrors the engine formula (`-1.0 + xRatio*1.2`, `-0.42 + yRatio*0.84` for left). Values match current `Rules.ts` implementation but are computed from an embedded `FORMATIONS` dict (`gnn_graph_builder.py:324–393`). | Yes — embedded `FORMATIONS` and coordinate math can drift from `Rules.ts` if either is updated independently. |
| **scenario context** | `GameEngine.ts:21` — `activeScenario: ScenarioConfig \| null`; full `ScenarioConfig` available (`src/types/football.ts:233–259`) | `gmn_pettingzoo.py` — scenario id passed at construction (`__init__` line 374) and via `set_scenario()` line 467; not included in per-step `info` dict. | Yes — SCENARIO node built from embedded `SCENARIOS` dict (`gnn_graph_builder.py:909–959`) and placed at `graph["scenario"]` (not in `nodes` array). | No — graph builder uses a Python-side embedded copy of `ScenarioRegistry.ts`. Any scenario config changes must be manually synced. |
| **exact possession semantics** | `GameEngine.ts:1127–1138` — possession assigned on ball-catch contact; `ball.ownerId` set to exact player id; `hasBall` flag toggled | `gmn_pettingzoo.py:373–378` — `ground_truth.current_ball_owner` resolves exact `agent_id` + `team` | Partial — BALL node `ownership` is team-level (`left`/`right`/`none`) from observation one-hot. POSSESSES edge connects closest player of owning team (`gnn_graph_builder.py:1291–1310`), not the exact `ownerId`. | Yes — exact player-id possession semantics (including `lastOwnerId`, `lastOwnerTeam`, `isShotInFlight` state) are not represented in the graph. |
| **player-player relative geometry** | Engine has full `players` array with `position` and `velocity`; pairwise distances computable | Not exposed as structured features in info dict. | Partial — PLAYER nodes carry absolute `position` and `velocity`. Derived features include `nearest_teammate_dist` and `nearest_opponent_dist` (`gnn_graph_builder.py:840–848`). TEAMMATE and OPPONENT edges exist but carry no geometric attributes. NEAR edges carry `distance` and `threshold` (`gnn_graph_builder.py:1064–1070`). | Yes — no explicit all-pairs relative geometry matrix or richer edge attributes (e.g., relative velocity, heading delta). |
| **team shape connectivity** | `GameEngine.ts` does not compute team shape metrics; `getFormationPositions()` provides template positions | Not exposed in info dict. | Yes — `TEAM_SHAPE` nodes (`gnn_graph_builder.py:962–980`) with `formation_deviation` metrics computed via gap-detection on live positions: `inter_line_spacing_variance`, `mean_line_compactness`, `num_lines`, `num_lanes` (`_compute_team_shape()` lines 436–496). | No — metrics are derived from actual positions only; template-formation deviation is not computed. |
| **ball-player geometry** | `GameEngine.ts:988–1005` — `checkBallPossession()` computes per-player `dist` to ball; `PhysicsEngine` tracks ball `position`/`velocity` | Not exposed as per-player ball distances in info dict. | Partial — BALL node has `position` and `velocity` (`gnn_graph_builder.py:864–882`). PLAYER nodes have absolute `position`. No explicit ball-player distance edges or features beyond POSSESSES and NEAR. | Yes — no ball-player distance features on PLAYER nodes, no BALL→PLAYER proximity edges for all players. |
| **player-goal geometry** | `GameEngine.ts:779` — `opponentGoalX` computed per-player; `Rules.ts:58–66` — `isGoalMouthPoint()` defines goal mouth geometry | Not exposed in info dict. | Partial — GOAL nodes carry position and geometry (`gnn_graph_builder.py:885–906`). No PLAYER→GOAL distance edges or shooting-angle features. | Yes — no explicit player-goal distance or angle features. |
| **edge directionality** | Not a native engine concept; edges are graph-level abstractions | N/A | Undirected in practice — TEAMMATE/NEAR/FORMATION edges use `i < j` ordering (`gnn_graph_builder.py:1038`, `1057`, `1135`, `1164`, `1182`). OPPONENT edges are ordered left→right (lines 1046–1052). POSSESSES is directed player→ball (line 1308). Schema does not declare a `directed` flag. | Yes — directionality semantics are implicit and inconsistent across edge types. |
| **zScenario availability** | `GameEngine.ts:269` — `zScenario: TaskEncoder.encode(...)` included in `RLObservation`; 8-dim `Float32Array` (`TaskEncoder.ts:6–47`) | **Not exposed.** Bridge sends only `observation.rawVector` (127-dim) in binary frames (`bridge_server.ts:922–925`). Step/reset JSON responses do not include `zScenario`. | **No.** `build_graph()` signature accepts only `observation` (127-dim) and `info` (`gnn_graph_builder.py:7`). SCENARIO node embeds static scenario config, not dynamic `zScenario`. | Yes — bridge must serialize `zScenario`; Python env must receive it; `build_graph()` must accept and consume it. |

---

## 3. Current Graph Support

### Node Types
The graph currently supports the following node types, defined in `training/gnn_graph_schema.json` lines 19–35 and constructed in `training/gnn_graph_builder.py`:

| Node Type | Count / Occurrence | Construction Source | Status |
|-----------|-------------------|---------------------|--------|
| `PLAYER` | Up to 22 (left_0..left_10, right_0..right_10) | `_build_player_nodes()` lines 668–861; positions/velocities from 127-dim obs; roles from embedded `SCENARIOS` dict | Implemented; tested (`test_gnn_graph_builder.py` lines 135–156, 455–473) |
| `BALL` | 1 | `_build_ball_node()` lines 864–882; position/velocity from obs offsets 88–93; ownership from obs one-hot 94–96 | Implemented; tested (lines 207–229) |
| `GOAL` | 2 (goal_left, goal_right) | `_build_goal_nodes()` lines 885–906; hardcoded geometry | Implemented |
| `SCENARIO` | 1 (top-level `graph["scenario"]`, not in `nodes` array) | `_build_scenario_node()` lines 909–959; embedded `SCENARIOS` + `info["score"]` | Implemented; tested (lines 326–361) |
| `TEAM_SHAPE` | 2 (left, right) | `_build_team_shape_nodes()` lines 962–980; gap-detection on live positions | Implemented; tested (lines 299–323) |
| `FORMATION_SLOT` | 22 for `11_vs_11`, 0 otherwise | `_build_formation_slot_nodes()` lines 983–1011; embedded `FORMATIONS` dict | Implemented for `11_vs_11` only; tested (lines 363–386) |

**Missing node type:** `zScenario` is not represented as a node or feature in the graph.

### Edge Types
The graph currently supports the following edge types, defined in `training/gnn_graph_schema.json` lines 41–63 and constructed in `training/gnn_graph_builder.py`:

| Edge Type | Construction | Attributes | Status |
|-----------|-------------|------------|--------|
| `TEAMMATE` | `_build_edges()` lines 1257–1264; all unordered same-team pairs | `source`, `target` | Implemented; tested (lines 135–156) |
| `OPPONENT` | `_build_edges()` lines 1266–1272; all ordered cross-team pairs (left→right) | `source`, `target` | Implemented; tested (lines 135–156) |
| `NEAR` | `_build_edges()` lines 1274–1288; cross-team pairs with Euclidean dist < 0.065 | `source`, `target`, `distance`, `threshold` | Implemented; tested (lines 158–205) |
| `POSSESSES` | `build_graph()` lines 1291–1310; closest player of owning team to ball | `source`, `target` (fixed `"ball"`) | Implemented; tested (lines 207–229) |
| `FORMATION_ADJACENCY` | `_build_formation_edges()` lines 1173–1187; k=2 nearest FORMATION_SLOT neighbors in `(xRatio, yRatio)` space | `source`, `target` | Implemented for `11_vs_11` only |
| `FORMATION_LINE` | `_build_formation_edges()` lines 1113–1141; FORMATION_SLOT pairs sharing xRatio band (GK/Def/Mid/Att) | `source`, `target`, `line_id` | Implemented for `11_vs_11` only; tested implicitly via schema validation |
| `FORMATION_LANE` | `_build_formation_edges()` lines 1143–1170; FORMATION_SLOT pairs sharing yRatio band (Left/Center/Right) | `source`, `target`, `lane_id` | Implemented for `11_vs_11` only; tested implicitly via schema validation |

**Missing edge types:** No PASS, SHOT, TACKLE, INTERCEPTION, or GOAL-SCORING edges. No ball-player proximity edges for non-owning players.

### Determinism Guarantees
- Node ordering: PLAYER (left_0..left_N, right_0..right_M), BALL, GOAL (left, right), TEAM_SHAPE (left, right), FORMATION_SLOT (left, right) — `gnn_graph_builder.py:1238–1247`.
- Edge ordering: sorted lexicographically by `(edge_type, source, target)` — `gnn_graph_builder.py:1317`.
- Tested via `test_determinism` (`test_gnn_graph_builder.py` lines 388–400) and `test_node_ordering_deterministic` / `test_edge_ordering_deterministic` (lines 402–453).

---

## 4. Gaps Requiring Phase 3 Action

### G1. zScenario Not Available to Graph Builder
- **Impact:** The 8-dim `zScenario` task vector (`TaskEncoder.ts:6–47`) is produced by the engine and included in `RLObservation` (`GameEngine.ts:269`), but the bridge never serializes it. The Python `build_graph()` function cannot include it.
- **Action required:** 
  1. Extend `bridge_server.ts` binary frame and/or JSON responses to include `zScenario`.
  2. Extend `gmn_pettingzoo.py` frame parsing to extract `zScenario`.
  3. Add `zScenario` parameter to `build_graph()` signature and include it as a node or node feature in the graph.

### G2. Exact Player-Id Ball Ownership vs. Team-Level Ownership
- **Impact:** The graph's POSSESSES edge and BALL node `ownership` field are derived from the 127-dim observation one-hot (team-level: `left`/`right`/`none`). The engine's exact `ball.ownerId` (e.g., `"left_2"`) and `ground_truth.current_ball_owner` are available but unused by `build_graph()`.
- **Action required:** Either pass `ground_truth` / `ballOwnerAgentIdx` into `build_graph()`, or re-derive exact ownership from observation + scenario config + active player index.

### G3. Dynamic Controlled Player ID
- **Impact:** `is_controlled` on PLAYER nodes is computed from static scenario config (`gnn_graph_builder.py:715–721`). The engine's `controlledPlayerId` can change at runtime via `SWITCH_PLAYER` (`GameEngine.ts:926`) or possession change (`GameEngine.ts:1137`).
- **Action required:** Accept `controlledPlayerId` as a parameter to `build_graph()` and use it to set `is_controlled` dynamically.

### G4. Passing Geometry Absent
- **Impact:** Engine tracks `currentPassTracking` (`GameEngine.ts:742–747`) with `passerId`, `targetId`, `offsideReceiverIds`, and `lastKickedBy`/`lastKickedTeam`. None of this is exposed in the graph.
- **Action required:** Decide whether to add PASS edges (passer→receiver) or pass-geometry features to PLAYER nodes, and extend bridge/info to carry pass events or reconstruct them from `ground_truth`.

### G5. Embedded Data Drift Risk
- **Impact:** `gnn_graph_builder.py` contains embedded copies of `SCENARIOS` (lines 44–318), `FORMATIONS` (lines 324–393), and goal geometry constants (lines 885–906). These mirror authoritative TypeScript sources but are not auto-synced.
- **Action required:** Prefer importing from a generated contract or syncing via `npm run sync-contracts` / equivalent. At minimum, add a validation test that asserts Python embedded data matches TypeScript sources.

### G6. Formation Slot Scope Limitation
- **Impact:** FORMATION_SLOT nodes and formation edges (`FORMATION_ADJACENCY`, `FORMATION_LINE`, `FORMATION_LANE`) are only generated for `11_vs_11` (`gnn_graph_builder.py:985`, `1083`). Academy scenarios that fall back to formation layouts (e.g., `11_vs_11`-sized but not the `11_vs_11` scenario id) receive no formation nodes.
- **Action required:** Clarify whether formation slots should be generated for any scenario that uses `getFormationPositions()` fallback, or remain `11_vs_11`-only.

### G7. Schema `$id` / Version Mismatch
- **Impact:** `training/gnn_graph_schema.json` line 3 declares `$id: "gnn_graph_schema_v1.json"` while line 4 declares `"title": "GNN Graph Schema v3"`.
- **Action required:** Align `$id` with the actual version (e.g., `gnn_graph_schema_v3.json`) to avoid confusion in tooling and downstream consumers.

---

## 5. Key Decisions / Constraints

1. **Node ordering is fixed by team + index.** PLAYER nodes follow the 127-dim observation ordering: left_0..left_10, then right_0..right_10. This is enforced by the schema `global_id` pattern (`^ (left|right)_[0-9]+$`) and by `build_graph()` assembly order (`gnn_graph_builder.py:1238–1247`).

2. **Edges are sorted lexicographically within type.** `edges.sort(key=lambda e: (e["edge_type"], e["source"], e.get("target", "")))` is applied twice (`gnn_graph_builder.py:1090`, `1317`) to guarantee deterministic output.

3. **Line/lane IDs use gap-detection on actual positions, not formation templates.** For non-`11_vs_11` scenarios, `line_id` and `lane_id` are derived from 1D gap-detection on live x/y coordinates with threshold 0.15 (`gnn_graph_builder.py:782–787`), not from `FORMATIONS`. The schema description explicitly notes this (`gnn_graph_schema.json` lines 229–234).

4. **SCENARIO node lives at `graph["scenario"]`, not in `graph["nodes"]`.** This is a deliberate structural choice documented in `gnn_graph_builder.py:1240–1245` and reflected in the schema (`gnn_graph_schema.json` lines 7–15).

5. **Graph builder is Python-side only.** The bridge (`training/bridge_server.ts`) sends binary frames with 127-dim observations; graph construction does not happen in TypeScript.

6. **Non-ASCII characters in source files are intentional.** `training/test_event_code_wire.py` documents a prior mojibake fix; characters such as ω, ×, ≈, µ, ✓, § in `src/engine/Physics.ts`, `src/components/AgentArenaPanel.tsx`, and `training/validate_learned_policy.ts` are UTF-8 artifacts.

7. **Binary frame protocol:** Batched binary action frames use `0xFF` magic prefix `[0xFF, B, N, B×N actions]`; plain multi-agent buffers have no header. Bridge parser routes based on magic byte presence (`bridge_server.ts:1209`).

8. **zScenario is currently dead code for the graph pipeline.** It is computed and stored in `RLObservation` but never transmitted across the bridge or consumed by Python.

---

## 6. Risk Notes

1. **Drift between embedded Python data and TypeScript sources.** The Python `SCENARIOS` dict (`gnn_graph_builder.py:44–318`) and `FORMATIONS` dict (`gnn_graph_builder.py:324–393`) are manual mirrors of `ScenarioRegistry.ts` and `Rules.ts`. Any scenario or formation change in TypeScript requires a corresponding Python edit. Risk: silent graph construction errors or schema validation failures.

2. **Controlled player mismatch during episodes.** If the engine switches `controlledPlayerId` (e.g., `SWITCH_PLAYER` action at `GameEngine.ts:926`), the graph's `is_controlled` flag will remain stale because it is computed from static scenario config. Risk: RL policies trained on graph input receive incorrect active-player signals.

3. **Exact possession semantics are approximated.** The graph infers possession from the observation one-hot and picks the closest player of the owning team (`gnn_graph_builder.py:1291–1310`). This is not guaranteed to match `ball.ownerId` in edge cases (e.g., two players equidistant, or ownership transition frames). Risk: POSSESSES edge points to wrong player.

4. **zScenario absence limits task-conditioned GNN training.** Without `zScenario`, the graph cannot represent task progression (time remaining, pass progress, goal progress, possession state) beyond the static SCENARIO node. Risk: Phase 3 task-conditioned architectures require a separate workaround or retraining.

5. **Schema `$id` inconsistency may break automated tooling.** Some JSON Schema validators or documentation generators use `$id` as the canonical identifier. The mismatch between `$id` (`gnn_graph_schema_v1.json`) and `title` (`GNN Graph Schema v3`) could cause versioning confusion.

6. **Formation slot nodes are silently absent for most scenarios.** Tests verify that academy scenarios contain zero FORMATION_SLOT nodes (`test_gnn_graph_builder.py:363–375`), but this is a current limitation, not a guaranteed invariant. If Phase 3 expands formation edges to smaller scenarios, the gap-detection-based `line_id`/`lane_id` on PLAYER nodes and the template-based FORMATION_SLOT nodes may produce conflicting line/lane semantics.

7. **Bridge binary protocol has no reserved space for zScenario.** The current binary frame layout (`bridge_server.ts:878–892`) is tightly packed: 18-byte header + 508-byte observation + 19-byte mask = 545 bytes for non-rondo. Adding `zScenario` (32 bytes) requires a protocol version bump or a new frame variant.
