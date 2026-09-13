# GNN_PHASE4_GRAPH_DIAGNOSTICS

**Date:** 2026-09-13  
**Scope:** First runnable code phase. `training/gnn_graph_builder.py` produces schema-valid graphs from live engine observations. `training/tests/test_gnn_graph_builder.py` is the correctness suite.

---

## 1. What `build_graph` does not populate and why

| Field / Node | Status | Reason |
|---|---|---|
| `ballOwnedPlayer` (specific index) | **Not populated** | The 127-dim observation vector only carries a 3-float ball-ownership one-hot (`none/left/right`), not the owning player's index. `ObservationEncoder.ts` computes `ballOwnedPlayer` internally but does not serialize it into the raw vector sent to the bridge. `POSSESSES` edges therefore use a closest-player heuristic rather than exact ownership. |
| `is_controlled` (exact engine state) | **Approximated** | `controlledPlayerId` is an internal `GameEngine` field not exposed in the observation. For academy scenarios, `isControlled` from `ScenarioConfig.setup` is used. For `11_vs_11`, `left_0` defaults to controlled (matching `GameEngine.ts`'s `players[0]` fallback). |
| `objective.is_completed` / `is_failed` | **Always `false`** | The bridge's `info` dict does not include live objective completion state (only `ground_truth` on terminal ticks, and even then without per-objective completion flags). `GameEngine.evaluateScenarioConditions()` state is not wire-transmitted. |
| `SEQUENCE_NEXT` edges | **Deferred** | Requires event timestamps and possession-chain history. Phase 0 §3.2 identified this as needing new engine instrumentation. |
| `CONSTRAINED_BY` edges | **Deferred** | Requires scenario attributes (`target_zone`, `forbidden_actions`, `max_touches`) that do not exist in `ScenarioConfig` (`types/football.ts:197-222`). |
| `FORMATION_ADJACENCY` / `FORMATION_LINE` / `FORMATION_LANE` for academy scenarios | **Deferred** | Academy scenarios lack a formation identity (`setup.leftPlayers` is non-empty, so `getFormationPositions` is never called). Track A (template-driven) is restricted to `11_vs_11` per Phase 2 Track A/B split. |
| Template-based formation deviation for academy scenarios | **Deferred** | Track B provides template-free `formation_deviation` (gap-detection lines/lanes). Measuring distance from a `FORMATIONS` template would require inventing a formation label for academy scenarios, which Track B explicitly avoids. |

---

## 2. Determinism guarantee

Node ordering is fixed by team and index:
- `left_0` … `left_10` (or up to `teamLeftPlayers` for academy)
- `right_0` … `right_10` (or up to `teamRightPlayers`)
- `ball`
- `goal_left`, `goal_right`
- `scenario`
- `team_shape_left`, `team_shape_right`
- `formation_slot_left_0` … `formation_slot_right_10` (11_vs_11 only)

Edge lists are sorted lexicographically by `(edge_type, source, target)` within each type group. The same `observation` + `info` + `scenario_id` tuple always produces byte-identical JSON output.

Gap-detection for `line_id` / `lane_id` is deterministic: sort by coordinate, split at all gaps `> 0.15`, assign sequential IDs. No random initialization.

---

## 3. Test coverage summary

| # | Requirement | Test method |
|---|---|---|
| 1 | NEAR edge distance correctness | Live reset of `academy_3_vs_1_defender_3`; independently compute all cross-team distances and compare NEAR edge set |
| 2 | TEAMMATE / OPPONENT team-sorting | Assert every TEAMMATE edge connects same-team pairs, every OPPONENT edge connects cross-team pairs |
| 3 | POSSESSES correctness | Assert POSSESSES presence/absence matches `ball.ownership` from the observation |
| 4 | Line/lane correctness | Independently recompute gap-detection from live positions; compare with `build_graph` output on both tick-0 and post-step |
| 5 | TEAM_SHAPE correctness | Independent from-scratch implementation of `inter_line_spacing_variance` and `mean_line_compactness` formulas; compare |
| 6 | SCENARIO node correctness | Verify `objective_vocabulary`, `reward_scoring`, `reward_completion`, `terminate_on_opponent_possession` against `ScenarioRegistry` for 3 scenarios |
| 7 | Formation (Track A) correctness | Assert zero `FORMATION_SLOT` nodes for academy/rondo; assert 22 `FORMATION_SLOT` nodes for `11_vs_11` |
| 8 | Schema validation on every case | `jsonschema.validate()` called in every test and in a smoke loop over all 12 scenarios |
| 9 | Determinism check | Call `build_graph` twice with identical inputs; assert `json.dumps(sort_keys=True)` equality |

---

## 4. Out of scope (explicit)

- §28 representation-diagnostic probes (recovering formation/role/distance from a trained embedding) — requires a GNN encoder, which is Phase 5+.
- Wiring `build_graph` into `gmn_pettingzoo.py`, `train_mappo.py`, or any MAPPO code path — explicitly Phase 5.
- `FORMATION_ADJACENCY` / `FORMATION_LINE` / `FORMATION_LANE` for academy scenarios — Phase 2 Track A/B split; academy scenarios use Track B (gap-detection) only.

---

## 5. How to run

```bash
# Full suite (requires live bridge, same pattern as prior phases)
python -m pytest training/tests/test_gnn_graph_builder.py -v

# Single test
python -m pytest training/tests/test_gnn_graph_builder.py::TestGNNGraphBuilder::test_near_edges_correctness -v
```

The tests boot a TypeScript bridge subprocess via `GMNMultiAgentEnv(auto_start_bridge=True)`, same as `test_reward_exploits.py` and other live-bridge tests.

---

*End of Phase 4 diagnostics document. No engine, scenario, or training-pipeline source files were modified.*
