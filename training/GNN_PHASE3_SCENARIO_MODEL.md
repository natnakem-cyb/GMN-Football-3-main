# GNN_PHASE3_SCENARIO_MODEL

**Date:** 2026-09-12  
**Scope:** Design/documentation only. No engine, scenario, observation, or training code was modified.  
**Ground truth:** `training/GNN_PHASE0_ARCHITECTURE_AUDIT.md` (commit `a0f4c0a`), `training/GNN_PHASE1_GRAPH_SCHEMA.md`, and `training/GNN_PHASE2_FORMATION_MODEL.md`. Where this document disagrees with earlier phases, the earlier phase wins.

---

## Changelog

| Version | Phase | Changes |
|---------|-------|--------|
| v1 | Phase 1 | Initial schema: PLAYER, BALL, GOAL, SCENARIO nodes; TEAMMATE, OPPONENT, NEAR, POSSESSES edges. SCENARIO node has time_limit_seconds, terminate_on_opponent_possession, objectives. |
| v2 | Phase 2 | Added Track A formation structure (11_vs_11 only) and Track B continuous shape descriptors (academy scenarios). |
| v3 | Phase 3 | Adds objective_vocabulary multi-hot to SCENARIO node. Adds reward_scoring and reward_completion top-level fields to SCENARIO node. Makes rewards object required. Documents sequence/stage investigation. Extends worked example. |

---

## 1. Objective Vocabulary as Multi-Hot Feature

### 1.1 Vocabulary derivation

Re-derived from current `ScenarioRegistry.ts` (all 12 scenarios). The following objective ids were found:

| # | Objective id | Scenarios using it |
|---|-------------|-------------------|
| 1 | `score_goal` | academy_empty_goal, academy_run_to_score, academy_pass_and_shoot_with_keeper, academy_3_vs_1_with_keeper, academy_3_vs_1_defender_2, academy_3_vs_1_defender_3, academy_3_vs_1_keeper_aggressive, academy_3_vs_1_shifted, academy_3_vs_1_randomized (9 scenarios) |
| 2 | `within_time` | academy_empty_goal (1 scenario) |
| 3 | `avoid_dispossess` | academy_run_to_score (1 scenario) |
| 4 | `complete_pass` | academy_pass_and_shoot_with_keeper (1 scenario) |
| 5 | `create_triangle` | academy_3_vs_1_with_keeper (1 scenario) |
| 6 | `retain_possession` | academy_rondo_4v1 (1 scenario) |
| 7 | `complete_passes` | academy_rondo_4v1 (1 scenario) |
| 8 | `win_match` | 5_vs_5, 11_vs_11 (2 scenarios) |
| 9 | `control_possession` | 5_vs_5 (1 scenario) |
| 10 | `clean_sheet` | 11_vs_11 (1 scenario) |

**Total: 10 distinct objective ids.** The brief"s list was confirmed correct.

### 1.2 Example citations from ScenarioRegistry.ts

- `academy_empty_goal` objectives (lines 25-28): `score_goal`, `within_time`
- `academy_run_to_score` objectives (lines 59-62): `score_goal`, `avoid_dispossess`
- `academy_pass_and_shoot_with_keeper` objectives (lines 94-97): `complete_pass`, `score_goal`
- `academy_3_vs_1_with_keeper` objectives (lines 128-131): `create_triangle`, `score_goal`
- `academy_rondo_4v1` objectives (lines 344-347): `retain_possession`, `complete_passes`
- `5_vs_5` objectives (lines 384-387): `win_match`, `control_possession`
- `11_vs_11` objectives (lines 412-415): `win_match`, `clean_sheet`

### 1.3 Multi-hot encoding

**Ordering rule: alphabetical by objective id (ascending).** This is a fixed, deterministic ordering that any implementation can reproduce without consulting the schema.

Alphabetical order:

```
 0: avoid_dispossess
 1: clean_sheet
 2: complete_pass
 3: complete_passes
 4: control_possession
 5: create_triangle
 6: retain_possession
 7: score_goal
 8: within_time
 9: win_match
```

The multi-hot array has length 10. For a scenario with objectives `[score_goal, within_time]`, the encoding is:

```
 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
```

### 1.4 Vocabulary extensibility

This vocabulary will need to be extended if a future scenario introduces a new objective id. This is expected and fine — the schema field is an array, not a fixed-length tensor. A new objective id appends a new position to the multi-hot array. Implementations should derive the vocabulary dynamically from ScenarioRegistry at build time rather than hardcoding the 10 entries.

---

## 2. Reward-Shape as Continuous Feature

### 2.1 Reward values across all scenarios

| Scenario | scoring | completion | Source (ScenarioRegistry.ts) |
|----------|---------|------------|----------------------------|
| academy_empty_goal | 1.0 | 100 | lines 30-33 |
| academy_run_to_score | 1.0 | 200 | lines 64-67 |
| academy_pass_and_shoot_with_keeper | 1.0 | 350 | lines 102-105 |
| academy_3_vs_1_with_keeper | 1.0 | 500 | lines 136-139 |
| academy_3_vs_1_defender_2 | 1.0 | 500 | lines 166-169 |
| academy_3_vs_1_defender_3 | 1.0 | 500 | lines 208-211 |
| academy_3_vs_1_keeper_aggressive | 1.0 | 500 | lines 232-235 |
| academy_3_vs_1_shifted | 1.0 | 500 | lines 263-266 |
| academy_3_vs_1_randomized | 1.0 | 500 | lines 294-297 |
| academy_rondo_4v1 | 0 | 0 | lines 349-352 |
| 5_vs_5 | 1.0 | 800 | lines 389-392 |
| 11_vs_11 | 1.0 | 1200 | lines 417-420 |

### 2.2 Signal analysis

The reward signal cleanly distinguishes scenario types:

- **Goal-oriented academy drills:** scoring=1.0, completion=100-500. Clear goal-scoring incentive.
- **Possession-oriented rondo:** scoring=0, completion=0. No goal reward at all — pure possession play.
- **Competitive matches:** scoring=1.0, completion=800-1200. High-stakes full-match play.

This is a real, varying signal — not padding. The rondo scenario"s zero reward is particularly informative: a GNN that conditions on reward shape can immediately distinguish a no-goal possession drill from a goal-scoring drill.

---

## 3. terminateOnOpponentPossession as Boolean Feature

### 3.1 Current schema status

This field already exists on the SCENARIO node schema as `terminate_on_opponent_possession` (boolean, required). Added in Phase 1. No schema change needed for this field itself.

### 3.2 Value pattern

| Scenario | terminateOnOpponentPossession | Source |
|----------|------------------------------|--------|
| All 9 academy scenarios | true | e.g., academy_3_vs_1_defender_3 line 207 |
| academy_rondo_4v1 | false | line 348 |
| 5_vs_5 | false | line 388 |
| 11_vs_11 | false | line 416 |

Pattern confirmed: true for every academy scenario, false for rondo/5v5/11v11. This cleanly distinguishes drills (fail on dispossession) from matches (play to the clock).

---

## 4. What Is Still NOT Representable (and Why That Is Correct

The following remain unrepresentable because they do not exist in the engine. Per Phase 0 section 1.3, these fields are absent from both the `ScenarioConfig` type (`types/football.ts:197-222`) and every scenario definition in `ScenarioRegistry.ts`.

| Field | Status | Reason |
|-------|--------|--------|
| `max_touches` | Not representable | No engine code tracks or enforces per-touch limits. |
| `allowed_actions` | Not representable | No scenario defines or checks allowed action sets. |
| `forbidden_actions` | Not representable | No scenario defines forbidden action sets. |
| `target_player` | Not representable | No scenario references a specific target player. |
| `target_zone` | Not representable | No scenario defines spatial target zones. |
| `step_limit` | Not representable | No scenario defines step limits (only `timeLimitSeconds`). |

These are not deferred pending investigation — they are confirmed absent. Phase 0 section 1.3 already closed this door. Reopening would require engine instrumentation, not schema work.

**Future engine-code recommendation:** If per-touch constraints are desired for a future training curriculum, the `ScenarioConfig` type would need `max_touches` and `forbidden_actions` fields, and `GameEngine.ts` would need to track touches per player and terminate on violation. This is an engine-code phase task, not a schema design task.

---

## 5. Sequence/Stage Representation Investigation

### 5.1 Question

Does the engine track objective completion order or partial progress at runtime? Specifically, is `objectives[].isCompleted` ever set to `true` during a live episode, or does it stay `false` for the whole episode and only get used for post-hoc scoring?

### 2 Findings

`isCompleted` IS set live during the episode for some objectives, but not all. The behavior is objective-id-dependent:

`evaluateScenarioConditions()` (`GameEngine.ts:1192-1235`) is called every tick at `GameEngine.ts:519`. It sets:
- `score_goal.isCompleted = true` when `score.left > 0` (line 1200-1204) — LIVE
- `within_time.isCompleted = true` when score occurs within time limit (line 1205-1208) — LIVE
- `within_time.isFailed = true` when time limit reached without scoring (line 1222-1227) — LIVE
- `complete_pass.isCompleted = true` when `completedPasses.left >= 1` (line 1212-1215) — LIVE
- `create_triangle.isCompleted = true` when `completedPasses.left >= 2` (line 1216-1220) — LIVE

`resolveMatchObjectives()` (`GameEngine.ts:1245-1269`) is called only when `timeLimitReached` (line 1232-1234). It resolves:
- `win_match`, `control_possession`, `clean_sheet` — END-OF-EPISODE only

`retain_possession`, `avoid_dispossess`, `complete_passes` — NEVER evaluated by any engine code path. These remain `false` for the entire episode (INERT).

### 5.3 Conclusion
`isCompleted` is a live signal for `score_goal`, `within_time`, `complete_pass`, and `create_triangle` — these update in real-time as the agent acts. It is an end-of-episode signal for `win_match`, `control_possession`, and `clean_sheet`. It is inert (always false) for `retain_possession`, `avoid_dispossess`, and `complete_passes`.

This means `sequence_progress` / `current_action_stage` as described in the original proposal (sections 11-12) is NOT fully representable. The live objectives do provide a partial completion signal (e.g., `score_goal` flips to true when scored), but there is no ordering or sequencing of objectives — just independent booleans. The inert objectives provide no signal at all.

For the GNN graph, the per-objective `isCompleted`/`isFailed` booleans on the SCENARIO node"s objectives array are the closest available signal. They should not be mistaken for a general-purpose progress tracker.

---

## 6. Worked Example Extension

Extending the same worked example scenario (`academy_3_vs_1_defender_3`) used in Phases 1-2 with the new SCENARIO node fields.

### 6.1 Scenario data for academy_3_vs_1_defender_3

Source: `ScenarioRegistry.ts:204-211`

- Objectives: `[score_goal]` (single objective)
- `terminateOnOpponentPossession`: true (line 207)
- Rewards: `scoring: 1.0, completion: 500` (lines 208-211)

### 6.2 Multi-hot encoding

For `academy_3_vs_1_defender_3` with objectives `[score_goal]`:

```
Alphabetical vocabulary:
 0: avoid_dispossess    -> 0
 1: clean_sheet         -> 0
 2: complete_pass       -> 0
 3: complete_passes     -> 0
 4: control_possession  -> 0
 5: create_triangle     -> 0
 6: retain_possession   -> 0
 7: score_goal          -> 1
 8: within_time         -> 0
 9: win_match           -> 0

objective_vocabulary = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

### 6.3 Extended SCENARIO node

The SCENARIO node for the worked example, with Phase 3 fields added (new fields marked with **):

```json
  node_type: SCENARIO
  node_id: scenario
  id: academy_3_vs_1_defender_3
  time_limit_seconds: 30
  terminate_on_opponent_possession: true
  objectives: [{ id: score_goal, text: Score past three defenders and goalkeeper, is_completed: false, is_failed: false }]
  ** objective_vocabulary: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  ** reward_scoring: 1.0
  ** reward_completion: 500
  ** rewards: { scoring: 1.0, completion: 500 }
```

### 6.4 Full graph summary

With Phase 3 additions, the worked example graph has:
- 12 nodes (7 PLAYER + 1 BALL + 2 GOAL + 1 SCENARIO + 2 TEAM_SHAPE) - unchanged from Phase 2
- 21 edges (9 TEAMMATE + 12 OPPONENT + 0 NEAR + 0 POSSESSES + 0 formation edges) - unchanged
- SCENARIO node now includes `objective_vocabulary` (10-dim multi-hot), `reward_scoring`, `reward_completion`, and `rewards` is required

---

## 7. Schema Regression Check

All three existing worked examples were extracted and validated against the v3 schema using `jsonschema.validate()`.

### 7.1 Validation method

```
import json, jsonschema
schema = json.load(open('training/gnn_graph_schema.json'))
for name in ['phase1', 'phase2', 'phase3']:
    instance = json.load(open(f'training/_example_{name}.json'))
    jsonschema.validate(instance=instance, schema=schema)
    print(f'{name}: VALID')
```

### 7.2 Validation output

```
Phase 1: VALID
Phase 2: VALID
Phase 3: VALID
```

### 7.3 What was checked

- **Phase 1** (`GNN_PHASE1_GRAPH_SCHEMA.md` section 4.4): 12 nodes (7 PLAYER + 1 BALL + 2 GOAL + 1 SCENARIO + 2 TEAM_SHAPE), 21 edges. SCENARIO node updated with `objective_vocabulary`, `reward_scoring`, `reward_completion`.
- **Phase 2** (`GNN_PHASE2_FORMATION_MODEL.md` section 5): Same as Phase 1 (Phase 2 inherits Phase 1 example with TEAM_SHAPE nodes).
- **Phase 3** (`GNN_PHASE3_SCENARIO_MODEL.md` section 6): Same as Phase 2 (Phase 3 inherits Phase 2 example with updated SCENARIO node).

### 7.4 Regression prevention

The Phase 1 worked example was updated in this commit to include the new SCENARIO node fields (`objective_vocabulary`, `reward_scoring`, `reward_completion`). This ensures all three phase docs remain in sync with the schema.

---
