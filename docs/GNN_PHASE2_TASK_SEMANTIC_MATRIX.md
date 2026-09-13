# GNN Phase 2 — Task Semantic Matrix

One row per registered scenario. Every cell must be backed by the actual engine definition.
`UNKNOWN / NOT_DEFINED` is used where the engine does not currently expose the concept.

---

| Scenario | Task | Formation | Touch Rule | Dribble | Shoot | Pass Target | Goal Target | Turnover | Bounds | Sequence | Step Limit |
|---|---|---|---|---|---|---|---|---|---|---|
| `academy_empty_goal` | Score in empty net | 4-3-3 (1 player) | NOT_DEFINED | NOT_DEFINED | Permitted (required) | NOT_DEFINED | 1 | terminateOnOpponentPossession = true | NOT_DEFINED | NOT_DEFINED | 15s / 900 ticks |
| `academy_run_to_score` | Evade defender, score | 4-3-3 (1 player) | NOT_DEFINED | NOT_DEFINED | Permitted (required) | NOT_DEFINED | 1 | terminateOnOpponentPossession = true | NOT_DEFINED | NOT_DEFINED | 20s / 1200 ticks |
| `academy_pass_and_shoot_with_keeper` | Pass then score | 4-3-3 (2 players) | NOT_DEFINED | NOT_DEFINED | Permitted (required) | 1 (objective text) | 1 | terminateOnOpponentPossession = true | NOT_DEFINED | NOT_DEFINED | 25s / 1500 ticks |
| `academy_3_vs_1_with_keeper` | Triangle pass, score | 4-3-3 (3 players) | NOT_DEFINED | NOT_DEFINED | Permitted | 2+ (objective text) | 1 | terminateOnOpponentPossession = true | NOT_DEFINED | NOT_DEFINED | 30s / 1800 ticks |
| `academy_3_vs_1_defender_2` | Score past 2 defenders | 4-3-3 (3 players) | NOT_DEFINED | NOT_DEFINED | Permitted | NOT_DEFINED | 1 | terminateOnOpponentPossession = true | NOT_DEFINED | NOT_DEFINED | 30s / 1800 ticks |
| `academy_3_vs_1_defender_3` | Score past 3 defenders + GK | 4-3-3 (3 players) | NOT_DEFINED | NOT_DEFINED | Permitted | NOT_DEFINED | 1 | terminateOnOpponentPossession = true | NOT_DEFINED | NOT_DEFINED | 30s / 1800 ticks |
| `academy_3_vs_1_keeper_aggressive` | Score past aggressive GK | 4-3-3 (3 players) | NOT_DEFINED | NOT_DEFINED | Permitted | NOT_DEFINED | 1 | terminateOnOpponentPossession = true | NOT_DEFINED | NOT_DEFINED | 30s / 1800 ticks |
| `academy_3_vs_1_shifted` | Score past shifted defense | 4-3-3 (3 players) | NOT_DEFINED | NOT_DEFINED | Permitted | NOT_DEFINED | 1 | terminateOnOpponentPossession = true | NOT_DEFINED | NOT_DEFINED | 30s / 1800 ticks |
| `academy_3_vs_1_randomized` | Score under jitter | 4-3-3 (3 players) | NOT_DEFINED | NOT_DEFINED | Permitted | NOT_DEFINED | 1 | terminateOnOpponentPossession = true | NOT_DEFINED | NOT_DEFINED | 30s / 1800 ticks |
| `academy_rondo_4v1` | Keep-ball, complete passes | 4-3-3 (4 players) | NOT_DEFINED | NOT_DEFINED | Not permitted (no goal logic) | 10+ (objective text) | 0 (no goal) | Custom handler: defenderPossessionTime ≥ 2.0s or ballOutOfAreaTime ≥ 1.5s | Implicit rectangle |0.35 radius via handler | NOT_DEFINED | 20s / 1200 ticks |
| `5_vs_5` | Win match, >50% possession | 4-3-3 (5 players) | NOT_DEFINED | NOT_DEFINED | Permitted | NOT_DEFINED | NOT_DEFINED | terminateOnOpponentPossession = false | NOT_DEFINED | NOT_DEFINED | 90s / 5400 ticks |
| `11_vs_11` | Win match, clean sheet | 4-3-3 (11 players) | NOT_DEFINED | NOT_DEFINED | Permitted | NOT_DEFINED | NOT_DEFINED | terminateOnOpponentPossession = false | NOT_DEFINED | NOT_DEFINED | 180s / 10800 ticks |

---

## Notes

- **Touch Rule**: The engine does not track discrete touch events. `NOT_DEFINED` is explicit.
- **Dribble / Shoot**: The engine exposes `ActionType.DRIBBLE` and `ActionType.SHOT`, but no scenario-level constraint restricts them. The rondo description says "Do not shoot," but this is not enforced by the engine.
- **Sequence**: No sequence state machine exists in the engine.
- **Bounds**: Only the rondo handler uses an implicit spatial bound (`|x| ≤ 0.35 && |y| ≤ 0.35`). This is handler logic, not a scenario-level field.
- **Turnover**: `terminateOnOpponentPossession` is the only explicit termination rule in `ScenarioConfig`. Rondo uses `checkExtraTermination()` for custom conditions.
