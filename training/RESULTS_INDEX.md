# GMN-Football-3 — Canonical Results & Evidence Index

Single source of truth mapping each *reported* result to its checkpoint, evaluator,
code revision, and artifact. All checkpoint SHA-256 values below were computed from
the files in `training/models/` at revision `c22b053`.

## Lineage table

| Artifact / claim | Checkpoint | Checkpoint SHA-256 (prefix) | Evaluator | Notes |
|---|---|---|---|---|
| `training/validation_report.md` (v2, 500-episode, ground-truth bridge) | `mappo_academy_3_vs_1_with_keeper_seed42_best.pt` | `ddaf4d38558cf39a…` | `evaluator_version: v2_ground_truth_bridge`, 500 episodes | This resolves the earlier `FILE_NOT_FOUND` ambiguity: the report's model IS seed42_best. |
| Browser-deployed weights (`src/agents/mappo_weights.ts`, `public/models/mappo_policy.onnx`) | `mappo_academy_3_vs_1_with_keeper_seed44_best.pt` | `6fb28ff1a56a60d8…` | `training/export_onnx.py` (verified weights export) | Weights payload SHA `30e41e0c…` pinned in the same file. |
| Legacy heuristic-era behavioral reports (`comprehensive_eval_*_seed{43,137}*.json` with `possession≈50%`, `shot_acc==win_rate`) | pre-fix evaluator (superseded) | n/a | `eval_mappo_comprehensive.py` BEFORE the ground-truth fixes | Kept only as historical record; superseded numbers must not be cited. |
| Current canonical plain-baseline evals (`eval_mappo.py`, `eval_generalization.py`) | `mappo_academy_3_vs_1_with_keeper_best.pt` | `939e6ceefb95c880…` | `eval_mappo.py`, 50 episodes default | Default checkpoint path fixed at `c22b053`. |

## Rules for reporting results
1. Every number published in a report MUST name its checkpoint file AND its SHA-256 prefix (≥16 hex chars).
2. Every report MUST record the code revision (`git rev-parse --short HEAD`) of the evaluator used.
3. Superseded artifacts move to a `*_legacy*` name or are annotated; they are never deleted (audit trail).
4. Two results with different checkpoint SHAs are NEVER compared as if they describe the same policy (this killed the earlier "0.0% vs 53.2%" confusion).

## Known-good verification commands
```bash
certutil -hashfile training/models/mappo_academy_3_vs_1_with_keeper_seed42_best.pt SHA256
C:\Python314\python.exe -m pytest training/tests/test_reward_shaper.py -q          # 27 passed (1 pre-existing unrelated failure)
C:\Python314\python.exe training/test_reward_shape_e2e.py                          # wire test PASS
npx tsc --noEmit                                                                   # contract/type check
```

## Semantic fix log (2026-09-11)

| ID | File | Fix description | Verification |
|---|---|---|---|
| M1 | `training/gmn_pettingzoo.py` | `CooperativeRewardShaper.compute_shaped_rewards` now processes `PASS_INTERCEPTED`, `PASS_FAILED`, `TURNOVER_CONCEDED` regardless of the engine's `team` tag. Added `penalty_turnover` (-0.10) parameter. | New tests: `test_pass_intercepted_with_right_team_tag_penalizes_left_agent`, `test_turnover_conceded_with_right_team_tag_penalizes_left_agent`, updated `test_turnover_chain_break`. |
| M2 | `training/gmn_pettingzoo.py` | Removed terminal-step guard in `step()` that stripped shaped rewards on terminal frames. Implemented assisted-goal bonus: when `pass_chain_length > 0` and a goal is scored, all active left-team agents receive `r_assisted_goal` (+0.50). | New tests: `test_goal_bonus_received_on_terminal_step`, `test_assisted_goal_bonus_distributed_to_all_active_agents`, `test_goal_without_pass_chain_does_not_give_bonus`. |
| H1 | `training/bridge_server.ts` | `stepBatch` now passes the per-env `engine` variable into `runOnnxInference` and `_applyRuleBasedAction` fallback instead of always using `this.engine`. | Code review; no TS unit-test framework present. |
| M5 | `training/gmn_pettingzoo.py` | `step_batch` now maintains per-sub-env `pending_pass`, `reward_shaper`, and `last_actions`. Added `_build_shaper_events`, `_apply_shaping_for_env`, and refactored `_resolve_pending_pass_state` as stateless helper. Non-rondo batched rewards normalized to per-agent dicts. | New tests: `TestM5BatchedRewardShaping` (5 tests covering event mapping, pending-pass state machine, shaping application, and env-state initialization). |
