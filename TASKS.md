# GMN-Football-3 — Remediation Tracker

Tracking task vs progress. Status: `[ ]` = pending, `[~]` = in progress, `[x]` = done.

## Item 1 — Make reward shaping (CooperativeRewardShaper) actually functional
- [x] Bridge/wrapper feed shaper with the event types it can consume (`PASS_COMPLETED`, `SHOT_TAKEN`, `GOAL_SCORED`, `TURNOVER_CONCEDED`, `PASS_FAILED`)
- [x] Provide `agent_id` for ball-owner attribution (ball-hogging / per-agent credit)
- [x] Add a wired-path regression test asserting passing vs direct-shooting produce different shaped rewards
- [x] Fix rondo `Math.abs(currBallX) && Math.abs(currBallX)` copy-paste bug
- [x] Run the added/adjusted tests (pytest) to confirm

## Item 1b — Secure verified Item-1 fixes (prevents loss on checkout)
- [x] Commit the 7 verified modified source files
- [x] Add a TRUE end-to-end wire test (boot bridge → `GMNMultiAgentEnv.step()`) asserting shaped rewards differ for passing vs solitary-shooting — `training/test_reward_shape_e2e.py`: shaped +1.2126 vs unshaped +1.2627, `holder_ticks=36`, `ball_hogging_count=6` → PASS
- [x] Revisit `PASS_COMPLETED`-at-initiation semantics (over-counts; emit real `PASS_FAILED`) — pending-pass state machine in `GMNMultiAgentEnv._resolve_pending_pass()`: PASS_COMPLETED deferred until a teammate gains possession; PASS_FAILED on right-team possession / turnover event / 60-step loose-ball timeout; 5 new unit tests (17 passed) + e2e wire re-verified

## Item 2 — One canonical MAPPO trainer
- [x] Decide shaped vs plain as canonical path — `train_mappo.py` = canonical plain (baseline); `train_mappo_shaped.py` = canonical shaped variant (documented in its header). Not directly comparable by design.
- [x] Ensure the canonical trainer is committed; document divergence — both committed; header note in `train_mappo_shaped.py`; shaped checkpoints tracked (`..._seed42_shaped_*.pt`)

## Item 3 — Repo hygiene: commit / ignore untracked work
- [x] Commit `train_mappo_shaped.py`, shaped + rondo checkpoints (matches existing seed-checkpoint convention, ~430 KB each) + evidence CSVs — commit `6e1d632`
- [x] Gitignore `training/runs/`, `*.bak` (`win_rate_progress.csv.bak`), `*_temp.pt` (`*_shaped_temp.pt`) — `.gitignore` "Training artifacts (transient)" section
- [x] Verify `git status` is clean/coherent — confirmed clean after `6e1d632`

## Item 4 — Frontend hyperparameters match trainer
- [x] Fix `TrainingTelemetryService` defaults to match trainers — clipRange 0.2→0.15, miniBatchSize 64→256, targetTimesteps 200k (train_mappo.py default; shaped targets 500k, noted in comment), entropy anneal 0.01→0.005 documented. `npx tsc --noEmit` passes.

## Item 5 — Reconciled eval/README checkpoint paths
- [x] Point all defaults at an existing checkpoint — `mappo_..._trained.pt` did not exist; replaced with `mappo_..._best.pt` (eval_mappo.py, eval_generalization.py, export_onnx.py, validate_learned_policy.py, bridge_server.ts, TrainingJobService.ts fallback) and `..._seed44_best.pt` for the deployed-weights lineage (validate_learned_policy.ts, README). Remaining `_trained.pt` refs are guarded `fs.existsSync` fallbacks only. tsc + py_compile pass.
- [ ] Reconcile stale/scattered `comprehensive_eval_*.json` / reports

## Item 6 — Single evidence source of truth
- [ ] Pin reported results to checkpoint SHA + `git describe`
- [ ] Resolve report content thrash / FILE_NOT_FOUND class conflicts