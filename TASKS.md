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
- [ ] Decide shaped vs plain as canonical path
- [ ] Ensure the canonical trainer is committed; document divergence

## Item 3 — Repo hygiene: commit / ignore untracked work
- [ ] Commit `train_mappo_shaped.py`, `training/runs/`, shaped + rondo checkpoints (or .gitignore temp/backup)
- [ ] Remove/gitignore `win_rate_progress.csv.bak`, `*_shaped_temp.pt`
- [ ] Verify `git status` is clean/coherent

## Item 4 — Frontend hyperparameters match trainer
- [ ] Fix `TrainingTelemetryService` defaults (clip 0.15, entropy 0.01 -> 0.005, etc.) to match `train_mappo.py`

## Item 5 — Reconciled eval/README checkpoint paths
- [ ] Point `eval_mappo.py`, `validate_learned_policy.py`, README at an existing checkpoint
- [ ] Reconcile stale/scattered `comprehensive_eval_*.json` / reports

## Item 6 — Single evidence source of truth
- [ ] Pin reported results to checkpoint SHA + `git describe`
- [ ] Resolve report content thrash / FILE_NOT_FOUND class conflicts