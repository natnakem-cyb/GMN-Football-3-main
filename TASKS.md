# GMN-Football-3 — Remediation Tracker

Tracking task vs progress. Status: `[ ]` = pending, `[~]` = in progress, `[x]` = done.

## Item 1 — Make reward shaping (CooperativeRewardShaper) actually functional
- [x] Bridge/wrapper feed shaper with the event types it can consume (`PASS_COMPLETED`, `SHOT_TAKEN`, `GOAL_SCORED`, `TURNOVER_CONCEDED`, `PASS_FAILED`)
- [x] Provide `agent_id` for ball-owner attribution (ball-hogging / per-agent credit)
- [x] Add a wired-path regression test asserting passing vs direct-shooting produce different shaped rewards
- [x] Fix rondo `Math.abs(currBallX) && Math.abs(currBallX)` copy-paste bug
- [x] Split rondo rewards between attacker and defender teams (`computeRondoReward` returns `{attackerReward, defenderReward}`, `RondoScenarioHandler` exposes `getLastDefenderReward()`, bridge sends 22-byte header for rondo with defender reward at offset 20, Python client assigns per-team rewards, `CooperativeRewardShaper` excludes right-team agents) — commit `9bcbbd6`
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
- [x] Pin reported results to checkpoint SHA + `git describe` — created `training/RESULTS_INDEX.md`: lineage table (validation_report v2 → seed42_best `ddaf4d38…`, deployed weights → seed44_best `6fb28ff1…`, legacy heuristic JSONs marked superseded), reporting rules, verification commands. The old `FILE_NOT_FOUND` ambiguity is resolved: the report's model IS seed42_best.

---

# Architectural Review Remediation (Report Findings A.1–B.8)

These tasks address the 8 findings from the architectural review report. Immediate
items (A) first, then Medium (B).

## A.1 — Training Throughput for PPO/MAPPO
**Problem:** PPO/MAPPO train on a single environment (blocking round-trip to bridge).
**Fix:** Optional parallel stepping via `--n-envs N` (multiple bridge instances).
- [x] `training/train_ppo.py` supports `--n-envs` flag (spawns bridges on ports 5050..5050+N-1)
- [x] `training/train_mappo.py` updated for parallel env stepping
- [x] `training/mappo_rollout.py` supports multi-env rollout collection
- [x] `training/gmn_gym.py` / `gmn_pettingzoo.py` accept `port` param; auto-start bridge per port
- [x] `training/bridge_server.ts` accepts `GMN_BIND_PORT` env var
- [x] Legacy `n_envs=1` single-env path preserved
- [x] Smoke test passed: `python training/train_ppo.py 300 --n-envs 2` runs end-to-end

## A.2 — Automate Contract Sync in CI
**Problem:** `sync_contracts.ts` exists but is not enforced; drift goes undetected.
**Fix:** Add `--check` mode + non-zero exit, wire into CI and build.
- [x] `scripts/sync_contracts.ts` updated with `--check` flag (exits non-zero on drift)
- [x] `package.json`: `check:contracts` script added (runs `sync:contracts --check`)
- [x] `.github/workflows/ci.yml`: contract-sync step added to CI pipeline
- [x] Build enforces sync: `npm run check:contracts` fails fast if Python constants drift from `Contract.ts`
- [x] Manual "health check" references replaced by automated CI enforcement

## A.3 — Consolidate Requirements Files
**Problem:** Conflicting root `requirements.txt` and `training/requirements.txt`.
**Fix:** Delete root file, point everything at training/.
- [x] Root `requirements.txt` deleted (`git rm`)
- [x] `README.md` updated: instruct users to install from `training/requirements.txt`
- [x] Verified all training scripts reference `training/requirements.txt` (CI already did)
- [x] `training/README.md` documents where to add new Python dependencies

## A.4 — Stricter TypeScript Checks
**Problem:** `noUnusedLocals`/`noUnusedParameters: false` allow dead code.
**Fix:** Enable strict checks, fix all resulting errors.
- [x] `tsconfig.json`: `noUnusedLocals: true`, `noUnusedParameters: true`, `noImplicitReturns: true`
- [x] `noFallthroughCasesInSwitch` was already `true`
- [x] Fixed all unused-variable errors across ~45 `.ts`/`.tsx` files (prefix with `_` or remove)
- [x] `npx tsc --noEmit` passes clean (0 errors)

## B.5 — Self-Play / Opponent Pools
**Problem:** Only left team learns; opponent is fixed `RuleBasedAgent`.
**Fix:** Opponent pool with pluggable selection strategy.
- [x] `training/opponent_pool.py`: `OpponentPool` class with `uniform`/`cyclic`/`elo` strategies
- [x] `training/gmn_gym.py` / `gmn_pettingzoo.py`: `set_opponent_difficulty()` method
- [x] `training/bridge_server.ts`: `/opponent` POST endpoint to set bot difficulty
- [x] `training/train_mappo.py`: `--opponent-difficulty` flag wired through
- [x] Pool supports rule-based difficulties today; learned-snapshot playback noted as future work

## B.6 — Extend Training to Full Matches
**Problem:** Only drill scenarios trained; full 11v11 not reachable.
**Fix:** New training script on the existing `11_vs_11` scenario.
- [x] `training/train_ppo_full.py`: PPO trainer targeting the `11_vs_11` scenario (reuses `run_ppo_training`)
- [x] Leverages parallel stepping (`--n-envs`) from A.1
- [x] Reuses existing `11_vs_11` scenario from `src/scenarios/ScenarioRegistry.ts`
- [x] Full law set (kick-off, throw-ins, goal kicks) handled by authoritative `GameEngine`

## Follow-up — Fix pre-existing `test_ippo_shared_reward.py` failure (surfaced by A.4 constructor guard)
**Problem:** `test_ippo_shared_reward.py` crashed at the constructor (`ConnectionRefusedError`) because `GMNMultiAgentEnv.__init__` unconditionally called `_connect_ws()` + `reset()` even with `auto_start_bridge=False`. Once the constructor was guarded (Option A), two further pre-existing test bugs were exposed: (1) `MockWS.recv` didn't accept the `timeout` kwarg that `_recv_frame` passes, and (2) the test packed a stale 17-byte header (`<f??BBffB`) while the current contract expects 18 bytes (`<f??BBffBB`, incl. `ball_owner_agent_idx`). The test had never actually passed — the crash masked the other bugs.
**Fix:** Guard constructor WS connection + eager reset with `auto_start_bridge`; fix mock signature + header format in the test.
- [x] `training/gmn_pettingzoo.py`: `if self.auto_start_bridge:` guards both `_connect_ws()` and `self.reset()` in `__init__`
- [x] `training/test_ippo_shared_reward.py`: `MockWS.recv(self, timeout=None, decode=None)` now matches websockets API
- [x] `training/test_ippo_shared_reward.py`: header packed as `<f??BBffBB` (18 bytes, adds `ball_owner_agent_idx=0`)
- [x] `reset()`/`step()` still connect on-demand when `ws_client is None` — preserves externally-booted-bridge pattern (`test_reward_shape_e2e.py`)
- [x] Verified: `test_ippo_shared_reward.py` → rewards 0.75 ✓ | `test_reward_shape_e2e.py` → PASS ✓ | `npm test` determinism suites ✓ | `tsc --noEmit` clean ✓ | `check:contracts` ✓

## B.7 — Rewrite CONTRIBUTING.md
**Problem:** Generic boilerplate referencing unrelated project (Google/Tensor2Tensor).
**Fix:** Project-specific contribution guidelines.
- [x] `CONTRIBUTING.md` fully rewritten with:
  - Dev environment setup (Node + Python)
  - Coding standards (TypeScript strict, Python PEP8)
  - Testing expectations (`npm test`, `pytest`)
  - PR process (contract sync, docs updated)
  - Link to architecture overview in README
- [x] All references to other projects removed

## B.8 — Document All Training Scripts
**Problem:** `training/` has ~120 files, undocumented in README.
**Fix:** Comprehensive `training/README.md`.
- [x] `training/README.md` created documenting every script (trainers, eval, networks, debug, tests)
- [x] Each script tagged Stable / Reference / Experimental
- [x] Sample commands, flag tables, environment notes included
- [x] Root `README.md` updated with pointer to `training/README.md`
- [x] Resolve report content thrash / FILE_NOT_FOUND class conflicts — SHA cross-check proves the seemingly contradictory reports used different checkpoints (seed42_best vs seed44_best vs best); documented in RESULTS_INDEX.md. Remaining sub-item: mark the legacy `comprehensive_eval_*` JSONs with `_legacy` suffix when regenerating next eval run.

## Follow-up — Critical bridge/engine bug fixes (5 commits)
**Problem:** Five verified bugs in the bridge/engine pipeline, each fixed and pushed as a separate commit.
- [x] Bug 1: `encodeErrorStepBinary()` used 17-byte header instead of 18-byte (non-rondo) / 22-byte (rondo). Added `isRondo` parameter, explicit `ballOwnerAgentId` at offset 17, and `defenderReward` at offset 20 for rondo. Updated all 4 call sites. Commit `5ccdc03`.
- [x] Bug 2: 3x `forEach(async ...)` race conditions in `stepBatch()`, `step()`, and `stepMulti()` where `engine.step()` fired before ONNX inference resolved. Replaced with `await Promise.all(...)`. Commit `1b9c408`.
- [x] Bug 2 follow-up: POST `/step_multi` HTTP handler missed `await` on `bridge.stepMulti()`, serializing a Promise (`{}`) instead of the real result. Added `await` and end-to-end regression test `training/tests/test_step_multi_http.py` (real HTTP request, asserts non-empty JSON shape). Commit `f32ce94`.
- [x] Bug 3: `RondoScenarioHandler.ts` overwrote `prevDefenderDistToBall` before passing it to `computeRondoReward()`, making the distance-closing bonus dead code. Added `currentDefenderDistToBall` field, pass current distance as `defenderDistToBall`, previous as `prevDefenderDistToBall`, then update prev after reward computation. Commit `d79f241`.
- [x] Bug 4: `GameEngine.ts` `resetToKickoff()` did not clear `lastScenarioResolutionEmitted`, permanently suppressing `scenario_complete`/`scenario_failed` events after a kickoff reset. Added `this.lastScenarioResolutionEmitted = false`. Commit `02bdcf2`.
- [x] Verified after each fix: `npx tsc --noEmit` clean, `python -m pytest training/tests/ -x -q` → 33 passed. All commits pushed to origin/main.

## Follow-up — CurriculumScheduler for train_mappo.py
**Problem:** `train_stage2_ppo.py`'s `run_stage2_curriculum()` is a hardcoded step schedule, not a performance-driven system. Need an actual threshold-based promotion/demotion scheduler wired into `train_mappo.py` with persistence.
- [x] `training/curriculum_scheduler.py`: new `CurriculumScheduler` with rolling-window promotion/demotion and JSON persistence. Stage ladder: `academy_empty_goal → academy_run_to_score → academy_pass_and_shoot_with_keeper → academy_3_vs_1_with_keeper → academy_3_vs_1_defender_2 → academy_3_vs_1_defender_3 → 5_vs_5 → 11_vs_11` (`academy_rondo_4v1` excluded as parallel track). Success detection reuses existing `gmn_pettingzoo.py` info dict. Commit `2a84dcb`.
- [x] `training/train_mappo.py`: `--curriculum` flag, `--scenario` required as starting stage, mid-run scenario switching via `scheduler.evaluate_and_step()` before each rollout, transition checkpoint saving (`mappo_curriculum_{promote|demote}_to_{stage}_step{N}.pt`), scheduler state persistence after every transition. Commit `2a84dcb`.
- [x] `training/mappo_rollout.py`: `completed_episodes` now includes per-episode `success` flag derived via `is_scenario_success()`.
- [x] `training/gmn_pettingzoo.py`: added `set_scenario()` helper for curriculum transitions.
- [x] `training/opponent_pool.py`: updated stale module docstring (ONNX snapshot playback now wired and verified).
- [x] Unit tests (`training/tests/test_curriculum_scheduler.py`): promotion, demotion, `min_episodes_before_promotion` floor, window reset on transition, persistence save/reload, mid-run reload with partial window, history logging. 18 tests passed.
- [x] Integration tests (`training/tests/test_curriculum_integration.py`): mocked `train_mappo.py` training loop verifies mid-run scenario promotion and transition checkpoint path. 2 tests passed.
- [x] Verified: `python -m pytest training/tests/ -x -q` → 53 passed. Commit `2a84dcb` pushed.
- [x] Bug fix: `min_episodes_before_promotion` was incorrectly gated on `total_episodes` (lifetime counter), allowing later stages to promote after only 1 episode. Fixed by adding `episodes_in_stage` counter, reset on promotion/demotion, and gating on per-stage counter. Persisted in `to_dict()`/`load()`. Added regression test for back-to-back promotion gating and persistence test for `episodes_in_stage` round-trip. Commit `81c3673`.