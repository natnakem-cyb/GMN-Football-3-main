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

## Audit fix log (2026-09-12 — P0/P1 audit fixes, branch `fix/audit-p0-p1`)

| Issue | File(s) | Fix | Verification |
|---|---|---|---|
| 1 (P0) | `training/mappo_rollout.py` | All three collectors (`collect_rollout`, `collect_rollout_parallel`, `collect_rollout_batched`) now build `terminal_info` (score + event) and emit `"success"` (via `is_scenario_success`) on every `completed_episodes` entry, alongside the legacy `goal` metric. Parallel collector previously had NO success field; batched likewise. Also fixed: parallel collector's `_terminal_info` now reads `score` from `info` (the `_last_frame_score` diagnostic attr is single-path only); batched collector's `goal` metric now reads the top-level shared info `score.left` (real batched env delivers a top-level info dict, not per-agent). | `training/tests/test_rollout_success_emission.py` (fake single/parallel/batched envs; 5_vs_5 win, rondo, academy goal-stage cases) — all green. |
| 2 (P0) | `src/engine/ObservationEncoder.ts`, `src/engine/scenarios/RondoScenarioHandler.ts` | Rondo defender reward is now a **transition** reward: `defenderJustWonPossession` flag computed in the handler (ownership change to right) and paid +0.2 ONCE — not +0.2/tick (was ≈12.0 over 60 ticks). Attacker +0.01 possession / +0.1 pass behavior unchanged. | `npx tsc --noEmit` clean; `training/test_rondo_transition_reward.ts` — 60 ticks of continuous right possession = 0.2, transition tick = 0.2, non-transition = 0. |
| 3 (P0) | `src/engine/ObservationEncoder.ts`, `src/engine/GameEngine.ts` | Ball-progress checkpoint is now **possession-gated** (Option A): `computeReward` takes `ballOwnerTeam`; when supplied (GameEngine always supplies it), checkpoint AND high-water-mark update only occur while the controlled training team owns the ball. Loose ball / right-driven X advance pays 0 and does not raise the mark (regaining possession is not penalized). Legacy ungated behavior preserved when the param is omitted (undefined) for old call sites. | `training/test_reward_equivalence.ts` — left-owned advance pays, loose-ball and right-owned advance pay 0, legacy path preserved. |
| 4 (P0/P1) | `src/scenarios/ScenarioRegistry.ts` | Renamed `create_triangle` objective text from "Connect 2+ passes without losing possession" to "Complete 2+ passes in the episode" to match the engine's aggregate `completedPasses` evaluation (chain semantics deferred — see residual risks). Scenario id unchanged (wire key). | `npx tsx training/test_scenario_completion.ts` — text-match test plus ≥2-passes completion tests green. |
| 5 (P1) | `training/curriculum_scheduler.py`, `training/README.md` | Documented: `CURRICULUM_STAGES` is a strict subset of the 12-scenario registry (rondo = parallel track; aggressive/shifted/randomized = held-out variants); `academy_3_vs_1_with_keeper` is actually 3A_vs_1D_plus_GK (id counts outfield opponents; ids must not be renamed). | Docs only. |
| 6 (P1) | `training/mappo_update.py` | Documented (no code change): critic loss samples scale as T×num_agents because the joint state is repeated per agent. For shared returns the repeats are identical pairs so loss scale is invariant; for per-agent returns the state-only critic receives the same state with different targets — a "mean over agents" normalization would change critic semantics, so it is deferred. | Code review; follow-up opened in this log. |

Residual risks / deferred: `create_triangle` chain semantics (needs per-possession pass-chain tracking in engine stats); critic per-agent-return normalization (semantic redesign); offside FIFA suite; physics real-world calibration; live-bridge 5k smoke re-run after M1b (pre-existing).

## Semantic fix log (2026-09-11)

| ID | File | Fix description | Verification |
|---|---|---|---|
| M1 | `training/gmn_pettingzoo.py` | `CooperativeRewardShaper.compute_shaped_rewards` now processes `PASS_INTERCEPTED`, `PASS_FAILED`, `TURNOVER_CONCEDED` regardless of the engine's `team` tag. Added `penalty_turnover` (-0.10) parameter. | New tests: `test_pass_intercepted_with_right_team_tag_penalizes_left_agent`, `test_turnover_conceded_with_right_team_tag_penalizes_left_agent`, updated `test_turnover_chain_break`. |
| M2 | `training/gmn_pettingzoo.py` | Removed terminal-step guard in `step()` that stripped shaped rewards on terminal frames. Implemented assisted-goal bonus: when `pass_chain_length > 0` and a goal is scored, all active left-team agents receive `r_assisted_goal` (+0.50). | New tests: `test_goal_bonus_received_on_terminal_step`, `test_assisted_goal_bonus_distributed_to_all_active_agents`, `test_goal_without_pass_chain_does_not_give_bonus`. |
| H1 | `training/bridge_server.ts` | `stepBatch` now passes the per-env `engine` variable into `runOnnxInference` and `_applyRuleBasedAction` fallback instead of always using `this.engine`. | Code review; no TS unit-test framework present. |
| M5 | `training/gmn_pettingzoo.py` | `step_batch` now maintains per-sub-env `pending_pass`, `reward_shaper`, and `last_actions`. Added `_build_shaper_events`, `_apply_shaping_for_env`, and refactored `_resolve_pending_pass_state` as stateless helper. Non-rondo batched rewards normalized to per-agent dicts. | New tests: `TestM5BatchedRewardShaping` (5 tests covering event mapping, pending-pass state machine, shaping application, and env-state initialization). |
| M1b | `training/gmn_pettingzoo.py` | **Closed the M1 attribution gap.** Added `previous_left_ball_carrier` state to `CooperativeRewardShaper`. On `PASS_INTERCEPTED`/`PASS_FAILED`/`TURNOVER_CONCEDED`, the shaper now falls back to the last remembered left carrier when the event's `agent_id` is missing or non-left (the live wire frame reports `ball_owner_agent_idx=255`, so events arrive with no `agent_id`). Carrier is updated on every left-owned frame, preserved across ownerless/right-owned frames (interceptions arrive precisely as ownerless frames — the old `current_holder_id` was reset to `None` there, which is why it could not be reused), and cleared after a victim event resolves it. Explicit left `agent_id` on a well-formed event still takes precedence. | New tests: `test_m1_interception_without_agent_id_uses_previous_left_carrier` (live-style: drives `_build_shaper_events(...,255)` → `compute_shaped_rewards`, asserts `-0.10` on the previous carrier), `test_m1_interception_without_prior_carrier_charges_nothing` (negative case), `test_m1_explicit_agent_id_takes_precedence_over_carrier`. Integration verification confirmed `-0.10` delivered end-to-end through the real `step()` event-building path. |

## Phase 1 Forensic Smoke Test (2026-09-11)

Command:
```bash
python -m training.train_mappo \
  --scenario academy_3_vs_1_with_keeper \
  --seed 42 \
  --timesteps 5000 \
  --n-envs 1
```

### M1 – Turnover / Interception penalty

**Status: NOT VERIFIABLE from this smoke test**

Observations:
- The smoke test generated `event_code=15` (`pass_intercepted`) terminal events.
- Debug trace showed `ball_owner_agent_idx=255` (out-of-range sentinel) and `pending_pass=None` for these events.
- `total_pass_completed=0` throughout the entire 5k run — the agents never initiated a pass, so no pending-pass state existed to attribute the turnover to a specific left passer.
- Terminal step rewards for interception events were `0.0` or `-0.006667` (ball-hogging), **not** the expected `-0.033333` average (`-0.10 / 3` agents) from the turnover penalty.
- `total_turnovers` counter did increment (19 → 36 → 56 → 77 across checkpoints), confirming turnover events are counted. However, counter increment does not prove the per-agent penalty was applied.

Evidence snippet (terminal JSONL episode ending with interception):
```json
{"terminal_frame_reward": 0.0, "terminal_shared_reward": 0.0, "reward_before_terminal": -0.203410, "episode_reward": -0.209743, "terminal_event_code": 15, "terminal_score": {"left": 0, "right": 0}, "episode_length": 62}
```

**Unit-test verification:**
- `test_pass_intercepted_with_right_team_tag_penalizes_left_agent` — PASSES
- `test_turnover_conceded_with_right_team_tag_penalizes_left_agent` — PASSES

**Gap (CLOSED in M1b):** The engine's `pass_intercepted` binary frame reports `ball_owner_agent_idx=255` (no owner) and does not embed the left passer's ID. The original M1 fix correctly penalizes the left agent **when the event dict carries a valid left `agent_id`**, but the engine does not provide that ID in the observed scenario. Without a pending-pass state (agents don't pass), the shaper could not attribute the penalty.

**Resolution (M1b):** Implemented option (b) — the shaper now tracks `previous_left_ball_carrier` across steps and uses it as the penalty target when `agent_id` is missing or a right-team player. This was preferred over an engine contract change because it requires no wire-format change and the engine-side `passerAgentId` can be added later as an optional hardening. An integration check confirmed the `-0.10` penalty is now delivered end-to-end through the real `step()` event-building path (`_build_shaper_events(...,255)` → `compute_shaped_rewards`), even though a full live-bridge smoke run has not yet been re-executed.

### M2 – Terminal goal reward (no assist)

**Status: VERIFIED**

Observations:
- Two terminal goal events (`event_code=1`) observed in the smoke test.
- Engine raw reward on goal steps: `binary_frame=2.008532` and `binary_frame=2.008626`.
- `collector_shared=2.008532` and `2.008626` — the terminal shaped reward equals the engine reward. No assisted-goal bonus was added because `pass_chain_length=0` (no passes occurred).

Evidence snippet (terminal JSONL goal episode):
```json
{"terminal_frame_reward": 2.008532, "terminal_shared_reward": 2.008532, "reward_before_terminal": -0.120000, "episode_reward": 1.888532, "terminal_event_code": 1, "terminal_score": {"left": 1, "right": 0}, "episode_length": 49}
```

- Confirmed: old hardcoded `+2.00` shaper bonus is **not** still being added (total equals engine reward, not engine + 2.00).

### M2 – Assisted-goal bonus

**Status: NOT VERIFIABLE from this smoke test**

- `total_pass_completed=0` throughout. No pass chains were built, so no assisted-goal bonus could be triggered.
- Unit tests (`test_assisted_goal_bonus_distributed_to_all_active_agents`) pass.

### Diagnostics

| Checkpoint | total_pass_completed | total_goals | total_turnovers |
|---|---|---|---|
| Step 1024 | 0 | 0 | 19 |
| Step 2048 | 0 | 1 | 36 |
| Step 3072 | 0 | 3 | 56 |
| Step 4096 | 0 | 3 | 77 |

Counters move monotonically in the expected direction when corresponding events occur.

### Recommendation

**M1 attribution is now closed (shaper-side, option b).** The `-0.10` per-agent penalty is delivered to the previous left ball carrier even when the interception frame reports `ball_owner_agent_idx=255` and carries no `agent_id`. This was verified end-to-end through the real `step()` event-building path.

**Remaining gap before the 200k baseline:** The *live-bridge* 5k smoke has **not** been re-run after M1b. The fix is proven at the unit/integration level (real `_build_shaper_events` → `compute_shaped_rewards` path, all 8 `TestM1M2SemanticFixes` tests green), but the original forensic observation that turnovers are "counted but the −0.10 is not delivered" has not yet been confirmed as *fixed* in a live bridge episode. Re-run the forensic smoke (`python -m training.train_mappo --scenario academy_3_vs_1_with_keeper --seed 42 --timesteps 5000 --n-envs 1`) to observe a live −0.10 on an interception before launching 200k.

Note: `total_pass_completed=0` in prior smoke means passing behavior has not yet emerged, so the carrier is established by dribbling/possession frames, not by passes. The fix correctly handles both origins.
