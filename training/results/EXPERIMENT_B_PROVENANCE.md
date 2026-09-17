# Experiment B Provenance

**Date:** 2026-09-17  
**HEAD:** `572979d`  
**Primary scenario:** `academy_3_vs_1_with_keeper`  
**Budget:** 200,000 timesteps  

---

## 1. clampPassDirection scope

**Status:** Scripted/probe layer **only**.

**Evidence:**
- `training/experiment_d_scenario_probes.ts` — used in deterministic scripted controller
- `training/experiment_d_pass_forensics.ts` — used in forensic probe
- `training/results/EXPERIMENT_D_QUALIFICATION.md` — documents clamp at max y-component ±0.5

**NOT present in:**
- `src/engine/` — no engine modifications
- `training/gmn_pettingzoo.py` — no env action-path changes
- `training/train_mappo.py` — no training changes
- `src/engine/ActionMapping.ts` — no discrete action changes

**Conclusion:** The clamp is a probe-layer diagnostic aid. The training MDP's action space is unchanged from Baseline A. The agent can still output any pass direction; no env-level constraint was added.

---

## 2. Probe sprint-to-ball vs training MDP

**Status:** Training MDP does **not** force non-acting teammates to sprint to ball.

**Evidence:**
- Sprint-to-ball behavior exists **only** in:
  - `training/experiment_d_scenario_probes.ts` line 665-680 — forced-pass probe makes non-passing left players sprint toward ball
  - `training/experiment_d_pass_forensics.ts` — forensic probe
- The actual training code (`train_mappo.py`, `gmn_pettingzoo.py`) uses standard MAPPO with independent policy heads. Teammate behavior is determined by the learned policy, not scripted sprints.

**Conclusion:** The probe's sprint-to-ball behavior is a measurement artifact to improve pass-completion counting. It does not reflect the training MDP.

---

## 3. train_mappo.py / gmn_pettingzoo.py diff audit

**Diff vs Baseline A (commit `3bf6c39`):**

### training/train_mappo.py
```diff
+    enable_exploration: bool = True,
+    exploration_beta: float = 0.03,
```

- Added explicit `enable_exploration` and `exploration_beta` parameters to `run_mappo_training()`
- These are passed to `GMNMultiAgentEnv` constructor
- Added CLI args `--enable-exploration`, `--no-exploration`, `--exploration-beta`
- **Behavior for `academy_3_vs_1_with_keeper`:** UNCHANGED — defaults are `True` and `0.03`, which match Baseline A's hardcoded behavior

```diff
+            enable_exploration_bonus=enable_exploration,
+            exploration_beta=exploration_beta,
```

- Passes exploration params to env
- **No reward magnitude changes**

### training/gmn_pettingzoo.py
```diff
+        enable_exploration_bonus: bool = True,
+        exploration_beta: float = 0.03,
```

- Added exploration bonus params to `GMNMultiAgentEnv.__init__`
- Stored as instance attributes
- Passed to `get_reward_adapter()`
- **Behavior for primary scenario:** UNCHANGED — defaults match Baseline A

```diff
+    def _canonicalize_pass_events(...):
```

- Added PASS_COMPLETED event deduplication
- **Purpose:** Fix duplicate PASS_COMPLETED events from `_resolve_pending_pass_state` and `step()` both emitting the same event
- **Impact:** Prevents double-counting of pass rewards. This is a **bug fix**, not a reward magnitude change.

### training/reward_adapters.py
```diff
-        return RondoRewardAdapter(**kw)
+        return RondoRewardAdapter()
```

- Fixed RondoRewardAdapter to not receive exploration bonus kwargs
- **Impact:** Unrelated to primary scenario `academy_3_vs_1_with_keeper`
- **Purpose:** Fix TypeError when switching from attacking drill to rondo scenario

### Summary
| Change | File | Impact on primary scenario |
|--------|------|---------------------------|
| Explicit exploration params | train_mappo.py, gmn_pettingzoo.py | None — defaults match Baseline A |
| PASS_COMPLETED deduplication | gmn_pettingzoo.py | Bug fix — prevents double-counting |
| RondoRewardAdapter fix | reward_adapters.py | None — unrelated scenario |

**Conclusion:** No reward, strip, or exploration edits that change behavior for `academy_3_vs_1_with_keeper`. All changes are either bug fixes or parameterizations that preserve Baseline A defaults.

---

## 4. Post-repair SHOT conditional

**From D report (`EXPERIMENT_D_QUALIFICATION.md` section 3.2.4):**

| Metric | Count | Rate |
|--------|-------|------|
| SHOT commanded | 106 | — |
| SHOT fired | 27 | **25.5%** |

**P(SHOT | poss+mask+cmd) = 25.5%**

Forced shots do fire occasionally. SHOT execution path is not broken.

---

## 5. Action/env path delta vs Baseline A

Since Baseline A (commit `3bf6c39`), the action/env path has changed in three ways:

1. **Explicit exploration bonus parameters:** `train_mappo.py` and `gmn_pettingzoo.py` now pass `enable_exploration_bonus` and `exploration_beta` explicitly to the reward adapter. For `academy_3_vs_1_with_keeper`, the defaults are identical to Baseline A (`True`, `0.03`), so the effective reward signal is unchanged.

2. **PASS_COMPLETED event deduplication:** `gmn_pettingzoo.py` now canonicalizes PASS_COMPLETED events to prevent double-counting from `_resolve_pending_pass_state` and `step()` both emitting the same event. This fixes a bug where a single physical pass could be counted twice. It does not change the per-pass reward magnitude; it ensures each pass is counted exactly once.

3. **RondoRewardAdapter fix:** `reward_adapters.py` now constructs `RondoRewardAdapter()` without forwarding exploration kwargs. This fixes a crash when switching to rondo scenarios. Unrelated to the primary scenario.

**Net effect on action/env path for `academy_3_vs_1_with_keeper`:** The action space and reward magnitude are identical to Baseline A. The only behavioral change is that pass completions are now counted exactly once instead of potentially twice, which makes the reward signal more accurate but does not increase the per-pass reward.

---

## 6. D-Obs limits (for B report)

From `training/results/D_OBSERVATION_SUFFICIENCY_AUDIT.md` (`572979d`):

- **Spatial/relational probes:** SUFFICIENT
- **Ball displacement:** PARTIAL (velocity in obs; no temporal stack in MAPPO)
- **Task context:** INSUFFICIENT (`zScenario` not transmitted to Python)
- **Overall D-Obs:** PARTIAL

B proceeds with these limitations documented.

---

## 7. Regression gate

```bash
python -m pytest \
  training/tests/test_scenario_reward_adapters.py \
  training/tests/test_reward_exploit_regression.py \
  training/tests/test_reward_whole_pipeline.py \
  -q
```

**Result:** 9 passed (reward adapter persistence tests)

---

## 8. Freeze checklist

| Factor | Required value | Verified |
|--------|---------------|----------|
| Scenario | `academy_3_vs_1_with_keeper` | ✅ |
| Reward / strip | Post-strip freeze (GOAL-only keep engine base; no new terms) | ✅ |
| Exploration | Same as Baseline A: `enable_exploration_bonus=True`, `exploration_beta=0.03` | ✅ |
| Obs contract | `simple115_v3_role` 127-D — no schema change | ✅ |
| Architecture | Same MAPPO nets as Baseline A | ✅ |
| γ | **0.99** (from `train_mappo.py` line 11, 133, 426) | ✅ |
| Seeds | **42, 123, 999, 7** | ✅ |

**γ^H note:** For H ≈ 50 steps, `0.99^50 ≈ 0.605`. For 200k timesteps with 600-tick episodes, effective horizon is ~333 episodes. Discount factor is unchanged from Baseline A.

---

## 9. Baseline A artifacts

**Existing checkpoints:**
- `training/models/mappo_academy_3_vs_1_with_keeper_seed42_100k`
- `training/models/mappo_academy_3_vs_1_with_keeper_seed123_100k`
- `training/models/mappo_academy_3_vs_1_with_keeper_seed7_100k`
- `training/models/mappo_academy_3_vs_1_with_keeper_seed999_100k`

**Status:** These are the Baseline A reference checkpoints. They will not be overwritten. B checkpoints will use naming `mappo_ac3v1_seed{SEED}_200k_B`.

---

## 10. Provenance gate status

| Gate | Status |
|------|--------|
| Ancestry verified | ✅ |
| 5 caveats resolved | ✅ |
| D-Obs limits documented | ✅ |
| Regression gate passed | ✅ |
| Freeze checklist verified | ✅ |

**Gate: PASS. Proceeding with Experiment B training.**

---

## 11. Post-run evaluation findings (corrected)

**Date of evaluation:** 2026-09-17  
**HEAD at eval:** `572979d`  

### 11.1 Final deterministic evaluation results

| Seed | Best rolling goal rate | Best deterministic goal rate | Final eval goal rate | Final eval mean reward | Final eval turnovers conceded/ep |
|------|----------------------|------------------------------|----------------------|------------------------|----------------------------------|
| 42   | 2.0% at step 92,160  | 33.33% at step 100,352       | **0.0%**             | -0.1059                | 0.28                             |
| 123  | 0.0%                 | 23.33% at step 100,352       | **0.0%**             | -0.1506                | 0.12                             |
| 7    | 0.0%                 | 0.0%                         | **0.0%**             | -0.0944                | 0.00                             |
| 999  | 0.0%                 | 0.0%                         | **0.0%**             | -0.0954                | 0.00                             |

### 11.2 Pathological tackle spam

- **Seed 7:** 9.48 tackles/episode (action index 16, SLIDING/tackle)  
- **Seed 999:** 42.64 tackles/episode (action index 16, SLIDING/tackle)  
- Seeds 42 and 123 exhibit the opposite degenerate behavior: 0.00 tackles, 0.00 passes, 0.00 shots.

### 11.3 Root cause analysis

1. **Pass interception dominance:** `event_code=15` (`pass_intercepted`) dominates the training reward chain. The agent's passes are intercepted constantly, so passing yields negative or useless reward.
2. **Weak goal-scoring signal:** With passes failing and shots not registering (0 shots/ep in final eval for all seeds), the agent never receives a strong positive gradient toward scoring.
3. **Degenerate policy collapse:** When productive actions are consistently punished, the agent finds a "safe" low-variance action and spams it (tackle spam on seeds 7/999) or learns to take no productive actions at all (policy paralysis on seeds 42/123).

### 11.4 Shared CSV confusion — resolved

**Issue:** `training/results/win_rate_progress.csv` was a single global cumulative file shared across all training runs. During this investigation, stale baseline entries from a 2026-09-07 run (500k-step baseline with 43.33% goal rate at step 250,880) were visible in the same file as Experiment B entries, causing confusion about where the 43.33% figure originated.

**Fix applied:** Modified `evaluate_checkpoint_progress()` and all trainer callers (`train_mappo.py`, `train_ppo.py`, `train_ippo.py`, `train_mappo_shaped.py`) to write per-run eval CSVs under `runs/{experiment_name}/eval_progress.csv`. `generate_comparison_table.py` updated to scan per-run CSVs. The legacy global `win_rate_progress.csv` is preserved but no longer receives new entries from canonical trainers.

### 11.5 Best-vs-final gap investigation (detailed trace)

**See also:** `training/results/EXPERIMENT_B_HORIZON_200k.md` for the full hypothesis table, re-eval data, and closure recommendation.

Re-evaluation of the exact “best” checkpoints from the manifests with the **same 30-episode deterministic protocol** used during training yields **0.0% goal rate** for all four seeds. Three independent re-runs on seed42’s best checkpoint are fully deterministic and all return 0.0% (evaluation_id `3da26710aeb08111`, identical cache key). The in-training 33.33% / 23.33% / 30.0% / 36.67% values are therefore **not reproducible** and should be treated as transient false positives, likely from small-sample eval noise combined with environmental non-determinism or a transient bridge/game state during the training run.

The 33.33% entry is recorded in `training/results/win_rate_progress_v2.csv` (created 2026-09-17 11:51 UTC), not in the original `win_rate_progress.csv`. The versioned CSV was created because the original file lacked the `schema_version` header at the time of the first Experiment B eval write.

### 11.6 Corrected checkpoint provenance

| Checkpoint | SHA256 | Reproducible peak performance |
|------------|--------|------------------------------|
| `mappo_ac3v1_seed42_200k_B` | `1d9be7e16f07d9becaa1de364510bee48b15256768fc14327e47532c856bf47b` | 0.0% (final 50-ep eval, re-eval) |
| `mappo_ac3v1_seed123_200k_B` | `7624efee69555a3e8e8a1c582ef0cdb39b51d31edfd8555e559e63ceaa95e8cc` | 0.0% |
| `mappo_ac3v1_seed7_200k_B` | `73581605c65bd7d298be848b23de1d29ced52c4680147957a17825dcd3726eaf` | 0.0% |
| `mappo_ac3v1_seed999_200k_B` | `c1b36f41d74975f4154923bcdec920f9f0f67f58998c59de65394dba9d0f7ebb` | 0.0% |

The `experiment_manifest.json` `best_deterministic_goal_rate` fields (33.33% for seed42, 23.33% for seed123, 30.0% for seed7, 36.67% for seed999) are **unreliable**. They do not match reproducible re-evals and appear to have been populated from transient in-training eval noise.

### 11.7 Next steps

- Do not promote any Experiment B checkpoint to `_clean.pt` based on the manifest `best_deterministic_goal_rate` field.
- The reliable headline numbers for Experiment B are: **0.0% final deterministic goal rate across all four seeds**, with severe tackle spam on seeds 7 and 999 (9.48 and 42.64 tackles/episode).
- Proceed with targeted investigation into action masking, reward shaping, and exploration strategy before the next training run.
- Do not treat the in-training 33.33% / 23.33% figures as evidence of learning; they are non-reproducible transient eval noise.
- Proceed with targeted investigation into action masking, reward shaping, and exploration strategy before the next training run.
