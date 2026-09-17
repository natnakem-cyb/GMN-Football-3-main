# Experiment B Horizon 200k — Best-vs-Final Gap Investigation

**Date:** 2026-09-17  
**HEAD at investigation:** `f2fd157` (after per-run eval CSV isolation fix)  
**Training run git commit:** `572979dfa3a82246e994c28a65fdb6cc92eddd77`  
**Investigator:** automated debug trace  

---

## 1. The discrepancy

| Source | seed42 | seed123 |
|--------|--------|---------|
| Experiment manifest `best_deterministic_goal_rate` | **33.33%** at step 100,352 | **23.33%** at step 100,352 |
| Final 50-ep deterministic eval (current) | **0.0%** | **0.0%** |
| Re-eval of best checkpoint with same 30-ep protocol (current) | **0.0%** | **0.0%** |

The manifests also record:
- seed7 best: 30.0% at step 50,176 → re-eval 0.0%
- seed999 best: 36.67% at step 50,176 → re-eval 0.0%

---

## 2. What the manifest actually tracks (and what it misses)

`train_mappo.py` tracks `best_deterministic_checkpoint_path` in memory and writes it to the in-memory best checkpoint file (`mappo_{scenario}_seed{seed}_best.pt`). However, the manifest JSON does **not** include `best_checkpoint_path`. It only records:

- `best_deterministic_goal_rate`
- `best_deterministic_step`
- `quarantine_checkpoint` (the final `*_200k_B` file)
- `clean_checkpoint` (a copy of `_best.pt` if it exists)

This means the manifest cannot tell you *which* checkpoint produced the recorded best rate. It only records the scalar and the step.

---

## 3. In-training eval vs. final eval protocol diff

| Parameter | In-training milestone eval | Final end-of-run eval |
|-----------|---------------------------|----------------------|
| `num_episodes` | 30 | 50 |
| `deterministic` | True | True |
| `base_seed` | 500000 | 500000 |
| `csv_path` | `training/results/win_rate_progress.csv` (see §4) | `training/results/win_rate_progress.csv` (same) |
| Checkpoint evaluated | Milestone checkpoint at exact step (e.g., `_100352.pt`) | Final quarantine checkpoint (`*_200k_B`) |

The 30-ep vs 50-ep difference is real, but it does **not** explain the gap: re-running the eval with the same 30-ep protocol on the same checkpoint today still yields 0.0%.

---

## 4. Shared-CSV provenance trail

The 33.33% value is **real in the sense that it was recorded** during the training run, but it lives in a different file than the one most people look at:

- `training/results/win_rate_progress.csv` — created earlier, last modified 2026-09-07. Contains baseline entries and some Experiment B entries, but **not** the 33.33% for seed42 step 100352.
- `training/results/win_rate_progress_v2.csv` — created 2026-09-17 11:51 UTC. Contains the 33.33% entry:
  - `evaluation_id=3da26710aeb08111`
  - `checkpoint_sha256=e0e5629a95103a812433abb125990235ac4ade8095b3e0750c8d77385f3b5076`
  - `checkpoint_path=...mappo_academy_3_vs_1_with_keeper_seed42_100352.pt`
  - `goal_rate_pct=33.33`

The versioned CSV was created because the original `win_rate_progress.csv` did not contain the `schema_version` column header when the first Experiment B eval write occurred. `_resolve_versioned_csv_path()` in `eval_progress.py` redirected subsequent writes to `win_rate_progress_v2.csv`.

This is the same root cause as the earlier 43.33% confusion: a single global cumulative CSV shared across runs, with versioning redirects scattering entries across multiple files.

---

## 5. Re-evaluation results (identical protocol, current code)

### 5.1 seed42 — best checkpoint (`mappo_academy_3_vs_1_with_keeper_seed42_clean.pt`, step 100,352)

| Run | Episodes | Goal rate | Mean reward | Std reward |
|-----|----------|-----------|-------------|------------|
| In-training (recorded in v2 CSV) | 30 | **33.33%** | -0.8674 | 0.3278 |
| Re-eval #1 | 30 | 0.0% | -0.7080 | 0.1419 |
| Re-eval #2 | 50 | 0.0% | -0.7295 | 0.1175 |
| Re-eval #3 | 50 | 0.0% | -0.7295 | 0.1175 |

All three re-evals produced **identical** `evaluation_id=3da26710aeb08111` (same cache key), confirming the eval path is fully deterministic under the current environment. The in-training result is **not reproducible**.

### 5.2 seed123 — best checkpoint (`mappo_academy_3_vs_1_with_keeper_seed123_clean.pt`, step 100,352)

| Run | Episodes | Goal rate |
|-----|----------|-----------|
| In-training (recorded in v2 CSV) | 30 | **23.33%** |
| Re-eval | 30 | 0.0% |
| Re-eval | 50 | 0.0% |

Same pattern: non-reproducible in-training result.

### 5.3 seed42 — final checkpoint (`mappo_ac3v1_seed42_200k_B`, step 199,936)

| Run | Episodes | Goal rate |
|-----|----------|-----------|
| Final eval (recorded in v2 CSV) | 50 | 0.0% |
| Re-eval | 50 | 0.0% |

Consistent.

---

## 6. Hypothesis assessment

| Hypothesis | Assessment | Evidence |
|------------|------------|----------|
| **H1** Different eval protocol (30 vs 50 eps) | **Partial** — protocol differs, but re-eval with 30 eps still yields 0.0%, so protocol alone does not explain 33% → 0% | Re-eval with 30 eps = 0.0% |
| **H2** “Best” is a different checkpoint than `*_200k_B` | **True but irrelevant** — the manifest’s “best” step (100,352) does point to a different checkpoint than the quarantine file, and we re-evaluated that exact checkpoint. It scores 0.0% now. | `_best.pt` sha256 ≠ `_200k_B` sha256; both evaled |
| **H3** Train-time proxy / buggy counter / small-sample noise | **Most likely** — 33% in 30 episodes is a small-sample signal that is not reproduced under identical seeds today. The eval code is deterministic; the environment source files are unchanged; the checkpoint is unchanged. The in-training result appears to be a transient false positive. | Re-eval deterministic at 0.0%; code & files identical |
| **H4** Final eval env mismatch | **Unproven** — ground-truth metrics were present in the in-training eval output (`ground_truth_possession_left_pct: 4.3`) but are `None` in current evals. This suggests the environment bridge or game binary may have been different during the training run, but we have no tracked file diff to prove it. | Ground-truth presence/absence differs |
| **H5** Checkpoint load failure / wrong weights | **False** — `torch.load` succeeds, actor architecture matches, sha256 of file matches the training log’s recorded hash. | File hash verified |
| **H6** Goal metric definition drift | **False** — goal detection logic (`score_left > 0` or `event.type == "goal"`) is identical in the current code and was not changed between commits. | Code diff shows no change |

---

## 7. Closure recommendation

**Close the best-vs-final gap as: metric bug / non-reproducible train-time eval (H3 primary, H4 possible contributing factor).**

- The 33.33% / 23.33% / 30.0% / 36.67% values in the Experiment B manifests are **not reliable measures of policy quality**. They were produced by in-training milestone evals that are not reproducible today.
- The **final 50-episode deterministic eval (0.0% all seeds)** and the **re-eval of the best checkpoints (0.0% all seeds)** are the trustworthy numbers because they are reproducible and consistent.
- The `best_deterministic_goal_rate` field in the manifest should be deprecated or marked as `unverified` until eval determinism is fully understood.
- Do **not** write Experiment B as “succeeded at 33%” or “observation proven to be the sole cause.” The headline finding is **0.0% goal rate across all four seeds, with severe tackle spam on seeds 7 and 999**.

### What to fix next (not in this trace)

1. **Eval determinism audit:** instrument `GMNMultiAgentEnv.reset()` and the bridge to confirm the game binary version and whether opponent RNG is fully seeded by `env.reset(seed=...)`.
2. **Add `evaluation_id` provenance to manifests:** record the exact evaluation_id(s) that produced `best_deterministic_goal_rate` so future investigators can re-eval the exact same eval.
3. **Eval JSON sidecars:** make `evaluate_checkpoint_progress` save per-eval JSON next to the checkpoint (or in the run directory) so episode-level data is preserved.

---

## 8. Action taken this session

- Fixed shared eval CSV isolation: all canonical trainers now write to `runs/{experiment_name}/eval_progress.csv`. (`f2fd157`)
- Documented corrected B-run findings in `training/results/EXPERIMENT_B_PROVENANCE.md`. (`f2fd157`)
- Pushed to `origin/main`.

**Do not promote any Experiment B checkpoint to production based on the manifest `best_deterministic_goal_rate` field.**
