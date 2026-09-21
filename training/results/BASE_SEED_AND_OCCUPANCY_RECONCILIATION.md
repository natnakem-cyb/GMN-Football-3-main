# BASE_SEED AND OCCUPANCY RECONCILIATION

**Date:** 2026-09-21  
**HEAD:** `472b92de6e768756f62669af58add3fce48b1091`  
**Phase:** Critic/GAE Horizon Forensics — Base Seed Lock + Occupancy Reconciliation

---

## 1. Canonical Base Seed Source of Truth

The canonical `base_seed` for all deterministic evaluation measurements is **500,000**.

### Primary Sources

| Source | Location | Value |
|--------|----------|-------|
| CANONICAL_METRICS_CONTRACT.md §2.1 | Line 73 | `Base seed \| 500,000` |
| CANONICAL_METRICS_CONTRACT.md §2.1 | Line 74 | `Episode seeds \| 500000, 501009, 502018, ..., 549441 (base + episode×1009)` |
| `training/eval_actor_reweight.py` | Line 26 | `parser.add_argument("--base-seed", type=int, default=500000)` |

### Script Default Fix

`training/eval_canonical_three_agent_measurement.py` previously defaulted to `DEFAULT_BASE_SEED = 700000`, which conflicted with the canonical contract. This has been corrected to `DEFAULT_BASE_SEED = 500000`.

---

## 2. Seed 42 `n_onball` Swing: 17 → 74 → 13

Three distinct values were observed for seed 42's on-ball frame count across measurement runs. The table below traces each value to its source.

| Value | Source File | Base Seed | Measurement Method | Commit / Code |
|-------|-------------|-----------|-------------------|---------------|
| **17** | `training/results/onball_occupancy_summary.csv` | 700,000 | **Post-step** (defective) | `5908073` |
| **74** | `training/results/onball_occupancy_prestep_summary.csv` | 700,000 | **Pre-step** (corrected) | `085ec85` |
| **13** | `training/results/post_reweight_logit_prestep_reconciled_summary.csv` | 500,000 | **Pre-step** (corrected) | `472b92d` |

### 2.1 Swing 17 → 74: Measurement Method Fix

The jump from 17 to 74 is caused by the **temporal-alignment bug fix** in commit `085ec85`.

**Defective implementation (pre-085ec85):**
```python
# Old collector used POST-step ball ownership to determine retention:
ball_owner_agent_idx = getattr(env, "_last_ball_owner_agent_idx", 255)
agent0_has_ball = (ball_owner_agent_idx == 0)   # post-step, wrong
```

This retained frames where agent 0 happened to own the ball **after** the step, rather than **before** the step. In many cases, the ball changes hands during a step, so post-step ownership is a strict subset of pre-step team possession.

**Corrected implementation (085ec85):**
```python
# New predicate uses PRE-step team possession:
from training.prestep_onball import is_prestep_onball
# retain iff pre_step_obs[95] == 1.0 (left-team has ball pre-step)
```

The corrected method retains **all** ticks where the left team has the ball before the agent acts, regardless of who specifically owns it post-step. This expands the retained frame set.

**Verification:** `training/tests/test_prestep_onball_temporal_alignment.py` covers four synthetic cases:

| Case | Pre-step obs[95] | Post-step owner | Retained (old) | Retained (new) |
|------|------------------|-----------------|----------------|----------------|
| A | 1 (left) | ≠ 0 | False | **True** |
| B | 0 (none) | 0 | **True** | False |
| C | 1 (left) | 0 | True | True |
| D | 0 (none) | ≠ 0 | False | False |

Case B is the critical regression: the old code incorrectly retained frames where the team did **not** have the ball pre-step but agent 0 happened to get it post-step. Case A is the primary expansion: frames where the team had the ball pre-step but agent 0 did not own it post-step were previously dropped.

All 9 synthetic tests pass under the corrected implementation.

### 2.2 Swing 74 → 13: Base Seed Change

The drop from 74 to 13 is caused by changing `base_seed` from **700,000** to **500,000**.

Different base seeds produce different episode seeds via the formula:

```
ep_seed = base_seed + episode_index * 1009
```

For `base_seed=700000`: episode seeds are `700000, 701009, 702018, ...`  
For `base_seed=500000`: episode seeds are `500000, 501009, 502018, ...`

Different episode seeds drive different trajectory randomizations, which change:
- Ball spawn positions
- Opponent movement patterns
- Agent decision points

These trajectory differences alter how often the left team has the ball at pre-step, producing different `n_onball` counts.

### 2.3 Summary

The 17 → 74 → 13 swing decomposes as:

```
17 (old post-step, base_seed=700000)
  ↓ fix measurement method (post-step → pre-step)
74 (corrected pre-step, base_seed=700000)
  ↓ change base_seed (700000 → 500000, canonical)
13 (corrected pre-step, base_seed=500000) ← CANONICAL
```

Both changes are orthogonal: the method fix changes **which frames are retained**, while the base seed change changes **which trajectories are evaluated**.

---

## 3. Canonical `n_onball` for All Seeds (base_seed=500000)

The final canonical measurement was performed under:

- **Script:** `training/eval_canonical_three_agent_measurement.py`
- **Commit:** `472b92de6e768756f62669af58add3fce48b1091`
- **Scenario:** `academy_3_vs_1_with_keeper_onball`
- **Episodes:** 50 per seed
- **Base seed:** 500,000
- **Deterministic:** argmax
- **Measurement method:** Pre-step obs[95] team possession

### 3.1 On-Ball Frame Counts

| Seed | Checkpoint | `n_onball` | P(on-ball) | `n_selected_ps` | P(PASS+SHOT | on-ball) |
|------|------------|------------|------------|-----------------|------------------------|
| 42 | `seed42_actorreweight_49920.pt` | **13** | 0.170% | 8 | 61.54% |
| 123 | `seed123_actorreweight_49920.pt` | **96** | 1.255% | 79 | 82.23% |
| 7 | `seed7_actorreweight_49920.pt` | **18** | 0.235% | 17 | 94.44% |
| 999 | `seed999_actorreweight_49920.pt` | **42** | 0.549% | 30 | 71.43% |

**Note:** `n_onball` counts **all-agent** on-ball frames across the full 7650-decision scope (50 episodes × 51 ticks × 3 agents = 7,650 decisions). The per-agent breakdown is:

| Seed | Agent 0 | Agent 1 | Agent 2 | Total |
|------|---------|---------|---------|-------|
| 42 | 13 | 0 | 0 | 13 |
| 123 | 29 | 25 | 42 | 96 |
| 7 | 11 | 5 | 2 | 18 |
| 999 | 9 | 11 | 22 | 42 |

### 3.1.1 Small-n Qualification

The on-ball samples in Section 3.1 are sparse. A single additional selected PASS+SHOT frame shifts the conditional rate by `1 / n_onball`, expressed in percentage points:

| Seed | `n_onball` | `n_selected_ps` | 1-frame shift (pp) |
|------|------------|-----------------|--------------------|
| 42 | 13 | 8 | **7.69 pp** |
| 123 | 96 | 79 | 1.04 pp |
| 7 | 18 | 17 | **5.56 pp** |
| 999 | 42 | 30 | 2.38 pp |

**Seed 42 (n=13) and seed 7 (n=18) are the smallest and most fragile samples in this table.** Their conditional rates — 61.54% and 94.44% respectively — should be read as descriptive observations from a sparse sample, not as stable seed-level properties. A single frame moving from non-selected to selected changes seed 42's conditional rate by 7.69 percentage points and seed 7's by 5.56 percentage points.

This qualification applies to every percentage in the canonical occupancy table above. None of these numbers should be presented as precise or low-variance elsewhere in the repo.

---

### 3.2 Canonical Scope Reconciliation Table

The table below compares the corrected all-three-agent measurement (7650-decision scope: 50 episodes × 51 ticks × 3 agents) against the rebuilt `actor_reweight_retest_eval_summary.csv` rates.

**Tolerance:** ±0.05 percentage points. This tolerance is pre-stated and accounts for minor floating-point and eval-script variance between the two measurement paths. It is not tight enough to force a match label; all four seeds reconcile within a much tighter margin (≤0.005 pp).

| Seed | `n_decisions` | `n_pass_shot` | `canonical_rate_pct` | `rebuilt_rate_pct` | `delta_pp` | `match` |
|------|---------------|---------------|----------------------|--------------------|------------|---------|
| 42 | 7650 | 57 | 0.7451% | 0.75% | −0.0049 pp | ✅ |
| 123 | 7650 | 128 | 1.6732% | 1.67% | +0.0032 pp | ✅ |
| 7 | 7650 | 66 | 0.8627% | 0.86% | +0.0027 pp | ✅ |
| 999 | 7650 | 79 | 1.0327% | 1.03% | +0.0027 pp | ✅ |

**Verification:** `n_decisions = 7650` is confirmed for every seed (50 episodes × 51 ticks × 3 agents = 7,650 agent-decisions). The canonical rate is computed as `100 × n_pass_shot / n_decisions`. The rebuilt rate is taken from `actor_reweight_retest_eval_summary.csv` column `pass_shot_rate_pct` at the matching checkpoint step (50,000). All four seeds reconcile within the stated ±0.05 pp tolerance.

---

### 3.3 π-Floor Checks (All Seeds Pass)

| Seed | `n_onball` | `n_selected_ps` | `mean_pi_ps` | Floor (action space) | Floor (observed masks) | Pass |
|------|------------|-----------------|--------------|----------------------|------------------------|------|
| 42 | 13 | 8 | 0.790306 | 0.032389 | 0.034188 | ✅ |
| 123 | 96 | 79 | 0.310327 | 0.043311 | 0.045718 | ✅ |
| 7 | 18 | 17 | 0.336986 | 0.049708 | 0.052469 | ✅ |
| 999 | 42 | 30 | 0.404780 | 0.037594 | 0.039683 | ✅ |

All π-floor inequalities hold:
- Inequality 1: `n_selected × mean_pi_selected / n_onball ≤ mean_pi_ps`
- Inequality 2: `floor_from_observed_masks ≤ mean_pi_ps`
- Inequality 3: `mean_pi_selected ≤ mean_pi_ps_given_selected`

---

## 4. Measurement Artifact Provenance

### 4.1 Canonical Artifacts (base_seed=500000)

| Artifact | Path | Description |
|----------|------|-------------|
| Detail JSON | `training/results/post_reweight_logit_prestep_reconciled_detail.json` | Raw per-agent decision frames (30,600 frames) |
| Summary CSV | `training/results/post_reweight_logit_prestep_reconciled_summary.csv` | Per-seed aggregates |
| Reconciliation CSV | `training/results/post_reweight_canonical_scope_reconciliation.csv` | 7650-decision scope reconciliation |

### 4.2 Verification

Two independent verification scripts validate the canonical artifacts:

1. **`training/verify_prestep_measurement.py`** — Recomputes all aggregates from raw frame records using an independent softmax implementation. **PASSED** (exit 0).

2. **`training/verify_canonical_artifacts.py`** — Validates JSON structure, frame counts, checkpoint SHAs, temporal alignment, π-floor, and canonical rate consistency. **PASSED** (exit 0).

3. **`training/generate_final_report.py`** — Generates human-readable gate report. **All gates passed.**

### 4.3 Checkpoint Inventory

All measurements use the **fresh-training 50k checkpoints** verified against `training/results/retest_checkpoint_inventory.csv`:

| Seed | Checkpoint Path | SHA-256 |
|------|-----------------|---------|
| 42 | `training/models/mappo_academy_3_vs_1_with_keeper_onball_seed42_actorreweight.pt` | `eb9e1403b6a65c3c288397b752c64319cf8f9022116492fa05a355c08682fd24` |
| 123 | `training/models/mappo_academy_3_vs_1_with_keeper_onball_seed123_actorreweight.pt` | `72e56e385a031953f5ace01f2e56ff658c9730b7cc5c5e1ad88fd9c637acf4be` |
| 7 | `training/models/mappo_academy_3_vs_1_with_keeper_onball_seed7_actorreweight.pt` | `7b29a3a465a3e1044a4908b205ce5fae2aa589b9d39ba258047314335d8b8524` |
| 999 | `training/models/mappo_academy_3_vs_1_with_keeper_onball_seed999_actorreweight.pt` | `0365a1d95f9a31f9517bdd42401aa3f4a4bfed3e713be610ae91bcf95d526097` |

---

## 5. Historical vs Canonical Comparison

| Seed | Historical `n_onball` (post-step, base_seed=700000) | Canonical `n_onball` (pre-step, base_seed=500000) | Change |
|------|---------------------------------------------------|--------------------------------------------------|--------|
| 42 | 17 | 13 | -4 (-23.5%) |
| 123 | 27 | 96 | +69 (+255.6%) |
| 7 | 23 | 18 | -5 (-21.7%) |
| 999 | 11 | 42 | +31 (+281.8%) |

The cross-seed ranking also shifts substantially. Under the historical measurement:
- Seed 42 had the **lowest** on-ball occupancy (17 frames)
- Seed 999 had the **second lowest** (11 frames)

Under the canonical measurement:
- Seed 42 still has the **lowest** on-ball occupancy (13 frames)
- Seed 7 has the **second lowest** (18 frames)
- Seed 999 jumps to **third** (42 frames)

This ranking shift is an arithmetic consequence of both the method fix and the base seed change. The historical values are **not comparable** to the canonical values and should not be mixed in analysis.

---

## 6. Conclusions

1. **Canonical base_seed = 500,000.** This is locked in `CANONICAL_METRICS_CONTRACT.md` and `training/eval_actor_reweight.py`. The script default in `training/eval_canonical_three_agent_measurement.py` has been corrected to match.

2. **Seed 42 `n_onball` swing 17 → 74 → 13** is fully explained by two orthogonal changes:
   - **17 → 74:** Measurement method fix from post-step ball ownership to pre-step team possession (commit `085ec85`).
   - **74 → 13:** Base seed change from 700,000 to 500,000 (canonical).

3. **Canonical `n_onball` values under base_seed=500000:**
   - Seed 42: **13**
   - Seed 123: **96**
   - Seed 7: **18**
   - Seed 999: **42**

4. **Canonical scope reconciliation:** All four seeds reconcile within ±0.05 pp of the rebuilt `actor_reweight_retest_eval_summary.csv` rates under the confirmed-correct base_seed=500000. No seed exceeds the tolerance.

5. **No training, reward, GAE, mask, or network changes** were made. This is a measurement-only reconciliation.

---

## 7. Superseded Prior Reports

The following reports and artifacts from this investigation thread are superseded by this document. They should not be cited as authoritative; this document is the canonical reference for occupancy, π-floor, and scope-reconciliation numbers under base_seed=500000.

| Report / Artifact | Reason Superseded |
|-------------------|-------------------|
| `5908073` — `ONBALL_OCCUPANCY_CHECK.md` and `onball_occupancy_summary.csv` | Used defective post-step ball-ownership measurement (`_last_ball_owner_agent_idx == 0`) and wrong base_seed=700000. Both the method and the seed are incorrect under the canonical contract. |
| `085ec85` — `ONBALL_OCCUPANCY_PRESTEP_CHECK.md` and `onball_occupancy_prestep_summary.csv` | Corrected the measurement method but still used base_seed=700000, which conflicts with `CANONICAL_METRICS_CONTRACT.md` §2.1 (base_seed=500000). |
| `085ec85` — `post_reweight_logit_prestep_summary.csv` / `post_reweight_logit_prestep_detail.json` | Corrected method but wrong base_seed=700000. The `n_onball` values (42=74, 123=121, 7=91, 999=92) are not comparable to the canonical values. |
| "Final Measurement Gate" report (`generate_final_report.py` output) | Used base_seed=500000 but was generated from artifacts that mixed wrong-base_seed data and did not include the canonical scope reconciliation table with explicit tolerance. This document replaces it. |

---

## 8. Allowed Conclusions

From this reconciliation, the only permitted conclusions are:

- The canonical base_seed is 500,000.
- The canonical on-ball measurement uses pre-step team possession (`obs[95] == 1.0`).
- The seed 42 `n_onball` swing is explained by the measurement-method fix and base-seed change.
- The canonical `n_onball` values are as listed in Section 3.1.
- The canonical scope reconciliation table (Section 3.2) shows all four seeds match the rebuilt rates within ±0.05 pp.

No interpretation of team dynamics, occupancy sufficiency, or policy behavior is authorized at this stage. D-Obs results are required before any such conclusions may be drawn.
