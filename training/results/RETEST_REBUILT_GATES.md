# RETEST_REBUILT_GATES — Corrected-Gate Retest Official Scorecard

**Date:** 2026-09-20  
**HEAD:** `a8d57ae`  
**Source of truth (behavioral):** `training/results/actor_reweight_retest_eval_summary.csv`  
**Inventory:** `training/results/retest_checkpoint_inventory.csv`  
**Share logs:** `training/logs/actor_loss_reweight_share_seed{42,123,7,999}_academy_3_vs_1_with_keeper_onball.csv`

---

## Void Notice — Stale Table Supersession

The original `training/results/ACTOR_LOSS_REWEIGHT_RETEST.md` (pre-`a8d57ae`) contained trajectory and Phase 2 tables that were **digit-identical** to the 880091d buggy-gate run. **Those tables are void.**

- Stale CSV: `training/results/actor_reweight_eval_summary.csv` (mtime `2026-09-20 12:40:47 PM`, predates retest training 14:44–15:00)
- Stale claims voided:
  - Seed 999 final PASS+SHOT% **2.34%** → rebuilt **1.03%**
  - "1/4 primary" → rebuilt **0/4 primary**
  - "Secondary PASS (2/4)" → rebuilt **1/4**
  - Digit-identical 10-point trajectory series across all seeds

**This document is the official scorecard.** All behavioral claims below derive exclusively from the 40-row rebuilt CSV.

---

## Task 1 — CSV Integrity

| Check | Result |
|-------|--------|
| Total rows | **40** |
| Seeds | **42, 123, 7, 999** |
| Steps per seed | **10** (5k, 10k, 15k, 20k, 25k, 30k, 35k, 40k, 45k, 50k) |
| Seed 999 final `pass_shot_rate_pct` | **1.03** (matches A4 standalone eval) |
| Final checkpoint basename (seed 999) | `mappo_academy_3_vs_1_with_keeper_onball_seed999_actorreweight.pt` |
| Final checkpoint SHA256 | `0365a1d95f9a31f9517bdd42401aa3f4a4bfed3e713be610ae91bcf95d526097` |

**CSV integrity: PASS**

---

## Task 2 — Official Phase 1 / Phase 2 Tables

### 2a. Trajectory Table (PASS+SHOT%, from rebuilt CSV)

| Seed | 5k | 10k | 15k (Phase 1 end) | 20k | 25k | 30k | 35k | 40k | 45k | 50k (tail) |
|------|----|-----|-------------------|-----|-----|-----|-----|-----|-----|------------|
| 42 | 0.75 | 0.75 | 0.75 | 0.75 | 0.78 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 123 | 1.02 | 0.78 | 1.52 | 1.28 | 1.42 | 1.76 | 1.82 | 1.42 | 1.65 | 1.67 |
| 7 | 0.80 | 0.76 | 0.82 | 1.05 | 0.80 | 0.90 | 0.85 | 0.99 | 0.86 | 0.86 |
| 999 | 0.75 | 0.73 | 0.75 | 0.73 | 0.73 | 1.02 | 1.03 | 1.03 | 1.03 | 1.03 |

### 2b. Phase 1 End (step ≤ 15000 closest to 15k)

| Seed | Step | PASS+SHOT% | Passes/Ep | Shots/Ep |
|------|------|------------|-----------|----------|
| 42 | 15000 | 0.75 | 1.14 | 0.00 |
| 123 | 15000 | 1.52 | 2.32 | 0.00 |
| 7 | 15000 | 0.82 | 1.26 | 0.00 |
| 999 | 15000 | 0.75 | 1.14 | 0.00 |

### 2c. Phase 2 Tail (max step = 50000)

| Seed | Step | PASS+SHOT% | Passes/Ep | Shots/Ep | Mean Reward | Checkpoint Basename |
|------|------|------------|-----------|----------|-------------|---------------------|
| 42 | 50000 | 0.75 | 1.14 | 0.00 | -0.7690 | `mappo_academy_3_vs_1_with_keeper_onball_seed42_actorreweight.pt` |
| 123 | 50000 | 1.67 | 2.54 | 0.02 | -0.7348 | `mappo_academy_3_vs_1_with_keeper_onball_seed123_actorreweight.pt` |
| 7 | 50000 | 0.86 | 1.32 | 0.00 | -0.7542 | `mappo_academy_3_vs_1_with_keeper_onball_seed7_actorreweight.pt` |
| 999 | 50000 | 1.03 | 1.58 | 0.00 | -0.7459 | `mappo_academy_3_vs_1_with_keeper_onball_seed999_actorreweight.pt` |

---

## Task 3 — Gate Scorecard

### Protocol Constants

| Seed | Canonical Baseline PASS+SHOT% | Primary Target |
|------|-------------------------------|----------------|
| 42 | 0.00 | 1.50 |
| 123 | 1.96 | 3.46 |
| 7 | 0.00 | 1.50 |
| 999 | 0.00 | 1.50 |

### Gate Evaluation

| Seed | Tail Rate | Baseline | Target | Primary | Passes/Ep > 0 | Shots/Ep > 0 | Secondary | Below Baseline? | Guard |
|------|-----------|----------|--------|---------|---------------|--------------|-----------|-----------------|-------|
| 42 | 0.75% | 0.00% | 1.50% | **NO** | Yes | No | **NO** | No | NO |
| 123 | 1.67% | 1.96% | 3.46% | **NO** | Yes | Yes | **YES** | Yes | **FAIL** |
| 7 | 0.86% | 0.00% | 1.50% | **NO** | Yes | No | **NO** | No | NO |
| 999 | 1.03% | 0.00% | 1.50% | **NO** | Yes | No | **NO** | No | NO |

### Aggregate Gates

- **Primary (≥3/4 seeds exceed target): FAIL** (0/4)
- **Secondary (≥2/4 seeds with pass/ep > 0 AND shot/ep > 0): FAIL** (1/4 — seed 123 only)
- **Guard (no seed below baseline): FAIL** (violator: seed 123, 1.67% < 1.96%)

---

## Task 3 — Programme Verdict

### Claim Assessment

| Claim | Conclusion from rebuilt evidence |
|-------|----------------------------------|
| Corrected gate implements rare-action M | **Supported** — code on `main` uses `selected ∈ {9..12} ∧ selected_legal`; old `pass_shot_mask.any()` removed from optimizer path |
| Phase 1 PASS+SHOT **share** raised into multi-percent band | **Supported** — share logs show 26–44% under M=500 vs 0.3–0.6% under M=1 (~76–82×) |
| Unscripted multi-seed π meets primary bar after M off | **Not supported** — 0/4 seeds exceed target at tail |
| Stale "999 unique pass at 2.34%" | **Void** — rebuilt value is 1.03%; stale table was copy-paste from pre-retest CSV |
| "Starvation-relief sufficient for persistent policy" | **Not supported** — share shift did not translate into PASS+SHOT rate increases in canonical eval |
| "Consolidation residual" as explanation of 1/4 near-miss | **Not justified** — rebuilt data shows 0/4 primary, not 1/4; "consolidation residual" was narrative fitted to stale table |

### Required One-Line Verdict

> Corrected-gate M=500 moves aggregate PASS/SHOT actor-loss share as intended; under rebuilt canonical eval, unscripted PASS+SHOT at 50k remains below primary targets for **all four** seeds (0/4); secondary and guard also fail. Frequency reweight is **not** a sufficient policy fix under these bars.

---

## Task 4 — Doc Hygiene

### RETEST_REBUILT_GATES.md

This document is the **official scorecard**. It supersedes all behavioral claims in `ACTOR_LOSS_REWEIGHT_RETEST.md` that relied on the stale `actor_reweight_eval_summary.csv`.

### ACTOR_LOSS_REWEIGHT_RETEST.md Status

That file now carries a **CORRECTION** banner and points readers here for the verified scorecard. Do not cite its original Task 5 verdict ("starvation-relief real; consolidation residual") as accepted science.

---

## Task 5 — Next-Step Recommendation

**Close frequency-reweight as a policy solution under current pre-registered bars.**

- The share-log mechanism is real and validated (26–44% vs 0.3–0.6%).
- The behavioral outcome under canonical eval does not meet primary/secondary/guard criteria (0/4 primary, 1/4 secondary, guard fail on 123).
- Do not run another fixed-M experiment without a new, written persistence mechanism.
- Optional later work: measurement-only logit comparison on verified retest checkpoints using rebuilt rates (not "999 winner" framing). Do not open a basin-forensics training campaign until this scorecard is committed and the verdict is agreed.

---

## Appendix

### A.1 Phase 1 Surrogate-Loss Share Summary (from share logs)

| Seed | Mean Share M=500 | Mean Share M=1 | Ratio | Max Share M=500 | Max Share M=1 |
|------|-----------------|---------------|-------|----------------|--------------|
| 42 | 0.4117 (41.2%) | 0.0053 (0.5%) | 77.5× | 0.7902 (79.0%) | 0.0144 (1.4%) |
| 123 | 0.2828 (28.3%) | 0.0036 (0.4%) | 78.8× | 0.7453 (74.5%) | 0.0171 (1.7%) |
| 7 | 0.4362 (43.6%) | 0.0057 (0.6%) | 76.3× | 0.8619 (86.2%) | 0.0181 (1.8%) |
| 999 | 0.2605 (26.1%) | 0.0032 (0.3%) | 81.7× | 0.7311 (73.1%) | 0.0181 (1.8%) |

### A.2 Rebuilt Summary CSV

`training/results/retest_rebuilt_gates_summary.csv` (optional one-row-per-seed summary) is not required for the official scorecard but may be generated for convenience.

---

## Confirmation Checklist

- [x] Behavioral numbers derived exclusively from 40-row rebuilt CSV
- [x] Stale 2.34% / digit-identical trajectory explicitly voided
- [x] Primary/secondary/guard recomputed from file
- [x] Share success separated from policy failure
- [x] No new training, no M change, no reward/GAE/mask edits
- [x] 880091d findings not overwritten
