# POST_REWEIGHT_LOGIT_SNAPSHOT — Verified Retest Checkpoint Actor Measurement

**Date:** 2026-09-20  
**HEAD:** `154ace9`  
**Scenario:** `academy_3_vs_1_with_keeper_onball`  
**Eval protocol:**
- Deterministic rollouts, `base_seed=700000`, 50 episodes per checkpoint
- Frame filter: on-ball frames where agent 0 has the ball (`obs[95] == 1.0`)
- Legal PASS/SHOT mask reported per frame; no frame filtering on mask (all on-ball frames included)

**Source of truth (behavioral rates):** `training/results/actor_reweight_retest_eval_summary.csv`  
**Checkpoint inventory:** `training/results/retest_checkpoint_inventory.csv`  
**Summary CSV:** `training/results/post_reweight_logit_summary.csv`  
**Detail JSON:** `training/results/post_reweight_logit_detail.json` (local/LFS if large)

**Frequency-reweight status:** Closed as a sufficient policy fix under current pre-registered bars (0/4 primary, 1/4 secondary, guard fail on seed 123). This snapshot does not authorize a new M experiment.

---

## Task 1 — Checkpoint Selection

| Seed | Phase 1 end (15k) | SHA256 (15k) | Phase 2 final (50k) | SHA256 (final) |
|------|-------------------|--------------|---------------------|----------------|
| 42 | `mappo_academy_3_vs_1_with_keeper_onball_seed42_actorreweight_15000.pt` | `7f005b376d62cf39...` | `mappo_academy_3_vs_1_with_keeper_onball_seed42_actorreweight.pt` | `eb9e1403b6a65c3c...` |
| 123 | `mappo_academy_3_vs_1_with_keeper_onball_seed123_actorreweight_15000.pt` | `3e1d781088f49be8...` | `mappo_academy_3_vs_1_with_keeper_onball_seed123_actorreweight.pt` | `72e56e385a031953...` |
| 7 | `mappo_academy_3_vs_1_with_keeper_onball_seed7_actorreweight_15000.pt` | `3680ded7f2044633...` | `mappo_academy_3_vs_1_with_keeper_onball_seed7_actorreweight.pt` | `7b29a3a465a3e104...` |
| 999 | `mappo_academy_3_vs_1_with_keeper_onball_seed999_actorreweight_15000.pt` | `c01e0eb2b0503f87...` | `mappo_academy_3_vs_1_with_keeper_onball_seed999_actorreweight.pt` | `0365a1d95f9a31f9...` |

All 8 checkpoints verified against `retest_checkpoint_inventory.csv`. No 880091d hashes used.

---

## Task 2 — On-Ball Actor Snapshot

### 2a. Rebuilt Rate Context (from `actor_reweight_retest_eval_summary.csv`)

| Seed | 15k PASS+SHOT% | 50k PASS+SHOT% |
|------|----------------|----------------|
| 42 | 0.75% | 0.75% |
| 123 | 1.52% | 1.67% |
| 7 | 0.82% | 0.86% |
| 999 | 0.75% | 1.03% |

### 2b. On-Ball Snapshot @ 15k (Phase 1, M=500 on)

| Seed | n_frames | π(MOVE) | π(PASS) | π(SHOT) | H(π) | Δ_PASS-MOVE | Δ_SHOT-MOVE | P(PASS legal) | P(SHOT legal) |
|------|----------|---------|---------|---------|------|-------------|-------------|---------------|---------------|
| 42 | 21 | 0.272 | 0.538 | 0.0169 | 1.710 | 1.873 | -0.995 | 0.905 | 0.905 |
| 123 | 41 | 0.529 | 0.157 | 0.0114 | 2.486 | 1.127 | 0.024 | 0.268 | 0.268 |
| 7 | 17 | 0.380 | 0.365 | 0.0437 | 2.421 | 1.538 | 0.195 | 0.647 | 0.647 |
| 999 | 8 | 0.201 | 0.637 | 0.0301 | 1.524 | 2.893 | 0.287 | 0.875 | 0.875 |

### 2c. On-Ball Snapshot @ 50k (Phase 2, M=1.0 off)

| Seed | n_frames | π(MOVE) | π(PASS) | π(SHOT) | H(π) | Δ_PASS-MOVE | Δ_SHOT-MOVE | P(PASS legal) | P(SHOT legal) |
|------|----------|---------|---------|---------|------|-------------|-------------|---------------|---------------|
| 42 | 17 | 0.254 | 0.550 | 0.0167 | 1.703 | 2.066 | -0.812 | 0.882 | 0.882 |
| 123 | 27 | 0.409 | 0.344 | 0.0328 | 2.403 | 1.303 | -0.340 | 0.778 | 0.778 |
| 7 | 23 | 0.422 | 0.298 | 0.0456 | 2.516 | 0.856 | -0.071 | 0.652 | 0.652 |
| 999 | 11 | 0.323 | 0.450 | 0.0488 | 2.050 | 1.467 | -0.286 | 0.818 | 0.818 |

---

## Task 3 — Contrasts

### 3a. Phase 1 → Phase 2 Within-Seed Change

| Seed | Δ π(PASS) | Δ π(SHOT) | Δ H(π) | Δ Δ_PASS-MOVE | Δ Δ_SHOT-MOVE | Direction |
|------|-----------|-----------|--------|---------------|---------------|-----------|
| 42 | +0.012 | -0.0002 | -0.007 | +0.193 | +0.183 | PASS stable, SHOT flat, entropy ↓, deltas ↑ |
| 123 | +0.187 | +0.0214 | -0.083 | +0.176 | -0.364 | PASS ↑, SHOT ↑, entropy ↓, Δ_PASS-MOVE ↑, Δ_SHOT-MOVE ↓ |
| 7 | -0.067 | +0.0019 | +0.095 | -0.681 | -0.266 | PASS ↓, SHOT flat, entropy ↑, deltas ↓ |
| 999 | -0.187 | +0.0187 | +0.526 | -1.426 | -0.573 | PASS ↓, SHOT ↑, entropy ↑, deltas ↓ |

**Note:** Seed 123 is the only seed showing π(PASS) and π(SHOT) both increasing from Phase 1 to Phase 2. Seeds 7 and 999 show π(PASS) declining after M is removed.

### 3b. Cross-Seed at Final (50k), Ordered by Rebuilt PASS+SHOT%

| Rank | Seed | Rebuilt PASS+SHOT% | π(PASS) | π(SHOT) | H(π) | Δ_PASS-MOVE | Δ_SHOT-MOVE |
|------|------|-------------------|---------|---------|------|-------------|-------------|
| 1 | 123 | 1.67% | 0.344 | 0.0328 | 2.403 | 1.303 | -0.340 |
| 2 | 999 | 1.03% | 0.450 | 0.0488 | 2.050 | 1.467 | -0.286 |
| 3 | 7 | 0.86% | 0.298 | 0.0456 | 2.516 | 0.856 | -0.071 |
| 4 | 42 | 0.75% | 0.550 | 0.0167 | 1.703 | 2.066 | -0.812 |

**Observation:** Logits do **not** track behavioral rate order. Seed 42 has the highest π(PASS) (0.550) but the lowest behavioral rate (0.75%). Seed 999 has higher π(PASS) and π(SHOT) than seed 123 but lower behavioral rate (1.03% vs 1.67%). This suggests behavioral rates are not determined solely by on-ball logit magnitudes in this snapshot.

### 3c. Seed 123 vs 999 (No "Winner" Framing)

Both seeds are below primary targets (123: 1.67% < 3.46%; 999: 1.03% < 1.50%).

| Metric | Seed 123 @ 50k | Seed 999 @ 50k |
|--------|----------------|----------------|
| π(PASS) | 0.344 | 0.450 |
| π(SHOT) | 0.0328 | 0.0488 |
| H(π) | 2.403 | 2.050 |
| Δ_PASS-MOVE | 1.303 | 1.467 |
| Δ_SHOT-MOVE | -0.340 | -0.286 |
| n_frames | 27 | 11 |

Seed 999 shows higher π(PASS) and π(SHOT) but also higher entropy (2.050 vs 2.403? Actually 2.403 > 2.050, so seed 123 has higher entropy). Wait, let me correct: seed 123 entropy=2.403, seed 999 entropy=2.050. So seed 123 has higher entropy despite lower π(PASS)/π(SHOT). This could indicate more diffuse policy mass.

### 3d. Mask On-Ball

| Seed | 15k P(PASS legal) | 15k P(SHOT legal) | 50k P(PASS legal) | 50k P(SHOT legal) | Mask defect? |
|------|-------------------|-------------------|-------------------|-------------------|--------------|
| 42 | 0.905 | 0.905 | 0.882 | 0.882 | Minor (~10% of on-ball frames) |
| 123 | 0.268 | 0.268 | 0.778 | 0.778 | **Yes** at 15k (73% of on-ball frames missing PASS/SHOT legal) |
| 7 | 0.647 | 0.647 | 0.652 | 0.652 | **Yes** (~35% of on-ball frames missing PASS/SHOT legal) |
| 999 | 0.875 | 0.875 | 0.818 | 0.818 | Minor (~18–20% of on-ball frames) |

**Flag:** Seeds 123 and 7 show mask defects where PASS/SHOT are not legal at a substantial fraction of on-ball frames. This limits the interpretability of π(PASS)/π(SHOT) at those checkpoints because the policy cannot select those actions when they are masked out.

---

## Task 4 — Interpretation

The on-ball logit snapshot shows heterogeneous Phase 1→Phase 2 dynamics:

1. **Seed 123** is the only seed with both π(PASS) and π(SHOT) increasing after M is removed (+0.187 and +0.021), yet its rebuilt behavioral rate (1.67%) remains below its canonical baseline (1.96%). The mask defect at 15k (27% PASS/SHOT legal) complicates the Phase 1 baseline.

2. **Seed 999** shows the highest Phase 1 π(PASS) (0.637) and a strong Δ_PASS-MOVE (2.893), but π(PASS) collapses by -0.187 after M is removed, ending at 0.450 with behavioral rate 1.03%. The "999 winner" framing from the stale table is void.

3. **Seeds 42, 7, 999** all show π(PASS) declining or flat after M is removed, with entropy increasing (seeds 7, 999) or stable (seed 42). This is inconsistent with a simple "consolidation" narrative where logits should remain stable while action selection drops.

4. **Cross-seed logits do not predict behavioral rates.** Seed 42 has the highest π(PASS) (0.550) but the lowest behavioral rate (0.75%). This suggests that on-ball logit magnitudes alone are insufficient to explain canonical eval outcomes; shot/tackle/dribble selection, mask dynamics, and opponent-zone geometry also matter.

5. **Mask defects** (seeds 123 at 15k, seed 7 throughout) mean that PASS/SHOT were not available as actions on a sizable fraction of on-ball frames, which depresses both π(PASS)/π(SHOT) and behavioral rates.

**Bottom line:** The share-log mechanism moved aggregate actor-loss share as intended (26–44% vs 0.3–0.6%), but the on-ball policy snapshot shows no consistent translation into sustained π(PASS)/π(SHOT) after M is removed. The heterogeneity across seeds (123 up, 7/999 down, 42 flat) points to basin-dependent dynamics rather than a uniform consolidation failure. Frequency reweight remains not a sufficient policy fix under current bars.

---

## Non-Claims

- Policy bars still failed 0/4 primary; this snapshot does not change that verdict.
- No new training is authorized by this measurement.
- The "999 held under reweight" narrative is void; seed 999 π(PASS) declined from 0.637 to 0.450 after M was removed.
- No anneal-M, entropy, mix-script, or reward changes are recommended here.

---

## Confirmation Checklist

- [x] All snapshots tied to inventory SHA256s
- [x] Behavioral rates cited only from rebuilt CSV / RETEST_REBUILT_GATES
- [x] No stale 2.34% / 880091d trajectories cited
- [x] No training performed
- [x] No reward/GAE/mask/network/entropy changes

---

## Files

- `training/results/POST_REWEIGHT_LOGIT_SNAPSHOT.md` — this report
- `training/results/post_reweight_logit_summary.csv` — aggregate snapshot
- `training/results/post_reweight_logit_detail.json` — per-frame detail (optional, local/LFS)
- `training/eval_post_reweight_logits.py` — eval-only script
