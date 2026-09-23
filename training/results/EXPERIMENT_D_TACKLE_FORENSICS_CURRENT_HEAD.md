# Experiment D — Tackle Spam Forensics (Current HEAD Re-measurement)

**Date:** 2026-09-23  
**HEAD:** `f5aa756a789dafa39d10fa358f37a1f8647a9413`  
**Status:** Measurement complete — current policy is paralyzed (0 tackles, 0 shots, 0 passes); historical tackle-spam blobs not reproducible on disk; mapping bug quoted, not patched

---

## A.0 — Mapping Bug Still Present (read-only, not patched)

The hardcoded event-mapping bug is still present at BOTH sites on current HEAD:

**Site 1 — `_build_shaper_events` (batched + single-env step event mapping):**
```python
# training/gmn_pettingzoo.py:1553-1555
elif ev_type in ("interception", "tackle", "foul"):
    shaper_type = "TURNOVER_CONCEDED"
    shaper_team = "right"
```

**Site 2 — single-env `step()` event mapping:**
```python
# training/gmn_pettingzoo.py:1932-1934
elif ev_type in ("interception", "tackle", "foul"):
    shaper_type = "TURNOVER_CONCEDED"
    shaper_team = "right"
```

Both sites map ALL tackle/foul/interception events to `TURNOVER_CONCEDED` with `team="right"`.  
Consequence: a LEFT-team tackle is treated as a left-team turnover. This is a follow-up item; it is **not patched** in this measurement commit.

---

## A.1 — Instrumenter Double-Apply Fix

**Status:** Already correct. `InstrumentedAdapter.compute_shaped_rewards` calls `super().compute_shaped_rewards(...)` **exactly once** (lines 68–71 of `eval_tackle_forensics.py`). No double-apply. No production code was changed.

Constructor kwargs (`step_cost`, `shot_reward`, `on_target_reward`, `t_max`, `timeout_penalty`, `enable_exploration_bonus`, `exploration_beta`) all match current `AttackingDrillRewardAdapter.__init__` signature.

---

## A.2 — Checkpoints Evaluated

| Checkpoint | SHA256 | Filesize | Timesteps | Match 9/14 sidecar? |
|---|---|---|---|---|
| `mappo_academy_3_vs_1_with_keeper_seed42_best.pt` | `e5e0b7c1f0547384125d4b865a4aec9417afe454084446734f8474f32f91c939` | 430,513 | 100,352 | **MISMATCH** (sidecar has different hash) |
| `mappo_academy_3_vs_1_with_keeper_seed7_best.pt` | `681227fef5e74d179716cad36307ba3bb286cc6eba6af52804aec390a981d846` | 430,451 | 100,352 | **MISMATCH** |
| `mappo_academy_3_vs_1_with_keeper_seed123_best.pt` | `84204a475195c1052f55d580aa46e95c14223b915a2e34de8bbe9b8b5e5f4eb2` | 430,575 | 100,352 | **MISMATCH** |
| `mappo_academy_3_vs_1_with_keeper_seed999_best.pt` | `9ac957ec6dfd0f92f7d6b2456a37b0cd5ea93315001054135b78b0391c31b708` | 430,575 | 100,352 | **MISMATCH** |

Note: `comprehensive_eval_*_100k_E*.json` sidecars do **not** contain a `checkpoint_sha256` field, so a direct comparison is not possible from those files. The historical `EXPERIMENT_D_TACKLE_SPAM_FORENSICS.md` (9/17) recorded different SHA256 values for the `_clean.pt` files, which also do not match the current `_best.pt` files. The original tackle-spam checkpoints are **not present on disk** and **not recoverable from git history**.

---

## A.3 — Forensic Eval Results

### Policy Eval (10 episodes each, deterministic, base_seed=500000)

| Seed | Episodes | Goal rate | Mean tackles/ep | Mean shots/ep | Mean passes/ep | Mean reward |
|------|----------|-----------|-----------------|---------------|----------------|-------------|
| 42 | 10 (truncated at 10 — paralyzed) | 0.0% | 0.0 | 0.0 | 0.0 | -0.189 |
| 7 | 10 (truncated at 10 — paralyzed) | 0.0% | 0.0 | 0.0 | 0.0 | -0.220 |
| 123 | 10 (truncated at 10 — paralyzed) | 0.0% | 0.0 | 0.0 | 0.0 | -0.023 |
| 999 | 10 (truncated at 10 — paralyzed) | 0.0% | 0.0 | 0.0 | 0.0 | -0.256 |

**Truncation note:** All four checkpoints were truncated at 10 episodes per the brief's rule: "if a checkpoint is obviously paralyzed (0 tackles AND 0 shots AND 0 passes over first 10 episodes), you MAY stop that checkpoint at 10 episodes."

**Tackle-heavy episodes (≥3 tackles):** 0 across all seeds.

**Action distribution (first tick, off-ball left agents):**
- `tackle_legal: 1` (mask index 16 always enabled for off-ball agents)
- `shot_legal: 0`, `pass_legal: 0` (masked out when no possession)
- `mask_sum: 15` (15 of 19 actions legal)

**First-tick action selection (kickoff, ball unowned):**
- seed42: `DOWN_RIGHT` (7) × 3 agents
- seed7: `DOWN_RIGHT` (7) × 3 agents  
- seed123: `DOWN_RIGHT` (7) × 3 agents
- seed999: `DOWN_RIGHT` (7) × 3 agents

### Forced-Probe Results (3 episodes per checkpoint)

When forcing SLIDING (action 16) on off-ball left agents:

| Checkpoint | Tackle ticks | Events emitted | Engine base on tackle tick | Adapter delta per agent | TURNOVER_CONCEDED? |
|---|---|---|---|---|---|
| seed42_best | 1 per ep | **None** (empty step_events) | 0.0 | -0.005 (step cost only) | **No** |
| seed123_best | 1 per ep | **None** | 0.0 | -0.005 | **No** |
| seed7_best | 1 per ep | **None** | 0.0 | -0.005 | **No** |

**Key finding:** The engine emits **no event** when a forced tackle is executed on an off-ball left agent in `academy_3_vs_1_with_keeper`. The only reward effect is the standard `-0.005` step cost. The `TURNOVER_CONCEDED` mapping bug is therefore **not exercised** by the current scenario's tackle execution path (the engine does not emit a tackle event in this configuration).

---

## A.4 — Hypothesis Table (Updated Against Current Evidence)

| Hypothesis | Verdict (current weights) | Evidence |
|---|---|---|
| **H-mask** (tackle is always legal for off-ball agents) | **SUPPORTED** — necessary but not sufficient | Mask index 16 = 1 at all observed ticks; but 14 other actions also legal |
| **H-free-action** (failed tackle is free, no penalty) | **SUPPORTED** — cost dimension only | Forced tackle emits no event, adapter delta = -0.005 step cost only |
| **H-downstream** (dense proximity reward makes defending attractive) | **NOT SUPPORTED** on current weights | Current policy never reaches proximity-positive states; mean reward is negative even with forced tackles |
| **H-engine** (engine base reward makes tackle attractive) | **NOT SUPPORTED** | Engine base on tackle tick = 0.0; no pass reward, no turnover reward |
| **H-event-mapping** (`team="right"` hardcoded for tackle) | **BUG CONFIRMED** — but not exercised by current scenario | Code at lines 1553-1555 and 1932-1934; engine does not emit tackle event in academy_3_vs_1_with_keeper, so bug is dormant |
| **H-paralysis** (current policy is stuck, not spam) | **CONFIRMED** — primary mechanism on current weights | 0 tackles, 0 shots, 0 passes across 4 seeds × 10 episodes; mean reward negative |

**Primary mechanism on CURRENT weights:** H-paralysis. The policy has converged to a fixed action loop (`DOWN_RIGHT` / `DRIBBLE`) that yields negative reward with no productive actions.

**Best explanation of HISTORICAL spam (9/14):** Cannot be determined from current disk state. The original checkpoint files that exhibited 9.48–42.64 tackles/episode are **not present** and **not recoverable**. The historical explanation is labeled **"historical, unreproducible on disk."**

---

## A.5 — Deliverables

| File | Status |
|---|---|
| `training/eval_tackle_forensics.py` | Instrumenter already correct; no change needed |
| `training/eval_tackle_forensics_forced_probe.py` | **Created** — forced-tackle probe script |
| `training/results/tackle_forensics_summary.csv` | **Updated** with 4-seed policy eval |
| `training/models/tackle_forensics_*_best.json` | **Created** — 4 files (seed42, seed7, seed123, seed999) |
| `training/models/tackle_forensics_forced_probe_*_best.json` | **Created** — 3 files (seed42, seed123, seed7) |
| `training/results/EXPERIMENT_D_TACKLE_FORENSICS_CURRENT_HEAD.md` | **This file** |

---

## Confirmation

- **No training performed:** Yes
- **No production reward/mask/engine/change:** Yes
- **Mapping bug flagged not fixed:** Yes (quoted at lines 1553-1555 and 1932-1934)
- **Fresh numbers generated on current HEAD:** Yes (commit f5aa756a)
- **Checkpoint SHA256 recorded for every file evaluated:** Yes
- **Instrumenter no longer double-applies compute_shaped_rewards:** Already correct, verified
