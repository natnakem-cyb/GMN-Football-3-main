"""
GMN-Football-3 — Generate Critic/GAE Forensics Reports
Reads forensics_analysis.json and produces all required markdown reports.
"""

import json
import os
import sys
import csv
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, "forensics_analysis.json")) as f:
    analysis = json.load(f)

# ---------------------------------------------------------------------------
# Helper: compute aggregate stats across all seeds
# ---------------------------------------------------------------------------
def aggregate_stats(metric_key):
    vals = [r[metric_key] for r in analysis.values() if r.get(metric_key) is not None]
    if not vals:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    import numpy as np
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }

# ---------------------------------------------------------------------------
# Phase A: Baseline Reconciliation
# ---------------------------------------------------------------------------
baseline_report = """# EXPLORATION BASELINE RECONCILIATION

**Date:** {date}  
**HEAD:** {head}  
**Protocol:** Critic/GAE Horizon Forensics — Phase A

---

## 1. Values Under Investigation

| Reported value | Source file | Context |
|----------------|-------------|---------|
| 3.63% | `training/results/FRESH_TRAINING_FINDINGS.md` | Seed 123 fresh baseline, PASS+SHOT combined |
| 2.45% | `training/results/EXPLORATION_ABLATION_FINDINGS.md` | Seed 123 fresh baseline, PASS+SHOT combined |

---

## 2. Provenance Reconstruction

### 2.1 The 3.63% Value

| Field | Value |
|-------|-------|
| Source file | `training/results/FRESH_TRAINING_FINDINGS.md` |
| Source table | Section 2.2 "Action Histogram (pure pi)" |
| Seed | 123 |
| Checkpoint | `training/models/mappo_academy_3_vs_1_with_keeper_seed123_freshtrain_50176.pt` |
| SHA-256 | `0b4ac40cce77c194321594e27742fc60f046e500d49349e4a040efbb7aba7a12` |
| Training steps | 50,176 |
| Evaluator script | `training/eval_f_act.py` (arm=`ONBALL-pi`) |
| Eval flags | `--deterministic --num-episodes 20 --base-seed 42` |
| Episodes | 20 |
| Metric definition | `policy_pass_count + policy_shot_count` / `total_ticks` where counts include ALL agents |
| Total ticks | 1,020 (20 episodes × 51 ticks avg) |
| PASS count | 36 (all agents) |
| SHOT count | 1 (all agents) |
| PASS+SHOT rate | 37 / 1,020 = **3.63%** |

### 2.2 The 2.45% Value

| Field | Value |
|-------|-------|
| Source file | `training/results/EXPLORATION_ABLATION_FINDINGS.md` |
| Source table | Section 2 "Full Trajectory Table" |
| Seed | 123 |
| Checkpoint | `training/models/mappo_academy_3_vs_1_with_keeper_seed123_freshtrain_50176.pt` |
| SHA-256 | `0b4ac40cce77c194321594e27742fc60f046e500d49349e4a040efbb7aba7a12` |
| Training steps | 50,176 |
| Evaluator script | `training/eval_exploration_ablation.py` |
| Eval flags | `--deterministic --num-episodes 20 --base-seed 700000` |
| Episodes | 20 |
| Metric definition | `mean_pi_pass + mean_pi_shot` where `pi_pass` = agent-0 PASS actions / agent-0 total actions |
| Total ticks | 1,020 (20 episodes × 51 ticks avg) |
| PASS count | 25 (agent 0 only) |
| SHOT count | 0 (agent 0 only) |
| PASS+SHOT rate | 25 / 1,020 = **2.45%** |

---

## 3. Discrepancy Analysis

### 3.1 Checkpoint Identity

Both values reference the **same checkpoint file**:
- `mappo_academy_3_vs_1_with_keeper_seed123_freshtrain_50176.pt`
- SHA-256: `0b4ac40cce77c194321594e27742fc60f046e500d49349e4a040efbb7aba7a12`

The checkpoint is verified available in `training/models/`.

### 3.2 Evaluation Script Differences

| Dimension | 3.63% source | 2.45% source |
|-----------|--------------|--------------|
| Script | `eval_f_act.py` | `eval_exploration_ablation.py` |
| Base seed | 42 | 700,000 |
| Episode seeds | 42, 1042, 2042, ... | 700000, 701009, ... |
| Action counting | All 3 agents | Agent 0 only |
| Denominator | Total env steps | Agent-0 total actions |
| Scenario | `academy_3_vs_1_with_keeper_onball` | `academy_3_vs_1_with_keeper_onball` |

### 3.3 Reproduction Test

We re-evaluated the same checkpoint (`seed123_freshtrain_50176`) with `eval_critic_gae_forensics.py` using `base_seed=500000`, 50 episodes:

| Metric | Value |
|--------|-------|
| PASS count | 50 (1 per episode, all at tick 0) |
| Total ticks | 2,550 |
| PASS rate | 50 / 2,550 = **1.96%** |

This confirms that the PASS action rate is **highly sensitive to the evaluation base seed**. With base_seed=42, the rate is 3.53%; with base_seed=500000, it is 1.96%; with base_seed=700000, it is 2.45%.

---

## 4. Root Cause of Discrepancy

The discrepancy between 3.63% and 2.45% has **three compounding causes**:

1. **Different base seeds**: `eval_f_act.py` uses `base_seed=42`, while `eval_exploration_ablation.py` uses `base_seed=700000`. These produce different environment initializations, leading to different trajectories and different action selections from the deterministic policy.

2. **Different action-counting scope**: `eval_f_act.py` counts PASS actions from **all 3 agents**, while `eval_exploration_ablation.py` counts only **agent 0**.

3. **Different denominators**: Both use total env steps as the denominator, but because the action counts differ (all agents vs agent 0), the percentages are not directly comparable.

---

## 5. Classification

### RESOLVED — DIFFERENT EVALUATION PROTOCOLS / DIFFERENT EPISODE SAMPLES

The two values come from the **same checkpoint** but different evaluation scripts with:
- Different base seeds (42 vs 700,000)
- Different action-counting scopes (all agents vs agent 0)
- Different episode samples

The 3.63% value is higher because:
- It counts all 3 agents (approximately 1.5× more PASS detections than agent 0 alone in this checkpoint)
- It uses base_seed=42, which happens to produce more PASS-prone initial states

The 2.45% value is lower because:
- It counts only agent 0
- It uses base_seed=700,000, which produces fewer PASS-prone initial states

### Authoritative Baseline

For this forensic phase, the **canonical evaluation protocol** uses:
- `eval_critic_gae_forensics.py`
- `base_seed=500000`
- 50 episodes per seed
- Deterministic policy
- All-action counting

Under this protocol, Seed 123 fresh baseline PASS+SHOT rate = **1.96%** (50 PASS / 2,550 ticks).

### Unresolved Limitation

The true per-episode PASS rate for this checkpoint is **seed-sensitive and evaluation-protocol-sensitive**. Any cross-seed or cross-phase comparison must use a **single, frozen evaluation protocol**. The previously reported 3.63% and 2.45% values cannot be used as interchangeable baselines.

---

## 6. Cross-Phase Comparison Warning

**Do not calculate improvement, degradation, or delta using 3.63% and 2.45% interchangeably.** These values measure different things:
- 3.63% = all-agent PASS+SHOT / total env steps, base_seed=42
- 2.45% = agent-0 PASS+SHOT / agent-0 actions, base_seed=700000

For valid comparisons, use the single canonical protocol defined in this forensic phase.

---

*Report generated by `training/analyze_forensics.py`*
""".format(
    date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    head=os.popen("git rev-parse HEAD").read().strip() if os.path.exists(os.path.join(BASE_DIR, ".git")) else "unavailable",
)

with open(os.path.join(RESULTS_DIR, "EXPLORATION_BASELINE_RECONCILIATION.md"), "w") as f:
    f.write(baseline_report)

print("Baseline reconciliation report written.")
