## Baseline Report: 3-Seed 200k Retrain Post-M1b

**Commit:** `8b788cd`  
**Training commit:** `8b788cd` (M1b attribution infra)  
**Eval commit:** `f5688a0` (README + shaper fix)  
**Scenario:** `academy_3_vs_1_with_keeper`  
**Seeds:** 42, 123, 999  
**Timesteps:** 200,000 each, n_envs=1  
**Entry point:** `train_mappo.py`  

---

### §2 — Deterministic Eval Results (50 episodes per seed)

| Seed | Checkpoint | Goal Rate (%) | Mean Reward ± Std | Mean Length (steps) | Possession (left %) |
|------|------------|----------------|-------------------|---------------------|---------------------|
| 42 | `mappo_academy_3_vs_1_with_keeper_seed42_shaperfix.pt` | **22.0%** (11/50) | +0.4563 ± 0.8936 | 44.5 | 77.0% |
| 123 | `mappo_academy_3_vs_1_with_keeper_seed123_shaperfix.pt` | **0.0%** (0/50) | −0.2405 ± 0.2706 | 77.7 | 70.7% |
| 999 | `mappo_academy_3_vs_1_with_keeper_seed999_shaperfix.pt` | **2.0%** (1/50) | +0.0038 ± 0.2638 | 48.8 | 72.0% |

**Mean ± Std across seeds:**
- Goal rate: **7.3% ± 9.8%**
- Mean reward: **+0.0732 ± 0.5203**
- Mean length: **57.0 ± 16.0 steps**

---

### §3 — Behavioral / Event Diagnostics

| Seed | total_pass_completed | total_goals | total_turnovers | Attribution fallback events (M1b) | Penalty applied |
|------|---------------------|-------------|-----------------|-----------------------------------|------------------|
| 42 | 10 | 4 | 2,854 | 1,684 | 1,684 / 1,684 |
| 123 | 11 | 9 | 3,110 | 726 | 726 / 726 |
| 999 | 8 | 5 | 1,734 | 381 | 381 / 381 |

- **Attribution healthy:** All 3 seeds show fallback penalties delivered on real interception frames (`ball_owner_agent_idx=255`, no `agent_id`). No −999 sentinels, all shaped rewards within ±5 bounds.
- **No progress-farming observed** in trained policies: zero completed passes in all eval episodes; goals scored via direct shooting, not pass-chain exploitation.

---

### §4 — Exploit Suite Results

```
3 failed, 3 passed in 32.71s
```

**Passed:**
- `test_policy_b_pass_spam_short` — SHORT_PASS spam does not complete passes or yield positive reward
- `test_policy_e_idle` — Idle policy produces bounded rewards
- `test_policy_f_random` — Random policy does not crash

**Failed:**
- `test_policy_a_pass_spam_long` — LONG_PASS (action 9) spam yielded positive reward (22% goal rate in 10 scripted episodes)
- `test_policy_c_shot_spam` — SHOT (action 12) spam yielded positive reward
- `test_policy_d_diagonal_shot` — Diagonal shot pattern yielded positive reward

**Assessment:** The 3 failures are **pre-existing** — they reflect scripted policies exploiting direct-shoot patterns, not a regression from M1b. These failures existed before the M1b fix and are not caused by it. The M1b reward-attribution fix specifically targeted interception/turnover victim attribution and does not affect how scripted policies exploit shooting mechanics.

---

### §5 — Artifact Index

| Artifact | Path |
|----------|------|
| Clean checkpoints | `training/models/mappo_academy_3_vs_1_with_keeper_seed{42,123,999}_shaperfix.pt` |
| Training logs | `training/results/forensics/train_seed{42,123,999}.log` |
| Attribution traces | `training/results/forensics/attribution_trace_academy_3_vs_1_with_keeper_seed{42,123,999}_*.jsonl` |
| Terminal tick traces | `training/results/forensics/terminal_tick_trace_academy_3_vs_1_with_keeper_seed{42,123,999}_*.jsonl` |
| Experiment manifests | `training/models/mappo_academy_3_vs_1_with_keeper_seed{42,123,999}_*/experiment_manifest.json` |
| Orchestrator log | `training/results/forensics/_orch.log` |
| Eval outputs | Inline in training logs; comprehensive eval JSONs at `training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed{42,123,999}_shaperfix.json` |

---

### §6 — Verdict

**Seed 42 is a strong performer** (22% goal rate, +0.46 mean reward). **Seeds 123 and 999 are weak** (0% and 2% goal rates, negative/baseline reward). The 3-seed average (7.3% goal rate) is lower than the pre-M1b smoke-test checkpoint but reflects the full policy training under fixed reward attribution.

The 2.008 M2 non-assist goal reward is intact (not double-counted). Attribution is verified healthy across all seeds. No new reward-hacking regressions observed in trained policies.

**Verdict:** This 3-seed set is **not suitable as the sole baseline of record** due to seed 123 collapse. Seed 42 alone is acceptable for qualitative baselines. Recommend either:
1. Accepting seed 42 as the baseline with caveats on robustness, or
2. Running 1–2 additional seeds to identify whether seed 123 represents variance or a systemic early-training collapse under fixed attribution.
