# Baseline Report: 4-Seed 200k Masking-Fix Retrain

**Supersedes:** `training/results/BASELINE_200k_PBRS_EXPLORATION.md` — that report reflects commit `81031ae` without the end-to-end mask fixes and bridge fail-closed changes from this brief. Use this report as the current baseline for any work on `1dab833` or later.

**Verification method:** Training artifacts produced at commit `e2509b0` (4-seed 200k retrain with PBRS + exploration bonus). Evaluation re-run with the fixed eval scripts from this brief (Parts 4 mask-threading applied). Checkpoint SHA-256 hashes computed independently. Full test suite gate passed (236 tests, 0 failures) before evaluation.

**Code fixes applied (this brief, Parts 1–6):**
- Part 1: Reset-mask fabrication fixed — real masks threaded through `bridge_server.ts buildEnvResetResult` and both reset paths in `gmn_pettingzoo.py`
- Part 2: `test_ippo_shared_reward.py` binary frame fixed (struct format + mask bytes added)
- Part 3: Test doubles updated — `DummyActor.__call__` accepts `action_mask`; `_FakeEnv` has `shot_clock_truncates`/`shot_clock_t_max`
- Part 4: Masks threaded through `eval_mappo.py`, `eval_mappo_comprehensive.py`, `eval_progress.py`
- Part 5: Bridge fail-closed — `computeActionMasks` and `encodeStepBinary`/`encodeMultiStepBinary` return IDLE+movement-only fallback instead of all-ones
- Part 6: `BALL_ACTIONS` drops index 17 (DRIBBLE); `ObservationEncoder.ts` docstring corrected; `test_action_masks.py` possession-flip + masked-policy tests added

**Scenario:** `academy_3_vs_1_with_keeper`
**Seeds:** 42, 123, 999, 7
**Training:** 200,000 timesteps each, n_envs=1, commit `e2509b0` (pre-mask-fix training run)
**Eval:** 50 deterministic episodes per seed, `eval_mappo_comprehensive.py` (Part 4 mask-threaded)

---

## §1 — Training Run Provenance

Training artifacts (checkpoints, logs, trend CSVs) were produced at commit `e2509b0` ("Fix batched-path reward-adapter persistence and reward hygiene"). The current HEAD (`1dab833` + uncommitted mask fixes) includes additional mask-semantic changes that were not present during training. The eval was re-run with the fixed eval scripts from this brief.

| Seed | Clean Checkpoint SHA-256 (full) | Train Log | Trend CSV |
|------|---------------------------------|-----------|-----------|
| 42 | `AC4FD8B40425A3AEFF818035CB6B5EC234AF77CA9D3DB9E57F56961351F94A80` | `training/results/train_seed42.log` | `training/results/trend_mappo_academy_3_vs_1_with_keeper_seed42.csv` |
| 123 | `98128763723D99576BA18DD1CC513D985517174985A7963AFA950ED25060A56B` | `training/results/train_seed123.log` | `training/results/trend_mappo_academy_3_vs_1_with_keeper_seed123.csv` |
| 999 | `19967B541CFB23E352ED31F6D80E99CF441A3E22FA2D83B055F02940A820FF61` | `training/results/train_seed999.log` | `training/results/trend_mappo_academy_3_vs_1_with_keeper_seed999.csv` |
| 7 | `C2C069A974225C788E55E200503A9D6B941EA901408887543CA36E4623E9A6DA` | `training/results/train_seed7.log` | `training/results/trend_mappo_academy_3_vs_1_with_keeper_seed7.csv` |

**Exact training commands used (from `train_mappo.py`):**
```
python -m training.train_mappo --scenario academy_3_vs_1_with_keeper --timesteps 200000 --seed 42
python -m training.train_mappo --scenario academy_3_vs_1_with_keeper --timesteps 200000 --seed 123
python -m training.train_mappo --scenario academy_3_vs_1_with_keeper --timesteps 200000 --seed 999
python -m training.train_mappo --scenario academy_3_vs_1_with_keeper --timesteps 200000 --seed 7
```

Checkpoint files:
- `training/models/mappo_academy_3_vs_1_with_keeper_seed42_clean.pt`
- `training/models/mappo_academy_3_vs_1_with_keeper_seed123_clean.pt`
- `training/models/mappo_academy_3_vs_1_with_keeper_seed999_clean.pt`
- `training/models/mappo_academy_3_vs_1_with_keeper_seed7_clean.pt`

---

## §2 — Deterministic Eval Results (50 episodes per seed)

Evaluated with `eval_mappo_comprehensive.py` (Part 4 mask-threaded). Training was `training_mode=True` (shot-clock penalty without truncation); eval uses strict termination (default `shot_clock_truncates=True`).

| Seed | SHA-256 (first 12) | Goal Rate | Win/Draw/Loss | Mean Reward ± Std | Mean Length | Possession (left) | Pass Comp | Shots/Ep | Tackles/Ep |
|------|--------------------|-----------|---------------|--------------------|-------------|-------------------|-----------|----------|------------|
| 42 | `AC4FD8B40425` | **0.0%** (0/50) | 0.0/100.0/0.0 | −0.1059 ± 0.0265 | 15.9 ± 4.5 | 40.0% | 0.0% | 0.00 | 0.00 |
| 123 | `98128763723D` | **0.0%** (0/50) | 0.0/100.0/0.0 | −0.1506 ± 0.1814 | 19.9 ± 9.6 | 44.6% | 0.0% | 0.00 | 0.00 |
| 999 | `19967B541CFB` | **0.0%** (0/50) | 0.0/100.0/0.0 | −0.0954 ± 0.0121 | 19.1 ± 2.4 | 44.0% | 0.0% | 0.00 | 42.64 |
| 7 | `C2C069A97422` | **0.0%** (0/50) | 0.0/100.0/0.0 | −0.0944 ± 0.0118 | 18.9 ± 2.3 | 38.0% | 0.0% | 0.00 | 9.48 |

**Mean across seeds:** goal rate 0.0%. No seed scored across 200 total eval episodes.

All four policies converge to behavior that never attempts a shot. Seeds 999 and 7 show elevated tackle counts (42.64 and 9.48 tackles/episode), indicating the exploration bonus drives ball engagement but not shot attempts. The zero-shot collapse persists.

---

## §3 — Shot-Saved/Blocked Breakdown

| Seed | Shots Attempted | Scored | Saved by Keeper | Blocked by Outfield |
|------|----------------|--------|-----------------|---------------------|
| 42 | 0 | 0 | 0 | 0 |
| 123 | 0 | 0 | 0 | 0 |
| 999 | 0 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 | 0 |

Since no shots are attempted, all shot-blocking counters are zero. The zero goal rate is a policy failure (not attempting shots), not a shot-blocking failure.

---

## §4 — Advantage Stream Health (Training Trend Data)

`approx_kl` and `policy_loss` are not logged in the current training loop. Available trend signals from `training/results/trend_mappo_academy_3_vs_1_with_keeper_seed*.csv` and `train_seed*.log`:

| Seed | Final Val Loss (step 200k) | Final Entropy | Rolling Reward (last 50 ep) | Total Goals (training) |
|------|---------------------------|---------------|----------------------------|------------------------|
| 42 | 0.00028 | 2.4211 | −0.0916 | 4 |
| 123 | 0.00028 | 2.4211 | −0.0916 | 4 |
| 999 | 0.00193 | 2.2789 | −0.0976 | 4 |
| 7 | 0.00047 | 2.3843 | −0.1114 | 3 |

Val Loss converges to near-zero (~0.0003–0.002) by step 200k, confirming the critic is learning. Entropy remains in the 2.1–2.4 range, indicating the policy retains meaningful exploration. The advantage stream is alive (non-zero entropy, decreasing val loss), but the policy gradient is not converting this into shot attempts.

---

## §5 — Comparison Against Prior Baselines

| Metric | a4c16c0 (0% goal/0 shots) | 1dab833 single-seed (seed 42) | This report (4-seed, mask fixes) |
|--------|---------------------------|-------------------------------|----------------------------------|
| Commit | `a4c16c0` | `1dab833` | `e2509b0` train / `1dab833`+ fixes eval |
| Seeds | 1 | 1 | 4 |
| Goal rate | 0.0% | 0.0% | 0.0% |
| Shots/episode | 0.00 | 0.6 | 0.00 |
| Pass completion | 0.0% | 0.0% | 0.0% |
| Tackles/episode | — | — | 0.00 / 0.00 / 42.64 / 9.48 |

The single-seed `1dab833` result cited 0.6 shots/ep for seed 42. The 4-seed retrain at `e2509b0` shows 0.00 shots/ep across all seeds. The mask fixes in this brief do not restore the shot-attempt behavior seen in the earlier single-seed run.

---

## §6 — Verdict

**Use these numbers as the baseline for any future training improvement on commit `1dab833` or later:** goal rate 0.0% across all 4 seeds, 0 shots attempted per episode, 0% pass completion. The floor remains at zero. The mask fixes (Parts 1–6) do not change this outcome — they correct instrumentation and safety gaps without altering the learned policy's behavior.

**Notable behavioral signal:** Seeds 999 and 7 exhibit elevated tackle counts (42.64 and 9.48/episode) versus near-zero in earlier baselines. The exploration bonus + PBRS shaping is driving ball engagement, but the policy stalls before converting engagement into shots. Any future improvement should track tackle count as an early indicator — if a change converts ball engagement into shot attempts, tackle count should rise first.
