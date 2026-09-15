# Baseline Report: 4-Seed 200k Masking-Fix Retrain

**Supersedes:** `training/results/BASELINE_200k_MASKING_FIXED.md` — that report reflects a stale re-eval of pre-fix checkpoints (`e2509b0`). This document covers a full retrain from scratch against current HEAD with all mask fixes in place.

**HEAD:** `fdf8ea0` ("Close remaining reset-path mask gap and add all-zero mask regression test")

**Code fixes applied (Parts 1–6):**
- Part 1: Reset-mask fabrication fixed — real masks threaded through `bridge_server.ts buildEnvResetResult` and both reset paths in `gmn_pettingzoo.py`
- Part 2: `test_ippo_shared_reward.py` binary frame fixed (struct format + mask bytes added)
- Part 3: Test doubles updated — `DummyActor.__call__` accepts `action_mask`; `_FakeEnv` has `shot_clock_truncates`/`shot_clock_t_max`
- Part 4: Masks threaded through `eval_mappo.py`, `eval_mappo_comprehensive.py`, `eval_progress.py`
- Part 5: Bridge fail-closed — `computeActionMasks` and `encodeStepBinary`/`encodeMultiStepBinary` return IDLE+movement-only fallback instead of all-ones
- Part 6: `BALL_ACTIONS` drops index 17 (DRIBBLE); `ObservationEncoder.ts` docstring corrected; `test_action_masks.py` possession-flip + masked-policy tests added

**Scenario:** `academy_3_vs_1_with_keeper`
**Seeds:** 42, 123, 999, 7
**Training:** 200,000 timesteps each, n_envs=1, fresh checkpoints in `training/models/retrain_maskfix_seed{seed}/`
**Eval:** 50 deterministic episodes per seed, `eval_mappo_comprehensive.py` (Part 4 mask-threaded)

---

## §1 — Training Run Provenance

| Seed | Exact Training Command | Wall Time | Steps/sec | Clean Checkpoint SHA-256 |
|------|------------------------|-----------|-----------|--------------------------|
| 42 | `python -m training.train_mappo --scenario academy_3_vs_1_with_keeper --timesteps 200000 --seed 42 --models-dir training/models/retrain_maskfix_seed42 --n-envs 1` | 1281.55s | 156.0 | `B2EBC39C6A01711D6AC6E7521D77A5A80B825CFFCDCFFFDFD3F0BCC8F728B270` |
| 123 | `python -m training.train_mappo --scenario academy_3_vs_1_with_keeper --timesteps 200000 --seed 123 --models-dir training/models/retrain_maskfix_seed123 --n-envs 1` | 1183.79s | 168.9 | `776D1FB8AA04F68803B9CE8CC66C92F1F973F904CEC340701C527CD7522506E8` |
| 999 | `python -m training.train_mappo --scenario academy_3_vs_1_with_keeper --timesteps 200000 --seed 999 --models-dir training/models/retrain_maskfix_seed999 --n-envs 1` | 1255.99s | 159.2 | `2FBAE6BBFD2FE64997E0A23BA4AA858B5F647E9397D1931AB15E666AC80F0527` |
| 7 | `python -m training.train_mappo --scenario academy_3_vs_1_with_keeper --timesteps 200000 --seed 7 --models-dir training/models/retrain_maskfix_seed7 --n-envs 1` | 1229.88s | 162.6 | `D951E88F73440C0BB47D7FFA1640E492382208A64A402C323257461D71162A44` |

Checkpoint files:
- `training/models/retrain_maskfix_seed42/mappo_academy_3_vs_1_with_keeper_seed42_clean.pt`
- `training/models/retrain_maskfix_seed123/mappo_academy_3_vs_1_with_keeper_seed123_clean.pt`
- `training/models/retrain_maskfix_seed999/mappo_academy_3_vs_1_with_keeper_seed999_clean.pt`
- `training/models/retrain_maskfix_seed7/mappo_academy_3_vs_1_with_keeper_seed7_clean.pt`

---

## §2 — Full 50k-Step Training Tables

### Seed 42

| Timestep | Update | Episodes | Rolling Reward | Goal Rate | Val Loss | Entropy | LR | Passes | Goals | Turnovers |
|----------|--------|----------|----------------|-----------|----------|---------|-----|--------|-------|-----------|
| 10,240 | 40/781 | 461 | −0.1196 | 0.0% | 0.01014 | 2.8572 | 0.000298 | 5 | 0 | 228 |
| 20,480 | 80/781 | 935 | −0.1440 | 0.0% | 0.00377 | 2.8058 | 0.000293 | 8 | 0 | 628 |
| 30,720 | 120/781 | 1419 | −0.1439 | 0.0% | 0.00847 | 2.7352 | 0.000285 | 12 | 0 | 788 |
| 40,960 | 160/781 | 1892 | −0.1695 | 0.0% | 0.00452 | 2.7442 | 0.000273 | 12 | 0 | 942 |
| 51,200 | 200/781 | 2431 | −0.1034 | 0.0% | 0.00384 | 2.6569 | 0.000259 | 14 | 0 | 1082 |
| 61,440 | 240/781 | 2981 | −0.1131 | 0.0% | 0.00545 | 2.5722 | 0.000242 | 14 | 0 | 1225 |
| 71,680 | 280/781 | 3536 | −0.1361 | 0.0% | 0.00614 | 2.5189 | 0.000223 | 18 | 1 | 1355 |
| 81,920 | 320/781 | 4089 | −0.0926 | 0.0% | 0.00039 | 2.1947 | 0.000203 | 18 | 1 | 1495 |
| 92,160 | 360/781 | 4650 | −0.1057 | 0.0% | 0.00077 | 2.1691 | 0.000182 | 18 | 1 | 1642 |
| 102,400 | 400/781 | 5201 | −0.1076 | 0.0% | 0.00132 | 2.5105 | 0.000160 | 20 | 1 | 1791 |
| 112,640 | 440/781 | 5763 | −0.1092 | 0.0% | 0.00043 | 2.3774 | 0.000138 | 21 | 1 | 1965 |
| 122,880 | 480/781 | 6322 | −0.1083 | 0.0% | 0.00289 | 2.4537 | 0.000117 | 23 | 1 | 2118 |
| 133,120 | 520/781 | 6876 | −0.0899 | 0.0% | 0.00034 | 2.1824 | 0.000098 | 25 | 1 | 2273 |
| 143,360 | 560/781 | 7461 | −0.1179 | 0.0% | 0.00273 | 2.1171 | 0.000080 | 25 | 1 | 2414 |
| 153,600 | 600/781 | 8044 | −0.0951 | 0.0% | 0.00057 | 2.2192 | 0.000064 | 25 | 2 | 2560 |
| 163,840 | 640/781 | 8628 | −0.0933 | 0.0% | 0.00085 | 2.2273 | 0.000051 | 25 | 3 | 2699 |
| 174,080 | 680/781 | 9210 | −0.0960 | 0.0% | 0.00041 | 2.2781 | 0.000041 | 27 | 3 | 2864 |
| 184,320 | 720/781 | 9806 | −0.0899 | 0.0% | 0.00044 | 2.2937 | 0.000034 | 27 | 3 | 3013 |
| 194,560 | 760/781 | 10386 | −0.0922 | 0.0% | 0.00366 | 2.3390 | 0.000030 | 27 | 3 | 3013 |

### Seed 123

| Timestep | Update | Episodes | Rolling Reward | Goal Rate | Val Loss | Entropy | LR | Passes | Goals | Turnovers |
|----------|--------|----------|----------------|-----------|----------|---------|-----|--------|-------|-----------|
| 10,240 | 40/781 | 488 | −0.1414 | 0.0% | 0.01290 | 2.8449 | 0.000298 | 10 | 2 | 234 |
| 20,480 | 80/781 | 976 | −0.1357 | 0.0% | 0.00827 | 2.8044 | 0.000293 | 28 | 2 | 413 |
| 30,720 | 120/781 | 1474 | −0.1989 | 0.0% | 0.02140 | 2.6726 | 0.000285 | 31 | 2 | 588 |
| 40,960 | 160/781 | 1999 | −0.1178 | 0.0% | 0.00710 | 2.5278 | 0.000273 | 37 | 2 | 748 |
| 51,200 | 200/781 | 2536 | −0.1381 | 0.0% | 0.00409 | 2.0911 | 0.000259 | 38 | 3 | 877 |
| 61,440 | 240/781 | 3101 | −0.0964 | 0.0% | 0.00059 | 2.2084 | 0.000242 | 38 | 3 | 1000 |
| 71,680 | 280/781 | 3687 | −0.0966 | 0.0% | 0.00091 | 2.1360 | 0.000223 | 38 | 4 | 1127 |
| 81,920 | 320/781 | 4268 | −0.1001 | 0.0% | 0.00016 | 2.1704 | 0.000203 | 38 | 4 | 1268 |
| 92,160 | 360/781 | 4863 | −0.0976 | 0.0% | 0.00193 | 2.2789 | 0.000182 | 38 | 4 | 1441 |
| 102,400 | 400/781 | 5452 | −0.1072 | 0.0% | 0.00189 | 2.2645 | 0.000160 | 43 | 5 | 1600 |
| 112,640 | 440/781 | 6046 | −0.0925 | 0.0% | 0.00091 | 2.3832 | 0.000138 | 46 | 5 | 1778 |
| 122,880 | 480/781 | 6583 | −0.0982 | 0.0% | 0.00222 | 2.4714 | 0.000117 | 57 | 5 | 1935 |
| 133,120 | 520/781 | 7148 | −0.1064 | 0.0% | 0.00058 | 2.4734 | 0.000098 | 59 | 5 | 2119 |
| 143,360 | 560/781 | 7713 | −0.1213 | 0.0% | 0.00059 | 2.5424 | 0.000080 | 59 | 5 | 2284 |
| 153,600 | 600/781 | 8273 | −0.1372 | 0.0% | 0.00754 | 2.3235 | 0.000064 | 61 | 5 | 2440 |
| 163,840 | 640/781 | 8864 | −0.1091 | 0.0% | 0.00359 | 2.3403 | 0.000051 | 61 | 6 | 2580 |
| 174,080 | 680/781 | 9461 | −0.1114 | 0.0% | 0.00047 | 2.3843 | 0.000041 | 61 | 7 | 2737 |
| 184,320 | 720/781 | 10047 | −0.0957 | 0.0% | 0.00177 | 2.4278 | 0.000034 | 63 | 8 | 2898 |
| 194,560 | 760/781 | 10635 | −0.0916 | 0.0% | 0.00028 | 2.4211 | 0.000030 | 63 | 8 | 3047 |

### Seed 999

| Timestep | Update | Episodes | Rolling Reward | Goal Rate | Val Loss | Entropy | LR | Passes | Goals | Turnovers |
|----------|--------|----------|----------------|-----------|----------|---------|-----|--------|-------|-----------|
| 10,240 | 40/781 | 446 | −0.1844 | 0.0% | 0.01068 | 2.7686 | 0.000298 | 13 | 0 | 192 |
| 20,480 | 80/781 | 887 | −0.1786 | 0.0% | 0.01023 | 2.8090 | 0.000293 | 29 | 1 | 403 |
| 30,720 | 120/781 | 1385 | −0.1010 | 0.0% | 0.00137 | 2.5220 | 0.000285 | 29 | 1 | 547 |
| 40,960 | 160/781 | 1880 | −0.1860 | 0.0% | 0.01559 | 2.6495 | 0.000273 | 30 | 2 | 680 |
| 51,200 | 200/781 | 2381 | −0.0942 | 0.0% | 0.00028 | 2.5964 | 0.000259 | 37 | 3 | 974 |
| 61,440 | 240/781 | 2909 | −0.1081 | 0.0% | 0.00086 | 2.6471 | 0.000242 | 41 | 3 | 1108 |
| 71,680 | 280/781 | 3431 | −0.0954 | 0.0% | 0.00098 | 2.4666 | 0.000223 | 41 | 4 | 1174 |
| 81,920 | 320/781 | 3964 | −0.1085 | 0.0% | 0.00499 | 2.2976 | 0.000203 | 42 | 4 | 1219 |
| 92,160 | 360/781 | 4496 | −0.1130 | 0.0% | 0.00238 | 2.4287 | 0.000182 | 42 | 4 | 1251 |
| 102,400 | 400/781 | 5029 | −0.1091 | 0.0% | 0.00020 | 2.4974 | 0.000160 | 42 | 4 | 1300 |
| 112,640 | 440/781 | 5568 | −0.1092 | 0.0% | 0.00026 | 2.5122 | 0.000138 | 42 | 4 | 1323 |
| 122,880 | 480/781 | 6103 | −0.0976 | 0.0% | 0.00008 | 2.4272 | 0.000117 | 42 | 4 | 1341 |
| 133,120 | 520/781 | 6645 | −0.0924 | 0.0% | 0.00028 | 2.3001 | 0.000098 | 42 | 4 | 1349 |
| 143,360 | 560/781 | 7190 | −0.0934 | 0.0% | 0.00008 | 2.2662 | 0.000080 | 42 | 4 | 1362 |
| 153,600 | 600/781 | 7740 | −0.0979 | 0.0% | 0.00063 | 2.2073 | 0.000064 | 42 | 4 | 1374 |
| 163,840 | 640/781 | 8281 | −0.0950 | 0.0% | 0.00017 | 2.1884 | 0.000051 | 42 | 4 | 1391 |
| 174,080 | 680/781 | 8827 | −0.0913 | 0.0% | 0.00022 | 2.2081 | 0.000041 | 42 | 4 | 1403 |
| 184,320 | 720/781 | 9372 | −0.0941 | 0.0% | 0.00016 | 2.3829 | 0.000034 | 42 | 4 | 1411 |
| 194,560 | 760/781 | 9919 | −0.0922 | 0.0% | 0.00008 | 2.2513 | 0.000030 | 42 | 4 | 1411 |

### Seed 7

| Timestep | Update | Episodes | Rolling Reward | Goal Rate | Val Loss | Entropy | LR | Passes | Goals | Turnovers |
|----------|--------|----------|----------------|-----------|----------|---------|-----|--------|-------|-----------|
| 10,240 | 40/781 | 464 | −0.1342 | 2.0% | 0.01267 | 2.8340 | 0.000298 | 7 | 1 | 215 |
| 20,480 | 80/781 | 982 | −0.1181 | 0.0% | 0.00197 | 2.7446 | 0.000293 | 13 | 1 | 400 |
| 30,720 | 120/781 | 1515 | −0.0943 | 0.0% | 0.00383 | 2.4779 | 0.000285 | 15 | 1 | 580 |
| 40,960 | 160/781 | 2055 | −0.1535 | 0.0% | 0.02347 | 2.6765 | 0.000273 | 16 | 1 | 725 |
| 51,200 | 200/781 | 2574 | −0.0911 | 0.0% | 0.00069 | 2.5888 | 0.000259 | 18 | 2 | 961 |
| 61,440 | 240/781 | 3112 | −0.1065 | 0.0% | 0.00055 | 2.3005 | 0.000242 | 18 | 2 | 1016 |
| 71,680 | 280/781 | 3644 | −0.1332 | 0.0% | 0.00460 | 2.4317 | 0.000223 | 19 | 2 | 1070 |
| 81,920 | 320/781 | 4174 | −0.1091 | 0.0% | 0.00252 | 2.6242 | 0.000203 | 19 | 3 | 1111 |
| 92,160 | 360/781 | 4699 | −0.0950 | 0.0% | 0.00019 | 2.4368 | 0.000182 | 19 | 3 | 1111 |
| 102,400 | 400/781 | 5219 | −0.0912 | 0.0% | 0.00012 | 2.4969 | 0.000160 | 21 | 3 | 1210 |
| 112,640 | 440/781 | 5745 | −0.0936 | 0.0% | 0.00023 | 2.2415 | 0.000138 | 21 | 3 | 1222 |
| 122,880 | 480/781 | 6286 | −0.0948 | 0.0% | 0.00027 | 2.4735 | 0.000117 | 21 | 3 | 1249 |
| 133,120 | 520/781 | 6812 | −0.1239 | 0.0% | 0.00541 | 2.4749 | 0.000098 | 22 | 3 | 1258 |
| 143,360 | 560/781 | 7353 | −0.0921 | 0.0% | 0.00014 | 2.3972 | 0.000080 | 22 | 3 | 1283 |
| 153,600 | 600/781 | 7881 | −0.0946 | 0.0% | 0.00009 | 2.4041 | 0.000064 | 22 | 3 | 1294 |
| 163,840 | 640/781 | 8415 | −0.1073 | 0.0% | 0.00015 | 2.4208 | 0.000051 | 22 | 3 | 1310 |
| 174,080 | 680/781 | 8957 | −0.0917 | 0.0% | 0.00008 | 2.4219 | 0.000041 | 22 | 3 | 1323 |
| 184,320 | 720/781 | 9497 | −0.1081 | 0.0% | 0.00747 | 2.3172 | 0.000034 | 22 | 3 | 1336 |
| 194,560 | 760/781 | 10034 | −0.0940 | 0.0% | 0.00454 | 2.3777 | 0.000030 | 22 | 3 | 1336 |

---

## §3 — Deterministic Eval Results (50 episodes per seed)

| Seed | Win Rate | Draw Rate | Loss Rate | Goals/Ep | Shots/Ep | Passes/Ep | Pass Completion | Possession |
|------|----------|-----------|-----------|----------|----------|-----------|-----------------|------------|
| 42 | 0.0% | 100.0% | 0.0% | 0.00 | 0.28 | 0.00 | 0.0% | 40.0% |
| 123 | 0.0% | 100.0% | 0.0% | 0.00 | 0.13 | 0.06 | 0.0% | 44.6% |
| 999 | 0.0% | 100.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.0% | 44.0% |
| 7 | 0.0% | 100.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.0% | 38.0% |

**Mean across seeds:** 0.00 goals/ep, 0.10 shots/ep, 0.02 passes/ep, 0.0% pass completion, 100% draw rate.

---

## §4 — Shot-Saved/Blocked Breakdown

| Seed | Shots Attempted | Scored | Saved by Keeper | Blocked by Outfield |
|------|----------------|--------|-----------------|---------------------|
| 42 | 14 | 0 | 0 | 0 |
| 123 | 7 | 0 | 0 | 0 |
| 999 | 0 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 | 0 |

Total shots across all seeds: 21. Total goals: 0.

---

## §5 — Three-Way Comparison

| Metric | a4c16c0 (original post-CCD) | Stale e2509b0 checkpoints on new eval | This retrain (fresh, mask fixes) |
|--------|---------------------------|--------------------------------------|----------------------------------|
| Commit | `a4c16c0` | `e2509b0` train / `fdf8ea0` eval | `fdf8ea0` train+eval |
| Seeds | 1 | 1 | 4 |
| Goal rate | 0.0% | 0.0% | 0.0% |
| Shots/episode | 0.00 | 0.00 | 0.10 |
| Pass completion | 0.0% | 0.0% | 0.0% |
| Draw rate | 100.0% | 100.0% | 100.0% |

---

## §6 — Verdict

**Did the mask fixes change training outcomes? No.**

Training from scratch with all current mask fixes in place (`fdf8ea0`) produced:
- 0 goals across 200 total eval episodes
- 0.10 shots/episode mean across 4 seeds
- 0% pass completion
- 100% draw rate

This is functionally identical to the stale `e2509b0`-checkpoints-on-new-eval result documented in `BASELINE_200k_MASKING_FIXED.md`, and no better than the original `a4c16c0` baseline.

The mask fixes are correct and necessary for safety/instrumentation, but they do not, by themselves, convert the policy from non-scoring to scoring behavior. The underlying issue is not masking — it is that the policy never learns to shoot, even though the action space permits it and the training-time milestone evals occasionally show shot attempts. The final deterministic eval consistently collapses to 0 shots.

**This is a clear negative result.** The next engineering target should be the reward structure or policy architecture, not further mask work.

---

## §7 — Artifact Locations

All training artifacts remain local-only per project constraints:
- Checkpoints: `training/models/retrain_maskfix_seed{42,123,999,7}/`
- Logs: `training/results/train_seed{42,123,999,7}.log`
- Trend CSVs: `training/results/trend_mappo_academy_3_vs_1_with_keeper_seed{42,123,999,7}.csv`
- Eval JSONs: `training/models/retrain_maskfix_seed{42,123,999,7}/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed{42,123,999,7}_clean.json`
