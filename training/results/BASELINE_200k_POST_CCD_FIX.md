# Baseline Report: 4-Seed 200k Post-CCD-Fix Retrain

**Superseded by:** `training/results/BASELINE_200k_PBRS_EXPLORATION.md` — that report reflects commit `81031ae`, which adds PBRS shaping and a count-based exploration bonus on top of this same post-CCD game. Use the newer report as the current baseline for any work on `81031ae` or later. This report is retained for historical comparison against the pre-CCD game.

**Supersedes (for post-CCD comparisons):** `training/results/BASELINE_200k_VERIFIED.md` — that report reflects the pre-CCD-fix game (shots tunneled through defenders with no resistance). This one reflects current HEAD with real shot-blocking in play. See "Comparison against the old baseline" below.

**Verification method:** Fresh training runs from scratch against commit `3a8dc60`, followed by `eval_mappo_comprehensive.py` with `--force` (no cached results reused). Checkpoint SHA-256 hashes computed independently via `Get-FileHash`.

**Scenario:** `academy_3_vs_1_with_keeper`
**Seeds:** 42, 123, 999, 7
**Training:** 200,000 timesteps each, n_envs=1, no curriculum, commit `3a8dc60`
**Eval:** 50 deterministic episodes per seed, `eval_mappo_comprehensive.py --force`

---

## §1 — Training Run Provenance

| Seed | Wall Time | Throughput | Clean Checkpoint SHA-256 (full) | Best Det Goal Rate | Best Step |
|------|-----------|------------|---------------------------------|-------------------|----------|
| 42 | 1120.27s (18.7 min) | 178.5 steps/sec | `49DF95A5B1E2183F67D5660E42F88EC3E5266030FB72B4CF10AD8C5A29CD0189` | 0.0% | 50176 |
| 123 | 1042.30s (17.4 min) | 191.8 steps/sec | `AE74C3D004448FD457255E29EA197309217790BC9EA66C657AF2DCF0E271F9C0` | 0.0% | 50176 |
| 999 | 1062.73s (17.7 min) | 188.1 steps/sec | `0CCB18EA8A6B26EDB2709ADF5E4C1FB3363EAB0909FAA965320878BA7F383462` | 0.0% | 50176 |
| 7 | 1017.24s (17.0 min) | 196.5 steps/sec | `02780709F108A7C5042217138484FC7F979B7D9FED572DC0A95EDF7B0A20A75C` | 0.0% | 50176 |

**Exact commands used:**
```
python -m training.train_mappo --scenario academy_3_vs_1_with_keeper --timesteps 200000 --seed 42
python -m training.train_mappo --scenario academy_3_vs_1_with_keeper --timesteps 200000 --seed 123
python -m training.train_mappo --scenario academy_3_vs_1_with_keeper --timesteps 200000 --seed 999
python -m training.train_mappo --scenario academy_3_vs_1_with_keeper --timesteps 200000 --seed 7
```

All runs completed without error.

Checkpoint files:
- `training/models/mappo_academy_3_vs_1_with_keeper_seed42_clean.pt`
- `training/models/mappo_academy_3_vs_1_with_keeper_seed123_clean.pt`
- `training/models/mappo_academy_3_vs_1_with_keeper_seed999_clean.pt`
- `training/models/mappo_academy_3_vs_1_with_keeper_seed7_clean.pt`

---

## §2 — Deterministic Eval Results (50 episodes per seed)

| Seed | SHA-256 (first 12) | Goal Rate | Win/Draw/Loss | Mean Reward ± Std | Mean Length | Possession | Pass Comp | Shots/Ep |
|------|--------------------|-----------|---------------|--------------------|-------------|------------|-----------|----------|
| 42 | `49df95a5b1e2` | **0.0%** (0/50) | 0.0/100.0/0.0 | −0.1092 ± 0.1952 | 24.9 ± 10.7 | 61.0% | 0.0% | 0.00 |
| 123 | `ae74c3d00444` | **0.0%** (0/50) | 0.0/100.0/0.0 | −1.0256 ± 1.4702 | 73.2 ± 74.9 | 64.5% | 0.0% | 0.00 |
| 999 | `0ccb18ea8a6b` | **0.0%** (0/50) | 0.0/100.0/0.0 | 0.0000 ± 0.0000 | 19.0 ± 2.4 | 43.0% | 0.0% | 0.00 |
| 7 | `02780709f108` | **0.0%** (0/50) | 0.0/100.0/0.0 | 0.0000 ± 0.0000 | 19.0 ± 2.4 | 43.0% | 0.0% | 0.00 |

**Mean across seeds:** goal rate 0.0%. No seed scored across 200 total eval episodes.

All four policies converged to behavior that never attempts a shot. Seeds 42/123 have longer episodes (25–73 steps) with moderate possession (61–65%); seeds 999/7 terminate quickly (~19 steps) at low possession (43%). None ever trigger a shot action.

---

## §3 — CCD Shot-Blocking Breakdown

These counters (`shots_saved_left`, `shots_blocked_left` from `gmn_pettingzoo.py` `ground_truth`, added in `d5cf1df`) are new — they had no meaning under the old tunneling bug.

| Seed | Shots Attempted | Scored | Saved by Keeper | Blocked by Outfield |
|------|----------------|--------|-----------------|---------------------|
| 42 | 0 | 0 | 0 | 0 |
| 123 | 0 | 0 | 0 | 0 |
| 999 | 0 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 | 0 |

Since no shots are attempted, all shot-blocking counters are zero. The CCD fix is not being exercised because the policies never reach the point of shooting. The zero goal rate is a policy failure (not attempting shots), not a shot-blocking failure (attempts being saved).

---

## §4 — Comparison Against the Old Baseline

| Metric | Old Baseline (pre-CCD, 3 seeds) | New Baseline (post-CCD, 4 seeds) |
|--------|--------------------------------|----------------------------------|
| Seeds | 42, 123, 999 | 42, 123, 999, 7 |
| Commit | `3566a6a` + rondo/curriculum fixes | `3a8dc60` (includes CCD fix `d5cf1df`) |
| Goal rate range | 0.0% – 4.0% | 0.0% – 0.0% |
| Mean goal rate | 2.0% | 0.0% |
| Pass completion | 0.0% (all) | 0.0% (all) |
| Shots/episode | Not reported | 0.00 (all) |

**Why these numbers are not directly comparable:** The CCD fix changed shooting mechanics — a shot in a defender's path used to tunnel through and score; now it is blocked (outfield) or save-rolled (keeper, 15–75%). The old baseline reflected an easier game. Under the current game, the policies learned to never shoot.

**The meaningful comparison is not pre/post CCD (different games), but this baseline against any future ablation on the same commit.** This is the number to beat: 0% goal rate, 0 shots attempted, 0% pass completion.

---

## §5 — Verdict

**Use these numbers as the baseline for any future GNN ablation or training improvement on commit `3a8dc60` or later:** goal rate 0.0% across all 4 seeds, 0 shots attempted per episode, 0% pass completion. This is a weaker baseline than the pre-CCD one — not because the policies are necessarily worse in a vacuum, but because the game is genuinely harder now and the policies adapted by not shooting at all.

**This baseline should be treated as unstable and weak.** A 0% goal rate across 200 episodes with 4 seeds means any future improvement (even a single goal) would be statistically notable, but also that the absolute floor is very low. Any claim of improvement should be measured against this 0% baseline, not against the old 2–4% numbers, because the underlying game has changed.
