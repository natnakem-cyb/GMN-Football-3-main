Baseline Report: 4-Seed 200k PBRS + Exploration-Bonus Retrain

**Supersedes (for post-CCD comparisons):** `training/results/BASELINE_200k_POST_CCD_FIX.md` — that report reflects the post-CCD-fix game with the original reward model. This one reflects commit `81031ae`, which adds PBRS potential-based shaping (γ_shaping=1.0, fixing the discounting leak) and a count-based exploration bonus on top of the same post-CCD game. See "Comparison against the post-CCD baseline" below.

**Verification method:** Fresh training runs from scratch against commit `81031ae`, followed by `eval_mappo_comprehensive.py` with `--force` (no cached results reused). Checkpoint SHA-256 hashes computed independently via `Get-FileHash`.

**Scenario:** `academy_3_vs_1_with_keeper`
**Seeds:** 42, 123, 999, 7
**Training:** 200,000 timesteps each, n_envs=1, no curriculum, commit `81031ae`
**Eval:** 50 deterministic episodes per seed, `eval_mappo_comprehensive.py --force`

---

## §1 — Training Run Provenance

| Seed | Wall Time | Throughput | Clean Checkpoint SHA-256 (full) | Best Det Goal Rate |
|------|-----------|------------|---------------------------------|-------------------|
| 42 | 1281.55s (21.4 min) | 156.0 steps/sec | `5DBCCB936FC8B5AC676909146E5401A57B8FADF596675CF2E9E021E82A25EC76` | 0.0% |
| 123 | 1183.79s (19.7 min) | 168.9 steps/sec | `98128763723D99576BA18DD1CC513D985517174985A7963AFA950ED25060A56B` | 0.0% |
| 999 | 1255.99s (20.9 min) | 159.2 steps/sec | `19967B541CFB23E352ED31F6D80E99CF441A3E22FA2D83B055F02940A820FF61` | 0.0% |
| 7 | 1229.88s (20.5 min) | 162.6 steps/sec | `C2C069A974225C788E55E200503A9D6B941EA901408887543CA36E4623E9A6DA` | 0.0% |

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
| 42 | `5dbccb936fc8` | **0.0%** (0/50) | 0.0/100.0/0.0 | −0.1059 ± 0.0265 | 15.9 ± 4.5 | 40.0% | 0.0% | 0.00 |
| 123 | `98128763723d` | **0.0%** (0/50) | 0.0/100.0/0.0 | −0.1506 ± 0.1814 | 19.9 ± 9.6 | 44.6% | 0.0% | 0.00 |
| 999 | `19967b541cfb` | **0.0%** (0/50) | 0.0/100.0/0.0 | −0.0954 ± 0.0121 | 19.1 ± 2.4 | 44.0% | 0.0% | 0.00 |
| 7 | `c2c069a97422` | **0.0%** (0/50) | 0.0/100.0/0.0 | −0.0944 ± 0.0118 | 18.9 ± 2.3 | 38.0% | 0.0% | 0.00 |

**Mean across seeds:** goal rate 0.0%. No seed scored across 200 total eval episodes.

All four policies converged to behavior that never attempts a shot — the same failure mode as the post-CCD baseline. However, there is a notable behavioral difference: seeds 999 and 7 show markedly elevated tackle counts (42.64 and 9.48 tackles/episode respectively), suggesting the exploration bonus succeeded in driving the agents to engage with the ball more actively than the post-CCD baseline (where those seeds showed passive ~19-step episodes). The engagement did not, however, progress to shot attempts.

---

## §3 — Shot-Blocking Breakdown

| Seed | Shots Attempted | Scored | Saved by Keeper | Blocked by Outfield |
|------|----------------|--------|-----------------|---------------------|
| 42 | 0 | 0 | 0 | 0 |
| 123 | 0 | 0 | 0 | 0 |
| 999 | 0 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 | 0 |

Since no shots are attempted, all shot-blocking counters are zero. As with the post-CCD baseline, the zero goal rate is a policy failure (not attempting shots), not a shot-blocking failure.

---

## §4 — Comparison Against the Post-CCD Baseline

This is the direct, same-protocol comparison the brief calls for: identical scenario, seeds, timesteps, and eval — the only difference is the reward model (commit `81031ae` adds PBRS shaping + exploration bonus on top of the post-CCD game from `3a8dc60`).

| Metric | Post-CCD Baseline (`3a8dc60`) | PBRS + Exploration (`81031ae`) |
|--------|-------------------------------|-------------------------------|
| Seeds | 42, 123, 999, 7 | 42, 123, 999, 7 |
| Commit | `3a8dc60` | `81031ae` |
| Goal rate (all seeds) | 0.0% | 0.0% |
| Shots/episode (all seeds) | 0.00 | 0.00 |
| Pass completion (all seeds) | 0.0% | 0.0% |
| Mean goal rate | 0.0% | 0.0% |

**Headline: shooting behavior did not change.** The exploration bonus + PBRS shaping did not produce a single shot attempt or goal across 200 eval episodes over 4 seeds. The zero-shot collapse persists unchanged.

**What did change (behavioral, not outcome):** Seeds 999 and 7 show substantially more ball engagement under the exploration bonus — seed 999 recorded 42.64 tackles/episode and seed 7 recorded 9.48 tackles/episode, versus near-zero defensive action in the post-CCD baseline. This indicates the exploration bonus is having *some* effect on the learned policy (more active ball pursuit), but that effect stalls before the shot action. The agents approach and contest the ball without ever committing to a shot.

**Why this is not a contradiction:** The exploration bonus rewards visiting under-visited pitch regions (novelty), and tackling/ball-pursuit moves the ball around the pitch, generating novelty reward. But the shot action itself is gated on being close to the goal with possession, and the local reward landscape there (PBRS potential + shot-clock + keeper save probability) apparently still does not overcome the policy's reluctance to shoot within 200k timesteps.

---

## §5 — Verdict

**The PBRS discounting leak fix and count-based exploration bonus, as implemented, did not resolve the zero-shot collapse.** Goal rate remains 0.0% across all 4 seeds with 0 shots attempted per episode — statistically indistinguishable from the post-CCD baseline on the headline metric.

**This is a negative result, but an informative one.** The exploration bonus demonstrably changed agent behavior (increased ball engagement via tackles) without changing the outcome (shots/goals). This suggests the bottleneck is not purely exploration — the agents can be driven to interact with the ball but not to shoot. Possible explanations: (1) the shot action's expected value remains below the exploration-driven alternative within the training budget, (2) the keeper's save probability makes shooting unrewarding at the distances the agents reach, or (3) 200k timesteps is insufficient for the exploration bonus to reshape the policy all the way to shooting.

**Use these numbers as the baseline for any future reward-model or exploration improvement on commit `81031ae` or later:** goal rate 0.0% across all 4 seeds, 0 shots attempted per episode, 0% pass completion. The floor remains at zero. Any future improvement should be measured against this 0% baseline. The elevated tackle counts in seeds 999/7 are a secondary signal worth tracking in future ablations — if a future change converts that ball engagement into shots, the tackle count should rise first, providing an early indicator before goals appear.