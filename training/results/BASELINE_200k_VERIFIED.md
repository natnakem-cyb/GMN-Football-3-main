# Baseline Report: 3-Seed 200k Shaperfix Retrain — INDEPENDENTLY VERIFIED

**Supersedes:** `training/results/BASELINE_200k_M1b_report.md` (§2 and §5 of that report contain errors — see "Corrections" below). Do not cite that report's headline numbers going forward; cite this one.

**Verification method:** Fresh full clone of the repo, HEAD confirmed matching remote. Re-ran `training/eval_mappo_comprehensive.py` from scratch against the exact `*_shaperfix.pt` checkpoint files already committed, on a freshly booted live bridge. Independently confirmed checkpoint SHA-256 hashes. Separately drove a live bridge instance by hand (dribble + shot) to confirm the engine's own terminal/goal signaling is correct, independent of any training/eval code.

**Scenario:** `academy_3_vs_1_with_keeper`
**Seeds:** 42, 123, 999
**Training:** 200,000 timesteps each, n_envs=1, reward-shaper wiring fix applied (commit `3566a6a`)
**Eval:** 50 deterministic episodes per seed, `eval_mappo_comprehensive.py`

---

## §1 — Verified Deterministic Eval Results (50 episodes per seed)

| Seed | Checkpoint SHA-256 (first 12 chars) | Goal Rate | Mean Reward ± Std | Mean Length (steps) | Possession (left) | Pass Completion |
|------|--------------------------------------|-----------|--------------------|-----------------------|--------------------|------------------|
| 42   | `0176b46fca28` | **4.0%** (2/50)  | +0.0673 ± 0.4202 | 75.9 | 78.0% | 0.0% |
| 123  | `fbed2afa5207` | **0.0%** (0/50)  | −0.2593 ± 0.2941 | 42.4 | 70.3% | 0.0% |
| 999  | `33bf0bf9b4bb` | **2.0%** (1/50)  | −0.0159 ± 0.2340 | 49.0 | 79.0% | 0.0% |

**Mean across seeds:** goal rate 2.0%, highly seed-sensitive (0.0%–4.0% range).

Independently re-run twice (once per verification pass) — both passes reproduced these exact numbers to the checkpoint's own JSON output, with matching SHA-256 hashes each time. These numbers are trustworthy.

Pass completion is 0.0% in all three seeds — the policy is not passing at all in this eval, consistent with prior findings that this scenario tends to converge on direct-shoot behavior rather than team coordination.

---

## §2 — Engine correctness check (independent of training/eval code)

To rule out an instrumentation problem before trusting the low goal rates above, a live bridge instance was driven by hand with a simple scripted dribble-then-shoot sequence (unrelated to any trained checkpoint):

- Player picked up the ball, dribbled toward goal, and fired `SHOT`.
- On the scoring step, the bridge correctly returned `terminated: true`, `info.score.left: 1`, `info.event: "goal"`, and a reward of `+2.006`.

**Conclusion: the engine's goal-scoring and episode-termination logic is functioning correctly.** The low goal rates in §1 reflect genuine policy weakness, not a broken termination signal.

---

## §3 — Corrections to the superseded M1b report

The prior report (`BASELINE_200k_M1b_report.md`) contained three verified errors:

1. **Seed 42 headline goal rate was wrong.** It reported 22.0% (11/50) for the `seed42_shaperfix` checkpoint. The actual eval JSON for that exact checkpoint (SHA-256 confirmed identical) shows 4.0% (2/50). The source of the 22%/37% figures the report may have drawn on traces to an unrelated, older checkpoint (`seed42_best.pt`, commit `4b6f48d`, dated Sep 7) — a different training run entirely, not part of this retrain.
2. **Cited artifacts did not exist.** The report's artifact index listed `training/results/forensics/train_seed{42,123,999}.log`. No such files exist anywhere in the repo or in git history. A similarly-named but unrelated file, `retrain_seed42_validation.log` (commit `7991e879`, an earlier 50k-step single-seed validation run), appears to be the likely source of confusion.
3. **An uncatalogued `seed4242` trace pair** (`episode_reward_trace_..._seed4242_*.csv`, `terminal_tick_trace_..._seed4242_*.jsonl`) was committed alongside the real seed 42/123/999 artifacts but never mentioned in the report. No seed-doubling logic exists anywhere in `train_mappo.py`, `forensic_reproduce.py`, or `diagnostic_shot_spam_score_check.py` that could produce this value — it is most likely a manual one-off smoke-test invocation (`--seed 4242`) that was left uncleaned, timestamped ~13 minutes before the real seed 42 run in the same session.

One item from the original report remains open and **unresolved by this verification pass**: the seed42 episode-reward training trace closest to the eval timestamp shows `terminal=False` and `score_left=0` across all 3,741 logged rows. §2 above shows the underlying engine handles termination correctly in isolation, so this is most likely a logging/instrumentation gap specific to that CSV-writing code path during training — not a sign that no goals were scored during that run. This has not been root-caused and should not be assumed fixed.

---

## §4 — Verdict

**Use these numbers as the Row-A baseline for any GNN ablation work (Section 22 of the GNN architecture proposal):** goal rate 4.0% / 0.0% / 2.0% across seeds 42/123/999, all with 0.0% pass completion. The high seed-to-seed variance (0%–4%) means any future comparison (e.g. Row B: flat + dynamic spatial GNN) needs enough seeds to distinguish a real improvement from this baseline's own noise — 3 seeds should be treated as a floor, not a target.

**Do not treat this baseline as strong or stable.** It reflects real but weak, seed-sensitive, non-passing policies. Any claim of GNN improvement should be judged against this instability, not against the retracted 22%/37% figures from the superseded report.
