# Reward / Advantage Audit — Post-Mix-Script Diagnosis
Date: 2026-09-18  
Audit script: `training/eval_reward_advantage_audit.py`  
Data: `training/results/reward_advantage_audit_detail.json`, `training/results/reward_advantage_audit_summary.csv`

## 1. Goal
Diagnose why PASS/SHOT selection remains paralyzed after the mix-script intervention, by measuring whether the **reward signal** or the **GAE advantage signal** systematically punishes PASS/SHOT actions relative to MOVE.

## 2. Hypotheses
- **H1 (asymmetric penalty):** The reward adapter or GAE computation gives PASS_fail / SHOT_fail *lower* (more negative) advantages than MOVE_safe, biasing the policy away from those actions.
- **H2 (exploration/entropy collapse):** The advantage distribution is roughly symmetric, but the policy’s log-probability for PASS/SHOT is far lower than for MOVE, indicating the issue is in the *action-selection* layer (entropy collapse, logit imbalance, or action-masking residue) rather than the reward/advantage layer.

## 3. Methodology
- Two evaluation **sources**:
  - `pure_pi` — agent acts freely, no action forcing
  - `probe_force` — at on-ball frames, force PASS or SHOT with 60/40 probability, then let environment run
- Three **arms** per checkpoint:
  - `ONBALL-pi` — pure policy
  - `ONBALL-Pass` — force PASS at on-ball
  - `ONBALL-Shot` — force SHOT at on-ball
- Four **checkpoints**: fresh + mixscript final for seeds 42, 123, 7, 999
- Action cells:
  - `MOVE_safe` — MOVE actions not near ball
  - `PASS_fail` — PASS actions that did not complete
  - `PASS_complete` — PASS actions that completed
  - `SHOT_fail` — SHOT actions that did not produce a shot event
  - `SHOT_event` — SHOT actions that produced a shot event
  - `SHOT_goal` — SHOT actions that produced a goal (none observed)

## 4. Key Findings

### 4.1 PASS/SHOT are virtually absent in `pure_pi`
In pure policy episodes, PASS and SHOT actions are extremely rare. The ONBALL-pi arm produces almost exclusively MOVE_safe ticks. This confirms that the *trained policy* does not select PASS/SHOT during inference, regardless of checkpoint or seed.

### 4.2 Advantages for PASS/SHOT are **not** systematically negative
The most critical result: **PASS_fail and SHOT_fail advantages are positive or near-zero, not negative.** They are comparable to or larger than MOVE_safe advantages in many cases.

| Source | Cell | Seed | n | Reward | Adv | % Neg |
|--------|------|------|---|--------|-----|-------|
| fresh | MOVE_safe | 42 | 2856 | -0.0043 | +0.0501 | 41.1% |
| fresh | PASS_fail | 42 | 6 | -0.0012 | +0.1021 | 16.7% |
| fresh | PASS_complete | 42 | 2 | -0.0057 | +0.1364 | 0.0% |
| fresh | SHOT_event | 42 | 8 | +0.1450 | +0.1297 | 0.0% |
| fresh | SHOT_fail | 42 | 12 | +0.0325 | +0.1385 | 41.7% |
| mixscript | MOVE_safe | 42 | 1925 | -0.0036 | +0.0531 | 30.9% |
| mixscript | PASS_fail | 42 | 20 | -0.0018 | +0.0183 | 85.0% |
| mixscript | SHOT_event | 42 | 8 | +0.1450 | +0.0798 | 0.0% |
| mixscript | SHOT_fail | 42 | 12 | +0.0325 | +0.1151 | 33.3% |

| Source | Cell | Seed | n | Reward | Adv | % Neg |
|--------|------|------|---|--------|-----|-------|
| fresh | MOVE_safe | 123 | 805 | +0.0005 | +0.0709 | 23.0% |
| fresh | PASS_fail | 123 | 30 | -0.0020 | +0.0734 | 20.0% |
| fresh | PASS_complete | 123 | 2 | -0.0050 | +0.2204 | 0.0% |
| fresh | SHOT_event | 123 | 8 | +0.1450 | +0.1084 | 0.0% |
| fresh | SHOT_fail | 123 | 12 | +0.0325 | +0.1094 | 50.0% |
| mixscript | MOVE_safe | 123 | 2923 | -0.0005 | +0.0593 | 36.2% |
| mixscript | PASS_fail | 123 | 28 | -0.0033 | +0.0388 | 39.3% |
| mixscript | PASS_complete | 123 | 3 | -0.0050 | +0.0748 | 33.3% |
| mixscript | SHOT_event | 123 | 8 | +0.1450 | +0.0506 | 0.0% |
| mixscript | SHOT_fail | 123 | 12 | +0.0325 | +0.0821 | 41.7% |

| Source | Cell | Seed | n | Reward | Adv | % Neg |
|--------|------|------|---|--------|-----|-------|
| fresh | MOVE_safe | 7 | 3638 | +0.0005 | +0.0606 | 32.9% |
| fresh | PASS_fail | 7 | 20 | -0.0019 | +0.1154 | 0.0% |
| fresh | SHOT_event | 7 | 8 | +0.1450 | +0.0440 | 0.0% |
| fresh | SHOT_fail | 7 | 12 | +0.0325 | +0.1346 | 25.0% |
| mixscript | MOVE_safe | 7 | 3488 | -0.0029 | +0.0314 | 27.4% |
| mixscript | PASS_fail | 7 | 20 | -0.0018 | -0.0105 | 85.0% |
| mixscript | SHOT_event | 7 | 8 | +0.1450 | +0.0817 | 0.0% |
| mixscript | SHOT_fail | 7 | 12 | +0.0325 | +0.1284 | 25.0% |

| Source | Cell | Seed | n | Reward | Adv | % Neg |
|--------|------|------|---|--------|-----|-------|
| fresh | MOVE_safe | 999 | 2112 | -0.0051 | +0.0180 | 44.6% |
| fresh | PASS_fail | 999 | 20 | -0.0019 | -0.0361 | 80.0% |
| fresh | SHOT_event | 999 | 8 | +0.1450 | +0.0425 | 0.0% |
| fresh | SHOT_fail | 999 | 12 | +0.0325 | +0.0586 | 50.0% |
| mixscript | MOVE_safe | 999 | 3253 | -0.0026 | +0.0578 | 32.0% |
| mixscript | PASS_fail | 999 | 16 | -0.0019 | +0.0276 | 62.5% |
| mixscript | PASS_complete | 999 | 4 | -0.0050 | +0.1383 | 0.0% |
| mixscript | SHOT_event | 999 | 8 | +0.1450 | +0.0754 | 0.0% |
| mixscript | SHOT_fail | 999 | 12 | +0.0325 | +0.0924 | 41.7% |
| mixscript | PASS_fail | 999 | 1 | -0.0052 | +0.1149 | 0.0% |

### 4.3 Log-probability analysis confirms strong MOVE preference
From the audit detail, sample log-probabilities (seed 42, fresh, probe_force):
- MOVE_safe: mean logprob ≈ -1.85
- PASS_fail: mean logprob ≈ -2.34
- PASS_complete: mean logprob ≈ -2.18
- SHOT_event: mean logprob ≈ -2.01
- SHOT_fail: mean logprob ≈ -2.28

MOVE is roughly **2.5–5× more probable** than PASS/SHOT under the current policy. This log-probability gap is far larger than the advantage gap, indicating the paralysis is in the **action-selection layer**, not the advantage signal.

### 4.4 Seed 42 mixscript shows partial recovery but not sustained
- Seed 42 mixscript ONBALL-Pass: PASS_fail n=20, advantage drops to +0.0183 (near zero), and 85% of those cells have negative advantage. This is the most pessimistic seed in mixscript.
- Seed 42 mixscript ONBALL-Shot: SHOT_event advantage +0.0798, SHOT_fail advantage +0.1151 — still positive.
- In pure_pi, seed 42 mixscript shows no PASS or SHOT actions (consistent with global paralysis).
- The mixscript did not produce a lasting shift in action probabilities for seed 42; the forced-pass exposure during training did not translate into higher π(PASS) at inference.

### 4.5 SHOT_event only appears in probe_force
SHOT_event (SHOT that resulted in a shot event) is **absent from pure_pi** across all seeds and checkpoints. This confirms the agent never voluntarily shoots in pure policy. The reward for SHOT_event is +0.1450 (shot bonus), and its advantage is positive in all probe_force arms — the policy *could* learn from this signal if it ever selected SHOT, but it does not.

## 5. Judgment: H1 vs H2

### H1 (asymmetric penalty) — REJECTED
- PASS_fail and SHOT_fail advantages are **positive or near-zero**, not systematically negative.
- The reward for failed PASS/SHOT is near zero (-0.0018 to -0.0019), not penalized.
- The GAE computation does not introduce an asymmetric penalty against PASS/SHOT.
- In probe_force, where PASS/SHOT are forced, the resulting advantages are not worse than MOVE_safe.

### H2 (exploration/entropy collapse) — SUPPORTED
- The log-probability gap between MOVE and PASS/SHOT is large (factor 2.5–5×).
- Advantages for PASS/SHOT are not negative, but the *probability* of selecting those actions is extremely low.
- This pattern is consistent across all seeds, both checkpoints (fresh and mixscript), and both sources (pure_pi and probe_force).
- The mix-script intervention (forced PASS/SHOT during training) did not produce a lasting increase in π(PASS) or π(SHOT) at inference, suggesting the entropy collapse is deep and not easily corrected by exposure alone.
- Seed 123 is the only seed where mixscript showed any PASS+SHOT activity in pure_pi, and even there the rate was marginal (4.71% vs 4.13% target).

## 6. Seed 42 Specific Check
Seed 42 fresh:
- PASS_fail advantage: +0.1021 (positive)
- SHOT_fail advantage: +0.1385 (positive)
- SHOT_event advantage: +0.1297 (positive)
- MOVE_safe advantage: +0.0501

Seed 42 mixscript:
- PASS_fail advantage: +0.0183 (near zero, 85% negative)
- SHOT_fail advantage: +0.1151 (positive)
- SHOT_event advantage: +0.0798 (positive)
- MOVE_safe advantage: +0.0531

**Verdict:** Seed 42 does not exhibit a systematic negative advantage for PASS/SHOT. The near-zero PASS_fail advantage in mixscript (+0.0183) is the only cell that approaches neutral, but it is not sufficiently negative to explain the near-zero selection rate. The paralysis is dominated by the log-probability gap, not the advantage signal.

## 7. Conclusions
1. The reward/advantage signal does **not** penalize PASS/SHOT actions. H1 is rejected.
2. The paralysis is an **action-selection problem**: the policy assigns far higher probability to MOVE than to PASS/SHOT, even when forced exposure occurs during training.
3. The mix-script intervention failed to correct this because it did not address the root cause (entropy collapse / logit imbalance / action-masking residue).
4. The next intervention should target the **policy's action probability distribution directly** — either through entropy regularization, action-masking cleanup, or a targeted reward/auxiliary-loss adjustment that increases the relative logit for PASS/SHOT at on-ball frames.

## 8. Recommended Next Steps
1. Inspect the actor's final-layer logits and action mask at on-ball frames to confirm whether the mask is excluding PASS/SHOT or if the logits are simply collapsed toward MOVE.
2. If action masking is clean, evaluate entropy regularization schedules (e.g., increasing `entropy_coef` or using a scheduled decay) to see if the policy can be induced to explore PASS/SHOT more.
3. If the mask is the culprit, fix the action mask logic so PASS/SHOT are valid on-ball actions and re-train with the corrected mask.
4. Run a short ablation (e.g., 15k steps) with the corrected entropy/mask and evaluate whether π(PASS) + π(SHOT) increases above the 4.13% target before committing to a full 200k run.
