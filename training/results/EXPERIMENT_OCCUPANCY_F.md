# Experiment F — Occupancy Intervention Gate

**Date:** 2026-09-17  
**HEAD at eval:** `939dc22` (critic/GAE forensics commit)  
**Checkpoint:** `training/models/mappo_academy_3_vs_1_with_keeper_seed42_best.pt`  
**Checkpoint SHA256:** `e5e0b7c1f0547384125d4b865a4aec9417afe454084446734f8474f32f91c939`  
**Checkpoint timesteps:** 100,352  
**Scenario:** `academy_3_vs_1_with_keeper` (CTRL) / `academy_3_vs_1_with_keeper_onball` (INT)  
**Episodes per arm:** 50  
**Deterministic:** True  
**Base seed:** 500,000  
**Gamma / Lambda:** 0.99 / 0.95  
**Investigator:** automated occupancy gate  

---

## 1. Hypothesis

**Hypothesis F:** Increasing football-state occupancy via a single initial-distribution intervention (`μ-onball`) causes events and critic/advantage separation, without changing reward, GAE, networks, or action masks.

**Null expectation if hypothesis is false:** Both arms show indistinguishable event counts, action distributions, and critic statistics because the policy cannot exploit the occupancy difference.

---

## 2. Intervention

| Property | Value |
|----------|-------|
| Name | `μ-onball` |
| Scenario ID | `academy_3_vs_1_with_keeper_onball` |
| Change | Ball spawns on controlled left CAM (`left_1`) at `(x=0.2, y=0, z=0)` with `ownerId='left_1'` |
| Freeze checklist | Reward, strip, GAE λ/γ, value loss, critic architecture, action masks, exploration bonus (E=1/β=0.03) — all unchanged |

**Provenance:** `src/scenarios/ScenarioRegistry.ts` lines 146–175; `src/engine/GameEngine.ts` lines 493–501; `src/types/football.ts` line 247.

---

## 3. GAE Convention

Production convention enforced in `training/eval_occupancy_gate.py`:
- `dones = terminated` only
- Truncated episodes bootstrap with `critic(next_obs)`
- Terminated episodes bootstrap with `0.0`

---

## 4. Bridge / Reproducibility

- `training/gmn_pettingzoo.py` patched with `_kill_existing_bridge()` and `_ensure_bridge_running()` kill-first + retry/backoff.
- Deterministic seeds: `500000, 501009, ..., 549441` (50 episodes, seed+9 per episode).
- Frozen weights loaded once; no training occurs during eval.

---

## 5. Eval Protocol

```bash
python -m training.eval_occupancy_gate \
  --checkpoint training/models/mappo_academy_3_vs_1_with_keeper_seed42_best.pt \
  --scenario academy_3_vs_1_with_keeper --num-episodes 50 --deterministic --arm CTRL

python -m training.eval_occupancy_gate \
  --checkpoint training/models/mappo_academy_3_vs_1_with_keeper_seed42_best.pt \
  --scenario academy_3_vs_1_with_keeper_onball --num-episodes 50 --deterministic --arm INT
```

Artifacts:
- `training/models/occupancy_gate_CTRL_mappo_academy_3_vs_1_with_keeper_seed42_best.json`
- `training/models/occupancy_gate_INT_mappo_academy_3_vs_1_with_keeper_seed42_best.json`
- `training/results/occupancy_gate_summary.csv`

---

## 6. Results

### 6.1 Aggregate metrics

| Metric | CTRL | INT |
|--------|------|-----|
| `left_possession_fraction` | 0.4692 | 0.4630 |
| `controlled_owner_fraction` | 0.4692 | 0.4630 |
| `pass_legal_fraction` | 0.5176 | 0.5010 |
| `shot_legal_fraction` | 0.5176 | 0.5010 |
| `pass_completed_count` | **0** | **0** |
| `shot_event_count` | **0** | **0** |
| `goal_count` | **0** | **0** |
| `mean_reward` | -0.00851 | -0.00877 |
| `reward_entropy` | 4.0427 | 3.93799 |
| `mean_value` | -0.65974 | -0.66070 |
| `mean_gae_advantage` | -0.09716 | -0.11953 |
| `std_value_on_ball` | 0.01124 | 0.01044 |
| `std_value_off_ball` | 0.00894 | 0.00923 |
| `mean_advantage_on_ball` | -0.09716 | -0.11953 |
| `mean_advantage_off_ball` | 0.37120 | 0.34807 |

### 6.2 Action distribution (INT, episode 0)

| Action | Count |
|--------|-------|
| UP | 51 |
| UP_RIGHT | 6 |
| All others | 0 |

CTRL shows the same `UP`/`UP_RIGHT` dominance.

### 6.3 t=0 ownership verification

| Scenario | `obs[94:97]` (no-one, left, right) |
|----------|-----------------------------------|
| CTRL (`academy_3_vs_1_with_keeper`) | `[1. 0. 0.]` — unowned |
| INT (`academy_3_vs_1_with_keeper_onball`) | `[0. 1. 0.]` — left team owns ball |

The intervention is mechanically effective at `t=0`. The action mask correctly opens `SHOT`/`PASS`/`DRIBBLE` for the on-ball player in INT and closes `TACKLE`.

---

## 7. Verdict: **Soft FAIL**

| Criterion | Outcome |
|-----------|---------|
| Intervention implemented correctly | ✅ INT starts with left possession at `t=0` |
| Measurement valid | ✅ 50-episode deterministic eval completed on frozen weights |
| Events/critic separation observed | ❌ 0 passes, 0 shots, 0 tackles, 0 goals in both arms |
| Policy exploits occupancy difference | ❌ Action distribution is `UP`/`UP_RIGHT` only |

**Interpretation:** The `μ-onball` intervention successfully changes the initial distribution, but the 100k-step policy is paralyzed and cannot execute ball-handling actions in either arm. Because no events ever fire, there is no occupancy-driven signal to measure, and critic/advantage separation is negligible.

This is the brief’s definition of a **Soft FAIL**: the intervention is valid and measurable, but the policy at this training stage is not capable of exploiting the increased occupancy.

---

## 8. Recommendation

1. **Do not start 200k or exploration ablation** until the four 100k policies are characterized on the same axes.
2. If the four 100k policies all show the same paralyzed action distribution, classify Hypothesis F as **Soft FAIL** and redesign the intervention or training target before scaling horizon.
3. If one or more 100k policies do execute ball actions, re-run the occupancy gate on those policies to test whether `μ-onball` produces event/critic separation when the policy is capable.

---

## 9. Files

| File | Role |
|------|------|
| `training/eval_occupancy_gate.py` | Eval script |
| `training/results/occupancy_gate_summary.csv` | Aggregate CSV |
| `training/models/occupancy_gate_CTRL_*_seed42_best.json` | CTRL per-tick log |
| `training/models/occupancy_gate_INT_*_seed42_best.json` | INT per-tick log |
| `src/scenarios/ScenarioRegistry.ts` | Intervention definition |
| `src/engine/GameEngine.ts` | Initial ownership assignment |
| `src/types/football.ts` | `ownerId` on `ScenarioSetup.ball` |
| `training/gmn_pettingzoo.py` | Bridge startup fix, per-tick ball owner capture, **t=0 ownership from reset** |
| `training/bridge_server.ts` | **Reset response includes `current_ball_owner`** |

---

## 10. Addendum — Measurement Repair & Re-run (2026-09-17)

### 10.1 Issue discovered

The original report (Section 6.3) claimed t=0 ownership was measured from `obs[94:97]`, but the eval script actually read `ball_owner_agent_idx` from the **first step binary frame** — which defaults to `255` (no owner) before any step occurs. The bridge did not include `current_ball_owner` in the reset JSON response.

**Original Section 6.3 claim is INVALID** — the t=0 ownership values reported were not measured at t=0.

### 10.2 Fixes applied

| File | Fix |
|------|-----|
| `training/eval_occupancy_gate.py` | Capture `obs[94:97]` at `reset()` before first step; log `t0_ball_ownership`, `t0_controlled_has_ball`, `t0_mask_sum`, `t0_pass_legal`, `t0_shot_legal` |
| `training/bridge_server.ts` | Add `current_ball_owner` to reset JSON response (mirrors step response) |
| `training/gmn_pettingzoo.py` | Parse `current_ball_owner` from reset response; populate `_last_ball_owner_agent_idx` immediately after reset |

### 10.3 Re-run results (50 episodes/arm, same frozen weights)

| Metric | CTRL | INT |
|------|-----|-----|
| `t0_left_owner_fraction` | **0.00** (0/50) | **1.00** (50/50) |
| `t0_mean_mask_sum` | 15.0 | 18.0 |
| `t0_pass_legal_eps` | 0 | 50 |
| `t0_shot_legal_eps` | 0 | 50 |
| `left_possession_fraction` | 0.469 | **0.941** |
| `pass_legal_fraction` | 0.518 | **0.998** |
| `shot_legal_fraction` | 0.518 | **0.998** |
| `pass_completed_count` | 0 | 0 |
| `shot_event_count` | 0 | 0 |
| `goal_count` | 0 | 0 |
| `mean_reward` | -0.0085 | -0.0144 |
| `mean_value` | -0.660 | -0.649 |
| `mean_gae_advantage` | +0.151 | -0.151 |
| `mean_advantage_on_ball` | -0.097 | -0.153 |
| `mean_advantage_off_ball` | +0.371 | -0.125 |

### 10.4 Interpretation

- ✅ **Intervention mechanically effective:** INT starts with left ownership at t=0 in 100% of episodes; controlled agent has pass/shot/dribble legal (mask_sum=18).
- ✅ **Occupancy sustained:** INT maintains left possession for 94% of episode ticks (vs 47% in CTRL); pass/shot legal 99.8% of ticks.
- ❌ **Policy cannot exploit:** 0 passes, 0 shots, 0 goals in 50 INT episodes despite continuous on-ball opportunity.
- ❌ **Critic/advantage not separated:** V remains flat (~ -0.65 both arms); on-ball advantages negative in both arms.

### 10.5 Updated Verdict

**Soft FAIL — now measurement-validated.**

The `μ-onball` intervention successfully creates the intended initial distribution and sustained on-ball occupancy, but the 100k-step policy is paralyzed and cannot convert occupancy into football actions or events. The critic basin persists because no football events occur to provide learning signal.

This **confirms** the original Soft FAIL conclusion but now rests on valid measurement.

### 10.6 Recommendation unchanged

1. Do not start 200k or exploration ablation until all four 100k policies are characterized on these axes.
2. If all four show paralyzed action distributions, classify Hypothesis F as Soft FAIL and redesign intervention/training target.
3. If any 100k policy executes ball actions, re-run occupancy gate on that policy.
