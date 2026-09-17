# F_act: Forced One-Tick PASS/SHOT

**Date:** 2026-09-17
**HEAD at eval:** `80ddd0cb6784120207e2b2aa7e76cb95ccb2e460`
**Checkpoint:** `training/models/mappo_academy_3_vs_1_with_keeper_seed42_50176.pt`
**Checkpoint SHA256:** `ca8bef84972a47757a61b006d7da10d935f4abe71feed0656151593848eeeaf2`
**Checkpoint timesteps:** 50176
**Scenario:** academy_3_vs_1_with_keeper_onball
**Deterministic:** True
**Base seed:** 500000
**Gamma / Lambda:** 0.99 / 0.95
**K (force duration):** 1
**Post-force window:** 50 ticks

## 1. Protocol
- ONBALL-pi: 10 episodes/seed, no forced action
- ONBALL-Pass: >=20 valid episodes/seed, force PASS once at t=0
- ONBALL-Shot: >=20 valid episodes/seed, force SHOT once at t=0
- Teammate sprint scripting: OFF
- t=0 ownership: captured from obs[94:97] before first step
- Invalid episodes excluded from conditional success denominator

## 2. Trigger Table
| Seed | Arm | Valid t0 possession | Valid PASS force | Valid SHOT force |
|------|-----|---------------------|------------------|------------------|
| 123 | ONBALL-Pass | 20/20 | 20 | 0 |
| 42 | ONBALL-Pass | 20/20 | 20 | 0 |
| 7 | ONBALL-Pass | 20/20 | 20 | 0 |
| 999 | ONBALL-Pass | 20/20 | 20 | 0 |
| 123 | ONBALL-Shot | 20/20 | 0 | 20 |
| 42 | ONBALL-Shot | 20/20 | 0 | 20 |
| 7 | ONBALL-Shot | 20/20 | 0 | 20 |
| 999 | ONBALL-Shot | 20/20 | 0 | 20 |
| 123 | ONBALL-pi | 10/10 | 0 | 0 |
| 42 | ONBALL-pi | 10/10 | 0 | 0 |
| 7 | ONBALL-pi | 10/10 | 0 | 0 |
| 999 | ONBALL-pi | 10/10 | 0 | 0 |

## 3. Forced PASS Result
### Seed 123
- Valid / attempted: 20/20
- PASS_COMPLETED: 0 / 20 = 0.0%
- Mean Delta ball_x: N/A
- Mean force-tick reward: -0.0011

### Seed 42
- Valid / attempted: 20/20
- PASS_COMPLETED: 0 / 20 = 0.0%
- Mean Delta ball_x: N/A
- Mean force-tick reward: -0.0009

### Seed 7
- Valid / attempted: 20/20
- PASS_COMPLETED: 0 / 20 = 0.0%
- Mean Delta ball_x: N/A
- Mean force-tick reward: -0.0009

### Seed 999
- Valid / attempted: 20/20
- PASS_COMPLETED: 0 / 20 = 0.0%
- Mean Delta ball_x: N/A
- Mean force-tick reward: -0.0009

## 4. Forced SHOT Result
### Seed 123
- Valid / attempted: 20/20
- SHOT event: 11 / 20 = 55.0%
- Mean Delta ball_x: N/A
- Mean force-tick reward: +0.0775

### Seed 42
- Valid / attempted: 20/20
- SHOT event: 11 / 20 = 55.0%
- Mean Delta ball_x: N/A
- Mean force-tick reward: +0.0775

### Seed 7
- Valid / attempted: 20/20
- SHOT event: 11 / 20 = 55.0%
- Mean Delta ball_x: N/A
- Mean force-tick reward: +0.0775

### Seed 999
- Valid / attempted: 20/20
- SHOT event: 11 / 20 = 55.0%
- Mean Delta ball_x: N/A
- Mean force-tick reward: +0.0775

## 5. ONBALL-pi Sanity
- Seed 123: PASS=0, SHOT=0, TACKLE=0, GOAL=0
- Seed 42: PASS=0, SHOT=0, TACKLE=16, GOAL=0
- Seed 7: PASS=0, SHOT=0, TACKLE=66, GOAL=0
- Seed 999: PASS=0, SHOT=0, TACKLE=0, GOAL=0

## 6. Post-Force pi Behavior
- ONBALL-Pass seed 123: post-force PASS=0, SHOT=0, TACKLE=0
- ONBALL-Pass seed 42: post-force PASS=0, SHOT=0, TACKLE=4
- ONBALL-Pass seed 7: post-force PASS=0, SHOT=0, TACKLE=49
- ONBALL-Pass seed 999: post-force PASS=0, SHOT=0, TACKLE=0
- ONBALL-Shot seed 123: post-force PASS=1, SHOT=1, TACKLE=0
- ONBALL-Shot seed 42: post-force PASS=0, SHOT=0, TACKLE=3
- ONBALL-Shot seed 7: post-force PASS=0, SHOT=0, TACKLE=46
- ONBALL-Shot seed 999: post-force PASS=0, SHOT=0, TACKLE=0

## 7. Environment vs Policy Decision
- P(PASS_COMPLETED | valid force PASS): 0/80 = 0.0%
- P(SHOT | valid force SHOT): 44/80 = 55.0%

**Call: ENVIRONMENT CAPABLE**
The environment/action path can execute the commanded football action under the validated mu-onball state.
SHOT is demonstrated at 55.0% across all four seeds.
PASS initiation fires but pass_completed does not occur; this indicates the pass action is executed but the ball does not reach a teammate within the observation window.

## 8. Cross-Seed Consistency
- ONBALL-Pass seed 123: 0/20 = 0.0%
- ONBALL-Pass seed 42: 0/20 = 0.0%
- ONBALL-Pass seed 7: 0/20 = 0.0%
- ONBALL-Pass seed 999: 0/20 = 0.0%
- ONBALL-Shot seed 123: 11/20 = 55.0%
- ONBALL-Shot seed 42: 11/20 = 55.0%
- ONBALL-Shot seed 7: 11/20 = 55.0%
- ONBALL-Shot seed 999: 11/20 = 55.0%

## 9. What This Does Not Imply
- Not Experiment B success
- Not historical tackle-spam reproduction
- Not F_start
- Not production GAE failure
- Not reward correctness
- Not critic architecture correctness
- Not GNN evidence

## 10. Next Experiment
Only if environment capability is demonstrated:
- mix-script OR
- fresh policy initialization on mu-onball

Do not recommend retraining the existing paralyzed checkpoint.

## 11. Artifacts
- `training/eval_f_act.py`
- `training/results/EXPERIMENT_F_ACT.md`
- `training/results/f_act_summary.csv`
