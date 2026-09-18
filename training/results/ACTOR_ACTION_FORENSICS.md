ACTOR ACTION FORENSICS REPORT
==================================================
HEAD:                         7ce61c00688ace1833af8267a0b3c1062dfd7957
Checkpoints audited:          mixscript_early (15k), fresh_final (50k), mixscript_final (50k)
Intermediate checkpoints available: yes — mixscript_early (15k) for all seeds
Seeds:                        42, 123, 7, 999
Scenario:                     academy_3_vs_1_with_keeper_onball
Episodes / on-ball frames:    8440 total on-ball frames
Act mode:                     deterministic

TASK 1 — MASK VERIFICATION
----------------------------------------
  seed42 mixscript_early (t=15104, n=746):
    P(PASS legal | on-ball):  1.000
    P(SHOT legal | on-ball):  1.000
    Mask defect suspected:    no
  seed42 fresh_final (t=50176, n=864):
    P(PASS legal | on-ball):  1.000
    P(SHOT legal | on-ball):  1.000
    Mask defect suspected:    no
  seed42 mixscript_final (t=50176, n=864):
    P(PASS legal | on-ball):  1.000
    P(SHOT legal | on-ball):  1.000
    Mask defect suspected:    no
  seed123 mixscript_early (t=15104, n=590):
    P(PASS legal | on-ball):  1.000
    P(SHOT legal | on-ball):  1.000
    Mask defect suspected:    no
  seed123 fresh_final (t=50176, n=208):
    P(PASS legal | on-ball):  0.990
    P(SHOT legal | on-ball):  0.990
    Mask defect suspected:    yes
  seed123 mixscript_final (t=50176, n=208):
    P(PASS legal | on-ball):  0.990
    P(SHOT legal | on-ball):  0.990
    Mask defect suspected:    yes
  seed7 mixscript_early (t=15104, n=829):
    P(PASS legal | on-ball):  1.000
    P(SHOT legal | on-ball):  1.000
    Mask defect suspected:    no
  seed7 fresh_final (t=50176, n=1017):
    P(PASS legal | on-ball):  1.000
    P(SHOT legal | on-ball):  1.000
    Mask defect suspected:    no
  seed7 mixscript_final (t=50176, n=1017):
    P(PASS legal | on-ball):  1.000
    P(SHOT legal | on-ball):  1.000
    Mask defect suspected:    no
  seed999 mixscript_early (t=15104, n=690):
    P(PASS legal | on-ball):  1.000
    P(SHOT legal | on-ball):  1.000
    Mask defect suspected:    no
  seed999 fresh_final (t=50176, n=683):
    P(PASS legal | on-ball):  1.000
    P(SHOT legal | on-ball):  1.000
    Mask defect suspected:    no
  seed999 mixscript_final (t=50176, n=683):
    P(PASS legal | on-ball):  1.000
    P(SHOT legal | on-ball):  1.000
    Mask defect suspected:    no
  GLOBAL mask defect found:  yes
  NOTE: rates are ≥99%; if any 'yes' entries correspond to very small
  on-ball frame counts, flag as minor edge-case, not root cause.

TASK 2 — ACTOR PIPELINE (on-ball, aggregate)
----------------------------------------
  Checkpoint                         n    π(MOVE)    π(PASS)    π(SHOT)       H(π)    Δ_PASS-MOVE    Δ_SHOT-MOVE
  seed42 mixscript_early        746     0.5404     0.1372     0.0444     2.7945        -0.6725        -0.9736
  seed42 fresh_final            864     0.5190     0.1300     0.0300     2.7455        -0.5390        -1.3865
  seed42 mixscript_final        864     0.5190     0.1300     0.0300     2.7455        -0.5390        -1.3865
  seed123 mixscript_early        590     0.5011     0.1592     0.0535     2.8250        -0.3051        -0.5974
  seed123 fresh_final            208     0.5545     0.1489     0.0348     2.7840        -0.4651        -1.2546
  seed123 mixscript_final        208     0.5545     0.1489     0.0348     2.7840        -0.4651        -1.2546
  seed7 mixscript_early        829     0.5748     0.1077     0.0532     2.7882        -0.9998        -0.7976
  seed7 fresh_final           1017     0.6084     0.0902     0.0416     2.7553        -1.3159        -1.2242
  seed7 mixscript_final       1017     0.6084     0.0902     0.0416     2.7553        -1.3159        -1.2242
  seed999 mixscript_early        690     0.4961     0.1711     0.0249     2.7793        -0.3370        -1.6609
  seed999 fresh_final            683     0.6027     0.1141     0.0166     2.7059        -0.9208        -2.2297
  seed999 mixscript_final        683     0.6027     0.1141     0.0166     2.7059        -0.9208        -2.2297

TASK 3 — SEED 123 COMPARISON
----------------------------------------
  t=15104: seed123 π(PASS)=0.1592 vs others avg=0.1387
          seed123 π(SHOT)=0.0535 vs others avg=0.0408
          seed123 H(π)=2.8250 vs others avg=2.7873
          seed123 Δ_PASS-MOVE=-0.3051 vs others avg=-0.6698
          seed123 Δ_SHOT-MOVE=-0.5974 vs others avg=-1.1441
  Interpretation: seed123 shows distinguishably better football-action logits vs 42/7/999
  t=50176: seed123 π(PASS)=0.1489 vs others avg=0.1198
          seed123 π(SHOT)=0.0348 vs others avg=0.0310
          seed123 H(π)=2.7840 vs others avg=2.7202
          seed123 Δ_PASS-MOVE=-0.4651 vs others avg=-0.8990
          seed123 Δ_SHOT-MOVE=-1.2546 vs others avg=-1.5519
  Interpretation: seed123 shows distinguishably better football-action logits vs 42/7/999

TASK 4 — CROSS-CHECKPOINT TRAJECTORY
----------------------------------------
  Data available: yes — mixscript_early (15k) -> mixscript_final (50k)
  Fresh intermediate checkpoints: no — only fresh_final (50k) exists

  seed42:
    mixscript_early  (15k): π(PASS)=0.1372, π(SHOT)=0.0444, H(π)=2.7945
    mixscript_final  (50k): π(PASS)=0.1300, π(SHOT)=0.0300, H(π)=2.7455
    Progressive collapse:    yes
  seed123:
    mixscript_early  (15k): π(PASS)=0.1592, π(SHOT)=0.0535, H(π)=2.8250
    mixscript_final  (50k): π(PASS)=0.1489, π(SHOT)=0.0348, H(π)=2.7840
    Progressive collapse:    yes
  seed7:
    mixscript_early  (15k): π(PASS)=0.1077, π(SHOT)=0.0532, H(π)=2.7882
    mixscript_final  (50k): π(PASS)=0.0902, π(SHOT)=0.0416, H(π)=2.7553
    Progressive collapse:    yes
  seed999:
    mixscript_early  (15k): π(PASS)=0.1711, π(SHOT)=0.0249, H(π)=2.7793
    mixscript_final  (50k): π(PASS)=0.1141, π(SHOT)=0.0166, H(π)=2.7059
    Progressive collapse:    yes

OVERALL JUDGMENT
----------------------------------------
  H2a (stable actor prior) vs H2b (entropy/exploration re-collapse): H2b-leaning with early movement baseline
  Evidence:
    All comparable seeds show progressive collapse in π(PASS)/π(SHOT) from 15k→50k, plus entropy decline. The movement prior is present at 15k, but it intensifies during training, which is the hallmark of H2b (entropy/exploration re-collapse).
    Mean π(PASS) across all on-ball frames: 0.1338
    Mean π(SHOT) across all on-ball frames: 0.0350
    Mean H(π) across all on-ball frames:    2.7506
    Mean Δ_PASS-MOVE:                       -0.6458
    Mean Δ_SHOT-MOVE:                       -1.3141
    Progressive collapse (15k→50k):         4/4 seeds
  Mask defect: yes — 0.3% of frames

CONFIRMATIONS
----------------------------------------
  No reward/GAE/mask/network/entropy changes: yes
  No training runs performed:                  yes
  Prior result files untouched:                yes

FILES WRITTEN
----------------------------------------
  - training/results/ACTOR_ACTION_FORENSICS.md
  - training/results/actor_forensics_summary.csv
  - training/results/actor_forensics_detail.json

NEXT INTERVENTION CLASS
----------------------------------------
  entropy-exploration intervention: progressive collapse documented across all seeds.
  Before changing entropy_coef, run a 15k ablation with a scheduled entropy bonus
  or count-based exploration reward targeting on-ball football actions, and measure
  whether π(PASS)+π(SHOT) stabilizes above the 4.13% target.
