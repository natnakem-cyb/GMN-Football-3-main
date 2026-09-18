# Exploration Ablation Findings — Form A: Targeted On-Ball Entropy Bonus
Date: 2026-09-18
HEAD: dc2718d7fbf50e4a17effd9553023b0bdad4d812

## 1. Protocol Summary

- **Mechanism:** Targeted entropy bonus on legal PASS/SHOT marginal during Phase 1 (0–15k), alpha=0.50
- **Phase 1:** 0–15k steps, bonus ON
- **Phase 2:** 15k–30k steps, bonus OFF (unscripted tail)
- **Checkpoints:** 5k, 10k, 15k, 20k, 25k, 30k for seeds 42, 123, 7, 999
- **Baseline:** fresh-training final (50k) for each seed
- **Evaluation:** pure-π, deterministic, 20 episodes/checkpoint/seed, no forcing

## 2. Full Trajectory Table (Phase 1 context, non-evidentiary)

> **Note:** Phase 1 numbers are diagnostic-only context per protocol. Success is judged on Phase 2 tail (30k) only.

| Checkpoint | Seed | n_ep | π(PASS) | π(SHOT) | π(MOVE) | Football Rate (legal) | H(π) proxy | PASS+SHOT | Pass Completed | Shot Events | Goals | Mean Reward |
|------------|------|------|---------|---------|---------|----------------------|------------|-----------|----------------|-------------|-------|-------------|
| 5k (P1) | 42 | 20 | 0.0000 | 0.0000 | 0.4980 | 0.0000 | — | 0.0000 | 0 | 0 | 0 | -0.7208 |
| 5k (P1) | 123 | 20 | 0.0167 | 0.0059 | 0.8725 | 0.0204 | — | 0.0225 | 9 | 0 | 0 | -0.5423 |
| 5k (P1) | 7 | 20 | 0.0088 | 0.0000 | 0.9490 | 0.0051 | — | 0.0088 | 0 | 0 | 0 | -0.5057 |
| 5k (P1) | 999 | 20 | 0.0098 | 0.0039 | 0.5667 | 0.0185 | — | 0.0137 | 0 | 0 | 0 | -0.7109 |
| 10k (P1) | 42 | 20 | 0.0000 | 0.0000 | 0.0196 | 0.0000 | — | 0.0000 | 0 | 0 | 0 | -0.8528 |
| 10k (P1) | 123 | 20 | 0.0196 | 0.0000 | 0.4676 | 0.0159 | — | 0.0196 | 14 | 0 | 0 | -0.5333 |
| 10k (P1) | 7 | 20 | 0.0000 | 0.0000 | 0.9922 | 0.0000 | — | 0.0000 | 0 | 0 | 0 | -0.4368 |
| 10k (P1) | 999 | 20 | 0.0069 | 0.0020 | 0.8961 | 0.0049 | — | 0.0088 | 3 | 0 | 0 | -0.8006 |
| 15k (P1) | 42 | 20 | 0.0010 | 0.0000 | 0.4392 | 0.0005 | — | 0.0010 | 1 | 0 | 0 | -0.7345 |
| 15k (P1) | 123 | 20 | 0.0039 | 0.0029 | 0.1647 | 0.0067 | — | 0.0069 | 2 | 0 | 0 | -0.4641 |
| 15k (P1) | 7 | 20 | 0.0000 | 0.0088 | 0.9167 | 0.0046 | — | 0.0088 | 0 | 0 | 0 | -0.4770 |
| 15k (P1) | 999 | 20 | 0.0069 | 0.0055 | 0.9227 | 0.0073 | — | 0.0124 | 2 | 2 | 2 | -0.5310 |
| 20k (P2) | 42 | 20 | 0.0000 | 0.0000 | 0.4657 | 0.0000 | — | 0.0000 | 0 | 0 | 0 | -0.7857 |
| 20k (P2) | 123 | 20 | 0.0029 | 0.0030 | 0.2231 | 0.0039 | — | 0.0059 | 0 | 1 | 0 | -1.0523 |
| 20k (P2) | 7 | 20 | 0.0000 | 0.0186 | 0.8804 | 0.0100 | — | 0.0186 | 0 | 0 | 0 | -0.4946 |
| 20k (P2) | 999 | 20 | 0.0020 | 0.0125 | 0.8692 | 0.0088 | — | 0.0145 | 1 | 5 | 4 | -0.4690 |
| 25k (P2) | 42 | 20 | 0.0000 | 0.0000 | 0.8088 | 0.0000 | — | 0.0000 | 0 | 0 | 0 | -0.8398 |
| 25k (P2) | 123 | 20 | 0.0010 | 0.0029 | 0.1441 | 0.0022 | — | 0.0039 | 0 | 0 | 0 | -0.6566 |
| 25k (P2) | 7 | 20 | 0.0000 | 0.0196 | 0.8637 | 0.0109 | — | 0.0196 | 2 | 0 | 0 | -0.4974 |
| 25k (P2) | 999 | 20 | 0.0039 | 0.0050 | 0.8661 | 0.0061 | — | 0.0089 | 2 | 2 | 2 | -0.7446 |
| 30k (P2) | 42 | 20 | 0.0010 | 0.0000 | 0.8363 | 0.0005 | — | 0.0010 | 0 | 0 | 0 | -0.8288 |
| 30k (P2) | 123 | 20 | 0.0010 | 0.0068 | 0.0904 | 0.0048 | — | 0.0078 | 1 | 4 | 3 | -0.6045 |
| 30k (P2) | 7 | 20 | 0.0000 | 0.0196 | 0.8569 | 0.0110 | — | 0.0196 | 2 | 0 | 0 | -0.4990 |
| 30k (P2) | 999 | 20 | 0.0029 | 0.0048 | 0.8905 | 0.0056 | — | 0.0078 | 1 | 4 | 3 | -0.6027 |
| fresh 50k (baseline) | 42 | 20 | 0.0059 | 0.0000 | 0.4725 | 0.0044 | — | 0.0059 | 4 | 0 | 0 | -0.7614 |
| fresh 50k (baseline) | 123 | 20 | 0.0245 | 0.0000 | 0.2608 | 0.3709 | — | 0.0245 | 12 | 0 | 0 | -0.4230 |
| fresh 50k (baseline) | 7 | 20 | 0.0000 | 0.0000 | 0.9853 | 0.0000 | — | 0.0000 | 0 | 0 | 0 | -0.4530 |
| fresh 50k (baseline) | 999 | 20 | 0.0000 | 0.0000 | 0.3657 | 0.0000 | — | 0.0000 | 0 | 0 | 0 | -0.4795 |

## 3. Phase 2 Tail-End Gate Judgment (30k, primary criterion)

### Primary Criterion: ≥3/4 seeds exceed fresh baseline by ≥1.5pp on PASS+SHOT combined rate

| Seed | Fresh Baseline (PASS+SHOT) | 30k Ablation (PASS+SHOT) | Δ | Meets ≥1.5pp? |
|------|---------------------------|--------------------------|---|---------------|
| 42 | 0.0059 | 0.0010 | -0.0049 | NO |
| 123 | 0.0245 | 0.0078 | -0.0167 | NO |
| 7 | 0.0000 | 0.0196 | +0.0196 | YES |
| 999 | 0.0000 | 0.0078 | +0.0078 | NO |

**Primary criterion result: 1/4 seeds meet the ≥1.5pp threshold.**

### Secondary Criterion: ≥2/4 seeds show non-forced pass_completed + shot events on tail

| Seed | pass_completed | shot_events | goals | Meets both? |
|------|----------------|-------------|-------|--------------|
| 42 | 0 | 0 | 0 | NO |
| 123 | 1 | 4 | 3 | YES |
| 7 | 2 | 0 | 0 | NO |
| 999 | 1 | 4 | 3 | YES |

**Secondary criterion result: 2/4 seeds meet both event thresholds.**

### Guard Criterion (Reversion Check): Any seed below own fresh baseline?

| Seed | Fresh Baseline (PASS+SHOT) | 30k Ablation (PASS+SHOT) | Reverted? |
|------|---------------------------|--------------------------|-----------|
| 42 | 0.0059 | 0.0010 | YES — below baseline |
| 123 | 0.0245 | 0.0078 | YES — below baseline |
| 7 | 0.0000 | 0.0196 | NO |
| 999 | 0.0000 | 0.0078 | NO |

## 4. Phase 1 Context (Diagnostic Only)

Phase 1 (0–15k) shows whether the entropy bonus mechanism was active.

| Checkpoint | Seed | π(PASS) | π(SHOT) | Football Rate (legal) | PASS+SHOT |
|------------|------|---------|---------|----------------------|-----------|
| 5k | 42 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 5k | 123 | 0.0167 | 0.0059 | 0.0204 | 0.0225 |
| 5k | 7 | 0.0088 | 0.0000 | 0.0051 | 0.0088 |
| 5k | 999 | 0.0098 | 0.0039 | 0.0185 | 0.0137 |
| 10k | 42 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 10k | 123 | 0.0196 | 0.0000 | 0.0159 | 0.0196 |
| 10k | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 10k | 999 | 0.0069 | 0.0020 | 0.0049 | 0.0088 |
| 15k | 42 | 0.0010 | 0.0000 | 0.0005 | 0.0010 |
| 15k | 123 | 0.0039 | 0.0029 | 0.0067 | 0.0069 |
| 15k | 7 | 0.0000 | 0.0088 | 0.0046 | 0.0088 |
| 15k | 999 | 0.0069 | 0.0055 | 0.0073 | 0.0124 |

## 5. Collapse / Recovery Shape

| Seed | 15k PASS+SHOT | 30k PASS+SHOT | 50k Baseline PASS+SHOT | Phase 2 Δ | Phase 1→2 Δ |
|------|---------------|---------------|------------------------|-----------|-------------|
| 42 | 0.0010 | 0.0010 | 0.0059 | +0.0000 | -0.0049 |
| 123 | 0.0069 | 0.0078 | 0.0245 | +0.0009 | -0.0167 |
| 7 | 0.0088 | 0.0196 | 0.0000 | +0.0108 | +0.0196 |
| 999 | 0.0124 | 0.0078 | 0.0000 | -0.0046 | +0.0078 |

## 6. Overall Judgment

**Judgment: ABLATION INSUFFICIENT**

Primary criterion failed (1/4 seeds ≥1.5pp improvement). Secondary criterion: 2/4 seeds with both event types.

**Guard violations:** Seeds 42, 123 fell below their own fresh baseline on the Phase 2 tail.

## 7. Next Recommendation

The ablation did not produce sufficient improvement on the Phase 2 tail. Recommend either (a) increasing alpha, (b) extending Phase 2 duration, or (c) trying a count-based exploration reward instead of an entropy bonus.

CONFIRMATIONS
----------------------------------------
  No reward/GAE/mask/network/entropy changes: yes
  Evaluation-only run, no training performed: yes
  Prior result files untouched: yes

FILES WRITTEN
----------------------------------------
  - training/results/exploration_ablation_summary.csv
  - training/results/EXPLORATION_ABLATION_FINDINGS.md
