# Fresh Training Protocol — µ-onball, Repaired PASS Path

**Pre-registered:** 2026-09-18  
**HEAD:** `05c95eb` (docs: freeze PASS semantics + frozen-pi re-eval)  
**Scenario:** `academy_3_vs_1_with_keeper_onball`  
**Seeds:** 42, 123, 7, 999  
**Total timesteps:** 50,176 per seed  
**Checkpoint naming:** `mappo_academy_3_vs_1_with_keeper_seed{N}_freshtrain_50176.pt`

---

## 1. Scope and Constraints

- **No reward changes.** Reward function, GAE γ/λ, action masks are frozen.
- **No PASS/SHOT mechanics changes.** Nearest-teammate PASS resolution is already on `main`.
- **No option (b)** (passing-lane aiming). No tackle special-casing.
- **No mix-script / scripted assistance.**
- **No warm-start.** Fresh initialization from scratch.
- **Frozen configuration logged** at start of each run: reward function hash/version, GAE γ=0.99 λ=0.95, mask configuration.

---

## 2. Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | MAPPO (SharedActor + CentralizedCritic) |
| Architecture | MLP 64×64 |
| Rollout length | 256 |
| Mini-batch size | 256 |
| PPO epochs | 4 |
| Learning rate | 3e-4 (cosine anneal to 3e-5) |
| Clip range | 0.15 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0.01 → 0.005 (linear decay) |
| Max grad norm | 0.5 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Opponent difficulty | medium |
| Self-play | disabled |
| Curriculum | disabled |

---

## 3. Success Criteria (Pre-Registered)

### Primary Criterion

**Post-training PASS+SHOT combined selection rate must exceed the Phase B frozen-π baseline (0.06% + 0.09% = 0.15% combined) by at least an order of magnitude (≥1.5%) in at least 3 of the 4 seeds independently.**

This threshold is evaluated per-seed on the pure-π (no force) evaluation, not on the pooled aggregate. The 1.5% floor is set at 10× the Phase B baseline combined rate. Justification:

- Phase B showed near-zero spontaneous PASS/SHOT selection (2 PASS + 3 SHOT across 3,416 ticks = 0.15%).
- Per-seed variance was large (some seeds had 0 PASS and 0 SHOT; others had 1 each), so a pooled-only bar would mask seed-level failures.
- An order-of-magnitude improvement is the minimum signal that indicates genuine learning rather than noise.
- 1.5% corresponds to roughly 1 PASS or SHOT selection per 67 ticks, which is a sparse but detectable preference change.

**Pass/fail:** A seed passes the primary criterion if its post-training PASS+SHOT combined selection rate ≥ 1.5%. The experiment passes overall if ≥ 3 of 4 seeds pass individually.

### Secondary Criterion

**At least one non-forced `pass_completed` event and at least one non-forced `shot` event must occur under pure π control in at least 2 of 4 seeds.**

This is a behavioral floor: it confirms the learned policy can actually execute football actions end-to-end, not just select the action category. Goals are not required.

**Pass/fail:** A seed passes the secondary criterion if it produces ≥1 `pass_completed` and ≥1 `shot` event in its 20-episode pure-π evaluation. The experiment passes overall if ≥ 2 of 4 seeds pass individually.

---

## 4. Evaluation Protocol

Reuse the Phase B evaluation exactly:

- **Script:** `training/eval_f_act.py` with arm `ONBALL-pi`
- **Scenario:** `academy_3_vs_1_with_keeper_onball`
- **Episodes:** 20 per seed, 80 total
- **Deterministic:** `True` (argmax)
- **No forced actions, no clamp, no teammate scripting**
- **Full-episode rollout** (no post-force window truncation)
- **t=0 validity** captured from `obs[94:97]` ownership and action masks

Command pattern per seed:
```bash
python training/eval_f_act.py \
  --checkpoint training/models/mappo_academy_3_vs_1_with_keeper_seed{N}_freshtrain_50176.pt \
  --scenario academy_3_vs_1_with_keeper_onball \
  --seed {N} \
  --num-episodes 20 \
  --arm ONBALL-pi \
  --deterministic \
  --base-seed 42 \
  --output-dir training/results
```

---

## 5. Reporting Requirements

- Report **per-seed always**, matching Phase B format:
  - Selection rates and counts (PASS, SHOT, TACKLE, DRIBBLE, MOVE, IDLE)
  - Event counts (`pass` initiation, `pass_completed`, `shot`, `goal`)
  - Total ticks per seed
  - t=0 validity fraction
- Never report only the pooled aggregate.
- Explicit pass/fail judgment per criterion, per seed, in the findings document.

---

## 6. Artifacts

| Artifact | Location |
|----------|----------|
| Protocol (this document) | `training/results/FRESH_TRAINING_PROTOCOL.md` |
| Training checkpoints | `training/models/mappo_academy_3_vs_1_with_keeper_seed{N}_freshtrain_50176.pt` |
| Evaluation JSON | `training/results/f_act_ONBALL-pi_seed{N}_mappo_academy_3_vs_1_with_keeper_seed{N}_freshtrain_50176.json` |
| Findings | `training/results/FRESH_TRAINING_FINDINGS.md` |

---

## 7. Confirmations

- No reward / GAE / mask / network / spawn changes.
- No warm-start from existing checkpoints.
- No option (b) implemented.
- PASS path uses production nearest-teammate resolution (unchanged).
