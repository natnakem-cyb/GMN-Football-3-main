# Frozen-π Re-evaluation Post PASS Fix

## 1. Provenance

- **HEAD:** `5a9dfdda449af14f2bed8b5c94161bf5dfe48acf`
- **Date:** 2026-09-18
- **Scenario:** `academy_3_vs_1_with_keeper_onball`
- **Arm:** `ONBALL-π` (pure frozen policy, no force)
- **Checkpoints:** 4 × 50,176-step MAPPO (seeds 42, 123, 7, 999)
- **Episodes:** 80 total (20 per seed)
- **Deterministic:** True

## 2. Protocol

- Pure frozen policy inference (`deterministic=True`, argmax).
- No forced actions, no clamp, no teammate scripting, no direction override.
- Full-episode rollout (no post-force window truncation).
- t=0 validity captured from reset observation (`obs[94:97]` ownership, action masks).
- PASS direction resolution: nearest same-team teammate (Euclidean) when `action.direction` is absent.
- No training, no reward/GAE/mask/network/spawn changes.

## 3. t=0 Validity

| Seed | Valid t0 possession | Fraction |
|------|--------------------:|---------|
| 42   | 20 / 20             | 100%     |
| 123  | 20 / 20             | 100%     |
| 7    | 20 / 20             | 100%     |
| 999  | 20 / 20             | 100%     |

Aggregate: **80 / 80 = 100%** valid on-ball state at reset.

## 4. Action Histogram (pure π)

| Action category | Count | Rate (of 3,416 ticks) |
|-----------------|------:|----------------------:|
| PASS (9–11)     | 2     | 0.06%                 |
| SHOT (12)       | 3     | 0.09%                 |
| TACKLE (16)     | 173   | 5.06%                 |
| DRIBBLE (17)    | 55    | 1.61%                 |
| MOVE (1–8)      | 2,578 | 75.47%                |
| IDLE (0)        | 19    | 0.56%                 |
| **Total ticks** | 3,416 | 100%                  |

**Note:** Counts include all 3 controlled left-team players per tick. The `action_selected` field in per-tick CSV/trace data records only agent 0; aggregate counts above are derived from per-agent episode summaries.

## 5. Event Counts

| Event               | Count |
|---------------------|------:|
| `pass` initiation   | 1     |
| `pass_completed`    | 0     |
| `shot`              | 1     |
| `goal`              | 0     |

No goals scored under pure π in any seed.

## 6. Per-Seed Snapshot

| Seed | Total ticks | PASS | SHOT | TACKLE | DRIBBLE | Goals |
|------|------------:|-----:|-----:|-------:|--------:|------:|
| 42   | 836         | 0    | 0    | 54     | 0       | 0     |
| 123  | 870         | 1    | 1    | 0      | 8       | 0     |
| 7    | 813         | 1    | 1    | 119    | 32      | 0     |
| 999  | 897         | 0    | 1    | 0      | 15      | 0     |

## 7. Comparison to Pre-Repair Baseline

There is no pre-repair pure-π (`ONBALL-pi`) baseline on record. The closest prior measurement is the post-force π window in `F_act` forced-arm runs, which reported **0 PASS / 0 SHOT / 0 TACKLE** selections in the 50-tick post-force window across all seeds.

Post-repair pure π over full episodes shows:

- **PASS/SHOT:** still vanishingly rare (0.06% / 0.09%). No measurable change in spontaneous pass/shot selection.
- **TACKLE:** non-zero (5.06%), with strong seed variance (seed 7 = 119, seed 42 = 54, seeds 123/999 = 0).
- **Goals:** 0 across all seeds.

## 8. Interpretation

**Behaviour change vs pre-repair frozen π:** Minor / none for PASS and SHOT; TACKLE is newly non-zero but was not measured in a prior pure-π baseline. The rare spontaneous PASS/SHOT events (2 and 3 respectively) are within noise and do not indicate environment-coupled behaviour change.

**Paralysis independent of PASS aiming defect:** **Yes.** The frozen policy still selects football actions at a negligible rate. The repaired PASS path (nearest-teammate aiming) does not alter π behaviour because π almost never chooses PASS/SHOT. Paralysis is a policy-selection problem, not an environment-mechanics problem.

**Evidence summary:**
Pure π over 80 episodes selects PASS 2 times (0.06%) and SHOT 3 times (0.09%), produces 1 spontaneous `pass` event and 1 `shot` event, and scores 0 goals. TACKLE is the only football action with meaningful frequency (173 selections, 5.06%), driven entirely by seeds 7 and 42. Because the PASS fix only changes environment execution of an already-rare policy action, frozen-π behaviour is effectively unchanged. The next step should be a learning experiment (fresh train or mix-script), not further environment mechanics work.

## 9. Artifacts

- `training/results/f_act_frozen_pi_postfix.json` — aggregate per-seed JSON
- `training/results/f_act_frozen_pi_postfix_summary.csv` — one row per seed
- `training/results/f_act_ONBALL-pi_seed{42,123,7,999}_mappo_academy_3_vs_1_with_keeper_seed*_50176.json` — per-seed raw JSON
- `training/results/FROZEN_PI_POST_PASS_FIX.md` — this report

## 10. Confirmations

- No training / reward / GAE / mask / network / spawn changes: **yes**
- No force arm used: **yes**
- Pre-fix & PASS diagnostic files untouched: **yes**
- Option (b) passing-lane not implemented: **yes**
