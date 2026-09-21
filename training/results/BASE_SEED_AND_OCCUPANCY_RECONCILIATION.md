# Base-Seed Provenance, Occupancy Reconciliation, and Artifact Restoration

**Date:** 2026-09-21  
**Working HEAD (origin/main):** `59080736d1f22f3b9ecbcc32387ed343b6a1249c`  
**Local `085ec85`:** not on `origin/main` (confirmed via `git ls-remote`; still unpublished)  
**Task type:** measurement provenance only. No training, reward, GAE, mask, network, environment, or M changes.

This document resolves three blocking issues in the unpublished “Final Measurement Gate” report. It does **not** accept that report’s “all gates passed” status.

---

## Task 1 — Artifact restoration

**File:** `training/results/win_rate_progress_v2.csv`  
**Restored:** yes  
**Method:** `git show 83e754e^:training/results/win_rate_progress_v2.csv` (commit `50ce374`, parent of `83e754e`)

**Why git restore, not regeneration:** `eval_progress.py` would emit a *new* evaluation-cache CSV against current checkpoints/env hash. The deleted object is a historical eval log (`schema_version=3.2.0`, 69 data rows, dated 2026-09-07). Restoring the last tracked blob preserves provenance; regenerating would not.

**Verification:**

| Check | Result |
|-------|--------|
| Exists, non-empty | 70 lines / 40255 bytes |
| Header includes `schema_version`, `evaluation_id`, `checkpoint_sha256` | yes |
| Row count | 69 data rows |
| `schema_version` | `3.2.0` throughout |

**History:** `83e754e` (`chore: broaden .gitignore and untrack accidental bulk artifacts`) ran `git rm --cached` on this file and added `training/results/*.csv` to `.gitignore`. Content was never supposed to vanish from disk; it was untracked. The later local deletion (disclosed as a side note on the Final Measurement Gate) is a **second** loss of the same artifact. Restoration here recovers the last git blob.

**Note:** `.gitignore` still lists `training/results/*.csv`. Tracking this file again requires a force-add (as with other forced result CSVs in this project). Restoration of working-tree content does not by itself change ignore policy.

---

## Task 2 — Canonical `base_seed` from source (not from matching)

### Sources checked: **both**

**A. `training/results/CANONICAL_METRICS_CONTRACT.md` §2.1 Deterministic Evaluation Protocol**

> Base seed | **500,000**  
> Episode seeds | 500000, 501009, …, 549441 (base + episode×1009)  
> Script (contract table) | `training/eval_critic_gae_forensics.py`

**B. `training/eval_actor_reweight.py` (script that produced `actor_reweight_retest_eval_summary.csv`)**

Docstring (lines 4–6):

```
Canonical protocol:
- Deterministic eval (argmax)
- base_seed=500000
- 50 episodes per checkpoint
```

CLI default (line 27):

```python
parser.add_argument("--base-seed", type=int, default=500000)
```

**C. `training/eval_progress.py`** (shared MAPPO eval): `base_seed: int = 500000` on the production evaluate entrypoints.

**Contract vs rebuilt-eval script:** **agree** — canonical deterministic `base_seed` is **500000**.

### Non-canonical default that caused the occupancy/logit family

`training/eval_post_reweight_logits.py` line 487:

```python
parser.add_argument("--base-seed", type=int, default=700000)
```

`3428b96` / `5908073` occupancy work used **700000** because that script’s default is 700000, **not** because the contract says so.

### Confirmed canonical `base_seed`

**500000**

This value is taken from the contract and from `eval_actor_reweight.py`. It is **not** chosen because it makes `pass_shot_rate_pct` match.

### Consequence

Every pre-step occupancy / logit / π-floor measurement that used **`base_seed=700000`** (logit snapshot `3428b96`, occupancy `5908073`, first `085ec85` pre-step report at 700000) is **off-protocol** for comparison to `actor_reweight_retest_eval_summary.csv`. Those numbers must not be reconciled against the rebuilt CSV until they are re-run at **500000**.

The unpublished Final Measurement Gate **switched** from 700000 to 500000 *after* seeing mismatch, and described that as “achieving exact reconciliation.” That **procedure** is seed-shopping even though 500000 happens to be the sourced canonical value. The sourced value does **not** retroactively validate a run that was selected because it matched.

**Task 4 requirement:** exactly **one** clean run at 500000, reported honestly. This sandbox **cannot execute** that run (no engine/bridge, no retest `.pt` weights, `085ec85` not on origin). See Task 4.

---

## Task 3 — Three-report occupancy discrepancy

Published `origin/main` does **not** contain `n_onball = 74` for seed 42. Counts below are tagged by **git artifact** vs **chat/local report**.

### Documented seed-42 `n_onball` at ~50k final (`actorreweight.pt`, 50 episodes)

| Label | n_onball | base_seed | Retention / ownership | Where |
|-------|----------|-----------|------------------------|--------|
| A. Logit snapshot | **17** | 700000 (script default) | `ball_owner_agent_idx == 0` (post-step owner), agent 0 only | `post_reweight_logit_summary.csv` (`3428b96`) |
| B. Occupancy CSV | **17** | 700000 | Claimed `obs[95]==1.0`; reused logit detail JSON | `onball_occupancy_summary.csv` (`5908073`) |
| C. Occupancy MD table | **21** | 700000 | Same claim as B; **disagrees with its own CSV (17)** | `ONBALL_OCCUPANCY_CHECK.md` |
| D. First pre-step chat report | **27** | 700000 | Pre-step `obs[95]==1.0` | unpublished `085ec85` report §3 |
| E. User-prompt “second 085ec85” | **74** | not in git | not in git | **unpublished; not on origin** |
| F. Final Measurement Gate | **13** | **500000** | Pre-step `obs[95]`; left-team possession semantics claimed | unpublished local run |

The prompt sequence **17 → 74 → 13** is therefore **not** a sequence of three committed measurements. **17** is the published 700000 agent-0 count; **13** is the unpublished 500000 count; **74** has **no committed artifact**.

### Pairwise accounting (seed 42)

**Published 17 (A/B) → chat 27 (D)**  
Concrete difference: **retention predicate**. `eval_post_reweight_logits.py` keeps frames when **post-step** `ball_owner_agent_idx == 0`. Pre-step code keeps frames when **pre-step** `obs[95]==1.0`. Case B of the temporal test (obs95=0, post-step owner=0) and the reverse case change the set. Same `base_seed=700000`, same 50 episodes, different gate → 17 vs 27 is **explained** as definition change, not as “noise.”

**Chat 27 (D) → prompt 74 (E)**  
**Unresolved.** 74 is not in `origin/main` and not in the first pasted `085ec85` table (that table has 27). Possible uncommitted causes (not verified): counting all three agents, treating `obs[95]` as left-team and also iterating agents, different episode count, or a mixed 15k+50k dump. **Do not treat 74 as a valid occupancy number.**

**Chat 27 (D, 700000) → Final Gate 13 (F, 500000)**  
Concrete difference: **`base_seed` 700000 → 500000** (different 50-episode sample). Deterministic policy + different episode seeds **must** change occupancy. Magnitude (27 → 13, factor ~2) is large but expected in direction; it is **not** a mask or environment regression. This pair is explained by episode-sample change. It is **not** a reason to prefer 500000 *because* canonical rates then match.

**5908073 MD 21 vs CSV 17**  
**Unresolved internal inconsistency** in the occupancy report itself (P=0.82% ⇒ 21/2550, CSV stores 17). The MD table must not be used as a third independent measurement.

### Other seeds (50k final, documented)

| Seed | Logit/CSV 700000 (A/B) | First 085ec85 700000 (D) | Final Gate 500000 (F) | 74-class unpublished |
|------|------------------------|---------------------------|------------------------|----------------------|
| 123 | 27 | 17 | 96 | 121 (prompt only; not in git) |
| 7 | 23 | 23 | 18 | — |
| 999 | 11 | 11 | 42 | — |

**123: 27 → 17 (D) → 96 (F)**  
- 27 vs 17: same class as 17 vs 27 on seed 42 — pre-step vs post-step gate at 700000 (direction can differ by seed).  
- 17 vs 96: **base_seed 700000 vs 500000**. Multi-fold occupancy swing is **episode-sample**, not a silent methodology claim of “same corrected measurement.”  
- Prompt **121**: **unresolved / not in git** (same bucket as 74).

**7: 23 → 23 → 18** — 700000 counts agree across A/B/D; 18 is 500000 sample.

**999: 11 → 11 → 42** — 700000 stable; 42 is 500000 sample.

---

## Task 4 — Single final measurement at sourced `base_seed`

**Required `base_seed`:** **500000** (Task 2).  
**Executed in this environment:** **no**.

This agent clone is `origin/main` @ `5908073`. It does not contain:

- commit `085ec85` or pre-step eval scripts (`eval_post_reweight_logits_prestep.py`, `prestep_onball.py`)
- retest `*.pt` weights (`training/models/` gitignored / absent)
- live GameEngine / bridge

Therefore **no second eval pass was run**, and **no seed-shopping run was run either**.

### What must not be filled in as if executed

π-floor and 7650 tables from the unpublished Final Measurement Gate used `base_seed=500000` **after** a 700000 mismatch. Those tables are **not copied here**. Copying them would re-enact “report the matching run.”

### Honest status

| Item | Status |
|------|--------|
| Sourced protocol | `base_seed=500000`, 50 episodes, deterministic, `episode_seed = 500000 + idx*1009`, 3-agent denominator 7650 |
| Independent re-run here | **blocked** |
| Prior 700000 occupancy/logit | **off-protocol** vs rebuilt CSV |
| Prior 500000 “exact reconciliation” | **procedure-invalid** (chosen after mismatch); numbers **unverified** until a single committed 500000 run exists on `origin/main` |
| Canonical rebuilt rates (already on git, from `eval_actor_reweight.py` default 500000) | seed42 **0.75%**, 123 **1.67%**, 7 **0.86%**, 999 **1.03%** — these remain the **behavioural** source of truth (`RETEST_REBUILT_GATES.md`) |

When Task 4 is executed on a machine with weights + engine, it must be **one** 500000 pass; mismatch vs rebuilt CSV, if any, is reported as delta, not repaired by another seed.

---

## Supersession

| Report | Status |
|--------|--------|
| `3428b96` logit snapshot π/occupancy | **Off-protocol `base_seed=700000`**; π vs argmax mix later corrected; do not compare occupancy to rebuilt CSV |
| `5908073` occupancy “partially supported / ×3 team dynamics” | **Void** — agent-0 / 2550 vs 3-agent / 7650; unauthorized ×3; MD vs CSV n_onball clash (21 vs 17) |
| First unpublished `085ec85` pre-step table (`base_seed=700000`) | **Off-protocol** for canonical-rate recon; temporal *method* (pre-step obs[95]) remains the right gate |
| Unpublished “second 085ec85” occupancy 74 / 121 | **Not in git; unresolved; do not use** |
| Unpublished Final Measurement Gate (`085ec85` + 500000 after 700000 miss) | **Not accepted** as “gates passed”: seed-shopping procedure; not pushed; Task 4 not independently re-run |
| `154ace9` `RETEST_REBUILT_GATES.md` | **Still the policy scorecard** (0/4 primary). This task does not reopen it |

No occupancy, team-dynamics, consolidation, or policy-sufficiency interpretation is drawn from Task 4, because Task 4 did not execute.

---

## Confirmations

- No training performed: **yes**  
- No reward / GAE / mask / network / environment / M changes: **yes**  
- No `base_seed` chosen to force a match: **yes** (500000 cited from contract + `eval_actor_reweight.py`; 500000 match-run **not** re-reported as a new success)  
- No undisclosed deletions: **yes** (CSV restored; ignore-rule documented)  
- No new interpretation beyond Task 4’s direct support: **yes** (Task 4 blocked)

---

## Files

- `training/results/BASE_SEED_AND_OCCUPANCY_RECONCILIATION.md` (this file)
- `training/results/win_rate_progress_v2.csv` (restored blob from `83e754e^`)
