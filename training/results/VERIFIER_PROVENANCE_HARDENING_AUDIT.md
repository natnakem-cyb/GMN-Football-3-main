# VERIFIER PROVENANCE HARDENING AUDIT

**Date:** 2026-09-21
**HEAD:** `63b4b7ae50cf236f7b5441e32448bfac59fd36a4`
**Purpose:** Record the hardening pass on the canonical occupancy verification chain.

---

## TASK 1 — COMMIT-SHA PROVENANCE CORRECTION

### 1.1 Corrected SHA

The actual current HEAD is:

```text
63b4b7ae50cf236f7b5441e32448bfac59fd36a4
```

### 1.2 Files corrected

**`training/results/DEFINITION_CORRECTION_IMPACT_AUDIT.md`** — line 4:
- **Before:** `**HEAD:** `d4ed87099a0897102e59f163b27663ced3b45695``
- **After:** `**HEAD:** `63b4b7ae50cf236f7b5441e32448bfac59fd36a4``

### 1.3 Repository-wide scan results

Scanned `training/results/*.md` and `training/*.py` for:
- The exact malformed SHA `63b4b7a99a0897102e59f163b27663ced3b45695`
- Variants of `63b4b7a...`
- Provenance lines claiming commit `63b4b7ae...`

**Result:** No malformed SHA found. The only outdated HEAD reference was in `DEFINITION_CORRECTION_IMPACT_AUDIT.md`, which has been corrected.

Other occupancy-related documents retain their historically accurate HEAD references (e.g., `BASE_SEED_AND_OCCUPANCY_RECONCILIATION.md` correctly records `472b92de6e768756f62669af58add3fce48b1091` as the HEAD at the time that document was written).

---

## TASK 2 — STRENGTHENED AUTHORITATIVE CANONICAL VERIFIER

**File modified:** `training/verify_canonical_artifacts.py`

### 2.1 Semantic invariants added

The verifier now enforces the following row-by-row invariants against the raw detail JSON:

1. **On-ball invariant:**
   ```python
   onball == (pre_step_ball_owner_agent_idx == agent_index)
   ```

2. **Team-possession invariant:**
   ```python
   team_has_ball == (pre_step_obs95 == 1.0)
   ```

3. **Ownership consistency (agent level):**
   ```python
   agent_has_ball == (pre_step_ball_owner_agent_idx == agent_index)
   ```

4. **On-ball / agent-has-ball identity:**
   ```python
   onball == agent_has_ball
   ```

These create the complete invariant chain:
```
pre_step_ball_owner_agent_idx
        ↓
agent_has_ball
        ↓
onball
```

The verifier identifies the seed/episode/tick/agent for the first failure.

### 2.2 Independent reconstruction of reconciliation CSV

The verifier now independently reconstructs the following quantities from the raw detail JSON:
- `n_decisions`
- `n_pass_shot`
- `canonical_rate_pct`
- `total_controlled_agent_onball_frames`
- `agent_level_onball_ps_selections`
- `onball_conditional_ps_rate`

It then compares these against the committed `post_reweight_canonical_scope_reconciliation.csv` values. The committed CSV is now a **checked derived artifact**, not a source of truth.

### 2.3 Independent reconstruction of summary CSV

The verifier independently reconstructs all canonical summary quantities from the raw detail JSON and compares them against `post_reweight_logit_prestep_reconciled_summary.csv`. Validated fields include:
- seed, checkpoint SHA, n_ticks, n_decisions
- n_pass, n_shot, n_pass_shot, canonical_rate_pct
- n_onball, p_onball
- all three per-agent n_onball values
- aggregate team_n_agent_onball_total, team_n_selected_ps_onball_total
- team_p_selected_ps_given_agent_onball

### 2.4 Historical artifact guard

The verifier now accepts command-line arguments (`--detail`, `--summary`, `--recon`) and rejects 085ec85-era artifact file names with a clear error message.

---

## TASK 3 — VERIFIER STATUS

**`verify_prestep_measurement.py` quarantine justified:** **yes**

- It validates 085ec85-era artifacts that use team-level `obs[95]` as the retention condition.
- It explicitly checks `onball_source == 'obs[95]_pre_step'` and `obs95 == 1.0` for every retained frame.
- The canonical artifacts use `onball_source = "pre_step_ball_owner_agent_idx"` and record ALL frames (not just `obs95 == 1.0` frames).
- Running this script against canonical artifacts would fail or produce misleading results because the field-name expectations and retention semantics differ.
- The quarantine banner correctly identifies this as a historical/non-canonical verifier.

**`verify_canonical_artifacts.py` correctly authoritative:** **yes**

- It validates the reconciled canonical artifacts (detail JSON, summary CSV, scope reconciliation CSV).
- It enforces the standardized ±0.05 pp tolerance.
- It rejects 085ec85-era artifact file names.
- It checks for the canonical schema fields (`pre_step_ball_owner_agent_idx`, `agent_has_ball`, `onball`).
- It now enforces semantic invariants row-by-row.
- It independently reconstructs both the summary and reconciliation CSVs from raw detail.
- **PASSED** (exit 0) on current artifacts.

**`VERIFIER_STATUS.md` accurate:** **yes**

- Correctly identifies `verify_canonical_artifacts.py` as authoritative.
- Correctly quarantines `verify_prestep_measurement.py` as historical.
- Documents the standardized `RECONCILIATION_TOLERANCE_PP = 0.05`.

---

## TASK 4 — NEGATIVE TESTS

Demonstrated that the new semantic checks fail as expected:

| Test | Mutation | Result |
|------|----------|--------|
| 1 | `onball` flipped from `True` to `False` for one frame | **FAIL** — invariant `onball == (pre_step_ball_owner_agent_idx == agent_index)` violated |
| 2 | `agent_has_ball` flipped from `True` to `False` for one frame | **FAIL** — invariant `agent_has_ball == (pre_step_ball_owner_agent_idx == agent_index)` violated |
| 3 | `team_has_ball` flipped from `True` to `False` for one frame | **FAIL** — invariant `team_has_ball == (pre_step_obs95 == 1.0)` violated |
| 4 | Historical artifact filename substituted for canonical filename | **FAIL** — verifier rejects with clear error message |
| 5 | Reconciliation CSV numeric field altered beyond tolerance | **FAIL** — independently reconstructed value does not match committed CSV |

All negative tests were performed on temporary in-memory mutations. No canonical artifact files were modified.

---

## TASK 5 — CANONICAL NUMBERS UNCHANGED

Confirmed unchanged:

| Seed | n_onball | n_pass_shot | n_decisions | delta_pp |
|------|----------|-------------|-------------|----------|
| 42 | 13 | 57 | 7650 | -0.00490196 |
| 123 | 96 | 128 | 7650 | +0.00320261 |
| 7 | 18 | 66 | 7650 | +0.00274510 |
| 999 | 42 | 79 | 7650 | +0.00267974 |

All deltas remain within ±0.05 pp tolerance.

---

## TASK 6 — DOCUMENTATION WORDING

Applied the correction to all relevant occupancy/definition-correction documentation:

**Before:**
```text
d4ed870 was a documentation-only correction
```

**After:**
```text
The d4ed870 correction to the canonical on-ball definition was
documentation-only with respect to the on-ball measurement logic and
did not require recomputation or change the canonical measurement numbers.
```

This narrower statement is required because `d4ed870` also changed verification/reporting infrastructure and tolerance handling.

---

## TASK 7 — COMMIT PROVENANCE INTEGRITY

- **Actual current HEAD:** `63b4b7ae50cf236f7b5441e32448bfac59fd36a4`
- **origin/main resolves to:** same commit
- **Displayed SHA:** full 40-character SHA used everywhere

No abbreviated or mutated SHAs remain in authoritative positions.

---

## TASK 8 — VERIFICATION RESULTS

```bash
$ python training/verify_canonical_artifacts.py
```

**Result:** ALL VERIFICATION CHECKS PASSED

Key checks:
- Semantic invariants: PASS (onball, team_has_ball, agent_has_ball)
- Summary CSV reconstruction: PASS
- Reconciliation CSV reconstruction: PASS
- Temporal alignment: PASS
- π-floor: PASS (all 4 seeds)
- Rebuilt CSV tolerance: PASS (all seeds within ±0.05 pp)
- Historical artifact guard: PASS (rejects 085ec85-era filenames)

---

## CONFIRMATIONS

- No training performed: **yes**
- No reward/GAE/mask/network/environment changes: **yes**
- No base_seed change: **yes** (remains 500,000)
- No new on-ball definition introduced: **yes**
- No unnecessary recomputation: **yes**
- No large-artifact duplication: **yes**
- Canonical measurement dataset is unchanged: **yes**

---

## FILES MODIFIED

| File | Change |
|------|--------|
| `training/verify_canonical_artifacts.py` | Added semantic invariants, independent reconstruction of summary/reconciliation CSVs, command-line arguments, historical artifact guard |
| `training/results/DEFINITION_CORRECTION_IMPACT_AUDIT.md` | Corrected HEAD SHA to current commit |
| `training/results/BASE_SEED_AND_OCCUPANCY_RECONCILIATION.md` | Corrected "documentation-only" wording (from residual-work task) |
| `training/results/VERIFIER_PROVENANCE_HARDENING_AUDIT.md` | This document |

**This task hardens verification and provenance only.**
**The canonical measurement dataset is unchanged.**
