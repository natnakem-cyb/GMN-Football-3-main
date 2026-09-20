# ONBALL_OCCUPANCY_PRESTEP_CHECK — Corrected Pre-Step On-Ball Measurement (50k actor-reweight checkpoints)

**Date:** 2026-09-20
**Type:** Measurement-only correction. No training, no reward/GAE/mask/actor/critic/optimiser/entropy/horizon change, no checkpoint-weight change.

**Measurement code commit (git HEAD at measurement time):** `59080736d1f22f3b9ecbcc32387ed343b6a1249c`
The corrected measurement code (collector + shared pre-step module + verifier + tests) was an uncommitted working-tree version during the run, so it is additionally identified by per-file SHA-256 digests (also recorded inside the artifact JSON as `provenance.measurement_code_sha256`):

| File | SHA-256 |
|------|---------|
| `training/prestep_onball.py` | `f630492af654dee7eadc7954217434c2535c34e6a703a78d2c6c02cd88f401c4` |
| `training/eval_post_reweight_logits_prestep.py` | `528e82e9a0984e1437d485ba16e6ab585676ea7f435cde5a26bf04bf7ce2cfe6` |
| `training/verify_prestep_measurement.py` | `f8f8ab804e3d1b86e264c537f7146c3de7edc0ad7e9c679a291a4c3173e0a490` |
| `training/tests/test_prestep_onball_temporal_alignment.py` | `a0a2cba793c917f48923d4af55bb54e39cb0dc666d2178778fbb0c01cdc94cdb` |

**Artifacts (new; historical `3428b96` / `5908073` artifacts were not modified):**

- `training/results/post_reweight_logit_prestep_detail.json` — per-frame detail, strict JSON (2.7 MB, 378 retained frames)
- `training/results/post_reweight_logit_prestep_summary.csv` — per-seed aggregates + provenance (4 rows)
- `training/results/onball_occupancy_prestep_summary.csv` — occupancy table (4 rows)
- this report

**Scope / non-goals:** this pass measures only. It does not reopen the frequency-reweight policy-sufficiency verdict, does not recommend or authorise training, and makes no claim about consolidation, actor collapse, occupancy causality, cross-seed mechanisms, other-agent coordination, mask defects or reward dynamics.

---

## 1. Validation Gate

**Test module:** `training/tests/test_prestep_onball_temporal_alignment.py`
**Result:** **9/9 PASSED** (`python training/tests/test_prestep_onball_temporal_alignment.py`).

| # | Test | Result |
|---|------|--------|
| 1 | `test_case_a_pre_step_onball_post_step_not_agent0` (obs[95]=1, post-step owner=2 → RETAIN) | PASS |
| 2 | `test_case_b_pre_step_not_onball_post_step_is_agent0` (obs[95]=0, post-step owner=0 → **NOT** RETAIN) | PASS |
| 3 | `test_case_c_both_agree_agent0` (obs[95]=1, post-step owner=0 → RETAIN) | PASS |
| 4 | `test_case_d_neither_indicates_agent0` (obs[95]=0, post-step owner=255 → NOT RETAIN) | PASS |
| 5 | `test_post_step_owner_cannot_override_pre_step_retention` (owners 0/1/2/255) | PASS |
| 6 | `test_retained_frame_uses_pre_step_mask` | PASS |
| 7 | `test_retained_frame_uses_pre_step_observation_and_logits` | PASS |
| 8 | `test_deterministic_action_from_pre_step_policy` | PASS |
| 9 | `test_policy_probability_not_confused_with_action_frequency` | PASS |

The tests import the **production** retention predicate, frame builder and aggregator from `training/prestep_onball.py` (the same functions the collector calls), so a regression in production logic fails the test instead of passing against a duplicated copy.

**Mutation check (defect-detection proof).** Re-introducing the defective predicate
(`should_retain_prestep_onball_frame = lambda obs, owner=None: owner == 0`) makes Case B fail with
`frame with pre-step obs[95] == 0 must NOT be retained even when the post-step ball owner becomes agent 0`.
The test therefore detects exactly the historical defect.

**Runtime alignment assertions inside the collector** (fail loudly rather than silently mis-measure):
every retained frame must satisfy `pre_step_onball == True` **and** `frame["action_taken"] == applied action`.

## 2. Protocol

```text
scenario            academy_3_vs_1_with_keeper_onball
seeds               42, 123, 7, 999
episodes            50 per checkpoint
deterministic       True (argmax of masked pre-step actor logits)
base_seed           700000 ; episode_seed = base_seed + episode_index * 1009
checkpoint label    50k  (human label)
internal timestep   49920 (actual value stored in each checkpoint; NOT relabelled)
frame gate          retain iff PRE-step agent-0 observation satisfies obs[95] == 1.0
onball_source       "obs[95]_pre_step"
platform            headless TS bridge (npx tsx training/bridge_server.ts), port 5050
```

**Checkpoint identity (verified against `training/results/retest_checkpoint_inventory.csv` *before* measuring;
the collector hard-fails on any mismatch):**

| seed | checkpoint file | SHA-256 | internal timesteps |
|------|-----------------|---------|--------------------|
| 42 | `mappo_academy_3_vs_1_with_keeper_onball_seed42_actorreweight.pt` | `eb9e1403b6a65c3c288397b752c64319cf8f9022116492fa05a355c08682fd24` | 49920 |
| 123 | `mappo_academy_3_vs_1_with_keeper_onball_seed123_actorreweight.pt` | `72e56e385a031953f5ace01f2e56ff658c9730b7cc5c5e1ad88fd9c637acf4be` | 49920 |
| 7 | `mappo_academy_3_vs_1_with_keeper_onball_seed7_actorreweight.pt` | `7b29a3a465a3e1044a4908b205ce5fae2aa589b9d39ba258047314335d8b8524` | 49920 |
| 999 | `mappo_academy_3_vs_1_with_keeper_onball_seed999_actorreweight.pt` | `0365a1d95f9a31f9517bdd42401aa3f4a4bfed3e713be610ae91bcf95d526097` | 49920 |

4/4 checkpoints evaluated. The human label `50k` and the internal timestep `49920` are reported together and never conflated.

**Determinism evidence:** the full measurement was executed twice; the summary CSV of run 2 is **byte-identical** to run 1
(`filecmp.cmp(..., shallow=False) == True`). The statistics below are from the second run.

**Temporal-alignment semantics (all from one pre-step tick):**

| Quantity | Source |
|----------|--------|
| retention gate | **pre-step** `agent0_obs[95] == 1.0` |
| legality (`mask[9..12]`) | **pre-step** `mask_matrix[0]` |
| raw / masked logits, π | **pre-step** actor forward on `(agent0_obs, agent0_mask)` |
| deterministic action | argmax of that same pre-step masked logit vector |
| `post_step_ball_owner_agent_idx` | recorded as a **diagnostic only**, never used for retention |

**Semantics note (needed to read the tables correctly).** `obs[94:97]` is a one-hot over
`[no-one, left, right]` **team-level** ownership (`src/engine/ObservationEncoder.ts:33,136-140`), so
`obs[95] == 1.0` means *the left team has the ball* (agent 0, 1 or 2 may be the individual owner).
Individual possession is exposed by the per-agent mask: the engine sets `mask[9..12] = 1` only when the
acting player personally has possession (`ObservationEncoder.getActionMask`), which is why
`n_pass_legal == n_shot_legal` in every row (one `hasPossession` gate drives all four slots).

**JSON safety:** the detail artifact is valid strict JSON. `-inf` masked-logit slots and any empty population
are encoded as `null` (documented in the artifact's `schema` block); no `Infinity` / `-Infinity` / `NaN` literal is emitted
(`json.dump(..., allow_nan=False)` plus a `json_safe` pass; the verifier re-parses with `parse_constant` rejection).

## 3. Corrected On-Ball Occupancy

Numerator is strictly `pre-step obs[95] == 1.0` for the agent-0 decision state. Denominator `n_ticks` is the
number of decision ticks in the 50 episodes (51 ticks/episode × 50 episodes).

| seed | n_ticks | n_onball | P(on-ball) |
|------|---------|----------|------------|
| 42 | 2550 | 74 | 0.029019607843137254 (2.9020%) |
| 123 | 2550 | 121 | 0.04745098039215686 (4.7451%) |
| 7 | 2550 | 91 | 0.03568627450980392 (3.5686%) |
| 999 | 2550 | 92 | 0.03607843137254902 (3.6078%) |
| **total** | **10200** | **378** | **0.03705882352941176 (3.7059%)** |

Retainment denominators: every one of the 50 episodes per seed contained at least one retained frame
(maximum retained frames in a single episode: 6 / 7 / 10 / 5 for seeds 42 / 123 / 7 / 999).

## 4. Legality on the retained pre-step on-ball population

Legality is read from the **pre-step** `mask_matrix[0]`. PASS = any of actions {9,10,11}; SHOT = {12}.
`n_pass_legal == n_shot_legal == n_pass_or_shot_legal` in every row because the engine derives all four
slots from one `hasPossession` gate (see the semantics note in §2); they are reported separately so the
equality is measurable rather than assumed.

| seed | n_onball | n_pass_legal | pass_legal_rate | n_shot_legal | shot_legal_rate | n_pass_or_shot_legal | pass_or_shot_legal_rate |
|------|----------|--------------|-----------------|--------------|-----------------|----------------------|-------------------------|
| 42 | 74 | 67 | 0.9054054054054054 (90.54%) | 67 | 0.9054054054054054 (90.54%) | 67 | 0.9054054054054054 (90.54%) |
| 123 | 121 | 77 | 0.6363636363636364 (63.64%) | 77 | 0.6363636363636364 (63.64%) | 77 | 0.6363636363636364 (63.64%) |
| 7 | 91 | 73 | 0.8021978021978022 (80.22%) | 73 | 0.8021978021978022 (80.22%) | 73 | 0.8021978021978022 (80.22%) |
| 999 | 92 | 61 | 0.6630434782608695 (66.30%) | 61 | 0.6630434782608695 (66.30%) | 61 | 0.6630434782608695 (66.30%) |

Reading note: the complement `n_onball - n_pass_legal` (7 / 44 / 18 / 31) are retained ticks on which the
**left team** held the ball but **agent 0 personally did not**, so agent 0's PASS/SHOT slots are illegal and its
π over those slots is exactly 0. Frame selection was never filtered on legality, so those frames remain in
Population A (§5) and are removed only in Populations B/C/D.

## 5. Policy Probability (actor distribution, not selected actions)

π values are the actor's masked softmax mass on the **same pre-step state** used for the gate and the mask.
Four explicitly defined populations are reported; every mean carries its `n`.

| seed | n_onball (A) | π_PASS mean (A) | π_SHOT mean (A) | π_PASS+SHOT mean (A) | n_PASS_legal (B) | π_PASS \| PASS-legal (B) | n_SHOT_legal (C) | π_SHOT \| SHOT-legal (C) | n_either_legal (D) | π_PASS+SHOT \| either-legal (D) |
|------|--------------|-----------------|-----------------|----------------------|------------------|--------------------------|------------------|--------------------------|--------------------|---------------------------------|
| 42 | 74 | 0.7143105601237433 | 0.014840312704846665 | 0.7291508728285899 | 67 | 0.788940021629209 | 67 | 0.016390793136696318 | 67 | 0.8053308147659053 |
| 123 | 121 | 0.3092540629392813 | 0.027468404494041254 | 0.33672246743332257 | 77 | 0.48597067033315633 | 77 | 0.043164635633493396 | 77 | 0.5291353059666497 |
| 7 | 91 | 0.3829062996575466 | 0.05240714955297145 | 0.435313449210518 | 73 | 0.4773215516279005 | 73 | 0.06532946040164934 | 73 | 0.5426510120295498 |
| 999 | 92 | 0.44338834534763644 | 0.026910473187656506 | 0.47029881853529293 | 61 | 0.6687168487210254 | 61 | 0.04058628743056391 | 61 | 0.7093031361515894 |

Population definitions (also encoded in the detail JSON `schema` block):

- **A — all pre-step on-ball frames** (`n_onball`): the full retained population, including ticks where agent 0's own PASS/SHOT slots are illegal. Illegal-action frames are **not** discarded from A.
- **B — PASS-legal on-ball frames** (`n_pass_legal`): A ∧ `mask_pass_legal == 1`.
- **C — SHOT-legal on-ball frames** (`n_shot_legal`): A ∧ `mask_shot_legal == 1`.
- **D — PASS-or-SHOT-legal on-ball frames** (`n_pass_or_shot_legal`): A ∧ `mask_pass_or_shot_legal == 1`.

That Populations B/C/D give higher means than A in every seed is the arithmetic consequence of masking:
in A, 7/74, 44/121, 18/91 and 31/92 retained ticks contribute exact zeros for π_PASS/π_SHOT.

## 6. Deterministic Behavioral Frequency (separate from π)

This table counts the **selected deterministic action**, i.e. a behavioural action-selection statistic.
It is deliberately labelled `P(selected PASS+SHOT | on-ball)` and must not be quoted as π_PASS, π_SHOT or
"policy probability".

| seed | n_onball | n_PASS+SHOT selected | P(selected PASS+SHOT \| on-ball) |
|------|----------|----------------------|----------------------------------|
| 42 | 74 | 61 | 0.8243243243243243 (61/74) |
| 123 | 121 | 72 | 0.5950413223140496 (72/121) |
| 7 | 91 | 68 | 0.7472527472527473 (68/91) |
| 999 | 92 | 58 | 0.6304347826086957 (58/92) |

None of the selected PASS+SHOT actions occurred on a tick where the corresponding slots were illegal
(the deterministic action is the argmax of the masked logits, which is verified for every frame by the
independent checker).

## 7. Small-n Qualification

Every per-seed statement above must be read with its denominator. The 50-episode deterministic evaluation
produces **sparse** on-ball samples in this scenario, so the reported percentages are high-variance
descriptive measurements, not stable seed-level properties.

| seed | n_ticks | n_onball (A) | n_pass_legal (B) | n_shot_legal (C) | n_pass_or_shot_legal (D) | n_PASS+SHOT selected |
|------|---------|--------------|------------------|------------------|--------------------------|----------------------|
| 42 | 2550 | 74 | 67 | 67 | 67 | 61 |
| 123 | 2550 | 121 | 77 | 77 | 77 | 72 |
| 7 | 2550 | 91 | 73 | 73 | 73 | 68 |
| 999 | 2550 | 92 | 61 | 61 | 61 | 58 |

Explicit sparsity statement: P(on-ball) rests on 74–121 retained ticks out of 2550 decision ticks; the
legality rates rest on 61–77 of those; the conditional π means in Populations B/C/D rest on the same 61–77
ticks; the behavioural fractions rest on 58–72 of 74–121 ticks. Ratios of this size (e.g. 61/74, 58/92,
67/74, 72/121) shift by several percentage points when a single frame changes, so they must not be treated
as stable seed-level characteristics. No `n` is hidden behind a percentage anywhere in this report or in the
CSV artifacts.

## 8. Old vs Corrected Measurement (implementation audit only)

**Old source:** `training/results/post_reweight_logit_summary.csv` (rows with `checkpoint_timesteps = 49920`)
and `training/results/onball_occupancy_summary.csv` (rows with `step = 49920`), produced by
`training/eval_post_reweight_logits.py` at commits `3428b96` / `5908073`.
**New source:** `training/results/post_reweight_logit_prestep_summary.csv` / `onball_occupancy_prestep_summary.csv`.

### 8a. Why the two populations are different sets

The old collector selected a tick with `getattr(env, "_last_ball_owner_agent_idx", 255) == 0`, i.e.
**post-step individual ownership**, while reporting pre-step masks/logits/π. The corrected collector selects
a tick with **pre-step `obs[95] == 1.0`**, i.e. pre-step *team-level* left-team ownership. Therefore:

```text
old population       = {pre-step left-team on-ball AND post-step owner 0}
                       UNION {pre-step NOT left-team on-ball AND post-step owner 0}  <- acquisition ticks
corrected population = {pre-step left-team on-ball}   (regardless of post-step owner)
```

This is a **measurement-definition change** (which state decides retention, and team-level vs
individual-level possession), not an environment or policy change: no reward/mask/GAE/network/env code was
touched, and the corrected collector reproduces byte-identical statistics on a repeat run.

### 8b. Side-by-side (seeds in the committed artifact order 42 / 123 / 7 / 999)

| quantity | seed 42 old → new | seed 123 old → new | seed 7 old → new | seed 999 old → new |
|----------|-------------------|--------------------|------------------|--------------------|
| n_onball | 17 → **74** | 27 → **121** | 23 → **91** | 11 → **92** |
| P(on-ball) | 0.0067 (0.67%) → **0.029020 (2.90%)** | 0.0106 (1.06%) → **0.047451 (4.75%)** | 0.0090 (0.90%) → **0.035686 (3.57%)** | 0.0043 (0.43%) → **0.036078 (3.61%)** |
| deterministic PASS+SHOT frequency | 0.5294 (9/17) → **0.8243 (61/74)** | 0.5926 (16/27) → **0.5950 (72/121)** | 0.4348 (10/23) → **0.7473 (68/91)** | 0.5455 (6/11) → **0.6304 (58/92)** |
| π_PASS mean (Population A) | 0.549843 → **0.714311** | 0.343665 → **0.309254** | 0.297574 → **0.382906** | 0.449949 → **0.443388** |
| π_SHOT mean (Population A) | 0.016714 → **0.014840** | 0.032752 → **0.027468** | 0.045626 → **0.052407** | 0.048792 → **0.026910** |
| legality rate (pre-step PASS/SHOT legal) | 0.882353 → **0.905405** | 0.777778 → **0.636364** | 0.652174 → **0.802198** | 0.818182 → **0.663043** |

`n_ticks` is 2550 in both measurements, so the P(on-ball) denominators match. The old π and legality values
are taken from the committed `post_reweight_logit_summary.csv`; the old occupancy/frequency values from the
committed `onball_occupancy_summary.csv`.

### 8c. Structural evidence that the change is definitional (not environmental)

1. **The old gate selects a subset plus acquisition ticks — not a different sample.** Of the corrected
   frames, the number whose *diagnostic* post-step owner is agent 0 is 15 / 21 / 15 / 9, versus old
   `n_onball` = 17 / 27 / 23 / 11. Hence old = `{corrected ∧ post-step owner 0}` ∪ {2, 6, 8, 2 acquisition
   ticks} in which the pre-step state did **not** report left-team possession but the post-step owner became
   agent 0 — ticks the corrected rule excludes by construction.
2. **The old gate conditions on the outcome of the action (survivorship).** In the corrected population,
   59 / 100 / 76 / 83 of 74 / 121 / 91 / 92 retained frames have a post-step owner that is **not** agent 0;
   these are overwhelmingly the ticks on which agent 0 disposed of the ball (seed 42: 61 of 74 frames are
   HIGH_PASS selections). Requiring "still owner after the step" mechanically removes exactly those ticks,
   which is why the old population was smaller and its PASS+SHOT selection frequency lower.
3. **The old legality numbers are not a mask property.** The engine derives `mask[9..12]` from a single
   `hasPossession` expression, so `pass_legal_rate == shot_legal_rate` in both measurements (old:
   0.882 / 0.778 / 0.652 / 0.818; new: 0.905 / 0.636 / 0.802 / 0.663). In the corrected measurement the
   legality rate is exactly *the fraction of pre-step team-possession ticks on which agent 0 was the
   individual owner* (67/74, 77/121, 73/91, 61/92). The previous report's "mask defect" reading for
   seeds 123 (at 15k) and 7 was derived from the temporally misaligned population; it is neither confirmed
   nor refuted here, and re-stating it as a mask property would need a separate, explicitly scoped
   measurement.
4. **Provenance defect in the old artifact:** every row of `post_reweight_logit_summary.csv` has an **empty**
   `checkpoint_sha256` (the old collector hard-coded `""`), so those aggregate rows are not self-verifying.
   The corrected CSV populates `checkpoint_sha256` in every row, and the collector refuses to measure any
   checkpoint whose computed SHA-256 differs from the committed inventory value.

**Attribution statement.** The differences in §8b are recorded as an *implementation-level* change of
measurement definition (temporal gate, and team-level vs individual-level possession). No environment
regression, policy regression, mask regression or reward-dynamics change is implied or claimed — and none
may be inferred from a change in an observed legality rate alone.

## 9. Interpretation

### 9.1 Mandatory gate status (all boxes must pass)

```text
[x] temporal-alignment synthetic test passes             -> 9/9 PASSED (+ mutation check)
[x] corrected collector uses pre-step obs[95]            -> is_prestep_onball() from training/prestep_onball.py
[x] pre-step mask and logits are from the same state     -> single (agent0_obs, agent0_mask) pair per tick
[x] deterministic action uses that same pre-step state   -> runtime equality assertion per retained frame
[x] 4/4 50k checkpoints evaluated                        -> seeds 42, 123, 7, 999
[x] checkpoint SHA-256 verified                          -> matched retest_checkpoint_inventory.csv before measuring
[x] n values present for every statistic                 -> every table and every CSV column carries n
[x] policy probabilities separated from action frequency  -> separate column families, separate sections
[x] legality populations reported separately             -> Populations A/B/C/D with explicit definitions
[x] corrected detail artifact is machine-readable        -> strict JSON parse, 378 frames, 4 results
[x] summary CSV is internally consistent                 -> independent checker exit code 0
```

Every gate passed, so a descriptive interpretation is permitted. It is kept strictly proportional to the
sample sizes in §7 and states no causal or dynamical mechanism.

### 9.2 Descriptive read-out

- **Population A (all pre-step team-possession ticks; n = 74 / 121 / 91 / 92):** mean π_PASS+SHOT is
  0.729 / 0.337 / 0.435 / 0.470. Part of each of these means is contributed by retained ticks on which
  agent 0 personally did not hold the ball (7 / 44 / 18 / 31 ticks), where π_PASS and π_SHOT are exactly 0.
- **Populations B/C/D (agent 0 individually in possession; n = 67 / 77 / 73 / 61):** the same quantity,
  restricted to ticks where PASS and/or SHOT were actually playable, is
  0.805 / 0.529 / 0.543 / 0.709 (π_PASS+SHOT | either-legal), with mean π_PASS | PASS-legal of
  0.789 / 0.486 / 0.477 / 0.669.
- **π_SHOT is small in every seed and every population:** 0.015–0.052 in Population A and
  0.016–0.065 in Population C. Where the ball-play slots are legal, the actor's mass is dominated by the
  PASS group rather than by SHOT; this holds for all four seeds under this protocol and sample size.
- **Selected-action behaviour (separate statistic):** deterministic PASS+SHOT selection covers
  61/74, 72/121, 68/91 and 58/92 of retained ticks. The selected action is always consistent with the
  pre-step mask and equals the argmax of the pre-step masked logits for every one of the 378 frames
  (independently re-derived by `training/verify_prestep_measurement.py`).
- Seed-to-seed differences in the means above (e.g. π_PASS | PASS-legal 0.477 vs 0.789) sit on 61–77
  frames; a single frame moves such a mean by roughly 0.01–0.02, so these are descriptive differences at
  this sample size, not established seed-level properties.

### 9.3 Explicit non-claims

- No statement is made about consolidation dynamics, actor collapse, policy sufficiency, occupancy
  causality, cross-seed behavioural mechanisms, other-agent coordination, mask defects, or reward dynamics.
- The frequency-reweight policy-sufficiency verdict is **not** reopened or revised: it remains closed as
  previously recorded, and this measurement does not change it.
- No further training is recommended or authorised by this pass.
- No claim is made that the old population was behaviourally "wrong"; only that it was temporally
  misaligned and therefore not comparable to this one.

## 10. Artifact Integrity, Verification and Reproduction

Independent checker: `python training/verify_prestep_measurement.py` → **exit code 0**
(`CONSISTENCY CHECK PASSED - artifacts are internally consistent`). It re-derives every reported number
from the raw per-frame records in the detail JSON, using its own masked-softmax implementation rather than
the collector's helper.

| Task 12 check | Result |
|---------------|--------|
| strict `json.loads` of the detail JSON (with `parse_constant` rejecting `Infinity`/`NaN`) | PASS |
| no `Infinity` / `NaN` value literals in the artifact text | PASS |
| detail JSON parses; both CSVs parse with `csv.DictReader` | PASS |
| row counts = expected checkpoints (4) | PASS |
| `checkpoint_sha256` non-empty in every row and equal to the committed inventory | PASS |
| `n` values reconcile with the raw detail data (frames, episodes, aggregates) | PASS |
| stored π equals independently recomputed π from raw logits + mask | PASS (`max_abs_err ≤ 1e-6`, float32-vs-float64 tolerance) |
| deterministic action equals the argmax of the recomputed distribution for all 378 frames | PASS |
| summary + occupancy CSVs equal the recomputed values | PASS |
| reproducibility: second full run produced a byte-identical summary CSV | PASS |

**Reproduction commands**

```bash
# 1. temporal-alignment gate (no bridge required)
python training/tests/test_prestep_onball_temporal_alignment.py

# 2. corrected measurement (starts the headless TS bridge, port 5050, verified 4/4 checkpoints)
python training/eval_post_reweight_logits_prestep.py

# 3. independent consistency check of the emitted artifacts
python training/verify_prestep_measurement.py
```

`--output-suffix` appends a suffix to every artifact name (used for a throwaway single-episode smoke run
that was not committed); `--num-episodes`, `--seeds`, `--base-seed`, `--scenario`, `--bridge-port`,
`--model-dir` and `--inventory` reproduce or vary the protocol. `--stochastic` exists but is **not** the
canonical protocol.

**Files**

| Path | Role |
|------|------|
| `training/prestep_onball.py` | Production pre-step predicate, frame builder, aggregator (measurement-only) |
| `training/eval_post_reweight_logits_prestep.py` | Corrected collector + runners + writers |
| `training/verify_prestep_measurement.py` | Independent artifact consistency checker |
| `training/tests/test_prestep_onball_temporal_alignment.py` | Synthetic regression gate (9 tests) |
| `training/results/post_reweight_logit_prestep_detail.json` | Per-frame detail (378 frames, strict JSON) |
| `training/results/post_reweight_logit_prestep_summary.csv` | Per-seed aggregates incl. provenance and all `n` |
| `training/results/onball_occupancy_prestep_summary.csv` | Occupancy table |
| `training/results/post_reweight_logit_summary.csv`, `training/results/onball_occupancy_summary.csv`, `training/results/POST_REWEIGHT_LOGIT_SNAPSHOT.md`, `training/results/ONBALL_OCCUPANCY_CHECK.md`, `training/results/post_reweight_logit_detail.json`, `training/eval_post_reweight_logits.py` | **Historical artifacts, left unmodified and un-rewritten** |

**Non-modification confirmation.** The defect-bearing historical collector `training/eval_post_reweight_logits.py`
is left byte-identical so the `3428b96` / `5908073` artifacts remain reproducible. Nothing in
`reward / GAE / masks / actor / critic / optimiser / entropy / learning rate / action taxonomy / environment /
training horizon / checkpoint weights` was changed, and no model was trained.





