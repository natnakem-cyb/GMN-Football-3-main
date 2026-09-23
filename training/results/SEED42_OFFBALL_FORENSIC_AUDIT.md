# SEED-42 OFF-BALL CLAIM — FORENSIC ANALYTICAL AUDIT, ERROR IDENTIFICATION, AND EVIDENCE-BACKED CORRECTION

Audit type: read-only forensic re-examination of the Seed-42 "off-ball PASS+SHOT"
clarification (`training/results/SEED42_OFFBALL_CLAIM_CLARIFICATION.md`, untracked,
commit-less at the time of writing).
Audit date: 2026-09-21.
Canonical artifact audited: `training/results/post_reweight_logit_prestep_reconciled_detail.json`
(326,583,222 bytes, 315,197,111 characters, 30,600 frames).
Repository HEAD at audit start: `fdfe7355c1e046b63f8c4a71f8401a30d49963f4`.

---

## 1. Executive finding

**The prior clarification's raw observations about the Seed-42 off-ball set are correct.
Its causal interpretation and its derived conclusions are not.**

Confirmed raw facts (independently reproduced from the canonical detail JSON):

* Seed 42 records 7,650 agent-decisions, 57 PASS+SHOT selections, `n_onball = 13`,
  and exactly 49 recorded off-ball PASS+SHOT selections.
* All 49 recorded off-ball PASS+SHOT frames are at `tick = 0`, `agent_index = 0`
  (`left_1`), `action_taken = 10` (`HIGH_PASS`), `mask_sum = 18`,
  `pre_step_ball_owner_agent_idx = 255`, `pre_step_obs95 = 1.0`,
  `team_has_ball = True`, `event_code = 5`.

Refuted / unsupported interpretations:

* The tick-0 action mask is **not** an "all-ones" mask and **not** a synthetic
  PettingZoo fallback. The recorded vector is
  `[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1]` — 18 of 19 actions legal, with index 16
  (`SLIDING`) illegal. This is exactly the engine's **possession mask** produced by
  `ObservationEncoder.getActionMask` when the acting player has the ball
  (`src/engine/ObservationEncoder.ts:394-424`), transmitted by the bridge's reset
  path (`training/bridge_server.ts:467-530`).
* No "synthetic all-ones mask" code path exists in the reset path. The Python
  fallback used when a mask is absent is **fail-closed** (`mask[0:9]=1`,
  `mask_sum=9`, `training/mappo_rollout.py:64-77`); no such mask appears in the
  artifact at all (`mask_sum=9` count: 0).
* The 49 frames are **valid canonical decision frames**: zero canonical-scope
  violations, zero temporal-alignment mismatches, zero non-finite values, all
  actions produced by the actor's deterministic masked argmax and executed by
  `env.step()`.
* The 49 frames are **not genuinely off-ball**. Independent of the recorded
  `onball` field, the same 49 frames carry the engine's possession mask, and all
  150 tick-0 frames carry `pre_step_obs95 == 1.0`, which the encoder can only emit
  when `ball.ownerId` resolves to a left-team player.
* The recorded `pre_step_ball_owner_agent_idx = 255` at tick 0 is explained by a
  **wrapper reset-path bookkeeping defect**: `GMNMultiAgentEnv.step` empties
  `self.agents` on a terminal tick (`training/gmn_pettingzoo.py:2106-2107`), while
  `GMNMultiAgentEnv.reset` resolves the reset-time owner with
  `self.agents.index(owner_id)` *before* refreshing `self.agents`
  (`training/gmn_pettingzoo.py:1413-1422` vs `1431-1432`). Only the first episode
  (whose reset follows the constructor's warm-up reset at line 542-546) resolves
  correctly; every later episode falls through to the `255` default.
* Therefore the "8 genuine PASS+SHOT" statement is **not supported**: the 57
  recorded PASS+SHOT selections are 57 policy actions by agent 0, of which 50 occur
  on the kickoff frame where the engine granted agent 0 possession (the scenario
  defines `setup.ball.ownerId = 'left_1'`) and 7 occur at `tick > 0` with
  `pre_step_ball_owner_agent_idx = 0`.
* The claim `0.7451% = 8/7650` is **arithmetically incorrect**. `8/7650 = 0.104575%`;
  `0.7451%` is `57/7650`.
* The claim that the reconciliation CSV "excludes these artifacts" is **not
  supported**: the same CSV row reports `team_wide_canonical_rate_pct =
  0.7450980392156863` (= 57/7650, all PASS+SHOT) alongside
  `team_wide_n_ps_selected = 8` (on-ball-restricted). It carries both statistics.
* The "fully explained by agent-0 occupancy arithmetic, with no off-ball
  contribution" verdict is **not supported as stated**, because 49/7650 =
  0.640523% of the 0.745098% total is labelled off-ball by the artifact. It
  becomes defensible only under the corrected reading that those 49 frames are
  engine-possession frames mislabelled by the recorded `onball` field.

**Canonical numbers are unchanged.** The audit does not alter any canonical
artifact, and no canonical metric is recomputed or replaced.

---

## 2. Scope and prohibitions observed

| Prohibition | Status |
| --- | --- |
| No training | not performed |
| No change to reward / GAE / masks / actor / critic / optimizer / entropy / env / taxonomy / scenarios / architecture / checkpoint weights | none made |
| `base_seed` unchanged | unchanged (500000) |
| Canonical scenario unchanged | `academy_3_vs_1_with_keeper_onball` |
| Canonical episode count unchanged | 50 |
| Canonical 7650-decision scope unchanged | 7650 |
| Collector not re-run; no new data collected | not run |
| Canonical detail JSON / summary CSV / reconciliation CSV not regenerated | untouched (byte-identical; see §30) |
| No new definition of `onball` introduced | none — the audit distinguishes the *recorded* field from the *engine-derived* possession indicator and never rewrites the field |
| 49 records not declared invalid for convenience | they are classified as in-scope and validated |
| Existing Seed-42 analysis not deleted or overwritten | `SEED42_OFFBALL_CLAIM_CLARIFICATION.md` and `analyze_offball_mechanism.py` left byte-identical |
| Hypotheses not treated as facts | §23/§24/§25 split proven facts, supported inference, unresolved questions |
| Prior markdown / chat output not used as proof of raw-data claims | all claims here derive from the artifact plus code inspection |

---

## 3. Canonical definitions (unchanged)

```text
canonical scope          : 50 episodes x 51 ticks x 3 controlled agents = 7650 agent-decisions
PASS action IDs          : 9, 10, 11        (LONG_PASS, HIGH_PASS, SHORT_PASS)
SHOT action ID           : 12               (SHOT)
PASS+SHOT                : {9, 10, 11, 12}
onball                   : (pre_step_ball_owner_agent_idx == agent_index)
team_has_ball            : (pre_step_obs95 == 1.0)
canonical rate           : ((passes + shots) / (num_episodes * 51 * 3)) * 100
```

Sources: `training/eval_canonical_three_agent_measurement.py:26-29, 68-84, 479-493`,
`training/prestep_onball.py:74-86`, and the artifact's own `schema` block.

Additional engine semantics relevant to this audit (not new definitions — they are
the environment's own contract, quoted from source):

```text
hasPossession    : player.hasBall || engine.ball.ownerId === player.id
BALL_ACTIONS     : {9, 10, 11, 12}  -> legal iff hasPossession
TACKLE (index 16): legal iff !hasPossession
```

`src/engine/ObservationEncoder.ts:394-424`.

The two masks the engine can emit for a controlled left player are therefore:

```text
possession mask   : [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1]   sum 18, illegal {16}
non-possession    : [1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1]   sum 15, illegal {9,10,11,12}
```

Both vectors are observed in the artifact (§9).

---

## 4. Analysis-script audit — `training/analyze_seed42_disconnect.py`

The script (added at HEAD, commit `fdfe735`, 54 lines) contains one computation
block and one claim block. Complete frame classification (lines 19-25):

```python
for seed in [42, 123, 7, 999]:
    seed_frames = by_seed[seed]
    n_total = len(seed_frames)
    n_onball = sum(1 for f in seed_frames if f['onball'])
    n_offball = n_total - n_onball
    n_selected_ps_onball = sum(1 for f in seed_frames if f['onball'] and f['action_taken'] in PASS_SHOT_ACTION_IDS)
    n_pass_shot_total = sum(1 for f in seed_frames if f['action_taken'] in PASS_SHOT_ACTION_IDS)
    n_selected_ps_offball = n_pass_shot_total - n_selected_ps_onball
```

Claim block (lines 36-41):

```python
print(f'  n_onball=13 ({13/7650*100:.3f}% of all decisions)')
print(f'  n_selected_ps_onball=8 (61.54% conditional rate)')
print(f'  n_pass_shot_total=57 (0.745% unconditional rate)')
print(f'  n_selected_ps_offball={57-8} ({49/7650*100:.3f}% of all decisions)')
print(f'  Off-ball PASS+SHOT selections: {49} out of 57 total (85.96%)')
```

Answers to the twelve required questions:

1. **Frame classification**: the stored canonical `onball` field combined with
   `action_taken` membership in `{9,10,11,12}`. Nothing else.
2. **Total PASS+SHOT**: `n_pass_shot_total = count(action_taken in {9,10,11,12})`.
   **On-ball PASS+SHOT**: `n_selected_ps_onball = count(onball and action_taken in
   {9,10,11,12})`. **Off-ball PASS+SHOT**: `n_selected_ps_offball =
   n_pass_shot_total - n_selected_ps_onball` (a subtraction of the two above, not an
   independent count).
3. Does it use `tick`, `agent_index`, `mask_sum`, reset state, or any other field to
   classify a frame as invalid? **No.** None of those fields is referenced anywhere
   in the script.
4. Does it exclude tick 0? **No.**
5. Does it exclude reset frames? **No** (it has no notion of a reset frame).
6. Does it exclude synthetic masks? **No** (it never reads `action_mask`/`mask_sum`).
7. Does it exclude any particular agent? **No.**
8. Does it exclude frames where the ball owner is `255`? **No** (that value is never
   read).
9. Does it use the canonical `onball` field or recompute its own definition? **It
   uses the stored canonical `onball` field verbatim**; it never recomputes it from
   `pre_step_ball_owner_agent_idx`.
10. Is its `49` a raw count or literally `57 - 8`? **Both, on different lines**: line
    25 computes it as the difference of two genuine raw counts, line 40 prints the
    literal `{57-8}`. The value 49 is nevertheless a true raw-frame count, verified
    independently in §5/§6 as
    `count((not onball) and action_taken in {9,10,11,12}) == 49`.

### Required conclusion

> **Does `analyze_seed42_disconnect.py` itself establish that the 49 off-ball
> selections are tick-0 synthetic artifacts?**
>
> **NO.**
>
> The script contains no tick predicate, no agent predicate, no mask predicate, no
> reset predicate and no validity predicate. It cannot establish anything about
> tick 0, mask provenance, or frame validity; it merely partitions PASS+SHOT
> selections by the stored `onball` field. The prior clarification's sentence
> "The analysis script counts ALL PASS+SHOT selections including tick=0 frames with
> synthetic PettingZoo masks" attributes to this script behaviour it does not have:
> the script's only "tick=0" content is that the data happens to contain tick-0
> frames, and its only "mask" content is nothing at all.


---

## 5. Raw inventory of all 49 recorded off-ball PASS+SHOT frames

Construction (executed by `training/forensic_seed42_offball_inventory.py`, which
streams the canonical artifact and never calls `json.load` on the whole document):

```python
seed42_frames = [f for f in frames if f['seed'] == 42]
offball_ps_frames = [f for f in seed42_frames
                    if (not f['onball']) and f['action_taken'] in {9, 10, 11, 12}]
assert len(offball_ps_frames) == 49      # verified independently
```

Independent verification output: `len(seed42_offball_ps) = 49`, and
`len(seed42_all_ps) == len(seed42_onball_ps) + len(seed42_offball_ps)`
(`57 = 8 + 49`).

Every record below has `seed = 42`, `agent_id = left_1`, and the same
`checkpoint = training\models\mappo_academy_3_vs_1_with_keeper_onball_seed42_actorreweight.pt`,
`checkpoint_timesteps = 49920`,
`checkpoint_sha256 = eb9e1403b6a65c3c288397b752c64319cf8f9022116492fa05a355c08682fd24`,
`scenario = academy_3_vs_1_with_keeper_onball`, `deterministic = true`,
`code_commit = 472b92de6e768756f62669af58add3fce48b1091`.

### 5.1 Decision-identity table (all 49 rows)

| # | seed | episode | tick | agent_index | agent_id | action_taken | action_taken_name | ep_seed |
| -: | -: | -: | -: | -: | --- | -: | --- | -: |
| 1 | 42 | 1 | 0 | 0 | left_1 | 10 | HIGH_PASS | 501009 |
| 2 | 42 | 2 | 0 | 0 | left_1 | 10 | HIGH_PASS | 502018 |
| 3 | 42 | 3 | 0 | 0 | left_1 | 10 | HIGH_PASS | 503027 |
| 4 | 42 | 4 | 0 | 0 | left_1 | 10 | HIGH_PASS | 504036 |
| 5 | 42 | 5 | 0 | 0 | left_1 | 10 | HIGH_PASS | 505045 |
| 6 | 42 | 6 | 0 | 0 | left_1 | 10 | HIGH_PASS | 506054 |
| 7 | 42 | 7 | 0 | 0 | left_1 | 10 | HIGH_PASS | 507063 |
| 8 | 42 | 8 | 0 | 0 | left_1 | 10 | HIGH_PASS | 508072 |
| 9 | 42 | 9 | 0 | 0 | left_1 | 10 | HIGH_PASS | 509081 |
| 10 | 42 | 10 | 0 | 0 | left_1 | 10 | HIGH_PASS | 510090 |
| 11 | 42 | 11 | 0 | 0 | left_1 | 10 | HIGH_PASS | 511099 |
| 12 | 42 | 12 | 0 | 0 | left_1 | 10 | HIGH_PASS | 512108 |
| 13 | 42 | 13 | 0 | 0 | left_1 | 10 | HIGH_PASS | 513117 |
| 14 | 42 | 14 | 0 | 0 | left_1 | 10 | HIGH_PASS | 514126 |
| 15 | 42 | 15 | 0 | 0 | left_1 | 10 | HIGH_PASS | 515135 |
| 16 | 42 | 16 | 0 | 0 | left_1 | 10 | HIGH_PASS | 516144 |
| 17 | 42 | 17 | 0 | 0 | left_1 | 10 | HIGH_PASS | 517153 |
| 18 | 42 | 18 | 0 | 0 | left_1 | 10 | HIGH_PASS | 518162 |
| 19 | 42 | 19 | 0 | 0 | left_1 | 10 | HIGH_PASS | 519171 |
| 20 | 42 | 20 | 0 | 0 | left_1 | 10 | HIGH_PASS | 520180 |
| 21 | 42 | 21 | 0 | 0 | left_1 | 10 | HIGH_PASS | 521189 |
| 22 | 42 | 22 | 0 | 0 | left_1 | 10 | HIGH_PASS | 522198 |
| 23 | 42 | 23 | 0 | 0 | left_1 | 10 | HIGH_PASS | 523207 |
| 24 | 42 | 24 | 0 | 0 | left_1 | 10 | HIGH_PASS | 524216 |
| 25 | 42 | 25 | 0 | 0 | left_1 | 10 | HIGH_PASS | 525225 |
| 26 | 42 | 26 | 0 | 0 | left_1 | 10 | HIGH_PASS | 526234 |
| 27 | 42 | 27 | 0 | 0 | left_1 | 10 | HIGH_PASS | 527243 |
| 28 | 42 | 28 | 0 | 0 | left_1 | 10 | HIGH_PASS | 528252 |
| 29 | 42 | 29 | 0 | 0 | left_1 | 10 | HIGH_PASS | 529261 |
| 30 | 42 | 30 | 0 | 0 | left_1 | 10 | HIGH_PASS | 530270 |
| 31 | 42 | 31 | 0 | 0 | left_1 | 10 | HIGH_PASS | 531279 |
| 32 | 42 | 32 | 0 | 0 | left_1 | 10 | HIGH_PASS | 532288 |
| 33 | 42 | 33 | 0 | 0 | left_1 | 10 | HIGH_PASS | 533297 |
| 34 | 42 | 34 | 0 | 0 | left_1 | 10 | HIGH_PASS | 534306 |
| 35 | 42 | 35 | 0 | 0 | left_1 | 10 | HIGH_PASS | 535315 |
| 36 | 42 | 36 | 0 | 0 | left_1 | 10 | HIGH_PASS | 536324 |
| 37 | 42 | 37 | 0 | 0 | left_1 | 10 | HIGH_PASS | 537333 |
| 38 | 42 | 38 | 0 | 0 | left_1 | 10 | HIGH_PASS | 538342 |
| 39 | 42 | 39 | 0 | 0 | left_1 | 10 | HIGH_PASS | 539351 |
| 40 | 42 | 40 | 0 | 0 | left_1 | 10 | HIGH_PASS | 540360 |
| 41 | 42 | 41 | 0 | 0 | left_1 | 10 | HIGH_PASS | 541369 |
| 42 | 42 | 42 | 0 | 0 | left_1 | 10 | HIGH_PASS | 542378 |
| 43 | 42 | 43 | 0 | 0 | left_1 | 10 | HIGH_PASS | 543387 |
| 44 | 42 | 44 | 0 | 0 | left_1 | 10 | HIGH_PASS | 544396 |
| 45 | 42 | 45 | 0 | 0 | left_1 | 10 | HIGH_PASS | 545405 |
| 46 | 42 | 46 | 0 | 0 | left_1 | 10 | HIGH_PASS | 546414 |
| 47 | 42 | 47 | 0 | 0 | left_1 | 10 | HIGH_PASS | 547423 |
| 48 | 42 | 48 | 0 | 0 | left_1 | 10 | HIGH_PASS | 548432 |
| 49 | 42 | 49 | 0 | 0 | left_1 | 10 | HIGH_PASS | 549441 |

### 5.2 Ownership / mask / status table (all 49 rows)

| # | onball | agent_has_ball | team_has_ball | pre_step_obs95 | pre_step_ball_owner_agent_idx | mask_sum | mask_pass_legal | mask_shot_legal | mask_pass_or_shot_legal |
| -: | --- | --- | --- | -: | -: | -: | -: | -: | -: |
| 1 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 2 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 3 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 4 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 5 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 6 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 7 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 8 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 9 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 10 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 11 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 12 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 13 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 14 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 15 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 16 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 17 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 18 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 19 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 20 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 21 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 22 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 23 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 24 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 25 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 26 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 27 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 28 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 29 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 30 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 31 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 32 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 33 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 34 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 35 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 36 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 37 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 38 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 39 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 40 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 41 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 42 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 43 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 44 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 45 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 46 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 47 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 48 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |
| 49 | False | False | True | 1.0 | 255 | 18 | 1 | 1 | 1 |

### 5.3 Action-mask vectors and post-step status (all 49 rows)

`action_mask` is the 19-element pre-step legality vector; `event_code`, `done`,
`terminated` and `truncated` are the post-step diagnostics stored alongside it.
They are not inputs to the actor decision (see §18).

| # | action_mask | event_code | done | terminated | truncated |
| -: | --- | -: | --- | --- | --- |
| 1 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 2 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 3 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 4 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 5 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 6 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 7 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 8 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 9 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 10 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 11 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 12 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 13 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 14 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 15 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 16 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 17 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 18 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 19 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 20 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 21 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 22 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 23 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 24 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 25 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 26 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 27 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 28 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 29 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 30 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 31 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 32 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 33 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 34 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 35 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 36 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 37 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 38 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 39 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 40 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 41 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 42 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 43 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 44 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 45 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 46 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 47 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 48 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |
| 49 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]` | 5 | False | False | False |


---

## 6. Tick distribution (the "all 49 are tick=0" claim)

`Counter(f['tick'] for f in offball_ps_frames)` for Seed 42:

```text
tick=0 : 49
tick>0 : 0
```

```text
Number of off-ball PASS+SHOT frames at tick 0 : 49
Number at tick > 0                            :  0
Minimum tick                                  :  0
Maximum tick                                  :  0
Unique ticks                                  : {0}
```

The same holds for the other three seeds (each has its own 49-frame set):

```text
seed  7 : n_off_PS=49   ticks {0: 49}
seed 42 : n_off_PS=49   ticks {0: 49}
seed 123: n_off_PS=49   ticks {0: 49}
seed 999: n_off_PS=49   ticks {0: 49}
```

**Disposition: CONFIRMED.** The prior statement "All 49 off-ball PASS+SHOT
selections for seed 42 occur at tick=0" is true of the raw data. No frame of the 49
occurs at tick > 0, so the escape clause in the audit brief ("if even one occurs at
tick > 0 the statement is false") does not apply.

Note the prior report's accompanying sentence "Zero off-ball PS selections occur at
tick>0 across all seeds" is also confirmed by the same table.

---

## 7. Agent distribution (the "all 49 are agent 0" claim)

`Counter(f['agent_index'] for f in offball_ps_frames)` for Seed 42:

```text
agent 0 : 49
agent 1 :  0
agent 2 :  0
```

`Counter(f['agent_id'] ...)`: `{'left_1': 49}`.

**Disposition: CONFIRMED.** All 49 belong to `agent_index = 0` (`left_1`). The
adjacent prior claim that "agents 1 and 2 never selected PASS or SHOT while on-ball"
is consistent with the artifact (`agent1_n_selected_ps_when_onball = 0`,
`agent2_n_selected_ps_when_onball = 0` in the summary CSV).

---

## 8. Action distribution (the "all 49 are HIGH_PASS" claim)

`Counter(f['action_taken'] ...)` and `Counter(f['action_taken_name'] ...)` for Seed 42:

```text
action  9 (LONG_PASS)  :  0
action 10 (HIGH_PASS)  : 49
action 11 (SHORT_PASS) :  0
action 12 (SHOT)       :  0

action_taken_name 'HIGH_PASS' : 49
```

**Disposition: CONFIRMED for Seed 42.**

**Disposition: REFUTED as a cross-seed generalisation.** The prior report asserted
that "All 196 off-ball PASS+SHOT selections across all 4 seeds (49 per seed) share
identical characteristics: ... Action HIGH_PASS (action_id=10)". The artifact shows
the action differs by seed:

```text
seed  7 : actions {10: 49}   (HIGH_PASS)
seed 42 : actions {10: 49}   (HIGH_PASS)
seed 123: actions {11: 49}   (SHORT_PASS)
seed 999: actions { 9: 49}   (LONG_PASS)
```

That erroneous generalisation matters, because it was used to argue that the 49 were
a mechanical, policy-independent artifact. A tick-0 selection that varies with the
policy (three different PASS variants across four independently trained
checkpoints) is a policy-driven decision, not a fixed environment artefact.


---

## 9. Mask distribution (the "mask_sum=18 means synthetic / all-ones mask" claim)

Distinct masks among the 49 Seed-42 frames — one single vector, not a family:

```text
mask vector : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
mask_sum    : 18
legal       : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18]
illegal     : [16]   (['SLIDING'])
count       : 49
all_ones    : False
```

Comparison with the rest of the Seed-42 corpus:

```text
mask_sum distribution, ALL seed-42 frames        : {15: 7588, 18: 62}
mask_sum distribution, seed-42 tick>0 frames     : {15: 7488, 18: 12}
mask_sum distribution, seed-42 tick0 frames      : {18: 50 (agent 0), 15: 100 (agents 1-2)}
mask_sum == 9  (the Python fail-closed fallback) : 0 occurrences
```

Findings:

1. **Is the mask genuinely all-ones?** No. It is 18-of-19; index 16 (`SLIDING`) is 0.
   The prior report simultaneously printed the correct 18-of-19 vector and labelled
   it "(all actions legal)" / "synthetic all-ones mask" — two mutually inconsistent
   statements.
2. **Is it a canonical legal mask?** Yes: it is exactly the vector the engine emits
   for a player with possession (`ObservationEncoder.getActionMask`, possession
   branch, `src/engine/ObservationEncoder.ts:395-416`).
3. **Does it differ from the normal masks of non-tick-0 frames?** No: seed 42 has 12
   non-tick-0 frames carrying this exact vector, and those 12 frames are precisely
   its 12 non-tick-0 on-ball frames.
4. **Is it produced by a fallback mechanism?** No. The only fallbacks are
   `_fail_closed_fallback_mask()` (`training/mappo_rollout.py:64-77`, sum 9) and the
   bridge's unknown-player-id fallback (`training/bridge_server.ts:356-366`, sum 9).
   Neither can produce sum 18.
5. **Is there code evidence that it is synthetic?** None. The reset-time mask is
   transmitted by the bridge using the engine's real legality function
   (`training/bridge_server.ts:311, 319, 326, 505, 511, 521`) and stored verbatim by
   the wrapper (`training/gmn_pettingzoo.py:1445-1455`).
6. **Does the environment/bridge explicitly create such a mask at reset?** Yes, and
   non-synthetically: `computeActionMasks` calls
   `ObservationEncoder.getActionMask(player, engine)` per controllable id at reset
   (`training/bridge_server.ts:356-366`); the canonical scenario places the ball on
   `left_1` (`src/scenarios/ScenarioRegistry.ts:163`:
   `ball: { x: 0.2, y: 0, z: 0, ownerId: 'left_1' }`), which
   `GameEngine.loadScenario` applies via `ball.ownerId = owner.id;
   owner.hasBall = true` (`src/engine/GameEngine.ts:493-501`). left_1's reset mask is
   therefore the possession mask.
7. **Is the mask 18-of-19, and which action is illegal?** Yes: `SLIDING` (index 16),
   matching the engine's possession rule
   (`TACKLE_INDEX = 16; mask[16] = hasPossession ? 0 : 1`).

**Disposition: REFUTED.** `mask_sum = 18` is neither an all-ones mask nor a
synthetic mask. Observed fact: an 18-of-19 vector with `SLIDING` illegal.
Code-confirmed mechanism: the engine's possession mask, delivered through the
bridge's reset path.

---

## 10. Ball-owner distribution

`Counter(f['pre_step_ball_owner_agent_idx'] ...)` for the 49 frames:

```text
255 : 49
```

and for **all** tick-0 frames of Seed 42 (`150` frames = 50 episodes x 3 agents):

```text
tick0 owner distribution : {0: 3, 255: 147}
```

`255` is a documented sentinel: `training/gmn_pettingzoo.py:500`
(`self._last_ball_owner_agent_idx: int = 255  # OCCUPANCY-EXP`) and the read site
`training/gmn_pettingzoo.py:1413-1422` / `579-581`. `255` is not "right team", not
"ball in flight" and not "unknown" in the recorded field; in this pipeline it means
"the wrapper did not resolve a left-team owner for this tick".

The three frames with `owner = 0` are the three agent rows of episode 0
(`(agent, owner)` counts per seed:

```text
seed 42: {(0,0):1, (0,255):49, (1,0):1, (1,255):49, (2,0):1, (2,255):49}
```

identical shape for seeds 7, 123 and 999). So exactly one of the 50 episodes
resolved a reset owner; the other 49 did not. See §15 and §24.

---

## 11. Team-possession distribution

`Counter(f['team_has_ball'] ...)` and `Counter(f['pre_step_obs95'] ...)` for the 49:

```text
team_has_ball  : {True: 49}
pre_step_obs95 : {1.0: 49}
pre_step_obs_ball_ownership_slice : {(0.0, 1.0, 0.0): 49}
```

All 150 tick-0 frames, not only the 49:

```text
tick0 obs95 distribution         : {1.0: 150}
tick0 team_has_ball distribution : {True: 150}
```

This is the decisive internal inconsistency of the prior interpretation. `obs[94..96]`
is a team-level one-hot produced **only** from `ball.ownerId`
(`src/engine/ObservationEncoder.ts:57-71, 136-141`): the value `(0.0, 1.0, 0.0)`
(=`obs95 == 1.0`, "left team has the ball") can only be emitted when
`engine.ball.ownerId` resolves to a **left-team player**. A frame therefore cannot
simultaneously (a) carry `obs95 == 1.0` and (b) be a frame in which no player owns
the ball. The recorded `pre_step_ball_owner_agent_idx = 255` in those same frames
contradicts the observation vector recorded next to it — the contradiction is in the
recorded field, not in the environment.


---

## 12. Tick-0 investigation (what tick 0 actually is)

Seed-42 tick-0 population (all agents, all actions):

```text
tick0 frames total                 : 150
tick0 by (agent_index, agent_id)   : {(0,'left_1'):50, (1,'left_2'):50, (2,'left_3'):50}
tick0 episodes                     : 50
tick0 owner distribution           : {0: 3, 255: 147}
tick0 mask-sum distribution        : {18: 50, 15: 100}
tick0 mask VECTOR distribution     : {(1x16,0,1,1): 50, (1x9,0,0,0,0,1,1,1,1,1,1): 100}
tick0 obs95 distribution           : {1.0: 150}
tick0 onball distribution          : {True: 1, False: 149}
tick0 temporal_aligned             : {True: 150}
tick0 action distribution (agent 0)   : {10: 50}
tick0 action distribution (agents 1-2): {0: 100}
```

Key structural facts about tick 0:

* Tick 0 is not a "reset frame" that the collector skipped or synthesised: it is the
  **first recorded agent decision of every episode**, taken from the state returned
  by `env.reset(...)` (see §16).
* Tick 0 is present for **all three agents**, in **all 50 episodes**, of **all four
  seeds** (150 frames per seed; 600 tick-0 frames in the artifact).
* The masks at tick 0 are agent-specific and are the two real engine masks:
  agent 0 receives the possession mask (sum 18) in every episode; agents 1 and 2
  receive the non-possession mask (sum 15) in every episode. A single "synthetic
  reset mask" cannot be agent-specific, and neither of these is all-ones.
* `pre_step_obs95 == 1.0` in all 150 tick-0 frames, i.e. the observation the policy
  actually consumed at tick 0 encodes left-team possession in every episode.
* Only 1 of the 150 tick-0 frames is recorded as on-ball, although 50 of them are
  agent-0 frames in which the engine mask grants ball actions. The 49-frame
  "off-ball" set is therefore exactly "agent-0 tick-0 frames of episodes 1..49".

---

## 13. Agent-0 investigation

Agent 0 (`left_1`) is the scenario's controlled CAM
(`src/scenarios/ScenarioRegistry.ts:165`: `{ role: 'CAM', pos: { x: 0.2, y: 0 }, isControlled: true }`)
and the scenario's designated kickoff ball carrier (`ScenarioRegistry.ts:163`).
Consequences verified in the artifact:

* agent 0 holds the possession mask at tick 0 in 50/50 episodes (per seed);
* agent 0 is the only agent that selected PASS+SHOT at Seed-42 tick 0
  (agents 1-2 selected action 0 = IDLE in all 100 tick-0 rows);
* agents 1-2 selected PASS+SHOT **zero** times over the whole Seed-42 episode set
  (`PASS+SHOT selections by agent_index, per seed: seed 42 -> {0: 57}`), so Seed-42's
  entire canonical numerator is agent 0's.

For completeness, the same statistic for the other seeds is
`seed 7 -> {0:59, 1:5, 2:2}`, `seed 123 -> {0:74, 1:17, 2:37}`,
`seed 999 -> {0:57, 1:11, 2:11}`; agent 0's dominance is seed-42-specific.

---

## 14. HIGH_PASS investigation

The Seed-42 49-frame set is 100% `action_taken = 10` / `HIGH_PASS` (§8), and every
one of those frames has `deterministic_action_is_pass_shot = True`, consistent with
the collector deriving `action_taken` directly from the actor's deterministic
selection (`training/eval_canonical_three_agent_measurement.py:259, 294, 587-590`).

`HIGH_PASS` at tick 0 is legal for agent 0 in every episode, because the mask the
engine produced for agent 0 at kickoff grants indices 9-12
(`mask_pass_legal = 1`, `mask_shot_legal = 1`, `mask_pass_or_shot_legal = 1` for all
49). It is **not** legal for agents 1 and 2 (their masks clear indices 9-12), and
they duly did not select it.

The cross-seed variation (§8: seed 123 selects SHORT_PASS, seed 999 selects
LONG_PASS) shows that the tick-0 PASS choice is a policy property of each trained
checkpoint, not a fixed environment behaviour.

---

## 15. Synthetic-mask investigation (mechanism)

This section answers the eleven questions of the audit brief by separating
**observed fact**, **possible interpretation** and **code-confirmed mechanism**.

**Observed facts (artifact only)**

* the 49 masks are the single 18-of-19 vector with `SLIDING` illegal;
* the same vector occurs on the 12 non-tick-0 on-ball frames of the same seed;
* the complementary 15-of-19 vector (PASS/SHOT illegal) is used everywhere else;
* `mask_sum = 9` (the Python fail-closed fallback) never occurs;
* all 49 frames have `pre_step_obs95 = 1.0` with ownership slice `(0, 1, 0)`;
* all 150 tick-0 frames of every seed have `pre_step_obs95 = 1.0`.

**Possible interpretations that are NOT supported**

* "the wrapper synthesises all-ones masks at reset" — no all-ones mask exists in the
  artifact, and the wrapper's only synthesis path is fail-closed (sum 9);
* "the mask does not reflect true action legality" — the mask is byte-identical to
  the mask the engine emits under possession, i.e. it does reflect legality;
* "the mask is produced before the first physics step, so legality is unknown" — the
  mask is computed from the engine's reset state (`hasPossession` from the scenario's
  `ownerId`), which is fully determined at reset.

**Code-confirmed mechanism**

1. Scenario definition: `setup.ball.ownerId = 'left_1'`
   (`src/scenarios/ScenarioRegistry.ts:163`).
2. `GameEngine.loadScenario` applies it: `ball.ownerId = owner.id; owner.hasBall = true`
   (`src/engine/GameEngine.ts:493-501`).
3. Bridge reset builds per-agent masks from engine state:
   `computeActionMasks` -> `ObservationEncoder.getActionMask(player, engine)`
   (`training/bridge_server.ts:356-366, 311`), served by the `type: 'reset'` handler
   (`training/bridge_server.ts:1386-1388` -> `bridge.reset`, lines 467-530).
4. Mask rule: `hasPossession = player.hasBall || engine.ball.ownerId === player.id`;
   ball actions legal iff possession; TACKLE illegal iff possession
   (`src/engine/ObservationEncoder.ts:394-424`).
5. Wrapper stores the transmitted mask unchanged
   (`training/gmn_pettingzoo.py:1445-1455`, consumed at `551-552` and `574`).

Conclusion: the tick-0 mask is a **genuine engine legality mask** reflecting real
possession, not a synthetic artifact.


### 15.1 The recorded `owner = 255` at tick 0 — code-confirmed defect

The single remaining anomaly in the 49 frames is that
`pre_step_ball_owner_agent_idx` is recorded as `255` even though the same frame's
observation encodes left-team possession and its mask encodes individual possession.
The code path explains it:

```text
training/gmn_pettingzoo.py:1413-1422   reset(): reads data["info"]["current_ball_owner"]
                                       and resolves it with self.agents.index(owner_id),
                                       falling back to 255 when the id is not in
                                       self.agents  (self.agents is NOT yet refreshed)
training/gmn_pettingzoo.py:1431-1432   reset(): only AFTERWARDS sets
                                       self.possible_agents / self.agents from the
                                       reset response's controllableAgentIds
training/gmn_pettingzoo.py:2106-2107   step(): "if shared_term or shared_trunc:
                                       self.agents = []"  -> the list is emptied at
                                       the end of every episode
training/gmn_pettingzoo.py:542-546     __init__(): performs one warm-up reset
                                       ("Perform initial reset to discover
                                       controllable agents"), which is what leaves
                                       self.agents populated for episode 0's reset
```

Predicted and observed consequences:

| Episode | `self.agents` at reset time | correct resolution possible? | recorded owner |
| --- | --- | --- | --- |
| 0 (first collected episode, after the constructor warm-up reset) | populated | yes | `0` |
| 1..49 (previous episode ended with `self.agents = []`) | empty | no, falls back | `255` |

This reproduces the artifact exactly: exactly 1 of 50 episodes records a resolved
owner, and it is the first (episode 0), for all four seeds and for all three agents.
The artifact also shows that **every** episode ended with a terminal tick
(`Tick distribution of frames with done==True: {50: 150}` per seed), i.e. the
`self.agents = []` branch was taken at the end of all 50 episodes.

Consequence: at tick 0, the fields `pre_step_ball_owner_agent_idx`, `agent_has_ball`
and `onball` are **not reliable** for episodes 1..49; the observation
(`pre_step_obs95`) and the action mask *are* reliable, because both are computed
engine-side from the real ball owner.

Nothing in this section changes any canonical artifact. It is a diagnosis of a
recorded field, and it is the reason the "49 off-ball" set exists at all.

---

## 16. Canonical tick-0 validity investigation

Trace of the canonical collector
`training/eval_canonical_three_agent_measurement.py::collect_canonical_measurement`
(lines 523-681):

1. **`tick_idx` initialisation**: `tick_idx = 0` immediately after `env.reset(...)`
   (line 561). No offset, no skip.
2. **Pre-step observation**: `local_obs = np.stack([obs_dict[a] for a in
   current_agents])` from the dict returned by `env.reset`, re-assigned from
   `env.step` after every step (lines 571-573, 593-595).
3. **Pre-step mask**: `mask_matrix = _mask_matrix(current_ep_masks, current_agents)`
   where `current_ep_masks = unwrap_masks(obs_dict)` (lines 552, 574).
4. **`pre_step_ball_owner_agent_idx`**: `int(getattr(env,
   "_last_ball_owner_agent_idx", 255))`, read *before* `env.step()` (lines 576-581).
5. **Is the actor run at tick 0?** Yes — `_batched_actor_quantities(actor, local_obs,
   mask_matrix)` is invoked unconditionally inside the loop (line 584) on the reset
   observation and reset mask.
6. **Does `env.step()` occur?** Yes, after the pre-step capture (line 593).
7. **Is the tick-0 decision recorded?** Yes — one frame per agent is appended on
   every loop iteration, including the first (lines 609-633).
8. **Does the tick-0 decision contribute to `n_decisions`?** Yes:
   `"n_decisions": len(all_decisions)` (line 677) counts it, and
   `n_ticks = 2550 = 50 episodes x 51 ticks` confirms ticks 0..50 are counted.
9. **Does the canonical denominator include it?** Yes:
   `canonical_denominator = num_episodes * DEFAULT_TICKS_PER_EPISODE * NUM_AGENTS`
   with `DEFAULT_TICKS_PER_EPISODE = 51` (lines 81, 1144, 1212) = 7650; the artifact
   schema states the same contract.
10. **Is there any "discard initial state" logic?** None. There is no skip, no
    `continue` on tick 0, no filter by tick anywhere in the collector. The retained-
    frames contract is explicitly the opposite: the artifact schema records
    `"retention_rule": "retain ALL agent decisions; on-ball =
    pre_step_ball_owner_agent_idx == agent_index"`.

### Required answer

> **Is tick 0 an invalid artefact under the canonical measurement contract, or is
> tick 0 an intentionally included agent-decision frame?**

**Tick 0 is an intentionally included agent-decision frame.** The contract counts
51 ticks per episode (`0..50`), the collector records the first pre-step decision
without exception, and `n_decisions = len(all_decisions)` includes it. Tick 0 is
**not** removed from any canonical metric by this audit.


---

## 17. Canonical-scope investigation

For each of the 49 records the audit evaluated: correct seed; episode within 0..49;
tick within the canonical rollout (0..50); `agent_index` in 0..2; action actually
emitted by the actor; pre-step observation present; pre-step mask present and
19-dimensional; frame recorded by the canonical collector; canonical checkpoint
identity (path, SHA-256, `checkpoint_timesteps = 49920`); canonical scenario;
`deterministic = true`; `ep_seed == 500000 + episode * 1009`; action equal to the
masked pre-step argmax.

```text
scope violations found: 0
```

Per-field conformance (all 49 rows):

```text
checkpoint_timesteps : {49920: 49}
deterministic_flag   : {True: 49}
scenario             : {'academy_3_vs_1_with_keeper_onball': 49}
checkpoint           : {'training\\models\\mappo_academy_3_vs_1_with_keeper_onball_seed42_actorreweight.pt': 49}
checkpoint_sha256    : {'eb9e1403b6a65c3c288397b752c64319cf8f9022116492fa05a355c08682fd24': 49}
code_commit          : {'472b92de6e768756f62669af58add3fce48b1091': 49}
episode              : {1..49, exactly one frame each}
ep_seed              : {501009..549441, exactly one frame each}
```

The recorded `episode`/`ep_seed` distributions are exhaustive single-occurrence
ranges (episode 1..49, `ep_seed = 500000 + episode*1009`), which also proves that
the 49 frames come from 49 distinct episodes — the episode-0 tick-0 frame is the one
recorded as on-ball and is therefore not part of the 49.

### Required conclusion

> **All 49 are canonical decision frames.** None of the 49 falls outside the
> canonical decision scope. "Off-ball" is not equivalent to "out of scope", and no
> scope rule in the collector, the summary CSV or the reconciliation CSV excludes
> tick-0 frames.

---

## 18. Temporal-alignment investigation

Canonical alignment rule (identical to the verifier's,
`training/verify_canonical_artifacts.py:442-456`):

```text
expected = argmax_i probs[i] over i with action_mask[i] == 1
aligned  = (action_taken == expected)
```

Results for the 49 frames:

```text
temporal_aligned distribution (n=49) : {True: 49}
```

and for the whole tick-0 population of Seed 42:

```text
tick0 temporal_aligned : {True: 150}
```

```text
number aligned    : 49
number misaligned :  0
```

The authoritative verifier independently re-checks alignment over all 30,600 frames
and reports `PASS: temporal alignment (action == pre-step masked argmax)`.

The 49 frames are therefore genuine recorded policy decisions — not artifacts of
temporal misalignment, not post-step events recorded as actions, and not
environment-derived actions: `action_taken` is written from
`actor_q["deterministic_action"]` (the masked pre-step argmax) and the identical
vector is what the collector passes to `env.step(action_dict)`
(`training/eval_canonical_three_agent_measurement.py:587-590` and `:259`).

---

## 19. Raw vs summary vs reconciliation comparison

Seed-42 values, reconstructed from the raw detail (this audit) versus the committed
artifacts (read, not regenerated):

| Quantity | Raw detail (this audit) | `..._reconciled_summary.csv` | `..._canonical_scope_reconciliation.csv` |
| --- | ---: | ---: | ---: |
| total PASS+SHOT (`n_pass_shot`) | 57 | 57 (`n_pass=57`, `n_shot=0`) | 57/7650 implied by `team_wide_canonical_rate_pct = 0.7450980392156863` |
| decisions (`n_decisions`) | 7650 | 7650 | 7650 |
| on-ball PASS+SHOT | 8 | 8 (`team_n_selected_ps_onball_total`, `agent0_n_selected_ps_when_onball`) | 8 (`team_wide_n_ps_selected`, `agent_level_onball_ps_selections`) |
| recorded off-ball PASS+SHOT | 49 | not present as a column | not present as a column |
| `n_onball` | 13 | 13 | 13 (`total_controlled_agent_onball_frames`) |
| canonical rate | 0.7450980392156863% | 0.7450980392156863% | 0.7450980392156863% (`team_wide_canonical_rate_pct`, `team_wide_rate_from_raw_decisions`) |

Code that produces those columns:

* `canonical_rate_pct` / `team_wide_canonical_rate_pct`:
  `_compute_canonical_rate` counts **all** PASS+SHOT actions over all 7650 decisions
  (`training/eval_canonical_three_agent_measurement.py:479-493`) — it is written to
  the reconciliation row at line 1107 and to the summary row at line 771.
* `team_wide_n_ps_selected` / `agent_level_onball_ps_selections`: the sum of
  `agent{i}_n_selected_ps_when_onball`
  (`training/eval_canonical_three_agent_measurement.py:1097-1110`) — an on-ball-only
  statistic.

### Required relationship

```text
raw total PASS+SHOT (57) == summary n_pass_shot (57)
raw on-ball PASS+SHOT (8) == summary/reconciliation on-ball PS (8)
raw off-ball PASS+SHOT (49) == raw total (57) - raw on-ball (8)
reconciliation canonical rate == raw total / 7650 == 0.7450980392156863%
```

The reconciliation CSV does **not** "reject" the 49: it publishes the on-ball
conditional statistic *and* the all-PASS+SHOT canonical rate computed from the raw
57. The prior clarification's sentence — "The CSV ... already correctly excludes
these artifacts by counting only on-ball PASS+SHOT selections
(`team_wide_n_ps_selected=8`)" — mischaracterises a co-reported statistic as an
exclusion rule.


---

## 20. Occupancy decomposition (metric unchanged)

Raw-frame decomposition of the canonical unconditional rate for all four seeds
(`N = 7650` in every case):

```text
seed  7: rate = 66/7650 = 0.862745% | onball contribution = 17/7650 = 0.222222% | offball contribution = 49/7650 = 0.640523%
seed 42: rate = 57/7650 = 0.745098% | onball contribution =  8/7650 = 0.104575% | offball contribution = 49/7650 = 0.640523%
seed 123: rate = 128/7650 = 1.673203% | onball contribution = 79/7650 = 1.032680% | offball contribution = 49/7650 = 0.640523%
seed 999: rate = 79/7650 = 1.032680% | onball contribution = 30/7650 = 0.392157% | offball contribution = 49/7650 = 0.640523%
```

Seed 42 two-component decomposition
(`P(PS) = P(onball)·P(PS|onball) + P(offball)·P(PS|offball)`):

```text
N (decisions)                 = 7650
n_onball                      = 13      P(onball)  = 0.0016993464
n_offball                     = 7637    P(offball) = 0.9983006536
on-ball PS selections         = 8
off-ball PS selections        = 49
total PS selections           = 57
P(PS | onball)                = 8/13   = 0.6153846154
P(PS | offball)               = 49/7637 = 0.0064161320
on-ball contribution to rate  = 8/7650  = 0.104575%
off-ball contribution to rate = 49/7650 = 0.640523%
total unconditional rate      = 57/7650 = 0.745098%
decomposition check           = 0.007450980392 vs 0.007450980392  match=True
```

### Required interpretation

1. **How much of the unconditional rate is attributable to on-ball frames?**
   8/7650 = 0.104575% (14.04% of the total rate) under the recorded `onball` field.
2. **How much is attributable to off-ball frames?** 49/7650 = 0.640523% (85.96% of
   the total) under the recorded `onball` field.
3. **Does low on-ball occupancy suppress the on-ball contribution?** Yes, and
   measurably: `P(onball) = 13/7650 = 0.17%`, so even a 61.5% conditional rate can
   only contribute 0.1046 pp. This part of the prior analysis is sound.
4. **Does low on-ball occupancy explain the entire unconditional rate?** **No.** The
   decomposition shows a second, larger term. Removing it would be a silent
   redefinition of the metric.
5. **Do the 49 off-ball selections remain unexplained, explained, or invalid?**
   They are **not invalid** (§17, §18) and they are **explained** — but by a defect in
   the recorded ownership field (§15.1), not by "synthetic masks":

   ```text
   seed  poss_mask  recorded_onball  poss&onball  poss&offball(recorded)  tick0_poss  tick>0_poss  PS_with_illegal_ball
      7         67               18           18                      49          50           17                     0
     42         62               13           13                      49          50           12                     0
    123        145               96           96                      49          50           95                     0
    999         91               42           42                      49          50           41                     0
   ```

   For every seed: `frames whose mask grants ball actions == recorded n_onball + 49`,
   and the extra 49 are exactly the tick-0 frames of episodes 1..49. Zero PASS+SHOT
   selections anywhere in the corpus were made under a mask that forbade ball
   actions. Under the engine's possession semantics the Seed-42 picture becomes:

   ```text
   possession frames (engine mask grants {9,10,11,12}) : 62
   PASS+SHOT selections among them                     : 57
   conditional rate given possession                   : 57/62 = 91.94%
   possession occupancy                                : 62/7650 = 0.810458%
   unconditional rate check: (62/7650)·(57/62) = 57/7650 = 0.745098%
   ```

   This revised decomposition is *still* an occupancy explanation — the policy
   passes or shoots in ~92% of its possession frames, but possession occurs in only
   0.81% of decisions — but the conditional number is very different from the
   reported 61.54% (8/13), and it is not produced by discarding the 49 selections.

---

## 21. Arithmetic validation

```text
8/7650  = 0.1045751634%
49/7650 = 0.6405228758%
57/7650 = 0.7450980392%
13/7650 = 0.1699346405%
```

* `0.745098...% == 57/7650` — mathematically correct.
* `0.7451% == 8/7650` — mathematically **false**; `8/7650 = 0.1046%`.
* `49/7650` and `8/7650` do sum to `57/7650`, so the canonical rate is consistent
  with the raw partition; only the labelling of the rate with the numerator `8` was
  wrong.


---

## 21a. Metric dictionary (the quantities that were conflated)

| Metric | Definition | Denominator | Seed-42 value | Source column / computation |
| --- | --- | ---: | ---: | --- |
| `n_decisions` | all recorded agent-decisions in the canonical scope | — | 7650 | summary `n_decisions`; raw `len(frames[seed==42])` |
| `n_onball` | decisions with `onball == True` (as recorded) | 7650 | 13 | summary `n_onball`; raw `count(onball)` |
| `n_offball` | decisions with `onball == False` (as recorded) | 7650 | 7637 | `7650 - 13` (derived; not stored) |
| `n_pass_shot_total` | PASS+SHOT actions over **all** decisions | 7650 | **57** | summary `n_pass_shot`; raw `count(action ∈ {9,10,11,12})` |
| `n_selected_ps_onball` | PASS+SHOT actions with `onball == True` | 13 | **8** | summary `team_n_selected_ps_onball_total`; reconciliation `team_wide_n_ps_selected` |
| `n_selected_ps_offball` | PASS+SHOT actions with `onball == False` | 7637 | **49** | derived: `57 - 8`; not a stored column |
| `canonical_rate_pct` | *Statistic A*: PASS+SHOT / 7650 × 100 | 7650 | **0.7450980392%** | summary `canonical_rate_pct`; reconciliation `team_wide_canonical_rate_pct` |
| `p_selected_ps_given_onball` | *Statistic B*: conditional on-ball rate | 13 | **0.6153846154** | summary `team_p_selected_ps_given_agent_onball`; reconciliation `onball_conditional_ps_rate` |
| `p_onball` | on-ball occupancy | 7650 | **0.0016993464** | summary `p_onball` |
| `rebuilt_pass_shot_rate_pct` | independent rebuilt-run rate | 7650 | **0.75** | reconciliation `canonical_rebuilt_csv_rate` (delta −0.004902 pp, `match=1`) |

Numerical relationships:

```text
57 = 8 + 49
0.745098% = 57/7650            (Statistic A  — the canonical metric)
0.6153846 = 8/13               (Statistic B  — conditional on-ball)
0.104575% = 8/7650             (Statistic B expressed over the full scope)
0.169935% = 13/7650            (occupancy)
```

The prior clarification replaced Statistic A with Statistic B (`8/7650`) while
keeping the label "canonical unconditional rate". That single substitution is the
root of the numerical contradictions in that report.

---

## 21b. Required raw-evidence table (Seed 42)

| Property | Seed 42 result | Evidence source | Status |
| --- | ---: | --- | --- |
| Total decisions | 7650 | raw detail | CONFIRMED |
| Total PASS+SHOT | 57 | raw detail | CONFIRMED |
| On-ball PASS+SHOT (recorded) | 8 | raw detail | CONFIRMED |
| Off-ball PASS+SHOT (recorded) | 49 | raw detail | CONFIRMED |
| Off-ball at tick 0 | 49 | raw detail | CONFIRMED |
| Off-ball at tick > 0 | 0 | raw detail | CONFIRMED |
| Off-ball agent 0 | 49 | raw detail | CONFIRMED |
| Off-ball agent 1 | 0 | raw detail | CONFIRMED |
| Off-ball agent 2 | 0 | raw detail | CONFIRMED |
| HIGH_PASS (action 10) count | 49 | raw detail | CONFIRMED |
| SHOT (action 12) count | 0 | raw detail | CONFIRMED |
| `mask_sum = 18` count | 49 | raw detail | CONFIRMED (18-of-19, `SLIDING` illegal) |
| all-ones mask count | 0 | raw detail | REFUTED (no all-ones mask exists) |
| temporal mismatches | 0 of 49 (0 of 30600 corpus-wide) | raw detail + verifier | CONFIRMED |
| canonical-scope violations | 0 | raw detail | CONFIRMED |
| engine-possession-mask frames | 62 | raw detail (mask semantics) | CONFIRMED (= 13 recorded on-ball + 49 tick-0) |
| PASS+SHOT under a mask forbidding ball actions | 0 (all seeds) | raw detail | CONFIRMED |

---

## 21c. Required numerical reconciliation table (raw vs recorded)

| Seed | Raw `n_onball` | Recorded `n_onball` | Match | Raw `n_pass_shot` | Recorded `n_pass_shot` | Match | Raw rate | Recorded rate |
| ---- | -------------: | ------------------: | ----- | ----------------: | ---------------------: | ----- | -------: | ------------: |
| 42   | 13 | 13 | yes | 57 | 57 | yes | 0.7450980392156863% | 0.7450980392156863% |
| 123  | 96 | 96 | yes | 128 | 128 | yes | 1.6732026143790852% | 1.6732026143790852% |
| 7    | 18 | 18 | yes | 66 | 66 | yes | 0.8627450980392156% | 0.8627450980392156% |
| 999  | 42 | 42 | yes | 79 | 79 | yes | 1.0326797385620916% | 1.0326797385620916% |

All values were re-derived from the raw detail by this audit; none was copied from
the previous report. `n_pass_shot` per seed decomposes as
`seed 7: 17 + 49`, `seed 42: 8 + 49`, `seed 123: 79 + 49`, `seed 999: 30 + 49`.

---

## 21d. Required Seed-42 decomposition table

| Quantity | Formula | Seed 42 |
| --- | --- | ------: |
| Total decisions | `N` | 7650 |
| On-ball decisions | `n_onball` | 13 |
| Off-ball decisions | `N - n_onball` | 7637 |
| On-ball PS selections | `n_onball_PS` | 8 |
| Off-ball PS selections | `n_offball_PS` | 49 |
| Total PS selections | `onball_PS + offball_PS` | 57 |
| On-ball contribution | `onball_PS / N` | 0.1045751634% |
| Off-ball contribution | `offball_PS / N` | 0.6405228758% |
| Total unconditional rate | `total_PS / N` | 0.7450980392% |
| Conditional PS rate (recorded on-ball) | `onball_PS / n_onball` | 61.5384615385% |
| Conditional PS rate (engine possession, §20) | `total_PS / n_possession_frames` | 91.9354838710% |
| Possession occupancy (engine, §20) | `n_possession_frames / N` | 0.8104575163% |

This table makes `8/13`, `8/7650`, `49/7650` and `57/7650` mutually unambiguous.


---

## 22. MAJOR ANALYTICAL ERRORS / INCONSISTENCIES

### Error A — arithmetic inconsistency

```text
ERROR:                  8/7650 presented as 0.7451%
Original claim:         "Seed 42 unconditional rate: 0.7451% (8/7650 x 100) — unchanged from canonical"
Why it was made:        the canonical 0.7451% (57/7650) was carried over and re-attached to the smaller
                        numerator 8 after the 49 frames had been declared artifacts
Evidence available:     canonical summary/reconciliation CSVs both store 0.7450980392156863
                        (= 57/7650); raw detail gives count(action in {9,10,11,12}) = 57
What code/data show:    8/7650 = 0.1045751634%; 57/7650 = 0.7450980392%
Classification:         ARITHMETIC ERROR
Impact:                 the report asserts a rate is "unchanged from canonical" while changing its
                        numerator by a factor of 7.1x
Canonical numbers:      unchanged (the CSV value 0.7451% is correct)
Required correction:    state 0.7451% = 57/7650; if the on-ball-restricted statistic is intended, write
                        "8/7650 = 0.1046% (on-ball-restricted)" and label it as such
```

### Error B — conflating the on-ball statistic with the total

```text
ERROR:                  8 on-ball selections treated as "8 total genuine selections"
Original claim:         "Seed 42 total genuine PASS+SHOT = 8"; "Total PASS+SHOT (genuine, excluding
                        tick=0 artifacts) = 8  Matches CSV canonical value"
Why it was made:        the reconciliation CSV's team_wide_n_ps_selected = 8 was read as the canonical
                        number, and the same CSV's team_wide_canonical_rate_pct (57/7650) was ignored
Evidence available:     summary n_pass_shot = 57; reconciliation team_wide_canonical_rate_pct =
                        0.7450980392156863; raw count = 57
What code/data show:    the CSVs publish both statistics; 57 != 8 unless an exclusion rule is invoked,
                        and no such rule exists in the contract
Classification:         statistic conflation (Statistic A vs Statistic B, §21a)
Impact:                 converts an occupancy decomposition into an artifact-removal claim
Canonical numbers:      unchanged
Required correction:    report 57 as the canonical numerator and 8 as the on-ball-restricted count
```

### Error C — treating "off-ball" as "invalid" without a validity rule

```text
ERROR:                  49 frames declared artifacts because onball == False
Original claim:         "These are initialization artifacts, not off-ball actions"; "should be excluded
                        from behavioral counts"
Why it was made:        onball == False was read as evidence of invalidity
Evidence available:     the canonical scope rules (episode/tick/agent/action/mask/checkpoint/
                        determinism/temporal alignment) contain no ownership-based exclusion
What code/data show:     all 49 satisfy every scope rule (§17); temporal alignment 49/49 (§18); zero
                        non-finite values; verifier passes on the full corpus
Classification:         unsupported validity rule
Impact:                 the retraction of the behavioural finding is not justified
Canonical numbers:      unchanged
Required correction:    classify the 49 as valid canonical decisions that are mislabelled by the
                        recorded ownership field (§15.1)
```

### Error D — tick-0 attribution

```text
ERROR:                  none (claim is factually correct)
Original claim:         "All 49 off-ball PS selections for seed 42 occur at tick=0"
Why it was made:        — (observation)
Evidence available:     raw detail, full tick distribution
What code/data show:    tick=0: 49, tick>0: 0 independently reproduced
Classification:         CONFIRMED observation, used to support an unsupported causal story
Impact:                 the correct observation was harnessed to the incorrect mask interpretation
Canonical numbers:      unchanged
Required correction:    keep the observation, detach it from the "synthetic mask" narrative
```

### Error E — agent attribution

```text
ERROR:                  none (claim is factually correct for seed 42)
Original claim:         "agent 0 (left_1) only"
Why it was made:        — (observation)
Evidence available:     raw detail, full agent distribution
What code/data show:    agent 0: 49, agent 1: 0, agent 2: 0
Classification:         CONFIRMED
Impact:                 none
Canonical numbers:      unchanged
Required correction:    none
```

### Error F — cross-seed generalisation of "all HIGH_PASS"

```text
ERROR:                  "all 196 off-ball PS across 4 seeds are HIGH_PASS"
Original claim:         "All 196 off-ball PASS+SHOT selections across all 4 seeds (49 per seed) share
                        identical characteristics: Action HIGH_PASS (action_id=10)"
Why it was made:        the Seed-42 set was assumed to generalise to the other seeds
Evidence available:     raw detail per seed
What code/data show:    seed 42 -> {10: 49}; seed 7 -> {10: 49}; seed 123 -> {11: 49} SHORT_PASS;
                        seed 999 -> {9: 49} LONG_PASS
Classification:         REFUTED generalisation (Seed-42 sub-claim CONFIRMED)
Impact:                 the "identical across all seeds" pattern was used to argue the frames were
                        mechanical/environmental; the cross-seed variation shows policy dependence
Canonical numbers:      unchanged
Required correction:    restrict the HIGH_PASS claim to seeds 42 and 7
```


### Error G — unsupported synthetic-mask attribution

```text
ERROR:                  "mask_sum=18 means a synthetic all-ones mask"
Original claim:         "PettingZoo wrapper synthesizes all-ones action masks because the true mask is
                        unknown before the first physics step"; table row "mask_sum 18 (all actions legal)"
Why it was made:        mask_sum=18 was read as "19 actions minus something immaterial", and a remembered
                        wrapper behaviour was assumed rather than inspected
Evidence available:     raw action_mask vector [1x16,0,1,1]; summary pi_floor_mean_mask_sum_given_selected
                        = 18.0; wrapper/bridge/engine source
What code/data show:    the vector is 18-of-19 with SLIDING illegal; it is byte-identical to the mask on
                        the 12 non-tick-0 on-ball frames of the same seed; the Python fallback is
                        fail-closed (sum 9, zero occurrences); bridge reset transmits engine-computed masks
Classification:         REFUTED
Impact:                 the entire "artifact" mechanism collapses
Canonical numbers:      unchanged
Required correction:    describe the mask as the engine possession mask delivered at reset
```

### Error H — metric / denominator confusion

```text
ERROR:                  interchangeable use of 8/13, 8/7650, 49/7650, 57/7650
Original claim:         "0.7451% (8/7650)" and "61.54% conditional rate" and "49 out of 57 (85.96%)"
                        presented as facets of one metric
Why it was made:        the three quantities share the seed-42 numerator space but have different
                        denominators and different meanings (§21a)
Evidence available:     the CSV columns and code lines cited in §21a
What code/data show:    8/13 = 61.5385% (conditional, denominator 13); 8/7650 = 0.1046%;
                        49/7650 = 0.6405%; 57/7650 = 0.7451% (canonical)
Classification:         metric conflation
Impact:                 the report's headline rate and its headline count refer to different metrics
Canonical numbers:      unchanged
Required correction:    use the metric dictionary in §21a for every numerical statement
```

### Error I — "fully explained" overreach

```text
ERROR:                  verdict upgraded from "partially explained" to "fully explained ... with no
                        off-ball contribution"
Original claim:         "Fully explained by agent-0 occupancy arithmetic, with no off-ball contribution"
Why it was made:        the 49 frames had been declared artifacts, which removed the off-ball term from
                        the decomposition by assumption rather than by measurement
Evidence available:     raw decomposition (§20): the recorded off-ball term is 49/7650 = 0.640523%,
                        85.96% of the 0.745098% total
What code/data show:    the term exists and dominates the recorded decomposition; only the ownership
                        label is defective, not the frame
Classification:         overreach / unsupported
Impact:                 the Seed-42 puzzle was closed on a premise the raw data contradicts
Canonical numbers:      unchanged
Required correction:    see §26 — the occupancy explanation survives, but with possession-based numbers
                        (57/62 = 91.94% conditional, 62/7650 = 0.81% occupancy) and without discarding
                        any recorded selection
```

### Error J — misattributing the discrepancy to the analysis script

```text
ERROR:                  "The analysis script counts ALL PASS+SHOT selections including tick=0 frames
                        with synthetic PettingZoo masks, rather than excluding them as the canonical
                        CSV does"
Original claim:         as quoted
Why it was made:        the script's output number (57) was used as if it were a code feature
Evidence available:     the script's full source (§4)
What code/data show:    the script has no tick logic, no mask logic and no exclusion logic; it reports
                        the stored fields verbatim. The canonical CSV does not exclude tick-0 frames
                        either (§19); 57 is the canonical numerator in both places
Classification:         misattribution + mischaracterisation of the canonical CSV
Impact:                 the "bug" was located in a script that cannot produce it, while the real defect
                        (the wrapper's reset-time ownership resolution, §15.1) went unnoticed
Canonical numbers:      unchanged
Required correction:    see §15.1
```

### Error K — the missed defect (not an error in the prior report, but the real mechanism)

```text
ERROR:                  tick-0 pre_step_ball_owner_agent_idx = 255 for 49 of 50 episodes
Location:               training/gmn_pettingzoo.py:1413-1422 (owner resolution uses self.agents before
                        line 1431-1432 refreshes it) together with 2106-2107 (self.agents emptied on
                        every terminal step) and 542-546 (constructor warm-up reset)
Status:                 recorded-field defect reproduced by code inspection and by the 1-of-50 pattern
Impact on metrics:      none — the canonical metric counts actions, not ownership labels
Impact on interpretation: HIGH — it is the sole reason a 49-frame "off-ball" set exists
Note:                   this audit does not patch the wrapper (measurement-only mandate, and the
                        canonical artifacts were produced by the current code). It documents the defect
                        so that any future ownership-based statistic is interpreted with it in mind
```


---

## 23. Layer 1 — Proven facts (raw code/data only)

F1. The canonical detail artifact contains 30,600 frames: 4 seeds x 7650 decisions
    (50 episodes x 51 ticks x 3 agents per seed).
F2. Seed 42: 57 PASS+SHOT selections; 13 decisions with `onball == True`; 49 with
    `onball == False`; `57 = 8 + 49`; canonical rate `57/7650 = 0.7450980392%`.
F3. All 49 recorded off-ball PASS+SHOT frames have `tick = 0`, `agent_index = 0`,
    `agent_id = left_1`, `action_taken = 10`, `action_taken_name = HIGH_PASS`,
    `mask_sum = 18`, `pre_step_ball_owner_agent_idx = 255`, `pre_step_obs95 = 1.0`,
    `team_has_ball = True`, `agent_has_ball = False`, `onball = False`,
    `event_code = 5`, `done = terminated = truncated = False`,
    `deterministic_action_is_pass_shot = True`, exactly one frame per episode 1..49.
F4. Their `action_mask` is exactly `[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1]`
    (18 of 19 legal; index 16 `SLIDING` illegal).
F5. Seed 42 corpus mask sums: `{15: 7588, 18: 62}`; the 18-vector also occurs on 12
    non-tick-0 frames, which are exactly the 12 non-tick-0 on-ball frames. The
    fail-closed sum-9 mask occurs 0 times.
F6. Tick-0 population per seed: 150 frames; agent 0 always carries the 18-vector and
    agents 1-2 always carry the 15-vector; `pre_step_obs95 == 1.0` in all 150; exactly
    1 frame is recorded on-ball (episode 0, agent 0); agent 0 selects action 10 in all
    50 Seed-42 episodes, action 10 for seed 7, action 11 for seed 123, action 9 for
    seed 999.
F7. Zero PASS+SHOT selections anywhere in the corpus (all seeds, all 30,600 frames)
    were made under a mask with ball actions illegal.
F8. For every seed, `count(mask grants {9,10,11,12}) == recorded n_onball + 49`
    (seed 7: 67 = 18+49; seed 42: 62 = 13+49; seed 123: 145 = 96+49; seed 999: 91 = 42+49).
F9. All 49 frames pass every canonical scope rule (0 violations) and all 49 are
    temporally aligned (`action_taken == masked pre-step argmax`).
F10. The canonical collector records tick 0 (`tick_idx = 0`, appended before
     `env.step`) and `n_decisions = len(all_decisions)` includes it; the canonical
     denominator is `50 x 51 x 3 = 7650`.
F11. `analyze_seed42_disconnect.py` contains no tick, agent, mask, reset or validity
     predicate; it partitions PASS+SHOT by the stored `onball` field.
F12. The reconciliation CSV row for seed 42 contains both
     `team_wide_n_ps_selected = 8` and
     `team_wide_canonical_rate_pct = 0.7450980392156863`.
F13. `8/7650 = 0.1045751634%`, `49/7650 = 0.6405228758%`, `57/7650 = 0.7450980392%`.
F14. `training/verify_canonical_artifacts.py` reports
     `ALL VERIFICATION CHECKS PASSED` with exit code 0 (raw output in §29).
F15. Source-code facts: `ObservationEncoder.getActionMask` grants ball actions iff
     `player.hasBall || engine.ball.ownerId === player.id`
     (`src/engine/ObservationEncoder.ts:394-424`); the team-possession one-hot is
     computed solely from `ball.ownerId` (`:57-71, 136-141`); the canonical scenario
     sets `setup.ball.ownerId = 'left_1'` (`src/scenarios/ScenarioRegistry.ts:163`),
     applied by `GameEngine.loadScenario` (`src/engine/GameEngine.ts:493-501`); the
     bridge transmits engine masks at reset (`training/bridge_server.ts:311, 319,
     356-366, 467-530, 1386-1388`); the wrapper stores them verbatim
     (`training/gmn_pettingzoo.py:1445-1455`); the wrapper empties `self.agents` on
     terminal steps (`:2106-2107`) and resolves the reset owner before refreshing
     `self.agents` (`:1413-1422` vs `:1431-1432`); the constructor performs a warm-up
     reset (`:542-546`).

---

## 24. Layer 2 — Supported inference (follows from proven facts)

I1. The 49 frames are possession frames mislabelled as off-ball. Chain: engine
    possession mask (F4/F5) + `pre_step_obs95 == 1.0` on all 150 tick-0 frames
    (F6) + scenario `ownerId = 'left_1'` applied at reset (F15) + the wrapper
    reset-path defect explaining the 255 label (F15, S15.1). No validity rule is
    invoked; the defect is in the recorded ownership field, not the decision.
I2. The possession-corrected Seed-42 picture is: 62 engine-possession frames,
    57 PASS+SHOT selections on those frames, i.e. 57/62 = 91.94%
    possession-conditional rate and 62/7650 = 0.81% possession occupancy. This
    is an inference (re-labelling by mask/observation), not a new canonical
    metric.
I3. The prior "synthetic mask / artifact / invalid" mechanism is not viable: it
    contradicts F4, F5, F7 and the fail-closed fallback fact in F15, and no code
    path supporting it was found.
I4. `analyze_seed42_disconnect.py` did not create the 49; it faithfully reported
    the stored `onball` partition (F11). The error was downstream: treating that
    partition as a validity partition.

---

## 25. Layer 3 — Unresolved questions (not determinable from artifact)

U1. Why the Seed-42 policy selects HIGH_PASS at every kickoff (50/50 episodes)
    while other seeds select other kickoff actions — a policy question needing
    checkpoint/activation analysis, outside the measurement artifact.
U2. Why recorded on-ball occupancy at tick > 0 is so low for Seed 42 (12 frames)
    versus Seed 123 (95 frames) — a dynamics/rollout question outside the
    measurement artifact's scope.
U3. Whether future ownership-based statistics should use the recorded
    `pre_step_ball_owner_agent_idx` at tick 0 for episodes 1..49 without
---

## 26. Revised Seed-42 conclusion

Seed 42 canonical measurement stands: 57 PASS+SHOT / 7650 = 0.7450980392%.
Decomposition by recorded `onball`: 8 recorded on-ball + 49 recorded off-ball,
where the 49 are valid canonical tick-0 kickoff decisions by agent 0 (HIGH_PASS
under the engine possession mask) whose recorded ownership label (255, hence
`onball = False`) is wrong for episodes 1..49 because of the wrapper reset-path
defect (S15.1).

Consequences:

* Canonical 57/7650 is unchanged and remains the only canonical unconditional
  rate. 8/7650 = 0.1046% is an on-ball-restricted statistic, not canonical.
* Occupancy explanation survives in corrected form but does NOT license "fully
  explained with no off-ball contribution": on recorded fields the off-ball
  term (49/7650 = 0.6405%) dominates; possession-corrected reading is low
  possession occupancy (62/7650 = 0.81%) with high conditional rate
  (57/62 = 91.94%).
* Disconnect classification: PARTIALLY EXPLAINED — raw partition (57 = 8 + 49)
  and ownership-label defect proven; behavioural question (kickoff policy, low
  tick>0 occupancy) remains open per S25.

---

## 27. Confirmation that canonical numbers were not changed

No training, no re-collection, no regeneration, no patching of any canonical
artifact, network, reward, mask, environment or scenario file.

```text
post_reweight_logit_prestep_reconciled_detail.json : 326,583,222 bytes, 30,600 frames
```

Reconciliation (raw-derived, see S21c):

```text
seed 42 : n_onball 13 == 13, n_pass_shot 57 == 57, rate 0.7450980392156863% == recorded
seed 123: n_onball 96 == 96, n_pass_shot 128 == 128, rate 1.6732026143790852% == recorded
seed 7  : n_onball 18 == 18, n_pass_shot 66 == 66,  rate 0.8627450980392156% == recorded
seed 999: n_onball 42 == 42, n_pass_shot 79 == 79,  rate 1.0326797385620916% == recorded
```

`training/verify_canonical_artifacts.py` reports ALL VERIFICATION CHECKS PASSED
(exit 0) on untouched artifacts (raw output in S29).

---

## 28. Git provenance
---

## 29. Verification output (verbatim, authoritative verifier)

Command: `python training/verify_canonical_artifacts.py` — exit code 0.
Full raw output (177 lines) preserved at
`training/results/forensic_verify_canonical_artifacts_output.txt`; tail:

```text
PASS: reconciliation CSV matches raw-detail reconstruction
PASS: seed 42: canonical rate matches raw counts
PASS: seed 123: canonical rate matches raw counts
PASS: seed 7: canonical rate matches raw counts
PASS: seed 999: canonical rate matches raw counts
PASS: seed 42: canonical rate within 0.05pp of rebuilt CSV
PASS: seed 123: canonical rate within 0.05pp of rebuilt CSV
PASS: seed 7: canonical rate within 0.05pp of rebuilt CSV
PASS: seed 999: canonical rate within 0.05pp of rebuilt CSV
PASS: historical filename guard
PASS: canonical scope guard
PASS: summary/reconciliation checksums match committed values
PASS: verification coverage guard

ALL VERIFICATION CHECKS PASSED
```

What "verifier passed" means: every structural, count, field, alignment,
semantic-invariant, summary-reconstruction, reconciliation-reconstruction, rate
and provenance-guard check holds on untouched artifacts. It does not prove the
49 "invalid" — the verifier checks stored `onball` semantics, not engine-truth
of the tick-0 ownership label; the mislabelling diagnosis (S15.1) is a
code-level finding on top of passing verification.

Forensic derivation (read-only; canonical artifacts never written):

```bash
python training/forensic_seed42_offball_inventory.py --out "$env:TEMP\seed42_offball_frames.json"
python training/verify_canonical_artifacts.py
```

---

## 30. Cleanup / git status (verbatim)

```text
git status --short (after audit, before any commit):
?? _tmp_gen_section5.py
?? training/analyze_offball_mechanism.py
?? training/forensic_seed42_offball_inventory.py
?? training/results/SEED42_OFFBALL_CLAIM_CLARIFICATION.md
?? training/results/SEED42_OFFBALL_FORENSIC_AUDIT.md
?? training/results/forensic_seed42_offball_inventory_output.txt
?? training/results/forensic_verify_canonical_artifacts_output.txt

git diff --check:
(no output — no whitespace errors)
```

Deliberate audit artifacts: this report (required deliverable);
`training/forensic_seed42_offball_inventory.py` (read-only streaming script);
`training/results/forensic_seed42_offball_inventory_output.txt` (359 lines);
`training/results/forensic_verify_canonical_artifacts_output.txt` (177 lines).
Untouched pre-existing untracked files: SEED42_OFFBALL_CLAIM_CLARIFICATION.md,
`training/analyze_offball_mechanism.py`. Scratch `_tmp_gen_section5.py` removed
before commit. No tracked file modified; no checkpoint/JSON/training output
added; 326 MB LFS detail artifact not duplicated.

---

## FINAL DISPOSITION

```text
FINAL DISPOSITION
=================

Prior "49 off-ball = tick-0 synthetic artifact" claim:
    REFUTED

Prior "all 49 are agent 0" claim:
    CONFIRMED (raw observation; causal "artifact" reading REFUTED)

Prior "all 49 are HIGH_PASS" claim:
    CONFIRMED (raw observation; causal "artifact" reading REFUTED)

Prior "mask_sum=18 means synthetic mask" claim:
    REFUTED

Prior "8/7650 = 0.7451%" claim:
    ARITHMETICALLY INCORRECT

Prior "fully explained by low on-ball occupancy" claim:
    PARTIALLY SUPPORTED (recorded decomposition coherent; "fully explained with
    no off-ball contribution" REFUTED — recorded off-ball term 49/7650 dominates;
    corrected story is low possession occupancy with high conditional rate, S26)

Canonical dataset changed:
    NO

Canonical n_onball changed:
    NO

Canonical n_pass_shot changed:
    NO

Canonical unconditional rate changed:
    NO

Training performed:
    NO

Canonical measurement re-run:
    NO
```

Forensic script stdout (359 lines: per-seed reconstruction, 49-frame
inventory, frequency tables, alignment/scope checks) preserved at
`training/results/forensic_seed42_offball_inventory_output.txt`; head:

```text
artifact: training\results\post_reweight_logit_prestep_reconciled_detail.json
bytes   : 326583222
chars   : 315197111
 seed  frames  onball pass_shot  onb_PS  off_PS tick0_fr tick0_PS  t0_onb  t0_onbPS
    7    7650      18        66      17      49      150       50       1         1
   42    7650      13        57       8      49      150       50       1         1
  123    7650      96       128      79      49      150       50       1         1
  999    7650      42        79      30      49      150       50       1         1
```


```text
HEAD at audit start and end: fdfe7355c1e046b63f8c4a71f8401a30d49963f4
Branch: main (origin/main, origin/HEAD)
```

Recent history head (`git log --oneline -20`):

```text
fdfe735 (HEAD -> main, origin/main, origin/HEAD) docs/verification: canonical pi snapshot and seed-42 disconnect re-examination
b46d3f0 docs: add SHA_AND_EVIDENCE_CORRECTION.md with full provenance evidence
69806f4 docs: add CLOSE_OUT_REPORT.md from canonical close-out task
8b8e177 docs/verification: harden canonical verifier with semantic invariants and provenance fix
63b4b7a docs/verification: definition-correction impact audit confirms Branch A (doc-only fix)
```

No tracked file modified. New untracked audit artifacts listed in S30.
`git diff --check` reports no whitespace errors (verbatim in S30).

    correction — open until the wrapper reset-path defect (Error K) is fixed in
    code; this audit deliberately does not patch it (measurement-only mandate).
U4. The game-semantic meaning of `event_code = 5` beyond its observed constancy
    on the 49 frames — the artifact records the code but contains no codebook.

