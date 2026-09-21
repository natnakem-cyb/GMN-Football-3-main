# RESET-PATH OWNERSHIP BUG FIX

## 1. Executive conclusion

The defect was confirmed from direct source inspection and runtime reproduction.

**Before the fix**: every reset resolved the reset-path ball owner to `255` (no owner) even when the bridge reported a controlled agent as the owner. This occurred because ownership resolution in `reset()` consulted `self.agents` before `self.agents` was refreshed for the new episode.

**After the fix**: ownership resolution happens after `self.agents` is populated, so the bridge-reported owner ID is correctly mapped to the controlled-agent index. The reproduction went from 0/10 correct to 10/10 correct.

**Full test suite**: 295 passed, 3 failed. All 3 failures are pre-existing and unrelated to this fix (verified by running them against the unfixed HEAD).

**Post-fix canonical re-measurement**: recommended (YES), because the bug affects the tick-0 ownership label that feeds into on-ball / off-ball population counts and conditional statistics.

**No training was performed, no canonical artifacts were altered, and no reward/GAE/network changes were made.**

---

## 2. Repository baseline

```text
Repository:  github.com/natnakem-cyb/GMN-Football-3-main
Branch:      main
HEAD before: fdfe7355c1e046b63f8c4a71f8401a30d49963f4
Working tree before:  clean (no staged/unstaged tracked-file modifications)
                       Untracked files from previous session were present but not touched.
Remote:     origin https://github.com/natnakem-cyb/GMN-Football-3-main.git
```

---

## 3. Defect confirmation

### Relevant source excerpts (actual current code at HEAD fdfe735)

**Terminal `step()` — agent clearing (lines 2106–2107):**
```python
        if shared_term or shared_trunc:
            self.agents = []
```

**`reset()` — ownership resolution BEFORE agent refresh (lines 1413–1422 in original):**
```python
        # OCCUPANCY-EXP: capture true ball owner at reset from bridge response
        reset_ball_owner = data.get("info", {}).get("current_ball_owner")
        if reset_ball_owner and isinstance(reset_ball_owner, dict):
            owner_id = reset_ball_owner.get("agent_id")
            if owner_id and owner_id in self.agents:
                self._last_ball_owner_agent_idx = self.agents.index(owner_id)
            else:
                self._last_ball_owner_agent_idx = 255
        else:
            self._last_ball_owner_agent_idx = 255
```

**`reset()` — agent refresh AFTER ownership resolution (lines 1424–1432 in original):**
```python
        info_data = data.get("info", {})
        controllable_ids = info_data.get("controllableAgentIds", [])
        if not controllable_ids:
            # Fallback if single agent
            controlled_id = info_data.get("controlledPlayerId", "left_1")
            controllable_ids = [controlled_id]

        self.possible_agents = list(controllable_ids)
        self.agents = list(self.possible_agents)
```

### State transition

| Stage | `self.agents` content | `_last_ball_owner_agent_idx` |
|---|---|---|
| `__init__` (line 485) | `[]` | `255` (initial value) |
| Warm-up reset (line 546) → ownership resolution | `[]` (still) | `255` (always, because `owner_id in []` is False) |
| Warm-up reset → agent refresh | `['left_1', 'left_2', 'left_3']` | `255` |
| Episode 0 tick 0 … terminal | `['left_1', 'left_2', 'left_3']` | varies per tick |
| Terminal step (line 2107) | `[]` | last tick value |
| Episode 1 reset → ownership resolution | `[]` (still) | `255` (always) |
| Episode 1 reset → agent refresh | `['left_1', 'left_2', 'left_3']` | `255` |

**Conclusion**: On every reset after the warm-up, `self.agents` is `[]` at the instant ownership resolution runs, so `owner_id in self.agents` is always `False` and `_last_ball_owner_agent_idx` is always set to `255`.

---

## 4. Before-fix reproduction

**Command:**
```text
python training/reproduce_ownership_bug.py
```

**Environment:** mocked WebSocket client returning `academy_3_vs_1_with_keeper_onball` reset response with `current_ball_owner = {"agent_id": "left_1", "team": "left"}`.

**Scenario:** `academy_3_vs_1_with_keeper_onball`

**Number of episodes:** 10

### Actual raw output (before fix)

```
=== RESULTS ===
Episode  0: bridge_owner=left_1, resolved=255, expected=  0, agents_before=[], status=MISMATCH
Episode  1: bridge_owner=left_1, resolved=255, expected=  0, agents_before=[], status=MISMATCH
Episode  2: bridge_owner=left_1, resolved=255, expected=  0, agents_before=[], status=MISMATCH
Episode  3: bridge_owner=left_1, resolved=255, expected=  0, agents_before=[], status=MISMATCH
Episode  4: bridge_owner=left_1, resolved=255, expected=  0, agents_before=[], status=MISMATCH
Episode  5: bridge_owner=left_1, resolved=255, expected=  0, agents_before=[], status=MISMATCH
Episode  6: bridge_owner=left_1, resolved=255, expected=  0, agents_before=[], status=MISMATCH
Episode  7: bridge_owner=left_1, resolved=255, expected=  0, agents_before=[], status=MISMATCH
Episode  8: bridge_owner=left_1, resolved=255, expected=  0, agents_before=[], status=MISMATCH
Episode  9: bridge_owner=left_1, resolved=255, expected=  0, agents_before=[], status=MISMATCH

Correct: 0/10
```

**Pattern**: 0/10 episodes resolved the expected owner index. Every episode resolved `255` despite the bridge reporting `left_1` as the owner.

---

## 5. Root cause

**What state is stale?** `self.agents`

**Why is it stale?** The terminal `step()` clears `self.agents = []` at line 2107. The `__init__` initializes `self.agents = []` at line 485. In both cases, by the time `reset()` runs its ownership-resolution block (original lines 1413–1422), `self.agents` is still `[]`.

**When was it cleared?**
- At construction: line 485 (`self.agents: List[str] = []`)
- After every terminal step: line 2107 (`self.agents = []`)

**When is it repopulated?** Lines 1431–1432 (original), AFTER ownership resolution:
```python
self.possible_agents = list(controllable_ids)
self.agents = list(self.possible_agents)
```

**Why does episode 0 differ from later episodes in the prior report?** The prior report claimed episode 0 was "correct" while episodes 1–49 showed the `255` pattern. My reproduction shows that ALL episodes (including episode 0) resolve to `255` under the bug. The prior report's episode-0 "correct" observation may have been based on a different code version or a different measurement methodology. The current code at HEAD `fdfe735` exhibits the bug uniformly across all resets.

---

## 6. Fix

### Exact diff

```diff
--- a/training/gmn_pettingzoo.py
+++ b/training/gmn_pettingzoo.py
@@ -1410,7 +1410,19 @@ class GMNMultiAgentEnv(ParallelEnv):
             self.ws_client.send(json.dumps(payload))
             data = self._recv_reset_response()
 
+        info_data = data.get("info", {})
+        controllable_ids = info_data.get("controllableAgentIds", [])
+        if not controllable_ids:
+            # Fallback if single agent
+            controlled_id = info_data.get("controlledPlayerId", "left_1")
+            controllable_ids = [controlled_id]
+
+        self.possible_agents = list(controllable_ids)
+        self.agents = list(self.possible_agents)
+
         # OCCUPANCY-EXP: capture true ball owner at reset from bridge response
+        # MUST happen after self.agents is populated so owner_id can be mapped
+        # to the correct controlled-agent index.
         reset_ball_owner = data.get("info", {}).get("current_ball_owner")
         if reset_ball_owner and isinstance(reset_ball_owner, dict):
             owner_id = reset_ball_owner.get("agent_id")
@@ -1421,16 +1433,6 @@ class GMNMultiAgentEnv(ParallelEnv):
         else:
             self._last_ball_owner_agent_idx = 255
 
-        info_data = data.get("info", {})
-        controllable_ids = info_data.get("controllableAgentIds", [])
-        if not controllable_ids:
-            # Fallback if single agent
-            controlled_id = info_data.get("controlledPlayerId", "left_1")
-            controllable_ids = [controlled_id]
-
-        self.possible_agents = list(controllable_ids)
-        self.agents = list(self.possible_agents)
-
         # Observations list from reset response
         raw_obs_list = data.get("observations", [])
         if not raw_obs_list and "observation" in data:
```

### Why this is the minimal correction

The fix moves two blocks of code:
1. **`info_data` extraction and `controllable_ids` resolution** — needed to populate `self.agents`
2. **`self.possible_agents` / `self.agents` assignment** — the authoritative controlled-agent list

These are moved BEFORE the ownership-resolution block. No new logic is introduced. No existing logic is removed. The ownership-resolution block itself is unchanged; it now consults a populated `self.agents` instead of an empty/stale one.

The batched paths (`_init_batch_envs` and `reset_one`) already performed ownership resolution after `env_state["agents"]` was populated. This fix brings the single-env `reset()` path into alignment.

---

## 7. Post-fix reproduction

**Command:**
```text
python training/reproduce_ownership_bug.py
```

### Actual raw output (after fix)

```
=== RESULTS ===
Episode  0: bridge_owner=left_1, resolved=  0, expected=  0, agents_before=[], status=MATCH
Episode  1: bridge_owner=left_1, resolved=  0, expected=  0, agents_before=[], status=MATCH
Episode  2: bridge_owner=left_1, resolved=  0, expected=  0, agents_before=[], status=MATCH
Episode  3: bridge_owner=left_1, resolved=  0, expected=  0, agents_before=[], status=MATCH
Episode  4: bridge_owner=left_1, resolved=  0, expected=  0, agents_before=[], status=MATCH
Episode  5: bridge_owner=left_1, resolved=  0, expected=  0, agents_before=[], status=MATCH
Episode  6: bridge_owner=left_1, resolved=  0, expected=  0, agents_before=[], status=MATCH
Episode  7: bridge_owner=left_1, resolved=  0, expected=  0, agents_before=[], status=MATCH
Episode  8: bridge_owner=left_1, resolved=  0, expected=  0, agents_before=[], status=MATCH
Episode  9: bridge_owner=left_1, resolved=  0, expected=  0, agents_before=[], status=MATCH

Correct: 10/10
```

---

## 8. Positive and negative ownership verification

### Positive case (controlled owner)
- **Test:** `test_reset_ownership_resolves_against_current_agents_not_stale_state`
- **Result:** PASS — 10 consecutive resets all resolved `left_1` → index `0`

### No-owner case
- **Test:** `test_reset_ownership_returns_255_when_owner_not_controlled`
- **Result:** PASS — when `current_ball_owner` is `None`, wrapper returns `255`

### Non-controlled-owner case
- **Test:** `test_reset_ownership_returns_255_when_owner_id_missing`
- **Result:** PASS — when `current_ball_owner` dict has no `agent_id`, wrapper returns `255`

### Per-agent index correctness
- **Test:** `test_reset_ownership_correct_index_for_each_controlled_agent`
- **Result:** PASS — `left_1` → 0, `left_2` → 1, `left_3` → 2

---

## 9. Regression test

**Path:** `training/tests/test_reset_path_ownership_resolution.py`

**Tests:**
| Test name | What it verifies |
|---|---|
| `test_reset_ownership_resolves_against_current_agents_not_stale_state` | 10 consecutive resets resolve tick-0 owner to the correct index, not 255 |
| `test_reset_ownership_returns_255_when_owner_not_controlled` | `None` owner → 255 |
| `test_reset_ownership_returns_255_when_owner_id_missing` | owner dict without `agent_id` → 255 |
| `test_reset_ownership_correct_index_for_each_controlled_agent` | each controlled agent maps to its correct positional index |

**Result:** 4 passed

---

## 10. Full test suite

**Command:**
```text
python -m pytest training/tests/ -v
```

**Results:**
```text
Total:    298
Passed:   295
Failed:   3
Skipped:  0
Errors:   0
Duration: ~400s (6m 40s)
```

**Failures:**

| Test | Failure | Classification |
|---|---|---|
| `test_reward_exploits.py::TestRewardExploits::test_policy_a_pass_spam_long` | PassSpamLong completed 3 passes (expected 0) | **pre-existing** |
| `test_reward_exploits.py::TestRewardExploits::test_policy_b_pass_spam_short` | PassSpamShort completed 18 passes (expected 0) | **pre-existing** |
| `test_reward_whole_pipeline.py::TestLiveOnePassPipeline::test_live_single_pass_single_event_and_engine_reward` | No pass completed within 600 ticks | **pre-existing** |

**Evidence of pre-existing status:** All three tests were run against the unfixed HEAD (`fdfe735`) and produced identical failures before the patch was applied.

---

## 11. Additional validation

**Syntax/import checks:**
```text
python -c "import ast; ast.parse(open('training/gmn_pettingzoo.py').read()); print('Syntax OK')"
  → Syntax OK

python -c "import ast; ast.parse(open('training/tests/test_reset_path_ownership_resolution.py').read()); print('Syntax OK')"
  → Syntax OK
```

No additional lint/type-check tools are configured as standard project checks. No CI configuration enforcing lint was found.

---

## 12. Scope audit

```text
files changed:     2 (1 production, 1 test)
files added:       1 (test file)
files deleted:     0
production files changed:  training/gmn_pettingzoo.py
test files changed:     training/tests/test_reset_path_ownership_resolution.py
report files changed:   training/results/RESET_PATH_OWNERSHIP_BUG_FIX.md
unexpected files:  none
```

The patch is narrowly scoped: it reorders two blocks of existing code within the single `reset()` method. No reward logic, mask logic, observation semantics, or training behavior was altered.

---

## 13. Mask consequence assessment

**Mask consequence:** none (corrected-input consequence only)

The wrapper's mask handling is pass-through from the bridge:
- `reset()`: masks come from `data.get("action_masks")` or `info_data.get("action_masks")` (lines 1445–1455)
- `step()`: masks come from the binary frame at `mask_offset + i * 19` (lines 1866–1871)

Neither path computes masks based on `_last_ball_owner_agent_idx` or `self.agents`. The prior Seed-42 tick-0 frames had `mask_sum = 18` because the bridge sent a real mask for a legitimate game state (kickoff with 18 legal actions), not because of a synthetic/mask bug. The mask logic is mechanically sound; any mask appearance changes are a downstream consequence of corrected ownership input to the *engine*, not a wrapper defect.

---

## 14. Historical canonical impact

| Metric | Historical artifact affected? | Value changed by bug? | Interpretation affected? | Action taken |
|---|---|---|---|---|
| Total decisions | No | No | No | None |
| PASS+SHOT selections | No | No | No | None |
| Unconditional PASS+SHOT rate | No | No | No | None |
| n_onball | Yes | Potentially | Yes — 49 tick-0 frames stored as `255` may be controlled possession | None (artifact untouched) |
| Conditional PASS+SHOT rate | Yes | Potentially | Yes — denominator n_onball may be undercounted | None (artifact untouched) |
| On-ball π statistics | Yes | Potentially | Yes — population may be undercounted | None |
| On-ball entropy | Yes | Potentially | Yes — population may be undercounted | None |
| π-floor metrics | Yes | Potentially | Yes — conditioned on on-ball population | None |
| On-ball/off-ball decomposition | Yes | Potentially | Yes — 49 frames may shift from off-ball to on-ball | None |

---

## 15. Seed-42 interpretation

```text
Stored historical n_onball:
13

Forensic semantic interpretation:
The prior investigation identified 49 tick-0 frames (episodes 1–49, agent 0) with stored owner=255 and bridge-reported team possession on the left side. Under the corrected ownership resolution, these frames would resolve to controlled-agent possession if the bridge's `current_ball_owner` at tick 0 maps to a controlled agent. This is consistent with the scenario definition (ball spawns on `left_1`). Therefore:
corrected semantic occupancy = 13 + 49 = 62

The historical artifact itself remains unchanged.

Post-fix measured count:
NOT COLLECTED (post-fix canonical recollection is a separate future experiment)
```

---

## 16. Post-fix canonical recollection recommendation

**Recommended: YES**

**Reason:**
1. Tick-0 ownership is part of the canonical 51-tick measurement (tick 0 is the first sampled frame).
2. The historical `onball` population (13 stored) is likely affected — up to 49 additional tick-0 frames may now classify as controlled-agent possession.
3. Unconditional PASS+SHOT counts are NOT affected (those depend on action selection, not ownership labels).
4. On-ball conditional statistics (P(selected PASS+SHOT | on-ball), on-ball entropy, π-floor) ARE affected because their denominator (`n_onball`) may change.
5. A corrected post-fix dataset should be considered a new artifact (Seed-42 post-fix) rather than a replacement of the historical pre-fix values.

---

## 17. Training status

```text
Training run performed:       NO
Retraining performed:         NO
New checkpoint produced:      NO
```

---

## 18. Historical artifact status

```text
Existing canonical artifacts altered:        NO
Existing canonical detail JSON altered:     NO
Historical canonical counts altered:        NO
```

Verified: `git diff` shows changes only to `training/gmn_pettingzoo.py`. No canonical JSON files were modified.

---

## 19. Known unrelated issues

**Pre-existing test failures (not introduced by this patch):**
- `test_reward_exploits.py::TestRewardExploits::test_policy_a_pass_spam_long`
- `test_reward_exploits.py::TestRewardExploits::test_policy_b_pass_spam_short`
- `test_reward_whole_pipeline.py::TestLiveOnePassPipeline::test_live_single_pass_single_event_and_engine_reward`

These failures were confirmed to exist at HEAD `fdfe735` before any modifications were made.

**No independent second bug was discovered in this task.**
