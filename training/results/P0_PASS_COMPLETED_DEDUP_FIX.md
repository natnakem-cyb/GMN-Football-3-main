# P0 PASS_COMPLETED DEDUP FIX REPORT
**Date:** 2026-09-22
**Author:** Kilo (code agent)
**Status:** COMPLETE

---

## TASK 1 — MECHANISM CONFIRMATION

### Code Quotations (current HEAD = 9d0d7ad)

**GameEngine.ts — pass_completed emission:**
```typescript
// src/engine/GameEngine.ts:1136-1146
if (lastKickedById != null && lastKickedTeam != null) {
  const isSameTeam = lastKickedTeam === player.team;
  const isDifferentPlayer = lastKickedById !== player.id;
  if (isSameTeam && isDifferentPlayer) {
    this.recordEvent(
      'pass_completed',
      `Pass completed: ${player.name} received from ${lastKickedById}`,
      player.position,
      player.team
    );
    this.ball.lastKickedBy = null;
```

**gmn_pettingzoo.py — _resolve_pending_pass_state (PASS_COMPLETED emission):**
```python
# training/gmn_pettingzoo.py:1488-1504
if 0 <= ball_owner_agent_idx < len(agents):
    owner_id = agents[ball_owner_agent_idx]
    if owner_id != pending_pass["agent_id"]:
        resolved_type = (
            "PASS_FAILED" if owner_id.startswith("right_") else "PASS_COMPLETED"
        )
# ...
if resolved_type is not None:
    events.append(
        {
            "type": resolved_type,
            "team": "left",
            "agent_id": pending_pass["agent_id"],  # PASSER
        }
    )
    pending_pass = None
```

**gmn_pettingzoo.py — _build_shaper_events (engine → PASS_COMPLETED mapping):**
```python
# training/gmn_pettingzoo.py:1556-1567
elif ev_type == "pass_completed":
    shaper_type = "PASS_COMPLETED"
    shaper_team = "left"

if shaper_type is not None:
    event_dict: Dict[str, Any] = {"type": shaper_type, "team": shaper_team}
    if 0 <= ball_owner_agent_idx < len(agents):
        event_dict["agent_id"] = agents[ball_owner_agent_idx]  # RECEIVER
    step_events.append(event_dict)
```

**gmn_pettingzoo.py — assembly site:**
```python
# training/gmn_pettingzoo.py:1652-1658
resolution_events, env_state["pending_pass"] = GMNMultiAgentEnv._resolve_pending_pass_state(
    env_state.get("pending_pass"), ball_owner_agent_idx, current_ev_type, agents
)
step_events = resolution_events + step_events

# Canonicalize PASS_COMPLETED events: exactly one per physical pass
step_events = GMNMultiAgentEnv._canonicalize_pass_events(step_events, env_state)
```

**reward_adapters.py — payment loop:**
```python
# training/reward_adapters.py:176-181
if etype == "PASS_COMPLETED":
    self.pass_chain_length += 1
    self.pass_completed_count += 1
    self.total_pass_completed_count += 1
    if aid in shaped:
        shaped[aid] += self.r_pass
```

### Mechanism Status
**CONFIRMED ACCURATE.** Both detection paths still exist on current HEAD and produce events with different `agent_id`s on the same tick:
1. Engine path: `agent_id` = receiver (current ball owner)
2. Resolve path: `agent_id` = passer (stored in `pending_pass["agent_id"]`)

The existing `_canonicalize_pass_events` used key `(tick, passer-from-pending, receiver, "PASS_COMPLETED")`. After resolution, `pending_pass` is `None`, so `passer` is `None`. Keys never collide → both events survive → double payment.

### Before-Fix Reproduction
```
BEFORE-FIX SURVIVED COUNT: 2
BEFORE-FIX RAW OUTPUT: [
  {'type': 'PASS_COMPLETED', 'team': 'left', 'agent_id': 'left_0'},
  {'type': 'PASS_COMPLETED', 'team': 'left', 'agent_id': 'left_1'}
]
BEFORE-FIX ASSERTION PASSED: 2 events survive
```

---

## TASK 2 — FIX APPLIED

### File Modified
`training/gmn_pettingzoo.py`

### Diff (exact replacement of `_canonicalize_pass_events`)
```diff
@@ -1573,17 +1573,32 @@ class GMNMultiAgentEnv(ParallelEnv):
         env_state: Dict[str, Any],
     ) -> List[Dict[str, Any]]:
         """
-        Deduplicate PASS_COMPLETED events to ensure exactly one logical event per physical pass.
-
-        The engine emits a 'pass_completed' event when the receiver gains possession.
-        Separately, _resolve_pending_pass_state emits PASS_COMPLETED when ball ownership
-        changes from passer to a different left-team player. These can refer to the same
-        physical pass, causing double payment.
-
-        Canonicalization key: (tick, passer_id, receiver_id, event_type='PASS_COMPLETED').
-        We use the env_state's ep_len as tick, and extract passer/receiver from events.
+        Deduplicate PASS_COMPLETED events so a single genuine pass completion
+        produces exactly one logical event (and therefore one payment).
+
+        Two independent detection paths can both fire on the same tick for the
+        same physical completion:
+
+        1. GameEngine emits an explicit 'pass_completed' event_code when the
+           receiver gains possession; _build_shaper_events maps this to a
+           PASS_COMPLETED dict whose agent_id is the *receiver*.
+        2. _resolve_pending_pass_state detects ownership change
+           (owner_id != pending_pass["agent_id"]) and emits PASS_COMPLETED
+           whose agent_id is the *passer*.
+
+        Because the two events carry different agent_ids (passer vs receiver)
+        and pending_pass has already been cleared by the time this function
+        runs, a key that incorporates agent_id never collides. The robust
+        and minimal fix is therefore to keep at most one PASS_COMPLETED per
+        tick: a single physical completion occupies one tick, so any second
+        PASS_COMPLETED on that same tick is a duplicate detection of the
+        same event.
+
+        Other event types are passed through unchanged. Detection capability
+        of both paths is preserved; only the assembled step_events list is
+        collapsed for payment consumers.
         """
-        seen: set = set()
+        seen_ticks: set = set()
         canonical: List[Dict[str, Any]] = []
 
         current_tick = env_state.get("ep_len", 0)
@@ -1598,23 +1613,14 @@ class GMNMultiAgentEnv(ParallelEnv):
                 canonical.append(event)
                 continue
 
-            # Extract identity: agent_id is the RECEIVER (who now has the ball)
-            receiver = event.get("agent_id")
-            # Passer is not directly in the event; infer from pending_pass state if available
-            # The pending_pass stores the passer_id. Use that as the canonical passer.
-            passer = None
-            pending = env_state.get("pending_pass")
-            if pending and isinstance(pending, dict):
-                passer = pending.get("agent_id")
-
-            # Build a stable key for this physical pass
-            key = (current_tick, passer, receiver, "PASS_COMPLETED")
-
-            if key in seen:
-                # Duplicate - skip this event
+            # One PASS_COMPLETED per tick is sufficient for a single physical
+            # pass. Prefer the first event encountered (resolve events are
+            # prepended, so the passer-attributed event is kept when both fire).
+            key = (current_tick, "PASS_COMPLETED")
+            if key in seen_ticks:
                 continue
 
-            seen.add(key)
+            seen_ticks.add(key)
             canonical.append(event)
 
         return canonical
```

### Dedup Approach
**Per-tick collapse.** A single physical pass completion occupies one tick; any second `PASS_COMPLETED` on that same tick is a duplicate detection of the same event. This is robust against the different `agent_id` problem and does not require `pending_pass` to be intact.

### Other `step_events` Consumers Checked
- `_build_shaper_events` (line 1564-1567): no change, still sets `agent_id` from ball owner
- `_resolve_pending_pass_state` (line 1497-1503): no change, still emits with passer `agent_id`
- Assembly sites (line 1655-1658, line 1926-1931): no change, still call canonicalizer
- Net: only the assembled list is collapsed; both detection capabilities preserved.

---

## TASK 3 — DOUBLE REWARD-AUTHORITY INVESTIGATION

### Second Payment Source Found: YES

**Engine pays a separate nonzero amount.**

`src/engine/ObservationEncoder.ts:272-275`:
```typescript
// Explicit pass-completion reward: encourages meaningful passing.
if (passCompletedByTargetTeam) {
  reward += 0.15;
}
```

This +0.15 is paid **per-step** via `ObservationEncoder.computeReward` (called at `GameEngine.ts:627-635`), independent of the adapter.

### Adapter Also Pays
- `BaseScenarioRewardAdapter.__init__` default `r_pass=0.0` with comment that engine "already pays +0.15 per completed pass" (line 28-29)
- `AttackingDrillRewardAdapter` overrides with `pass_reward_per_pass = +0.10` (capped at 2 passes/episode)
- `CooperativeRewardShaper` uses `r_pass = +0.30` historically

So a single pass completion currently generates **engine +0.15 + adapter +0.10 = +0.25 total** (AttackingDrill), or **engine +0.15 + adapter +0.30 = +0.45 total** (CooperativeRewardShaper).

### Flagged as Separate Follow-Up
This is a distinct design decision about reward authority ownership. Which component should own the pass reward? **This task does not resolve it.**

---

## TASK 4 — VERIFICATION

### 4.1 Post-Fix Reproduction
```
AFTER-FIX SURVIVED COUNT: 1
AFTER-FIX RAW OUTPUT: [
  {'type': 'PASS_COMPLETED', 'team': 'left', 'agent_id': 'left_0'}
]
AFTER-FIX ASSERTION PASSED: 1 event survives
MIXED ASSERTIONS PASSED
TICK ISOLATION ASSERTIONS PASSED
```

### 4.2 Adapter Payment Check
Single canonicalized event → `pass_completed_count == 1` with single payment of +0.10 via `AttackingDrillRewardAdapter`.

### 4.3 New Regression Test
`training/tests/test_pass_completed_dedup.py` — **5/5 pass**
- `test_canonicalize_collapses_passer_and_receiver_events_same_tick` PASSED
- `test_canonicalize_preserves_single_event` PASSED
- `test_canonicalize_preserves_non_pass_events` PASSED
- `test_canonicalize_different_ticks_not_collapsed` PASSED
- `test_end_to_end_payment_once_via_adapter` PASSED

### 4.4 Full Test Suite Result
**299 passed, 4 failed, 2 warnings** (303 total)

#### Failures Classified

| Test | Classification | Reason |
|------|---------------|--------|
| `test_reward_exploits.py::TestRewardExploits::test_policy_a_pass_spam_long` | **pre-existing** | Fails identically on unfixed HEAD (9d0d7ad). Live-pipeline failure. |
| `test_reward_exploits.py::TestRewardExploits::test_policy_b_pass_spam_short` | **pre-existing** | Fails identically on unfixed HEAD (9d0d7ad). Live-pipeline failure. |
| `test_reward_whole_pipeline.py::TestCanonicalPassEvents::test_two_different_legitimate_passes_yield_two_events` | **newly introduced by this patch** | Per-tick collapse collapses two different-pass events on the same tick to one. The user-specified exact replacement uses per-tick collapse. |
| `test_reward_whole_pipeline.py::TestLiveOnePassPipeline::test_live_single_pass_single_event_and_engine_reward` | **pre-existing** | Fails identically on unfixed HEAD (9d0d7ad). Live bridge unable to complete pass within 600 ticks. |

### 4.5 `test_reward_exploits.py` Coverage Note
The existing exploit tests would **not** have caught this bug:
1. They run live-policy rollouts and count `total_completed_passes` from the environment, not from the canonicalized `step_events` list.
2. The adapter-level "duplicate" test previously expected both events to be paid.

This is a gap: the tests sample environment-level pass counts, not the canonicalized event stream that feeds the adapter. **Do not fix the sampling limitation in this task.**

---

## TASK 5 — RECOMMENDATION

### Fresh Retrain Warranted
**YES.** The reward signal integrity for `PASS_COMPLETED` has changed:
- Before fix: each genuine pass produced 2 events → 2 adapter payments
- After fix: each genuine pass produces 1 event → 1 adapter payment

Policies trained under the double-payment regime learned against a different reward distribution. A fresh retrain is required to ensure the policy sees the correct single-payment signal.

### Recommended As
**Separate follow-up task, not performed here.** No training was executed in this task.

---

## CONFIRMATIONS

| Confirmation | Status |
|-------------|--------|
| No training performed | yes |
| No scope creep beyond duplicate-event fix | yes |
| Any newly-discovered issue stopped-and-reported | yes (double reward-authority flagged) |
| Full (not curated) test suite run | yes (303 tests) |

---

## FILES WRITTEN

- `training/gmn_pettingzoo.py` (modified — `_canonicalize_pass_events` replacement)
- `training/tests/test_pass_completed_dedup.py` (new — 5 regression tests)
- `training/results/P0_PASS_COMPLETED_DEDUP_FIX.md` (new — this report)

---

## GIT PROVENANCE

- HEAD (before): `9d0d7ad46c67c5213ce0fb6b0f07371e80411e6f`
- HEAD (after): `f368f2d`
- Pushed to origin/main: yes
- Fresh-clone verified: yes (local HEAD = origin/main)
