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

The broken canonicalizer used key `(tick, passer-from-pending, receiver, "PASS_COMPLETED")`. After resolution, `pending_pass` is `None`, so `passer` is `None`. Keys never collide → both events survive → double payment.

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
@@ -1573,17 +1573,70 @@ class GMNMultiAgentEnv(ParallelEnv):
          env_state: Dict[str, Any],
      ) -> List[Dict[str, Any]]:
          """
-        Deduplicate PASS_COMPLETED events so a single genuine pass completion
-        produces exactly one logical event (and therefore one payment).
-
-        Two independent detection paths can both fire on the same tick for the
-        same physical completion:
-
-        1. GameEngine emits an explicit 'pass_completed' event_code when the
-           receiver gains possession; _build_shaper_events maps this to a
-           PASS_COMPLETED dict whose agent_id is the *receiver*.
-        2. _resolve_pending_pass_state detects ownership change
-           (owner_id != pending_pass["agent_id"]) and emits PASS_COMPLETED
-           whose agent_id is the *passer*.
-
-        Because the two events carry different agent_ids (passer vs receiver)
-        and pending_pass has already been cleared by the time this function
-        runs, a key that incorporates agent_id never collides. The robust
-        and minimal fix is therefore to keep at most one PASS_COMPLETED per
-        tick: a single physical completion occupies one tick, so any second
-        PASS_COMPLETED on that same tick is a duplicate detection of the
-        same event.
-
-        Other event types are passed through unchanged. Detection capability
-        of both paths is preserved; only the assembled step_events list is
-        collapsed for payment consumers.
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
+        Canonicalization key: (tick, passer_id, receiver_id, event_type='PASS_COMPLETED').
+        passer_id is taken from env_state["pending_pass"]["agent_id"] when available;
+        receiver_id is the event's agent_id. When pending_pass is None (the resolved
+        pass has already been cleared), any multiple PASS_COMPLETED events on the
+        same tick are the same physical pass detected by both paths and are collapsed
+        to one. When pending_pass is present, a new pass was just initiated, so
+        multiple PASS_COMPLETED events are engine events for different passes and
+        are preserved.
          """
-        seen_ticks: set = set()
+        canonical: List[Dict[str, Any]] = []
+
+        current_tick = env_state.get("ep_len", 0)
+        pending = env_state.get("pending_pass")
+        passer_id = pending.get("agent_id") if isinstance(pending, dict) else None
+
+        # Collect PASS_COMPLETED events for this tick
+        pass_events = [
+            e for e in step_events
+            if isinstance(e, dict) and e.get("type") == "PASS_COMPLETED"
+        ]
+
+        if len(pass_events) <= 1:
+            # No collision possible
+            return list(step_events)
+
+        if passer_id is not None:
+            # Use (tick, passer, receiver) key so different passes on the
+            # same tick are preserved.
+            seen: set = set()
+            for event in step_events:
+                if not isinstance(event, dict):
+                    canonical.append(event)
+                    continue
+
+                etype = event.get("type")
+                if etype != "PASS_COMPLETED":
+                    canonical.append(event)
+                    continue
+
+                receiver_id = event.get("agent_id")
+                key = (current_tick, passer_id, receiver_id, "PASS_COMPLETED")
+                if key in seen:
+                    continue
+
+                seen.add(key)
+                canonical.append(event)
+        else:
+            # pending_pass is None: the resolved pass has already been
+            # cleared, so any multiple PASS_COMPLETED events on this tick
+            # are the same physical pass detected by both paths (passer +
+            # receiver). Keep only the first event (the resolve event is
+            # prepended, so passer attribution is preserved).
+            kept_pass = False
+            for event in step_events:
+                if isinstance(event, dict) and event.get("type") == "PASS_COMPLETED":
+                    if not kept_pass:
+                        canonical.append(event)
+                        kept_pass = True
+                else:
+                    canonical.append(event)
+
+        return canonical
```

### Dedup Approach
**Pending-pass-aware key with per-tick fallback.** The canonicalizer now distinguishes two cases:
- `pending_pass` is present: a new pass was just initiated. The key is `(tick, passer_id, receiver_id, "PASS_COMPLETED")`, where `passer_id` comes from `env_state["pending_pass"]["agent_id"]` and `receiver_id` comes from the event's `agent_id`. Different passes on the same tick are preserved.
- `pending_pass` is `None`: the resolved pass has already been cleared. Any multiple `PASS_COMPLETED` events on the same tick are the same physical pass detected by both paths (passer + receiver). They are collapsed to one, keeping the first event (the resolve event is prepended, so passer attribution is preserved).

### Other `step_events` Consumers Checked
- `_build_shaper_events` (line 1564-1567): no change, still sets `agent_id` from ball owner
- `_resolve_pending_pass_state` (line 1497-1503): no change, still emits with passer `agent_id`
- Assembly sites (line 1655-1658, line 1926-1931): no change, still call canonicalizer
- Net: only the assembled list is collapsed; both detection capabilities preserved.

---

## TASK 3 — DOUBLE REWARD-AUTHORITY INVESTIGATION

### Second Payment Source Found: YES

**Engine pays a separate nonzero amount, unconditionally alongside the adapter.**

`src/engine/ObservationEncoder.ts:272-275`:
```typescript
// Explicit pass-completion reward: encourages meaningful passing.
if (passCompletedByTargetTeam) {
  reward += 0.15;
}
```

This fires in `ObservationEncoder.computeReward` whenever `passCompletedByTargetTeam` is true. In `GameEngine.ts:619-635`, that flag is set purely from the engine's own event stream:
```typescript
const passCompletedByLeft = newEventsThisTick.some(
  (e) => e.type === 'pass_completed' && e.team === 'left'
);
// ...
let { reward, checkpoint, newMaxBallProgressX } = ObservationEncoder.computeReward(
  prevBallX, this.ball.position.x, goalScoredThisTick,
  CONTROLLED_TRAINING_TEAM, this.maxBallProgressX,
  passCompletedByLeft, _ballOwnerTeam,
);
```

There is no gating on whether the adapter also pays, no consultation of `step_events`, and no check for duplicate detections. It is a separate reward authority.

### Adapter Also Pays
- `BaseScenarioRewardAdapter.__init__` default `r_pass=0.0` with the comment that the engine "already pays +0.15 per completed pass" (line 28-29)
- `AttackingDrillRewardAdapter` overrides this with `pass_reward_per_pass = +0.10` (capped at 2 passes/episode)
- `CooperativeRewardShaper` uses `r_pass = +0.30` historically

So for one real pass completion:
- Engine pays +0.15 via `ObservationEncoder.computeReward`
- Adapter pays +0.10 via `AttackingDrillRewardAdapter._pay_pass_rewards`
- **Total = +0.25**

Before the dedup fix, the adapter paid twice because it saw two `PASS_COMPLETED` events:
- Engine +0.15 + adapter +0.20 (2 × +0.10) = **+0.35**

After the dedup fix, the adapter pays once:
- Engine +0.15 + adapter +0.10 = **+0.25**

### Plain Report
The claim that "the reward signal is now correctly single-paid" is **false** if it refers to the total reward for a pass completion. The adapter is now single-paid, but the engine pays a separate +0.15 unconditionally. The total reward per pass is still engine + adapter, not adapter alone. This is a distinct design decision about reward authority ownership. **This task does not resolve it.**

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
Single canonicalized event → `pass_completed_count == 1` with single payment of +0.10 via `AttackingDrillRewardAdapter`. Non-event agent receives only dense possession - step cost = +0.005.

### 4.3 New Regression Test
`training/tests/test_pass_completed_dedup.py` — **5/5 pass**
- `test_canonicalize_collapses_passer_and_receiver_events_same_tick` PASSED
- `test_canonicalize_preserves_single_event` PASSED
- `test_canonicalize_preserves_non_pass_events` PASSED
- `test_canonicalize_different_ticks_not_collapsed` PASSED
- `test_end_to_end_payment_once_via_adapter` PASSED

### 4.4 Full Test Suite Result
**300 passed, 3 failed, 2 warnings** (303 total)

#### Failures Classified

| Test | Classification | Reason |
|------|---------------|--------|
| `test_reward_exploits.py::TestRewardExploits::test_policy_a_pass_spam_long` | **pre-existing** | Fails identically on unfixed HEAD (9d0d7ad). Live-pipeline failure. |
| `test_reward_exploits.py::TestRewardExploits::test_policy_b_pass_spam_short` | **pre-existing** | Fails identically on unfixed HEAD (9d0d7ad). Live-pipeline failure. |
| `test_reward_whole_pipeline.py::TestLiveOnePassPipeline::test_live_single_pass_single_event_and_engine_reward` | **pre-existing** | Fails identically on unfixed HEAD (9d0d7ad). Live bridge unable to complete pass within 600 ticks. |

**Zero new failures introduced by this patch.** `test_two_different_legitimate_passes_yield_two_events` now passes.

### 4.5 `test_reward_exploits.py` Coverage Note
The existing exploit tests would **not** have caught this bug:
1. They run live-policy rollouts and count `total_completed_passes` from the environment, not from the canonicalized `step_events` list.
2. The adapter-level "duplicate" test previously expected both events to be paid.

This is a gap: the tests sample environment-level pass counts, not the canonicalized event stream that feeds the adapter. **Do not fix the sampling limitation in this task.**

---

## TASK 5 — RECOMMENDATION

### Fresh Retrain Warranted
**YES.** The adapter-level reward signal for `PASS_COMPLETED` has changed:
- Before fix: each genuine pass produced 2 adapter payments (+0.20 for AttackingDrill)
- After fix: each genuine pass produces 1 adapter payment (+0.10 for AttackingDrill)

However, the engine still pays +0.15 per pass independently. So the total per-pass reward changed from +0.35 to +0.25 (AttackingDrill). Policies trained under the double-adapter-payment regime learned against a different reward distribution. A fresh retrain is required to ensure the policy sees the correct single-adapter-payment signal.

### Important Caveat
The "reward signal now correctly single-paid" claim is **false** for total pass reward. The adapter is single-paid, but the engine still pays +0.15 unconditionally alongside it. The total reward per pass is still engine + adapter, not adapter alone. This is a separate design decision about reward authority ownership that was investigated and flagged in Task 3.

### Recommended As
**Separate follow-up task, not performed here.** No training was executed in this task.

---

## CONFIRMATIONS

| Confirmation | Status |
|-------------|--------|
| No training performed | yes |
| No scope creep beyond duplicate-event fix | yes |
| Any newly-discovered issue stopped-and-reported | yes (double reward-authority flagged) |
| Full (not curated) test suite run | yes (303 tests, 300 passed, 3 pre-existing failures) |

---

## FILES WRITTEN

- `training/gmn_pettingzoo.py` (modified — `_canonicalize_pass_events` replacement)
- `training/tests/test_pass_completed_dedup.py` (new — 5 regression tests)
- `training/results/P0_PASS_COMPLETED_DEDUP_FIX.md` (new — this report)

---

## GIT PROVENANCE

- HEAD (before): `9d0d7ad46c67c5213ce0fb6b0f07371e80411e6f`
- HEAD (after): `1e66373470cc7a1e8d172cd17df5ba6a4560d539`
- Pushed to origin/main: yes
- Fresh-clone verified: no (see note below)

### Fresh-Clone Verification Note
A fresh-clone verification was attempted. The clone completed and HEAD matched `1e66373470cc7a1e8d172cd17df5ba6a4560d539`, but the checkout encountered a network error during blob download (`RPC failed; curl 18 transfer closed with outstanding read data remaining`). The partial clone reported `Clone succeeded, but checkout failed.` The three target files were present in the incomplete checkout. A subsequent retry with `--filter=blob:none` also failed with `early EOF`. Given these transient network failures on the remote side, fresh-clone verification is marked as **no** in the provenance block above, but the push to origin/main was successful and the commit is live.
