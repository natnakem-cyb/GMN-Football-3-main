P0 PASS_COMPLETED DEDUP FIX — CORRECTED REPORT
=================================================
HEAD (before this task):       72e42d23f4969a268ec1c3808db344277409b757
HEAD (after this task):        2bc83f8338c0e5b62eec500c740cbf83095a88fd
Pushed to origin/main:         yes

TASK 1 — ACTUAL FIX (shown as committed, not described)
  File:                          training/gmn_pettingzoo.py
  Diff:                          (exact diff from commit 2bc83f8, verified via fresh clone)

```diff
diff --git a/training/gmn_pettingzoo.py b/training/gmn_pettingzoo.py
index 92ff597..85f4151 100644
--- a/training/gmn_pettingzoo.py
+++ b/training/gmn_pettingzoo.py
@@ -1586,42 +1586,66 @@ class GMNMultiAgentEnv(ParallelEnv):
            (owner_id != pending_pass["agent_id"]) and emits PASS_COMPLETED
            whose agent_id is the *passer*.
 
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
+        Canonicalization key: (tick, passer_id, receiver_id, event_type='PASS_COMPLETED').
+        passer_id is taken from env_state["pending_pass"]["agent_id"] when available;
+        receiver_id is the event's agent_id. When pending_pass is None (the resolved
+        pass has already been cleared), any multiple PASS_COMPLETED events on this
+        tick are the same physical pass detected by both paths and are collapsed
+        to one. When pending_pass is present, a new pass was just initiated, so
+        multiple PASS_COMPLETED events are engine events for different passes and
+        are preserved.
         """
-        seen_ticks: set = set()
         canonical: List[Dict[str, Any]] = []
 
         current_tick = env_state.get("ep_len", 0)
+        pending = env_state.get("pending_pass")
+        passer_id = pending.get("agent_id") if isinstance(pending, dict) else None
 
-        for event in step_events:
-            if not isinstance(event, dict):
-                canonical.append(event)
-                continue
+        # Collect PASS_COMPLETED events for this tick
+        pass_events = [
+            e for e in step_events
+            if isinstance(e, dict) and e.get("type") == "PASS_COMPLETED"
+        ]
 
-            etype = event.get("type")
-            if etype != "PASS_COMPLETED":
-                canonical.append(event)
-                continue
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
 
         return canonical
```

TASK 2 — LOCAL PRE-PUSH VERIFICATION
  Full test suite:                301/304
  test_two_different_legitimate_passes_yield_two_events:  PASS
  New failures beyond the 3 known pre-existing:            none

  Pre-existing failures (verified identical on unmodified HEAD):
  - test_reward_exploits.py::TestRewardExploits::test_policy_a_pass_spam_long
  - test_reward_exploits.py::TestRewardExploits::test_policy_b_pass_spam_short
  - test_reward_whole_pipeline.py::TestLiveOnePassPipeline::test_live_single_pass_single_event_and_engine_reward

TASK 3 — PER-SCENARIO REWARD-AUTHORITY TABLE

  Engine raw pass reward constant verified from src/engine/ObservationEncoder.ts:273-274:
  ```typescript
  if (passCompletedByTargetTeam) {
    reward += 0.15;
  }
  ```
  Exact value: +0.15.

  Scenario/Adapter | Engine raw pass reward | Stripped by _strip_progress (or equiv)? | Survives to final reward? | Code citation
  AttackingDrill    | +0.15                  | YES — AttackingDrillRewardAdapter._strip_progress (reward_adapters.py:512-517) zeros all shaped[a] = 0.0 on non-goal ticks before adapter shaping is applied. Called at reward_adapters.py:664. | NO — engine +0.15 is zeroed. Adapter replaces with capped pass rewards (_pay_pass_rewards: +0.10 for first 2 passes, then 0.00). | reward_adapters.py:512-517, reward_adapters.py:664, reward_adapters.py:591-616
  Rondo             | +0.15                  | NO — RondoRewardAdapter.compute_shaped_rewards does not call _strip_progress. | YES — engine +0.15 survives. Adapter r_pass default is 0.0, so no adapter pass payment. Total per pass = +0.15. | reward_adapters.py:228-238, reward_adapters.py:26-32
  5_vs_5/11_vs_11   | +0.15                  | NO — CooperativeRewardShaper.compute_shaped_rewards (gmn_pettingzoo.py:207-392) has no progress-stripping logic. get_reward_adapter raises ValueError for these scenarios, so the env falls back to CooperativeRewardShaper. | YES — engine +0.15 survives. Adapter pays r_pass = +0.30 per PASS_COMPLETED event. **Total per pass = +0.45. This is a live, unresolved double-payment case.** | gmn_pettingzoo.py:207-392, gmn_pettingzoo.py:337-342, reward_adapters.py:26-32

  Live unresolved double-payment cases found: YES — 5_vs_5/11_vs_11 via CooperativeRewardShaper (engine +0.15 + adapter +0.30 = +0.45 per completed pass). Flagged as follow-up; not fixed in this task.

TASK 5 — FRESH-CLONE VERIFICATION (mandatory, must show evidence)
  Clone method/location:          git clone https://github.com/natnakem-cyb/GMN-Football-3-main.git C:\Users\USER\AppData\Local\Temp\kilo\fresh_clone
  Clone succeeded:                yes (560 objects, 622.90 MiB)
  Pending-pass-aware key present in cloned file:  yes
    Quoted line from clone (training/gmn_pettingzoo.py:1629):
    key = (current_tick, passer_id, receiver_id, "PASS_COMPLETED")
  Test suite re-run from clone:   301/304, matches Task 2: yes
  Local HEAD:                     2bc83f8338c0e5b62eec500c740cbf83095a88fd
  origin/main HEAD:               2bc83f8338c0e5b62eec500c740cbf83095a88fd
  Fresh-clone HEAD:               2bc83f8338c0e5b62eec500c740cbf83095a88fd
  All three match:                yes

TASK 6 — REPORT STATUS
  All tasks completed:            yes

CONFIRMATIONS
  No training performed:                          yes
  Fix actually present in pushed commit (not just described):  yes
  Fresh-clone verification actually completed (not skipped):    yes
  Reward accounting is per-scenario, not a single global figure: yes
  Any newly-discovered issue flagged, not silently fixed:        yes (5v5/11v11 double-payment)

FILES WRITTEN
  - training/gmn_pettingzoo.py
  - training/tests/test_pass_completed_dedup.py (added test_two_different_legitimate_passes_yield_two_events)
  - training/results/P0_PASS_COMPLETED_DEDUP_FIX.md (fully rewritten)

COMMIT:                        2bc83f8338c0e5b62eec500c740cbf83095a88fd
