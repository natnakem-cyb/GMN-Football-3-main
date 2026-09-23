# Open Item — Tackle/Foul/Interception Event-Mapping Bug

**Opened:** 2026-09-23  
**Status:** Open — confirmed present, not fixed (out of scope for current task)  
**Severity:** Medium (currently inert, but would activate if tackle behavior resumes)

---

## Exact Location

**File:** `training/gmn_pettingzoo.py`

**Site 1 — `_build_shaper_events` (lines 1553-1555):**
```python
elif ev_type in ("interception", "tackle", "foul"):
    shaper_type = "TURNOVER_CONCEDED"
    shaper_team = "right"
```

**Site 2 — single-env `step()` event mapping (lines 1932-1934):**
```python
elif ev_type in ("interception", "tackle", "foul"):
    shaper_type = "TURNOVER_CONCEDED"
    shaper_team = "right"
```

Both sites map ALL `interception`, `tackle`, and `foul` engine events to `TURNOVER_CONCEDED` with `team="right"`.

---

## Defect Description

The engine emits `tackle`/`foul`/`interception` events with a `team` field indicating which team performed the action. The current code **hardcodes `team="right"`** for all three event types, regardless of which team actually performed the action.

**What should happen:**
- A `tackle` by the **right** team (defender) → `team="right"` → left team is the victim → `TURNOVER_CONCEDED` for left victim → correct.
- A `tackle` by the **left** team (attacker) → `team="left"` → this is a **left-team defensive action**, not a turnover. It should NOT be mapped to `TURNOVER_CONCEDED` for the left team.

**Current behavior:**
- A left-team tackle is mapped to `TURNOVER_CONCEDED` with `team="right"` → the left team is penalized for its own defensive action. This is incorrect.

The same issue applies to `foul` and `interception` events: the `team` field from the engine should be used, not hardcoded to `"right"`.

---

## Current Consequentiality Assessment

**Currently inert on current policy weights.**

Task A (Experiment D re-measurement) found that the current policy across all four evaluated checkpoints (seeds 42, 7, 123, 999) is **fully paralyzed**: 0 tackles, 0 shots, 0 passes per episode. The forced-tackle probe showed that the engine does not emit a `tackle` event in `academy_3_vs_1_with_keeper` when SLIDING is forced on off-ball left agents — the only reward effect is the standard `-0.005` step cost.

Because the current policy never executes tackles (and the engine path may not emit tackle events in this scenario configuration), the bug is **not currently exercised** and produces no observable incorrect behavior.

**Would it matter in future work?**
Yes. If:
- Policy paralysis is resolved and agents begin selecting tackle actions regularly, OR
- The scenario configuration changes to emit tackle events, OR
- A future experiment explicitly tests tackle behavior

...then this bug would cause left-team tackles to be misclassified as turnovers, producing incorrect reward signals and corrupting any measurement or training that depends on correct tackle attribution.

---

## Recommendation for Future Fix

This bug should be fixed in a dedicated task with the following scope:

1. **Diagnose first:** Confirm the engine's `team` field semantics for `tackle`/`foul`/`interception` events (which team is reported as the actor?).
2. **Fix the mapping:** Replace the hardcoded `shaper_team = "right"` with logic that reads the event's actual `team` field and maps:
   - Right-team tackle/foul/interception → `TURNOVER_CONCEDED` for left victim (current behavior, correct).
   - Left-team tackle/foul/interception → do NOT map to `TURNOVER_CONCEDED`; instead, either:
     - Ignore (no reward effect), or
     - Map to a new event type if left-team defensive actions should be rewarded.
3. **Add regression tests:** Verify that left-team and right-team tackle events are mapped correctly.
4. **Do not fix inline:** This bug was found during Task A measurement and is explicitly out of scope for that task. It requires its own diagnose-first task.

**Priority:** Low until tackle behavior is observed in live policy or scenario configuration changes.

---

## References

- Task A report: `training/results/EXPERIMENT_D_TACKLE_FORENSICS_CURRENT_HEAD.md`
- Code sites: `training/gmn_pettingzoo.py:1553-1555` and `training/gmn_pettingzoo.py:1932-1934`
