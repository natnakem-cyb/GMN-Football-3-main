# Experiment D — Tackle Spam Forensic Investigation

**Date:** 2026-09-17  
**HEAD at investigation:** `f2fd157` (after per-run eval CSV isolation fix)  
**Status:** Measurement complete — mechanism not reproduced on current checkpoints  
**Objective:** Determine and eliminate the mechanism that makes tackle-related behavior an attractive policy strategy despite producing zero goals.  

---

## 1. Executive summary

The 9/14 deterministic evaluation of Experiment B checkpoints recorded **9.48 tackles/episode** (seed7) and **42.64 tackles/episode** (seed999). Those figures are **not reproducible** with the checkpoint files currently on disk. Re-evaluation of the same-named `_clean.pt` and `_200k_B` files today yields **0.0 tackles/episode** for all four seeds, including seeds 7 and 999.

Root cause: the checkpoint files on disk were **replaced after the 9/14 eval**. The SHA256 hashes recorded in the 9/14 `comprehensive_eval_*_clean.json` sidecars do not match the current files. The original files that exhibited tackle spam are no longer present in the working tree and are not recoverable from git history.

**Implication for D:** the "repeated behavioral failure signature" across B and C was observed on specific historical checkpoint weights that are no longer available. The current policy weights exhibit **policy paralysis** (0 tackles, 0 passes, 0 shots), not tackle spam. The forensic investigation therefore shifts from "why does the current policy tackle-spam" to "why did an older checkpoint version tackle-spam, and what changed."

---

## 2. Failed-tackle penalty question

**Answered definitively from code inspection:**

A failed tackle attempt (`Physics.ts executeTackle` returning `'miss'`) carries **no penalty** in the reward layer:
- No event is emitted for a miss.
- The agent pays only the standard step cost (`-0.005`) and any dense proximity reward earned that tick.
- There is no explicit "failed attempt" penalty in `reward_adapters.py`.

A `'foul'` outcome (15% deterministic roll when within 0.065m of the ball owner) emits a `foul` event, but the `AttackingDrillRewardAdapter` does **not** process `foul` events. The engine maps `interception`, `tackle`, and `foul` to `TURNOVER_CONCEDED` with `team="right"` (`gmn_pettingzoo.py:1750-1752`), which means:
- A right-team tackle → `TURNOVER_CONCEDED` for left victim → left victim gets `-0.10` (with tackle-spam suppression).
- A left-team tackle → also mapped to `TURNOVER_CONCEDED` with `team="right"` → this would penalize the left team even when they win the ball. **This is a likely event-mapping bug** (see §6).

**Conclusion:** a failed tackle is a free action in reward terms (H-free-action is supported for the cost dimension). The only deterrents are the 25-tick cooldown and the opportunity cost of not taking a productive action.

---

## 3. Current checkpoint behavior (re-eval, 2026-09-17)

Re-evaluated all four Experiment B `_200k_B` checkpoints with the instrumented eval script (`training/eval_tackle_forensics.py`), 10 episodes each, deterministic, `base_seed=500000`.

| Seed | Goal rate | Mean tackles/ep | Mean shots/ep | Mean passes/ep | Mean reward |
|------|----------|-----------------|---------------|----------------|-------------|
| 42   | 0.0%     | 0.0             | 0.0           | 0.0            | -0.708      |
| 123  | 0.0%     | 0.0             | 0.0           | 0.0            | -0.754      |
| 7    | 0.0%     | 0.0             | 0.0           | 0.0            | -0.683      |
| 999  | 0.0%     | 0.0             | 0.0           | 0.0            | -0.770      |

**Tackle-heavy episodes (≥3 tackles): 0 across all seeds.**

First-tick action selection at kickoff (ball unowned):
- seed7: all three agents select DRIBBLE (17)
- seed999: all three agents select LONG_PASS (10)
- seed123: left_1 and left_3 select LONG_PASS (10), left_2 selects SLIDING/tackle (16) — but this does not repeat; overall episode tackle count is 0.

First-tick mask for off-ball left agents:
- `tackle_legal: 1`
- `shot_legal: 0`
- `pass_legal: 0`
- `mask_sum: 15` (15 of 19 actions legal)

**Observation:** tackle is legal at kickoff for off-ball agents, but the current policy does not select it. The policy selects idle/movement/pass/dribble actions instead, all of which also yield negative reward.

---

## 4. Action-mask legality distribution

The instrumented eval logged the action mask at every tick where an off-ball left agent was active. Across all four seeds, 10 episodes each:

- **Tackle is always legal** for off-ball agents (mask index 16 = 1 at every observed tick).
- **Shot and pass are always illegal** for off-ball agents (mask indices 9, 10, 11, 12 = 0 at every observed tick) because the agent does not have possession.
- **Mask sum is consistently 15**, meaning 4 actions are masked out. The masked-out actions are: SHOT (12), SHORT_PASS (9), LONG_PASS (10), HIGH_PASS (11). All other actions (IDLE, movement, SPRINT, RELEASE_SPRINT, TACKLE, DRIBBLE, RELEASE_DRIBBLE) are legal.

**Conclusion (H-mask):** The mask structure is a necessary condition for tackle spam: if tackle were the only legal action, any policy with non-zero entropy would select it. But it is not sufficient: the current policy has 15 legal options and still selects 0 tackles. The mask alone does not explain the historical tackle spam.

---

## 5. Reward decomposition per tick

The instrumented eval monkey-patched `AttackingDrillRewardAdapter.compute_shaped_rewards` to log per-agent base reward, shaped reward, and delta at every tick. Across all four seeds:

- **Base reward** (from engine): `-1.0` for non-goal ticks, `0.0` for goal ticks.
- **Step cost**: `-0.005` per tick (applied to all agents).
- **Possession reward**: `+0.01` per tick to all left agents when left team has the ball.
- **PBRS proximity reward**: `0.02 * (phi(new_dist) - phi(prev_dist))` to all left agents, gated on left possession for the PBRS-to-goal term, but **not possession-gated** for the nearest-left-agent-to-ball proximity term.
- **Pass rewards**: capped at 2 passes per episode (`+0.10` each first two passes).
- **Shot rewards**: capped at 2 attempts (`+0.15` first, `+0.05` second).
- **Turnover penalty**: `-0.10` to the victim, suppressed within 10-tick grace window.

**No tick-level reward data for tackle actions exists in the current eval** because the current policy never selects tackle. The per-tick logs show only the standard penalty stream (step cost + occasional turnover penalty when the right team wins the ball).

**Conclusion (H-downstream / H-reward):** The current policy does not receive any positive reward from tackle actions because it never takes them. The dense proximity reward while defending is the only positive signal available to off-ball agents, but the current policy does not associate tackle with accessing that signal.

---

## 6. Engine event-mapping anomaly

A critical finding from code inspection is in `gmn_pettingzoo.py:1750-1752`:

```python
elif ev_type in ("interception", "tackle", "foul"):
    shaper_type = "TURNOVER_CONCEDED"
    shaper_team = "right"
```

This hardcodes `team="right"` for **all** tackle events, regardless of which team executed the tackle. In `reward_adapters.py`, `_LEFT_VICTIM_TYPES` only penalizes the left team when `team="right"`:

```python
_LEFT_VICTIM_TYPES = frozenset(("PASS_INTERCEPTED", "PASS_FAILED", "TURNOVER_CONCEDED"))
...
if etype in _LEFT_VICTIM_TYPES:
    victim = self._resolve_victim(shaped, aid)
    ...
    penalty_applied = self._penalize_victim(shaped, victim, etype)
```

**Effect:** if a left-team agent successfully tackles and the engine emits a `tackle` event, the reward adapter treats it as `TURNOVER_CONCEDED` for the left team and penalizes the victim `-0.10`. This means a successful left-team tackle is **rewarded as a turnover**, which is the opposite of what should happen.

This is a likely candidate for the historical tackle-spam mechanism: if the policy learned that tackle actions produce `TURNOVER_CONCEDED` events for the left team, it would be reinforced to avoid tackling — unless the penalty is suppressed or the policy has learned to tolerate it. The current policy appears to have learned to avoid tackling entirely (policy paralysis), which is consistent with this bug being present.

**This bug is present in the current code and has not been fixed.** It is a measurement-only finding; no code change is proposed in this brief.

---

## 7. Checkpoint provenance failure

The 9/14 eval sidecars (`comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed{7,999}_clean.json`) record `model_sha256` values that do **not** match the current `_clean.pt` files:

| Seed | 9/14 eval JSON SHA256 | Current `_clean.pt` SHA256 | Match? |
|------|----------------------|---------------------------|--------|
| 7    | `c2c069a974225c7...` | `eaa394cb30bab7e1...`     | No     |
| 999  | `19967b541cfb23e3...` | `5886b1b9255570b5...`     | No     |
| 42   | `e0e5629a95103a81...` | `1d9be7e16f07d9be...`     | No     |
| 123  | `d5aa860ee11699d0...` | `7624efee69555a3e...`     | No     |

The original checkpoint files that exhibited tackle spam are **not present** in the current working tree and are **not recoverable** from git history (the blobs are not in the object database).

The current `_clean.pt` and `_best.pt` files have `timesteps=50176` (step 50,176), which corresponds to the `best_deterministic_step: 50176` recorded in the Experiment B manifest for seeds 7 and 999. The manifest also records `best_deterministic_step: 100352` for seeds 42 and 123, but the corresponding `_best.pt` files also have `timesteps=50176`. This suggests the `_best.pt` files were overwritten at some point after the manifests were written, and the current files represent a mid-training checkpoint (step 50,176) rather than the best checkpoint (step 100,352) recorded in the manifest.

---

## 8. Hypothesis assessment

| Hypothesis | Assessment | Evidence |
|------------|------------|----------|
| **H-mask** | **Partially supported** — tackle is always legal for off-ball agents, but current policy does not select it. The mask is a necessary but not sufficient condition. | Mask data from 10 episodes × 4 seeds: tackle_legal=1 at every tick, but tackle_actions=0 |
| **H-free-action** | **Supported** — a failed tackle attempt carries no penalty. The 25-tick cooldown is the only cost. | Code inspection: `Physics.ts:336` returns `'miss'` with no event; `reward_adapters.py` does not process misses. |
| **H-downstream** | **Unproven** — the dense proximity reward while defending exists and is not possession-gated. A policy that learned to tackle to get closer to the ball could theoretically exploit this. But the current policy does not tackle, so we cannot measure the reward delta for tackle ticks. | Code inspection: `DENSE_PROXIMITY_REWARD = 0.02` pays to all left agents when nearest left agent gets closer to ball, regardless of possession. |
| **H-engine** | **Supported** — tackle success probability is 85% within 0.065m of ball owner, and the ball is deflected away (potentially creating a loose-ball scramble). The engine does not distinguish successful/failed tackles in the event stream for the reward adapter; all are mapped to `TURNOVER_CONCEDED`. | Code inspection: `Physics.ts:348-382`; `gmn_pettingzoo.py:1750-1752`. |

**Dominant mechanism for historical tackle spam (best-supported explanation):**

The combination of H-free-action + H-engine + the event-mapping bug (§6) is the most likely explanation for the historical tackle spam:

1. **Failed tackles are free** — no penalty, only step cost.
2. **Successful tackles have high probability** (85%) and deflect the ball, creating a loose-ball scramble where the dense proximity reward can pay.
3. **The event-mapping bug** maps ALL tackles (including left-team tackles) to `TURNOVER_CONCEDED` with `team="right"`. This means a left-team tackle is penalized as if the left team lost the ball, even though they won it. The tackle-spam protection suppresses duplicate penalties within 10 ticks, so rapid re-tackles are not repeatedly penalized.
4. **The net effect** for a historical policy might have been: tackle → sometimes succeed (win ball, get proximity reward) → sometimes fail (no penalty) → sometimes get penalized (but suppressed by grace window). The expected value could be slightly positive or at least less negative than idle/movement.

**Why the current policy does not tackle:** the current policy weights (step 50,176) have not learned to associate tackle with any positive outcome. They have converged to policy paralysis instead. The mechanism that produced tackle spam in the older checkpoint is no longer active in the current weights.

---

## 9. What the investigation cannot establish

- **Cannot replay the 9/14 tackle-spam episodes** because the original checkpoint files are gone.
- **Cannot measure per-tackle reward delta** for historical tackle spam because no current episode contains a tackle action.
- **Cannot confirm whether the event-mapping bug** was the primary driver of historical tackle spam or merely a contributing factor.

---

## 10. Recommended next steps (for a follow-up brief, not implemented here)

1. **Recover the original checkpoint files** from backup, CI artifacts, or external storage. The 9/14 eval sidecars contain the SHA256 hashes; any backup or artifact store that preserved the working tree on 9/14 should contain the original files.
2. **Fix the event-mapping bug** in `gmn_pettingzoo.py:1750-1752`: left-team tackles should not be mapped to `TURNOVER_CONCEDED` with `team="right"`. The mapping should distinguish left-team from right-team tackles.
3. **Re-evaluate the recovered checkpoint** with the instrumented eval to confirm tackle spam and measure per-tackle reward deltas.
4. **Introduce a single targeted intervention** only after the mechanism is confirmed on the recovered checkpoint. The intervention should change exactly one causal variable (e.g., fix the event mapping, add a failed-tackle penalty, or adjust the proximity reward gate).

---

## 11. Deliverables produced

| File | Purpose |
|------|---------|
| `training/eval_tackle_forensics.py` | Instrumented eval script (measurement only, no code changes to reward/mask/engine) |
| `training/models/tackle_forensics_mappo_ac3v1_seed{42,123,7,999}_200k_B.json` | Per-seed per-episode transition logs (0 tackles across all seeds) |
| `training/models/tackle_forensics_mappo_ac3v1_seed123_100k_E0.json` | Experiment C seed123 checkpoint (0 tackles) |
| `training/results/tackle_forensics_summary.csv` | Aggregated episode-level metrics |
| `training/results/EXPERIMENT_D_TACKLE_SPAM_FORENSICS.md` | This report |

---

## 12. Provenance

- B closed as negative: `training/results/EXPERIMENT_B_PROVENANCE.md`
- B best-vs-final gap: `training/results/EXPERIMENT_B_HORIZON_200k.md`
- C report: `training/results/EXPERIMENT_C_EXPLORATION_ABLATION_100k.md`
- D plan: `training/results/D_EXPERIMENT_TACKLE_FORENSICS.md`
- Experimental hierarchy: `training/experimental_hierarchy.md`
