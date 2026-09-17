# Experiment F Provenance

**Date:** 2026-09-17  
**HEAD at eval:** `939dc22`  
**Primary scenario:** `academy_3_vs_1_with_keeper` (CTRL) / `academy_3_vs_1_with_keeper_onball` (INT)  
**Frozen checkpoint:** `mappo_academy_3_vs_1_with_keeper_seed42_best.pt` (step 100,352)  

---

## 1. Intervention scope

**Files changed:**

| File | Change | Purpose |
|------|--------|---------|
| `src/types/football.ts` | `setup.ball` type extended with optional `ownerId?: string \| null` | Allow scenario to declare initial ball owner |
| `src/engine/GameEngine.ts` | After player setup in `loadScenario`, assign `ball.ownerId` and `player.hasBall` from `scenario.setup.ball.ownerId` if present | Materialize initial ownership |
| `src/scenarios/ScenarioRegistry.ts` | New scenario `academy_3_vs_1_with_keeper_onball` with `ball: { x: 0.2, y: 0, z: 0, ownerId: 'left_1' }` and matching CAM spawn | Single initial-distribution intervention |

**Not changed:**
- Reward shaping (`training/reward_adapters.py`)
- GAE / rollout (`training/mappo_rollout.py`)
- Networks (`training/mappo_networks.py`)
- Action masks (`src/engine/ObservationEncoder.ts::getActionMask` unchanged)
- Training loop (`training/train_mappo.py`)

---

## 2. Freeze checklist

| Freeze item | Status | Evidence |
|-------------|--------|----------|
| Reward | ✅ Unchanged | `reward_adapters.py` not modified for this experiment |
| Strip | ✅ Unchanged | No strip/scenario skin changes |
| GAE λ/γ | ✅ Unchanged | `GAMMA=0.99, LAM=0.95` in `eval_occupancy_gate.py` matches production `mappo_rollout.py` |
| Value loss / critic architecture | ✅ Unchanged | Same checkpoint, same `CentralizedCritic` class |
| Action masks | ✅ Unchanged | `ObservationEncoder.getActionMask` logic unchanged; INT mask opens ball actions for on-ball player only |
| Exploration bonus (E, β) | ✅ Unchanged | Defaults `enable_exploration_bonus=True, exploration_beta=0.03` match Baseline A |

---

## 3. GAE convention

`eval_occupancy_gate.py` enforces production-correct bootstrap:
```python
if ep_truncated_arr[-1] and not ep_terminated_arr[-1]:
    bootstrap_value = float(critic(...))
else:
    bootstrap_value = 0.0
```

---

## 4. Bridge startup fix

`training/gmn_pettingzoo.py`:
- `_kill_existing_bridge()` kills stale bridge subprocess and orphaned node/tsx listeners before each launch.
- `_ensure_bridge_running()` retries with backoff (`3s`, `5s`, `8s`) and aborts after 3 failed attempts.
- `close()` cleans up subprocess and orphaned listeners.

This prevents stale-bridge mask/observation leakage between CTRL and INT runs.

---

## 5. Probe reproducibility

| Parameter | Value |
|-----------|-------|
| Eval script | `training/eval_occupancy_gate.py` |
| Checkpoint path | `training/models/mappo_academy_3_vs_1_with_keeper_seed42_best.pt` |
| Seeds | 500,000 + 9×episode_index (50 episodes) |
| Arms | CTRL (`academy_3_vs_1_with_keeper`), INT (`academy_3_vs_1_with_keeper_onball`) |
| Git commit | `939dc22df1d7a52e32b47b1a03e9537638551092` |

---

## 6. Outcome provenance

- Verdict: **Soft FAIL**
- Primary evidence: 0 events in both arms; action distribution dominated by `UP`/`UP_RIGHT`
- `left_possession_fraction`: CTRL 0.4692, INT 0.4630 (no meaningful difference because policy never acts on ball)
- `t=0` ownership confirmed: CTRL `[1. 0. 0.]` (no-one), INT `[0. 1. 0.]` (left)
- Critic/advantage separation: negligible (`mean_advantage_on_ball` ≈ `mean_advantage_off_ball` in both arms)

---

## 7. Artifacts

| Artifact | Path |
|----------|------|
| Experiment report | `training/results/EXPERIMENT_OCCUPANCY_F.md` |
| Provenance report | `training/results/EXPERIMENT_OCCUPANCY_F_PROVENANCE.md` |
| Eval script | `training/eval_occupancy_gate.py` |
| Aggregate CSV | `training/results/occupancy_gate_summary.csv` |
| CTRL per-tick log | `training/models/occupancy_gate_CTRL_mappo_academy_3_vs_1_with_keeper_seed42_best.json` |
| INT per-tick log | `training/models/occupancy_gate_INT_mappo_academy_3_vs_1_with_keeper_seed42_best.json` |

---

## 8. Commit message

```
docs: add Experiment F occupancy gate report and provenance

Intervention: μ-onball (academy_3_vs_1_with_keeper_onball)
Verdict: Soft FAIL — policy paralysis, 0 events in both CTRL and INT arms
Checkpoint: mappo_academy_3_vs_1_with_keeper_seed42_best.pt @ 100352 steps
```
