# Phase 16 — Rondo Architecture Preservation

## Verified No Regressions

### RondoScenarioHandler
- File: `src/engine/scenarios/RondoScenarioHandler.ts`
- Status: UNCHANGED — no modifications made
- All rondo-specific state remains in handler class
- `GameEngine` has zero rondo-specific conditionals

### getLastDefenderReward()
- Bridge server binary header preserves defender reward at offset 20
- `encodeMultiStepBinary` and `encodeBatchedStepBinary` unchanged for rondo
- Python client `step_batch` and `_recv_step_response` preserve defender reward extraction

### 22-byte Rondo Bridge Header
- Header size: 22 bytes for rondo, 18 bytes for non-rondo
- Offset 20 carries defenderReward float32
- Status: PRESERVED

### Per-team PettingZoo Rewards
- `gmn_pettingzoo.py` step() method assigns:
  - `rewards[agent] = defender_reward` for right_* agents in rondo
  - `rewards[agent] = shared_reward` for left_* agents in rondo
- Status: PRESERVED (bug fix only corrected indentation, did not change logic)

### CooperativeRewardShaper Right-team Exclusion
- `compute_shaped_rewards()` filters: `if not agent_id.startswith("right_")`
- Status: PRESERVED

### MAPPO Per-agent Rewards
- `buffer["per_agent_rewards"]` shape: (n_steps, n_agents)
- `compute_gae()` accepts and uses per-agent rewards
- `ppo_update()` uses per-agent advantages
- Status: VERIFIED WORKING

## Changes Made (Non-Rondo)

### ObservationEncoder.computeReward
- Added `passCompletedByTargetTeam` and `shotEventByTargetTeam` parameters
- These are used for ALL scenarios, not rondo-specific
- Rondo uses `computeRondoReward()` exclusively, so unaffected

### CooperativeRewardShaper defaults
- `penalty_ball_hogging`: -0.005 -> -0.02
- `max_unassisted_hold_ticks`: 30 -> 15
- These affect both rondo and non-rondo scenarios uniformly
- Rondo-specific tests pass with new defaults

## Conclusion

**NO REGRESSIONS** in Rondo architecture. All rondo-specific components remain intact.
