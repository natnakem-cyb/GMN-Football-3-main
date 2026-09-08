# Bug Fix Correction: step_batch() Terminal Reward Drop

## Original Claim (Incorrect)

The commit message claimed:
> Fix critical bug: gmn_pettingzoo.py step() reward/termination/truncation/info assignment outside for loop (only last agent populated)

## Actual Code in Parent Commit (912f2dc)

In the parent commit, `step()` already had the per-agent assignments correctly inside the `for` loop:

```python
for i, agent in enumerate(self.agents):
    offset = header_size + i * obs_bytes
    obs = np.frombuffer(data, dtype="<f4", count=OBSERVATION_DIM, offset=offset).copy()
    observations[agent] = obs
    # Rondo: assign team-specific rewards; non-rondo keeps the shared broadcast.
    if is_rondo:
        rewards[agent] = defender_reward if agent.startswith("right_") else shared_reward
    else:
        rewards[agent] = shared_reward
    terminations[agent] = shared_term
    truncations[agent] = shared_trunc
    infos[agent] = dict(shared_info)
    if step_events:
        infos[agent]["step_events"] = step_events
```

**The per-agent loop in step() was already correct.** The claim that "only last agent was populated" does not match the code.

## Real Bug: step_batch() Terminal Reward Drop

The actual bug was in `step_batch()`, which `step()` delegates to whenever `batch_size > 1`:

```python
# OLD (buggy) code in step_batch()
if shared_term or shared_trunc:
    env_state["agents"] = []          # cleared FIRST
if is_rondo:
    env_rewards = {
        agent: defender_reward if agent.startswith("right_") else shared_reward
        for agent in env_state["agents"]    # now-empty list
    }
```

On terminal steps in batched rondo mode, `env_state["agents"]` was cleared before `env_rewards` was constructed, so the rewards dict became `{}` — silently dropping the final (usually highest-signal) reward.

## Fix Applied

Snapshot the agent list before clearing:

```python
terminal_agents = list(env_state["agents"])   # snapshot before clearing
if shared_term or shared_trunc:
    env_state["agents"] = []
if is_rondo:
    env_rewards = {
        agent: defender_reward if agent.startswith("right_") else shared_reward
        for agent in terminal_agents
    }
```

The observations loop already ran before the clear, so observations were unaffected.

## Test Added

`training/tests/test_rondo_batch_terminal_reward.py` — 2 tests:
1. `test_terminal_step_returns_non_empty_rewards_batched_rondo` — steps batch_size=2 rondo env until termination, asserts rewards non-empty and contains all expected agent keys
2. `test_terminal_step_rewards_match_expected_values_batched_rondo` — verifies numeric rewards on terminal step

## Impact

- **step() single-agent path**: No bug, no change needed
- **step_batch() batched path (batch_size > 1)**: Fixed terminal reward drop for rondo
- **Non-rondo batched path**: Unaffected (uses `shared_reward` scalar, not per-agent dict)
