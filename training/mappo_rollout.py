"""
GMN-Football-3 — MAPPO Rollout Collection & Generalized Advantage Estimation (GAE)
Data pipeline for Multi-Agent PPO with Centralized Critic.

Collects trajectory buffers with local observations for shared actors and
joint observations for the centralized critic.
"""

from typing import Dict, List, Optional, Tuple, Any
import json
import os
import numpy as np
import torch

from training.mappo_networks import SharedActor, CentralizedCritic
from training.curriculum_scheduler import is_scenario_success


def unwrap_obs(obs_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract raw observation arrays from the PettingZoo envelope format.

    The wrapper returns ``{agent_id: {"observation": np.ndarray,
    "action_mask": np.ndarray}}`` per agent. Rollout collectors need the bare
    arrays for ``np.stack`` / ``torch.tensor``. This helper accepts both the
    envelope format and already-unwrapped raw arrays (idempotent), so
    collectors stay robust across wrapper versions.

    G2 fix: previously each collector unwrapped inline at *some* sites but not
    at reset/bootstrap sites, so an episode ending mid-rollout stored envelope
    dicts into ``_mappo_obs`` and the next iteration crashed on
    ``dict.shape`` / ``np.stack(dicts)``.
    """
    unwrapped: Dict[str, Any] = {}
    for agent_id, obs_val in obs_dict.items():
        if isinstance(obs_val, dict) and "observation" in obs_val:
            unwrapped[agent_id] = obs_val["observation"]
        else:
            unwrapped[agent_id] = obs_val
    return unwrapped


def collect_rollout(
    env: Any,
    actor: SharedActor,
    critic: CentralizedCritic,
    num_steps: int = 256,
    terminal_jsonl_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Collects a multi-agent rollout from the PettingZoo environment.

    Returns a dict of numpy arrays with explicit shapes:
    - local_obs: shape (num_steps, num_agents, obs_dim) float32
    - global_state: shape (num_steps, obs_dim * num_agents) float32
    - actions: shape (num_steps, num_agents) int64
    - logprobs: shape (num_steps, num_agents) float32
    - values: shape (num_steps,) float32
    - rewards: shape (num_steps,) float32
    - dones: shape (num_steps,) bool — genuine terminations only (not truncations)
    - terminated: shape (num_steps,) bool — genuine terminations only
    - truncated: shape (num_steps,) bool — time-limit truncations only
    - next_local_obs: shape (num_agents, obs_dim) float32 — joint observation after the last rollout step
    - completed_episodes: list of dicts with {"reward": float, "length": int, "goal": int}
    """
    buffer: Dict[str, list] = {
        "local_obs": [],      # per-agent, shape (num_steps, num_agents, obs_dim)
        "global_state": [],   # shape (num_steps, obs_dim * num_agents)
        "actions": [],        # shape (num_steps, num_agents)
        "logprobs": [],       # shape (num_steps, num_agents)
        "values": [],         # shape (num_steps,) — one shared value per step
        "rewards": [],        # shape (num_steps,) — shared team reward (sum of per-agent rewards)
        "per_agent_rewards": [],  # shape (num_steps, num_agents) — individual agent rewards
        "dones": [],          # shape (num_steps,) — genuine terminations only (not truncations)
        "terminated": [],     # shape (num_steps,) — genuine terminations only
        "truncated": [],      # shape (num_steps,) — time-limit truncations only
    }
    completed_episodes = []

    # Retrieve or initialize persistent rollout state on env
    if not hasattr(env, "_mappo_obs") or env._mappo_obs is None:
        obs_dict, _ = env.reset()
        obs_dict = unwrap_obs(obs_dict)
        env._mappo_obs = obs_dict
        env._mappo_ep_rew = 0.0
        env._mappo_ep_len = 0
    else:
        obs_dict = unwrap_obs(env._mappo_obs)

    agent_order = list(env.agents if env.agents else env.possible_agents)
    num_agents = len(agent_order)
    first_obs = obs_dict[agent_order[0]]
    obs_dim = first_obs.shape[0] if hasattr(first_obs, "shape") else len(first_obs)

    for _ in range(num_steps):
        current_agents = list(env.agents if env.agents else env.possible_agents)
        local_obs_list = [obs_dict[a] for a in current_agents]
        local_obs = np.stack(local_obs_list, axis=0).astype(np.float32)
        global_state = local_obs.flatten().astype(np.float32)  # concat in agent_order

        # Verify concatenation integrity
        for idx, agent in enumerate(current_agents):
            start = idx * obs_dim
            end = (idx + 1) * obs_dim
            assert np.array_equal(global_state[start:end], local_obs[idx]), (
                f"Global state slice [{start}:{end}] does not match local_obs[{idx}] for agent {agent}"
            )

        with torch.no_grad():
            local_obs_t = torch.tensor(local_obs, dtype=torch.float32)
            dist = actor(local_obs_t)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            # Pass 3D tensor (1, num_agents, obs_dim) to CentralizedCritic (Deep Sets pooling)
            value = critic(local_obs_t.unsqueeze(0))

        action_dict = {a: int(actions[i].item()) for i, a in enumerate(current_agents)}

        # Step 2/3: capture reward-before-terminal strictly before the final tick
        reward_before_terminal = float(env._mappo_ep_rew)

        obs_dict, rewards, terminations, truncations, infos = env.step(action_dict)
        # G2 fix: step returns the envelope format ({agent: {"observation": ...,
        # "action_mask": ...}}); unwrap immediately so the next loop iteration's
        # np.stack sees raw arrays instead of dicts.
        obs_dict = unwrap_obs(obs_dict)

        terminated = any(terminations.values())
        truncated = any(truncations.values())
        done = terminated or truncated
        # Collect per-agent rewards for asymmetric scenarios (e.g. rondo)
        per_agent_rewards = np.array([rewards[a] for a in current_agents], dtype=np.float32)
        shared_reward = float(per_agent_rewards.mean())

        # Step 3: reward-chain trace on terminal tick
        terminal_frame_reward = getattr(env, "_last_frame_reward", float("nan"))
        terminal_shared_reward = getattr(env, "_last_shared_reward", float("nan"))
        terminal_event_code = getattr(env, "_last_frame_event_code", -1)
        terminal_score = getattr(env, "_last_frame_score", {"left": -1, "right": -1})
        if done:
            print(
                f"[REWARD-CHAIN] terminal_tick | "
                f"binary_frame={terminal_frame_reward:.6f} | "
                f"env_step_shared={terminal_shared_reward:.6f} | "
                f"collector_shared={shared_reward:.6f} | "
                f"reward_before_terminal={reward_before_terminal:.6f} | "
                f"event_code={terminal_event_code} | score={terminal_score}"
            )

        env._mappo_ep_rew += shared_reward
        env._mappo_ep_len += 1

        buffer["local_obs"].append(local_obs)
        buffer["global_state"].append(global_state)
        buffer["actions"].append(actions.cpu().numpy())
        buffer["logprobs"].append(logprobs.cpu().numpy())
        buffer["values"].append(float(value.item()))
        buffer["rewards"].append(shared_reward)
        buffer["per_agent_rewards"].append(per_agent_rewards)
        buffer["dones"].append(bool(terminated))
        buffer["terminated"].append(bool(terminated))
        buffer["truncated"].append(bool(truncated))

        if done:
            # Step 2: capture comprehensive terminal transition record
            episode_reward = float(env._mappo_ep_rew)
            terminal_record = {
                "terminal_frame_reward": terminal_frame_reward,
                "terminal_shared_reward": terminal_shared_reward,
                "reward_before_terminal": reward_before_terminal,
                "episode_reward": episode_reward,
                "terminal_event_code": terminal_event_code,
                "terminal_score": terminal_score,
                "episode_length": int(env._mappo_ep_len),
                "scenario": getattr(env, "scenario", "unknown"),
                "env_idx": 0,
            }
            if terminal_jsonl_path:
                os.makedirs(os.path.dirname(terminal_jsonl_path) or ".", exist_ok=True)
                with open(terminal_jsonl_path, "a") as f:
                    f.write(json.dumps(terminal_record) + "\n")

            # Check if left team scored a goal in the terminal step
            goal_scored = 0
            terminal_info = {}
            for agent_id, agent_info in infos.items():
                if agent_info.get("score", {}).get("left", 0) > 0:
                    goal_scored = 1
                terminal_info = agent_info  # any agent's info contains the shared score/event
                break

            success = is_scenario_success(
                getattr(env, "scenario", "academy_empty_goal"),
                terminal_info,
            )
            # Sanity-bound check: flag implausible episode rewards immediately.
            # For academy_3_vs_1_with_keeper (1800 ticks), per-tick reward is bounded
            # roughly [-1.0, +2.17], so cumulative episode reward should fall in
            # [-1800, +3600]. Values outside this range indicate a reward-accumulation
            # bug (e.g., double-counting, missing reset).
            _episode_reward = float(env._mappo_ep_rew)
            if not (-1800.0 <= _episode_reward <= 3600.0):
                print(
                    f"[WARN] Episode reward {_episode_reward:.2f} outside expected range "
                    f"[-1800, 3600] — possible reward-accumulation bug"
                )

            completed_episodes.append({
                "reward": _episode_reward,
                "length": int(env._mappo_ep_len),
                "goal": goal_scored,
                "success": success,
            })
            env._mappo_ep_rew = 0.0
            env._mappo_ep_len = 0
            obs_dict, _ = env.reset()
            obs_dict = unwrap_obs(obs_dict)

    env._mappo_obs = obs_dict

    # Convert to structured numpy arrays
    res_buffer = {
        "local_obs": np.array(buffer["local_obs"], dtype=np.float32),
        "global_state": np.array(buffer["global_state"], dtype=np.float32),
        "actions": np.array(buffer["actions"], dtype=np.int64),
        "logprobs": np.array(buffer["logprobs"], dtype=np.float32),
        "values": np.array(buffer["values"], dtype=np.float32),
        "rewards": np.array(buffer["rewards"], dtype=np.float32),
        "per_agent_rewards": np.array(buffer["per_agent_rewards"], dtype=np.float32),
        "dones": np.array(buffer["dones"], dtype=np.bool_),
        "terminated": np.array(buffer["terminated"], dtype=np.bool_),
        "truncated": np.array(buffer["truncated"], dtype=np.bool_),
        "next_local_obs": np.stack([unwrap_obs(obs_dict)[a] for a in agent_order], axis=0).astype(np.float32),
        "agent_order": agent_order,
        "completed_episodes": completed_episodes,
    }

    # Explicit shape assertions
    assert res_buffer["local_obs"].shape == (num_steps, num_agents, obs_dim), (
        f"local_obs shape mismatch: expected {(num_steps, num_agents, obs_dim)}, got {res_buffer['local_obs'].shape}"
    )
    assert res_buffer["global_state"].shape == (num_steps, obs_dim * num_agents), (
        f"global_state shape mismatch: expected {(num_steps, obs_dim * num_agents)}, got {res_buffer['global_state'].shape}"
    )
    assert res_buffer["actions"].shape == (num_steps, num_agents), (
        f"actions shape mismatch: expected {(num_steps, num_agents)}, got {res_buffer['actions'].shape}"
    )
    assert res_buffer["logprobs"].shape == (num_steps, num_agents), (
        f"logprobs shape mismatch: expected {(num_steps, num_agents)}, got {res_buffer['logprobs'].shape}"
    )
    assert res_buffer["values"].shape == (num_steps,), (
        f"values shape mismatch: expected {(num_steps,)}, got {res_buffer['values'].shape}"
    )
    assert res_buffer["rewards"].shape == (num_steps,), (
        f"rewards shape mismatch: expected {(num_steps,)}, got {res_buffer['rewards'].shape}"
    )
    assert res_buffer["dones"].shape == (num_steps,), (
        f"dones shape mismatch: expected {(num_steps,)}, got {res_buffer['dones'].shape}"
    )
    assert res_buffer["terminated"].shape == (num_steps,), (
        f"terminated shape mismatch: expected {(num_steps,)}, got {res_buffer['terminated'].shape}"
    )
    assert res_buffer["truncated"].shape == (num_steps,), (
        f"truncated shape mismatch: expected {(num_steps,)}, got {res_buffer['truncated'].shape}"
    )
    assert res_buffer["next_local_obs"].shape == (num_agents, obs_dim), (
        f"next_local_obs shape mismatch: expected {(num_agents, obs_dim)}, got {res_buffer['next_local_obs'].shape}"
    )

    return res_buffer


def collect_rollout_parallel(
    envs: List[Any],
    actor: SharedActor,
    critic: CentralizedCritic,
    num_steps: int = 256,
    terminal_jsonl_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parallel rollout collection across N independent environments.

    Each iteration steps every environment once (all N bridges step in an
    interleaved sequence, amortizing round-trips), and every environment's
    transition is appended to a single flat buffer. The resulting buffer has
    T = num_steps * len(envs) rows with the same layout that ``ppo_update``
    expects, so the update math is unchanged — it simply sees N times more
    on-policy data per update.

    Per-environment rollout state is stored on each env (``_mappo_obs`` etc.),
    mirroring the single-env contract of ``collect_rollout``.
    """
    buffers: Dict[str, list] = {
        "local_obs": [],
        "global_state": [],
        "actions": [],
        "logprobs": [],
        "values": [],
        "rewards": [],
        "per_agent_rewards": [],
        "dones": [],
        "terminated": [],
        "truncated": [],
    }
    completed_episodes: List[Dict[str, Any]] = []
    total_steps = 0

    for env in envs:
        if getattr(env, "_mappo_obs", None) is None:
            obs_dict, _ = env.reset()
            env._mappo_obs = unwrap_obs(obs_dict)
            env._mappo_ep_rew = 0.0
            env._mappo_ep_len = 0

    # Warm-up on env 0 only to derive shapes consistently
    ref_env = envs[0]
    obs_dict = unwrap_obs(ref_env._mappo_obs)
    agent_order = list(ref_env.agents if ref_env.agents else ref_env.possible_agents)
    num_agents = len(agent_order)
    first_obs = obs_dict[agent_order[0]]
    obs_dim = first_obs.shape[0] if hasattr(first_obs, "shape") else len(first_obs)

    for _ in range(num_steps):
        for env in envs:
            o_dict = unwrap_obs(env._mappo_obs)
            current_agents = list(env.agents if env.agents else env.possible_agents)
            if len(current_agents) != num_agents:
                # Defensive: scenario agent count is fixed per env.
                raise RuntimeError(
                    f"Parallel env agent-count mismatch: expected {num_agents}, got {len(current_agents)}"
                )
            local_obs = np.stack([o_dict[a] for a in current_agents], axis=0).astype(np.float32)
            global_state = local_obs.flatten().astype(np.float32)

            with torch.no_grad():
                dist = actor(torch.tensor(local_obs, dtype=torch.float32))
                actions_t = dist.sample()
                logprobs_t = dist.log_prob(actions_t)
                value_t = critic(torch.tensor(global_state, dtype=torch.float32).unsqueeze(0))

            actions = actions_t.numpy().astype(np.int64)
            logprobs = logprobs_t.numpy().astype(np.float32)
            value = float(value_t.item())

            obs_next, rewards, terminations, truncations, infos = env.step(
                {a: int(actions[i]) for i, a in enumerate(current_agents)}
            )

            per_agent_rewards = np.array([rewards[a] for a in current_agents], dtype=np.float32)
            shared_reward = float(per_agent_rewards.mean())
            terminated = bool(terminations[current_agents[0]])
            truncated = bool(truncations[current_agents[0]])

            buffers["local_obs"].append(local_obs)
            buffers["global_state"].append(global_state)
            buffers["actions"].append(actions)
            buffers["logprobs"].append(logprobs)
            buffers["values"].append(value)
            buffers["rewards"].append(shared_reward)
            buffers["per_agent_rewards"].append(per_agent_rewards)
            buffers["dones"].append(terminated)
            buffers["terminated"].append(terminated)
            buffers["truncated"].append(truncated)

            # Step 2/3: capture reward-before-terminal before incrementing
            reward_before_terminal = float(env._mappo_ep_rew)
            terminal_frame_reward = getattr(env, "_last_frame_reward", float("nan"))
            terminal_shared_reward = getattr(env, "_last_shared_reward", float("nan"))
            terminal_event_code = getattr(env, "_last_frame_event_code", -1)
            terminal_score = getattr(env, "_last_frame_score", {"left": -1, "right": -1})

            env._mappo_ep_rew += shared_reward
            env._mappo_ep_len += 1
            total_steps += 1

            if terminated or truncated:
                # Step 3: reward-chain trace on terminal tick
                print(
                    f"[REWARD-CHAIN] terminal_tick env={envs.index(env)} | "
                    f"binary_frame={terminal_frame_reward:.6f} | "
                    f"env_step_shared={terminal_shared_reward:.6f} | "
                    f"collector_shared={shared_reward:.6f} | "
                    f"reward_before_terminal={reward_before_terminal:.6f} | "
                    f"event_code={terminal_event_code} | score={terminal_score}"
                )

                info0 = infos[current_agents[0]]
                # Audit P0 fix: prefer the score carried in the step's info dict
                # (the real env contract, replicated per agent). The
                # _last_frame_* attributes are single-env diagnostics and are
                # not guaranteed to exist on every env implementation stepped
                # by this collector — an absent attribute used to yield
                # {"left": -1, "right": -1} and silently falsify match/rondo
                # success.
                _score_p = info0.get("score") if isinstance(info0, dict) else None
                if not isinstance(_score_p, dict):
                    _score_p = terminal_score if isinstance(terminal_score, dict) else {}
                episode_reward = float(env._mappo_ep_rew)
                terminal_record = {
                    "terminal_frame_reward": terminal_frame_reward,
                    "terminal_shared_reward": terminal_shared_reward,
                    "reward_before_terminal": reward_before_terminal,
                    "episode_reward": episode_reward,
                    "terminal_event_code": terminal_event_code,
                    "terminal_score": terminal_score,
                    "episode_length": int(env._mappo_ep_len),
                    "scenario": getattr(env, "scenario", "unknown"),
                    "env_idx": envs.index(env),
                }
                if terminal_jsonl_path:
                    os.makedirs(os.path.dirname(terminal_jsonl_path) or ".", exist_ok=True)
                    with open(terminal_jsonl_path, "a") as f:
                        f.write(json.dumps(terminal_record) + "\n")

                # Audit P0 fix: emit curriculum success signal for this episode.
                # Uses is_scenario_success so match/rondo stages use the correct
                # rule (left > right, or score.right==0 + scenario_complete)
                # instead of relying on goal==True as a proxy for success.
                # NOTE: preserve the raw event type here. Downstream
                # is_scenario_success distinguishes 'goal' from
                # 'scenario_complete', so mapping everything non-goal to ''
                # would break the rondo success rule.
                _scenario_id = getattr(env, "scenario", "academy_empty_goal")
                _raw_event_type = info0.get("event", {}).get("type") or ""
                _terminal_info = {
                    "score": _score_p,
                    "event": {"type": _raw_event_type},
                }
                _success = is_scenario_success(_scenario_id, _terminal_info)

                completed_episodes.append(
                    {
                        "reward": episode_reward,
                        "length": int(env._mappo_ep_len),
                        "goal": int(bool(info0.get("event", {}).get("type") == "goal")),
                        "success": _success,
                        "env": envs.index(env),
                    }
                )
                env._mappo_obs = None
                obs_reset, _ = env.reset()
                env._mappo_obs = unwrap_obs(obs_reset)
                env._mappo_ep_rew = 0.0
                env._mappo_ep_len = 0
            else:
                env._mappo_obs = unwrap_obs(obs_next)

    # Concatenate into the same layout produced by collect_rollout:
    # local_obs (T, num_agents, obs_dim), global_state (T, num_agents*obs_dim),
    # actions/logprobs (T, num_agents), scalars (T,).
    res_buffer: Dict[str, Any] = {}
    res_buffer["local_obs"] = np.stack(buffers["local_obs"], axis=0).astype(np.float32)
    res_buffer["global_state"] = np.stack(buffers["global_state"], axis=0).astype(np.float32)
    res_buffer["actions"] = np.stack(buffers["actions"], axis=0).astype(np.int64)
    res_buffer["logprobs"] = np.stack(buffers["logprobs"], axis=0).astype(np.float32)
    res_buffer["values"] = np.array(buffers["values"], dtype=np.float32)
    res_buffer["rewards"] = np.array(buffers["rewards"], dtype=np.float32)
    res_buffer["dones"] = np.array(buffers["dones"], dtype=bool)
    res_buffer["terminated"] = np.array(buffers["terminated"], dtype=bool)
    res_buffer["truncated"] = np.array(buffers["truncated"], dtype=bool)

    # next_local_obs: latest post-step observation (bootstrap), shape (num_agents, obs_dim)
    last_o = unwrap_obs(envs[-1]._mappo_obs)
    last_agents = list(envs[-1].agents if envs[-1].agents else envs[-1].possible_agents)
    res_buffer["next_local_obs"] = np.stack(
        [last_o[a] for a in last_agents], axis=0
    ).astype(np.float32)
    res_buffer["completed_episodes"] = completed_episodes
    res_buffer["total_steps"] = total_steps
    return res_buffer


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
    bootstrap_value: float = 0.0,
    next_local_obs: np.ndarray = None,
    critic = None,
    per_agent_rewards: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Generalized Advantage Estimation (GAE) and Returns backwards over the rollout.

    Args:
        rewards: shape (T,) — shared team reward used for value baseline
        values: shape (T,) — shared value estimates from centralized critic
        dones: shape (T,) — genuine terminations only (not truncations)
        gamma: discount factor (default 0.99)
        lam: GAE lambda parameter (default 0.95)
        bootstrap_value: value estimate of state at T when truly terminated (default 0.0)
        next_local_obs: shape (num_agents, obs_dim) — joint observation after the last rollout step,
            used to compute bootstrap_value when the rollout was truncated mid-episode.
        critic: CentralizedCritic instance. If provided and next_local_obs is given and the
            final step was not a termination, bootstrap_value is computed as
            critic(next_local_obs). Otherwise the provided bootstrap_value is used.
        per_agent_rewards: shape (T, num_agents) or None. If provided, returns per-agent
            advantages/returns of shape (T, num_agents). Otherwise falls back to shared
            advantages/returns of shape (T,).

    Returns:
        advantages: shape (T,) or (T, num_agents)
        returns: shape (T,) or (T, num_agents)
    """
    # If the rollout was truncated mid-episode and we have a critic + next_local_obs,
    # compute the real bootstrap value instead of defaulting to 0.0.
    effective_bootstrap = bootstrap_value
    if (
        critic is not None
        and next_local_obs is not None
        and len(rewards) > 0
        and not dones[-1]
    ):
        with torch.no_grad():
            # next_local_obs shape: (num_agents, obs_dim) -> (1, num_agents, obs_dim)
            assert next_local_obs.ndim == 2, (
                f"Expected next_local_obs shape (num_agents, obs_dim), got {next_local_obs.shape}"
            )
            obs_tensor = torch.tensor(next_local_obs, dtype=torch.float32).unsqueeze(0)
            effective_bootstrap = float(critic(obs_tensor).item())

    if per_agent_rewards is not None:
        # Per-agent GAE: each agent has its own reward signal but shares the same
        # value baseline V(s) from the centralized critic.
        T, num_agents = per_agent_rewards.shape
        advantages = np.zeros_like(per_agent_rewards, dtype=np.float32)
        for a in range(num_agents):
            last_gae = 0.0
            for t in reversed(range(T)):
                next_value = effective_bootstrap if t == T - 1 else values[t + 1]
                next_nonterminal = 0.0 if dones[t] else 1.0
                delta = per_agent_rewards[t, a] + gamma * next_value * next_nonterminal - values[t]
                last_gae = delta + gamma * lam * next_nonterminal * last_gae
                advantages[t, a] = last_gae
        returns = advantages + np.repeat(values[:, None], num_agents, axis=1)
        return advantages, returns

    # Shared GAE (original behavior)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = effective_bootstrap if t == len(rewards) - 1 else values[t + 1]
        next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def collect_rollout_batched(
    env: Any,
    actor: SharedActor,
    critic: CentralizedCritic,
    num_steps: int = 256,
    batch_size: int = 2,
    terminal_jsonl_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Vectorized rollout collection using the bridge's batched step protocol.

    Unlike ``collect_rollout_parallel``, this sends one stacked binary frame per
    step instead of N sequential round-trips. The returned buffer layout is
    identical so ``ppo_update`` is unchanged.
    """
    buffers: Dict[str, list] = {
        "local_obs": [],
        "global_state": [],
        "actions": [],
        "logprobs": [],
        "values": [],
        "rewards": [],
        "per_agent_rewards": [],
        "dones": [],
        "terminated": [],
        "truncated": [],
    }
    completed_episodes: List[Dict[str, Any]] = []
    total_steps = 0

    # Initialize all sub-environments via the batch reset path.
    batch_init = env.reset_batch([42 + i for i in range(batch_size)])
    if len(batch_init) != batch_size:
        raise RuntimeError(f"[collect_rollout_batched] Expected {batch_size} envs, got {len(batch_init)}")

    # Per-env rollout state, indexed by env_idx.
    env_states: List[Dict[str, Any]] = []
    ref_obs_dict, _ = batch_init[0]
    ref_obs_dict = unwrap_obs(ref_obs_dict)
    agent_order = list(ref_obs_dict.keys())
    num_agents = len(agent_order)
    obs_dim = ref_obs_dict[agent_order[0]].shape[0]

    for env_idx in range(batch_size):
        obs_dict, info = batch_init[env_idx]
        obs_dict = unwrap_obs(obs_dict)
        env_states.append({
            "obs_dict": obs_dict,
            "agents": list(obs_dict.keys()),
            "agent_order": agent_order,
            "obs_dim": obs_dim,
            "ep_rew": 0.0,
            "ep_len": 0,
            "info": info,
        })

    for _ in range(num_steps):
        # Build per-env action dicts from a single shared policy forward.
        # Every sub-env acts every tick: with the immediate per-env reset in
        # the terminal branch below, no env should ever be dead. If one
        # somehow is, revive it now rather than idling it through the step.
        action_sets: List[Dict[str, int]] = []
        local_obs_stack = []
        for env_idx in range(batch_size):
            state = env_states[env_idx]
            current_agents = state["agents"]
            if not current_agents:
                # Defensive: should be unreachable with reset-on-terminal.
                fresh_obs, fresh_info = env.reset_one(env_idx, seed=1000 + env_idx)
                state["obs_dict"] = unwrap_obs(fresh_obs)
                state["agents"] = list(state["obs_dict"].keys())
                state["info"] = fresh_info
                current_agents = state["agents"]
            local_obs = np.stack([state["obs_dict"][a] for a in current_agents], axis=0).astype(np.float32)
            local_obs_stack.append(local_obs)
            with torch.no_grad():
                dist = actor(torch.tensor(local_obs, dtype=torch.float32))
                actions_t = dist.sample()
                logprobs_t = dist.log_prob(actions_t)
                value_t = critic(torch.tensor(local_obs.flatten(), dtype=torch.float32).unsqueeze(0))
            action_dict = {a: int(actions_t[i].item()) for i, a in enumerate(current_agents)}
            action_sets.append(action_dict)
            state["_last_actions"] = actions_t.cpu().numpy().astype(np.int64)
            state["_last_logprobs"] = logprobs_t.cpu().numpy()
            state["_last_value"] = float(value_t.item())

        # One batched step across all sub-environments.
        batch_results = env.step_batch(action_sets)

        for env_idx in range(batch_size):
            state = env_states[env_idx]
            # Every sub-env is live (reset-on-terminal below), so every env
            # contributes exactly one transition per tick.
            observations, reward, terminated, truncated, info = batch_results[env_idx]
            current_agents = state["agents"]
            if isinstance(reward, dict):
                # Rondo-style per-agent rewards
                per_agent_rewards = np.array([reward[a] for a in current_agents], dtype=np.float32)
                shared_reward = float(per_agent_rewards.mean())
            else:
                per_agent_rewards = np.full(len(current_agents), float(reward), dtype=np.float32)
                shared_reward = float(reward)
            shared_term = (
                any(bool(v) for v in terminated.values())
                if isinstance(terminated, dict)
                else bool(terminated)
            )
            shared_trunc = (
                any(bool(v) for v in truncated.values())
                if isinstance(truncated, dict)
                else bool(truncated)
            )
            local_obs = np.stack([state["obs_dict"][a] for a in current_agents], axis=0).astype(np.float32)
            global_state = local_obs.flatten().astype(np.float32)

            buffers["local_obs"].append(local_obs)
            buffers["global_state"].append(global_state)
            buffers["actions"].append(state["_last_actions"])
            buffers["logprobs"].append(state["_last_logprobs"])
            buffers["values"].append(state["_last_value"])
            buffers["rewards"].append(shared_reward)
            buffers["per_agent_rewards"].append(per_agent_rewards)
            buffers["dones"].append(shared_term)
            buffers["terminated"].append(shared_term)
            buffers["truncated"].append(shared_trunc)

            state["obs_dict"] = unwrap_obs(observations)
            state["ep_rew"] += shared_reward
            state["ep_len"] += 1
            total_steps += 1

            # Step 2/3: capture reward-before-terminal before incrementing
            reward_before_terminal = float(state["ep_rew"] - shared_reward)
            terminal_frame_reward = float(reward) if not isinstance(reward, dict) else float(list(reward.values())[0])
            terminal_shared_reward = shared_reward

            if shared_term or shared_trunc:
                goal_scored = 0
                # Extract terminal score/event from the shared top-level info
                # dict FIRST (real batched contract: gmn_pettingzoo.step_batch
                # returns ONE shared info per sub-env), then derive the goal
                # metric. Keep a per-agent scan as a fallback for wrappers that
                # replicate shared info per agent.
                event_code = info.get("eventCode", -1) if isinstance(info, dict) else -1
                score = info.get("score", {"left": -1, "right": -1}) if isinstance(info, dict) else {"left": -1, "right": -1}
                if isinstance(score, dict) and score.get("left", 0) > 0:
                    goal_scored = 1
                else:
                    for agent_info in (info or {}).values():
                        if isinstance(agent_info, dict) and agent_info.get("score", {}).get("left", 0) > 0:
                            goal_scored = 1
                            break

                # Step 3: reward-chain trace on terminal tick
                print(
                    f"[REWARD-CHAIN] terminal_tick env={env_idx} | "
                    f"binary_frame={terminal_frame_reward:.6f} | "
                    f"env_step_shared={terminal_shared_reward:.6f} | "
                    f"collector_shared={shared_reward:.6f} | "
                    f"reward_before_terminal={reward_before_terminal:.6f} | "
                    f"event_code={event_code} | score={score}"
                )

                episode_reward = float(state["ep_rew"])
                terminal_record = {
                    "terminal_frame_reward": terminal_frame_reward,
                    "terminal_shared_reward": terminal_shared_reward,
                    "reward_before_terminal": reward_before_terminal,
                    "episode_reward": episode_reward,
                    "terminal_event_code": event_code,
                    "terminal_score": score,
                    "episode_length": int(state["ep_len"]),
                    "scenario": getattr(env, "scenario", "unknown"),
                    "env_idx": env_idx,
                }
                if terminal_jsonl_path:
                    os.makedirs(os.path.dirname(terminal_jsonl_path) or ".", exist_ok=True)
                    with open(terminal_jsonl_path, "a") as f:
                        f.write(json.dumps(terminal_record) + "\n")

                # Audit P0 fix: emit curriculum success signal for this episode.
                # Preserve the raw per-agent event type (goal vs
                # scenario_complete) so is_scenario_success can apply the
                # correct rule per stage.
                _scenario_id_b = getattr(env, "scenario", "academy_empty_goal")
                # The real batched env (gmn_pettingzoo.step_batch) returns ONE
                # shared top-level info dict per sub-env: {"score": {...},
                # "event": {"type": ...}, "eventCode": ..., ...}. Read the
                # event type from the top level first; keep a per-agent scan
                # as a fallback for wrappers that replicate shared info per
                # agent (same shape as the single-env step path).
                _raw_event_type_b = ""
                if isinstance(info, dict):
                    _ev_b = info.get("event")
                    if isinstance(_ev_b, dict):
                        _raw_event_type_b = _ev_b.get("type") or ""
                    if not _raw_event_type_b:
                        for agent_info in info.values():
                            if isinstance(agent_info, dict):
                                _raw_event_type_b = (
                                    agent_info.get("event", {}).get("type") or ""
                                )
                                if _raw_event_type_b:
                                    break

                _terminal_info_b = {
                    "score": score if isinstance(score, dict) else {},
                    "event": {"type": _raw_event_type_b},
                }
                _success_b = is_scenario_success(_scenario_id_b, _terminal_info_b)

                completed_episodes.append({
                    "reward": episode_reward,
                    "length": int(state["ep_len"]),
                    "goal": goal_scored,
                    "success": _success_b,
                    "env": env_idx,
                })
                state["ep_rew"] = 0.0
                state["ep_len"] = 0
                # G6 fix: immediate per-env reset via the reset_one wire
                # message, mirroring collect_rollout_parallel's reset-on-
                # terminal. The sub-env re-enters the rollout on the next tick
                # instead of idling on dead state until the whole batch dies.
                fresh_obs, fresh_info = env.reset_one(env_idx, seed=1000 + env_idx)
                state["obs_dict"] = unwrap_obs(fresh_obs)
                state["agents"] = list(state["obs_dict"].keys())
                state["info"] = fresh_info

    # Agent order from the (always-live) reference env state.
    agent_order = env_states[0]["agent_order"]

    res_buffer: Dict[str, Any] = {}
    res_buffer["local_obs"] = np.stack(buffers["local_obs"], axis=0).astype(np.float32)
    res_buffer["global_state"] = np.stack(buffers["global_state"], axis=0).astype(np.float32)
    res_buffer["actions"] = np.stack(buffers["actions"], axis=0).astype(np.int64)
    res_buffer["logprobs"] = np.stack(buffers["logprobs"], axis=0).astype(np.float32)
    res_buffer["values"] = np.array(buffers["values"], dtype=np.float32)
    res_buffer["rewards"] = np.array(buffers["rewards"], dtype=np.float32)
    res_buffer["per_agent_rewards"] = np.stack(buffers["per_agent_rewards"], axis=0).astype(np.float32)
    res_buffer["dones"] = np.array(buffers["dones"], dtype=bool)
    res_buffer["terminated"] = np.array(buffers["terminated"], dtype=bool)
    res_buffer["truncated"] = np.array(buffers["truncated"], dtype=bool)
    res_buffer["next_local_obs"] = np.stack([env_states[0]["obs_dict"][a] for a in agent_order], axis=0).astype(np.float32)
    res_buffer["completed_episodes"] = completed_episodes
    res_buffer["total_steps"] = total_steps

    # Shape assertions: flat layout identical to collect_rollout_parallel —
    # every sub-env contributes exactly one transition per tick, so
    # T = num_steps * batch_size rows of (num_agents, ...).
    T = num_steps * batch_size
    assert res_buffer["local_obs"].shape == (T, num_agents, obs_dim), (
        f"local_obs shape mismatch: got {res_buffer['local_obs'].shape}"
    )
    assert res_buffer["global_state"].shape == (T, obs_dim * num_agents), (
        f"global_state shape mismatch: got {res_buffer['global_state'].shape}"
    )
    assert res_buffer["actions"].shape == (T, num_agents), (
        f"actions shape mismatch: got {res_buffer['actions'].shape}"
    )
    assert res_buffer["logprobs"].shape == (T, num_agents), (
        f"logprobs shape mismatch: got {res_buffer['logprobs'].shape}"
    )
    assert res_buffer["values"].shape == (T,), (
        f"values shape mismatch: got {res_buffer['values'].shape}"
    )
    assert res_buffer["rewards"].shape == (T,), (
        f"rewards shape mismatch: got {res_buffer['rewards'].shape}"
    )
    assert res_buffer["dones"].shape == (T,), (
        f"dones shape mismatch: got {res_buffer['dones'].shape}"
    )
    assert res_buffer["next_local_obs"].shape == (num_agents, obs_dim), (
        f"next_local_obs shape mismatch: got {res_buffer['next_local_obs'].shape}"
    )
    return res_buffer
