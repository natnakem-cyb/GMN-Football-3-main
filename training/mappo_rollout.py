"""
GMN-Football-3 — MAPPO Rollout Collection & Generalized Advantage Estimation (GAE)
Data pipeline for Multi-Agent PPO with Centralized Critic.

Collects trajectory buffers with local observations for shared actors and
joint observations for the centralized critic.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import torch

from mappo_networks import SharedActor, CentralizedCritic


def collect_rollout(
    env: Any,
    actor: SharedActor,
    critic: CentralizedCritic,
    num_steps: int = 256,
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
        "rewards": [],        # shape (num_steps,) — shared team reward
        "dones": [],          # shape (num_steps,) — genuine terminations only (not truncations)
        "terminated": [],     # shape (num_steps,) — genuine terminations only
        "truncated": [],      # shape (num_steps,) — time-limit truncations only
    }
    completed_episodes = []

    # Retrieve or initialize persistent rollout state on env
    if not hasattr(env, "_mappo_obs") or env._mappo_obs is None:
        obs_dict, _ = env.reset()
        env._mappo_obs = obs_dict
        env._mappo_ep_rew = 0.0
        env._mappo_ep_len = 0
    else:
        obs_dict = env._mappo_obs

    agent_order = list(env.agents if env.agents else env.possible_agents)
    num_agents = len(agent_order)
    first_obs = obs_dict[agent_order[0]]
    obs_dim = first_obs.shape[0] if hasattr(first_obs, "shape") else len(first_obs)

    for _ in range(num_steps):
        current_agents = list(env.agents if env.agents else env.possible_agents)
        local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
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
        obs_dict, rewards, terminations, truncations, infos = env.step(action_dict)

        terminated = any(terminations.values())
        truncated = any(truncations.values())
        done = terminated or truncated
        shared_reward = float(rewards[current_agents[0]])  # identical across agents

        env._mappo_ep_rew += shared_reward
        env._mappo_ep_len += 1

        buffer["local_obs"].append(local_obs)
        buffer["global_state"].append(global_state)
        buffer["actions"].append(actions.cpu().numpy())
        buffer["logprobs"].append(logprobs.cpu().numpy())
        buffer["values"].append(float(value.item()))
        buffer["rewards"].append(shared_reward)
        buffer["dones"].append(bool(terminated))
        buffer["terminated"].append(bool(terminated))
        buffer["truncated"].append(bool(truncated))

        if done:
            # Check if left team scored a goal in the terminal step
            goal_scored = 0
            for agent_info in infos.values():
                if agent_info.get("score", {}).get("left", 0) > 0:
                    goal_scored = 1
                    break

            completed_episodes.append({
                "reward": float(env._mappo_ep_rew),
                "length": int(env._mappo_ep_len),
                "goal": goal_scored,
            })
            env._mappo_ep_rew = 0.0
            env._mappo_ep_len = 0
            obs_dict, _ = env.reset()

    env._mappo_obs = obs_dict

    # Convert to structured numpy arrays
    res_buffer = {
        "local_obs": np.array(buffer["local_obs"], dtype=np.float32),
        "global_state": np.array(buffer["global_state"], dtype=np.float32),
        "actions": np.array(buffer["actions"], dtype=np.int64),
        "logprobs": np.array(buffer["logprobs"], dtype=np.float32),
        "values": np.array(buffer["values"], dtype=np.float32),
        "rewards": np.array(buffer["rewards"], dtype=np.float32),
        "dones": np.array(buffer["dones"], dtype=np.bool_),
        "terminated": np.array(buffer["terminated"], dtype=np.bool_),
        "truncated": np.array(buffer["truncated"], dtype=np.bool_),
        "next_local_obs": np.stack([obs_dict[a] for a in agent_order], axis=0).astype(np.float32),
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
        "dones": [],
        "terminated": [],
        "truncated": [],
    }
    completed_episodes: List[Dict[str, Any]] = []
    total_steps = 0

    for env in envs:
        if getattr(env, "_mappo_obs", None) is None:
            obs_dict, _ = env.reset()
            env._mappo_obs = obs_dict
            env._mappo_ep_rew = 0.0
            env._mappo_ep_len = 0

    # Warm-up on env 0 only to derive shapes consistently
    ref_env = envs[0]
    obs_dict = ref_env._mappo_obs
    agent_order = list(ref_env.agents if ref_env.agents else ref_env.possible_agents)
    num_agents = len(agent_order)
    first_obs = obs_dict[agent_order[0]]
    obs_dim = first_obs.shape[0] if hasattr(first_obs, "shape") else len(first_obs)

    for _ in range(num_steps):
        for env in envs:
            o_dict = env._mappo_obs
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

            shared_reward = float(rewards[current_agents[0]])
            terminated = bool(terminations[current_agents[0]])
            truncated = bool(truncations[current_agents[0]])

            buffers["local_obs"].append(local_obs)
            buffers["global_state"].append(global_state)
            buffers["actions"].append(actions)
            buffers["logprobs"].append(logprobs)
            buffers["values"].append(value)
            buffers["rewards"].append(shared_reward)
            buffers["dones"].append(terminated)
            buffers["terminated"].append(terminated)
            buffers["truncated"].append(truncated)

            env._mappo_ep_rew += shared_reward
            env._mappo_ep_len += 1
            total_steps += 1

            if terminated or truncated:
                info0 = infos[current_agents[0]]
                completed_episodes.append(
                    {
                        "reward": env._mappo_ep_rew,
                        "length": env._mappo_ep_len,
                        "goal": int(bool(info0.get("event", {}).get("type") == "goal")),
                        "env": envs.index(env),
                    }
                )
                env._mappo_obs = None
                obs_reset, _ = env.reset()
                env._mappo_obs = obs_reset
                env._mappo_ep_rew = 0.0
                env._mappo_ep_len = 0
            else:
                env._mappo_obs = obs_next

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
    last_o = envs[-1]._mappo_obs
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Generalized Advantage Estimation (GAE) and Returns backwards over the rollout.

    Args:
        rewards: shape (T,)
        values: shape (T,)
        dones: shape (T,) — genuine terminations only (not truncations)
        gamma: discount factor (default 0.99)
        lam: GAE lambda parameter (default 0.95)
        bootstrap_value: value estimate of state at T when truly terminated (default 0.0)
        next_local_obs: shape (num_agents, obs_dim) — joint observation after the last rollout step,
            used to compute bootstrap_value when the rollout was truncated mid-episode.
        critic: CentralizedCritic instance. If provided and next_local_obs is given and the
            final step was not a termination, bootstrap_value is computed as
            critic(next_local_obs). Otherwise the provided bootstrap_value is used.

    Returns:
        advantages: shape (T,)
        returns: shape (T,)
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
