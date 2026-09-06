"""
GAE bootstrap regression test.

Constructs small synthetic rollouts where the expected advantages and returns
are known by hand, and asserts compute_gae produces the correct values for
both the terminated and truncated final-step cases.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from gmn_pettingzoo import OBSERVATION_DIM
from mappo_rollout import compute_gae
from mappo_networks import CentralizedCritic


def compute_expected_gae(rewards, values, dones, gamma, lam, bootstrap_value):
    """Hand-rolled GAE for verification."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = bootstrap_value if t == T - 1 else values[t + 1]
        next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def test_terminated_rollout():
    """
    3-timestep rollout terminated at the final step.
    bootstrap_value = 0.0 (correct for genuine termination).
    """
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    dones = np.array([False, False, True], dtype=np.bool_)
    gamma = 0.99
    lam = 0.95
    bootstrap_value = 0.0

    expected_adv, expected_ret = compute_expected_gae(
        rewards, values, dones, gamma, lam, bootstrap_value
    )

    # compute_gae should use bootstrap_value=0.0 at the final step because dones[-1]=True
    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=gamma,
        lam=lam,
        bootstrap_value=bootstrap_value,
        next_local_obs=np.ones((3, OBSERVATION_DIM), dtype=np.float32),
        critic=CentralizedCritic(obs_dim=OBSERVATION_DIM, hidden=8, mode="pool"),
    )

    np.testing.assert_allclose(advantages, expected_adv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(returns, expected_ret, rtol=1e-5, atol=1e-5)
    print("   [OK] Terminated rollout: advantages and returns match hand-computed values.")


def test_truncated_rollout():
    """
    3-timestep rollout truncated at the final step (episode ongoing).
    bootstrap_value should be computed from critic(next_local_obs), not defaulted to 0.0.
    """
    num_agents = 3
    obs_dim = OBSERVATION_DIM
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    dones = np.array([False, False, False], dtype=np.bool_)
    gamma = 0.99
    lam = 0.95

    # next_local_obs with a known value through the critic
    next_local_obs = np.ones((num_agents, obs_dim), dtype=np.float32) * 0.7
    critic = CentralizedCritic(obs_dim=obs_dim, hidden=8, mode="pool")

    # Compute what the critic would output for next_local_obs
    with torch.no_grad():
        obs_tensor = torch.tensor(next_local_obs, dtype=torch.float32).unsqueeze(0)
        expected_bootstrap = float(critic(obs_tensor).item())

    # Hand-compute expected advantages/returns using the real bootstrap value
    expected_adv, expected_ret = compute_expected_gae(
        rewards, values, dones, gamma, lam, expected_bootstrap
    )

    # compute_gae should automatically use critic(next_local_obs) because dones[-1]=False
    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=gamma,
        lam=lam,
        bootstrap_value=0.0,  # intentionally wrong; should be overridden
        next_local_obs=next_local_obs,
        critic=critic,
    )

    np.testing.assert_allclose(advantages, expected_adv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(returns, expected_ret, rtol=1e-5, atol=1e-5)
    print("   [OK] Truncated rollout: advantages and returns match critic-bootstrapped values.")


def test_no_bootstrap_when_critic_missing():
    """
    If critic is not provided, compute_gae should fall back to the provided
    bootstrap_value (preserving backward compatibility).
    """
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.5], dtype=np.float32)
    dones = np.array([False, False], dtype=np.bool_)

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=0.99,
        lam=0.95,
        bootstrap_value=0.0,
        next_local_obs=np.ones((3, OBSERVATION_DIM), dtype=np.float32),
        critic=None,
    )

    # With bootstrap_value=0.0 and no critic, this should match the old behavior
    expected_adv, expected_ret = compute_expected_gae(
        rewards, values, dones, 0.99, 0.95, 0.0
    )
    np.testing.assert_allclose(advantages, expected_adv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(returns, expected_ret, rtol=1e-5, atol=1e-5)
    print("   [OK] Backward compatibility: falls back to bootstrap_value when critic is missing.")


def test_bootstrap_uses_full_joint_state():
    """
    Verify that changing a non-first agent's observation changes the bootstrap value.
    This proves the bootstrap uses the full joint state, not just agent 0.
    """
    num_agents = 3
    obs_dim = OBSERVATION_DIM
    critic = CentralizedCritic(obs_dim=obs_dim, hidden=8, mode="pool")

    # Two different joint observations that differ only in agent 1
    joint_a = np.ones((num_agents, obs_dim), dtype=np.float32) * 0.1
    joint_b = np.ones((num_agents, obs_dim), dtype=np.float32) * 0.1
    joint_b[1, :] = 0.9  # change only agent 1

    with torch.no_grad():
        val_a = float(critic(torch.tensor(joint_a, dtype=torch.float32).unsqueeze(0)).item())
        val_b = float(critic(torch.tensor(joint_b, dtype=torch.float32).unsqueeze(0)).item())

    assert val_a != val_b, (
        f"Bootstrap values should differ when joint state differs, got {val_a} and {val_b}"
    )
    print(f"   [OK] Bootstrap is sensitive to full joint state: agent0-only={val_a:.4f}, agent1-changed={val_b:.4f}")


def test_bootstrap_shape_validation():
    """
    Verify that compute_gae raises a clear error when next_local_obs has wrong shape.
    """
    rewards = np.array([1.0], dtype=np.float32)
    values = np.array([0.5], dtype=np.float32)
    dones = np.array([False], dtype=np.bool_)
    critic = CentralizedCritic(obs_dim=OBSERVATION_DIM, hidden=8, mode="pool")

    try:
        compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_local_obs=np.ones(OBSERVATION_DIM, dtype=np.float32),  # wrong: 1D instead of 2D
            critic=critic,
        )
        assert False, "Expected AssertionError for wrong next_local_obs shape"
    except AssertionError as e:
        if "Expected next_local_obs shape" in str(e):
            print("   [OK] Shape validation raises clear error for 1D next_local_obs.")
        else:
            raise


def main():
    print("GAE Bootstrap Regression Test")
    print("=" * 50)

    test_terminated_rollout()
    test_truncated_rollout()
    test_no_bootstrap_when_critic_missing()
    test_bootstrap_uses_full_joint_state()
    test_bootstrap_shape_validation()

    print("\nAll GAE bootstrap tests passed.")


if __name__ == "__main__":
    main()
