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
    )

    np.testing.assert_allclose(advantages, expected_adv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(returns, expected_ret, rtol=1e-5, atol=1e-5)
    print("   [OK] Terminated rollout: advantages and returns match hand-computed values.")


def test_truncated_rollout():
    """
    3-timestep rollout truncated at the final step (episode ongoing).
    bootstrap_value should be computed from critic(next_obs), not defaulted to 0.0.
    """
    obs_dim = OBSERVATION_DIM
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    dones = np.array([False, False, False], dtype=np.bool_)
    gamma = 0.99
    lam = 0.95

    # next_obs with a known value through the critic
    next_obs = np.ones(obs_dim, dtype=np.float32) * 0.7
    critic = CentralizedCritic(global_state_dim=obs_dim * 3, hidden=8)

    # Compute what the critic would output for next_obs
    with torch.no_grad():
        obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        expected_bootstrap = float(critic(obs_tensor).item())

    # Hand-compute expected advantages/returns using the real bootstrap value
    expected_adv, expected_ret = compute_expected_gae(
        rewards, values, dones, gamma, lam, expected_bootstrap
    )

    # compute_gae should automatically use critic(next_obs) because dones[-1]=False
    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=gamma,
        lam=lam,
        bootstrap_value=0.0,  # intentionally wrong; should be overridden
        next_obs=next_obs,
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
        next_obs=np.ones(OBSERVATION_DIM, dtype=np.float32),
        critic=None,
    )

    # With bootstrap_value=0.0 and no critic, this should match the old behavior
    expected_adv, expected_ret = compute_expected_gae(
        rewards, values, dones, 0.99, 0.95, 0.0
    )
    np.testing.assert_allclose(advantages, expected_adv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(returns, expected_ret, rtol=1e-5, atol=1e-5)
    print("   [OK] Backward compatibility: falls back to bootstrap_value when critic is missing.")


def main():
    print("GAE Bootstrap Regression Test")
    print("=" * 50)

    test_terminated_rollout()
    test_truncated_rollout()
    test_no_bootstrap_when_critic_missing()

    print("\nAll GAE bootstrap tests passed.")


if __name__ == "__main__":
    main()
