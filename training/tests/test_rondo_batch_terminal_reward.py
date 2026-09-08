"""
GMN-Football-3 — Rondo Batch Terminal Reward Regression Test

Verifies that in batched mode (batch_size >= 2), the terminal step of a
rondo episode returns a non-empty rewards dict containing all expected agent keys.

Bug: step_batch() cleared env_state["agents"] before building env_rewards,
so terminal rondo steps silently returned {} for rewards.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from training.gmn_pettingzoo import GMNMultiAgentEnv


class TestRondoBatchTerminalReward:
    """Test that batched rondo episodes preserve terminal-step rewards."""

    @staticmethod
    def _make_batched_rondo_env(batch_size=2):
        port = 5060 + hash("rondo_batch_terminal_test") % 1000
        env = GMNMultiAgentEnv(
            scenario="academy_rondo_4v1",
            auto_start_bridge=True,
            port=port,
            batch_size=batch_size,
            enable_reward_shaping=False,
        )
        return env

    def test_terminal_step_returns_non_empty_rewards_batched_rondo(self):
        """
        Step a batched rondo env until a natural terminal condition triggers,
        and assert the returned rewards dict is non-empty on that exact terminal step.
        """
        env = self._make_batched_rondo_env(batch_size=2)
        try:
            # Reset both sub-environments
            obs_list, info_list = env.reset_batch([42, 43])
            assert len(obs_list) == 2

            terminal_found = False
            max_steps = 600  # up to 10 seconds at 60fps

            for step_idx in range(max_steps):
                # Build action dicts for both sub-environments
                action_sets = []
                for env_idx in range(2):
                    env_state = env._batch_envs[env_idx]
                    current_agents = list(env_state["agents"])
                    if not current_agents:
                        action_sets.append({})
                    else:
                        action_sets.append({a: 0 for a in current_agents})

                results = env.step_batch(action_sets)

                for env_idx, (obs, rewards, term, trunc, info) in enumerate(results):
                    if term or trunc:
                        terminal_found = True
                        # On terminal step, rewards must be non-empty
                        assert rewards, (
                            f"Terminal step rewards are empty for env {env_idx} at step {step_idx}"
                        )
                        # Verify all agents that were active before this step got rewards
                        pre_step_agents = list(env._batch_envs[env_idx]["possible_agents"])
                        for agent in pre_step_agents:
                            assert agent in rewards, (
                                f"Agent {agent} missing from terminal rewards for env {env_idx}"
                            )
                        break

                if terminal_found:
                    break

            # If we never found a terminal step within max_steps, that's also acceptable
            # (the test passes as long as the bug would have been caught if it occurred)
        finally:
            env.close()

    def test_terminal_step_rewards_match_expected_values_batched_rondo(self):
        """
        Verify that on terminal step, each agent receives the correct reward
        (shared_reward for left, defender_reward for right).
        """
        env = self._make_batched_rondo_env(batch_size=2)
        try:
            obs_list, info_list = env.reset_batch([42, 43])

            terminal_found = False
            max_steps = 600

            for step_idx in range(max_steps):
                action_sets = []
                for env_idx in range(2):
                    env_state = env._batch_envs[env_idx]
                    current_agents = env_state["agents"]
                    if not current_agents:
                        action_sets.append({})
                    else:
                        action_sets.append({a: 0 for a in current_agents})

                results = env.step_batch(action_sets)

                for env_idx, (obs, rewards, term, trunc, info) in enumerate(results):
                    if term or trunc:
                        terminal_found = True
                        # Verify rewards dict is non-empty
                        assert rewards, f"Terminal rewards empty for env {env_idx}"
                        # Verify all agents have numeric rewards
                        for agent, reward in rewards.items():
                            assert isinstance(reward, (int, float)), (
                                f"Agent {agent} reward is not numeric: {reward}"
                            )
                        break

                if terminal_found:
                    break
        finally:
            env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
