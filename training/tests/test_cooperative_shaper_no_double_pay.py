"""
Regression test for CooperativeRewardShaper double-payment fix.

Verifies that a single pass completion in a 5_vs_5/11_vs_11-style scenario
(which falls back to CooperativeRewardShaper because get_reward_adapter raises
ValueError for these scenario names) produces exactly one reward contribution
for that pass, not two.

Before fix: engine +0.15 (base_rewards) + adapter +0.30 (r_pass) = +0.45 for
the passing agent — a genuine double-payment.

After fix: engine +0.15 (base_rewards) + adapter +0.0 (r_pass=0.0) = +0.15 for
the passing agent — only the engine's base reward survives, matching the
established pattern in BaseScenarioRewardAdapter.
"""

import pytest

from training.gmn_pettingzoo import CooperativeRewardShaper


class TestCooperativeShaperNoDoublePay:
    """CooperativeRewardShaper must not double-pay for pass completions."""

    def test_single_pass_with_engine_reward_produces_single_contribution(self):
        """Engine +0.15 + adapter r_pass must equal +0.15, not +0.45."""
        shaper = CooperativeRewardShaper()
        shaper.reset()

        # Simulate engine broadcast: all agents receive +0.15 shared reward
        base_rewards = {"left_0": 0.15, "left_1": 0.15, "left_2": 0.15}
        step_events = [{"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"}]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0", "left_1", "left_2"]

        rewards = shaper.compute_shaped_rewards(
            base_rewards, step_events, ground_truth, active_agents
        )

        # Post-fix: engine +0.15 is the sole pass reward. Adapter r_pass=0.0.
        assert rewards["left_0"] == pytest.approx(0.15)
        assert rewards["left_1"] == pytest.approx(0.15)
        assert rewards["left_2"] == pytest.approx(0.15)

        # No double-payment: adapter did not add anything on top of engine.
        assert shaper.r_pass == 0.0

    def test_two_different_passes_same_tick_with_engine_reward(self):
        """Two distinct passes on the same tick must each earn exactly +0.15."""
        shaper = CooperativeRewardShaper()
        shaper.reset()

        base_rewards = {"left_0": 0.15, "left_1": 0.15, "left_2": 0.15}
        step_events = [
            {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"},
            {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"},
        ]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_1"}}
        active_agents = ["left_0", "left_1", "left_2"]

        rewards = shaper.compute_shaped_rewards(
            base_rewards, step_events, ground_truth, active_agents
        )

        # Each pass earns exactly the engine's +0.15; adapter contributes nothing.
        assert rewards["left_0"] == pytest.approx(0.15)
        assert rewards["left_1"] == pytest.approx(0.15)
        assert rewards["left_2"] == pytest.approx(0.15)

        assert shaper.pass_completed_count == 2
        assert shaper.pass_chain_length == 2

    def test_r_pass_default_is_zero(self):
        """CooperativeRewardShaper.r_pass must be 0.0 to avoid double-payment."""
        shaper = CooperativeRewardShaper()
        assert shaper.r_pass == 0.0
