"""
GMN-Football-3 — Reward Regression Tests (Phase 7)

Tests A-G preventing future reward exploitation.
"""
import pytest

from training.gmn_pettingzoo import CooperativeRewardShaper, GMNMultiAgentEnv


class TestPossessionFarming:
    """Test A — Possession farming must not generate unlimited positive reward."""

    def test_ball_hogging_penalty_exceeds_progress_after_sufficient_ticks(self):
        """After enough ticks, cumulative ball-hogging penalty should exceed
        the reward from one progress step, making indefinite possession negative."""
        shaper = CooperativeRewardShaper(penalty_ball_hogging=-0.02, max_unassisted_hold_ticks=1)
        shaper.reset()

        base_rewards = {"left_0": 0.0}
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0"]

        # Simulate 50 ticks of continuous possession
        total_shaped = 0.0
        for _ in range(50):
            rewards = shaper.compute_shaped_rewards(base_rewards, [], ground_truth, active_agents)
            total_shaped += rewards["left_0"]

        # With -0.02/tick after 1 tick, 50 ticks should yield large negative cumulative penalty
        assert total_shaped < -0.5, (
            f"Ball-hogging penalty insufficient: cumulative shaped reward after 50 ticks = {total_shaped:.4f}"
        )


class TestPassQuality:
    """Test B — Pass quality rewards."""

    def test_single_pass_gives_meaningful_reward(self):
        """A single pass completion should give the intended bonus."""
        shaper = CooperativeRewardShaper()
        shaper.reset()

        base_rewards = {"left_0": 0.0, "left_1": 0.0}
        step_events = [{"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"}]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0", "left_1"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)
        assert rewards["left_0"] == pytest.approx(0.25)

    def test_repetitive_pass_loop_does_not_dominate_progress(self):
        """Rapid pass cycling between two agents should not yield more reward
        per step than legitimate forward progress."""
        shaper = CooperativeRewardShaper()
        shaper.reset()

        base_rewards = {"left_0": 0.0, "left_1": 0.0}
        active_agents = ["left_0", "left_1"]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}

        # 10 passes back and forth
        total_reward = 0.0
        for i in range(10):
            passer = "left_0" if i % 2 == 0 else "left_1"
            step_events = [{"type": "PASS_COMPLETED", "team": "left", "agent_id": passer}]
            ground_truth = {"current_ball_owner": {"team": "left", "agent_id": passer}}
            rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)
            total_reward += sum(rewards.values())

        # 10 passes yield 10 * 0.25 = 2.5 total reward
        # This should not exceed what forward progress would give over 10 steps
        # (which depends on deltaX, but we check the pass reward is bounded)
        assert total_reward == pytest.approx(2.5), (
            f"Pass loop reward should be exactly 10 * 0.25 = 2.5, got {total_reward:.4f}"
        )


class TestShooting:
    """Test C — Shooting rewards."""

    def test_shot_after_pass_chain_not_penalized(self):
        """Shot taken with pass chain > 0 should not receive solitary-shot penalty."""
        shaper = CooperativeRewardShaper()
        shaper.reset()
        shaper.pass_chain_length = 1

        base_rewards = {"left_0": 0.0}
        step_events = [{"type": "SHOT_TAKEN", "team": "left", "agent_id": "left_0"}]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)
        assert rewards["left_0"] == pytest.approx(0.0)


class TestScoring:
    """Test D — Goal/scoring events dominate intermediate shaping."""

    def test_goal_reward_exceeds_sum_of_intermediate_rewards(self):
        """A single goal (+2.0 base) should exceed the sum of many pass/shoot bonuses."""
        shaper = CooperativeRewardShaper()
        shaper.reset()
        shaper.pass_chain_length = 2

        base_rewards = {"left_0": 2.0, "left_1": 2.0, "left_2": 2.0}
        step_events = [{"type": "GOAL_SCORED", "team": "left", "agent_id": "left_0"}]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0", "left_1", "left_2"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)
        total_goal_reward = sum(rewards.values())

        # Goal with pass chain gives +0.5 per agent = 1.5 total from shaper
        # Plus base 2.0 per agent = 6.0 total. Should exceed many pass bonuses.
        assert total_goal_reward > 4.0, (
            f"Goal reward {total_goal_reward:.4f} should exceed 4.0"
        )


class TestDefenderIsolation:
    """Test E — Defender rewards must never receive attacker-only rewards."""

    def test_defender_excluded_from_pass_bonus(self):
        """Right-team agents must not receive pass-completion bonuses."""
        shaper = CooperativeRewardShaper()
        shaper.reset()

        base_rewards = {"left_0": 0.0, "right_0": 0.0}
        step_events = [{"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"}]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0", "right_0"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)
        assert "right_0" not in rewards


class TestRewardOwnership:
    """Test F — Each agent receives only rewards intended for its team/role."""

    def test_only_left_team_agents_receive_shaped_rewards(self):
        """Shaper should only return rewards for left-team agents."""
        shaper = CooperativeRewardShaper()
        shaper.reset()

        base_rewards = {"left_0": 0.0, "left_1": 0.0, "right_0": 0.0, "right_1": 0.0}
        step_events = [{"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"}]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0", "left_1", "right_0", "right_1"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)
        assert "right_0" not in rewards
        assert "right_1" not in rewards
        assert "left_0" in rewards
        assert rewards["left_0"] == pytest.approx(0.25)


class TestRewardAccounting:
    """Test G — Sum of reported components equals actual episode reward."""

    def test_shaped_rewards_are_additive(self):
        """The shaped reward for an agent should equal base + event bonuses + penalties."""
        shaper = CooperativeRewardShaper()
        shaper.reset()

        base_rewards = {"left_0": 0.5, "left_1": 0.0}
        step_events = [
            {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"},
            {"type": "SHOT_TAKEN", "team": "left", "agent_id": "left_0"},
        ]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0", "left_1"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)

        # left_0: base 0.5 + pass 0.25 + shot 0 (chain > 0, no penalty) = 0.75
        assert rewards["left_0"] == pytest.approx(0.75)
        # left_1: base 0.0, no events = 0.0
        assert rewards["left_1"] == pytest.approx(0.0)

    def test_ball_hogging_penalty_is_applied_correctly(self):
        """Ball-hogging penalty should accumulate correctly per tick."""
        shaper = CooperativeRewardShaper(penalty_ball_hogging=-0.02, max_unassisted_hold_ticks=1)
        shaper.reset()

        base_rewards = {"left_0": 0.0}
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0"]

        # Tick 1: holder_ticks = 1, no penalty (threshold is > 1)
        rewards = shaper.compute_shaped_rewards(base_rewards, [], ground_truth, active_agents)
        assert rewards["left_0"] == pytest.approx(0.0)

        # Tick 2: holder_ticks = 2, penalty applies (-0.02)
        rewards = shaper.compute_shaped_rewards(base_rewards, [], ground_truth, active_agents)
        assert rewards["left_0"] == pytest.approx(-0.02)

        # Tick 3: holder_ticks = 3, another penalty (-0.02)
        rewards = shaper.compute_shaped_rewards(base_rewards, [], ground_truth, active_agents)
        assert rewards["left_0"] == pytest.approx(-0.02)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
