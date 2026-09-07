"""
GMN-Football-3 — CooperativeRewardShaper Unit Test Suite
Validates reward shaping mechanics:
- Pass completion bonuses
- Assisted goal bonuses
- Solitary shot penalties
- Ball-hogging penalties
- Turnover chain breaks
"""

import pytest

from training.gmn_pettingzoo import CooperativeRewardShaper


class TestCooperativeRewardShaper:
    """Test suite for CooperativeRewardShaper."""

    def test_pass_completion_rewards_passer_and_increments_chain(self):
        shaper = CooperativeRewardShaper()
        shaper.reset()

        base_rewards = {"left_0": 0.0, "left_1": 0.0, "left_2": 0.0}
        step_events = [{"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"}]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0", "left_1", "left_2"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)

        assert rewards["left_0"] == pytest.approx(0.25)
        assert rewards["left_1"] == pytest.approx(0.0)
        assert rewards["left_2"] == pytest.approx(0.0)
        assert shaper.pass_chain_length == 1

    def test_solitary_shot_penalty_when_no_pass_chain(self):
        shaper = CooperativeRewardShaper()
        shaper.reset()

        base_rewards = {"left_0": 0.0}
        step_events = [{"type": "SHOT_TAKEN", "team": "left", "agent_id": "left_0"}]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)

        assert rewards["left_0"] == pytest.approx(-0.30)
        assert shaper.pass_chain_length == 0

    def test_assisted_goal_bonus_distributed_to_all_active_agents(self):
        shaper = CooperativeRewardShaper()
        shaper.reset()
        shaper.pass_chain_length = 2

        base_rewards = {"left_0": 0.0, "left_1": 0.0, "left_2": 0.0}
        step_events = [{"type": "GOAL_SCORED", "team": "left", "agent_id": "left_0"}]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0", "left_1", "left_2"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)

        for agent in active_agents:
            assert rewards[agent] == pytest.approx(0.50)
        assert shaper.pass_chain_length == 0

    def test_ball_hogging_penalty_after_max_hold_ticks(self):
        shaper = CooperativeRewardShaper(max_unassisted_hold_ticks=2)
        shaper.reset()

        base_rewards = {"left_0": 0.0}
        step_events = []
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0"]

        # Tick 1: holder_ticks becomes 1, no penalty
        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)
        assert rewards["left_0"] == pytest.approx(0.0)
        assert shaper.holder_ticks == 1

        # Tick 2: holder_ticks becomes 2, still no penalty (threshold is > max_hold_ticks)
        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)
        assert rewards["left_0"] == pytest.approx(0.0)
        assert shaper.holder_ticks == 2

        # Tick 3: holder_ticks becomes 3, penalty applies
        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)
        assert rewards["left_0"] == pytest.approx(-0.005)
        assert shaper.holder_ticks == 3

    def test_turnover_chain_break(self):
        shaper = CooperativeRewardShaper()
        shaper.reset()
        shaper.pass_chain_length = 3

        base_rewards = {"left_0": 0.0}
        step_events = [{"type": "PASS_FAILED", "team": "left", "agent_id": "left_0"}]
        ground_truth = {"current_ball_owner": {"team": "right", "agent_id": "right_0"}}
        active_agents = ["left_0"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)

        assert rewards["left_0"] == pytest.approx(0.0)
        assert shaper.pass_chain_length == 0

    def test_non_left_team_events_ignored(self):
        shaper = CooperativeRewardShaper()
        shaper.reset()

        base_rewards = {"left_0": 0.0}
        step_events = [
            {"type": "PASS_COMPLETED", "team": "right", "agent_id": "right_0"},
            {"type": "SHOT_TAKEN", "team": "right", "agent_id": "right_0"},
        ]
        ground_truth = {"current_ball_owner": {"team": "right", "agent_id": "right_0"}}
        active_agents = ["left_0"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)

        assert rewards["left_0"] == pytest.approx(0.0)
        assert shaper.pass_chain_length == 0

    def test_reset_clears_state(self):
        shaper = CooperativeRewardShaper()
        shaper.pass_chain_length = 5
        shaper.current_holder_id = "left_0"
        shaper.holder_ticks = 30

        shaper.reset()

        assert shaper.pass_chain_length == 0
        assert shaper.current_holder_id is None
        assert shaper.holder_ticks == 0

    def test_missing_event_fields_do_not_crash(self):
        shaper = CooperativeRewardShaper()
        shaper.reset()

        base_rewards = {"left_0": 0.0}
        step_events = [
            {"type": "PASS_COMPLETED"},  # missing team, agent_id
            {"team": "left"},  # missing type
            "invalid_event",  # not a dict
            None,  # None
        ]
        ground_truth = {}
        active_agents = ["left_0"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)

        assert rewards["left_0"] == pytest.approx(0.0)
        assert shaper.pass_chain_length == 0

    def test_shot_after_pass_chain_not_penalized(self):
        shaper = CooperativeRewardShaper()
        shaper.reset()
        shaper.pass_chain_length = 1

        base_rewards = {"left_0": 0.0}
        step_events = [{"type": "SHOT_TAKEN", "team": "left", "agent_id": "left_0"}]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)

        assert rewards["left_0"] == pytest.approx(0.0)
        assert shaper.pass_chain_length == 1

    def test_goal_without_pass_chain_does_not_give_bonus(self):
        shaper = CooperativeRewardShaper()
        shaper.reset()
        shaper.pass_chain_length = 0

        base_rewards = {"left_0": 0.0, "left_1": 0.0}
        step_events = [{"type": "GOAL_SCORED", "team": "left", "agent_id": "left_0"}]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        active_agents = ["left_0", "left_1"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)

        for agent in active_agents:
            assert rewards[agent] == pytest.approx(0.0)
        assert shaper.pass_chain_length == 0

    def test_multiple_pass_completions_accumulate_chain(self):
        shaper = CooperativeRewardShaper()
        shaper.reset()

        base_rewards = {"left_0": 0.0, "left_1": 0.0, "left_2": 0.0}
        step_events = [
            {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"},
            {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"},
        ]
        ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_1"}}
        active_agents = ["left_0", "left_1", "left_2"]

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents)

        assert rewards["left_0"] == pytest.approx(0.25)
        assert rewards["left_1"] == pytest.approx(0.25)
        assert rewards["left_2"] == pytest.approx(0.0)
        assert shaper.pass_chain_length == 2
