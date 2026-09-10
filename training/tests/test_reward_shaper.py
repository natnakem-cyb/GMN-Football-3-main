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

        assert rewards["left_0"] == pytest.approx(0.30)
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

    def test_solitary_shot_penalty_attributed_via_shot_actions_when_no_agent_id(self):
        """Wire path: engine broadcasts SHOT_TAKEN while the ball is in flight, so
        the event arrives WITHOUT an agent_id. Every left-team agent that commanded
        a SHOT action that frame is the "shooter" and must pay the solitary penalty."""
        shaper = CooperativeRewardShaper()
        shaper.reset()

        base_rewards = {"left_0": 0.0, "left_1": 0.0, "left_2": 0.0}
        step_events = [{"type": "SHOT_TAKEN", "team": "left"}]  # no agent_id (ball loose)
        ground_truth = {"current_ball_owner": None}
        active_agents = ["left_0", "left_1", "left_2"]
        actions = {"left_0": 12, "left_1": 12, "left_2": 5}  # left_0, left_1 SHOT-spam

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents, actions)

        # -0.30 solitary-shot penalty ON TOP of the -0.01 per-ball-action cost
        # for each agent that commanded SHOT (12) this frame.
        assert rewards["left_0"] == pytest.approx(-0.31)
        assert rewards["left_1"] == pytest.approx(-0.31)
        assert rewards["left_2"] == pytest.approx(0.0)
        assert shaper.solitary_shot_count == 2
        assert shaper.pass_chain_length == 0

    def test_solitary_shot_penalty_after_pass_chain_not_applied(self):
        """A shot following a completed pass (pass_chain_length > 0) is assisted
        and must NOT pay the solitary penalty, even when the event lacks agent_id."""
        shaper = CooperativeRewardShaper()
        shaper.reset()
        shaper.pass_chain_length = 1

        base_rewards = {"left_0": 0.0}
        step_events = [{"type": "SHOT_TAKEN", "team": "left"}]  # no agent_id
        ground_truth = {"current_ball_owner": None}
        active_agents = ["left_0"]
        actions = {"left_0": 12}

        rewards = shaper.compute_shaped_rewards(base_rewards, step_events, ground_truth, active_agents, actions)

        # Assisted shot: NO solitary penalty; only the -0.01 per-ball-action cost
        # for commanding SHOT (12).
        assert rewards["left_0"] == pytest.approx(-0.01)
        assert shaper.solitary_shot_count == 0

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
        shaper = CooperativeRewardShaper(max_unassisted_hold_ticks=2, penalty_ball_hogging=-0.02)
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
        assert rewards["left_0"] == pytest.approx(-0.02)
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

        assert rewards["left_0"] == pytest.approx(0.30)
        assert rewards["left_1"] == pytest.approx(0.30)
        assert rewards["left_2"] == pytest.approx(0.0)
        assert shaper.pass_chain_length == 2

    def test_wired_path_pass_chain_differs_from_solitary_shot(self):
        shaper = CooperativeRewardShaper()
        shaper.reset()

        base_rewards = {"left_0": 0.0, "left_1": 0.0}
        active_agents = ["left_0", "left_1"]

        # Scenario A: pass chain completed then goal
        shaper.pass_chain_length = 1
        events_pass = [
            {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"},
            {"type": "GOAL_SCORED", "team": "left", "agent_id": "left_1"},
        ]
        gt_pass = {"current_ball_owner": {"team": "left", "agent_id": "left_1"}}
        rewards_pass = shaper.compute_shaped_rewards(base_rewards.copy(), events_pass, gt_pass, active_agents)

        # Scenario B: solitary shot then goal (no pass chain)
        shaper.reset()
        events_shot = [
            {"type": "SHOT_TAKEN", "team": "left", "agent_id": "left_0"},
            {"type": "GOAL_SCORED", "team": "left", "agent_id": "left_0"},
        ]
        gt_shot = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
        rewards_shot = shaper.compute_shaped_rewards(base_rewards.copy(), events_shot, gt_shot, active_agents)

        # Pass-chain path should yield strictly higher shaped reward than solitary-shot path
        total_pass = sum(rewards_pass[a] for a in active_agents)
        total_shot = sum(rewards_shot[a] for a in active_agents)
        assert total_pass > total_shot, (
            f"Pass-chain total reward {total_pass} must exceed solitary-shot total {total_shot}"
        )


class TestPendingPassStateMachine:
    """Tests for GMNMultiAgentEnv._resolve_pending_pass (no bridge required)."""

    @staticmethod
    def _make_env(agents):
        import types

        from training.gmn_pettingzoo import GMNMultiAgentEnv

        dummy = types.SimpleNamespace(
            agents=list(agents),
            _pending_pass=None,
            enable_reward_shaping=True,
            reward_shaper=object(),  # any non-None marker
        )
        # Bind the real method to the dummy so production logic is exercised.
        dummy._resolve_pending_pass = GMNMultiAgentEnv._resolve_pending_pass.__get__(dummy)
        return dummy

    def test_pass_resolves_completed_when_teammate_gains_possession(self):
        env = self._make_env(["left_0", "left_1"])
        # Pass initiated by left_0 (owner still left_0 on the initiation frame).
        events = env._resolve_pending_pass(0, "pass")
        assert events == []
        assert env._pending_pass["agent_id"] == "left_0"
        # Ball loose in flight for a step (owner byte 255).
        assert env._resolve_pending_pass(255, None) == []
        # Teammate left_1 gains possession.
        events = env._resolve_pending_pass(1, None)
        assert events == [{"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"}]
        assert env._pending_pass is None

    def test_pass_resolves_failed_when_right_team_gains_possession(self):
        env = self._make_env(["left_0", "right_0"])
        env._resolve_pending_pass(0, "pass")
        events = env._resolve_pending_pass(1, None)
        assert events == [{"type": "PASS_FAILED", "team": "left", "agent_id": "left_0"}]

    def test_pass_resolves_failed_on_turnover_event_while_loose(self):
        env = self._make_env(["left_0", "left_1"])
        env._resolve_pending_pass(0, "pass")
        events = env._resolve_pending_pass(255, "interception")
        assert events == [{"type": "PASS_FAILED", "team": "left", "agent_id": "left_0"}]

    def test_loose_ball_pass_times_out_to_failed(self):
        env = self._make_env(["left_0", "left_1"])
        env._resolve_pending_pass(0, "pass")
        for _ in range(60):
            env._resolve_pending_pass(255, None)  # ball loose, no outcome
        events = env._resolve_pending_pass(255, None)  # timeout step
        assert events == [{"type": "PASS_FAILED", "team": "left", "agent_id": "left_0"}]

    def test_new_pass_finalizes_previous_as_completed(self):
        env = self._make_env(["left_0", "left_1", "left_2"])
        env._resolve_pending_pass(0, "pass")
        # Rapid second pass before possession was observable: previous completes.
        events = env._resolve_pending_pass(1, "pass")
        assert events == [{"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"}]
        assert env._pending_pass["agent_id"] == "left_1"

