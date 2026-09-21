"""Regression test for reset-path ball-ownership resolution timing bug.

Verifies that _last_ball_owner_agent_idx is resolved against the current
episode's controlled-agent list, not a stale/empty self.agents from the
previous episode.

This guards against the bug where ownership resolution happened BEFORE
self.agents was refreshed in reset(), causing every non-first reset to
resolve to 255 (no owner) even when the bridge reported a controlled owner.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from training.gmn_pettingzoo import GMNMultiAgentEnv, OBSERVATION_DIM


def _make_reset_response(owner_id="left_1"):
    """Create a realistic reset JSON response with a controlled owner."""
    controllable_ids = ["left_1", "left_2", "left_3"]
    obs_list = [np.zeros(OBSERVATION_DIM, dtype=np.float32).tolist() for _ in controllable_ids]
    masks = [[1] * 19 for _ in controllable_ids]
    return {
        "type": "reset",
        "info": {
            "controllableAgentIds": controllable_ids,
            "current_ball_owner": {"agent_id": owner_id, "team": "left"},
            "action_masks": masks,
        },
        "observations": obs_list,
        "action_masks": masks,
    }


def _make_terminal_step_response(env):
    """Create a terminal step binary frame."""
    # Header format: <f??BBffBB (18 bytes)
    data = bytearray(18 + OBSERVATION_DIM * 4 * len(env.agents) + 19 * len(env.agents))
    data[4] = 1   # term = True
    data[5] = 0   # trunc = False
    data[6] = 0   # score_l
    data[7] = 0   # score_r
    data[16] = 11 # event_code = scenario_complete
    data[17] = 0  # ball_owner_agent_idx
    return bytes(data), {}, 0.0, [], None


def _create_env_with_mock_bridge(owner_id="left_1"):
    """Create a GMNMultiAgentEnv with a mocked WS client."""
    with patch('training.gmn_pettingzoo.GMNMultiAgentEnv._ensure_bridge_running', return_value=None), \
         patch('training.gmn_pettingzoo.GMNMultiAgentEnv._connect_ws', return_value=None):
        env = GMNMultiAgentEnv(
            scenario="academy_3_vs_1_with_keeper_onball",
            auto_start_bridge=False,
            port=9999,
            enable_reward_shaping=False,
            training_mode=False,
        )

    mock_client = MagicMock()
    mock_client.send = MagicMock()
    mock_client.recv = lambda *a, **kw: json.dumps(_make_reset_response(owner_id))
    env.ws_client = mock_client
    return env


def test_reset_ownership_resolves_against_current_agents_not_stale_state():
    """Tick-0 ownership must resolve to the correct controlled-agent index
    on every reset, including episodes after the first."""
    env = _create_env_with_mock_bridge(owner_id="left_1")
    num_episodes = 10

    for ep in range(num_episodes):
        obs, info = env.reset(seed=42)
        agents = list(env.agents)
        owner_idx = env._last_ball_owner_agent_idx

        # The bridge reports left_1 as owner; left_1 is at index 0 in the controllable list
        assert owner_idx == 0, (
            f"Episode {ep}: expected owner index 0 (left_1), got {owner_idx}. "
            f"agents={agents}"
        )

        # Step to terminal for next episode (except last)
        if ep < num_episodes - 1:
            with patch.object(env, '_recv_step_response', lambda *a, **kw: _make_terminal_step_response(env)):
                actions = {agent: 0 for agent in env.agents}
                obs, rewards, term, trunc, info = env.step(actions)


def test_reset_ownership_returns_255_when_owner_not_controlled():
    """When the bridge reports no owner or a non-controlled owner, the wrapper
    must return 255, not an incorrect index."""
    env = _create_env_with_mock_bridge(owner_id=None)
    env.reset(seed=42)
    assert env._last_ball_owner_agent_idx == 255


def test_reset_ownership_returns_255_when_owner_id_missing():
    """When current_ball_owner is present but has no agent_id, return 255."""
    # Patch the mock to return a response with a None agent_id owner
    env = _create_env_with_mock_bridge(owner_id="left_1")
    # Override the mock response for this test
    def mock_recv(*a, **kw):
        resp = _make_reset_response(owner_id=None)
        resp["info"]["current_ball_owner"] = {"team": "left"}  # no agent_id
        return json.dumps(resp)
    env.ws_client.recv = mock_recv
    env.reset(seed=42)
    assert env._last_ball_owner_agent_idx == 255


def test_reset_ownership_correct_index_for_each_controlled_agent():
    """Verify the correct index is returned for each possible controlled owner."""
    for owner_id, expected_idx in [("left_1", 0), ("left_2", 1), ("left_3", 2)]:
        env = _create_env_with_mock_bridge(owner_id=owner_id)
        env.reset(seed=42)
        assert env._last_ball_owner_agent_idx == expected_idx, (
            f"Owner {owner_id}: expected index {expected_idx}, got {env._last_ball_owner_agent_idx}"
        )
