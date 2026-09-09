"""
GMN-Football-3 — Bridge Error Frame Sentinel Regression Test

Verifies that the -999.0 sentinel reward error frame from the bridge is
detected and raises RuntimeError instead of being silently accumulated
into episode reward.

Frame layout (non-rondo, 18-byte header):
  Offset 0   (4B float32): reward (-999.0 for error frames)
  Offset 4   (1B uint8):   terminated (true for error frames)
  Offset 5   (1B uint8):   truncated
  Offset 6   (1B uint8):   scoreLeft
  Offset 7   (1B uint8):   scoreRight
  Offset 8   (4B float32): checkpointReward
  Offset 12  (4B float32): ballDistanceToGoal
  Offset 16  (1B uint8):   eventCode
  Offset 17  (1B uint8):   ballOwnerAgentId
"""

import os
import sys
import struct
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv, OBSERVATION_DIM, ACTION_SPACE_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_error_frame(
    scenario: str = "academy_3_vs_1_with_keeper",
    n_agents: int = 3,
) -> bytes:
    """Construct a synthetic error frame matching encodeErrorStepBinary output.
    
    Error frames have:
    - reward = -999.0
    - terminated = true
    - All other fields zeroed
    - Same frame length as normal frames
    """
    is_rondo = scenario == "academy_rondo_4v1"
    obs_bytes = OBSERVATION_DIM * 4
    mask_bytes = 19
    header_size = 22 if is_rondo else 18
    
    # For multi-agent, frame size is header + obs*agents + mask*agents
    # But error frames for single agent have nAgents=1 minimum
    total_len = header_size + obs_bytes * max(1, n_agents) + mask_bytes * max(1, n_agents)
    
    buf = bytearray(total_len)
    # Zero-fill the buffer (bytearray doesn't have fill() method)
    for i in range(total_len):
        buf[i] = 0
    
    # Error frame sentinel
    struct.pack_into("<f", buf, 0, -999.0)
    buf[4] = 1  # terminated = true
    # All other fields remain zero
    
    return bytes(buf)


def _build_normal_frame(
    reward: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
    n_agents: int = 3,
) -> bytes:
    """Construct a synthetic normal binary frame."""
    obs_bytes = OBSERVATION_DIM * 4
    header_size = 18
    mask_bytes = 19
    total_len = header_size + obs_bytes * n_agents + mask_bytes * n_agents
    
    buf = bytearray(total_len)
    # Header
    struct.pack_into("<f", buf, 0, reward)
    buf[4] = 1 if terminated else 0
    buf[5] = 1 if truncated else 0
    buf[6] = 0  # scoreLeft
    buf[7] = 0  # scoreRight
    struct.pack_into("<f", buf, 8, 0.0)  # checkpointReward
    struct.pack_into("<f", buf, 12, 0.0)  # ballDistanceToGoal
    buf[16] = 0  # eventCode
    buf[17] = 255  # ballOwnerAgentId
    
    # Observations and masks (zero-filled)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBridgeErrorFrameSentinel:
    """Verify that -999 error frames are detected and raise RuntimeError."""

    def test_error_frame_has_correct_sentinel_values(self):
        """An error frame must have reward=-999.0 and terminated=true."""
        frame = _build_error_frame()
        reward, term, trunc, score_l, score_r, cp_reward, dist_goal, event_code, ball_owner = struct.unpack_from(
            "<f??BBffBB", frame, 0
        )
        assert float(reward) == -999.0
        assert bool(term) is True

    def test_error_frame_detected_by_sentinel_check(self):
        """The sentinel check (reward <= -998.0 and term) must catch error frames."""
        frame = _build_error_frame()
        reward, term, trunc, *_ = struct.unpack_from("<f??BBffBB", frame, 0)
        assert float(reward) <= -998.0
        assert bool(term) is True

    def test_normal_frame_not_flagged_as_error(self):
        """Normal frames with reward > -998.0 must not trigger the sentinel check."""
        frame = _build_normal_frame(reward=-500.0, terminated=True)
        reward, term, trunc, *_ = struct.unpack_from("<f??BBffBB", frame, 0)
        assert not (float(reward) <= -998.0 and bool(term))

    def test_error_frame_length_matches_normal_frame(self):
        """Error frames must have the same length as normal frames so clients
        can decode them without hanging."""
        error_frame = _build_error_frame(n_agents=3)
        normal_frame = _build_normal_frame(n_agents=3)
        assert len(error_frame) == len(normal_frame)

    def test_negative_998_flagged_as_error(self):
        """A reward of exactly -998.0 IS flagged as an error frame
        (the threshold is <= -998.0)."""
        frame = _build_normal_frame(reward=-998.0, terminated=True)
        reward, term, trunc, *_ = struct.unpack_from("<f??BBffBB", frame, 0)
        assert float(reward) <= -998.0 and bool(term)

    def test_negative_999_without_termination_not_flagged(self):
        """A reward of -999.0 with terminated=false should not be flagged as
        an error frame (both conditions must be true)."""
        frame = _build_normal_frame(reward=-999.0, terminated=False)
        reward, term, trunc, *_ = struct.unpack_from("<f??BBffBB", frame, 0)
        assert not (float(reward) <= -998.0 and bool(term))


class TestBridgeErrorFrameRuntimeError:
    """Verify that receiving an error frame raises RuntimeError in step()."""

    def test_step_raises_runtime_error_on_error_frame(self, monkeypatch):
        """When the bridge returns an error frame, step() must raise RuntimeError."""
        env = GMNMultiAgentEnv.__new__(GMNMultiAgentEnv)
        env._forensic_debug = False
        env._episode_index = 1
        env._step_count = 0
        env._bridge_error_count = 0
        env.scenario = "academy_3_vs_1_with_keeper"
        env.agents = ["left_1", "left_2", "left_3"]
        env.possible_agents = ["left_1", "left_2", "left_3"]
        env._act_space = type('FakeSpace', (), {"n": ACTION_SPACE_SIZE})()
        env._last_frame_reward = 0.0
        env._last_frame_event_code = 0
        env._last_frame_score = {"left": 0, "right": 0}
        env._last_shared_reward = 0.0
        env.enable_reward_shaping = False
        env.reward_shaper = None
        env.debug_rewards = False
        env._pending_pass = None
        env.batch_size = 1
        env._batch_envs = []
        env._needs_bridge = False
        
        # Mock WebSocket client
        mock_ws = type('MockWS', (), {})()
        error_frame = _build_error_frame(n_agents=3)
        
        call_count = [0]
        def mock_send(data):
            call_count[0] += 1
        
        def mock_recv(timeout=None):
            return error_frame
        
        mock_ws.send = mock_send
        mock_ws.recv = mock_recv
        mock_ws.close = lambda: None
        env.ws_client = mock_ws
        env.ws_recv_timeout = 5.0
        
        with pytest.raises(RuntimeError, match="Bridge Error"):
            env.step({"left_1": 0, "left_2": 0, "left_3": 0})
