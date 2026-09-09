"""
GMN-Football-3 — Byte-Level Regression Tests for Terminal Binary Frame Parser

These tests verify the Python struct parser in gmn_pettingzoo.py correctly
unpacks the primary reward float and (for rondo) the defender reward from
their specified offsets without cross-contamination.

Frame layout (non-rondo, 18-byte header):
  Offset 0   (4B float32): reward
  Offset 4   (1B uint8):   terminated
  Offset 5   (1B uint8):   truncated
  Offset 6   (1B uint8):   scoreLeft
  Offset 7   (1B uint8):   scoreRight
  Offset 8   (4B float32): checkpointReward
  Offset 12  (4B float32): ballDistanceToGoal
  Offset 16  (1B uint8):   eventCode
  Offset 17  (1B uint8):   ballOwnerAgentId

Frame layout (rondo, 22-byte header):
  Offset 0   (4B float32): reward (attacker reward)
  Offset 4   (1B uint8):   terminated
  Offset 5   (1B uint8):   truncated
  Offset 6   (1B uint8):   scoreLeft
  Offset 7   (1B uint8):   scoreRight
  Offset 8   (4B float32): checkpointReward
  Offset 12  (4B float32): ballDistanceToGoal
  Offset 16  (1B uint8):   eventCode
  Offset 17  (1B uint8):   ballOwnerAgentId
  Offset 18  (4B float32): defenderReward
  Offset 22+ observations...
"""

import os
import sys
import struct

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv, OBSERVATION_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_standard_frame(
    reward: float = 1.0,
    terminated: bool = False,
    truncated: bool = False,
    score_left: int = 0,
    score_right: int = 0,
    checkpoint_reward: float = 0.0,
    ball_distance_to_goal: float = 0.0,
    event_code: int = 0,
    ball_owner_agent_idx: int = 255,
    obs: np.ndarray = None,
    mask: np.ndarray = None,
) -> bytes:
    """Construct a synthetic 18-byte standard binary frame."""
    obs_bytes = OBSERVATION_DIM * 4
    header_size = 18
    mask_bytes = 19
    total_len = header_size + obs_bytes + mask_bytes

    buf = bytearray(total_len)
    # Header
    struct.pack_into("<f", buf, 0, reward)
    buf[4] = 1 if terminated else 0
    buf[5] = 1 if truncated else 0
    buf[6] = max(0, min(255, score_left))
    buf[7] = max(0, min(255, score_right))
    struct.pack_into("<f", buf, 8, checkpoint_reward)
    struct.pack_into("<f", buf, 12, ball_distance_to_goal)
    buf[16] = max(0, min(255, event_code))
    buf[17] = max(0, min(255, ball_owner_agent_idx))

    # Observations
    if obs is None:
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    for i in range(OBSERVATION_DIM):
        struct.pack_into("<f", buf, header_size + i * 4, float(obs[i]))

    # Action mask
    if mask is None:
        mask = np.ones(19, dtype=np.uint8)
    for i in range(19):
        buf[header_size + obs_bytes + i] = int(mask[i]) if i < len(mask) else 1

    return bytes(buf)


def _build_rondo_frame(
    attacker_reward: float = 1.0,
    defender_reward: float = 0.5,
    terminated: bool = False,
    truncated: bool = False,
    score_left: int = 0,
    score_right: int = 0,
    event_code: int = 0,
    ball_owner_agent_idx: int = 255,
    obs: np.ndarray = None,
    mask: np.ndarray = None,
) -> bytes:
    """Construct a synthetic 22-byte rondo binary frame."""
    obs_bytes = OBSERVATION_DIM * 4
    header_size = 22
    mask_bytes = 19
    total_len = header_size + obs_bytes + mask_bytes

    buf = bytearray(total_len)
    # Base header (same as standard)
    struct.pack_into("<f", buf, 0, attacker_reward)
    buf[4] = 1 if terminated else 0
    buf[5] = 1 if truncated else 0
    buf[6] = max(0, min(255, score_left))
    buf[7] = max(0, min(255, score_right))
    struct.pack_into("<f", buf, 8, 0.0)  # checkpointReward
    struct.pack_into("<f", buf, 12, 0.0)  # ballDistanceToGoal
    buf[16] = max(0, min(255, event_code))
    buf[17] = max(0, min(255, ball_owner_agent_idx))

    # Defender reward at offset 18 (bytes 18-21)
    struct.pack_into("<f", buf, 18, defender_reward)

    # Observations
    if obs is None:
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    for i in range(OBSERVATION_DIM):
        struct.pack_into("<f", buf, header_size + i * 4, float(obs[i]))

    # Action mask
    if mask is None:
        mask = np.ones(19, dtype=np.uint8)
    for i in range(19):
        buf[header_size + obs_bytes + i] = int(mask[i]) if i < len(mask) else 1

    return bytes(buf)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStandardFrameParser:
    """Verify the standard 18-byte binary frame parser."""

    def test_primary_float_unpacked_from_offset_0(self):
        """The raw reward float at offset 0 must be read correctly."""
        known_reward = -782.7302
        frame = _build_standard_frame(reward=known_reward)

        parsed = struct.unpack_from("<f??BBffBB", frame, 0)
        parsed_reward = float(parsed[0])

        # float32 has ~7 decimal digits of precision; use rel tolerance
        assert parsed_reward == pytest.approx(known_reward, rel=1e-5), (
            f"Primary float mismatch: expected {known_reward}, got {parsed_reward}"
        )

    def test_terminal_flag_unpacked_from_offset_4(self):
        """Terminated flag at offset 4 must be read as bool."""
        frame = _build_standard_frame(terminated=True)
        parsed = struct.unpack_from("<f??BBffBB", frame, 0)
        assert bool(parsed[1]) is True

        frame = _build_standard_frame(terminated=False)
        parsed = struct.unpack_from("<f??BBffBB", frame, 0)
        assert bool(parsed[1]) is False

    def test_truncated_flag_unpacked_from_offset_5(self):
        """Truncated flag at offset 5 must be read as bool."""
        frame = _build_standard_frame(truncated=True)
        parsed = struct.unpack_from("<f??BBffBB", frame, 0)
        assert bool(parsed[2]) is True

    def test_scores_unpacked_from_offset_6_7(self):
        """scoreLeft and scoreRight at offsets 6 and 7 must be correct."""
        frame = _build_standard_frame(score_left=3, score_right=1)
        parsed = struct.unpack_from("<f??BBffBB", frame, 0)
        assert parsed[3] == 3
        assert parsed[4] == 1

    def test_checkpoint_reward_unpacked_from_offset_8(self):
        """checkpointReward float at offset 8 must be read correctly."""
        known_cp = 0.15
        frame = _build_standard_frame(checkpoint_reward=known_cp)
        parsed = struct.unpack_from("<f??BBffBB", frame, 0)
        assert float(parsed[5]) == pytest.approx(known_cp, abs=1e-5)

    def test_ball_distance_unpacked_from_offset_12(self):
        """ballDistanceToGoal float at offset 12 must be read correctly."""
        known_dist = 12.5
        frame = _build_standard_frame(ball_distance_to_goal=known_dist)
        parsed = struct.unpack_from("<f??BBffBB", frame, 0)
        assert float(parsed[6]) == pytest.approx(known_dist, abs=1e-5)

    def test_event_code_unpacked_from_offset_16(self):
        """eventCode uint8 at offset 16 must be read correctly."""
        frame = _build_standard_frame(event_code=14)  # pass_completed
        parsed = struct.unpack_from("<f??BBffBB", frame, 0)
        assert parsed[7] == 14

    def test_ball_owner_unpacked_from_offset_17(self):
        """ballOwnerAgentId uint8 at offset 17 must be read correctly."""
        frame = _build_standard_frame(ball_owner_agent_idx=2)
        parsed = struct.unpack_from("<f??BBffBB", frame, 0)
        assert parsed[8] == 2

    def test_frame_length_is_18_plus_obs_plus_mask(self):
        """Total frame length must be 18 + OBSERVATION_DIM*4 + 19."""
        frame = _build_standard_frame()
        expected_len = 18 + OBSERVATION_DIM * 4 + 19
        assert len(frame) == expected_len


class TestRondoFrameParser:
    """Verify the 22-byte rondo binary frame parser."""

    def test_standard_reward_at_offset_0(self):
        """Attacker reward must be read from offset 0 in a rondo frame."""
        attacker_reward = 1.234
        frame = _build_rondo_frame(attacker_reward=attacker_reward)

        parsed = struct.unpack_from("<f??BBffBB", frame, 0)
        parsed_reward = float(parsed[0])
        assert parsed_reward == pytest.approx(attacker_reward, abs=1e-5), (
            f"Attacker reward at offset 0 mismatch: expected {attacker_reward}, got {parsed_reward}"
        )

    def test_defender_reward_at_offset_18(self):
        """Defender reward must be read from offset 18 in a rondo frame."""
        defender_reward = -0.75
        frame = _build_rondo_frame(defender_reward=defender_reward)

        parsed_defender = struct.unpack_from("<f", frame, 18)[0]
        assert parsed_defender == pytest.approx(defender_reward, abs=1e-5), (
            f"Defender reward at offset 18 mismatch: expected {defender_reward}, got {parsed_defender}"
        )

    def test_attacker_reward_does_not_contaminate_offset_18(self):
        """A distinct attacker reward at offset 0 must NOT appear at offset 18."""
        attacker_reward = 3.14159
        defender_reward = -2.71828
        frame = _build_rondo_frame(
            attacker_reward=attacker_reward,
            defender_reward=defender_reward,
        )

        parsed_attacker = struct.unpack_from("<f", frame, 0)[0]
        parsed_defender = struct.unpack_from("<f", frame, 18)[0]

        assert parsed_attacker == pytest.approx(attacker_reward, abs=1e-5)
        assert parsed_defender == pytest.approx(defender_reward, abs=1e-5)
        # Cross-contamination guard: the two floats must differ
        assert abs(parsed_attacker - parsed_defender) > 1e-4

    def test_defender_reward_does_not_contaminate_offset_0(self):
        """A distinct defender reward at offset 18 must NOT appear at offset 0."""
        attacker_reward = 0.5
        defender_reward = -3.0
        frame = _build_rondo_frame(
            attacker_reward=attacker_reward,
            defender_reward=defender_reward,
        )

        parsed_attacker = struct.unpack_from("<f", frame, 0)[0]
        parsed_defender = struct.unpack_from("<f", frame, 18)[0]

        assert parsed_attacker == pytest.approx(attacker_reward, abs=1e-5)
        assert parsed_defender == pytest.approx(defender_reward, abs=1e-5)
        assert abs(parsed_attacker - parsed_defender) > 1e-4

    def test_rondo_header_length_is_22_plus_obs_plus_mask(self):
        """Total rondo frame length must be 22 + OBSERVATION_DIM*4 + 19."""
        frame = _build_rondo_frame()
        expected_len = 22 + OBSERVATION_DIM * 4 + 19
        assert len(frame) == expected_len

    def test_zero_rewards_do_not_cross_contaminate(self):
        """Zero rewards at both offsets must remain distinct and zero."""
        frame = _build_rondo_frame(attacker_reward=0.0, defender_reward=0.0)
        parsed_attacker = struct.unpack_from("<f", frame, 0)[0]
        parsed_defender = struct.unpack_from("<f", frame, 18)[0]
        assert parsed_attacker == 0.0
        assert parsed_defender == 0.0

    def test_nan_reward_propagates_through_parser(self):
        """NaN in the reward field must survive struct unpacking as NaN."""
        frame = _build_rondo_frame(attacker_reward=float("nan"), defender_reward=0.0)
        parsed_attacker = struct.unpack_from("<f", frame, 0)[0]
        assert np.isnan(parsed_attacker)

    def test_extreme_float_values_survive_round_trip(self):
        """Very large and very small float32 values must survive packing/unpacking."""
        for val in [1e30, -1e30, 1e-30, -1e-30, 65535.0, -65535.0]:
            frame = _build_rondo_frame(attacker_reward=val, defender_reward=-val)
            parsed_attacker = struct.unpack_from("<f", frame, 0)[0]
            parsed_defender = struct.unpack_from("<f", frame, 18)[0]
            # float32 precision: use rel tolerance for large magnitudes
            assert parsed_attacker == pytest.approx(val, rel=1e-6)
            assert parsed_defender == pytest.approx(-val, rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
