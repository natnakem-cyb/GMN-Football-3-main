"""
GMN-Football-3 - Regression tests for the gmn_gym WebSocket step-frame contract.

Background (root cause of the train_ppo first-step hang):
bridge_server.ts encodeStepBinary appends MASK_BYTES (19) action-mask bytes
after the observations, so a non-rondo single-agent step reply is
18 + 127*4 + 19 = 545 bytes (rondo: 22 + 127*4 + 19 = 549). GMNFootballEnv.step
still expected only header + observations (526 / 530) and silently skipped the
larger reply as an "unsolicited broadcast", so the next recv timed out:

    [GMN-Gym WS Timeout] No frame received within Ns during 'step'

These tests pin the accepted length set and drive the real step() code path
(fake WebSocket client, no bridge child process) against synthetic frames.
"""

import json
import os
import struct
import sys

import numpy as np
import pytest
from gymnasium import spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from training.gmn_gym import (  # noqa: E402
    ACTION_SPACE_SIZE,
    MASK_BYTES,
    OBSERVATION_DIM,
    GMNFootballEnv,
    accepted_step_frame_lengths,
)

RONDO_SCENARIO = "academy_rondo_4v1"


class _FakeWsClient:
    """Minimal stand-in for websockets.sync.client.connect(...)."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def recv(self, timeout=None):
        if not self.frames:
            raise TimeoutError("fake ws: no frame available")
        return self.frames.pop(0)

    def close(self):
        pass


def _make_env(frames, scenario="academy_empty_goal"):
    """Build a GMNFootballEnv with a fake transport and no bridge process."""
    env = object.__new__(GMNFootballEnv)
    env.scenario = scenario
    env.use_ws = True
    env.ws_client = _FakeWsClient(frames)
    env.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
    env._step_count = 0
    env.ws_recv_timeout = 0.1
    return env


def _build_step_frame(
    reward=1.25,
    terminated=False,
    truncated=False,
    score=(2, 1),
    checkpoint=0.5,
    dist_goal=12.0,
    event_code=0,
    ball_owner=0,
    scenario="academy_empty_goal",
    include_mask=True,
    obs_value=0.5,
):
    """Encode a single-agent step frame exactly like encodeStepBinary does."""
    header_size = 22 if scenario == RONDO_SCENARIO else 18
    total = header_size + OBSERVATION_DIM * 4 + (MASK_BYTES if include_mask else 0)
    buf = bytearray(total)

    struct.pack_into("<f", buf, 0, reward)
    buf[4] = 1 if terminated else 0
    buf[5] = 1 if truncated else 0
    buf[6] = score[0]
    buf[7] = score[1]
    struct.pack_into("<f", buf, 8, checkpoint)
    struct.pack_into("<f", buf, 12, dist_goal)
    buf[16] = event_code
    buf[17] = ball_owner

    for i in range(OBSERVATION_DIM):
        struct.pack_into("<f", buf, header_size + i * 4, obs_value + i)

    if include_mask:
        mask_offset = header_size + OBSERVATION_DIM * 4
        for i in range(MASK_BYTES):
            buf[mask_offset + i] = 1

    return bytes(buf)


def test_accepted_lengths_match_bridge_contract():
    """The accepted set must contain both current (masked) and legacy lengths."""
    assert MASK_BYTES == 19
    assert OBSERVATION_DIM == 127
    assert accepted_step_frame_lengths("academy_empty_goal") == {526, 545}
    assert accepted_step_frame_lengths(RONDO_SCENARIO) == {530, 549}


def test_step_accepts_masked_545_byte_frame():
    """Regression: the current 545-byte reply must be consumed, not skipped."""
    frame = _build_step_frame(reward=1.25, score=(2, 1), obs_value=0.5)
    assert len(frame) == 545

    env = _make_env([frame])
    obs, reward, terminated, truncated, info = env.step(3)

    assert obs.shape == (OBSERVATION_DIM,)
    assert obs.dtype == np.float32
    assert float(reward) == pytest.approx(1.25)
    assert terminated is False and truncated is False
    assert info["score"] == {"left": 2, "right": 1}
    assert float(info["checkpointReward"]) == pytest.approx(0.5)
    assert float(info["ballDistanceToGoal"]) == pytest.approx(12.0)
    # Observation parsing must start at header_size and ignore the mask bytes.
    assert float(obs[0]) == pytest.approx(0.5)
    assert float(obs[-1]) == pytest.approx(0.5 + OBSERVATION_DIM - 1)


def test_step_accepts_legacy_unmasked_526_byte_frame():
    """Backward compatibility with bridges predating the MASK_BYTES change."""
    frame = _build_step_frame(include_mask=False)
    assert len(frame) == 526

    env = _make_env([frame])
    obs, _, _, _, _ = env.step(0)
    assert obs.shape == (OBSERVATION_DIM,)


def test_step_accepts_masked_rondo_frame():
    frame = _build_step_frame(scenario=RONDO_SCENARIO)
    assert len(frame) == 549

    env = _make_env([frame], scenario=RONDO_SCENARIO)
    obs, _, _, _, _ = env.step(0)
    assert obs.shape == (OBSERVATION_DIM,)


def test_step_skips_broadcast_then_accepts_step_frame():
    """Unrecognised frames are still skipped; a valid step frame then wins."""
    broadcast = json.dumps({"type": "training_status", "step": 1}).encode("utf-8")
    frame = _build_step_frame()

    env = _make_env([broadcast, frame])
    obs, _, _, _, _ = env.step(0)
    assert obs.shape == (OBSERVATION_DIM,)


def test_step_raises_when_only_broadcasts():
    """60 broadcasts exhaust the loop with an actionable error message."""
    frames = [json.dumps({"type": "telemetry", "i": i}).encode("utf-8") for i in range(60)]

    env = _make_env(frames)
    with pytest.raises(RuntimeError, match="got only broadcast frames"):
        env.step(0)


def test_error_frame_sentinel_detected_on_masked_frame():
    """P0 #5: the -999 error frame sentinel must also work at 545 bytes."""
    frame = _build_step_frame(reward=-999.0, terminated=True)
    assert len(frame) == 545

    env = _make_env([frame])
    with pytest.raises(RuntimeError, match="Bridge returned an error frame"):
        env.step(0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
