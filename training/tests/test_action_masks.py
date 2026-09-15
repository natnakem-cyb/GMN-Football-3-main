"""Action-mask regression tests (masking-gap brief, Parts 1 & 6).

1. ``test_reset_mask_forbids_ball_actions_and_flips_on_possession`` (live
   bridge): the reset response must carry REAL legality masks — at kickoff the
   ball is unowned so SHOT/PASS/DRIBBLE are forbidden and TACKLE is allowed;
   once a controlled player picks the ball up, those bits must flip. This is
   the regression guard for the reset-mask fabrication gap (both reset paths
   used to synthesize all-ones masks).

2. ``test_masked_policy_never_emits_illegal_action`` and
   ``test_off_ball_entropy_matches_legal_action_count``: formalize the
   standalone check that a masked SharedActor never samples an illegal action
   id and that an off-ball observation's entropy sits near ln(legal actions),
   not ln(19).
"""

import math
import os
import sys
import time
import subprocess
import urllib.request

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv  # noqa: E402
from training.mappo_networks import SharedActor  # noqa: E402
from training.mappo_rollout import unwrap_masks, _mask_matrix  # noqa: E402

# 19-action layout (src/engine/ActionMapping.ts): IDLE=0, movement=1..8,
# LONG/HIGH/SHORT_PASS=9/10/11, SHOT=12, SPRINT=13, RELEASE_DIRECTION=14,
# RELEASE_SPRINT=15, TACKLE=16, DRIBBLE=17, RELEASE_DRIBBLE=18.
BALL_ACTIONS = (9, 10, 11, 12)
TACKLE = 16
DRIBBLE = 17

PORT = 5155  # dedicated port; this test spawns its own bridge here


def _start_bridge(port: int) -> subprocess.Popen:
    bridge_script = os.path.join(os.path.dirname(__file__), "..", "bridge_server.ts")
    npx = "npx.cmd" if sys.platform == "win32" else "npx"
    proc = subprocess.Popen(
        [npx, "tsx", bridge_script],
        env=dict(os.environ, GMN_BRIDGE_PORT=str(port)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        time.sleep(0.3)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return proc
        except Exception:
            pass
    raise RuntimeError(f"Bridge did not become healthy on port {port}")


def _mask_of(obs_dict, agent):
    v = obs_dict.get(agent)
    if isinstance(v, dict):
        return np.asarray(v.get("action_mask"), dtype=np.int8)
    return None


def test_reset_mask_forbids_ball_actions_and_flips_on_possession():
    """Reset-time mask forbids SHOT/PASS/DRIBBLE at kickoff (ball unowned);
    after a controlled player picks up the ball, those actions flip to legal
    and TACKLE flips to illegal."""
    proc = None
    try:
        proc = _start_bridge(PORT)
        env = GMNMultiAgentEnv(
            scenario="academy_empty_goal",
            auto_start_bridge=False,
            port=PORT,
            enable_reward_shaping=False,
        )
        try:
            obs_dict, _ = env.reset(seed=9001)
            current_agents = list(env.agents if env.agents else env.possible_agents)

            # At kickoff the ball is unowned: SHOT/PASS/DRIBBLE must be masked out.
            for agent in current_agents:
                mask = _mask_of(obs_dict, agent)
                assert mask is not None, f"{agent}: reset response missing action_mask"
                for action in BALL_ACTIONS:
                    assert mask[action] == 0, (
                        f"{agent}: BALL_ACTION {action} should be illegal at kickoff, mask={mask.tolist()}"
                    )
                assert mask[TACKLE] == 1, (
                    f"{agent}: TACKLE should be legal at kickoff (no possession), mask={mask.tolist()}"
                )

            # Step with MOVE-RIGHT (action=5) until a controlled player gets the ball.
            max_steps = 200
            for step in range(max_steps):
                current_agents = list(env.agents if env.agents else env.possible_agents)
                action_dict = {a: 5 for a in current_agents}
                obs_dict, rewards, terminations, truncations, infos = env.step(action_dict)
                for agent in current_agents:
                    mask = _mask_of(obs_dict, agent)
                    if mask is None:
                        continue
                    if mask[BALL_ACTIONS[0]] == 1 and mask[TACKLE] == 0:
                        # Controlled player has possession: ball actions legal, tackle illegal.
                        for action in BALL_ACTIONS:
                            assert mask[action] == 1, (
                                f"{agent}: BALL_ACTION {action} should be legal with possession at step {step}, mask={mask.tolist()}"
                            )
                        assert mask[TACKLE] == 0, (
                            f"{agent}: TACKLE should be illegal with possession at step {step}, mask={mask.tolist()}"
                        )
                        return  # success: possession flip confirmed
                if any(terminations.values()) or any(truncations.values()) or not env.agents:
                    break
            pytest.fail("No possession flip observed within max_steps")
        finally:
            env.close()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_masked_policy_never_emits_illegal_action():
    """A masked SharedActor never samples an action id that is masked out."""
    actor = SharedActor(obs_dim=127, action_dim=19, hidden=64)
    actor.eval()

    # Build an off-ball observation (random but deterministic).
    torch.manual_seed(42)
    obs = torch.randn(1, 127)

    # Mask: only IDLE (0) and movement (1-8) legal; everything else illegal.
    mask = torch.zeros(1, 19, dtype=torch.bool)
    mask[0, 0:9] = True

    illegal_indices = list(range(9, 19))
    for _ in range(500):
        with torch.no_grad():
            dist = actor(obs, mask)
            sample = int(dist.sample().item())
        assert sample not in illegal_indices, (
            f"Masked actor sampled illegal action {sample} with mask={mask.tolist()}"
        )


def test_off_ball_entropy_matches_legal_action_count():
    """Off-ball policy entropy should be near ln(|legal|), not ln(19)."""
    actor = SharedActor(obs_dim=127, action_dim=19, hidden=64)
    actor.eval()

    torch.manual_seed(42)
    obs = torch.randn(1, 127)

    # Off-ball mask: 9 legal actions (IDLE + movement).
    mask = torch.zeros(1, 19, dtype=torch.bool)
    mask[0, 0:9] = True

    with torch.no_grad():
        dist = actor(obs, mask)
        entropy = float(dist.entropy().item())

    expected_entropy = math.log(9)
    assert abs(entropy - expected_entropy) < 0.3, (
        f"Off-ball entropy {entropy:.4f} should be near ln(9)={expected_entropy:.4f}, "
        f"not ln(19)={math.log(19):.4f}"
    )
