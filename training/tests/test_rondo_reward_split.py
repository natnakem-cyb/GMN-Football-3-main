"""
Tests for the rondo (academy_rondo_4v1) reward-split fix.
"""

import os
import sys
import struct
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv, CooperativeRewardShaper, OBSERVATION_DIM

PASS_BONUS = 0.1
POSSESSION_TICK = 0.01
DEFENDER_POSSESSION = 0.2


def test_cooperative_shaper_excludes_right_team():
    """
    CooperativeRewardShaper must not shape rewards for right-team agents.
    """
    shaper = CooperativeRewardShaper()
    base_rewards = {"left_1": 0.1, "left_2": 0.0, "right_1": 0.0}
    step_events = [
        {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"},
    ]

    shaped = shaper.compute_shaped_rewards(
        base_rewards=base_rewards,
        step_events=step_events,
        info_ground_truth={"current_ball_owner": {"agent_id": "left_1", "team": "left"}},
        active_agents=["left_1", "left_2", "right_1"],
    )

    assert "right_1" not in shaped, "Right-team agent must not appear in shaped rewards"
    assert shaped["left_1"] == base_rewards["left_1"] + shaper.r_pass


def test_rondo_rewards_diverge_over_episode():
    """
    Over a short episode, attacker rewards must sometimes differ from defender rewards.
    This proves the split-reward path is active and not silently falling back.
    """
    port = 5054 + hash("rondo_diverge_test2") % 1000
    env = GMNMultiAgentEnv(
        scenario="academy_rondo_4v1",
        auto_start_bridge=True,
        port=port,
        enable_reward_shaping=False,
    )
    try:
        obs, info = env.reset(seed=123)
        attackers = [a for a in env.agents if a.startswith("left_")]
        defenders = [a for a in env.agents if a.startswith("right_")]

        diverged = False
        for step_idx in range(30):
            action_dict = {a: 0 for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(action_dict)

            # Check if any attacker reward differs from the defender reward this tick
            attacker_avg = np.mean([rewards[a] for a in attackers])
            defender_rew = rewards[defenders[0]]

            if abs(attacker_avg - defender_rew) > 1e-6:
                diverged = True
                break

            if any(terms.values()) or any(truncs.values()) or not env.agents:
                obs, info = env.reset(seed=123 + step_idx)

        assert diverged, (
            "Attacker and defender rewards never diverged over 30 steps; "
            "the split-reward path may not be active."
        )
    finally:
        env.close()


def test_defender_never_gets_pass_bonus():
    """
    The +0.1 pass-completion bonus is attacker-specific. The defender must never
    receive it. We verify by checking that defender rewards never equal the
    exact pass-bonus value when a pass event occurs.
    """
    port = 5054 + hash("rondo_defender_test2") % 1000
    env = GMNMultiAgentEnv(
        scenario="academy_rondo_4v1",
        auto_start_bridge=True,
        port=port,
        enable_reward_shaping=False,
    )
    try:
        obs, info = env.reset(seed=42)
        defenders = [a for a in env.agents if a.startswith("right_")]

        for step_idx in range(50):
            action_dict = {a: 0 for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(action_dict)

            # If the defender reward equals the pass bonus, that's a bug
            defender_rew = rewards[defenders[0]]
            assert defender_rew != PASS_BONUS, (
                f"Defender received pass-bonus reward {PASS_BONUS} at step {step_idx}; "
                "defender should not get attacker-specific shaping."
            )

            if any(terms.values()) or any(truncs.values()) or not env.agents:
                obs, info = env.reset(seed=42 + step_idx)
    finally:
        env.close()


if __name__ == "__main__":
    test_cooperative_shaper_excludes_right_team()
    print("[TEST] PASS: CooperativeRewardShaper excludes right team")

    test_rondo_rewards_diverge_over_episode()
    print("[TEST] PASS: Rondo rewards diverge over episode")

    test_defender_never_gets_pass_bonus()
    print("[TEST] PASS: Defender never gets pass bonus")

    print("[TEST] ALL PASSED")
