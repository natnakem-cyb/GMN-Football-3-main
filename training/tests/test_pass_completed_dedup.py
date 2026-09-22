"""Regression tests for P0 PASS_COMPLETED deduplication.

A single genuine pass completion currently produces two PASS_COMPLETED
events in step_events (one from GameEngine, one from _resolve_pending_pass_state).
These tests verify that _canonicalize_pass_events collapses them to exactly one.
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.reward_adapters import AttackingDrillRewardAdapter


def test_canonicalize_collapses_passer_and_receiver_events_same_tick():
    """Two PASS_COMPLETED events on the same tick (passer + receiver) collapse to one."""
    events = [
        {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"},
        {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"},
    ]
    result = GMNMultiAgentEnv._canonicalize_pass_events(
        events, {"ep_len": 42, "pending_pass": None}
    )
    survived = [e for e in result if e.get("type") == "PASS_COMPLETED"]
    assert len(survived) == 1


def test_canonicalize_preserves_single_event():
    """A single PASS_COMPLETED event passes through unchanged."""
    events = [
        {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"},
    ]
    result = GMNMultiAgentEnv._canonicalize_pass_events(
        events, {"ep_len": 42, "pending_pass": None}
    )
    survived = [e for e in result if e.get("type") == "PASS_COMPLETED"]
    assert len(survived) == 1
    assert survived[0]["agent_id"] == "left_0"


def test_canonicalize_preserves_non_pass_events():
    """Non-PASS_COMPLETED events are passed through unchanged."""
    events = [
        {"type": "GOAL_SCORED", "team": "left", "agent_id": "left_0"},
        {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"},
        {"type": "SHOT_TAKEN", "team": "left", "agent_id": "left_2"},
    ]
    result = GMNMultiAgentEnv._canonicalize_pass_events(
        events, {"ep_len": 42, "pending_pass": None}
    )
    assert len(result) == 3
    assert result[0]["type"] == "GOAL_SCORED"
    assert result[1]["type"] == "PASS_COMPLETED"
    assert result[2]["type"] == "SHOT_TAKEN"


def test_canonicalize_different_ticks_not_collapsed():
    """PASS_COMPLETED events on different ticks are NOT collapsed."""
    events_tick_42 = [
        {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"},
    ]
    result_42 = GMNMultiAgentEnv._canonicalize_pass_events(
        events_tick_42, {"ep_len": 42, "pending_pass": None}
    )
    assert len([e for e in result_42 if e["type"] == "PASS_COMPLETED"]) == 1

    events_tick_43 = [
        {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"},
    ]
    result_43 = GMNMultiAgentEnv._canonicalize_pass_events(
        events_tick_43, {"ep_len": 43, "pending_pass": None}
    )
    assert len([e for e in result_43 if e["type"] == "PASS_COMPLETED"]) == 1


def test_end_to_end_payment_once_via_adapter():
    """End-to-end: canonicalized events produce exactly one payment via AttackingDrillRewardAdapter."""
    canonical_events = [
        {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"},
    ]
    adapter = AttackingDrillRewardAdapter(enable_exploration_bonus=False)
    adapter.reset()
    base = {"left_0": 0.0, "left_1": 0.0}
    gt = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
    out = adapter.compute_shaped_rewards(
        base, canonical_events, gt, ["left_0", "left_1"]
    )
    assert adapter.pass_completed_count == 1
    assert adapter.total_pass_completed_count == 1
    # Single PASS_COMPLETED: pass reward (+0.10) + dense possession (+0.01) - step cost (-0.005) = +0.105
    assert out["left_0"] == pytest.approx(0.105)
    # Non-event agent receives only dense possession - step cost = +0.005
    assert out["left_1"] == pytest.approx(0.005)


def test_two_different_legitimate_passes_yield_two_events():
    """Two distinct same-tick passes with pending_pass present yield two events.

    When pending_pass is present (a new pass was just initiated), multiple
    PASS_COMPLETED events on the same tick represent different physical passes,
    not duplicate detections of the same pass. The pending-pass-aware key
    (tick, passer_id, receiver_id, "PASS_COMPLETED") preserves all distinct
    (passer, receiver) pairs.
    """
    events = [
        {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"},
        {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_2"},
    ]
    result = GMNMultiAgentEnv._canonicalize_pass_events(
        events, {"ep_len": 42, "pending_pass": {"agent_id": "left_0"}}
    )
    survived = [e for e in result if e.get("type") == "PASS_COMPLETED"]
    assert len(survived) == 2
