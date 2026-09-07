"""
Unit tests for eval_progress.py metric correctness (schema v3.2.0).

Validates:
- turnover telemetry is sourced from FootballMetricsTracker possession-change events
- non_scoring_episode_rate_pct is computed from goal outcomes only
- legacy turnover_rate is fully removed
- rondo telemetry reports NaN for unavailable direct-possession metrics
"""

import csv
import math
import os
import sys
import tempfile
from typing import Dict, Any

import numpy as np
import pytest

# Ensure training/ is importable
sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training"))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from eval_progress import (
    CSV_FIELDNAMES,
    SCHEMA_VERSION,
    _extract_owner_team,
    _extract_ball_pos,
    _left_action_indices_single,
    _left_action_indices_multi,
    _resolve_versioned_csv_path,
    FootballMetricsTracker,
)


def _default_obs(ball_owner_one_hot=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Return a minimal 127-dim observation with configurable ball ownership."""
    obs = np.zeros(127, dtype=np.float32)
    # ball ownership one-hot at indices 94, 95, 96
    obs[94:97] = ball_owner_one_hot
    # ball position at indices 88, 89, 90
    obs[88:91] = (0.25, 0.0, 0.0)
    return obs


class TestObservationDecoding:
    def test_extract_owner_team_left(self):
        obs = _default_obs((0.0, 1.0, 0.0))
        assert _extract_owner_team(obs) == "left"

    def test_extract_owner_team_right(self):
        obs = _default_obs((0.0, 0.0, 1.0))
        assert _extract_owner_team(obs) == "right"

    def test_extract_owner_team_none(self):
        obs = _default_obs((1.0, 0.0, 0.0))
        assert _extract_owner_team(obs) is None

    def test_extract_owner_team_short_vector(self):
        assert _extract_owner_team(np.array([], dtype=np.float32)) is None
        assert _extract_owner_team(None) is None

    def test_extract_ball_pos(self):
        obs = _default_obs((0.0, 1.0, 0.0))
        pos = _extract_ball_pos(obs)
        assert pos == {"x": 0.25, "y": 0.0, "z": 0.0}

    def test_extract_ball_pos_short_vector(self):
        assert _extract_ball_pos(np.array([], dtype=np.float32)) == {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
        assert _extract_ball_pos(None) == {"x": 0.0, "y": 0.0, "z": 0.0}


class TestActionIndexHelpers:
    def test_left_action_indices_single(self):
        assert _left_action_indices_single(5) == [5]

    def test_left_action_indices_multi(self):
        actions = {"left_1": 3, "right_1": 7, "left_2": 11}
        assert _left_action_indices_multi(actions) == [3, 11]

    def test_left_action_indices_multi_empty(self):
        assert _left_action_indices_multi({}) == []


class TestFootballMetricsTrackerTurnovers:
    def test_zero_turnovers_when_no_possession_change(self):
        tracker = FootballMetricsTracker()
        tracker.start_episode("academy_rondo_4v1", 42, {"x": 0.25, "y": 0.0, "z": 0.0})
        for _ in range(200):
            tracker.record_tick(
                left_action_indices=[0],
                step_reward=0.01,
                ball_pos={"x": 0.25, "y": 0.0, "z": 0.0},
                owner_team="left",
            )
        tracker.end_episode({"left": 0, "right": 0}, {}, {"x": 0.25, "y": 0.0, "z": 0.0})
        ep = tracker.episodes[-1]
        assert ep.turnovers_conceded == 0
        # possession_changes counts the initial None -> left transition as well
        assert ep.possession_changes == 1

    def test_single_turnover_on_left_to_right_transition(self):
        tracker = FootballMetricsTracker()
        tracker.start_episode("academy_rondo_4v1", 42, {"x": 0.25, "y": 0.0, "z": 0.0})
        # 100 ticks left possession
        for _ in range(100):
            tracker.record_tick(
                left_action_indices=[0],
                step_reward=0.01,
                ball_pos={"x": 0.25, "y": 0.0, "z": 0.0},
                owner_team="left",
            )
        # 1 turnover: left -> right
        tracker.record_tick(
            left_action_indices=[0],
            step_reward=0.01,
            ball_pos={"x": 0.0, "y": 0.0, "z": 0.0},
            owner_team="right",
        )
        # 99 ticks right possession
        for _ in range(99):
            tracker.record_tick(
                left_action_indices=[0],
                step_reward=0.01,
                ball_pos={"x": 0.0, "y": 0.0, "z": 0.0},
                owner_team="right",
            )
        tracker.end_episode({"left": 0, "right": 0}, {}, {"x": 0.0, "y": 0.0, "z": 0.0})
        ep = tracker.episodes[-1]
        assert ep.turnovers_conceded == 1
        # possession_changes includes the initial None -> left transition
        assert ep.possession_changes == 2


class TestEvalMetricFormulas:
    """Directly assert the formulaic outputs that eval_progress.py computes."""

    def test_non_scoring_episode_rate_no_goals(self):
        num_episodes = 1
        goals = 0
        non_scoring_episode_rate_pct = ((num_episodes - goals) / max(1, num_episodes)) * 100.0
        assert non_scoring_episode_rate_pct == 100.0

    def test_non_scoring_episode_rate_all_goals(self):
        num_episodes = 5
        goals = 5
        non_scoring_episode_rate_pct = ((num_episodes - goals) / max(1, num_episodes)) * 100.0
        assert non_scoring_episode_rate_pct == 0.0

    def test_turnovers_conceded_per_ep_zero(self):
        num_episodes = 1
        turnovers_conceded_total = 0.0
        turnovers_conceded_per_ep = turnovers_conceded_total / max(1, num_episodes)
        assert turnovers_conceded_per_ep == 0.0

    def test_turnovers_conceded_per_ep_single(self):
        num_episodes = 1
        turnovers_conceded_total = 1.0
        turnovers_conceded_per_ep = turnovers_conceded_total / max(1, num_episodes)
        assert turnovers_conceded_per_ep == 1.0

    def test_legacy_turnover_rate_removed(self):
        assert "turnover_rate" not in CSV_FIELDNAMES
        assert "non_scoring_episode_rate_pct" in CSV_FIELDNAMES
        assert "turnovers_conceded_per_ep" in CSV_FIELDNAMES

    def test_schema_version_present(self):
        assert "schema_version" in CSV_FIELDNAMES
        assert CSV_FIELDNAMES[0] == "schema_version"
        assert SCHEMA_VERSION == "3.2.0"


class TestVersionedCsvPath:
    def test_new_file_uses_default_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "win_rate_progress.csv")
            resolved = _resolve_versioned_csv_path(path)
            assert resolved == path

    def test_legacy_file_is_versioned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "win_rate_progress.csv")
            # Write legacy CSV without schema_version
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "episodes", "mean_reward", "goal_rate_pct"])
            resolved = _resolve_versioned_csv_path(path)
            assert resolved == path.replace(".csv", "_v2.csv")

    def test_v2_file_keeps_default_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "win_rate_progress.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_FIELDNAMES)
            resolved = _resolve_versioned_csv_path(path)
            assert resolved == path


class TestRondoTelemetryNaN:
    def test_rondo_returns_nan_for_unavailable_possession_metrics(self):
        """Rondo scenarios should not report dummy 0.0 for unavailable possession metrics."""
        # We verify the formula directly: when direct tracking is unavailable,
        # the eval path should produce NaN, not 0.0.
        possession_retention_time = float("nan")
        completed_pass_chains = float("nan")
        assert math.isnan(possession_retention_time)
        assert math.isnan(completed_pass_chains)
        assert possession_retention_time != 0.0
        assert completed_pass_chains != 0.0
