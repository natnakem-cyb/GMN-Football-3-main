"""GMN-Football-3 — Reward adapter persistence regression tests (bridge-free).

Guards the count-based exploration bonus's RUN-SCOPED state
(``AttackingDrillRewardAdapter._visit_counts``) against the two batched env
paths that used to throw it away:

  * ``reset_batch()`` -> ``_init_batch_envs()``: called at the top of every
    ``collect_rollout_batched()`` iteration (mappo_rollout.py:583), i.e. every
    ``n_steps`` (256) ticks per env.
  * ``reset_one()``: the per-episode terminal reset (mappo_rollout.py:776).

Both sites constructed a brand-new adapter, so with ``--n-envs > 1`` every
rollout (and every episode) looked novel again and ``beta / sqrt(1 + count)``
degenerated into a flat bonus. ``_scenario_adapter()`` now reuses the adapter
per env index when the scenario still maps to the same adapter class, calling
``reset()`` (which clears per-episode state but deliberately keeps counts) —
mirroring the single-env ``env.reset() -> reward_adapter.reset()`` path.

These tests need no WebSocket bridge and no subprocess: the env is built with
``auto_start_bridge=False``, ``_ensure_bridge_running`` is stubbed, and the
batch wire calls are stubbed with a fake ``ws_client``.
"""

import json
import logging
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.reward_adapters import (
    AttackingDrillRewardAdapter,
    RondoRewardAdapter,
)

SCENARIO = "academy_3_vs_1_with_keeper"
SAME_FAMILY_SCENARIO = "academy_3_vs_1_defender_2"
RONDO_SCENARIO = "academy_rondo_4v1"
NO_ADAPTER_SCENARIO = "5_vs_5"

VISITED_CELL = AttackingDrillRewardAdapter._cell_for_position(0.05, 0.05)


def _make_env(monkeypatch, scenario=SCENARIO, batch_size=2, enable_reward_shaping=True, **kwargs):
    """Build a fully offline env: no bridge subprocess, HTTP, or WebSocket."""
    monkeypatch.setattr(GMNMultiAgentEnv, "_ensure_bridge_running", lambda self: None)
    env = GMNMultiAgentEnv(
        scenario=scenario,
        auto_start_bridge=False,
        enable_reward_shaping=enable_reward_shaping,
        batch_size=batch_size,
        **kwargs,
    )
    # Adapter lifecycle is under test, not frame decoding, so the reset
    # response decoder is stubbed to a minimal per-env state envelope.
    env._decode_reset_result = lambda result: {
        "obs_dict": {},
        "info": {},
        "agents": ["left_0"],
        "possible_agents": ["left_0"],
        "agent_order": ["left_0"],
        "ep_len": 0,
        "ep_rew": 0.0,
        "obs_dim": 127,
    }
    # reset_one()/reset() send over the socket; only send() is exercised here.
    env.ws_client = types.SimpleNamespace(send=lambda payload: None)
    return env


def test_reset_batch_reuses_adapter_and_keeps_visit_counts(monkeypatch):
    """G1: reset_batch() must not wipe run-scoped visit counts.

    reset_batch() runs at the top of every batched rollout, so rebuilding the
    adapter there makes the 1/sqrt(1+count) decay effectively flat.
    """
    assert VISITED_CELL is not None
    env = _make_env(monkeypatch, batch_size=2)
    env._init_batch_envs([{}, {}])
    first = [state["reward_adapter"] for state in env._batch_envs]
    assert all(isinstance(adapter, AttackingDrillRewardAdapter) for adapter in first)

    for env_idx, adapter in enumerate(first):
        adapter._visit_counts[VISITED_CELL] = 3 + env_idx
        adapter.ticks_no_shot = 7
        adapter.pass_completed_count = 5

    # Second rollout boundary: same call, same scenario.
    env._init_batch_envs([{}, {}])

    for env_idx, adapter in enumerate(first):
        reused = env._batch_envs[env_idx]["reward_adapter"]
        assert reused is adapter, (
            "reset_batch() rebuilt the reward adapter; run-scoped exploration "
            "visit counts were wiped at a rollout boundary"
        )
        assert reused._visit_counts[VISITED_CELL] == 3 + env_idx, (
            "visit counts were wiped by reset_batch()"
        )
        # reset() must still clear per-episode state (but keep the counts).
        assert reused.ticks_no_shot == 0
        assert reused.pass_completed_count == 0


def test_reset_one_reuses_adapter_and_keeps_visit_counts(monkeypatch):
    """G1: the per-episode terminal reset must keep run-scoped visit counts."""
    env = _make_env(monkeypatch, batch_size=2)
    env._init_batch_envs([{}, {}])
    original = env._batch_envs[0]["reward_adapter"]
    original._visit_counts[VISITED_CELL] = 4
    original.ticks_no_shot = 9

    env._recv_reset_one_response = lambda: json.dumps({
        "type": "reset_one_result",
        "result": {
            "observations": [[0.0] * 127],
            "info": {"controllableAgentIds": ["left_0"]},
        },
    })
    env.reset_one(0, seed=1001)

    reusing = env._batch_envs[0]["reward_adapter"]
    assert reusing is original, (
        "reset_one() rebuilt the reward adapter; visit counts restarted every episode"
    )
    assert reusing._visit_counts[VISITED_CELL] == 4
    assert reusing.ticks_no_shot == 0
    # The sibling sub-env keeps its own adapter — no cross-env leakage.
    assert env._batch_envs[1]["reward_adapter"] is not original


def test_batch_visit_counts_are_isolated_per_env_index(monkeypatch):
    """Each sub-env owns its adapter (per-env novelty budget)."""
    env = _make_env(monkeypatch, batch_size=2)
    env._init_batch_envs([{}, {}])
    adapter_0 = env._batch_envs[0]["reward_adapter"]
    adapter_1 = env._batch_envs[1]["reward_adapter"]
    assert adapter_0 is not adapter_1
    adapter_0._visit_counts[VISITED_CELL] = 1
    assert VISITED_CELL not in adapter_1._visit_counts


def test_batch_reset_builds_fresh_adapter_across_adapter_class_change(monkeypatch):
    """A scenario change that maps to a DIFFERENT adapter class must not reuse."""
    env = _make_env(monkeypatch, batch_size=2)
    env._init_batch_envs([{}, {}])
    attacking = env._batch_envs[0]["reward_adapter"]
    assert isinstance(attacking, AttackingDrillRewardAdapter)

    env.scenario = RONDO_SCENARIO
    env._init_batch_envs([{}, {}])

    rondo = env._batch_envs[0]["reward_adapter"]
    assert isinstance(rondo, RondoRewardAdapter), (
        "cross-class scenario change reused the wrong adapter class"
    )
    assert rondo is not attacking
    # The rondo adapter has no exploration grid; nothing was leaked into it.
    assert getattr(rondo, "_visit_counts", {}) == {}


def test_set_scenario_same_family_keeps_visit_counts(monkeypatch):
    """Curriculum promotions inside one adapter class must keep novelty state."""
    env = _make_env(monkeypatch, batch_size=1)
    adapter = env.reward_adapter
    assert isinstance(adapter, AttackingDrillRewardAdapter)
    adapter._visit_counts[VISITED_CELL] = 5
    adapter.ticks_no_shot = 11

    env.set_scenario(SAME_FAMILY_SCENARIO)

    assert env.reward_adapter is adapter, (
        "set_scenario() rebuilt the adapter for a same-class scenario change"
    )
    assert env.reward_adapter._visit_counts[VISITED_CELL] == 5
    assert env.reward_adapter.ticks_no_shot == 0
    assert env.scenario == SAME_FAMILY_SCENARIO
    # A cross-class switch still replaces the adapter.
    env.set_scenario(RONDO_SCENARIO)
    assert isinstance(env.reward_adapter, RondoRewardAdapter)


def test_missing_adapter_logs_warning_and_degrades(monkeypatch, caplog):
    """G4: the factory's loud ValueError must not be swallowed silently.

    The env keeps the legacy-shaper fallback for scenarios without an adapter
    (full matches), but it must say so in the logs.
    """
    env = _make_env(monkeypatch, batch_size=1)
    env.scenario = NO_ADAPTER_SCENARIO
    with caplog.at_level(logging.WARNING):
        adapter = env._scenario_adapter()
    assert adapter is None
    assert any(
        NO_ADAPTER_SCENARIO in record.getMessage() for record in caplog.records
    ), "no warning was logged for a scenario with no reward adapter"


def test_shaping_disabled_yields_no_adapter(monkeypatch):
    """enable_reward_shaping=False must never construct an adapter."""
    env = _make_env(monkeypatch, batch_size=2, enable_reward_shaping=False)
    assert env.reward_adapter is None
    env._init_batch_envs([{}, {}])
    for state in env._batch_envs:
        assert state["reward_adapter"] is None


def test_exploration_bonus_not_paid_on_goal_tick():
    """G6: goal ticks are exempt from the exploration bonus, like the dense
    terms and the step cost (the count still increments: true visitation)."""
    beta = AttackingDrillRewardAdapter.EXPLORATION_BETA
    adapter = AttackingDrillRewardAdapter(max_hold=100000, p_hog=0.0, t_max=100000)
    adapter.reset()
    gt = {
        "current_ball_owner": {"team": "left", "agent_id": "left_0"},
        "ball_distance_to_goal": 1.0,  # constant => PBRS delta == 0
        "ball_x": 0.05,
        "ball_y": 0.05,
    }

    # Non-goal tick, first visit: step cost + dense possession + full beta.
    plain = adapter.compute_shaped_rewards({"left_0": 0.0}, [], gt, ["left_0"])
    expected_plain = (
        adapter.step_cost
        + AttackingDrillRewardAdapter.DENSE_POSSESSION_REWARD
        + beta
    )
    assert plain["left_0"] == pytest.approx(expected_plain, abs=1e-9)

    # Goal tick: base reward preserved whole, no exploration bonus, no step cost.
    goal_events = [{"type": "GOAL_SCORED", "team": "left", "agent_id": "left_0"}]
    scored = adapter.compute_shaped_rewards({"left_0": 2.0}, goal_events, gt, ["left_0"])
    assert scored["left_0"] == pytest.approx(2.0, abs=1e-6)

    # Visitation is still recorded on the goal tick.
    assert adapter._visit_counts[VISITED_CELL] == 2


def test_shot_clock_policy_is_wired_to_the_adapter(monkeypatch):
    """P2: shot_clock_truncates / shot_clock_t_max reach the adapter, and the
    defaults preserve the pre-existing eval semantics exactly.

    Regression guard for the training landmine: with the adapter default
    (t_max=50) the env truncated every episode at ~51 ticks, so a retrain never
    saw the 30 s drill. Training now passes shot_clock_truncates=False and a
    wider clock.
    """
    # Defaults: truncation ON (eval/committed-baseline semantics), no t_max
    # override, so the adapter keeps its own default of 50 ticks.
    env_default = _make_env(monkeypatch, batch_size=1)
    assert env_default.shot_clock_truncates is True
    assert env_default.shot_clock_t_max is None
    assert env_default.reward_adapter.t_max == 50

    # Training wiring: penalty only, episode not capped by the clock.
    env_train = _make_env(
        monkeypatch, batch_size=1, shot_clock_truncates=False, shot_clock_t_max=600
    )
    assert env_train.shot_clock_truncates is False
    assert env_train.reward_adapter.t_max == 600

    # The override must survive adapter reuse (reset_batch / scenario switch).
    env_train._init_batch_envs([{}, {}])
    assert env_train._batch_envs[0]["reward_adapter"].t_max == 600
    env_train.scenario = SAME_FAMILY_SCENARIO
    env_train._init_batch_envs([{}, {}])
    assert env_train._batch_envs[0]["reward_adapter"].t_max == 600
