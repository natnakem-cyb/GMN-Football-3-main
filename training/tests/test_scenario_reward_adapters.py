"""GMN-Football-3 — Scenario Reward Adapter Unit Tests.

Proves the Rondo and Attacking adapters produce different reward vectors
on identical synthetic events, and pins the attacking-adapter guarantees:
progress-checkpoint stripping, step cost, shot incentives, shot-clock.
"""

import pytest

from training.reward_adapters import (
    AttackingDrillRewardAdapter,
    RondoRewardAdapter,
    get_reward_adapter,
)

GT_NONE = {"current_ball_owner": None}
GT_L0 = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}


def test_factory_selects_adapter_by_scenario():
    assert isinstance(get_reward_adapter("academy_rondo_4v1"), RondoRewardAdapter)
    assert isinstance(
        get_reward_adapter("academy_3_vs_1_with_keeper"), AttackingDrillRewardAdapter
    )


def test_attacking_strips_progress_on_quiet_steps():
    ad = AttackingDrillRewardAdapter()
    ad.reset()
    base = {"left_0": 0.02, "left_1": 0.02}
    out = ad.compute_shaped_rewards(base, [], GT_NONE, ["left_0", "left_1"])
    assert out["left_0"] == pytest.approx(0.02 - 0.02 - 0.005)
    assert out["left_1"] == pytest.approx(0.02 - 0.02 - 0.005)


def test_attacking_preserves_goal_step_base():
    ad = AttackingDrillRewardAdapter()
    ad.reset()
    base = {"left_0": 2.0}
    evs = [{"type": "GOAL_SCORED", "team": "left", "agent_id": "left_0"}]
    out = ad.compute_shaped_rewards(base, evs, GT_L0, ["left_0"])
    # Goal step is exempt from step cost: the engine goal reward is kept whole.
    assert out["left_0"] == pytest.approx(2.0)


def test_attacking_pays_shot_taken_bonus():
    ad = AttackingDrillRewardAdapter()
    ad.reset()
    base = {"left_0": 0.0}
    evs = [{"type": "SHOT_TAKEN", "team": "left", "agent_id": "left_0"}]
    out = ad.compute_shaped_rewards(
        base, evs, GT_L0, ["left_0"], actions={"left_0": 12}
    )
    assert out["left_0"] == pytest.approx(0.25 - 0.005)
    assert ad.solitary_shot_count == 0


def test_attacking_no_action_cost_on_shot_or_pass():
    ad = AttackingDrillRewardAdapter()
    ad.reset()
    out = ad.compute_shaped_rewards(
        {"left_0": 0.0}, [], GT_NONE, ["left_0"], actions={"left_0": 12}
    )
    assert out["left_0"] == pytest.approx(-0.005)
    out2 = ad.compute_shaped_rewards(
        {"left_0": 0.0}, [], GT_NONE, ["left_0"], actions={"left_0": 9}
    )
    assert out2["left_0"] == pytest.approx(-0.005)


def test_attacking_step_cost_every_tick():
    ad = AttackingDrillRewardAdapter()
    ad.reset()
    out = ad.compute_shaped_rewards({"left_0": 0.0}, [], GT_NONE, ["left_0"])
    assert out["left_0"] == pytest.approx(-0.005)


def test_attacking_shot_clock_fires_after_tmax():
    ad = AttackingDrillRewardAdapter(t_max=50)
    ad.reset()
    for _ in range(51):
        ad.compute_shaped_rewards({"left_0": 0.0}, [], GT_NONE, ["left_0"])
    assert ad.check_shot_clock() == pytest.approx(-0.5)
    assert ad.timeout_count == 1


def test_attacking_shot_clock_timeout_penalty_is_applied_once():
    """The env contract: check_shot_clock returns the penalty exactly once,
    then None on every later call so the -0.5 cannot be double-charged even
    if env.step() keeps running while truncated is being set."""
    ad = AttackingDrillRewardAdapter(t_max=5)
    ad.reset()
    for _ in range(7):
        ad.compute_shaped_rewards({"left_0": 0.0}, [], GT_NONE, ["left_0"])
    assert ad.check_shot_clock() == pytest.approx(-0.5)
    assert ad.check_shot_clock() is None
    assert ad.check_shot_clock() is None
    assert ad.timeout_count == 1


def test_attacking_shot_clock_silenced_by_shot():
    ad = AttackingDrillRewardAdapter(t_max=50)
    ad.reset()
    ad.compute_shaped_rewards(
        {"left_0": 0.0},
        [{"type": "SHOT_TAKEN", "team": "left", "agent_id": "left_0"}],
        GT_L0,
        ["left_0"],
    )
    for _ in range(60):
        ad.compute_shaped_rewards({"left_0": 0.0}, [], GT_NONE, ["left_0"])
    assert ad.check_shot_clock() is None


def test_attacking_shot_saved_bonus_and_seen_shot():
    ad = AttackingDrillRewardAdapter()
    ad.reset()
    out = ad.compute_shaped_rewards(
        {"left_0": 0.0},
        [{"type": "SHOT_SAVED", "team": "left", "agent_id": "left_0"}],
        GT_NONE,
        ["left_0"],
    )
    assert out["left_0"] == pytest.approx(0.40 - 0.005)
    assert ad.check_shot_clock() is None or ad.seen_shot


def test_rondo_pays_legacy_penalty_for_shot_and_nothing_for_goal():
    rd = RondoRewardAdapter()
    rd.reset()
    shot = rd.compute_shaped_rewards(
        {"left_0": 0.0},
        [{"type": "SHOT_TAKEN", "team": "left", "agent_id": "left_0"}],
        GT_L0,
        ["left_0"],
        actions={"left_0": 12},
    )
    # Rondo retains the legacy shaper's solitary-shot penalty (-0.30) + the
    # -0.01 ball-action cost: shots are NOT the drill objective.
    assert shot["left_0"] == pytest.approx(-0.31)
    goal = rd.compute_shaped_rewards(
        {"left_0": 0.0},
        [{"type": "GOAL_SCORED", "team": "left", "agent_id": "left_0"}],
        GT_L0,
        ["left_0"],
    )
    # No pass chain -> no assisted-goal bonus; engine goal reward not present.
    assert goal["left_0"] == pytest.approx(0.0)


def test_rondo_keeps_pass_and_turnover_signals():
    rd = RondoRewardAdapter()
    rd.reset()
    out = rd.compute_shaped_rewards(
        {"left_0": 0.0},
        [{"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"}],
        GT_L0,
        ["left_0", "left_1"],
    )
    assert out["left_0"] == pytest.approx(0.30)
    out2 = rd.compute_shaped_rewards(
        {"left_0": 0.0},
        [{"type": "TURNOVER_CONCEDED", "team": "left", "agent_id": "left_0"}],
        GT_NONE,
        ["left_0"],
    )
    assert out2["left_0"] == pytest.approx(-0.10)


def test_adapters_differ_on_identical_shot_event():
    evs = [{"type": "SHOT_TAKEN", "team": "left", "agent_id": "left_0"}]
    ad = AttackingDrillRewardAdapter()
    ad.reset()
    rd = RondoRewardAdapter()
    rd.reset()
    a = ad.compute_shaped_rewards({"left_0": 0.0}, evs, GT_L0, ["left_0"])
    r = rd.compute_shaped_rewards(
        {"left_0": 0.0}, evs, GT_L0, ["left_0"], actions={"left_0": 5}
    )
    assert a["left_0"] != pytest.approx(r["left_0"])
    assert a["left_0"] > 0.0
    assert r["left_0"] <= 0.0
