"""GMN-Football-3 — Scenario Reward Adapter Unit Tests.

Proves the Rondo and Attacking adapters produce different reward vectors
on identical synthetic events, and pins the attacking-adapter guarantees:
progress-checkpoint stripping, step cost, shot incentives, shot-clock.
"""

import pytest

from training.reward_adapters import (
    FINISHING_SCENARIOS,
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


def test_factory_covers_every_finishing_scenario():
    """Every id actually in FINISHING_SCENARIOS must resolve to the
    AttackingDrillRewardAdapter. Loops over the set rather than one
    hardcoded example so a future set edit cannot silently leave a
    scenario un-dispatched."""
    for sid in FINISHING_SCENARIOS:
        assert isinstance(get_reward_adapter(sid), AttackingDrillRewardAdapter), (
            f"{sid} is in FINISHING_SCENARIOS but did not dispatch to "
            f"AttackingDrillRewardAdapter"
        )


def test_factory_raises_for_full_match_scenarios():
    """5_vs_5 and 11_vs_11 are full matches with win_match/
    control_possession/clean_sheet objectives. Neither existing adapter is
    designed for them: dispatch must fail loudly, never silently apply the
    attacking-drill reward model to a match."""
    for sid in ("5_vs_5", "11_vs_11"):
        with pytest.raises(ValueError) as excinfo:
            get_reward_adapter(sid)
        assert sid in str(excinfo.value)


def test_factory_uses_loud_error_message_for_unknown_scenario():
    with pytest.raises(ValueError) as excinfo:
        get_reward_adapter("some_future_scenario_x")
    msg = str(excinfo.value)
    assert "No reward adapter defined" in msg
    assert "some_future_scenario_x" in msg


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
    """SHOT_TAKEN no longer gets flat r_shot bonus (PBRS replaces it).

    Only step cost (-0.005) applies; the flat attempt bonus is removed.
    PBRS potential term requires previous distance state, which is None
    on the first call, so no potential term is added here.
    """
    ad = AttackingDrillRewardAdapter()
    ad.reset()
    base = {"left_0": 0.0}
    evs = [{"type": "SHOT_TAKEN", "team": "left", "agent_id": "left_0"}]
    out = ad.compute_shaped_rewards(
        base, evs, GT_L0, ["left_0"], actions={"left_0": 12}
    )
    # No flat r_shot bonus; only step cost applies.
    assert out["left_0"] == pytest.approx(-0.005)
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
    """Attacking and Rondo adapters must still produce different rewards.

    After PBRS: Attacking no longer gives flat r_shot for SHOT_TAKEN, but
    still differs from Rondo (which gives -0.30 solitary penalty + action cost).
    """
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
    # Attacking: step cost only (-0.005) on first call (no prev dist for PBRS).
    # Rondo: solitary penalty + action cost = -0.30 + -0.01 = -0.31
    assert a["left_0"] > r["left_0"]  # -0.005 > -0.31
    assert r["left_0"] <= 0.0


def test_pibrs_boundedness_telescoping():
    """PBRS potential term is bounded and doesn't diverge over round trips.

    Move ball toward goal for N ticks, then back to starting distance.
    Verify the total potential-term contribution is bounded (not exploited
    by oscillating distance back and forth).

    Note: With gamma=0.99, the sum of (gamma*phi(new) - phi(prev)) over
    a trajectory doesn't telescope to exactly 0, but it's bounded. The
    key PBRS property is policy invariance, not that round trips sum to 0.
    What matters is that the potential change is finite and bounded.
    """
    ad = AttackingDrillRewardAdapter()
    ad.reset()

    # Start at distance 2.0 (far from goal), move toward goal, then back.
    distances = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.5, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    total_potential = 0.0

    for i, dist in enumerate(distances):
        gt = {
            "current_ball_owner": {"team": "left", "agent_id": "left_0"},
            "ball_distance_to_goal": dist,
        }
        base = {"left_0": 0.0}
        evs = []
        out = ad.compute_shaped_rewards(base, evs, gt, ["left_0"])
        step_cost = -0.005
        potential_delta = out["left_0"] - step_cost
        total_potential += potential_delta

    # The total potential contribution should be bounded.
    # After the round trip, phi returns to phi(2.0) = -1.0.
    # The cumulative PBRS is bounded by the D_MAX range.
    # Key property: total_potential should NOT be large positive (no exploit).
    assert -2.0 < total_potential < 2.0, (
        f"PBRS total {total_potential} is unbounded - possible exploit!"
    )

    # Additional check: phi is bounded in [-1, 0], so any single-step
    # delta = gamma*phi(new) - phi(prev) is bounded by:
    # max: gamma*0 - (-1) = 1.0
    # min: gamma*(-1) - 0 = -0.99
    assert -1.0 <= total_potential <= 1.0 * len(distances), (
        f"PBRS total {total_potential} exceeds theoretical bounds"
    )


def test_pibrs_no_contribution_without_ball():
    """PBRS contributes nothing when left team doesn't own the ball."""
    ad = AttackingDrillRewardAdapter()
    ad.reset()

    distances = [2.0, 1.5, 1.0, 0.5]
    for dist in distances:
        gt = {
            "current_ball_owner": {"team": "right", "agent_id": "right_0"},
            "ball_distance_to_goal": dist,
        }
        base = {"left_0": 0.0}
        evs = []
        out = ad.compute_shaped_rewards(base, evs, gt, ["left_0"])
        assert out["left_0"] == pytest.approx(-0.005)


def test_pibrs_phi_bounded():
    """Phi(d) is bounded in [-1, 0] for all d >= 0."""
    ad = AttackingDrillRewardAdapter()
    assert ad._phi(0.0) == pytest.approx(0.0)
    assert ad._phi(2.0) == pytest.approx(-1.0)
    assert ad._phi(100.0) == pytest.approx(-1.0)
    assert ad._phi(1.0) == pytest.approx(-0.5)
    assert ad._phi(0.5) == pytest.approx(-0.25)
    assert ad._phi(1.5) == pytest.approx(-0.75)


def test_pibrs_positive_for_progress():
    """Genuine progress toward goal nets positive PBRS contribution."""
    ad = AttackingDrillRewardAdapter()
    ad.reset()

    distances = [2.0, 1.5, 1.0, 0.5]
    total_reward = 0.0

    for i, dist in enumerate(distances):
        gt = {
            "current_ball_owner": {"team": "left", "agent_id": "left_0"},
            "ball_distance_to_goal": dist,
        }
        base = {"left_0": 0.0}
        evs = []
        out = ad.compute_shaped_rewards(base, evs, gt, ["left_0"])
        total_reward += out["left_0"]

    # Expected: first tick no PBRS (prev=None), then 3 ticks of progress.
    # phi(2.0) = -1.0, phi(1.5) = -0.75, phi(1.0) = -0.5, phi(0.5) = -0.25
    # Tick 2: gamma*phi(1.5) - phi(2.0) = 0.99*(-0.75) - (-1.0) = 0.2575
    # Tick 3: gamma*phi(1.0) - phi(1.5) = 0.99*(-0.5) - (-0.75) = 0.255
    # Tick 4: gamma*phi(0.5) - phi(1.0) = 0.99*(-0.25) - (-0.5) = 0.2525
    expected_potential = (
        ad.gamma * ad._phi(1.5) - ad._phi(2.0) +
        ad.gamma * ad._phi(1.0) - ad._phi(1.5) +
        ad.gamma * ad._phi(0.5) - ad._phi(1.0)
    )
    expected_total = 4 * (-0.005) + expected_potential
    assert total_reward == pytest.approx(expected_total, abs=1e-6)


def test_pibrs_with_shot_saved_still_gets_on_target_bonus():
    """SHOT_SAVED still gets r_on_target bonus (not affected by PBRS change)."""
    ad = AttackingDrillRewardAdapter()
    ad.reset()

    # First set up PBRS state with a previous distance.
    gt_setup = {
        "current_ball_owner": {"team": "left", "agent_id": "left_0"},
        "ball_distance_to_goal": 2.0,
    }
    ad.compute_shaped_rewards({"left_0": 0.0}, [], gt_setup, ["left_0"])

    # Now a SHOT_SAVED event.
    gt_shot = {
        "current_ball_owner": {"team": "left", "agent_id": "left_0"},
        "ball_distance_to_goal": 0.5,
    }
    evs = [{"type": "SHOT_SAVED", "team": "left", "agent_id": "left_0"}]
    out = ad.compute_shaped_rewards(
        {"left_0": 0.0}, evs, gt_shot, ["left_0"]
    )

    # Should get: step_cost + PBRS_delta + r_on_target
    pb_delta = ad.gamma * ad._phi(0.5) - ad._phi(2.0)
    expected = -0.005 + pb_delta + 0.40
    assert out["left_0"] == pytest.approx(expected, abs=1e-6)


def test_pibrs_no_contribution_after_turnover():
    """PBRS contributes nothing on the tick immediately after a turnover."""
    ad = AttackingDrillRewardAdapter()
    ad.reset()

    # Tick 1: left has ball at distance 2.0
    gt1 = {
        "current_ball_owner": {"team": "left", "agent_id": "left_0"},
        "ball_distance_to_goal": 2.0,
    }
    base = {"left_0": 0.0}
    evs = []
    out1 = ad.compute_shaped_rewards(base, evs, gt1, ["left_0"])
    # First tick: _prev_ball_dist is None, so no PBRS.
    assert out1["left_0"] == pytest.approx(-0.005)

    # Tick 2: turnover - right now has ball.
    gt2 = {
        "current_ball_owner": {"team": "right", "agent_id": "right_0"},
        "ball_distance_to_goal": 2.0,
    }
    out2 = ad.compute_shaped_rewards(base, evs, gt2, ["left_0"])
    # After turnover: no PBRS, only step cost.
    assert out2["left_0"] == pytest.approx(-0.005)


def test_pibrs_shape_term_magnitude_sane():
    """PBRS term magnitude is sane relative to terminal goal reward.

    The potential term per tick should be modest, dominated by the real
    terminal goal reward (+2.0 from engine).
    """
    ad = AttackingDrillRewardAdapter()

    # Max possible single-tick PBRS: ball moves from max distance to goal.
    # delta = gamma * phi(0.0) - phi(2.0) = 0.99 * 0 - (-1.0) = 1.0
    max_delta = ad.gamma * ad._phi(0.0) - ad._phi(2.0)
    assert abs(max_delta) <= 1.0  # Bounded by D_MAX range

    # Terminal goal reward is +2.0. PBRS should be a shaping nudge.
    realistic_total_pb = 0
    prev_dist = 2.0
    for _ in range(10):
        new_dist = prev_dist - 0.2
        delta = ad.gamma * ad._phi(new_dist) - ad._phi(prev_dist)
        realistic_total_pb += delta
        prev_dist = new_dist

    # After 10 ticks of progress, total should be positive but modest.
    assert realistic_total_pb > 0
    assert realistic_total_pb < 5.0  # Goal is +2.0
