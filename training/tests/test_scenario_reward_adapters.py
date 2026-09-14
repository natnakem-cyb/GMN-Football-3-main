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
    """PBRS potential term telescopes to exactly zero over round trips.

    With PBRS_GAMMA = 1.0, the shaping term gamma*phi(new) - phi(prev) is a
    proper potential-based reward shaping term: over any trajectory that returns
    to the starting state, the sum telescopes to exactly phi(final) - phi(initial).
    When final == initial (round trip), this is exactly zero.

    This test verifies the strong property: a round trip from distance 2.0 back
    to distance 2.0 yields total PBRS contribution of exactly 0 (within float
    epsilon), not just "bounded."

    Note: We use max_hold=1000 to avoid ball hogging penalty during the
    test, since the default 15 would trigger after 15 ticks.
    """
    ad = AttackingDrillRewardAdapter(max_hold=1000)
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

    # With PBRS_GAMMA = 1.0, the round trip telescopes to exactly zero.
    # phi(2.0) = -1.0, and we started and ended at distance 2.0.
    # Sum of (phi(s_{t+1}) - phi(s_t)) = phi(s_final) - phi(s_initial) = 0.
    EPSILON = 1e-9
    assert abs(total_potential) < EPSILON, (
        f"PBRS round trip should telescope to exactly 0, got {total_potential}"
    )


def test_pibrs_stationary_hold_yields_zero():
    """Stationary ball yields exactly zero PBRS per tick (no discounting leak).

    This is the critical test that catches the discounting leak bug:
    with gamma < 1.0 (e.g., 0.99), a stationary ball at any distance d
    would earn (gamma - 1) * phi(d) > 0 every tick because phi(d) < 0.
    With PBRS_GAMMA = 1.0, this leak is eliminated: phi(s) - phi(s) = 0.

    Test: hold ball at constant distance for 50+ ticks, verify total PBRS = 0.

    Note: We use max_hold=1000 to avoid ball hogging penalty during the
    test, since the default 15 would trigger after 15 ticks.
    """
    ad = AttackingDrillRewardAdapter(max_hold=1000)
    ad.reset()

    # Hold ball at distance 1.0 (phi = -0.5) for 50 ticks.
    # With the bug (gamma=0.99), each tick would earn:
    #   (0.99 - 1) * (-0.5) = +0.005 per tick, totaling +0.25 over 50 ticks.
    # With the fix (PBRS_GAMMA=1.0), each tick earns 0.
    distance = 1.0
    num_ticks = 50
    total_potential = 0.0

    for _ in range(num_ticks):
        gt = {
            "current_ball_owner": {"team": "left", "agent_id": "left_0"},
            "ball_distance_to_goal": distance,
        }
        base = {"left_0": 0.0}
        evs = []
        out = ad.compute_shaped_rewards(base, evs, gt, ["left_0"])
        step_cost = -0.005
        potential_delta = out["left_0"] - step_cost
        total_potential += potential_delta

    EPSILON = 1e-9
    assert abs(total_potential) < EPSILON, (
        f"Stationary ball at distance {distance} for {num_ticks} ticks "
        f"should yield zero PBRS, got {total_potential}"
    )


def test_pibrs_many_cycle_oscillation_bounded_near_zero():
    """Many-cycle oscillation stays bounded near zero (telescoping property).

    This test directly supersedes the informal manual check in the bug report:
    oscillating between two distances for 50+ cycles (100+ ticks) should yield
    total PBRS near zero, not a growing linear accumulation.

    With the bug (gamma=0.99), 50 cycles of oscillation between d=0.8 and d=1.5
    would accumulate approximately +0.794 — a significant exploit.
    With the fix (PBRS_GAMMA=1.0), the sum telescopes to near-zero.

    Note: We use max_hold=1000 to avoid ball hogging penalty during the
    test, since the default 15 would trigger after 15 ticks.
    """
    ad = AttackingDrillRewardAdapter(max_hold=1000)
    ad.reset()

    # Oscillate between distance 0.8 and 1.5.
    # Each cycle: 0.8 -> 1.5 -> 0.8 (2 ticks per cycle).
    d1, d2 = 0.8, 1.5
    num_cycles = 50
    distances = []
    for _ in range(num_cycles):
        distances.append(d1)
        distances.append(d2)

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

    # With PBRS_GAMMA = 1.0, the sum telescopes:
    # Sum of (phi(s_{t+1}) - phi(s_t)) = phi(s_final) - phi(s_initial_for_PBRS)
    # First tick has no PBRS (prev_dist is None), so PBRS starts from tick 2.
    # After tick 1, prev_dist = d1.
    # The sum from tick 2 onwards telescopes to: phi(last_dist) - phi(d1)
    #   = phi(d2) - phi(d1) = -0.75 - (-0.4) = -0.35
    EPSILON = 1e-6  # Slightly larger epsilon for float accumulation over 100 ticks
    expected = ad._phi(distances[-1]) - ad._phi(distances[0])
    assert abs(total_potential - expected) < EPSILON, (
        f"Oscillation over {num_cycles} cycles should telescope to ~{expected}, "
        f"got {total_potential}"
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
    # With PBRS_GAMMA = 1.0:
    # Tick 2: 1.0*phi(1.5) - phi(2.0) = -0.75 - (-1.0) = 0.25
    # Tick 3: 1.0*phi(1.0) - phi(1.5) = -0.5 - (-0.75) = 0.25
    # Tick 4: 1.0*phi(0.5) - phi(1.0) = -0.25 - (-0.5) = 0.25
    PBRS_GAMMA = AttackingDrillRewardAdapter.PBRS_GAMMA
    expected_potential = (
        PBRS_GAMMA * ad._phi(1.5) - ad._phi(2.0) +
        PBRS_GAMMA * ad._phi(1.0) - ad._phi(1.5) +
        PBRS_GAMMA * ad._phi(0.5) - ad._phi(1.0)
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
    # With PBRS_GAMMA = 1.0:
    # delta = 1.0 * phi(0.5) - phi(2.0) = -0.25 - (-1.0) = 0.75
    PBRS_GAMMA = AttackingDrillRewardAdapter.PBRS_GAMMA
    pb_delta = PBRS_GAMMA * ad._phi(0.5) - ad._phi(2.0)
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
    # With PBRS_GAMMA = 1.0: delta = 1.0 * phi(0.0) - phi(2.0) = 0 - (-1.0) = 1.0
    PBRS_GAMMA = AttackingDrillRewardAdapter.PBRS_GAMMA
    max_delta = PBRS_GAMMA * ad._phi(0.0) - ad._phi(2.0)
    assert abs(max_delta) <= 1.0  # Bounded by D_MAX range

    # Terminal goal reward is +2.0. PBRS should be a shaping nudge.
    realistic_total_pb = 0
    prev_dist = 2.0
    for _ in range(10):
        new_dist = prev_dist - 0.2
        delta = PBRS_GAMMA * ad._phi(new_dist) - ad._phi(prev_dist)
        realistic_total_pb += delta
        prev_dist = new_dist

    # After 10 ticks of progress, total should be positive but modest.
    assert realistic_total_pb > 0
    assert realistic_total_pb < 5.0  # Goal is +2.0


# ---------------------------------------------------------------------------
# Part 2 — Count-based exploration bonus (arXiv:2503.13077 motivated).
#
# The bonus rewards reaching under-visited pitch regions. It is gated on left
# team possession (only left is paid) but tracks true visitation regardless of
# team. Counts persist across reset() — this is intentional and the opposite
# of every other piece of per-episode state in the adapter.
# ---------------------------------------------------------------------------

import math

GT_LEFT = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
BETA = AttackingDrillRewardAdapter.EXPLORATION_BETA


def _exploration_adapter():
    """Adapter configured to isolate the exploration bonus.

    max_hold is set huge and p_ball_hogging to 0 so the possession-based
    ball-hogging penalty never fires — these tests target the exploration
    bonus specifically, not the hold mechanism. The adapter is reset()ed and
    ready to use.
    """
    ad = AttackingDrillRewardAdapter(max_hold=100000, p_hog=0.0)
    ad.reset()
    return ad


def _bonus_for(ad, ball_x, ball_y, owner_gt, base_reward=0.0):
    """Run one shaped-reward tick and return the exploration bonus paid.

    Isolates the exploration component by subtracting the known step cost
    and the PBRS term (zero here because distance is constant and the first
    call has no previous distance). The remaining delta is the exploration
    bonus.
    """
    gt = {
        "current_ball_owner": owner_gt["current_ball_owner"],
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_distance_to_goal": 1.0,  # constant => PBRS delta == 0
    }
    out = ad.compute_shaped_rewards({"left_0": base_reward}, [], gt, ["left_0"])
    # step_cost applies every tick; PBRS is 0 (constant dist, no prior).
    return out["left_0"] - base_reward - ad.step_cost


def test_exploration_bonus_decreases_with_repeated_visits():
    """Req 1: bonus strictly decreases across repeated visits to one cell."""
    ad = _exploration_adapter()
    bonuses = []
    for _ in range(20):
        bonuses.append(_bonus_for(ad, 0.05, 0.05, GT_LEFT))
    # Each visit must pay strictly less than the previous.
    for i in range(1, len(bonuses)):
        assert bonuses[i] < bonuses[i - 1], (
            f"bonus[{i}]={bonuses[i]:.6f} not < bonus[{i-1}]={bonuses[i-1]:.6f}"
        )
    # First visit pays the full beta.
    assert bonuses[0] == pytest.approx(BETA, abs=1e-9)


def test_exploration_bonus_approaches_zero_after_many_visits():
    """Req 2: after 1000 visits to one cell, bonus is below a small threshold."""
    ad = _exploration_adapter()
    last_bonus = 0.0
    for _ in range(1000):
        last_bonus = _bonus_for(ad, 0.3, -0.1, GT_LEFT)
    # beta / sqrt(1 + 999) ~= 0.03 / 31.6 ~= 0.00095. Threshold 1e-3.
    assert last_bonus < 1e-3, f"bonus after 1000 visits was {last_bonus:.6f}"
    assert last_bonus > 0.0  # never goes negative


def test_exploration_cycling_does_not_sustain_constant_bonus():
    """Req 3: cycling between a few cells yields a decreasing per-cycle total.

    This is the exploration-bonus analog of the PBRS oscillation test: a
    pattern that pays the same or more on repetition is farmable. We cycle
    between two cells for 60 cycles and assert the per-cycle bonus total
    strictly decreases over successive cycles.
    """
    ad = _exploration_adapter()
    cell_a, cell_b = (0.05, 0.05), (0.25, -0.15)
    per_cycle_totals = []
    for cycle in range(60):
        b1 = _bonus_for(ad, cell_a[0], cell_a[1], GT_LEFT)
        b2 = _bonus_for(ad, cell_b[0], cell_b[1], GT_LEFT)
        per_cycle_totals.append(b1 + b2)
    # Each cycle's total must be strictly less than the previous cycle's.
    for i in range(1, len(per_cycle_totals)):
        assert per_cycle_totals[i] < per_cycle_totals[i - 1], (
            f"cycle {i} total {per_cycle_totals[i]:.6f} not < "
            f"cycle {i-1} total {per_cycle_totals[i-1]:.6f}"
        )


def test_exploration_visit_counts_survive_reset():
    """Req 4: visit counts persist across reset() — intentionally different
    from every other piece of per-episode state in this file."""
    ad = _exploration_adapter()
    # Visit a cell once.
    _bonus_for(ad, 0.05, 0.05, GT_LEFT)
    cell = AttackingDrillRewardAdapter._cell_for_position(0.05, 0.05)
    assert ad._visit_counts[cell] == 1
    # Reset and visit the same cell again.
    ad.reset()
    assert cell in ad._visit_counts, "visit counts were wiped by reset()"
    assert ad._visit_counts[cell] == 1, "count did not survive reset()"
    _bonus_for(ad, 0.05, 0.05, GT_LEFT)
    # The post-reset visit observes the combined count of 2.
    assert ad._visit_counts[cell] == 2


def test_exploration_bonus_magnitude_sane():
    """Req 5: the bonus stays small relative to PBRS and the terminal goal
    reward across a representative sequence."""
    ad = _exploration_adapter()
    # Representative sequence: 30 ticks of left possession, ball moving
    # around several cells (realistic exploration), distance held constant so
    # PBRS == 0 and we isolate the exploration component.
    positions = [(0.05, 0.05), (0.25, -0.15), (-0.35, 0.25), (0.65, -0.05)]
    total_bonus = 0.0
    for i in range(30):
        bx, by = positions[i % len(positions)]
        total_bonus += _bonus_for(ad, bx, by, GT_LEFT)
    # Total exploration bonus over 30 ticks must stay well below a single
    # terminal goal reward (+2.0) and below a realistic PBRS progress total.
    # With 4 cells in rotation the 30-tick sum is ~0.5; assert a firm ceiling
    # an order of magnitude below the goal reward.
    assert total_bonus < 1.0, f"total bonus {total_bonus:.4f} too large"
    assert total_bonus > 0.0


def test_exploration_bonus_only_paid_on_left_possession():
    """The bonus is only PAID when left owns the ball, but counts still
    increment on right/loose possession (true visitation tracking)."""
    ad = _exploration_adapter()
    gt_right = {"current_ball_owner": {"team": "right", "agent_id": "right_0"}}
    # Right possession: no bonus paid, but count increments.
    out_right = ad.compute_shaped_rewards(
        {"left_0": 0.0},
        [],
        {"current_ball_owner": gt_right["current_ball_owner"],
         "ball_x": 0.05, "ball_y": 0.05, "ball_distance_to_goal": 1.0},
        ["left_0"],
    )
    # Only step cost applies (no PBRS on first call, no exploration bonus).
    assert out_right["left_0"] == pytest.approx(-0.005, abs=1e-9)
    cell = AttackingDrillRewardAdapter._cell_for_position(0.05, 0.05)
    assert ad._visit_counts[cell] == 1, "count should increment even without left possession"
    # Loose ball (None owner): same — no bonus, count increments.
    ad.compute_shaped_rewards(
        {"left_0": 0.0},
        [],
        {"current_ball_owner": None,
         "ball_x": 0.05, "ball_y": 0.05, "ball_distance_to_goal": 1.0},
        ["left_0"],
    )
    assert ad._visit_counts[cell] == 2
    # Now left takes possession: bonus paid using prior_count == 2.
    bonus = _bonus_for(ad, 0.05, 0.05, GT_LEFT)
    assert bonus == pytest.approx(BETA / math.sqrt(1 + 2), abs=1e-9)


def test_exploration_cell_for_position_bounds():
    """_cell_for_position maps in-bounds positions to cells and rejects
    out-of-bounds positions (no tracking, no crash)."""
    # In-bounds center.
    assert AttackingDrillRewardAdapter._cell_for_position(0.0, 0.0) is not None
    # In-bounds corners (clamped into last cell, not rejected).
    assert AttackingDrillRewardAdapter._cell_for_position(1.0, 0.42) is not None
    assert AttackingDrillRewardAdapter._cell_for_position(-1.0, -0.42) is not None
    # Out-of-bounds: rejected.
    assert AttackingDrillRewardAdapter._cell_for_position(1.5, 0.0) is None
    assert AttackingDrillRewardAdapter._cell_for_position(0.0, 0.5) is None
    # None inputs: rejected.
    assert AttackingDrillRewardAdapter._cell_for_position(None, 0.0) is None
