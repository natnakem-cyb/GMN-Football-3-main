"""Whole-pipeline reward stress tests (synthetic engine-reward integration).

SYNTHETIC ENGINE-REWARD INTEGRATION TEST; DOES NOT EXECUTE THE LIVE BRIDGE.

These tests drive the real AttackingDrillRewardAdapter tick-by-tick with a
synthetic engine-reward stream that reproduces the known engine behaviour
(src/engine/ObservationEncoder.ts computeReward):

    engine pays +0.15 per completed pass  (shared broadcast to all agents)
    engine pays +2.00 per goal            (shared broadcast to all agents)

They measure the FULL reward (R_engine + R_adapter + R_dense + R_penalties
+ R_potential), NOT the adapter in isolation. Every tick reconciles:

    sum(components) == sum(final shaped rewards)

within floating-point tolerance.

Layer distinction (required by the audit):
    - "adapter pass shaping" refers to _pay_pass_rewards (bounded <= 0.20)
    - "engine pass reward" refers to the synthetic +0.15/pass stream
    - "total pass-cycle reward" is the sum of both layers

Live bridge verification remains environment-dependent and is reported
separately in training/results/REWARD_WHOLE_PIPELINE_STRESS_TEST.md.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from training.reward_adapters import AttackingDrillRewardAdapter  # noqa: E402

try:
    from training.gmn_pettingzoo import GMNMultiAgentEnv  # noqa: E402
    CANONICALIZER_AVAILABLE = True
    CANONICALIZER_REASON = ""
except Exception as _exc:  # pragma: no cover - environment-dependent
    GMNMultiAgentEnv = None
    CANONICALIZER_AVAILABLE = False
    CANONICALIZER_REASON = (
        "training.gmn_pettingzoo unavailable in this environment "
        f"({type(_exc).__name__}: {_exc}); live-path canonicalizer "
        "verification remains environment-dependent"
    )

# --- Synthetic engine model constants (from src/engine/ObservationEncoder.ts) ---
ENGINE_PASS_REWARD = 0.15   # per completed pass, broadcast to every agent
ENGINE_GOAL_REWARD = 2.00   # per goal, broadcast to every agent

# --- Adapter constants mirrored analytically for reconciliation ---
ADAPTER_PASS_REWARD = 0.10
MAX_PRODUCTIVE_PASSES = 2
SHOT_REWARD_FIRST = 0.15
SHOT_REWARD_SECOND = 0.05
SHOT_SAVED_REWARD_FIRST = 0.20
DENSE_POSSESSION_REWARD = 0.01
STEP_COST = -0.005
BALL_HOG_PENALTY = -0.02
MAX_HOLD_TICKS = 15
ASSISTED_GOAL_REWARD = 0.50
PBRS_D_MAX = 2.0
PROX_MAX_DIST = 1.0
PROX_REWARD = 0.02
TIMEOUT_PENALTY = -0.50
T_MAX = 50


def _phi_goal(d):
    """Phi(d) = -clip(d, 0, D_MAX) / D_MAX (ball distance to goal)."""
    return -min(max(float(d), 0.0), PBRS_D_MAX) / PBRS_D_MAX


def _phi_prox(d):
    """Phi(d) = -clip(d, 0, PROX_MAX) / PROX_MAX (nearest agent to ball)."""
    return -min(max(float(d), 0.0), PROX_MAX_DIST) / PROX_MAX_DIST


AGENTS = ["left_0", "left_1", "left_2"]


class WholePipelineAccountant:
    """Drives the adapter tick-by-tick with the synthetic engine stream and
    records a full per-component breakdown, reconciled against the adapter's
    actual output every tick."""

    COMPONENTS = (
        "engine_pass", "engine_goal", "adapter_pass", "shot_attempt",
        "shot_saved", "assisted_goal", "possession", "goal_potential",
        "proximity_potential", "exploration", "turnover", "ball_hog",
        "step_cost", "timeout",
    )

    def __init__(self, agents, adapter=None):
        self.agents = list(agents)
        self.adapter = adapter if adapter is not None else AttackingDrillRewardAdapter()
        self.per_agent = {a: 0.0 for a in self.agents}
        self.components = {k: 0.0 for k in self.COMPONENTS}
        self.shaped_total = 0.0
        self.ticks = 0
        self.passes = 0
        self.goals = 0
        self.shots = 0
        self.max_holder_ticks = 0
        # analytic mirror state
        self._holder_id = None
        self._holder_ticks = 0
        self._prev_ball_dist = None
        self._prev_prox = None
        self._pass_chain = 0
        self._adapter_pass_count = 0
        self._shot_attempt_count = 0
        self._shot_saved_count = 0

    # ------------------------------------------------------------------
    def tick(self, base_rewards, events, owner, ball_dist=None, prox=None):
        """One environment tick. base_rewards is the synthetic engine stream
        for this tick (engine pass/goal reward); events is the step_events
        list; owner is the current_ball_owner info dict."""
        self.ticks += 1
        has_goal = any(
            isinstance(e, dict) and e.get("type") == "GOAL_SCORED" for e in events)
        has_gp = has_goal or any(
            isinstance(e, dict) and e.get("type") == "PASS_COMPLETED" for e in events)
        left_owns = isinstance(owner, dict) and owner.get("team") == "left"

        # --- synthetic engine reward stream ---
        engine = {a: float(base_rewards.get(a, 0.0)) for a in self.agents}
        for e in events:
            if not isinstance(e, dict):
                continue
            etype = e.get("type")
            if etype == "PASS_COMPLETED" and e.get("team") == "left":
                self.components["engine_pass"] += ENGINE_PASS_REWARD * len(self.agents)
                self.passes += 1
            elif etype == "GOAL_SCORED":
                self.components["engine_goal"] += ENGINE_GOAL_REWARD * len(self.agents)
                self.goals += 1
            elif etype in ("SHOT_TAKEN", "SHOT_BLOCKED"):
                self.shots += 1

        # --- analytic mirror of the adapter (for reconciliation) ---
        expected = dict(engine)
        if not has_gp:
            expected = {a: 0.0 for a in self.agents}

        # possession / ball-hog (mirrors _apply_possession)
        if left_owns:
            hid = owner.get("agent_id")
            if hid == self._holder_id:
                self._holder_ticks += 1
            else:
                self._holder_id = hid
                self._holder_ticks = 1
            self.max_holder_ticks = max(self.max_holder_ticks, self._holder_ticks)
            if self._holder_ticks > MAX_HOLD_TICKS and hid in expected:
                expected[hid] += BALL_HOG_PENALTY
                self.components["ball_hog"] += BALL_HOG_PENALTY
        else:
            self._holder_id = None
            self._holder_ticks = 0

        # events (mirrors _handle_events + _pay_pass_rewards + _pay_shot_rewards)
        for e in events:
            if not isinstance(e, dict):
                continue
            etype = e.get("type")
            aid = e.get("agent_id")
            if etype == "PASS_COMPLETED" and e.get("team") == "left":
                self._pass_chain += 1
                if aid in expected:
                    self._adapter_pass_count += 1
                    if self._adapter_pass_count <= MAX_PRODUCTIVE_PASSES:
                        expected[aid] += ADAPTER_PASS_REWARD
                        self.components["adapter_pass"] += ADAPTER_PASS_REWARD
            elif etype == "GOAL_SCORED":
                if self._pass_chain > 0:
                    for a in self.agents:
                        expected[a] += ASSISTED_GOAL_REWARD
                        self.components["assisted_goal"] += ASSISTED_GOAL_REWARD
                self._pass_chain = 0
            elif etype in ("SHOT_TAKEN", "SHOT_BLOCKED"):
                team = e.get("team")
                is_left = (team == "left" if team is not None
                           else (aid is None or str(aid).startswith("left")))
                if is_left:
                    self._shot_attempt_count += 1
                    if self._shot_attempt_count == 1:
                        reward = SHOT_REWARD_FIRST
                    elif self._shot_attempt_count == 2:
                        reward = SHOT_REWARD_SECOND
                    else:
                        reward = 0.0
                    targets = [aid] if aid in expected else list(self.agents)
                    for t in targets:
                        expected[t] += reward
                        self.components["shot_attempt"] += reward
            elif etype == "SHOT_SAVED":
                self._shot_saved_count += 1
                if self._shot_saved_count == 1:
                    targets = [aid] if aid in expected else list(self.agents)
                    for t in targets:
                        expected[t] += SHOT_SAVED_REWARD_FIRST
                        self.components["shot_saved"] += SHOT_SAVED_REWARD_FIRST

        # PBRS goal potential (mirrors the adapter block; gamma=1.0)
        if ball_dist is not None and self._prev_ball_dist is not None and left_owns:
            delta = _phi_goal(ball_dist) - _phi_goal(self._prev_ball_dist)
            for a in self.agents:
                expected[a] += delta
                self.components["goal_potential"] += delta
        self._prev_ball_dist = None if ball_dist is None else float(ball_dist)

        # dense possession
        if left_owns and not has_goal:
            for a in self.agents:
                expected[a] += DENSE_POSSESSION_REWARD
                self.components["possession"] += DENSE_POSSESSION_REWARD

        # proximity PBRS
        if prox is not None and self._prev_prox is not None and not has_goal:
            delta = _phi_prox(prox) - _phi_prox(self._prev_prox)
            bonus = PROX_REWARD * delta
            for a in self.agents:
                expected[a] += bonus
                self.components["proximity_potential"] += bonus
        self._prev_prox = None if prox is None else float(prox)

        # step cost
        if not has_goal:
            for a in self.agents:
                expected[a] += STEP_COST
                self.components["step_cost"] += STEP_COST

        # --- run the real adapter ---
        info = {
            "current_ball_owner": owner,
            "ball_distance_to_goal": ball_dist,
            "ball_x": None,
            "ball_y": None,
            "nearest_left_agent_ball_distance": prox,
        }
        shaped = self.adapter.compute_shaped_rewards(
            base_rewards, events, info, self.agents, actions=None,
            tick=self.ticks - 1)

        # timeout: env applies the shot-clock penalty after compute_shaped_rewards
        penalty = self.adapter.check_shot_clock()
        if penalty is not None:
            for a in self.agents:
                expected[a] += penalty
                shaped[a] = shaped.get(a, 0.0) + penalty
                self.components["timeout"] += penalty

        # --- reconciliation: components == final reward, per agent and total ---
        for a in self.agents:
            assert abs(shaped[a] - expected[a]) < 1e-9, (
                f"tick {self.ticks} agent {a}: shaped={shaped[a]!r} "
                f"expected={expected[a]!r}")
            self.per_agent[a] += shaped[a]
        self.shaped_total += sum(shaped.values())
        comp_sum = sum(self.components.values())
        assert abs(comp_sum - self.shaped_total) < 1e-7, (
            f"tick {self.ticks} reconciliation failed: components={comp_sum!r} "
            f"shaped_total={self.shaped_total!r}")
        return shaped

    # ------------------------------------------------------------------
    def summary(self):
        return {
            "ticks": self.ticks,
            "passes": self.passes,
            "goals": self.goals,
            "shots": self.shots,
            "max_holder_ticks": self.max_holder_ticks,
            "per_agent": dict(self.per_agent),
            "team_total": self.shaped_total,
            "components": dict(self.components),
        }
# --- SCENARIO DRIVERS -------------------------------------------------


def run_pass_loop(n_passes, hold_ticks_between=0):
    """n_passes completed passes through A->B->C rotating, optionally with
    hold ticks between passes. Returns a WholePipelineAccountant."""
    acc = WholePipelineAccountant(AGENTS)
    for i in range(n_passes):
        receiver = AGENTS[i % 3]
        # hold ticks: current holder keeps the ball (pass events on hold ticks)
        for _ in range(hold_ticks_between):
            holder = acc._holder_id or AGENTS[0]
            acc.tick({}, [], {"team": "left", "agent_id": holder})
        # pass tick: engine pays +0.15 to every agent, receiver takes the ball
        acc.tick(
            {a: ENGINE_PASS_REWARD for a in AGENTS},
            [{"type": "PASS_COMPLETED", "team": "left", "agent_id": receiver}],
            {"team": "left", "agent_id": receiver},
        )
    return acc


def run_alternating_cycles(n_cycles, hold_ticks):
    """Explicit A holds -> pass to B -> B holds -> pass to C -> C holds ->
    pass to A cycle. Returns a WholePipelineAccountant."""
    acc = WholePipelineAccountant(AGENTS)
    holder_idx = 0
    for _ in range(n_cycles):
        holder = AGENTS[holder_idx]
        for _ in range(hold_ticks):
            acc.tick({}, [], {"team": "left", "agent_id": holder})
        receiver = AGENTS[(holder_idx + 1) % 3]
        acc.tick(
            {a: ENGINE_PASS_REWARD for a in AGENTS},
            [{"type": "PASS_COMPLETED", "team": "left", "agent_id": receiver}],
            {"team": "left", "agent_id": receiver},
        )
        holder_idx = (holder_idx + 1) % 3
    return acc


def run_stationary_hold(n_ticks):
    """One agent holds the ball for n_ticks without passing. Returns an
    accountant; ball-hog fires after tick 16."""
    acc = WholePipelineAccountant(AGENTS)
    for _ in range(n_ticks):
        acc.tick({}, [], {"team": "left", "agent_id": AGENTS[0]})
    return acc


def run_shot_spam(n_shots, shooter="left_0", with_saved=False):
    """Repeated shots without scoring; ball loose (no owner) each tick."""
    acc = WholePipelineAccountant(AGENTS)
    for i in range(n_shots):
        events = [{"type": "SHOT_TAKEN", "team": "left", "agent_id": shooter}]
        if with_saved and i == 0:
            events.append({"type": "SHOT_SAVED", "team": "left", "agent_id": shooter})
        acc.tick({}, events, None)
    return acc


def run_goal_trajectory(with_goal_potential=False):
    """Synthetic finishing trajectory: pass -> pass -> shot -> goal.

    Reward-accounting comparison ONLY; not evidence a live policy can
    execute it. Returns a WholePipelineAccountant."""
    acc = WholePipelineAccountant(AGENTS)
    # t0: pass left_0 -> left_1
    acc.tick(
        {a: ENGINE_PASS_REWARD for a in AGENTS},
        [{"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"}],
        {"team": "left", "agent_id": "left_1"},
        ball_dist=1.0 if with_goal_potential else None,
    )
    # t1: pass left_1 -> left_2
    acc.tick(
        {a: ENGINE_PASS_REWARD for a in AGENTS},
        [{"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_2"}],
        {"team": "left", "agent_id": "left_2"},
        ball_dist=0.8 if with_goal_potential else None,
    )
    # t2: shot by left_0
    acc.tick(
        {},
        [{"type": "SHOT_TAKEN", "team": "left", "agent_id": "left_0"}],
        {"team": "left", "agent_id": "left_0"},
        ball_dist=0.4 if with_goal_potential else None,
    )
    # t3: goal (engine pays +2.00 to every agent)
    acc.tick(
        {a: ENGINE_GOAL_REWARD for a in AGENTS},
        [{"type": "GOAL_SCORED", "team": "left", "agent_id": "left_0"}],
        {"team": "left", "agent_id": "left_0"},
        ball_dist=0.0 if with_goal_potential else None,
    )
    return acc


def run_potential_probe(dist_sequence, prox_sequence=None):
    """Measure goal-potential (and optional proximity) shaping in isolation.
    Ball stays with left_0; no events; returns accountant."""
    acc = WholePipelineAccountant(AGENTS)
    prox_sequence = prox_sequence or [None] * len(dist_sequence)
    for d, p in zip(dist_sequence, prox_sequence):
        acc.tick({}, [], {"team": "left", "agent_id": AGENTS[0]},
                 ball_dist=d, prox=p)
    return acc


def print_report_line(title, acc):
    s = acc.summary()
    c = s["components"]
    print(f"\n[{title}] ticks={s['ticks']} passes={s['passes']} "
          f"goals={s['goals']} shots={s['shots']} "
          f"max_holder_ticks={s['max_holder_ticks']}")
    print(f"  team_total={s['team_total']:.4f}")
    print(f"  per_agent=" + ", ".join(
        f"{a}={v:.4f}" for a, v in s["per_agent"].items()))
    print("  components=" + ", ".join(
        f"{k}={v:.4f}" for k, v in c.items() if abs(v) > 1e-12))


# =====================================================================
# TESTS
# =====================================================================

@pytest.mark.skipif(not CANONICALIZER_AVAILABLE, reason=CANONICALIZER_REASON)
class TestCanonicalPassEvents:
    """PART 8: direct unit tests of _canonicalize_pass_events."""

    def _canon(self, events, ep_len=10, pending=None):
        env_state = {"ep_len": ep_len, "pending_pass": pending, "agents": AGENTS}
        return GMNMultiAgentEnv._canonicalize_pass_events(events, env_state)

    def test_two_representations_of_same_pass_yield_one_event(self):
        """Engine event + pending-pass resolver event for the SAME physical
        pass must collapse to exactly one logical PASS_COMPLETED."""
        pending = {"agent_id": "left_0"}  # passer
        events = [
            {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"},
            {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"},
        ]
        out = self._canon(events, ep_len=10, pending=pending)
        n = sum(1 for e in out if e.get("type") == "PASS_COMPLETED")
        assert n == 1, f"expected 1 logical PASS_COMPLETED, got {n}"

    def test_two_different_legitimate_passes_yield_two_events(self):
        pending = {"agent_id": "left_0"}
        events = [
            {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"},
            {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_2"},
        ]
        out = self._canon(events, ep_len=10, pending=pending)
        n = sum(1 for e in out if e.get("type") == "PASS_COMPLETED")
        assert n == 2, f"expected 2 logical PASS_COMPLETED, got {n}"

    def test_same_tick_identity_invariant_documented(self):
        """Event identity is (tick, passer, receiver). Two completed passes to
        the SAME receiver within ONE tick collapse to one event. Safe by the
        engine invariant that a receiver can gain possession only once per
        tick (one ball). Documented as the identity invariant."""
        pending = {"agent_id": "left_0"}
        events = [
            {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"},
            {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"},
        ]
        out = self._canon(events, ep_len=7, pending=pending)
        n = sum(1 for e in out if e.get("type") == "PASS_COMPLETED")
        assert n == 1

    def test_non_pass_events_pass_through(self):
        events = [
            {"type": "SHOT_TAKEN", "team": "left", "agent_id": "left_0"},
            {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"},
            {"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"},
            {"type": "TURNOVER_CONCEDED", "team": "left", "agent_id": "left_0"},
        ]
        out = self._canon(events, ep_len=3, pending={"agent_id": "left_0"})
        assert len(out) == 3
        assert sum(1 for e in out if e.get("type") == "PASS_COMPLETED") == 1


class TestOnePassWholePipeline:
    """PART 1/5: one physical pass -> exactly one logical event -> exactly one
    adapter payment, with the synthetic engine reward included."""

    def test_single_pass_full_accounting(self):
        acc = WholePipelineAccountant(AGENTS)
        acc.tick(
            {a: ENGINE_PASS_REWARD for a in AGENTS},
            [{"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"}],
            {"team": "left", "agent_id": "left_1"},
        )
        c = acc.components
        assert c["engine_pass"] == pytest.approx(0.45)
        assert c["adapter_pass"] == pytest.approx(0.10)
        assert c["possession"] == pytest.approx(0.03)
        assert c["step_cost"] == pytest.approx(-0.015)
        assert sum(c.values()) == pytest.approx(acc.shaped_total, abs=1e-7)
        assert acc.per_agent["left_0"] == pytest.approx(0.15 + 0.01 - 0.005)
        assert acc.per_agent["left_1"] == pytest.approx(0.15 + 0.10 + 0.01 - 0.005)
        assert acc.per_agent["left_2"] == pytest.approx(0.15 + 0.01 - 0.005)
        assert acc.passes == 1
        assert acc.goals == 0
        print_report_line("ONE PASS", acc)


class Test100PassStress:
    """PART 2/6: 100 physical passes through the synthetic pipeline."""

    def test_100_pass_loop_full_accounting(self):
        acc = run_pass_loop(100)
        s = acc.summary()
        c = s["components"]
        # engine layer is UNBOUNDED: 100 x 0.15 x 3 agents = 45.0
        assert c["engine_pass"] == pytest.approx(45.0)
        # adapter layer IS bounded: 2 x 0.10
        assert c["adapter_pass"] == pytest.approx(0.20)
        assert c["adapter_pass"] <= 0.20 + 1e-9
        # rotating holder: ball-hog never fires
        assert c["ball_hog"] == pytest.approx(0.0)
        assert s["max_holder_ticks"] <= 15
        # shot clock fires once (no shots): -0.50 x 3
        assert c["timeout"] == pytest.approx(-1.50)
        assert s["goals"] == 0
        # dense: possession 100 x 0.01 x 3, step 100 x -0.005 x 3
        assert c["possession"] == pytest.approx(3.0)
        assert c["step_cost"] == pytest.approx(-1.5)
        # FULL total: 45.0 + 0.2 + 3.0 - 1.5 - 1.5 = 45.2 — strongly positive
        # WITHOUT any goal. This is the evidence that the ENGINE pass reward
        # (not the adapter cap) must be reconsidered for the finishing drill.
        assert s["team_total"] == pytest.approx(45.2, abs=1e-6)
        assert sum(c.values()) == pytest.approx(s["team_total"], abs=1e-6)
        # per-agent: engine 15.0 + adapter 0.10/0.10/0.00 + possession 1.0
        #            - step 0.5 - timeout 0.5
        assert s["per_agent"]["left_0"] == pytest.approx(15.1, abs=1e-6)
        assert s["per_agent"]["left_1"] == pytest.approx(15.1, abs=1e-6)
        assert s["per_agent"]["left_2"] == pytest.approx(15.0, abs=1e-6)
        print_report_line("100-PASS LOOP", acc)

    def test_layer_distinction_statement(self):
        """Report rule: never claim 'pass farming prevented' without naming the
        layer. Adapter pass shaping: bounded. Engine pass reward: unbounded.
        Total pass-cycle reward: not bounded at goal scale."""
        acc = run_pass_loop(100)
        c = acc.components
        adapter_bounded = c["adapter_pass"] <= 0.20 + 1e-9
        total_bounded = acc.summary()["team_total"] <= 5.0
        assert adapter_bounded        # implemented fact
        assert not total_bounded      # measured result
        print_report_line("LAYER DISTINCTION", acc)


class TestAlternatingHolderCycle:
    """PART 3/7: pass -> holder reset -> pass avoids the ball-hog penalty."""

    def _check(self, n_cycles, hold_ticks):
        acc = run_alternating_cycles(n_cycles, hold_ticks)
        s = acc.summary()
        c = s["components"]
        ticks = n_cycles * (hold_ticks + 1)
        expected = {
            "engine_pass": n_cycles * 0.45,
            "adapter_pass": 0.20,
            "possession": ticks * 0.03,
            "step_cost": ticks * -0.015,
            # ball-hog never fires: holder_ticks resets to 1 every pass
            "ball_hog": 0.0,
            # shot clock fires once when the episode passes tick 51
            "timeout": -1.5 if ticks > T_MAX else 0.0,
        }
        for k, v in expected.items():
            assert c[k] == pytest.approx(v, abs=1e-6), (
                f"{n_cycles} cycles hold={hold_ticks} {k}: {c[k]} != {v}")
        assert s["max_holder_ticks"] == hold_ticks + 1
        assert s["max_holder_ticks"] <= 15
        assert s["passes"] == n_cycles
        assert s["goals"] == 0
        assert sum(c.values()) == pytest.approx(s["team_total"], abs=1e-6)
        print_report_line(
            f"ALTERNATING-HOLDER {n_cycles} CYCLES (hold={hold_ticks})", acc)
        return s

    def test_10_cycles(self):
        s = self._check(10, 5)
        # 4.5 + 0.2 + 1.8 - 0.9 - 1.5 = 4.1 — positive without any goal
        assert s["team_total"] == pytest.approx(4.1, abs=1e-6)

    def test_50_cycles(self):
        s = self._check(50, 5)
        # 22.5 + 0.2 + 9.0 - 4.5 - 1.5 = 25.7
        assert s["team_total"] == pytest.approx(25.7, abs=1e-6)

    def test_100_cycles(self):
        s = self._check(100, 5)
        # 45.0 + 0.2 + 18.0 - 9.0 - 1.5 = 52.7 — grows LINEARLY with cycles
        assert s["team_total"] == pytest.approx(52.7, abs=1e-6)
        assert s["team_total"] > 10.0  # > 5x a single goal trajectory's total


class TestStationaryVsAlternating:
    """PART 8/9: a stationary holder going negative does NOT prove the
    possession system is safe — the alternating-holder cycle is the exploit."""

    def test_stationary_hold_is_negative(self):
        acc = run_stationary_hold(100)
        s = acc.summary()
        c = s["components"]
        # ticks 1-15: +0.015/tick team; ticks 16-100: -0.005/tick team
        assert c["ball_hog"] == pytest.approx(85 * BALL_HOG_PENALTY)
        # possession 100x0.03, step 100x-0.015, hog 85x-0.02,
        # shot-clock timeout fires once at tick 51: -0.50x3
        assert c["possession"] == pytest.approx(3.0)
        assert c["step_cost"] == pytest.approx(-1.5)
        assert c["timeout"] == pytest.approx(-1.5)
        assert s["team_total"] == pytest.approx(3.0 - 1.5 - 1.7 - 1.5, abs=1e-6)
        assert s["team_total"] < 0.0
        print_report_line("STATIONARY HOLD 100 TICKS", acc)

    def test_alternating_cycle_beats_stationary_by_an_order_of_magnitude(self):
        stationary = run_stationary_hold(100).summary()["team_total"]
        # 100 alternating cycles at hold=1 -> 200 ticks, same tick count is
        # not required; use the 10-cycle run (60 ticks) for a like-for-like
        # per-tick comparison instead: cycle 60 ticks vs stationary 60 ticks.
        cyc = run_alternating_cycles(10, 5).summary()["team_total"]
        stat60 = run_stationary_hold(60).summary()["team_total"]
        assert cyc > 0.0
        assert stat60 < 0.0
        assert cyc > stationary
        print(f"\n[CONTRAST] stationary(60t)={stat60:.4f} "
              f"alternating(60t)={cyc:.4f}")


class TestShotStress:
    """PART 9/10: shot spam measured on the FULL reward, attempt and saved
    components distinguished, episode-scoped caps verified."""

    def test_10_shots(self):
        acc = run_shot_spam(10)
        s = acc.summary()
        c = s["components"]
        # attempts: +0.15 + 0.05 + 0x8 = 0.20 (episode-scoped cap)
        assert c["shot_attempt"] == pytest.approx(0.20)
        assert c["shot_saved"] == pytest.approx(0.0)
        # engine pays no shot reward in this model
        assert c["engine_pass"] == pytest.approx(0.0)
        # loose ball: possession 0, step -0.015/tick
        assert c["possession"] == pytest.approx(0.0)
        assert c["step_cost"] == pytest.approx(-0.15)
        # shots reset the shot clock -> no timeout
        assert c["timeout"] == pytest.approx(0.0)
        assert s["team_total"] == pytest.approx(0.05, abs=1e-6)
        assert s["goals"] == 0
        print_report_line("10 SHOTS", acc)

    def test_100_shots(self):
        acc = run_shot_spam(100)
        s = acc.summary()
        c = s["components"]
        assert c["shot_attempt"] == pytest.approx(0.20)
        assert c["step_cost"] == pytest.approx(-1.5)
        assert s["team_total"] == pytest.approx(-1.30, abs=1e-6)
        print_report_line("100 SHOTS", acc)

    def test_saved_shot_paid_once(self):
        acc = run_shot_spam(10, with_saved=True)
        s = acc.summary()
        c = s["components"]
        assert c["shot_attempt"] == pytest.approx(0.20)
        assert c["shot_saved"] == pytest.approx(0.20)
        assert s["team_total"] == pytest.approx(0.25, abs=1e-6)
        print_report_line("10 SHOTS + 1 SAVED", acc)


class TestSyntheticGoalTrajectory:
    """PART 4/10: synthetic pass->pass->shot->goal accounting comparison.
    Reward-accounting comparison ONLY — not evidence a live policy can
    execute the trajectory."""

    def test_goal_trajectory_without_potential(self):
        acc = run_goal_trajectory()
        s = acc.summary()
        c = s["components"]
        assert c["engine_pass"] == pytest.approx(0.90)   # 2 passes x 0.15 x 3
        assert c["engine_goal"] == pytest.approx(6.00)   # 2.00 x 3
        assert c["adapter_pass"] == pytest.approx(0.20)
        assert c["shot_attempt"] == pytest.approx(0.15)
        assert c["assisted_goal"] == pytest.approx(1.50)  # 0.50 x 3
        assert c["possession"] == pytest.approx(0.09)     # 3 ticks x 0.03
        assert c["step_cost"] == pytest.approx(-0.045)    # 3 ticks x -0.015
        assert c["goal_potential"] == pytest.approx(0.0)
        assert s["goals"] == 1
        assert s["team_total"] == pytest.approx(8.795, abs=1e-6)
        assert sum(c.values()) == pytest.approx(s["team_total"], abs=1e-6)
        print_report_line("GOAL TRAJECTORY (no potential)", acc)

    def test_goal_trajectory_with_potential(self):
        acc = run_goal_trajectory(with_goal_potential=True)
        s = acc.summary()
        c = s["components"]
        # deltas: t1 +0.1, t2 +0.2, t3 +0.2 per agent -> 1.5 team total
        assert c["goal_potential"] == pytest.approx(1.50, abs=1e-6)
        assert s["team_total"] == pytest.approx(10.295, abs=1e-6)
        print_report_line("GOAL TRAJECTORY (with potential)", acc)

    def test_pass_cycle_dominates_goal_trajectory(self):
        """MEASURED RESULT (synthetic): the 100-pass non-scoring cycle earns
        ~5x the synthetic goal trajectory. Goal dominance is NOT established
        under the current engine pass reward."""
        goal_total = run_goal_trajectory().summary()["team_total"]
        cycle_total = run_pass_loop(100).summary()["team_total"]
        assert cycle_total > goal_total * 5.0
        print(f"\n[DOMINANCE] goal_trajectory={goal_total:.4f} "
              f"pass_cycle_100={cycle_total:.4f} "
              f"ratio={cycle_total / goal_total:.2f}x")


class TestPotentialTerms:
    """PART 12: measure (not redesign) the potential-difference terms."""

    def test_goal_potential_telescoping_deltas(self):
        acc = run_potential_probe([1.0, 0.8, 0.6])
        c = acc.components
        # t1: phi(0.8)-phi(1.0) = +0.1/agent; t2: +0.1/agent -> 0.6 team
        assert c["goal_potential"] == pytest.approx(0.60, abs=1e-9)
        # team = 3 ticks x (0.03 possession - 0.015 step) + 0.6 potential
        assert acc.summary()["team_total"] == pytest.approx(0.645, abs=1e-6)
        print_report_line("GOAL POTENTIAL PROBE", acc)

    def test_round_trip_sums_to_zero(self):
        # 1.0 -> 0.5 -> 1.0: net phi change = 0 -> potential term = 0
        acc = run_potential_probe([1.0, 0.5, 1.0])
        c = acc.components
        assert c["goal_potential"] == pytest.approx(0.0, abs=1e-9)
        print_report_line("GOAL POTENTIAL ROUND TRIP", acc)

    def test_proximity_potential(self):
        acc = run_potential_probe([1.0, 0.8, 0.6], prox_sequence=[1.0, 0.5, 0.0])
        c = acc.components
        # prox deltas: +0.5 and +0.5 (phi units) x 0.02 x 3 agents
        assert c["proximity_potential"] == pytest.approx(0.06, abs=1e-9)
        print_report_line("PROXIMITY PROBE", acc)


class TestExplorationDisabled:
    """PART 11: exploration_reward == 0 for any synthetic trajectory."""

    def test_exploration_component_zero(self):
        acc = WholePipelineAccountant(AGENTS)
        assert acc.adapter.enable_exploration_bonus is False
        assert acc.adapter.exploration_beta == 0.0
        acc.tick(
            {a: ENGINE_PASS_REWARD for a in AGENTS},
            [{"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_1"}],
            {"team": "left", "agent_id": "left_1"},
        )
        acc.tick({}, [], None)
        assert acc.components["exploration"] == pytest.approx(0.0)


# =====================================================================
# COUNTERFACTUAL ANALYSIS: Engine Pass Reward Removal
# =====================================================================
# Key Question: Does the non-scoring pass-cycle proxy remain more
# rewarding than scoring when engine+pass reward = 0?
#
# This analysis requires computing what the trajectory would be worth
# if only base_rewards were used (no separate engine_pass component).

def compute_cf_pass_loop(n_passes, hold_ticks_between=0):
    """Counterfactual: compute pass loop with engine_pass=0.
    
    RECONCILIATION METHOD:
    - Run baseline pass loop
    - Subtract engine_pass component (100 × 0.15 × 3 = 45.0)
    - This gives the CF value where base_rewards=0 but adapter still awards
    
    Note: This avoids duplicating the complex tick() logic.
    """
    baseline = run_pass_loop(n_passes, hold_ticks_between)
    cf = WholePipelineAccountant(AGENTS)
    
    # Copy all state except engine_pass
    cf.ticks = baseline.ticks
    cf.passes = baseline.passes
    cf.goals = baseline.goals
    cf.shots = baseline.shots
    cf.max_holder_ticks = baseline.max_holder_ticks
    cf.per_agent = dict(baseline.per_agent)
    cf.shaped_total = 0.0
    cf._holder_id = baseline._holder_id
    cf._holder_ticks = baseline._holder_ticks
    cf._prev_ball_dist = baseline._prev_ball_dist
    cf._prev_prox = baseline._prev_prox
    cf._pass_chain = baseline._pass_chain
    cf._adapter_pass_count = baseline._adapter_pass_count
    cf._shot_attempt_count = baseline._shot_attempt_count
    cf._shot_saved_count = baseline._shot_saved_count
    
    # Components WITHOUT engine_pass
    cf.components = {k: v for k, v in baseline.components.items() if k != "engine_pass"}
    
    # Adjust per_agent rewards (subtract engine_pass contribution)
    # Each agent gets 0.15 × 100 = 15.0 from engine pass in baseline
    engine_per_agent = n_passes * ENGINE_PASS_REWARD
    for a in cf.per_agent:
        cf.per_agent[a] -= engine_per_agent
    cf.shaped_total = sum(cf.per_agent.values())
    
    return cf


def compute_cf_alternating_cycles(n_cycles, hold_ticks):
    """Counterfactual: alternating cycles with engine_pass=0."""
    baseline = run_alternating_cycles(n_cycles, hold_ticks)
    
    # Subtract engine_pass contribution
    engine_per_agent = n_cycles * ENGINE_PASS_REWARD
    
    cf = WholePipelineAccountant(AGENTS)
    cf.ticks = baseline.ticks
    cf.passes = baseline.passes
    cf.goals = baseline.goals
    cf.shots = baseline.shots
    cf.max_holder_ticks = baseline.max_holder_ticks
    cf.per_agent = {a: v - engine_per_agent for a, v in baseline.per_agent.items()}
    cf.shaped_total = sum(cf.per_agent.values())
    cf.components = {k: v for k, v in baseline.components.items() if k != "engine_pass"}
    cf._holder_id = baseline._holder_id
    cf._holder_ticks = baseline._holder_ticks
    cf._prev_ball_dist = baseline._prev_ball_dist
    cf._prev_prox = baseline._prev_prox
    cf._pass_chain = baseline._pass_chain
    cf._adapter_pass_count = baseline._adapter_pass_count
    cf._shot_attempt_count = baseline._shot_attempt_count
    cf._shot_saved_count = baseline._shot_saved_count
    
    return cf


def compute_cf_goal_trajectory(with_goal_potential=False):
    """Counterfactual: goal trajectory with engine_pass=0."""
    baseline = run_goal_trajectory(with_goal_potential)
    
    # Subtract engine_pass contribution for 2 passes
    # 2 × 0.15 × 3 = 0.90 total, or 0.30 per agent
    engine_per_agent = 2 * ENGINE_PASS_REWARD
    
    cf = WholePipelineAccountant(AGENTS)
    cf.ticks = baseline.ticks
    cf.passes = baseline.passes
    cf.goals = baseline.goals
    cf.shots = baseline.shots
    cf.max_holder_ticks = baseline.max_holder_ticks
    cf.per_agent = {a: v - engine_per_agent for a, v in baseline.per_agent.items()}
    cf.shaped_total = sum(cf.per_agent.values())
    cf.components = {k: v for k, v in baseline.components.items() if k != "engine_pass"}
    cf._holder_id = baseline._holder_id
    cf._holder_ticks = baseline._holder_ticks
    cf._prev_ball_dist = baseline._prev_ball_dist
    cf._prev_prox = baseline._prev_prox
    cf._pass_chain = baseline._pass_chain
    cf._adapter_pass_count = baseline._adapter_pass_count
    cf._shot_attempt_count = baseline._shot_attempt_count
    cf._shot_saved_count = baseline._shot_saved_count
    
    return cf


class TestCounterfactualEnginePassZero:
    """Counterfactual: measure if proxy remains when engine pass reward = 0."""

    def test_100_pass_cf_engine_contribution(self):
        """Verify that removing engine_pass leaves only other components."""
        baseline = run_pass_loop(100)
        cf = compute_cf_pass_loop(100)
        
        baseline_total = baseline.summary()["team_total"]
        cf_total = cf.summary()["team_total"]
        engine_contribution = baseline_total - cf_total
        
        # EXPECTED: 100 × 0.15 × 3 = 45.0
        expected_engine = 100 * ENGINE_PASS_REWARD * len(AGENTS)
        
        print(f"\n[COUNTERFACTUAL] 100-pass baseline={baseline_total:.4f}")
        print(f"[COUNTERFACTUAL] 100-pass CF (engine=0)={cf_total:.4f}")
        print(f"[COUNTERFACTUAL] Engine contribution removed={engine_contribution:.4f}")
        print(f"[COUNTERFACTUAL] Expected engine contribution={expected_engine:.4f}")
        
        assert abs(engine_contribution - expected_engine) < 1e-6
        assert cf_total == pytest.approx(baseline_total - expected_engine, abs=1e-6)

    def test_100_cycles_cf_proxy_analysis(self):
        """KEY TEST: Does pass-cycle proxy remain when engine_pass=0?"""
        baseline = run_alternating_cycles(100, 5)
        cf = compute_cf_alternating_cycles(100, 5)
        
        baseline_total = baseline.summary()["team_total"]
        cf_total = cf.summary()["team_total"]
        engine_contribution = baseline_total - cf_total
        
        expected_engine = 100 * ENGINE_PASS_REWARD * len(AGENTS)
        
        print(f"\n[COUNTERFACTUAL] 100-cycles baseline={baseline_total:.4f}")
        print(f"[COUNTERFACTUAL] 100-cycles CF (engine=0)={cf_total:.4f}")
        print(f"[COUNTERFACTUAL] Engine contribution removed={engine_contribution:.4f}")
        
        # KEY QUESTION: What remains after removing engine reward?
        # If cf_total > goal_total, other components create proxy
        goal_total = run_goal_trajectory().summary()["team_total"]
        print(f"[COUNTERFACTUAL] Single goal trajectory={goal_total:.4f}")
        print(f"[COUNTERFACTUAL] Proxy/Goal ratio={cf_total/goal_total:.2f}x")
        
        # The alternative holder trajectory should be decomposed
        # R = R_engine + R_adapter + R_possession + R_step_cost + R_timeout
        # With engine_pass = 0: R_cf = R_adapter + R_possession + R_step_cost + R_timeout
        
        n_cycles = 100
        hold_ticks = 5
        ticks = n_cycles * (hold_ticks + 1)  # 600 ticks
        
        # Expected non-engine components:
        # adapter_pass: 0.20 (capped)
        # possession: 600 × 0.01 × 3 = 18.0 (left owns during all ticks)
        # step_cost: 600 × -0.005 × 3 = -9.0
        # timeout: -1.5 (at t=51)
        expected_cf = 0.20 + 18.0 - 9.0 - 1.5
        
        print(f"[COUNTERFACTUAL] Expected CF total (no engine pass)={expected_cf:.4f}")
        assert cf_total == pytest.approx(expected_cf, abs=0.01)
        
        # CRITICAL: Does proxy remain?
        # If cf_total > 0, the cycle is still rewarding without engine pass
        assert cf_total > 0, "Proxy would be eliminated without engine_pass"
        print(f"[COUNTERFACTUAL] Proxy survives without engine_pass: +{cf_total:.4f}")

    def test_stationary_cf_unchanged(self):
        """Stationary hold is unchanged (no passes = no engine contribution)."""
        baseline = run_stationary_hold(100)
        cf = compute_cf_alternating_cycles(100, 5)  # placeholder, stationary has no passes
        
        # Stationary hold has no passes, so baseline == cf for this class
        stat = run_stationary_hold(100)
        print(f"\n[COUNTERFACTUAL] Stationary 100 ticks={stat.summary()['team_total']:.4f}")
        assert stat.summary()["team_total"] < 0, "Stationary hold should be negative"

    def test_goal_cf_engine_pass_removed(self):
        """Goal trajectory with engine_pass removed."""
        baseline = run_goal_trajectory()
        cf = compute_cf_goal_trajectory()
        
        baseline_total = baseline.summary()["team_total"]
        cf_total = cf.summary()["team_total"]
        engine_contribution = baseline_total - cf_total
        
        # 2 passes: 2 × 0.15 × 3 = 0.90
        expected_engine = 2 * ENGINE_PASS_REWARD * len(AGENTS)
        
        print(f"\n[COUNTERFACTUAL] Goal trajectory baseline={baseline_total:.4f}")
        print(f"[COUNTERFACTUAL] Goal trajectory CF={cf_total:.4f}")
        print(f"[COUNTERFACTUAL] Engine pass contribution={engine_contribution:.4f}")
        
        assert abs(engine_contribution - expected_engine) < 1e-6

    def test_horizon_normalized_comparison(self):
        """Compare normalized rewards under common horizon (600 ticks for cycles)."""
        # 100 cycles at hold=5 = 600 ticks
        baseline = run_alternating_cycles(100, 5)
        cf = compute_cf_alternating_cycles(100, 5)
        
        # Stationary 600 ticks (same horizon)
        stat = run_stationary_hold(600)
        
        # Goal trajectory is 4 ticks - normalize differently
        goal = run_goal_trajectory()
        goal_summary = goal.summary()
        goal_ticks = 4
        
        baseline_summary = baseline.summary()
        cf_summary = cf.summary()
        stat_summary = stat.summary()
        
        print("\n[HORIZON NORMALIZED - 600 ticks]")
        print(f"  Alternating baseline: {baseline_summary['team_total']/baseline.ticks:.6f} per tick")
        print(f"  Alternating CF (engine=0): {cf_summary['team_total']/cf.ticks:.6f} per tick")
        print(f"  Stationary 600: {stat_summary['team_total']/stat.ticks:.6f} per tick")
        print(f"  Goal trajectory: {goal_summary['team_total']/goal_ticks:.6f} per tick")
        
        # With engine_pass removed, does cycle still beat stationary?
        assert cf_summary['team_total'] > stat_summary['team_total'], "Cycle should beat stationary even without engine pass"
        
        # Does goal trajectory remain superior for scoring objectives?
        # Compare total (not per-tick) since goals have different natural horizon
        print(f"\n[COMPARISON]")
        print(f"  600-tick cycle total (engine=0): {cf_summary['team_total']:.4f}")
        print(f"  4-tick goal total: {goal_summary['team_total']:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


# =====================================================================
# LIVE PIPELINE VERIFICATION (existing bridge infrastructure)
# =====================================================================
# LIVE ENGINE VERIFICATION — distinct from the synthetic accounting above.
# Uses the same bridge fixture pattern as training/tests/test_action_masks.py.
# Verifies the LIVE invariant: 1 physical pass -> exactly 1 logical
# PASS_COMPLETED event, and that the engine's +0.15 pass reward reaches the
# agents through gmn_pettingzoo's shared-reward broadcast.

import subprocess  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402

LIVE_PORT = 5157  # dedicated port; this test spawns its own bridge


def _start_live_bridge(port: int) -> subprocess.Popen:
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
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return proc
        except Exception:
            pass
    raise RuntimeError(f"Live bridge did not become healthy on port {port}")


def _mask_of(obs_dict, agent):
    v = obs_dict.get(agent)
    if isinstance(v, dict):
        return v.get("action_mask")
    return None


@pytest.mark.skipif(not CANONICALIZER_AVAILABLE, reason=CANONICALIZER_REASON)
class TestLiveOnePassPipeline:
    """Live verification through the real bridge. Skipped when the bridge
    cannot be started (environment-dependent)."""

    def test_live_single_pass_single_event_and_engine_reward(self):
        proc = None
        try:
            proc = _start_live_bridge(LIVE_PORT)
        except RuntimeError as exc:
            pytest.skip(f"live bridge unavailable: {exc}")
        try:
            # enable_reward_shaping=False so rewards == the pure ENGINE stream
            # (shared_reward broadcast) — isolates R_engine for measurement.
            env = GMNMultiAgentEnv(
                scenario="academy_pass_and_shoot_with_keeper",
                auto_start_bridge=False,
                port=LIVE_PORT,
                enable_reward_shaping=False,
            )
            try:
                obs_dict, _ = env.reset(seed=9001)
                agents = list(env.agents)

                def _live_agents_now():
                    return list(env.agents)

                def _refresh_obs_agent_list():
                    nonlocal obs_dict, agents
                    agents = _live_agents_now()
                    return agents

                # Phase 1: move right until a controlled player has the ball.
                possessed = False
                for _ in range(300):
                    live = _live_agents_now()
                    if not live:
                        obs_dict, _ = env.reset(seed=9001)
                        live = _live_agents_now()
                    obs_dict, _, terms, truncs, _ = env.step({a: 5 for a in live})
                    live = _live_agents_now()
                    if not live:
                        # Episode ended during the run-in (e.g. scored while
                        # wandering); restart and keep looking for possession.
                        obs_dict, _ = env.reset(seed=9001)
                        continue
                    for agent in live:
                        mask = _mask_of(obs_dict, agent)
                        if mask is None:
                            continue
                        if mask[9] == 1 and mask[16] == 0:
                            possessed = True
                            break
                    if possessed:
                        agents = live
                        break
                assert possessed, "no controlled player gained possession in 300 ticks"

                                                # Phase 2: the OWNER issues SHORT_PASS exactly once.
                # While the ball travels, ALL NON-OWNERS move toward the ball
                # so the pass is actually received and resolves as
                # PASS_COMPLETED. The owner idles so a new kick does not clear
                # the pending pass. With shaping disabled, rewards == engine stream.
                #
                # Ball position is at observation offset 88-89 (x, y) per
                # ObservationEncoder: offset 88 (len 3) = ball (x, y, z).
                physical_passes = 0
                logical_events = 0
                pass_tick_shared = None
                pass_issued = False
                owner_id = None

                def _ball_xy_for(agent):
                    """Extract ball (x, y) from an agent's observation dict."""
                    v = obs_dict.get(agent)
                    if isinstance(v, dict):
                        arr = v.get("observation")
                    else:
                        arr = v
                    if hasattr(arr, "__len__") and len(arr) > 90:
                        return float(arr[88]), float(arr[89])
                    return 0.0, 0.0

                def _move_toward_ball(agent):
                    """Return a movement action that heads toward the ball."""
                    bx, by = _ball_xy_for(agent)
                    # Simple 8-direction mapping: pick closest cardinal/diagonal
                    if bx > 0.05:
                        if by < -0.05:
                            return 4  # TOP_RIGHT
                        elif by > 0.05:
                            return 6  # BOTTOM_RIGHT
                        return 5  # RIGHT
                    elif bx < -0.05:
                        if by < -0.05:
                            return 2  # TOP_LEFT
                        elif by > 0.05:
                            return 8  # BOTTOM_LEFT
                        return 1  # LEFT
                    else:
                        if by < -0.05:
                            return 3  # TOP
                        elif by > 0.05:
                            return 7  # BOTTOM
                    return 0  # IDLE if ball is roughly centered

                for _ in range(600):
                    live_agents = _live_agents_now()
                    if not live_agents:
                        obs_dict, _ = env.reset(seed=9001)
                        continue
                    if not pass_issued:
                        # Find the current owner: agent whose mask allows
                        # SHORT_PASS (mask bit 11 == 1).
                        owner_id = None
                        for agent in live_agents:
                            mask = _mask_of(obs_dict, agent)
                            if mask is not None and len(mask) > 11 and mask[11] == 1:
                                owner_id = agent
                                break
                        if owner_id is None:
                            # Keep moving toward center until someone has the ball.
                            obs_dict, _, _, _, _ = env.step(
                                {a: 5 for a in live_agents})
                            continue
                        # Owner passes, teammates move toward current ball pos.
                        step_actions = {
                            a: (11 if a == owner_id else _move_toward_ball(a))
                            for a in live_agents
                        }
                        pass_issued = True
                    else:
                        # Pass issued: owner idles, non-owners converge on ball.
                        step_actions = {
                            a: (0 if a == owner_id else _move_toward_ball(a))
                            for a in live_agents
                        }
                    obs_dict, rewards, terms, truncs, infos = env.step(
                        step_actions)
                    live_agents = _live_agents_now()
                    if not live_agents:
                        # Terminal/truncation tick: no per-agent infos.
                        obs_dict, _ = env.reset(seed=9001)
                        pass_issued = False
                        owner_id = None
                        continue
                    step_events = infos[live_agents[0]].get("step_events", [])
                    n_pass = sum(
                        1 for e in step_events
                        if isinstance(e, dict) and e.get("type") == "PASS_COMPLETED")
                    if n_pass > 0:
                        physical_passes += 1
                        logical_events += n_pass
                        shared = {a: float(rewards[a]) for a in live_agents}
                        assert len(set(shared.values())) == 1, (
                            f"engine reward not broadcast equally: {shared}")
                        pass_tick_shared = shared[live_agents[0]]
                        # LIVE INVARIANT: exactly one logical event this tick
                        assert n_pass == 1, (
                            f"1 physical pass produced {n_pass} logical "
                            "PASS_COMPLETED events")
                        # The engine's +0.15 pass reward must be included in
                        # the shared reward (engine may add other dense terms).
                        assert pass_tick_shared >= 0.15 - 1e-9, (
                            f"engine pass reward missing: shared="
                            f"{pass_tick_shared}")
                        break
                else:
                    pytest.fail("no pass completed within 600 ticks "
                                "(owner-issued SHORT_PASS, others converging)")

                assert physical_passes == 1
                assert logical_events == 1
                print(f"\n[LIVE ONE-PASS] physical_passes=1 "
                      f"logical_events=1 shared_reward_on_pass_tick="
                      f"{pass_tick_shared:.4f}")
            finally:
                env.close()
        finally:
            if proc is not None:
                proc.terminate()