"""Scenario-scoped reward adapters.

Rondo 4v1 optimizes spatial control (pass + interception, no shot/goal
terms, keeps engine progress signal). Academy 3v1 optimizes finishing
(progress checkpoint stripped via event filter, shot incentives, step
cost, shot-clock timeout).

Preserves all CooperativeRewardShaper instrumentation: M1b attribution,
_pass counters, terminal emission semantics.
"""

from typing import Any, Dict, List, Optional


SHOT_ACTION_ID = 12
BALL_ACTION_IDS = frozenset((9, 10, 11, 12, 17))
SHOT_EVENT_TYPES = frozenset(("SHOT_TAKEN", "SHOT_SAVED", "SHOT_BLOCKED", "SHOT_MISSED"))
GOAL_OR_PASS_TYPES = frozenset(("GOAL_SCORED", "PASS_COMPLETED"))
_LEFT_VICTIM_TYPES = frozenset(("PASS_INTERCEPTED", "PASS_FAILED", "TURNOVER_CONCEDED"))
class BaseScenarioRewardAdapter:
    """Shared event logic ported from CooperativeRewardShaper."""

    def __init__(self, r_pass=0.30, r_assisted=0.50, p_solitary=-0.30,
                 p_hog=-0.02, p_turn=-0.10, max_hold=15, act_cost=-0.01):
        self.r_pass = r_pass
        self.r_assisted_goal = r_assisted
        self.p_solitary_shot = p_solitary
        self.p_ball_hogging = p_hog
        self.p_turnover = p_turn
        self.max_hold_ticks = max_hold
        self.action_cost = act_cost
        self.previous_left_ball_carrier = None
        self._attribution_log: List[Dict[str, Any]] = []
        self.total_pass_completed_count = 0
        self.total_goal_scored_count = 0
        self.total_turnover_conceded_count = 0
        self.reset()

    def reset(self) -> None:
        self.pass_chain_length = 0
        self.current_holder_id = None
        self.holder_ticks = 0
        self.previous_left_ball_carrier = None
        self.pass_completed_count = 0
        self.goal_scored_count = 0
        self.turnover_conceded_count = 0
        self.pass_intercepted_count = 0
        self.solitary_shot_count = 0
        self.assisted_goal_count = 0
        self.ball_hogging_count = 0
        self.shot_missed_count = 0

    def get_diagnostics(self) -> Dict[str, Any]:
        return {"pass_chain_length": self.pass_chain_length,
                "holder_ticks": self.holder_ticks,
                "pass_completed_count": self.pass_completed_count,
                "goal_scored_count": self.goal_scored_count,
                "turnover_conceded_count": self.turnover_conceded_count,
                "pass_intercepted_count": self.pass_intercepted_count,
                "solitary_shot_count": self.solitary_shot_count,
                "assisted_goal_count": self.assisted_goal_count,
                "ball_hogging_count": self.ball_hogging_count,
                "shot_missed_count": self.shot_missed_count}

    def get_cumulative_diagnostics(self) -> Dict[str, Any]:
        return {"total_pass_completed_count": self.total_pass_completed_count,
                "total_goal_scored_count": self.total_goal_scored_count,
                "total_turnover_conceded_count": self.total_turnover_conceded_count}

    def get_attribution_log(self) -> List[Dict[str, Any]]:
        return list(self._attribution_log)

    def check_shot_clock(self) -> Optional[float]:
        return None

    def _apply_action_cost(self, shaped, actions, exempt=None) -> None:
        if actions:
            for aid, aidx in actions.items():
                if aid in shaped and aidx in BALL_ACTION_IDS:
                    if exempt is not None and aidx in exempt:
                        continue
                    shaped[aid] += self.action_cost

    def _apply_possession(self, shaped, gt) -> None:
        owner = gt.get("current_ball_owner") if gt else None
        if isinstance(owner, dict) and owner.get("team") == "left":
            hid = owner.get("agent_id")
            if hid and hid == self.current_holder_id:
                self.holder_ticks += 1
            else:
                self.current_holder_id = hid
                self.holder_ticks = 1
            if hid and hid in shaped:
                self.previous_left_ball_carrier = hid
            if self.holder_ticks > self.max_hold_ticks and hid in shaped:
                shaped[hid] += self.p_ball_hogging
                self.ball_hogging_count += 1
        else:
            self.current_holder_id = None
            self.holder_ticks = 0

    def _resolve_victim(self, shaped, aid):
        if aid is not None and aid in shaped:
            return aid
        if self.previous_left_ball_carrier is not None and self.previous_left_ball_carrier in shaped:
            return self.previous_left_ball_carrier
        return None

    def _log_victim(self, etype, aid, victim, shaped) -> None:
        self._attribution_log.append({"event_type": etype, "event_agent_id": aid,
            "resolved_victim_id": victim,
            "fallback_used": victim is not None and aid not in shaped,
            "penalty_applied": victim is not None,
            "shaped_rewards": {k: float(v) for k, v in shaped.items()}})

    def _handle_shot(self, shaped, etype, aid, actions) -> None:
        """Solitary-shot penalty in the base shaper (Rondo + legacy).

        AttackingDrillRewardAdapter overrides this: in a finishing drill a
        shot attempt is exactly the desired behavior, so there is no
        solitary-shot penalty there.
        """
        if self.pass_chain_length > 0:
            return
        shooters: List[str] = []
        if aid is not None and aid in shaped:
            shooters = [aid]
        elif actions:
            shooters = [ag for ag, ax in actions.items()
                        if ag in shaped and ax == SHOT_ACTION_ID]
        for sh in shooters:
            shaped[sh] += self.p_solitary_shot
            self.solitary_shot_count += 1

    def _handle_events(self, shaped, step_events, active_agents, actions) -> None:
        for event in step_events:
            if not isinstance(event, dict):
                continue
            etype = event.get("type")
            eteam = event.get("team")
            aid = event.get("agent_id")
            if etype in _LEFT_VICTIM_TYPES:
                victim = self._resolve_victim(shaped, aid)
                self.pass_chain_length = 0
                self.total_turnover_conceded_count += 1
                if etype == "PASS_INTERCEPTED":
                    self.pass_intercepted_count += 1
                else:
                    self.turnover_conceded_count += 1
                if victim is not None:
                    shaped[victim] += self.p_turnover
                self.previous_left_ball_carrier = None
                self._log_victim(etype, aid, victim, shaped)
                continue
            if eteam != "left":
                continue
            if etype == "PASS_COMPLETED":
                self.pass_chain_length += 1
                self.pass_completed_count += 1
                self.total_pass_completed_count += 1
                if aid in shaped:
                    shaped[aid] += self.r_pass
                if aid is not None and aid in shaped:
                    self.previous_left_ball_carrier = aid
            elif etype == "SHOT_MISSED":
                self.shot_missed_count += 1
            elif etype == "SHOT_TAKEN":
                self._handle_shot(shaped, etype, aid, actions)
            elif etype == "GOAL_SCORED":
                if self.pass_chain_length > 0:
                    for ag in active_agents:
                        if ag in shaped:
                            shaped[ag] += self.r_assisted_goal
                    self.assisted_goal_count += 1
                self.goal_scored_count += 1
                self.total_goal_scored_count += 1
                self.pass_chain_length = 0

    def compute_shaped_rewards(self, base_rewards, step_events,
                               info_ground_truth, active_agents,
                               actions=None, tick=0, max_ticks=None):
        shaped = {a: base_rewards.get(a, 0.0) for a in active_agents
                  if not a.startswith("right_")}
        self._apply_action_cost(shaped, actions)
        self._apply_possession(shaped, info_ground_truth)
        self._handle_events(shaped, step_events, active_agents, actions)
        return shaped
class RondoRewardAdapter(BaseScenarioRewardAdapter):
    """Spatial control: pass + interception + possession. No shot/goal terms.

    Inherits the base event logic (pass bonus, turnover/intercept penalties,
    ball-hogging penalty, action cost) but NOT the Attacking drill's
    progress-strip / shot incentives / step cost / shot-clock. The engine's
    dense progress signal stays (advancing the ball under pressure is part
    of the drill); shot and goal events pay nothing here because the base
    event handler has no shot/goal reward terms for non-assisted contexts
    and rondo scenarios never emit GOAL_SCORED to this adapter's path.
    """

    @property
    def name(self) -> str:
        return "rondo"

    def get_diagnostics(self) -> Dict[str, Any]:
        d = super().get_diagnostics()
        d["adapter"] = self.name
        return d

    def compute_shaped_rewards(self, base_rewards, step_events,
                               info_ground_truth, active_agents,
                               actions=None, tick=0, max_ticks=None):
        shaped = {a: base_rewards.get(a, 0.0) for a in active_agents
                  if not a.startswith("right_")}
        self._apply_action_cost(shaped, actions)
        self._apply_possession(shaped, info_ground_truth)
        # Pass completion / turnover / interception handled by the base
        # event loop (identical semantics to CooperativeRewardShaper).
        self._handle_events(shaped, step_events, active_agents, actions)
        return shaped

class AttackingDrillRewardAdapter(BaseScenarioRewardAdapter):
    """Finishing drill: strip progress, reward shots, step cost, clock."""

    def __init__(self, step_cost=-0.005, shot_reward=0.25,
                 on_target_reward=0.40, t_max=50,
                 timeout_penalty=-0.50, **kw):
        super().__init__(**kw)
        self.step_cost = step_cost
        self.r_shot = shot_reward
        self.r_on_target = on_target_reward
        self.t_max = t_max
        self.timeout_penalty = timeout_penalty
        # Finishing drill: a shot attempt IS the objective. Never penalize
        # a solitary shot, never charge action cost on shot/pass, never
        # penalize a missed attempt beyond the missing on-target bonus.
        self.p_solitary_shot = 0.0
        self.action_cost = 0.0
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.ticks_no_shot = 0
        self.seen_shot = False
        self.timeout_count = 0
        self.shot_taken_count = 0
        self.shot_on_target_count = 0
        self.shot_missed_count = 0

    @property
    def name(self) -> str:
        return "attacking"

    def get_diagnostics(self) -> Dict[str, Any]:
        d = super().get_diagnostics()
        d["adapter"] = self.name
        d["ticks_no_shot"] = self.ticks_no_shot
        d["seen_shot"] = self.seen_shot
        d["timeout_count"] = self.timeout_count
        d["shot_taken_count"] = self.shot_taken_count
        d["shot_on_target_count"] = self.shot_on_target_count
        return d

    def check_shot_clock(self) -> Optional[float]:
        """Return timeout penalty once, after t_max ticks without a shot.

        The env applies the returned penalty to every agent's reward and
        emits truncated=True (not terminated) so GAE bootstraps. Repeated
        calls after the first return None (penalty applied exactly once).
        """
        if not self.seen_shot and self.ticks_no_shot > self.t_max:
            if self.timeout_count == 0:
                self.timeout_count += 1
                return self.timeout_penalty
        return None

    def _handle_shot(self, shaped, etype, aid, actions) -> None:
        """Finishing drill: never penalize a shot; reward attempts here.

        SHOT_TAKEN gets its positive reward in _pay_shots so the ordering
        with pass completion (chain-aware) stays single-source-of-truth.
        """
        # No solitary-shot penalty in this adapter. SHOT_TAKEN is paid
        # positively in _pay_shots. Count attempts here.
        if aid is not None:
            self.seen_shot = True
            self.ticks_no_shot = 0
            self.shot_taken_count += 1
        else:
            # No agent_id on the event: attribute to any agent that
            # actually commanded SHOT this frame (mirrors M1b).
            if actions:
                shooters = [ag for ag, ax in actions.items()
                            if ag in shaped and ax == SHOT_ACTION_ID]
                if shooters:
                    self.seen_shot = True
                    self.ticks_no_shot = 0
                    self.shot_taken_count += 1

    def _strip_progress(self, shaped, step_events) -> None:
        has_gp = any(isinstance(e, dict) and e.get("type") in GOAL_OR_PASS_TYPES
                     for e in step_events)
        if not has_gp:
            for a in shaped:
                shaped[a] = 0.0

    def _clear_shot_clock(self) -> None:
        self.seen_shot = True
        self.ticks_no_shot = 0

    def _pay_shot_rewards(self, shaped, step_events, active_agents) -> None:
        """Pay shot incentives and clear the shot-clock on any shot event."""
        for event in step_events:
            if not isinstance(event, dict):
                continue
            etype = event.get("type")
            aid = event.get("agent_id")
            if etype in SHOT_EVENT_TYPES:
                self._clear_shot_clock()
                if etype in ("SHOT_TAKEN", "SHOT_BLOCKED"):
                    if aid is not None:
                        self.shot_taken_count += 1
                    targets = ([aid] if aid in shaped
                               else [a for a in active_agents if a in shaped])
                    for t in targets:
                        shaped[t] += self.r_shot
                elif etype == "SHOT_SAVED":
                    self.shot_on_target_count += 1
                    targets = ([aid] if aid in shaped
                               else [a for a in active_agents if a in shaped])
                    # On target (keeper had to save it): full on-target reward.
                    for t in targets:
                        shaped[t] += self.r_on_target
                elif etype == "SHOT_MISSED":
                    # Missed: still an attempt, paid the attempt reward so
                    # the agent learns shooting is productive even when not
                    # on target (missed happens both on target and off).
                    targets = ([aid] if aid in shaped
                               else [a for a in active_agents if a in shaped])
                    for t in targets:
                        shaped[t] += max(self.r_shot, 0.0)

    def compute_shaped_rewards(self, base_rewards, step_events,
                               info_ground_truth, active_agents,
                               actions=None, tick=0, max_ticks=None):
        shaped = {a: base_rewards.get(a, 0.0) for a in active_agents
                  if not a.startswith("right_")}
        self._strip_progress(shaped, step_events)
        # No action cost: shot/pass/dribble are all allowed freely; the
        # -0.005 step cost is the only time pressure.
        self._apply_possession(shaped, info_ground_truth)
        self._handle_events(shaped, step_events, active_agents, actions)
        self._pay_shot_rewards(shaped, step_events, active_agents)
        # Do not charge step cost on a goal-scoring tick: time pressure is
        # meant to push toward completion, not tax the successful finish.
        has_goal = any(isinstance(e, dict) and e.get("type") == "GOAL_SCORED"
                       for e in step_events)
        if not has_goal:
            for a in shaped:
                shaped[a] += self.step_cost
        if not self.seen_shot:
            self.ticks_no_shot += 1
        return shaped


FINISHING_SCENARIOS = frozenset((
    "academy_3_vs_1_with_keeper",
    "academy_3_vs_1_keeper_aggressive",
    "academy_empty_goal",
    "academy_run_to_score",
    "academy_3_vs_1",
    "academy_2_vs_1",
    "5_vs_5",
    "11_vs_11",
))


def get_reward_adapter(scenario: str, **kw) -> BaseScenarioRewardAdapter:
    if scenario == "academy_rondo_4v1":
        return RondoRewardAdapter(**kw)
    return AttackingDrillRewardAdapter(**kw)


