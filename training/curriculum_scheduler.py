"""
GMN-Football-3 — Curriculum Scheduler

Performance-threshold-driven promotion/demotion system for the academy stage ladder.
Persists state to JSON so training runs can resume mid-curriculum after a crash.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional


# Stage ladder derived from ScenarioRegistry.ts (academy_rondo_4v1 is excluded as a parallel track).
CURRICULUM_STAGES: List[str] = [
    "academy_empty_goal",
    "academy_run_to_score",
    "academy_pass_and_shoot_with_keeper",
    "academy_3_vs_1_with_keeper",
    "academy_3_vs_1_defender_2",
    "academy_3_vs_1_defender_3",
    "5_vs_5",
    "11_vs_11",
]


def is_scenario_success(scenario_id: str, info: Dict[str, Any]) -> bool:
    """
    Determine whether an episode counts as a 'success' for curriculum purposes.

    Uses only fields already returned by gmn_pettingzoo.py step()/info dict:
    - score.left / score.right
    - event.type

    Rules:
    - Stages with a 'score_goal' objective (academy_empty_goal,
      academy_run_to_score, academy_pass_and_shoot_with_keeper,
      academy_3_vs_1_with_keeper, academy_3_vs_1_defender_2,
      academy_3_vs_1_defender_3): success = left team scored (score.left > 0).
    - Stages with a 'win_match' objective (5_vs_5, 11_vs_11):
      success = left team won (score.left > score.right).
    """
    score = info.get("score", {}) if isinstance(info, dict) else {}
    left = int(score.get("left", 0)) if isinstance(score, dict) else 0
    right = int(score.get("right", 0)) if isinstance(score, dict) else 0

    if scenario_id in ("5_vs_5", "11_vs_11"):
        return left > right
    return left > 0


class CurriculumScheduler:
    """
    Rolling-window curriculum scheduler with promotion, demotion, and persistence.

    Promotion: when the rolling success rate exceeds ``promote_threshold`` over
    ``window_size`` episodes AND at least ``min_episodes_before_promotion``
    episodes have been recorded.
    Demotion: when the rolling success rate drops below ``demote_threshold``
    after a promotion (regression protection).
    """

    def __init__(
        self,
        stages: List[str] = None,
        window_size: int = 100,
        promote_threshold: float = 0.6,
        demote_threshold: float = 0.1,
        min_episodes_before_promotion: int = 200,
    ):
        self.stages = list(stages or CURRICULUM_STAGES)
        self.window_size = window_size
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold
        self.min_episodes = min_episodes_before_promotion

        self.current_idx: int = 0
        self.window: List[bool] = []
        self.total_episodes: int = 0
        self.episodes_in_stage: int = 0
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record_result(self, success: bool, goal_scored: bool = False) -> None:
        """
        Record the outcome of one completed episode.

        Parameters
        ----------
        success : bool
            Whether the episode satisfied the current stage's primary objective.
        goal_scored : bool
            Legacy/compatibility signal: whether the left team scored at least
            once during the episode. Kept for logging/debugging.
        """
        self.window.append(success)
        self.total_episodes += 1
        self.episodes_in_stage += 1
        if len(self.window) > self.window_size:
            self.window.pop(0)

    def evaluate_and_step(self) -> str:
        """
        Evaluate the rolling window and promote/demote if thresholds are met.

        Returns
        -------
        str
            The current stage's scenario id after any promotion/demotion.
        """
        current = self.stages[self.current_idx]

        if self.episodes_in_stage < self.min_episodes or not self.window:
            return current

        success_rate = sum(self.window) / len(self.window)

        # Promote
        if (
            self.current_idx < len(self.stages) - 1
            and success_rate >= self.promote_threshold
        ):
            old_stage = self.stages[self.current_idx]
            self.current_idx += 1
            new_stage = self.stages[self.current_idx]
            self.window = []
            self.episodes_in_stage = 0
            self.history.append({
                "timestamp": time.time(),
                "from": old_stage,
                "to": new_stage,
                "success_rate": success_rate,
                "type": "promote",
                "total_episodes": self.total_episodes,
            })
            return new_stage

        # Demote (regression protection)
        if (
            self.current_idx > 0
            and success_rate <= self.demote_threshold
        ):
            old_stage = self.stages[self.current_idx]
            self.current_idx -= 1
            new_stage = self.stages[self.current_idx]
            self.window = []
            self.episodes_in_stage = 0
            self.history.append({
                "timestamp": time.time(),
                "from": old_stage,
                "to": new_stage,
                "success_rate": success_rate,
                "type": "demote",
                "total_episodes": self.total_episodes,
            })
            return new_stage

        return current

    @property
    def current_stage(self) -> str:
        return self.stages[self.current_idx]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stages": self.stages,
            "current_idx": self.current_idx,
            "window": self.window,
            "total_episodes": self.total_episodes,
            "episodes_in_stage": self.episodes_in_stage,
            "history": self.history,
            "window_size": self.window_size,
            "promote_threshold": self.promote_threshold,
            "demote_threshold": self.demote_threshold,
            "min_episodes": self.min_episodes,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> Optional["CurriculumScheduler"]:
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            state = json.load(f)
        scheduler = cls(
            stages=state.get("stages", CURRICULUM_STAGES),
            window_size=state.get("window_size", 100),
            promote_threshold=state.get("promote_threshold", 0.6),
            demote_threshold=state.get("demote_threshold", 0.1),
            min_episodes_before_promotion=state.get("min_episodes", 200),
        )
        scheduler.current_idx = state.get("current_idx", 0)
        scheduler.window = [bool(x) for x in state.get("window", [])]
        scheduler.total_episodes = int(state.get("total_episodes", 0))
        scheduler.episodes_in_stage = int(state.get("episodes_in_stage", 0))
        scheduler.history = state.get("history", [])
        return scheduler
