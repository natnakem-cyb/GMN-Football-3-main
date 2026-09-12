"""
Audit P0 Issue 1 — curriculum success propagation tests.

Verifies that ALL THREE rollout collectors (single, parallel, batched) emit a
"success" field in completed_episodes, computed via is_scenario_success with
the correct per-stage rules:

- 5_vs_5 / 11_vs_11: left > right (win), even when the terminal event is
  scenario_complete rather than 'goal'.
- academy_rondo_4v1: score.right == 0 AND terminal event scenario_complete
  (no goal is possible; 'goal' must NOT be required).
- Academy goal stages: score.left > 0.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mappo_rollout import (  # noqa: E402
    collect_rollout,
    collect_rollout_parallel,
    collect_rollout_batched,
)

OBS_DIM = 4
N_ACTIONS = 5


# ---------------------------------------------------------------------------
# Minimal actor/critic stand-ins matching the rollout collectors' interface
# ---------------------------------------------------------------------------

class DummyActor:
    """Returns a uniform Categorical over N_ACTIONS for the agent batch."""

    def __call__(self, obs: torch.Tensor):
        n = obs.shape[0]
        return torch.distributions.Categorical(
            logits=torch.zeros(n, N_ACTIONS)
        )


class DummyCritic:
    """Returns a scalar value for any (1, ...) or flattened input."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1)


# ---------------------------------------------------------------------------
# Fake envs matching each collector's step contract
# ---------------------------------------------------------------------------

def _obs_envelope(agents: List[str]) -> Dict[str, Dict[str, Any]]:
    """Envelope observation format emitted by the real bridge (G2 fix)."""
    return {
        a: {"observation": np.zeros(OBS_DIM, dtype=np.float32), "action_mask": None}
        for a in agents
    }


class FakeStepEnv:
    """
    PettingZoo-style env for collect_rollout / collect_rollout_parallel.

    Runs `episode_length` steps per episode; the final step is terminal and
    carries the configured terminal info (score + event type) in every
    agent's info dict — same shared score/event contract as gmn_pettingzoo.
    """

    def __init__(
        self,
        agents: List[str],
        scenario_id: str,
        terminal_score: Dict[str, int],
        terminal_event_type: str,
        episode_length: int = 2,
        episode_reward: float = 0.1,
    ):
        self.agents = list(agents)
        self.possible_agents = list(agents)
        self.scenario = scenario_id
        self.terminal_score = dict(terminal_score)
        self.terminal_event_type = terminal_event_type
        self.episode_length = episode_length
        self.episode_reward = episode_reward
        self.t = 0
        self.episodes_finished = 0

    # -- PettingZoo API ------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.t = 0
        return _obs_envelope(self.agents), {}

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        self.t += 1
        terminal = self.t >= self.episode_length
        obs = _obs_envelope(self.agents)
        rewards = {a: self.episode_reward for a in self.agents}
        terminations = {a: terminal for a in self.agents}
        truncations = {a: False for a in self.agents}
        if terminal:
            info = {
                "score": dict(self.terminal_score),
                "event": {"type": self.terminal_event_type},
            }
            self.episodes_finished += 1
        else:
            info = {"score": {"left": 0, "right": 0}, "event": {"type": ""}}
        infos = {a: dict(info) for a in self.agents}
        return obs, rewards, terminations, truncations, infos

class FakeBatchedEnv:
    """
    Batched-wire env for collect_rollout_batched.

    Contract (from bridge_server stepBatch / resetBatch / resetOne):
    - reset_batch(seeds) -> List[(obs_envelope, info)]
    - step_batch(action_sets) -> List[(obs_envelope, reward, terminated,
      truncated, info)] with per-agent dicts for reward/terminated/truncated.
    - reset_one(idx, seed) -> (obs_envelope, info)
    """

    def __init__(
        self,
        agents: List[str],
        scenario_id: str,
        n_sub_envs: int,
        terminal_score: Dict[str, int],
        terminal_event_type: str,
        episode_length: int = 2,
        episode_reward: float = 0.1,
    ):
        self.agents = list(agents)
        self.scenario = scenario_id
        self.n_sub_envs = n_sub_envs
        self.terminal_score = dict(terminal_score)
        self.terminal_event_type = terminal_event_type
        self.episode_length = episode_length
        self.episode_reward = episode_reward
        self._t = [0] * n_sub_envs

    def reset_batch(self, seeds: List[int]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        self._t = [0] * self.n_sub_envs
        return [(self._obs(), {}) for _ in range(self.n_sub_envs)]

    def reset_one(self, idx: int, seed: int = 0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._t[idx] = 0
        return self._obs(), {}

    def _obs(self) -> Dict[str, Dict[str, Any]]:
        return _obs_envelope(self.agents)

    def step_batch(
        self, action_sets: List[Dict[str, int]]
    ) -> List[Tuple[Dict[str, Any], Any, Any, Any, Dict[str, Any]]]:
        results = []
        for idx, action_dict in enumerate(action_sets):
            self._t[idx] += 1
            terminal = self._t[idx] >= self.episode_length
            obs = self._obs()
            # Real contract (gmn_pettingzoo.step_batch, lines 701-711): ONE
            # shared top-level info dict per sub-env, with score and a
            # conditional event dict.
            if terminal:
                info = {
                    "score": dict(self.terminal_score),
                    "eventCode": 0,
                    "event": {"type": self.terminal_event_type},
                }
            else:
                info = {"score": {"left": 0, "right": 0}, "eventCode": 0}
            reward = {a: self.episode_reward for a in self.agents}
            terminated = {a: terminal for a in self.agents}
            truncated = {a: False for a in self.agents}
            results.append((obs, reward, terminated, truncated, info))
        return results


AGENTS = ["agent_0", "agent_1"]

# ---------------------------------------------------------------------------
# Tests: is_scenario_success wiring in all three collectors
# ---------------------------------------------------------------------------

def _assert_last_episode(buffer: Dict[str, Any], expected_success: bool,
                         expected_goal: Optional[int] = None,
                         context: str = "") -> None:
    eps = buffer["completed_episodes"]
    assert len(eps) > 0, f"{context}: no completed episodes emitted"
    last = eps[-1]
    assert "success" in last, (
        f"{context}: completed_episodes entry missing 'success' field: {last}"
    )
    assert last["success"] is expected_success, (
        f"{context}: success={last['success']!r}, expected {expected_success} (entry={last})"
    )
    if expected_goal is not None:
        assert last["goal"] == expected_goal, (
            f"{context}: goal={last['goal']!r}, expected {expected_goal} (entry={last})"
        )


def test_single_collector_match_stage_scenario_complete_is_success():
    """5_vs_5: left 2, right 1, terminal scenario_complete -> success True."""
    env = FakeStepEnv(
        AGENTS, "5_vs_5",
        terminal_score={"left": 2, "right": 1},
        terminal_event_type="scenario_complete",
    )
    buffer = collect_rollout(env, DummyActor(), DummyCritic(), num_steps=2)
    # NOTE: 'goal' is a metrics field (left scored at all), not the success
    # signal — with left=2 goals it is legitimately 1 even though the terminal
    # event is scenario_complete. The success assertion is the contract under
    # test here.
    _assert_last_episode(buffer, expected_success=True,
                         context="single/5_vs_5 win via scenario_complete")


def test_single_collector_rondo_no_goal_required():
    """Rondo: right==0 + scenario_complete -> success True; goal not required."""
    env = FakeStepEnv(
        AGENTS, "academy_rondo_4v1",
        terminal_score={"left": 0, "right": 0},
        terminal_event_type="scenario_complete",
    )
    buffer = collect_rollout(env, DummyActor(), DummyCritic(), num_steps=2)
    _assert_last_episode(buffer, expected_success=True, expected_goal=0,
                         context="single/rondo success without any goal")


def test_single_collector_academy_goal_stage():
    """Academy goal stage: score.left > 0 -> success True (terminal 'goal')."""
    env = FakeStepEnv(
        AGENTS, "academy_3_vs_1_with_keeper",
        terminal_score={"left": 1, "right": 0},
        terminal_event_type="goal",
    )
    buffer = collect_rollout(env, DummyActor(), DummyCritic(), num_steps=2)
    _assert_last_episode(buffer, expected_success=True, expected_goal=1,
                         context="single/academy goal stage")


def test_parallel_collector_match_stage_scenario_complete_is_success():
    """Parallel 5_vs_5: left 2, right 1, scenario_complete -> success True."""
    envs = [
        FakeStepEnv(
            AGENTS, "5_vs_5",
            terminal_score={"left": 2, "right": 1},
            terminal_event_type="scenario_complete",
        )
        for _ in range(2)
    ]
    buffer = collect_rollout_parallel(envs, DummyActor(), DummyCritic(), num_steps=2)
    _assert_last_episode(buffer, expected_success=True, expected_goal=0,
                         context="parallel/5_vs_5 win via scenario_complete")
    for ep in buffer["completed_episodes"]:
        assert ep["success"] is True, f"parallel: entry success wrong: {ep}"

def test_parallel_collector_rondo_no_goal_required():
    """Parallel rondo: right==0 + scenario_complete -> success True, goal 0."""
    envs = [
        FakeStepEnv(
            AGENTS, "academy_rondo_4v1",
            terminal_score={"left": 0, "right": 0},
            terminal_event_type="scenario_complete",
        )
        for _ in range(2)
    ]
    buffer = collect_rollout_parallel(envs, DummyActor(), DummyCritic(), num_steps=2)
    _assert_last_episode(buffer, expected_success=True, expected_goal=0,
                         context="parallel/rondo success without any goal")


def test_parallel_collector_losing_match_is_not_success():
    """Parallel 5_vs_5 loss: left 1, right 2 -> success False."""
    envs = [
        FakeStepEnv(
            AGENTS, "5_vs_5",
            terminal_score={"left": 1, "right": 2},
            terminal_event_type="scenario_complete",
        )
        for _ in range(2)
    ]
    buffer = collect_rollout_parallel(envs, DummyActor(), DummyCritic(), num_steps=2)
    _assert_last_episode(buffer, expected_success=False, expected_goal=0,
                         context="parallel/5_vs_5 loss must not be success")


def test_batched_collector_match_stage_scenario_complete_is_success():
    """Batched 5_vs_5: left 2, right 1, scenario_complete -> success True."""
    env = FakeBatchedEnv(
        AGENTS, "5_vs_5", n_sub_envs=2,
        terminal_score={"left": 2, "right": 1},
        terminal_event_type="scenario_complete",
    )
    buffer = collect_rollout_batched(env, DummyActor(), DummyCritic(),
                                     num_steps=2, batch_size=2)
    # goal metric tracks score.left > 0: a 5_vs_5 win (2-1) legitimately has
    # goal=1. success=True is the important assertion (scenario_complete +
    # left > right), independent of the goal metric.
    _assert_last_episode(buffer, expected_success=True, expected_goal=1,
                         context="batched/5_vs_5 win via scenario_complete")
    envs_reporting = {ep["env"] for ep in buffer["completed_episodes"]}
    assert envs_reporting == {0, 1}, (
        f"batched: expected episodes from both sub-envs, got {envs_reporting}"
    )


def test_batched_collector_rondo_no_goal_required():
    """Batched rondo: right==0 + scenario_complete -> success True, goal 0."""
    env = FakeBatchedEnv(
        AGENTS, "academy_rondo_4v1", n_sub_envs=2,
        terminal_score={"left": 0, "right": 0},
        terminal_event_type="scenario_complete",
    )
    buffer = collect_rollout_batched(env, DummyActor(), DummyCritic(),
                                     num_steps=2, batch_size=2)
    _assert_last_episode(buffer, expected_success=True, expected_goal=0,
                         context="batched/rondo success without any goal")


def test_batched_collector_academy_goal_stage():
    """Batched academy goal stage: score.left > 0 -> success True, goal 1."""
    env = FakeBatchedEnv(
        AGENTS, "academy_3_vs_1_with_keeper", n_sub_envs=2,
        terminal_score={"left": 1, "right": 0},
        terminal_event_type="goal",
    )
    buffer = collect_rollout_batched(env, DummyActor(), DummyCritic(),
                                     num_steps=2, batch_size=2)
    _assert_last_episode(buffer, expected_success=True, expected_goal=1,
                         context="batched/academy goal stage")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)



