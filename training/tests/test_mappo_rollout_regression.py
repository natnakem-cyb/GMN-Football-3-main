"""
GMN-Football-3 — MAPPO Rollout Regression Tests (G2/G6 batch-1 fixes)

- G2: episode termination mid-collection must not crash or corrupt buffers.
  Forces termination with a tiny max_steps scenario budget (short time limit
  => truncated quickly) and asserts collect_rollout completes with exact
  buffer shapes and all-finite values.
- G6: batched sub-envs that terminate must be reset and keep contributing
  data. Runs collect_rollout_batched long enough for at least one sub-env to
  terminate and asserts completed_episodes is non-empty AND the rollout kept
  producing valid transitions afterwards (no dead-env stall).
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv, OBSERVATION_DIM, ACTION_SPACE_SIZE
from training.mappo_networks import SharedActor, CentralizedCritic
from training.mappo_rollout import collect_rollout, collect_rollout_batched, unwrap_obs


def _make_nets(obs_dim=OBSERVATION_DIM, n_actions=ACTION_SPACE_SIZE, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return SharedActor(obs_dim=obs_dim, action_dim=n_actions), CentralizedCritic(obs_dim=obs_dim)


def test_unwrap_obs_handles_both_formats():
    """unwrap_obs is idempotent: envelope dicts and raw arrays both work."""
    raw = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    envelope = {
        "left_1": {"observation": raw, "action_mask": np.ones(19, dtype=np.int8)},
        "left_2": {"observation": raw + 1, "action_mask": np.ones(19, dtype=np.int8)},
    }
    out = unwrap_obs(envelope)
    assert set(out.keys()) == {"left_1", "left_2"}
    assert isinstance(out["left_1"], np.ndarray) and out["left_1"].shape == (OBSERVATION_DIM,)
    # Already-unwrapped input passes through unchanged.
    out2 = unwrap_obs({"left_1": raw})
    assert out2["left_1"] is raw


class FakeStepEnv:
    """Single-env fake mimicking the PettingZoo envelope contract exactly.

    Forces termination/truncation at deterministic tick counts — a random
    policy cannot reliably finish a 900-tick live scenario inside a short
    test, so the G2 mid-rollout terminal path is exercised here instead.
    """

    def __init__(self, n_agents=2, terminate_at=None, truncate_at=None):
        self.possible_agents = [f"left_{i}" for i in range(n_agents)]
        self.agents = list(self.possible_agents)
        self._t = 0
        self._terminate_at = terminate_at
        self._truncate_at = truncate_at
        # Alternate end condition per episode so both paths are exercised:
        # even-indexed episodes terminate, odd-indexed episodes truncate.
        self._episode_idx = 0

    def _obs(self):
        return {
            a: {
                "observation": np.full(OBSERVATION_DIM, 0.5, dtype=np.float32),
                "action_mask": np.ones(ACTION_SPACE_SIZE, dtype=np.int8),
            }
            for a in self.agents
        }

    def reset(self):
        self.agents = list(self.possible_agents)
        self._t = 0
        return self._obs(), {}

    def step(self, actions):
        self._t += 1
        if self._episode_idx % 2 == 0:
            term = self._terminate_at is not None and self._t >= self._terminate_at
            trunc = False
        else:
            term = False
            trunc = self._truncate_at is not None and self._t >= self._truncate_at
        obs = self._obs()
        rewards = {a: 0.25 for a in self.agents}
        terminations = {a: term for a in self.agents}
        truncations = {a: trunc for a in self.agents}
        infos = {a: {"score": {"left": 1 if term else 0}} for a in self.agents}
        if term or trunc:
            self.agents = []
            self._episode_idx += 1
        return obs, rewards, terminations, truncations, infos


class FakeBatchEnv:
    """Batched fake mimicking GMNMultiAgentEnv's batch reset/step contract.

    Sub-env 0 terminates every ``terminate_env0_at`` ticks; sub-env 1
    truncates every ``truncate_env1_at`` ticks. Tracks reset_one calls so the
    G6 assertion can prove each terminated sub-env was actually reset.
    """

    def __init__(self, batch_size=2, terminate_env0_at=7, truncate_env1_at=11):
        self.batch_size = batch_size
        self._term_at = terminate_env0_at
        self._trunc_at = truncate_env1_at
        self._t = [0] * batch_size
        self._agents = [[f"left_{i}" for i in range(2)] for _ in range(batch_size)]
        self.reset_one_called = []

    @property
    def agents(self):
        return [f"left_{i}" for i in range(2)]

    def _obs(self, i):
        return {
            a: {
                "observation": np.full(OBSERVATION_DIM, float(i + 1), dtype=np.float32),
                "action_mask": np.ones(ACTION_SPACE_SIZE, dtype=np.int8),
            }
            for a in self._agents[i]
        }

    def reset_batch(self, seeds=None, options=None):
        self._t = [0] * self.batch_size
        self._agents = [[f"left_{i}" for i in range(2)] for _ in range(self.batch_size)]
        # Real GMNMultiAgentEnv.reset_batch contract: list of per-env
        # (obs_dict, info_dict) tuples.
        return [(self._obs(i), {}) for i in range(self.batch_size)]

    def reset_one(self, env_idx, seed=None, options=None):
        self.reset_one_called.append(env_idx)
        self._t[env_idx] = 0
        self._agents[env_idx] = [f"left_{i}" for i in range(2)]
        return self._obs(env_idx), {}

    def step_batch(self, action_sets):
        results = []
        for i in range(self.batch_size):
            self._t[i] += 1
            term = self._term_at is not None and i == 0 and self._t[i] >= self._term_at
            trunc = self._trunc_at is not None and i == 1 and self._t[i] >= self._trunc_at
            obs = self._obs(i)
            rewards = {a: 0.5 for a in self._agents[i]}
            terminations = {a: term for a in self._agents[i]}
            truncations = {a: trunc for a in self._agents[i]}
            infos = {a: {"score": {"left": 1 if term else 0}} for a in self._agents[i]}
            if term or trunc:
                self._agents[i] = []
            results.append((obs, rewards, terminations, truncations, infos))
        return results


def test_collect_rollout_survives_episode_termination_mid_collection():
    """G2 regression: termination AND truncation inside num_steps must not
    crash or misshape the rollout — the collector must reset, unwrap, and
    keep filling exact-shape buffers across multiple episode boundaries."""
    env = FakeStepEnv(terminate_at=8, truncate_at=13)
    actor, critic = _make_nets()
    buf = collect_rollout(env, actor, critic, num_steps=40)
    n_agents = 2
    assert buf["local_obs"].shape == (40, n_agents, OBSERVATION_DIM)
    assert buf["global_state"].shape == (40, n_agents * OBSERVATION_DIM)
    assert buf["actions"].shape == (40, n_agents)
    assert buf["logprobs"].shape == (40, n_agents)
    assert buf["values"].shape == (40,)
    assert buf["rewards"].shape == (40,)
    assert buf["per_agent_rewards"].shape == (40, n_agents)
    assert buf["dones"].shape == (40,)
    assert buf["next_local_obs"].shape == (n_agents, OBSERVATION_DIM)
    # Termination at t=8, truncation at t=13 -> episodes of length 8, 13, 8
    # complete inside 40 steps; the 4th is cut off by num_steps.
    assert len(buf["completed_episodes"]) == 3, (
        f"expected 3 completed episodes, got {len(buf['completed_episodes'])}"
    )
    assert [e["length"] for e in buf["completed_episodes"]] == [8, 13, 8]
    assert int(buf["dones"].sum()) == 2, "dones must count terminations only"
    assert int(buf["truncated"].sum()) == 1, "truncations must be recorded separately"
    assert np.all(np.isfinite(buf["local_obs"]))
    assert np.all(np.isfinite(buf["rewards"]))
    assert np.all(np.isfinite(buf["next_local_obs"]))
    # Persistent state left behind must be unwrapped raw arrays, not envelopes.
    for v in env._mappo_obs.values():
        assert isinstance(v, np.ndarray), f"_mappo_obs holds envelope dict: {type(v)}"


def test_collect_rollout_batched_resets_terminated_subenvs():
    """G6 regression: each terminated sub-env must be reset via reset_one and
    keep contributing exactly one transition per tick (no dead-env stall)."""
    env = FakeBatchEnv(batch_size=2, terminate_env0_at=7, truncate_env1_at=11)
    actor, critic = _make_nets()
    num_steps = 25
    buf = collect_rollout_batched(env, actor, critic, batch_size=2, num_steps=num_steps)
    n_agents = 2
    # Sub-env 0: terminates at t=7,14,21 -> 3 episodes; sub-env 1: truncates
    # at t=11,22 -> 2 episodes. All inside 25 batched steps.
    assert len(buf["completed_episodes"]) == 3 + 2, (
        f"expected 5 completed episodes, got {len(buf['completed_episodes'])}"
    )
    # Episodes append per-tick across envs (env0 t=7, env1 t=11, env0 t=14,
    # env0 t=21, env1 t=22) — same envs, interleaved order.
    assert sorted(e["env"] for e in buf["completed_episodes"]) == [0, 0, 0, 1, 1]
    # Each terminal sub-env was individually reset through reset_one.
    assert env.reset_one_called.count(0) == 3, "sub-env 0 not reset after every termination"
    assert env.reset_one_called.count(1) == 2, "sub-env 1 not reset after every truncation"
    # Flat layout identical to collect_rollout_parallel: every env contributes
    # every tick -> exactly num_steps * batch_size rows, no idling.
    assert buf["total_steps"] == num_steps * 2
    assert buf["local_obs"].shape == (num_steps * 2, n_agents, OBSERVATION_DIM)
    assert buf["global_state"].shape == (num_steps * 2, n_agents * OBSERVATION_DIM)
    assert buf["actions"].shape == (num_steps * 2, n_agents)
    assert buf["logprobs"].shape == (num_steps * 2, n_agents)
    assert buf["values"].shape == (num_steps * 2,)
    assert buf["rewards"].shape == (num_steps * 2,)
    assert buf["per_agent_rewards"].shape == (num_steps * 2, n_agents)
    assert buf["dones"].shape == (num_steps * 2,)
    assert buf["next_local_obs"].shape == (n_agents, OBSERVATION_DIM)
    assert int(buf["dones"].sum()) == 3
    assert int(buf["truncated"].sum()) == 2
    assert np.all(np.isfinite(buf["local_obs"]))
    assert np.all(np.isfinite(buf["rewards"]))
    assert np.all(np.isfinite(buf["next_local_obs"]))


def test_live_smoke_single_env_collection():
    """Live-bridge smoke: single collector runs against the real engine."""
    port = 5070 + abs(hash("g2_live_smoke")) % 500
    env = GMNMultiAgentEnv(
        scenario="academy_empty_goal",
        auto_start_bridge=True,
        port=port,
        enable_reward_shaping=False,
    )
    try:
        actor, critic = _make_nets()
        buf = collect_rollout(env, actor, critic, num_steps=16)
        n_agents = buf["local_obs"].shape[1]
        assert buf["local_obs"].shape == (16, n_agents, OBSERVATION_DIM)
        assert np.all(np.isfinite(buf["local_obs"]))
    finally:
        env.close()


def test_live_smoke_batched_env_collection():
    """Live-bridge smoke: batched collector runs against the real engine."""
    port = 5080 + abs(hash("g6_live_smoke")) % 500
    env = GMNMultiAgentEnv(
        scenario="academy_empty_goal",
        auto_start_bridge=True,
        port=port,
        batch_size=2,
        enable_reward_shaping=False,
    )
    try:
        actor, critic = _make_nets()
        buf = collect_rollout_batched(env, actor, critic, batch_size=2, num_steps=16)
        n_agents = buf["local_obs"].shape[1]
        assert buf["total_steps"] == 16 * 2
        assert buf["local_obs"].shape == (16 * 2, n_agents, OBSERVATION_DIM)
        assert np.all(np.isfinite(buf["local_obs"]))
        state = env._batch_envs[0]
        assert len(state["agents"]) > 0, "sub-env left dead after live collection"
    finally:
        env.close()
