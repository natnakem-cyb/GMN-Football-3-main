"""Focused integration coverage for GNN environment, MAPPO and checkpoints."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from training.checkpoint_contract import (
    create_policy_checkpoint_contract,
    load_mappo_actor,
    load_mappo_critic,
    validate_policy_checkpoint,
)
from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.gnn_graph_to_tensor import GraphTensor
from training.gnn_mappo_networks import GNNMAPPOActor, GNNMAPPOCritic
from training.mappo_networks import SharedActor
from training.mappo_rollout import collect_rollout, collect_rollout_batched, compute_gae
from training.mappo_update import ppo_update


AGENT = "left_1"
MASK = np.asarray([1] * 9 + [0] * 10, dtype=np.int8)


def test_bridge_cleanup_finds_windows_listening_pid_on_exact_port():
    lines = [
        "  TCP    0.0.0.0:53145      0.0.0.0:0      LISTENING       24680",
        "  TCP    0.0.0.0:53146      0.0.0.0:0      LISTENING       24681",
        "  TCP    127.0.0.1:53145    127.0.0.1:80   ESTABLISHED     24682",
    ]
    assert GMNMultiAgentEnv._parse_netstat_listening_pids(lines, 53145) == {"24680"}


def _graph(value: float = 0.0) -> GraphTensor:
    nodes = torch.full((3, 32), float(value), dtype=torch.float32)
    nodes[0, 17] = 1.0  # controlled-player feature
    return GraphTensor(
        node_features=nodes,
        node_type=torch.tensor([0, 0, 1], dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_type=torch.empty((0,), dtype=torch.long),
        edge_features=torch.empty((0, 10), dtype=torch.float32),
        node_mask=torch.ones(3, dtype=torch.float32),
        agent_node_indices=[0],
        graph_context=torch.full((8,), float(value), dtype=torch.float32),
        node_id_to_index={"left_0": 0, "left_1": 1, "ball": 2},
        scenario_id="academy_empty_goal",
    )


class _SocketStub:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)


def _uninitialized_env(graph_enabled: bool) -> GMNMultiAgentEnv:
    """Build the PettingZoo wrapper without starting its external bridge."""
    env = GMNMultiAgentEnv.__new__(GMNMultiAgentEnv)
    env.include_graph_observations = graph_enabled
    env.scenario = "academy_3_vs_1_with_keeper_onball"
    env.batch_size = 1
    env._needs_bridge = False
    env._episode_index = 0
    env._step_count = 0
    env.reward_components = []
    env.reward_shaper = None
    env.reward_adapter = None
    env.opponent_pool = None
    env._pending_pass = None
    env.enable_reward_shaping = False
    env.enable_exploration_bonus = False
    env.exploration_beta = 0.0
    env.debug_rewards = False
    env.agents = []
    env.possible_agents = []
    env.ws_client = _SocketStub()
    return env


def _reset_payload():
    return {
        "observations": [np.zeros(127, dtype=np.float32).tolist()],
        "action_masks": [MASK.tolist()],
        "info": {"controllableAgentIds": [AGENT], "score": {"left": 0, "right": 0}},
    }


def _step_frame() -> bytes:
    header = struct.pack("<f??BBffBB", 0.25, False, False, 0, 0, 0.0, 0.5, 0, 255)
    observation = np.full(127, 0.125, dtype="<f4").tobytes()
    return header + observation + MASK.astype("<u1").tobytes()


@pytest.mark.parametrize("graph_enabled", [False, True])
def test_gap1_reset_and_step_preserve_flat_contract_and_attach_graphs(
    monkeypatch, graph_enabled
):
    from training import gnn_graph_builder

    graph = {"sentinel": "graph"}
    calls = []

    def build_graph(observation, info, scenario):
        calls.append((observation, info, scenario))
        return graph

    monkeypatch.setattr(gnn_graph_builder, "build_graph", build_graph)
    env = _uninitialized_env(graph_enabled)
    env._recv_reset_response = _reset_payload

    reset_obs, reset_infos = env.reset(seed=3)
    assert reset_obs[AGENT]["observation"].shape == (127,)
    assert np.array_equal(reset_obs[AGENT]["action_mask"], MASK)
    if graph_enabled:
        assert reset_infos[AGENT]["graph_observation"] is graph
    else:
        assert "graph_observation" not in reset_infos[AGENT]

    env._recv_step_response = lambda num_agents: (_step_frame(), None, 0.0, None, None)
    step_obs, _, _, _, step_infos = env.step({AGENT: 0})
    assert step_obs[AGENT]["observation"].shape == (127,)
    assert np.array_equal(step_obs[AGENT]["action_mask"], MASK)
    if graph_enabled:
        assert step_infos[AGENT]["graph_observation"] is graph
        assert len(calls) == 2
        assert all(call[2] == "academy_3_vs_1_with_keeper" for call in calls)
        assert all(call[1]["controlledPlayerId"] == "left_0" for call in calls)
    else:
        assert "graph_observation" not in step_infos[AGENT]
        assert calls == []


@pytest.mark.parametrize("graph_enabled", [False, True])
def test_gap1_batched_reset_attaches_graphs_without_changing_flat_observations(
    monkeypatch, graph_enabled
):
    from training import gnn_graph_builder

    graph = {"batch": "graph"}
    calls = []
    monkeypatch.setattr(
        gnn_graph_builder,
        "build_graph",
        lambda observation, info, scenario: calls.append(
            (observation, info, scenario)
        ) or graph,
    )
    env = _uninitialized_env(graph_enabled)
    env.batch_size = 2
    env._scenario_adapter = lambda previous=None: None
    env._recv_reset_batch_response = lambda: __import__("json").dumps(
        {
            "results": [
                {
                    "observations": [np.zeros(127, dtype=np.float32).tolist()],
                    "action_masks": [MASK.tolist()],
                    "info": {"controllableAgentIds": [AGENT]},
                }
                for _ in range(2)
            ]
        }
    )

    results = env.reset_batch([10, 11])
    assert len(results) == 2
    for observations, infos in results:
        assert observations[AGENT]["observation"].shape == (127,)
        assert np.array_equal(observations[AGENT]["action_mask"], MASK)
        if graph_enabled:
            assert infos[AGENT]["graph_observation"] is graph
        else:
            assert "graph_observation" not in infos[AGENT]
    assert len(calls) == (2 if graph_enabled else 0)


def test_gap1_graph_helper_accepts_batched_step_shared_info_shape(monkeypatch):
    from training import gnn_graph_builder

    graph = {"batch": "step-graph"}
    monkeypatch.setattr(gnn_graph_builder, "build_graph", lambda *args: graph)
    env = _uninitialized_env(graph_enabled=True)
    observations = {AGENT: {"observation": np.zeros(127, dtype=np.float32)}}
    shared_info = {"score": {"left": 0, "right": 0}}

    env._attach_graph_observations(observations, shared_info)

    assert shared_info["graph_observations"] == {AGENT: graph}


class _FakeGraphEnv:
    """Two-step, single-agent PettingZoo fixture for the GNN rollout/update."""

    include_graph_observations = True
    scenario = "academy_empty_goal"
    possible_agents = [AGENT]

    def __init__(self):
        self.agents = [AGENT]
        self.tick = 0

    def _observations(self):
        obs = np.full(127, self.tick / 10.0, dtype=np.float32)
        return {AGENT: {"observation": obs, "action_mask": MASK.copy()}}

    def _infos(self):
        return {AGENT: {"graph_observation": _graph(self.tick / 10.0)}}

    def reset(self, seed=None):
        self.tick = 0
        return self._observations(), self._infos()

    def step(self, actions):
        self.tick += 1
        return (
            self._observations(),
            {AGENT: float(self.tick)},
            {AGENT: False},
            {AGENT: False},
            self._infos(),
        )


class _FakeGraphBatchedEnv:
    """Two graph-enabled sub-environments with independent episode clocks."""

    include_graph_observations = True
    scenario = "academy_empty_goal"
    possible_agents = [AGENT]

    def __init__(self, batch_size=2):
        self.batch_size = batch_size
        self.ticks = [0] * batch_size

    def _observations(self, env_idx):
        obs = np.full(127, (self.ticks[env_idx] + env_idx) / 10.0, dtype=np.float32)
        return {AGENT: {"observation": obs, "action_mask": MASK.copy()}}

    def _graph_info(self, env_idx):
        return {AGENT: {"graph_observation": _graph(self.ticks[env_idx] + env_idx)}}

    def reset_batch(self, seeds=None):
        self.ticks = [0] * self.batch_size
        return [
            (self._observations(i), self._graph_info(i))
            for i in range(self.batch_size)
        ]

    def step_batch(self, action_sets):
        results = []
        for env_idx in range(self.batch_size):
            self.ticks[env_idx] += 1
            terminated = self.ticks[env_idx] == env_idx + 2
            graphs = {AGENT: _graph(self.ticks[env_idx] + env_idx)}
            results.append((
                self._observations(env_idx),
                {AGENT: float(self.ticks[env_idx])},
                {AGENT: terminated},
                {AGENT: False},
                {"graph_observations": graphs},
            ))
        return results

    def reset_one(self, env_idx, seed=None):
        self.ticks[env_idx] = 0
        return self._observations(env_idx), self._graph_info(env_idx)


@pytest.mark.parametrize("encoder_type", ["mlp", "gat", "geometry"])
def test_gap2_gnn_rollout_and_ppo_update_train_actor_and_critic(encoder_type):
    torch.manual_seed(17)
    np.random.seed(17)
    env = _FakeGraphEnv()
    actor = GNNMAPPOActor(encoder_type=encoder_type, hidden_dim=16)
    critic = GNNMAPPOCritic(encoder_type=encoder_type, hidden_dim=16)
    buffer = collect_rollout(env, actor, critic, num_steps=3)

    assert len(buffer["graph_observations"]) == 3
    assert len(buffer["next_graph_observations"]) == 1
    actor_before = {key: value.detach().clone() for key, value in actor.state_dict().items()}
    critic_before = {key: value.detach().clone() for key, value in critic.state_dict().items()}
    actor_opt = torch.optim.SGD(actor.parameters(), lr=0.02)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=0.02)

    metrics = ppo_update(
        actor,
        critic,
        actor_opt,
        critic_opt,
        buffer,
        advantages=np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
        returns=np.asarray([0.5, 1.0, -0.5], dtype=np.float32),
        n_epochs=1,
        batch_size=3,
        entropy_coef=0.0,
        max_grad_norm=None,
    )

    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert any(not torch.equal(actor_before[k], value) for k, value in actor.state_dict().items())
    assert any(not torch.equal(critic_before[k], value) for k, value in critic.state_dict().items())


def test_gap2_graph_batched_rollout_ppo_update_and_per_env_gae():
    torch.manual_seed(31)
    np.random.seed(31)
    env = _FakeGraphBatchedEnv(batch_size=2)
    actor = GNNMAPPOActor(encoder_type="mlp", hidden_dim=16)
    critic = GNNMAPPOCritic(encoder_type="mlp", hidden_dim=16)
    buffer = collect_rollout_batched(
        env, actor, critic, num_steps=3, batch_size=2
    )

    assert buffer["graph_observations"] and len(buffer["graph_observations"]) == 6
    assert buffer["sequence_ids"].tolist() == [0, 1, 0, 1, 0, 1]
    assert buffer["episode_ends"].tolist() == [False, False, True, False, False, True]
    assert buffer["next_values"].shape == (6,)

    actor_before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    critic_before = {k: v.detach().clone() for k, v in critic.state_dict().items()}
    advantages, returns = compute_gae(
        rewards=buffer["rewards"],
        values=buffer["values"],
        dones=buffer["dones"],
        per_agent_rewards=buffer["per_agent_rewards"],
        next_values=buffer["next_values"],
        sequence_ids=buffer["sequence_ids"],
        episode_ends=buffer["episode_ends"],
    )
    metrics = ppo_update(
        actor,
        critic,
        torch.optim.SGD(actor.parameters(), lr=0.02),
        torch.optim.SGD(critic.parameters(), lr=0.02),
        buffer,
        advantages=advantages,
        returns=returns,
        n_epochs=1,
        batch_size=6,
        entropy_coef=0.0,
        max_grad_norm=None,
    )
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert any(not torch.equal(actor_before[k], v) for k, v in actor.state_dict().items())
    assert any(not torch.equal(critic_before[k], v) for k, v in critic.state_dict().items())

    # Interleaved env transitions must accumulate only within their own env.
    rewards = np.asarray([1, 10, 2, 20], dtype=np.float32)
    gae, _ = compute_gae(
        rewards=rewards,
        values=np.zeros(4, dtype=np.float32),
        dones=np.zeros(4, dtype=bool),
        per_agent_rewards=rewards[:, None],
        gamma=1.0,
        lam=1.0,
        next_values=np.zeros(4, dtype=np.float32),
        sequence_ids=np.asarray([0, 1, 0, 1]),
        episode_ends=np.zeros(4, dtype=bool),
    )
    assert np.array_equal(
        gae[:, 0], np.asarray([3, 30, 2, 20], dtype=np.float32)
    )


@pytest.mark.parametrize("architecture", ["gnn:mlp", "gnn:gat", "gnn:geometry"])
def test_gap3_versioned_gnn_checkpoint_contract_loads_models(architecture):
    encoder_type = architecture.split(":", 1)[1]
    actor = GNNMAPPOActor(encoder_type=encoder_type)
    critic = GNNMAPPOCritic(encoder_type=encoder_type)
    checkpoint = {
        "policy_architecture": architecture,
        "checkpoint_contract": create_policy_checkpoint_contract(architecture),
        "obs_dim": 127,
        "action_dim": 19,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
    }

    assert validate_policy_checkpoint(checkpoint) == (True, None)
    assert type(load_mappo_actor(checkpoint)) is GNNMAPPOActor
    assert type(load_mappo_critic(checkpoint)) is GNNMAPPOCritic

    incompatible = dict(checkpoint)
    incompatible["checkpoint_contract"] = dict(checkpoint["checkpoint_contract"])
    incompatible["checkpoint_contract"]["node_feature_dim"] += 1
    valid, reason = validate_policy_checkpoint(incompatible)
    assert not valid
    assert "node_feature_dim" in reason


def test_gap3_legacy_flat_checkpoint_and_evaluator_loaders(tmp_path):
    from training.eval_f_act import load_checkpoint as load_f_act_checkpoint
    from training.eval_canonical_three_agent_measurement import load_actor as load_canonical_actor

    actor = SharedActor(obs_dim=127, action_dim=19, hidden=64)
    legacy_flat = {"actor": actor.state_dict(), "obs_dim": 127, "action_dim": 19}
    assert validate_policy_checkpoint(legacy_flat) == (True, None)
    assert type(load_mappo_actor(legacy_flat)) is SharedActor

    checkpoint_path = tmp_path / "legacy_flat.pt"
    torch.save(legacy_flat, checkpoint_path)
    f_act_actor, _, obs_dim, action_dim, _ = load_f_act_checkpoint(str(checkpoint_path))
    canonical_actor, _, _ = load_canonical_actor(str(checkpoint_path))
    assert type(f_act_actor) is SharedActor
    assert type(canonical_actor) is SharedActor
    assert (obs_dim, action_dim) == (127, 19)

    gnn_actor = GNNMAPPOActor(encoder_type="mlp")
    gnn_critic = GNNMAPPOCritic(encoder_type="mlp")
    gnn_checkpoint = {
        "policy_architecture": "gnn:mlp",
        "checkpoint_contract": create_policy_checkpoint_contract("gnn:mlp"),
        "obs_dim": 127,
        "action_dim": 19,
        "actor": gnn_actor.state_dict(),
        "critic": gnn_critic.state_dict(),
    }
    gnn_checkpoint_path = tmp_path / "gnn_mlp.pt"
    torch.save(gnn_checkpoint, gnn_checkpoint_path)
    gnn_f_act_actor, gnn_f_act_critic, _, _, _ = load_f_act_checkpoint(
        str(gnn_checkpoint_path)
    )
    gnn_canonical_actor, _, _ = load_canonical_actor(str(gnn_checkpoint_path))
    assert type(gnn_f_act_actor) is GNNMAPPOActor
    assert type(gnn_f_act_critic) is GNNMAPPOCritic
    assert type(gnn_canonical_actor) is GNNMAPPOActor

    missing_contract = {
        "policy_architecture": "gnn:mlp",
        "obs_dim": 127,
        "action_dim": 19,
        "actor": GNNMAPPOActor(encoder_type="mlp").state_dict(),
    }
    valid, reason = validate_policy_checkpoint(missing_contract)
    assert not valid
    assert "checkpoint_contract" in reason
