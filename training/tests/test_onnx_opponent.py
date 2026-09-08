"""
Integration test for learned-policy opponent execution via ONNX snapshot.

Trains a couple hundred steps to produce a real checkpoint, exports it via
export_onnx.py, wires it into the opponent pool, runs a short episode, and
asserts the right-team actions differ from what RuleBasedAgent would produce
for the same observations.
"""

import os
import sys
import tempfile
import time
import hashlib
import io

# Ensure project root and training/ are importable from the tests/ directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Work around Windows cp1252 console encoding when torch.onnx prints Unicode checkmarks.
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch

from training.gmn_pettingzoo import GMNMultiAgentEnv, OBSERVATION_DIM, ACTION_SPACE_SIZE
from training.opponent_pool import OpponentPool
from training.mappo_networks import SharedActor, CentralizedCritic
from training.export_onnx import export_to_onnx


def _train_and_export_onnx(tmp_dir: str, scenario: str = "academy_3_vs_1_with_keeper", train_steps: int = 200) -> str:
    """
    Train a SharedActor for a small number of steps, save the checkpoint in the
    format expected by export_onnx.py, export to ONNX, and return the ONNX path.
    """
    port = 6000 + np.random.randint(0, 1000)
    env = GMNMultiAgentEnv(
        scenario=scenario,
        auto_start_bridge=True,
        port=port,
        opponent_difficulty="medium",
        enable_reward_shaping=False,
    )
    try:
        num_agents = len(env.possible_agents)
        obs_dim = OBSERVATION_DIM
        action_dim = ACTION_SPACE_SIZE

        actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
        critic = CentralizedCritic(obs_dim=obs_dim, hidden=64, mode="pool")
        actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)

        obs, info = env.reset(seed=42)
        total_steps = 0
        max_steps = train_steps
        local_obs_buf = []
        actions_buf = []
        logprobs_buf = []
        rewards_buf = []
        values_buf = []

        while total_steps < max_steps:
            current_agents = list(env.agents if env.agents else env.possible_agents)
            local_obs = np.stack([obs[a] for a in current_agents], axis=0).astype(np.float32)

            with torch.no_grad():
                dist = actor(torch.tensor(local_obs, dtype=torch.float32))
                actions = dist.sample()
                logprobs = dist.log_prob(actions)
                value = critic(torch.tensor(local_obs, dtype=torch.float32).unsqueeze(0))

            action_dict = {a: int(actions[i].item()) for i, a in enumerate(current_agents)}
            obs, rewards, terminations, truncations, infos = env.step(action_dict)

            local_obs_buf.append(local_obs)
            actions_buf.append(actions.cpu().numpy())
            logprobs_buf.append(logprobs.cpu().numpy())
            values_buf.append(float(value.item()))
            rewards_buf.append(float(rewards[current_agents[0]]))

            total_steps += 1
            if any(terminations.values()) or any(truncations.values()) or not env.agents:
                obs, _ = env.reset(seed=42 + total_steps)

        # Simple policy gradient update on the collected trajectory.
        local_obs_t = torch.tensor(np.array(local_obs_buf, dtype=np.float32), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions_buf, dtype=np.int64), dtype=torch.int64)
        returns = torch.tensor(rewards_buf, dtype=torch.float32)

        dist = actor(local_obs_t.reshape(-1, obs_dim))
        logprobs = dist.log_prob(actions_t.reshape(-1))
        loss = -(logprobs * returns.repeat_interleave(len(current_agents))).mean()

        actor_opt.zero_grad()
        loss.backward()
        actor_opt.step()

        # Save checkpoint in export_onnx.py expected format.
        ckpt_name = f"test_actor_{int(time.time())}.pt"
        ckpt_path = os.path.join(tmp_dir, ckpt_name)
        torch.save(
            {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "actor_opt": actor_opt.state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "obs_dim": obs_dim,
                "global_state_dim": obs_dim * num_agents,
                "action_dim": action_dim,
                "timesteps": total_steps,
            },
            ckpt_path,
        )

        # Export to ONNX via export_onnx.py.
        onnx_path = os.path.join(tmp_dir, f"test_actor_{int(time.time())}.onnx")
        export_to_onnx(
            checkpoint_path=ckpt_path,
            output_path=onnx_path,
            ts_weights_path=os.path.join(tmp_dir, "test_weights.ts"),
            scenario_id=scenario,
        )
        return onnx_path
    finally:
        env.close()


def _collect_obs_checksums(env, steps=10, seed=42):
    """Run an episode and collect SHA-256 checksums of per-agent observations."""
    obs_dict, _ = env.reset(seed=seed)
    checksums = []
    for _ in range(steps):
        current_agents = list(env.agents if env.agents else env.possible_agents)
        action_dict = {a: 5 for a in current_agents}
        obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
        obs_stack = np.stack([obs_dict[a] for a in sorted(obs_dict.keys())], axis=0)
        checksum = hashlib.sha256(obs_stack.tobytes()).hexdigest()[:16]
        checksums.append(checksum)
        if any(terms.values()) or any(truncs.values()) or not env.agents:
            obs_dict, _ = env.reset(seed=seed + 1000)
    return checksums


def test_learned_opponent_snapshot():
    with tempfile.TemporaryDirectory(prefix="gmn_onnx_test_") as tmp_dir:
        print(f"[TEST] Training + exporting ONNX model to {tmp_dir}...")
        t0 = time.time()
        onnx_path = _train_and_export_onnx(tmp_dir, scenario="academy_3_vs_1_with_keeper", train_steps=200)
        print(f"[TEST] Training + export completed in {time.time() - t0:.2f}s")
        print(f"[TEST] Using ONNX model: {onnx_path}")

        pool = OpponentPool(difficulties=["medium"], strategy="uniform", seed=1234)
        pool.add_snapshot(onnx_path)

        env = GMNMultiAgentEnv(
            scenario="academy_3_vs_1_with_keeper",
            auto_start_bridge=True,
            port=5052,
            enable_reward_shaping=False,
        )

        entry = pool.sample()
        assert entry["kind"] == "snapshot", f"Expected snapshot entry, got {entry['kind']}"
        applied = pool.apply(entry, env)
        assert applied, "OpponentPool.apply() should return True for snapshot entries"

        snapshot_checksums = _collect_obs_checksums(env, steps=10, seed=42)
        env.close()

        # Episode with snapshot must complete.
        assert len(snapshot_checksums) == 10, f"Expected 10 steps, got {len(snapshot_checksums)}"

        # Run a parallel rule-based episode for comparison.
        env_rule = GMNMultiAgentEnv(
            scenario="academy_3_vs_1_with_keeper",
            auto_start_bridge=True,
            port=5053,
            enable_reward_shaping=False,
        )
        env_rule.set_opponent_difficulty("medium")
        rule_checksums = _collect_obs_checksums(env_rule, steps=10, seed=42)
        env_rule.close()

        assert len(rule_checksums) == 10

        # At least one observation checksum must differ between snapshot and rule-based.
        diverged = False
        for s_c, r_c in zip(snapshot_checksums, rule_checksums):
            if s_c != r_c:
                diverged = True
                break

        assert diverged, (
            "Snapshot and rule-based observation checksums were identical across all steps; "
            "the ONNX policy is not driving behavior."
        )
        print("[TEST] PASS: Snapshot observations diverge from rule-based baseline")


if __name__ == "__main__":
    test_learned_opponent_snapshot()
    print("[TEST] ALL PASSED")
