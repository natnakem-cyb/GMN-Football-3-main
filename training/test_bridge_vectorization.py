"""
GMN-Football-3 — Bridge IPC Vectorization Smoke Test
Compares per-step outputs from the legacy multi-process path vs the new
batched single-bridge path on the same seed, to prove numerical equivalence.
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv, OBSERVATION_DIM, ACTION_SPACE_SIZE
from training.mappo_networks import SharedActor


def _build_actor():
    torch.manual_seed(424242)
    actor = SharedActor(obs_dim=OBSERVATION_DIM, action_dim=ACTION_SPACE_SIZE, hidden=64)
    for p in actor.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return actor


def _run_legacy(n_envs: int, steps: int, seed: int):
    """Legacy path: N separate GMNMultiAgentEnv instances on ports 5050..5050+N-1."""
    envs = []
    for i in range(n_envs):
        envs.append(
            GMNMultiAgentEnv(
                scenario="academy_3_vs_1_with_keeper",
                auto_start_bridge=True,
                port=5050 + i if i > 0 else None,
                enable_reward_shaping=False,
            )
        )
    actor = _build_actor()
    actor.eval()
    agent_order = list(envs[0].possible_agents)
    num_agents = len(agent_order)
    obs_dim = OBSERVATION_DIM

    per_env_records = [{"rewards": [], "obs_checksums": [], "action_checksums": []} for _ in range(n_envs)]
    for env_idx, env in enumerate(envs):
        obs_dict, _ = env.reset(seed=seed + env_idx)
        for _ in range(steps):
            current_agents = list(env.agents if env.agents else env.possible_agents)
            local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
            with torch.no_grad():
                dist = actor(torch.from_numpy(local_obs).float())
                actions = dist.logits.argmax(dim=-1)
            action_dict = {a: int(actions[i].item()) for i, a in enumerate(current_agents)}
            obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
            shared_reward = float(rewards[current_agents[0]])
            per_env_records[env_idx]["rewards"].append(shared_reward)
            per_env_records[env_idx]["obs_checksums"].append(
                hashlib.sha256(np.stack([obs_dict[a] for a in current_agents], axis=0).tobytes()).hexdigest()[:16]
            )
            per_env_records[env_idx]["action_checksums"].append(
                hashlib.sha256(np.array([action_dict[a] for a in current_agents], dtype=np.int64).tobytes()).hexdigest()[:16]
            )
            if any(terms.values()) or any(truncs.values()) or not env.agents:
                obs_dict, _ = env.reset(seed=seed + env_idx + 1000)
        env.close()
    return per_env_records


def _run_batched(n_envs: int, steps: int, seed: int):
    """Batched path: one GMNMultiAgentEnv with batch_size=n_envs."""
    env = GMNMultiAgentEnv(
        scenario="academy_3_vs_1_with_keeper",
        auto_start_bridge=True,
        batch_size=n_envs,
        enable_reward_shaping=False,
    )
    actor = _build_actor()
    actor.eval()
    batch_results = env.reset_batch([seed + i for i in range(n_envs)])
    per_env_records = [{"rewards": [], "obs_checksums": [], "action_checksums": []} for _ in range(n_envs)]
    for _ in range(steps):
        action_sets = []
        for env_idx in range(n_envs):
            obs_dict, _ = batch_results[env_idx]
            current_agents = list(obs_dict.keys())
            local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
            with torch.no_grad():
                dist = actor(torch.from_numpy(local_obs).float())
                actions = dist.logits.argmax(dim=-1)
            action_dict = {a: int(actions[i].item()) for i, a in enumerate(current_agents)}
            action_sets.append(action_dict)
        step_results = env.step_batch(action_sets)
        for env_idx in range(n_envs):
            obs_dict, reward, term, trunc, info = step_results[env_idx]
            shared_reward = float(reward)
            current_agents = list(obs_dict.keys())
            per_env_records[env_idx]["rewards"].append(shared_reward)
            per_env_records[env_idx]["obs_checksums"].append(
                hashlib.sha256(np.stack([obs_dict[a] for a in current_agents], axis=0).tobytes()).hexdigest()[:16]
            )
            per_env_records[env_idx]["action_checksums"].append(
                hashlib.sha256(np.array([action_sets[env_idx][a] for a in current_agents], dtype=np.int64).tobytes()).hexdigest()[:16]
            )
            if bool(term) or bool(trunc):
                batch_results[env_idx] = env.reset_batch([seed + env_idx + 1000])[env_idx]
                continue
            batch_results[env_idx] = (obs_dict, {a: info for a in current_agents})
    env.close()
    return per_env_records


def main():
    n_envs = 2
    steps = 10
    seed = 424242

    print("=" * 70)
    print("GMN-Football-3 — Bridge IPC Vectorization Smoke Test")
    print(f"Comparing legacy multi-process vs batched single-bridge ({n_envs} envs, {steps} steps)")
    print("=" * 70)

    t0 = time.time()
    legacy_records = _run_legacy(n_envs, steps, seed)
    legacy_ms = (time.time() - t0) * 1000
    print(f"[Legacy] Completed in {legacy_ms:.1f} ms")

    t0 = time.time()
    batched_records = _run_batched(n_envs, steps, seed)
    batched_ms = (time.time() - t0) * 1000
    print(f"[Batched] Completed in {batched_ms:.1f} ms")

    print()
    mismatches = 0
    for env_idx in range(n_envs):
        legacy = legacy_records[env_idx]
        batched = batched_records[env_idx]
        rewards_match = legacy["rewards"] == batched["rewards"]
        obs_match = legacy["obs_checksums"] == batched["obs_checksums"]
        actions_match = legacy["action_checksums"] == batched["action_checksums"]
        status = "OK" if (rewards_match and obs_match and actions_match) else "MISMATCH"
        if status == "MISMATCH":
            mismatches += 1
        print(f"Env {env_idx}: rewards={rewards_match}, obs={obs_match}, actions={actions_match} [{status}]")
        if status == "MISMATCH":
            for step_idx in range(steps):
                if legacy["rewards"][step_idx] != batched["rewards"][step_idx]:
                    print(f"  Step {step_idx} reward mismatch: legacy={legacy['rewards'][step_idx]}, batched={batched['rewards'][step_idx]}")
                if legacy["obs_checksums"][step_idx] != batched["obs_checksums"][step_idx]:
                    print(f"  Step {step_idx} obs mismatch: legacy={legacy['obs_checksums'][step_idx]}, batched={batched['obs_checksums'][step_idx]}")

    print()
    if mismatches == 0:
        print("[OK] NUMERICAL EQUIVALENCE VERIFIED: batched path matches legacy multi-process path.")
    else:
        print(f"[FAIL] {mismatches} env(s) had mismatches between legacy and batched paths.")
        sys.exit(1)


if __name__ == "__main__":
    main()
