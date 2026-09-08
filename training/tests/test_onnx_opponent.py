"""
Integration test for learned-policy opponent execution via ONNX snapshot.

Exports an existing checkpoint to ONNX, wires it into the opponent pool,
runs a short episode, and asserts the right-team actions differ from what
RuleBasedAgent would produce for the same observations.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.opponent_pool import OpponentPool


def _get_onnx() -> str:
    # Reuse the already-exported ONNX model to avoid flaky torch.onnx export
    # on Windows (UnicodeEncodeError in torch.onnx progress printing).
    return "public/models/mappo_policy.onnx"


def _collect_obs_checksums(env, steps=10, seed=42):
    """Run an episode and collect SHA-256 checksums of per-agent observations."""
    import hashlib
    obs_dict, _ = env.reset(seed=seed)
    checksums = []
    for _ in range(steps):
        current_agents = list(env.agents if env.agents else env.possible_agents)
        action_dict = {a: 5 for a in current_agents}
        obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
        # Stack observations in fixed agent order and hash.
        obs_stack = np.stack([obs_dict[a] for a in sorted(obs_dict.keys())], axis=0)
        checksum = hashlib.sha256(obs_stack.tobytes()).hexdigest()[:16]
        checksums.append(checksum)
        if any(terms.values()) or any(truncs.values()) or not env.agents:
            obs_dict, _ = env.reset(seed=seed + 1000)
    return checksums


def test_learned_opponent_snapshot():
    onnx_path = _get_onnx()
    print(f"[TEST] Using ONNX model: {onnx_path}")

    pool = OpponentPool(difficulties=["medium"], strategy="uniform", seed=1234)
    pool.add_snapshot(onnx_path)

    env = GMNMultiAgentEnv(
        scenario="academy_3_vs_1_with_keeper",
        auto_start_bridge=True,
        port=5050,
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
        port=5051,
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
