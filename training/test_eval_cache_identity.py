"""
Evaluation cache identity regression tests.

Verifies that the evaluation cache correctly identifies evaluations by full
evaluation identity (checkpoint SHA256 + scenario + algorithm + step + env_hash),
not just by training step or filename.
"""
import os
import sys
import tempfile
import unittest
import hashlib

sys.path.insert(0, os.path.abspath("."))

from training.eval_progress import (
    compute_env_hash,
    compute_evaluation_identity,
    check_existing_evaluation,
    append_progress_row,
    evaluate_checkpoint_progress,
    CSV_FIELDNAMES,
    DEFAULT_CSV_PATH,
    sha256_file,
)


def _write_fake_checkpoint(path: str, content: bytes) -> None:
    with open(path, "wb") as f:
        f.write(content)


class TestEvalCacheIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.tmp_dir.name, "win_rate_progress.csv")
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_a1_same_metadata_different_weights(self):
        """Cache miss when checkpoint SHA256 differs, even with same scenario/algorithm/step."""
        ckpt_a = os.path.join(self.tmp_dir.name, "checkpoint_A.pt")
        ckpt_b = os.path.join(self.tmp_dir.name, "checkpoint_B.pt")
        _write_fake_checkpoint(ckpt_a, b"weights_A")
        _write_fake_checkpoint(ckpt_b, b"weights_B")

        identity_a = compute_evaluation_identity(
            checkpoint_path=ckpt_a,
            scenario="academy_3_vs_1_with_keeper",
            algorithm="MAPPO",
            step=200000,
            base_seed=500000,
            num_episodes=30,
            deterministic=True,
            env_hash=compute_env_hash(),
        )
        identity_b = compute_evaluation_identity(
            checkpoint_path=ckpt_b,
            scenario="academy_3_vs_1_with_keeper",
            algorithm="MAPPO",
            step=200000,
            base_seed=500000,
            num_episodes=30,
            deterministic=True,
            env_hash=compute_env_hash(),
        )

        self.assertNotEqual(identity_a["checkpoint_sha256"], identity_b["checkpoint_sha256"])
        self.assertNotEqual(identity_a["evaluation_id"], identity_b["evaluation_id"])

        append_progress_row(self.csv_path, {
            **identity_a,
            "goal_rate_pct": "13.33",
            "mean_reward": "0.3616",
            "std_reward": "0.4338",
            "shots_per_ep": "3.73",
            "turnover_rate": "86.67",
            "episodes": "30",
            "deterministic": "True",
            "checkpoint_path": ckpt_a,
            "provenance": "test",
        })

        result = check_existing_evaluation(self.csv_path, identity_b["evaluation_id"])
        self.assertIsNone(result, "Expected cache miss for different checkpoint SHA256")

    def test_a2_identical_checkpoint_evaluated_twice(self):
        """Cache hit when evaluating the same checkpoint twice with identical config."""
        ckpt = os.path.join(self.tmp_dir.name, "checkpoint_A.pt")
        _write_fake_checkpoint(ckpt, b"weights_A")

        identity = compute_evaluation_identity(
            checkpoint_path=ckpt,
            scenario="academy_3_vs_1_with_keeper",
            algorithm="MAPPO",
            step=200000,
            base_seed=500000,
            num_episodes=30,
            deterministic=True,
            env_hash=compute_env_hash(),
        )

        append_progress_row(self.csv_path, {
            **identity,
            "goal_rate_pct": "13.33",
            "mean_reward": "0.3616",
            "std_reward": "0.4338",
            "shots_per_ep": "3.73",
            "turnover_rate": "86.67",
            "episodes": "30",
            "deterministic": "True",
            "checkpoint_path": ckpt,
            "provenance": "test",
        })

        result = check_existing_evaluation(self.csv_path, identity["evaluation_id"])
        self.assertIsNotNone(result, "Expected cache hit for identical evaluation identity")
        self.assertEqual(result["goal_rate_pct"], "13.33")

    def test_a3_evaluation_seed_changes(self):
        """Cache miss when evaluation seed changes."""
        ckpt = os.path.join(self.tmp_dir.name, "checkpoint_A.pt")
        _write_fake_checkpoint(ckpt, b"weights_A")

        identity_seed1 = compute_evaluation_identity(
            checkpoint_path=ckpt,
            scenario="academy_3_vs_1_with_keeper",
            algorithm="MAPPO",
            step=200000,
            base_seed=500000,
            num_episodes=30,
            deterministic=True,
            env_hash=compute_env_hash(),
        )
        identity_seed2 = compute_evaluation_identity(
            checkpoint_path=ckpt,
            scenario="academy_3_vs_1_with_keeper",
            algorithm="MAPPO",
            step=200000,
            base_seed=500001,
            num_episodes=30,
            deterministic=True,
            env_hash=compute_env_hash(),
        )

        self.assertNotEqual(identity_seed1["evaluation_id"], identity_seed2["evaluation_id"])

        append_progress_row(self.csv_path, {
            **identity_seed1,
            "goal_rate_pct": "13.33",
            "mean_reward": "0.3616",
            "std_reward": "0.4338",
            "shots_per_ep": "3.73",
            "turnover_rate": "86.67",
            "episodes": "30",
            "deterministic": "True",
            "checkpoint_path": ckpt,
            "provenance": "test",
        })

        result = check_existing_evaluation(self.csv_path, identity_seed2["evaluation_id"])
        self.assertIsNone(result, "Expected cache miss when evaluation seed changes")

    def test_a4_env_hash_changes(self):
        """Cache miss when env_hash changes."""
        ckpt = os.path.join(self.tmp_dir.name, "checkpoint_A.pt")
        _write_fake_checkpoint(ckpt, b"weights_A")

        identity_old = compute_evaluation_identity(
            checkpoint_path=ckpt,
            scenario="academy_3_vs_1_with_keeper",
            algorithm="MAPPO",
            step=200000,
            base_seed=500000,
            num_episodes=30,
            deterministic=True,
            env_hash="oldhash",
        )
        identity_new = compute_evaluation_identity(
            checkpoint_path=ckpt,
            scenario="academy_3_vs_1_with_keeper",
            algorithm="MAPPO",
            step=200000,
            base_seed=500000,
            num_episodes=30,
            deterministic=True,
            env_hash="newhash",
        )

        self.assertNotEqual(identity_old["evaluation_id"], identity_new["evaluation_id"])

        append_progress_row(self.csv_path, {
            **identity_old,
            "goal_rate_pct": "13.33",
            "mean_reward": "0.3616",
            "std_reward": "0.4338",
            "shots_per_ep": "3.73",
            "turnover_rate": "86.67",
            "episodes": "30",
            "deterministic": "True",
            "checkpoint_path": ckpt,
            "provenance": "test",
        })

        result = check_existing_evaluation(self.csv_path, identity_new["evaluation_id"])
        self.assertIsNone(result, "Expected cache miss when env_hash changes")

    def test_a5_legacy_entry_without_sha256(self):
        """Legacy cache entry without checkpoint_sha256 must not be an exact hit."""
        legacy_row = {
            "evaluation_id": "legacy_unverified",
            "checkpoint_sha256": "",
            "scenario": "academy_3_vs_1_with_keeper",
            "algorithm": "MAPPO",
            "step": "200000",
            "env_version": "3.1.0",
            "observation_schema_version": "simple115_v3_role",
            "action_schema_version": "discrete19_v1",
            "learning_rate": "0.0003",
            "goal_rate_pct": "13.33",
            "mean_reward": "0.3616",
            "std_reward": "0.4338",
            "shots_per_ep": "3.73",
            "turnover_rate": "86.67",
            "episodes": "30",
            "deterministic": "True",
            "checkpoint_path": "training/models/mappo_academy_3_vs_1_with_keeper_best.pt",
            "provenance": "legacy",
            "env_hash": compute_env_hash(),
        }
        append_progress_row(self.csv_path, legacy_row)

        # A real evaluation identity with the same scenario/algorithm/step but a real checkpoint hash
        ckpt = os.path.join(self.tmp_dir.name, "checkpoint_A.pt")
        _write_fake_checkpoint(ckpt, b"weights_A")
        real_identity = compute_evaluation_identity(
            checkpoint_path=ckpt,
            scenario="academy_3_vs_1_with_keeper",
            algorithm="MAPPO",
            step=200000,
            base_seed=500000,
            num_episodes=30,
            deterministic=True,
            env_hash=compute_env_hash(),
        )

        result = check_existing_evaluation(self.csv_path, real_identity["evaluation_id"])
        self.assertIsNone(result, "Expected cache miss for legacy entry without real SHA256")


if __name__ == "__main__":
    unittest.main()
