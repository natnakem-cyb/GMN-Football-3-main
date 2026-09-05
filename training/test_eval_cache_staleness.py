"""
Test: evaluation cache staleness detection via env_hash.

Verifies that evaluate_checkpoint_progress treats a cached entry as stale
when the environment hash changes, even if (scenario, algorithm, step) match.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath("."))

from training.eval_progress import (
    compute_env_hash,
    check_existing_evaluation,
    append_progress_row,
    evaluate_checkpoint_progress,
    CSV_FIELDNAMES,
    DEFAULT_CSV_PATH,
)


class TestEvalCacheStaleness(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.tmp_dir.name, "win_rate_progress.csv")
        # Ensure a clean slate
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_env_hash_changes_detect_stale_cache(self):
        """Cache miss when env_hash differs, even with same scenario/algorithm/step."""
        # Simulate an old cache entry with env_hash="oldhash"
        old_row = {
            "scenario": "academy_3_vs_1_with_keeper",
            "algorithm": "MAPPO",
            "step": "150528",
            "learning_rate": "0.0003",
            "goal_rate_pct": "13.33",
            "mean_reward": "0.3616",
            "std_reward": "0.4338",
            "shots_per_ep": "3.73",
            "turnover_rate": "86.67",
            "episodes": "30",
            "deterministic": "True",
            "checkpoint_path": "training/models/mappo_academy_3_vs_1_with_keeper_best.pt",
            "provenance": "test",
            "env_hash": "oldhash",
        }
        append_progress_row(self.csv_path, old_row)

        # Current env hash will be different from "oldhash"
        current_hash = compute_env_hash()
        self.assertNotEqual(current_hash, "oldhash")

        # Should return None because env_hash does not match
        result = check_existing_evaluation(
            self.csv_path,
            "academy_3_vs_1_with_keeper",
            "MAPPO",
            150528,
            current_hash,
        )
        self.assertIsNone(result, "Expected cache miss when env_hash differs")

    def test_env_hash_match_returns_row(self):
        """Cache hit when env_hash matches."""
        matching_hash = compute_env_hash()
        row = {
            "scenario": "academy_3_vs_1_with_keeper",
            "algorithm": "MAPPO",
            "step": "150528",
            "learning_rate": "0.0003",
            "goal_rate_pct": "0.00",
            "mean_reward": "0.1859",
            "std_reward": "0.1805",
            "shots_per_ep": "3.87",
            "turnover_rate": "100.00",
            "episodes": "30",
            "deterministic": "True",
            "checkpoint_path": "training/models/mappo_academy_3_vs_1_with_keeper_best.pt",
            "provenance": "test",
            "env_hash": matching_hash,
        }
        append_progress_row(self.csv_path, row)

        result = check_existing_evaluation(
            self.csv_path,
            "academy_3_vs_1_with_keeper",
            "MAPPO",
            150528,
            matching_hash,
        )
        self.assertIsNotNone(result, "Expected cache hit when env_hash matches")
        self.assertEqual(result["goal_rate_pct"], "0.00")


if __name__ == "__main__":
    unittest.main()
