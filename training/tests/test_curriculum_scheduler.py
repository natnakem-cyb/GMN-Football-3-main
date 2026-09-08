"""
GMN-Football-3 — CurriculumScheduler Unit & Persistence Tests
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from training.curriculum_scheduler import (
    CurriculumScheduler,
    CURRICULUM_STAGES,
    is_scenario_success,
)


class TestIsScenarioSuccess:
    def test_academy_stages_goal_scored(self):
        info = {"score": {"left": 1, "right": 0}}
        for stage in CURRICULUM_STAGES[:6]:
            assert is_scenario_success(stage, info) is True

    def test_academy_stages_no_goal(self):
        info = {"score": {"left": 0, "right": 0}}
        for stage in CURRICULUM_STAGES[:6]:
            assert is_scenario_success(stage, info) is False

    def test_match_stages_win(self):
        info = {"score": {"left": 2, "right": 1}}
        assert is_scenario_success("5_vs_5", info) is True
        assert is_scenario_success("11_vs_11", info) is True

    def test_match_stages_draw(self):
        info = {"score": {"left": 1, "right": 1}}
        assert is_scenario_success("5_vs_5", info) is False
        assert is_scenario_success("11_vs_11", info) is False

    def test_match_stages_loss(self):
        info = {"score": {"left": 0, "right": 2}}
        assert is_scenario_success("5_vs_5", info) is False
        assert is_scenario_success("11_vs_11", info) is False


class TestCurriculumSchedulerPromotion:
    def test_promotion_after_threshold(self):
        scheduler = CurriculumScheduler(
            stages=["easy", "hard"],
            window_size=5,
            promote_threshold=0.6,
            min_episodes_before_promotion=5,
        )
        # 3 successes out of 5 = 0.6, should promote on the 5th
        for i in range(5):
            scheduler.record_result(success=True)
        stage = scheduler.evaluate_and_step()
        assert stage == "hard"
        assert scheduler.current_idx == 1

    def test_no_promotion_below_threshold(self):
        scheduler = CurriculumScheduler(
            stages=["easy", "hard"],
            window_size=5,
            promote_threshold=0.6,
            min_episodes_before_promotion=5,
        )
        for i in range(5):
            scheduler.record_result(success=False)
        stage = scheduler.evaluate_and_step()
        assert stage == "easy"
        assert scheduler.current_idx == 0

    def test_no_promotion_before_min_episodes(self):
        scheduler = CurriculumScheduler(
            stages=["easy", "hard"],
            window_size=5,
            promote_threshold=0.6,
            min_episodes_before_promotion=10,
        )
        for i in range(9):
            scheduler.record_result(success=True)
        stage = scheduler.evaluate_and_step()
        assert stage == "easy"
        # 10th episode should allow promotion
        scheduler.record_result(success=True)
        stage = scheduler.evaluate_and_step()
        assert stage == "hard"

    def test_promotion_resets_window(self):
        scheduler = CurriculumScheduler(
            stages=["easy", "hard"],
            window_size=5,
            promote_threshold=0.6,
            min_episodes_before_promotion=5,
        )
        for i in range(5):
            scheduler.record_result(success=True)
        scheduler.evaluate_and_step()
        assert scheduler.current_idx == 1
        assert scheduler.window == []

    def test_demotion_after_regression(self):
        scheduler = CurriculumScheduler(
            stages=["easy", "hard"],
            window_size=5,
            promote_threshold=0.6,
            demote_threshold=0.1,
            min_episodes_before_promotion=5,
        )
        # Promote first
        for i in range(5):
            scheduler.record_result(success=True)
        scheduler.evaluate_and_step()
        assert scheduler.current_idx == 1
        # Now record failures to trigger demotion
        for i in range(5):
            scheduler.record_result(success=False)
        stage = scheduler.evaluate_and_step()
        assert stage == "easy"
        assert scheduler.current_idx == 0

    def test_second_promotion_gated_by_min_episodes_in_stage(self):
        """
        Regression test: min_episodes_before_promotion must gate each stage
        transition independently, not just the very first promotion. After
        promoting A->B, only 1 episode in stage B should NOT be enough to
        promote B->C even if total_episodes is already past the floor.
        """
        scheduler = CurriculumScheduler(
            stages=["A", "B", "C"],
            window_size=5,
            promote_threshold=0.6,
            demote_threshold=0.1,
            min_episodes_before_promotion=5,
        )
        # 5 episodes in A -> promote to B
        for _ in range(5):
            scheduler.record_result(success=True)
        stage = scheduler.evaluate_and_step()
        assert stage == "B"
        assert scheduler.current_idx == 1
        assert scheduler.episodes_in_stage == 0

        # Only 1 episode in B -> should NOT promote to C
        scheduler.record_result(success=True)
        stage = scheduler.evaluate_and_step()
        assert stage == "B"
        assert scheduler.current_idx == 1

        # 4 more episodes in B (total 5 in stage) -> now promote to C
        for _ in range(4):
            scheduler.record_result(success=True)
        stage = scheduler.evaluate_and_step()
        assert stage == "C"
        assert scheduler.current_idx == 2

    def test_episodes_in_stage_resets_on_demotion(self):
        """
        Demotion must also reset episodes_in_stage so the floor applies
        fresh when re-entering the easier stage.
        """
        scheduler = CurriculumScheduler(
            stages=["A", "B"],
            window_size=5,
            promote_threshold=0.6,
            demote_threshold=0.1,
            min_episodes_before_promotion=5,
        )
        # Promote A -> B
        for _ in range(5):
            scheduler.record_result(success=True)
        scheduler.evaluate_and_step()
        assert scheduler.current_idx == 1
        # 5 failures in B -> success rate = 0.0, demote_threshold = 0.1
        for _ in range(5):
            scheduler.record_result(success=False)
        assert scheduler.episodes_in_stage == 5
        # Demote back to A
        scheduler.evaluate_and_step()
        assert scheduler.current_idx == 0
        assert scheduler.episodes_in_stage == 0


class TestCurriculumSchedulerPersistence:
    def test_save_and_reload_preserves_state(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            scheduler = CurriculumScheduler(
                stages=["easy", "hard"],
                window_size=5,
                promote_threshold=0.6,
                demote_threshold=0.1,
                min_episodes_before_promotion=5,
            )
            for i in range(5):
                scheduler.record_result(success=True)
            scheduler.evaluate_and_step()
            assert scheduler.current_idx == 1

            scheduler.save(path)
            loaded = CurriculumScheduler.load(path)
            assert loaded is not None
            assert loaded.current_idx == 1
            assert loaded.total_episodes == 5
            assert loaded.window == []
            assert len(loaded.history) == 1
            assert loaded.history[0]["type"] == "promote"
            assert loaded.history[0]["to"] == "hard"
        finally:
            os.unlink(path)

    def test_reload_mid_run_preserves_window(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            scheduler = CurriculumScheduler(
                stages=["easy", "hard"],
                window_size=5,
                promote_threshold=0.6,
                min_episodes_before_promotion=5,
            )
            # Record 3 successes, save, reload, record 2 more
            for i in range(3):
                scheduler.record_result(success=True)
            scheduler.save(path)

            loaded = CurriculumScheduler.load(path)
            assert loaded is not None
            assert loaded.total_episodes == 3
            assert len(loaded.window) == 3

            for i in range(2):
                loaded.record_result(success=True)
            stage = loaded.evaluate_and_step()
            assert stage == "hard"
            assert loaded.current_idx == 1
        finally:
            os.unlink(path)

    def test_load_nonexistent_returns_none(self):
        assert CurriculumScheduler.load("/tmp/nonexistent_curriculum_xyz.json") is None

    def test_persistence_preserves_episodes_in_stage(self):
        """
        After promoting once and recording more episodes in the new stage,
        a save/load round-trip must preserve episodes_in_stage exactly —
        not reset it to 0 and not conflate it with total_episodes.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            scheduler = CurriculumScheduler(
                stages=["A", "B"],
                window_size=5,
                promote_threshold=0.6,
                demote_threshold=0.1,
                min_episodes_before_promotion=5,
            )
            # 5 episodes in A -> promote to B
            for _ in range(5):
                scheduler.record_result(success=True)
            scheduler.evaluate_and_step()
            assert scheduler.current_idx == 1
            assert scheduler.episodes_in_stage == 0

            # 3 episodes in B
            for _ in range(3):
                scheduler.record_result(success=True)
            assert scheduler.episodes_in_stage == 3
            assert scheduler.total_episodes == 8

            scheduler.save(path)
            loaded = CurriculumScheduler.load(path)
            assert loaded is not None
            assert loaded.current_idx == 1
            assert loaded.episodes_in_stage == 3, (
                f"episodes_in_stage should be 3 after reload, got {loaded.episodes_in_stage}"
            )
            assert loaded.total_episodes == 8
            assert len(loaded.window) == 3

            # Record 2 more in B -> total 5 in stage, then promote to C (if present)
            # or verify evaluate_and_step still gates correctly on reloaded state.
            for _ in range(2):
                loaded.record_result(success=True)
            assert loaded.episodes_in_stage == 5
            assert loaded.total_episodes == 10
        finally:
            os.unlink(path)


class TestCurriculumSchedulerHistory:
    def test_history_records_promotion(self):
        scheduler = CurriculumScheduler(
            stages=["a", "b"],
            window_size=3,
            promote_threshold=0.6,
            min_episodes_before_promotion=3,
        )
        scheduler.record_result(True)
        scheduler.record_result(True)
        scheduler.record_result(True)
        scheduler.evaluate_and_step()
        assert len(scheduler.history) == 1
        assert scheduler.history[0]["type"] == "promote"
        assert scheduler.history[0]["from"] == "a"
        assert scheduler.history[0]["to"] == "b"
        assert "success_rate" in scheduler.history[0]
        assert "timestamp" in scheduler.history[0]

    def test_history_records_demotion(self):
        scheduler = CurriculumScheduler(
            stages=["a", "b"],
            window_size=3,
            promote_threshold=0.6,
            demote_threshold=0.1,
            min_episodes_before_promotion=3,
        )
        scheduler.record_result(True)
        scheduler.record_result(True)
        scheduler.record_result(True)
        scheduler.evaluate_and_step()
        scheduler.record_result(False)
        scheduler.record_result(False)
        scheduler.record_result(False)
        scheduler.evaluate_and_step()
        assert len(scheduler.history) == 2
        assert scheduler.history[1]["type"] == "demote"
        assert scheduler.history[1]["from"] == "b"
        assert scheduler.history[1]["to"] == "a"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
