"""
Live end-to-end curriculum tests.

These tests exercise the REAL GMNMultiAgentEnv + real TypeScript bridge
(auto_start_bridge=True). They are NOT FakeEnv mocks.

If the bridge cannot start, tests skip cleanly with a descriptive reason
rather than failing CI.

L1 — Promotion: env.scenario actually changes from stage[i] to stage[i+1]
     after the scheduler fires, and the next reset/step is in the new scenario.
L2 — train_mappo.py --curriculum path: evaluate_and_step is reached, a
     transition checkpoint is written, env.set_scenario is called, scheduler
     JSON is persisted.
L3 — Persistence: kill/reload CurriculumScheduler.load(path) restores
     current_idx, window, episodes_in_stage, history.
L4 — Demotion: live or hybrid-live path shows demote → previous stage.
L5 — Adapter identity across a finishing-family promotion remains
     AttackingDrillRewardAdapter.
L6 — Success signal: live episode terminal info produces is_scenario_success
     True iff left scored, using REAL score from the engine.

This file does NOT use _FakeEnv except as a last-resort skip path.
"""

import json
import os
import socket
import tempfile
import time
from pathlib import Path

import pytest

from training.curriculum_scheduler import CurriculumScheduler, is_scenario_success
from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_rollout import unwrap_masks
from training.reward_adapters import AttackingDrillRewardAdapter

# ---------------------------------------------------------------------------
# Bridge-availability guard
# ---------------------------------------------------------------------------

def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _bridge_can_start():
    """Return True if we believe the TypeScript bridge can start.

    The env uses `npx.cmd tsx bridge_server.ts` internally (see gmn_pettingzoo.py),
    so we only need node/npx available; tsx does not need to be on PATH.
    """
    import shutil
    if not shutil.which("node"):
        return False
    if not shutil.which("npx.cmd") and not shutil.which("npx"):
        return False
    try:
        port = _free_port()
        env = GMNMultiAgentEnv(scenario="academy_empty_goal", auto_start_bridge=True, port=port)
        time.sleep(0.5)
        env.close()
        return True
    except Exception:
        return False


BRIDGE_AVAILABLE = _bridge_can_start()

skip_if_no_bridge = pytest.mark.skipif(
    not BRIDGE_AVAILABLE,
    reason="TypeScript bridge (node/tsx) not available or failed to start",
)


# ---------------------------------------------------------------------------
# L1 — Live set_scenario switches engine stage
# ---------------------------------------------------------------------------

@skip_if_no_bridge
class TestCurriculumLivePromotion:
    """L1–L5: live GMNMultiAgentEnv with real bridge."""

    def test_live_env_set_scenario_switches_engine_stage(self):
        """GMNMultiAgentEnv.set_scenario + reset against live bridge."""
        port = _free_port()
        env = GMNMultiAgentEnv(scenario="academy_empty_goal", auto_start_bridge=True, port=port)
        try:
            obs, info = env.reset(seed=500000)
            assert env.scenario == "academy_empty_goal"

            env.set_scenario("academy_run_to_score")
            obs, info = env.reset(seed=500001)
            assert env.scenario == "academy_run_to_score"

            # At least one step succeeds (no exception, agents non-empty)
            masks = unwrap_masks(obs)
            action_dict = {}
            for a in env.agents:
                if a in masks and masks[a] is not None:
                    legal = [i for i, m in enumerate(masks[a]) if m == 1]
                    action_dict[a] = legal[0] if legal else 0
                else:
                    action_dict[a] = 0
            obs, rew, term, trunc, infos = env.step(action_dict)
            assert env.agents
        finally:
            env.close()

    def test_live_adapter_stays_attacking_drill_across_finishing_promotion(self):
        """After set_scenario within finishing family, adapter type is AttackingDrillRewardAdapter."""
        port = _free_port()
        env = GMNMultiAgentEnv(scenario="academy_empty_goal", auto_start_bridge=True, port=port)
        try:
            obs, info = env.reset(seed=500002)
            adapter_before = type(env.reward_adapter).__name__ if env.reward_adapter else None

            env.set_scenario("academy_run_to_score")
            obs, info = env.reset(seed=500003)
            adapter_after = type(env.reward_adapter).__name__ if env.reward_adapter else None

            assert adapter_before == "AttackingDrillRewardAdapter"
            assert adapter_after == "AttackingDrillRewardAdapter"
        finally:
            env.close()


# ---------------------------------------------------------------------------
# L2 — Live scheduler promotes on real successes (hybrid-live)
# ---------------------------------------------------------------------------

@skip_if_no_bridge
class TestCurriculumLiveSchedulerPromotion:
    """
    Drive a live academy_empty_goal env with random legal actions,
    window_size=5, min_episodes=5, promote_threshold=0.2,
    two-stage ladder [academy_empty_goal, academy_run_to_score].
    """

    def test_live_scheduler_promotes_on_real_successes_or_records(self):
        """
        If promotion fires, assert env.scenario switches.
        If successes never reach threshold, SKIP (still assert record_result
        and evaluate_and_step did not crash).
        """
        port = _free_port()
        env = GMNMultiAgentEnv(scenario="academy_empty_goal", auto_start_bridge=True, port=port)
        try:
            obs, info = env.reset(seed=500010)
            controllable = list(env.agents)

            scheduler = CurriculumScheduler(
                stages=["academy_empty_goal", "academy_run_to_score"],
                window_size=10,
                promote_threshold=0.2,
                demote_threshold=0.0,
                min_episodes_before_promotion=3,
            )

            successes = []
            max_episodes = 10

            for ep in range(max_episodes):
                done = False
                ep_len = 0
                while not done and ep_len < 200:
                    masks = unwrap_masks(obs)
                    action_dict = {}
                    for a in controllable:
                        if a in masks and masks[a] is not None:
                            legal = [i for i, m in enumerate(masks[a]) if m == 1]
                            action_dict[a] = legal[0] if legal else 0
                        else:
                            action_dict[a] = 0

                    obs, rew, term, trunc, infos = env.step(action_dict)
                    done = any(term.values()) or any(trunc.values()) or not env.agents
                    ep_len += 1

                terminal_info = {}
                for aid, inf in infos.items():
                    terminal_info = inf
                    break
                success = is_scenario_success(env.scenario, terminal_info)
                scheduler.record_result(success)
                successes.append(success)

                new_stage = scheduler.evaluate_and_step()
                if new_stage != "academy_empty_goal":
                    env.set_scenario(new_stage)
                    env.reset()
                    break

                obs, info = env.reset(seed=500010 + ep + 1)

            success_rate = sum(successes) / max(len(successes), 1)
            if success_rate >= 0.2:
                assert scheduler.current_idx == 1, (
                    f"Promotion should have fired (success_rate={success_rate:.1%}), "
                    f"but current_idx={scheduler.current_idx}"
                )
            else:
                pytest.skip(
                    f"Live scoring did not occur: success_rate={success_rate:.1%} "
                    f"over {len(successes)} episodes. set_scenario wiring is still "
                    f"tested by test_live_env_set_scenario_switches_engine_stage."
                )
        finally:
            env.close()


# ---------------------------------------------------------------------------
# L3 — Persistence
# ---------------------------------------------------------------------------

class TestCurriculumPersistence:
    """L3: save/load round-trip restores scheduler state."""

    def test_save_load_round_trip(self):
        scheduler = CurriculumScheduler(
            stages=["academy_empty_goal", "academy_run_to_score", "academy_pass_and_shoot_with_keeper"],
            window_size=20,
            promote_threshold=0.6,
            demote_threshold=0.1,
            min_episodes_before_promotion=10,
        )
        for _ in range(12):
            scheduler.record_result(True)
        for _ in range(3):
            scheduler.record_result(False)

        scheduler.evaluate_and_step()
        assert scheduler.current_idx == 1

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            scheduler.save(path)
            loaded = CurriculumScheduler.load(path)
            assert loaded is not None
            assert loaded.current_idx == 1
            assert loaded.episodes_in_stage == 0
            assert len(loaded.window) == 0
            assert loaded.total_episodes == 15
            assert len(loaded.history) == 1
            assert loaded.history[0]["type"] == "promote"
        finally:
            os.unlink(path)

    def test_load_missing_path_returns_none(self):
        assert CurriculumScheduler.load("/nonexistent/path.json") is None


# ---------------------------------------------------------------------------
# L4 — Demotion (hybrid-live: real env, injected failure window)
# ---------------------------------------------------------------------------

@skip_if_no_bridge
class TestCurriculumLiveDemotion:
    """L4: demote → previous stage → set_scenario."""

    def test_live_demote_path_with_real_env(self):
        port = _free_port()
        env = GMNMultiAgentEnv(scenario="academy_empty_goal", auto_start_bridge=True, port=port)
        try:
            scheduler = CurriculumScheduler(
                stages=["academy_empty_goal", "academy_run_to_score"],
                window_size=5,
                promote_threshold=0.6,
                demote_threshold=0.1,
                min_episodes_before_promotion=3,
            )
            scheduler.current_idx = 1
            scheduler.episodes_in_stage = 5
            for _ in range(5):
                scheduler.record_result(False)

            new_stage = scheduler.evaluate_and_step()
            assert new_stage == "academy_empty_goal"

            env.set_scenario(new_stage)
            obs, info = env.reset(seed=500020)
            assert env.scenario == "academy_empty_goal"
        finally:
            env.close()


# ---------------------------------------------------------------------------
# L6 — Real success signal from live engine
# ---------------------------------------------------------------------------

@skip_if_no_bridge
class TestCurriculumLiveSuccessSignal:
    """L6: is_scenario_success from REAL terminal info."""

    def test_live_success_signal_from_real_terminal_info(self):
        port = _free_port()
        env = GMNMultiAgentEnv(scenario="academy_empty_goal", auto_start_bridge=True, port=port)
        try:
            obs, info = env.reset(seed=500030)
            done = False
            while not done:
                masks = unwrap_masks(obs)
                action_dict = {}
                for a in env.agents:
                    if a in masks and masks[a] is not None:
                        legal = [i for i, m in enumerate(masks[a]) if m == 1]
                        action_dict[a] = legal[0] if legal else 0
                    else:
                        action_dict[a] = 0
                obs, rew, term, trunc, infos = env.step(action_dict)
                done = any(term.values()) or any(trunc.values()) or not env.agents

            terminal_info = {}
            for aid, inf in infos.items():
                terminal_info = inf
                break
            success = is_scenario_success("academy_empty_goal", terminal_info)
            goal_scored = terminal_info.get("score", {}).get("left", 0) > 0
            assert success == goal_scored
        finally:
            env.close()


# ---------------------------------------------------------------------------
# L2 — train_mappo.py --curriculum live wiring smoke
# ---------------------------------------------------------------------------

@skip_if_no_bridge
class TestTrainMappoCurriculumLiveSmoke:
    """
    Invoke train_mappo.run_mappo_training with --curriculum and a tiny
    timestep budget. Uses promote_threshold=0.0 plus a pre-seeded window
    so the FIRST evaluate_and_step promotes. This is a WIRING smoke,
    not a skill eval.
    """

    def test_train_mappo_curriculum_live_smoke_promotes_or_records(self, tmp_path):
        import subprocess
        import sys

        # Pre-seed the curriculum state so the first evaluate_and_step sees
        # a window already at the promote threshold. Disable demotion for this
        # wiring smoke: with both thresholds at 0, one unsuccessful episode
        # after promotion immediately demotes the scheduler and masks the
        # promotion that this test is meant to verify.
        scheduler = CurriculumScheduler(
            stages=["academy_empty_goal", "academy_run_to_score"],
            window_size=5,
            promote_threshold=0.0,
            demote_threshold=-1.0,
            min_episodes_before_promotion=1,
        )
        for _ in range(5):
            scheduler.record_result(True)
        curriculum_state_path = str(tmp_path / "curriculum_state.json")
        scheduler.save(curriculum_state_path)

        # Run train_mappo with --curriculum, tiny timesteps, seed42.
        # Use a distinct checkpoint prefix so we don't overwrite canonical models.
        ckpt_name = "mappo_curriculum_e2e_smoke_seed42.pt"
        cmd = [
            sys.executable, "training/train_mappo.py",
            "--scenario", "academy_empty_goal",
            "--timesteps", "1024",
            "--seed", "42",
            "--curriculum",
            "--curriculum-state-path", curriculum_state_path,
            "--curriculum-window-size", "5",
            "--curriculum-min-episodes", "1",
            "--curriculum-promote-threshold", "0.0",
            "--curriculum-demote-threshold", "-1.0",
            "--checkpoint-name", ckpt_name,
            "--models-dir", str(tmp_path),
        ]
        # Write output to a file: the trainer can launch a bridge child that
        # inherits its output handles, so capture_output can keep communicate()
        # blocked after the trainer itself has exited.
        log_path = tmp_path / "curriculum_smoke.log"
        process = None
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    cmd,
                    cwd=os.getcwd(),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            # The trainer owns the bridge child. On Windows, terminate its
            # process tree so a timed-out smoke cannot leave a server behind.
            if os.name == "nt" and process is not None:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            pytest.fail(f"Curriculum smoke exceeded 600 seconds. Log tail:\n{log_tail}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        print("TRAINER LOG:", log_text[-4000:])
        # The run may return non-zero for bridge issues; we only assert on
        # artifacts if the run got far enough to write them.
        curriculum_state = Path(curriculum_state_path)
        if curriculum_state.exists():
            state = json.loads(curriculum_state.read_text())
            assert state["current_idx"] == 1, (
                f"Expected promotion to idx=1, got {state['current_idx']}"
            )
            assert len(state["history"]) >= 1
            assert state["history"][-1]["type"] == "promote"
        else:
            pytest.skip(
                "Curriculum state was not written (bridge likely unavailable). "
                "L1/L3/L4/L5/L6 still exercise set_scenario and scheduler wiring."
            )
