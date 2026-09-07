"""
GMN-Football-3 — End-to-end (wire) regression test for CooperativeRewardShaper.

Objective: prove the reward-shaping FIX is live through the REAL bridge, not just
in unit tests.

Historical bug: the wired GMNMultiAgentEnv.step() path built `step_events` with
engine (lowercase) types and no `agent_id`, so CooperativeRewardShaper matched
nothing — every shaped reward equaled the base reward (dead shaper).

This test boots the actual TypeScript bridge (npx tsx bridge_server.ts), drives
GMNMultiAgentEnv on `academy_empty_goal` with a deterministic MOVE-RIGHT script
(the single controllable agent picks up the ball, carries it forward, and scores).
With ball-hogging shaping enabled, the resulting cumulative reward MUST differ
from the identical unshaped script — proving attribution + shaping flow end-to-end.

Additionally it asserts the shaper actually observed possession (holder_ticks>0)
through the wire.

Run:  python training/test_reward_shape_e2e.py
"""

import os
import sys
import time
import subprocess
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmn_pettingzoo import GMNMultiAgentEnv  # noqa: E402

PORT = int(os.environ.get("GMN_SHAPE_E2E_PORT", "5067"))
SCENARIO = "academy_empty_goal"
MAX_STEPS = 400
SEED = 9001


def start_bridge_server(port: int) -> subprocess.Popen:
    """Boot the real TypeScript bridge (same pattern as determinism test)."""
    bridge_script = os.path.join(os.path.dirname(__file__), "bridge_server.ts")
    npx = "npx.cmd" if sys.platform == "win32" else "npx"
    proc = subprocess.Popen(
        [npx, "tsx", bridge_script],
        env=dict(os.environ, GMN_BRIDGE_PORT=str(port)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(40):
        time.sleep(0.3)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return proc
        except Exception:
            pass
    raise RuntimeError(f"Bridge did not become healthy on port {port}")


def _run_episode(enable_shaping: bool):
    """Deterministic carry-to-goal. Returns (cumulative_reward, shaper_diag_or_None)."""
    env = GMNMultiAgentEnv(
        scenario=SCENARIO,
        port=PORT,
        auto_start_bridge=False,  # bridge already booted by this harness
        enable_reward_shaping=enable_shaping,
    )
    cumulative = 0.0
    step = 0
    try:
        obs, info = env.reset(seed=SEED)
        while env.agents and step < MAX_STEPS:
            acts = {ag: int(5) for ag in env.agents}  # pure MOVE RIGHT
            obs, rews, terms, truncs, infos = env.step(acts)
            a0 = list(rews.keys())[0]
            cumulative += float(rews[a0])
            step += 1
            if terms[a0] or truncs[a0]:
                break
    finally:
        env.close()
    diag = env.reward_shaper.get_diagnostics() if enable_shaping and env.reward_shaper else None
    return float(cumulative), diag


def main() -> int:
    proc = start_bridge_server(PORT)
    try:
        time.sleep(0.5)
        rew_shaped, diag = _run_episode(enable_shaping=True)
        rew_plain, _ = _run_episode(enable_shaping=False)

        print(f"shaped cumulative reward : {rew_shaped:+.4f}")
        print(f"unshaped cumulative reward: {rew_plain:+.4f}")
        print(f"shaper diagnostics       : {diag}")

        assert diag is not None, "shaper diagnostics missing with shaping enabled"
        holder_ticks = diag.get("holder_ticks", 0)
        assert holder_ticks > 0, (
            "shaper never observed possession through the wire — "
            "agent_id attribution is broken end-to-end"
        )
        assert rew_shaped != rew_plain, (
            "shaped reward identical to unshaped reward — shaper is dead on the wire"
        )
        print("PASS: end-to-end wire shaping is live and alters rewards.")
        return 0
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
