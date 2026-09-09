"""
GMN-Football-3 — End-to-end (wire) Event-Code Transmission Regression Test

Historical bug (root cause of the archived "0% pass accuracy" Phase 10 finding):
GameEngine.step() set `info.event` to the human-readable event DESCRIPTION (e.g.
"GOAL! Team Left scored! (1 - 0)"), and the bridge encoders mapped that into the
binary EVENT_CODE_MAP byte with an exact indexOf lookup. Description strings never
equal the canonical type token, so event_code was ALWAYS 0 on every tick where a
goal / shot / pass / pass_completed / ... legitimately occurred. Downstream, the
PettingZoo wrapper rebuilt `info["event"]` from event_code, so RL shaping and
evaluation never saw any events.

This test boots the REAL TypeScript bridge and drives scripted policies that
reliably pass / shoot / score, then asserts that the transmitted event stream on
the wire matches the engine's ground-truth counters:

  - PASS episode  : every attempted pass produces an event_code (pass-family
                    completeness vs stats.passes.left), and every completed pass
                    produces a pass_completed code (== completed_passes_left).
  - SHOT episode  : every shot produces a shot-family code (>= total_shots_left).
  - GOAL episode  : every goal produces exactly one code == 1 (== score.left).
  - Consistency   : EVENT_CODE_MAP[event_code] == info.event.type on every step
                    where event_code > 0.

Run:  python training/test_event_code_wire.py
"""

import os
import sys
import time
import subprocess
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmn_pettingzoo import GMNMultiAgentEnv  # noqa: E402
from gmn_gym import EVENT_CODE_MAP  # noqa: E402

PORT = int(os.environ.get("GMN_EVENT_CODE_TEST_PORT", "5068"))

# Discrete action indices (src/engine/ActionMapping.ts)
MOVE_RIGHT = 5
SHORT_PASS = 11
SHOT = 12


def start_bridge_server(port: int) -> subprocess.Popen:
    """Boot the real TypeScript bridge (same pattern as test_reward_shape_e2e)."""
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
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=1.0
            ) as resp:
                if resp.status == 200:
                    return proc
        except Exception:
            pass
    raise RuntimeError(f"Bridge did not become healthy on port {port}")


def run_episode(scenario: str, policy, seed: int, max_steps: int):
    """Drive one episode on the real bridge.

    Returns (event_count_by_type, event_codes_seen, terminal_ground_truth,
             final_score_left).
    """
    env = GMNMultiAgentEnv(
        scenario=scenario,
        port=PORT,
        auto_start_bridge=False,  # bridge already booted by this harness
        enable_reward_shaping=False,
    )
    counts: dict = {}
    codes_seen: list = []
    ground_truth: dict = {}
    final_score_left = 0
    step = 0
    try:
        obs, info = env.reset(seed=seed)
        while env.agents and step < max_steps:
            current_agents = list(env.agents)
            acts = {}
            for ag in current_agents:
                local_obs = obs[ag]["observation"] if isinstance(obs[ag], dict) else obs[ag]
                acts[ag] = int(policy(local_obs, ag, step))
            obs, rews, terms, truncs, infos = env.step(acts)
            step += 1

            # Collect the transmitted event stream (the wrapper rebuilds
            # info["event"] from event_code).
            for ag, inf in infos.items():
                ev_code = inf.get("eventCode", 0)
                if ev_code:
                    codes_seen.append(ev_code)
                ev = inf.get("event")
                if isinstance(ev, dict) and ev.get("type"):
                    counts[ev["type"]] = counts.get(ev["type"], 0) + 1

            if any(terms.values()) or any(truncs.values()):
                # Capture ground-truth engine counters on the terminal step.
                for inf in infos.values():
                    if isinstance(inf, dict) and "ground_truth" in inf:
                        ground_truth = inf["ground_truth"]
                    score = inf.get("score")
                    if score:
                        final_score_left = int(score.get("left", 0))
                break
    finally:
        env.close()
    return counts, codes_seen, ground_truth, final_score_left
def _ownership_left(obs) -> bool:
    """Ownership one-hot at obs[94..96] = [no-one, left, right]."""
    if len(obs) < 97:
        return False
    return bool(obs[95] > 0.5)


def pass_policy(obs, agent, step):
    """Drive with the ball, and short-pass every 4th tick while in possession."""
    if _ownership_left(obs) and step % 4 == 0:
        return SHORT_PASS
    return MOVE_RIGHT


def shot_policy(obs, agent, step):
    """Move right; shoot every 20th tick while in possession (midfield shots)."""
    if _ownership_left(obs) and step % 20 == 0:
        return SHOT
    return MOVE_RIGHT


def carry_policy(obs, agent, step):
    """Pure dribble-right carry to goal (mirrors test_reward_shape_e2e)."""
    return MOVE_RIGHT


def check(cond: bool, label: str) -> bool:
    if cond:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}")
    return cond


def verify_wire_consistency(codes_seen: list) -> bool:
    ok = True
    for code in codes_seen:
        if not (0 < code < len(EVENT_CODE_MAP)):
            print(f"  [FAIL] event_code {code} out of EVENT_CODE_MAP range")
            ok = False
    return ok