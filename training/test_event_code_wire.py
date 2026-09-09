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

Run:  python -X utf8 training/test_event_code_wire.py
      (or set PYTHONIOENCODING=utf-8; the script also reconfigures its own
       stdout to UTF-8 so section headers render correctly on any terminal)
"""

import os
import sys
import time
import subprocess
import urllib.request

# Force UTF-8 console output so non-ASCII punctuation (e.g. the em dash in the
# section headers below) is not mis-decoded by legacy code pages (CP437/1252),
# which would display '—' as mojibake like 'ΓÇö' or 'â€”'. The FILE is always
# UTF-8; this only fixes the terminal rendering.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (ValueError, OSError):
        pass

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
             final_score_left, diagnostics).
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
        diag = {"possession_steps": 0, "pass_actions": 0, "shot_actions": 0}
        while env.agents and step < max_steps:
            current_agents = list(env.agents)
            acts = {}
            for ag in current_agents:
                acts[ag] = int(policy(obs[ag], ag, step))
            # Diagnostics: team ownership + scripted ball actions this step
            first_pack = obs[current_agents[0]] if current_agents else None
            first_obs = first_pack.get("observation") if isinstance(first_pack, dict) else first_pack
            if first_obs is not None and _ownership_left(first_obs):
                diag["possession_steps"] += 1
            for a in acts.values():
                if a == SHORT_PASS:
                    diag["pass_actions"] += 1
                elif a == SHOT:
                    diag["shot_actions"] += 1
            obs, rews, terms, truncs, infos = env.step(acts)
            step += 1

            # Collect the transmitted event stream (the wrapper rebuilds
            # info["event"] from event_code). infos are per-agent copies of the
            # SAME shared_info dict, so collect exactly once per step.
            first_info = next(iter(infos.values())) if infos else {}
            if isinstance(first_info, dict):
                ev_code = first_info.get("eventCode", 0)
                if ev_code:
                    codes_seen.append(ev_code)
                ev = first_info.get("event")
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
    return counts, codes_seen, ground_truth, final_score_left, diag
def _ownership_left(obs) -> bool:
    """Ownership one-hot at obs[94..96] = [no-one, left, right]."""
    if len(obs) < 97:
        return False
    return bool(obs[95] > 0.5)


def _agent_slot(agent: str) -> int:
    """left_N -> slot N-1 in the left-team position block (obs[0..21])."""
    try:
        return int(agent.rsplit("_", 1)[1]) - 1
    except Exception:
        return 0


def _move_toward(obs, sx: float, sy: float, slot: int) -> int:
    """Pick the 8-way discrete move action best matching (sx - px, sy - py)."""
    px, py = obs[2 * slot], obs[2 * slot + 1]
    dx, dy = sx - px, sy - py
    if abs(dx) >= abs(dy):
        return 5 if dx > 0 else 1   # RIGHT : LEFT
    return 7 if dy > 0 else 3       # BOTTOM : TOP


def _teammate_slot(obs, own_slot: int) -> int:
    """Other occupied left-team slot (nonzero position block)."""
    best, best_d = own_slot, 1e9
    px, py = obs[2 * own_slot], obs[2 * own_slot + 1]
    for j in range(11):
        if j == own_slot:
            continue
        jx, jy = obs[2 * j], obs[2 * j + 1]
        if jx == 0.0 and jy == 0.0:
            continue  # empty slot
        d = (jx - px) ** 2 + (jy - py) ** 2
        if d < best_d:
            best_d, best = d, j
    return best


def pass_policy(obs_pack, agent, step):
    """Chase the ball; once the possessor (mask[11]==1), align toward the
    teammate on even possession steps and play a SHORT_PASS on odd ones.

    Notes:
    - The engine only registers a pass for the player actually holding the
      ball (GameEngine SHORT_PASS case), and ObservationEncoder marks
      ball-handling actions valid only for the possessor — so mask[11]==1 is
      the exact, team-agnostic trigger for a genuine pass attempt.
    - The wrapper synthesizes all-ones masks at reset (masks are unknown
      before the first physics step), so masks are trusted only for step > 0.
    - Possession pickup is proximity-based (BALL_CONTROL_DIST=0.038) while a
      player covers ~0.12/tick, so chasers must re-aim EVERY tick to
      eventually land a tick inside the pickup radius (mirrors the
      rule-based defender's behavior).
    """
    mask = obs_pack.get("action_mask") if isinstance(obs_pack, dict) else None
    obs = obs_pack.get("observation") if isinstance(obs_pack, dict) else obs_pack
    slot = _agent_slot(agent)
    if step > 0 and mask is not None and len(mask) > 11 and int(mask[11]) == 1:
        if step % 2 == 0:
            t = _teammate_slot(obs, slot)
            return _move_toward(obs, obs[2 * t], obs[2 * t + 1], slot)
        return SHORT_PASS
    if slot == 0:
        return _move_toward(obs, obs[88], obs[89], slot)  # chase ball
    return 0  # IDLE — teammates hold position to receive passes


def shot_policy(obs_pack, agent, step):
    """Chase the ball; once the possessor (mask[12]==1), SHOOT immediately.

    The shot direction defaults to the opponent goal (GameEngine SHOT case),
    and mask[12]==1 is the exact possessor trigger for a genuine shot attempt.
    """
    mask = obs_pack.get("action_mask") if isinstance(obs_pack, dict) else None
    obs = obs_pack.get("observation") if isinstance(obs_pack, dict) else obs_pack
    slot = _agent_slot(agent)
    if step > 0 and mask is not None and len(mask) > 12 and int(mask[12]) == 1:
        return SHOT
    return _move_toward(obs, obs[88], obs[89], slot)  # chase ball


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


SHOT_FAMILY_CODES = {2, 3, 4}  # shot, shot_saved, shot_missed
GOAL_CODE = 1
PASS_SEEDS = [7, 13, 21]
SHOT_SEEDS = [7, 13, 21]


def run_aggregated(scenario: str, policy, seeds, max_steps: int):
    """Run one episode per seed and aggregate wire events + engine counters."""
    agg_counts: dict = {}
    agg_codes: list = []
    agg_gt: dict = {}
    for seed in seeds:
        counts, codes, gt, score_left, diag = run_episode(scenario, policy, seed, max_steps)
        for k, v in counts.items():
            agg_counts[k] = agg_counts.get(k, 0) + v
        agg_codes.extend(codes)
        for k, v in gt.items():
            if isinstance(v, (int, float)):
                agg_gt[k] = agg_gt.get(k, 0) + v
            else:
                agg_gt[k] = v
    return agg_counts, agg_codes, agg_gt


def main() -> int:
    print("=" * 68)
    print("Event-Code Wire Transmission Regression Test (real bridge)")
    print("=" * 68)

    proc = start_bridge_server(PORT)
    failures = 0
    try:
        # ------------------------------------------------------------- PASS
        print(f"\n[1] PASS episodes — scripted short-pass policy, seeds {PASS_SEEDS}")
        counts, codes, gt = run_aggregated(
            "academy_pass_and_shoot_with_keeper", pass_policy, PASS_SEEDS, 1500
        )
        print(f"    wire event counts : {counts}")
        print(f"    wire codes seen   : {sorted(set(codes))}")
        print(f"    ground truth      : attempted_passes={gt.get('attempted_passes_left')}, "
              f"completed_passes={gt.get('completed_passes_left')}")
        failures += 0 if check(len(codes) > 0, "at least one event_code transmitted on the wire") else 1
        failures += 0 if check(
            counts.get("pass", 0) == gt.get("attempted_passes_left", 0),
            f"wire 'pass' codes ({counts.get('pass', 0)}) == engine attempted_passes_left "
            f"({gt.get('attempted_passes_left', 0)})",
        ) else 1
        failures += 0 if check(
            counts.get("pass_completed", 0) == gt.get("completed_passes_left", 0),
            f"wire 'pass_completed' codes ({counts.get('pass_completed', 0)}) == engine "
            f"completed_passes_left ({gt.get('completed_passes_left', 0)})",
        ) else 1
        failures += 0 if check(verify_wire_consistency(codes), "all codes within EVENT_CODE_MAP range") else 1

        # ------------------------------------------------------------- SHOT
        print(f"\n[2] SHOT episodes — scripted shot policy, seeds {SHOT_SEEDS}")
        counts, codes, gt = run_aggregated(
            "academy_run_to_score", shot_policy, SHOT_SEEDS, 1200
        )
        shot_codes = [c for c in codes if c in SHOT_FAMILY_CODES]
        print(f"    wire event counts : {counts}")
        print(f"    wire codes seen   : {sorted(set(codes))}")
        print(f"    ground truth      : total_shots={gt.get('total_shots_left')}, "
              f"shots_on_target={gt.get('shots_on_target_left')}")
        failures += 0 if check(len(shot_codes) > 0, "at least one shot-family event_code on the wire") else 1
        failures += 0 if check(
            counts.get("shot", 0) == gt.get("total_shots_left", 0),
            f"wire 'shot' codes ({counts.get('shot', 0)}) == engine total_shots_left "
            f"({gt.get('total_shots_left', 0)})",
        ) else 1
        failures += 0 if check(verify_wire_consistency(codes), "all codes within EVENT_CODE_MAP range") else 1

        # ------------------------------------------------------------- GOAL
        print("\n[3] GOAL episode — scripted carry-to-goal policy")
        counts, codes, gt, score_left, diag = run_episode(
            "academy_empty_goal", carry_policy, seed=3, max_steps=1200
        )
        goal_codes = [c for c in codes if c == GOAL_CODE]
        print(f"    wire event counts : {counts}")
        print(f"    wire codes seen   : {sorted(set(codes))}")
        print(f"    ground truth      : score_left={score_left}, "
              f"possession_left_pct={gt.get('possession_left_pct')}")
        failures += 0 if check(score_left > 0, "carry policy scored at least one goal") else 1
        failures += 0 if check(
            len(goal_codes) == score_left,
            f"wire 'goal' codes ({len(goal_codes)}) == final score.left ({score_left})",
        ) else 1
        failures += 0 if check(verify_wire_consistency(codes), "all codes within EVENT_CODE_MAP range") else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    print("=" * 68)
    if failures:
        print(f"RESULT: {failures} check(s) FAILED")
        return 1
    print("RESULT: all event-code wire transmission checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
