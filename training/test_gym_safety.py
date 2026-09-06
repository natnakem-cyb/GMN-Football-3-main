"""
GMN-Football-3 - Gym WS Safety Unit Tests (P0 #6, P1 #13)

Covers, without requiring a live bridge:
  - reset-response structural validation (unrelated JSON rejected)
  - WS receive timeout raises a clear, context-tagged RuntimeError (no hang)
  - contract constants: 127-dim observation, Discrete(19), additive offside code
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(__file__))

import gmn_gym  # noqa: E402
from gmn_gym import GMNFootballEnv, OBSERVATION_DIM, ACTION_SPACE_SIZE, EVENT_CODE_MAP  # noqa: E402

failures = 0


def check(cond, label):
    global failures
    if cond:
        print(f"   OK  {label}")
    else:
        failures += 1
        print(f"   FAIL {label}")


def make_env():
    # Bypass bridge startup: construct without __init__ side effects.
    env = GMNFootballEnv.__new__(GMNFootballEnv)
    env.ws_recv_timeout = 0.2
    return env


def test_reset_validation():
    print("\n[P1 #13] Reset response structural validation")
    good_single = {"observation": [0.0] * OBSERVATION_DIM, "info": {}}
    good_multi = {"observations": [[0.0] * OBSERVATION_DIM for _ in range(3)], "info": {}}
    bad_wrong_dim = {"observation": [0.0] * (OBSERVATION_DIM - 1)}
    bad_ack = {"status": "subscribed_metrics"}
    bad_broadcast = {"type": "training_status", "data": {"isRunning": False}}
    bad_error = {"type": "error", "data": {"message": "boom"}}
    not_json_like = ["observation", 42]

    check(env_is_valid(good_single), "valid single-agent reset accepted")
    check(env_is_valid(good_multi), "valid multi-agent reset accepted")
    check(not env_is_valid(bad_wrong_dim), "wrong observation dimension rejected")
    check(not env_is_valid(bad_ack), "ack JSON rejected as reset response")
    check(not env_is_valid(bad_broadcast), "training_status broadcast rejected as reset response")
    check(not env_is_valid(bad_error), "error message rejected as reset response")
    check(not env_is_valid(not_json_like), "non-dict payload rejected")


def env_is_valid(candidate):
    return GMNFootballEnv._is_valid_reset_response(candidate)


class FakeTimeoutSocket:
    """Simulates a bridge that never answers."""

    def recv(self, timeout=None):
        raise TimeoutError(f"timed out after {timeout}s")


class FakeErrorSocket:
    def recv(self, timeout=None):
        raise ConnectionError("connection closed")


def test_recv_timeout():
    print("\n[P0 #6] WS receive timeout / error handling")
    env = make_env()

    env.ws_client = FakeTimeoutSocket()
    try:
        env._recv_frame("step")
        check(False, "timeout must raise")
    except RuntimeError as e:
        check("WS Timeout" in str(e) and "'step'" in str(e), f"timeout raises context-tagged RuntimeError ({e})")

    env.ws_client = FakeErrorSocket()
    try:
        env._recv_frame("reset")
        check(False, "connection error must raise")
    except RuntimeError as e:
        check("'reset'" in str(e) and "Connection" in str(e) or "connection" in str(e), f"error raises context-tagged RuntimeError ({e})")

    env.ws_client = None
    try:
        env._recv_frame("step")
        check(False, "missing connection must raise")
    except RuntimeError as e:
        check("No WebSocket connection" in str(e), "missing connection raises")


def test_contract_constants():
    print("\n[Contract] Python constants parity")
    check(OBSERVATION_DIM == 127, "OBSERVATION_DIM == 127")
    check(ACTION_SPACE_SIZE == 19, "ACTION_SPACE_SIZE == 19")
    check(len(EVENT_CODE_MAP) == 14, f"EVENT_CODE_MAP has 14 entries (got {len(EVENT_CODE_MAP)})")
    check(EVENT_CODE_MAP[13] == "offside", "code 13 = offside (additive)")
    check(EVENT_CODE_MAP[1] == "goal" and EVENT_CODE_MAP[12] == "scenario_failed", "codes 1-12 unchanged")


def main():
    print("====================================================")
    print("GMN-FOOTBALL-3 - GYM WS SAFETY UNIT TESTS")
    print("====================================================")
    test_reset_validation()
    test_recv_timeout()
    test_contract_constants()
    print("\n====================================================")
    if failures == 0:
        print("OK  ALL GYM SAFETY CHECKS PASSED")
    else:
        print(f"FAIL {failures} CHECK(S) FAILED")
        sys.exit(1)
    print("====================================================")


if __name__ == "__main__":
    main()
