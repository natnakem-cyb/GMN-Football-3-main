"""P0.5d regression tests for gmn_gym bridge lifecycle (bridge-free).

  Defect 1 - _ensure_bridge_running must RAISE, not return None, when the
             bridge never becomes healthy within the retry budget.
  Defect 3 - close() must reap an *adopted* bridge on its port.
  Safety   - the netstat parser ignores UDP rows (e.g. a system svchost
             binding the same port number) and non-LISTENING rows.
"""
import sys
import unittest
from unittest import mock

import requests

sys.path.insert(0, r"C:\Users\USER\Documents\Project\GMN-Football-3-main")

from training import gmn_gym
from training.gmn_gym import GMNFootballEnv, _parse_netstat_listening_pids


def _healthy_response():
    resp = mock.Mock(status_code=200)
    resp.json.return_value = {"observation_dim": 127, "action_space_size": 19}
    return resp


def make_env(port=5399, auto_start_bridge=True):
    """Construct GMNFootballEnv WITHOUT touching a real bridge or port.

    The constructor calls _ensure_bridge_running(); we satisfy the health
    probe with a mock so no node process is ever spawned and no real socket
    wait occurs. Tests that target _ensure_bridge_running itself patch the
    probe afterwards.
    """
    with mock.patch.object(gmn_gym.requests, "get", return_value=_healthy_response()):
        return GMNFootballEnv(
            scenario="academy_empty_goal",
            port=port,
            use_ws=False,
            auto_start_bridge=auto_start_bridge,
        )


class TestDefect1PostLoopVerification(unittest.TestCase):
    """Defect 1: the retry loop must not exit silently."""

    def test_raises_when_health_never_succeeds(self):
        env = make_env(auto_start_bridge=False)
        with mock.patch.object(gmn_gym.time, "sleep"), \
             mock.patch.object(gmn_gym.requests, "get",
                               side_effect=requests.exceptions.ConnectionError("refused")):
            with self.assertRaises(RuntimeError) as ctx:
                env._ensure_bridge_running()
        msg = str(ctx.exception)
        self.assertIn("Bridge Startup Failed", msg)
        self.assertIn("5399", msg)                       # port
        self.assertIn("ConnectionError", msg)            # last health error
        self.assertIn("health checks", msg)              # attempt count
        self.assertIn("distinct from a later", msg)      # startup vs recv

    def test_attempt_count_matches_budget(self):
        env = make_env(auto_start_bridge=False)
        with mock.patch.object(gmn_gym.time, "sleep"), \
             mock.patch.object(gmn_gym.requests, "get",
                               side_effect=requests.exceptions.ConnectionError("refused")):
            with self.assertRaises(RuntimeError) as ctx:
                env._ensure_bridge_running()
        self.assertIn(str(gmn_gym.BRIDGE_STARTUP_ATTEMPTS), str(ctx.exception))

    def test_sleeps_only_budgeted_amount(self):
        """Budget unchanged. No spawn -> N x 0.4 s = 10.0 s.

        (With a spawn it is 1.0 + 24 x 0.4 = 10.6 s; asserted separately.)
        """
        env = make_env(auto_start_bridge=False)
        slept = []
        with mock.patch.object(gmn_gym.time, "sleep", side_effect=slept.append), \
             mock.patch.object(gmn_gym.requests, "get",
                               side_effect=requests.exceptions.ConnectionError("refused")):
            with self.assertRaises(RuntimeError):
                env._ensure_bridge_running()
        total = sum(slept)
        expected = gmn_gym.BRIDGE_STARTUP_ATTEMPTS * gmn_gym.BRIDGE_STARTUP_SLEEP_BETWEEN
        self.assertAlmostEqual(total, expected, places=6)
        self.assertAlmostEqual(total, 10.0, places=6)

    def test_reported_budget_matches_sleeps(self):
        env = make_env(auto_start_bridge=False)
        slept = []
        with mock.patch.object(gmn_gym.time, "sleep", side_effect=slept.append), \
             mock.patch.object(gmn_gym.requests, "get",
                               side_effect=requests.exceptions.ConnectionError("refused")):
            with self.assertRaises(RuntimeError) as ctx:
                env._ensure_bridge_running()
        self.assertIn("retry budget ~10.0s", str(ctx.exception))

    def test_returns_cleanly_when_healthy(self):
        env = make_env(auto_start_bridge=False)
        resp = mock.Mock(status_code=200)
        resp.json.return_value = {"observation_dim": 127, "action_space_size": 19}
        with mock.patch.object(gmn_gym.requests, "get", return_value=resp):
            self.assertIsNone(env._ensure_bridge_running())

    def test_contract_mismatch_still_raises_immediately(self):
        env = make_env(auto_start_bridge=False)
        resp = mock.Mock(status_code=200)
        resp.json.return_value = {"observation_dim": 115, "action_space_size": 19}
        with mock.patch.object(gmn_gym.requests, "get", return_value=resp):
            with self.assertRaises(RuntimeError) as ctx:
                env._ensure_bridge_running()
        self.assertIn("Contract Mismatch", str(ctx.exception))


class TestDefect3ReapOnClose(unittest.TestCase):
    """Defect 3: close() must reap the bridge this env SPAWNED (port tree-kill),
    and must NOT kill an ADOPTED bridge it does not own."""

    SPAWNED_NETSTAT = (
        "  TCP    0.0.0.0:5399    0.0.0.0:0    LISTENING    12345\n"
        "  TCP    0.0.0.0:5399    0.0.0.0:0    LISTENING    12399\n"
    )

    def test_spawned_close_reaps_port_listeners(self):
        env = make_env()
        env.bridge_process = mock.Mock()      # we own it
        with mock.patch.object(gmn_gym.subprocess, "check_output",
                               return_value=self.SPAWNED_NETSTAT), \
             mock.patch.object(gmn_gym.subprocess, "check_call") as cc:
            cc.return_value = 0
            with mock.patch.object(gmn_gym, "_process_command_line",
                                   return_value="node ... training/bridge_server.ts"):
                env.close()
        kills = [c for c in cc.call_args_list if "taskkill" in c.args[0]]
        self.assertTrue(kills, "spawned close() did not tree-kill the port listener")
        self.assertTrue(any("12345" in c.args[0] for c in kills))
        self.assertTrue(any("12399" in c.args[0] for c in kills))

    def test_adopted_close_does_not_kill_shared_bridge(self):
        """Regression guard: train_ppo shares port 5050 with eval_progress's env.

        Killing an adopted bridge broke that sharing (ConnectionClosedError in
        the final eval), so the adopted path must not port-kill.
        """
        env = make_env()
        env.bridge_process = None            # adopted
        with mock.patch.object(gmn_gym.subprocess, "check_output",
                               return_value=self.SPAWNED_NETSTAT), \
             mock.patch.object(gmn_gym.subprocess, "check_call") as cc:
            cc.return_value = 0
            with mock.patch.object(gmn_gym, "_process_command_line",
                                   return_value="node ... training/bridge_server.ts"):
                env.close()
        self.assertFalse([c for c in cc.call_args_list if "taskkill" in c.args[0]],
                         "close() killed an ADOPTED bridge it does not own")

    def test_close_does_not_kill_unrelated_listener(self):
        env = make_env()
        env.bridge_process = mock.Mock()      # spawned -> reaper runs
        netstat = "  TCP    0.0.0.0:5399    0.0.0.0:0    LISTENING    999\n"
        with mock.patch.object(gmn_gym.subprocess, "check_output",
                               return_value=netstat), \
             mock.patch.object(gmn_gym.subprocess, "check_call") as cc:
            cc.return_value = 0
            with mock.patch.object(gmn_gym, "_process_command_line",
                                   return_value="svchost.exe -k netsvcs"):
                env.close()
        self.assertFalse([c for c in cc.call_args_list if "taskkill" in c.args[0]],
                         "close() killed a non-bridge listener")

    def test_spawned_path_still_terminates_tracked_process(self):
        env = make_env()
        sentinel = mock.Mock()
        env.bridge_process = sentinel
        with mock.patch.object(gmn_gym.subprocess, "check_output",
                               return_value=""), \
             mock.patch.object(gmn_gym.subprocess, "check_call"):
            env.close()
        sentinel.terminate.assert_called_once()
        self.assertIsNone(env.bridge_process)


class TestNetstatParserSafety(unittest.TestCase):
    """The parser must ignore UDP rows for the same port number."""

    def test_ignores_udp_row(self):
        lines = [
            "  UDP    0.0.0.0:5050    *:*    6748",
            "  TCP    0.0.0.0:5050    0.0.0.0:0    LISTENING    3464",
        ]
        self.assertEqual(_parse_netstat_listening_pids(lines, 5050), {"3464"})

    def test_ignores_non_listening_tcp(self):
        lines = [
            "  TCP    127.0.0.1:5050    127.0.0.1:5555    ESTABLISHED    100",
        ]
        self.assertEqual(_parse_netstat_listening_pids(lines, 5050), set())

    def test_port_exact_match_only(self):
        lines = ["  TCP    0.0.0.0:15050   0.0.0.0:0    LISTENING    7"]
        self.assertEqual(_parse_netstat_listening_pids(lines, 5050), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)

