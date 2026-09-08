"""
GMN-Football-3 — POST /step_multi End-to-End HTTP Regression Test

Verifies that the HTTP bridge server's POST /step_multi endpoint returns
valid non-empty JSON with the expected step-result shape.

Bug: bridge.stepMulti() was converted to async but one HTTP handler call site
(forgot await), so JSON.stringify(multiResult) serialized a Promise object
({}) instead of the real result.
"""
import json
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from training.gmn_pettingzoo import GMNMultiAgentEnv


class TestStepMultiHTTP:
    """End-to-end test for POST /step_multi HTTP handler."""

    @staticmethod
    def _make_env():
        port = 5060 + hash("step_multi_http_test") % 1000
        env = GMNMultiAgentEnv(
            scenario="academy_empty_goal",
            auto_start_bridge=True,
            port=port,
            enable_reward_shaping=False,
        )
        return env

    def test_step_multi_returns_valid_json_shape(self):
        """
        Reset the env, then hit POST /step_multi directly via HTTP.
        Assert the response is valid non-empty JSON containing the expected
        top-level keys from bridge.stepMulti().
        """
        env = self._make_env()
        try:
            # Reset to discover controllable agents and ensure bridge is ready
            obs, info = env.reset(seed=42)
            controllable_ids = info.get("agent_order") or list(obs.keys())
            n_agents = len(controllable_ids)
            assert n_agents >= 1

            # Build a minimal valid action array (all no-ops)
            actions = [0] * n_agents

            base_url = env.base_url
            resp = requests.post(
                f"{base_url}/step_multi",
                json={"actions": actions},
                timeout=10.0,
            )

            # The endpoint must return 200, not a 500 or empty body
            assert resp.status_code == 200, (
                f"POST /step_multi returned {resp.status_code}: {resp.text}"
            )

            # Body must be valid non-empty JSON
            data = resp.json()
            assert isinstance(data, dict), f"Expected dict, got {type(data)}: {data}"
            assert data, "Response body is empty — likely a serialized Promise from missing await"

            # Expected shape from bridge.stepMulti()
            expected_keys = {"reward", "terminated", "truncated", "info", "observations", "controllableIds"}
            assert expected_keys.issubset(data.keys()), (
                f"Missing keys in /step_multi response. Expected {expected_keys}, got {set(data.keys())}"
            )

            # reward must be numeric, not undefined
            assert isinstance(data["reward"], (int, float)), (
                f"reward should be numeric, got {type(data['reward'])}: {data['reward']}"
            )

            # observations must be a list with one entry per controllable agent
            assert isinstance(data["observations"], list), (
                f"observations should be list, got {type(data['observations'])}"
            )
            assert len(data["observations"]) == n_agents, (
                f"Expected {n_agents} observations, got {len(data['observations'])}"
            )

            # controllableIds must match what we sent
            assert data["controllableIds"] == controllable_ids, (
                f"controllableIds mismatch: {data['controllableIds']} != {controllable_ids}"
            )
        finally:
            env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
