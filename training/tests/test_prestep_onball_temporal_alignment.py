"""Synthetic regression test: PRE-STEP obs[95] temporal alignment.

Verifies the corrected on-ball measurement rule: a decision tick is retained
ONLY when the PRE-step observation of agent 0 reports left-team ball ownership
(``pre_step_obs[95] == 1.0``). Post-step ownership must never decide retention.

Earlier measurement code determined retention from the POST-step individual
ball owner while reporting pre-step masks/logits/probabilities:

    ball_owner_agent_idx = getattr(env, "_last_ball_owner_agent_idx", 255)
    agent0_has_ball = (ball_owner_agent_idx == 0)          # post-step, wrong

Cases covered:

* Case A: pre_step_obs[95] = 1, post_step_owner != 0  -> RETAIN = True
* Case B: pre_step_obs[95] = 0, post_step_owner = 0   -> RETAIN = False
          (CRITICAL: these frames were retained by the defective implementation)
* Case C: pre_step_obs[95] = 1, post_step_owner = 0   -> RETAIN = True
* Case D: pre_step_obs[95] = 0, post_step_owner != 0  -> RETAIN = False

Additional alignment assertions on the retained frame:

* the action mask used is the PRE-step mask;
* the observation/logits used are the PRE-step observation's;
* the deterministic action is the argmax of the PRE-step masked distribution;
* policy probabilities (pi_PASS/pi_SHOT/pi_PASS+SHOT) are distinct from the
  behavioural deterministic PASS+SHOT action frequency.

These tests import the PRODUCTION predicate/frame-builder/aggregator from
``training.prestep_onball`` (used verbatim by the corrected collector), so a
regression in production logic fails here instead of silently passing against a
duplicated local copy. Runnable under pytest or as a plain script.
"""

import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training.mappo_networks import SharedActor  # noqa: E402
from training.prestep_onball import (  # noqa: E402
    OBS_L_TEAM_OWNERSHIP_INDEX,
    ONBALL_SOURCE_PRESTEP,
    PASS_ACTION_IDS,
    SHOT_ACTION_IDS,
    build_prestep_onball_frame,
    is_prestep_onball,
    summarize_prestep_frames,
)

OBS_DIM = 127
ACTION_DIM = 19
PASS_SHOT_ACTION_IDS = (9, 10, 11, 12)

ACTION_NAMES = [
    "IDLE", "LEFT", "RIGHT", "UP", "DOWN",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
    "LONG_PASS", "HIGH_PASS", "SHORT_PASS",
    "SHOT", "SPRINT",
    "RELEASE_DIRECTION", "RELEASE_SPRINT",
    "SLIDING", "DRIBBLE", "RELEASE_DRIBBLE",
]


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
def _obs(owner: str, marker: float = 0.0) -> np.ndarray:
    """Build a synthetic pre-step observation with a one-hot ownership slice.

    ``owner`` is one of ``"none"`` (obs[94]), ``"left"`` (obs[95], agent 0's
    team) or ``"right"`` (obs[96]). ``marker`` perturbs obs[0] so two states
    can be told apart by the actor.
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = marker
    idx = {"none": 94, "left": 95, "right": 96}[owner]
    obs[idx] = 1.0
    return obs


def _mask(pass_legal: bool, shot_legal: bool) -> np.ndarray:
    """Movement/IDLE legal; PASS (9/10/11) and SHOT (12) toggled explicitly."""
    mask = np.ones(ACTION_DIM, dtype=np.int8)
    for i in PASS_ACTION_IDS:
        mask[i] = 1 if pass_legal else 0
    for i in SHOT_ACTION_IDS:
        mask[i] = 1 if shot_legal else 0
    mask[16] = 0  # TACKLE illegal while in possession (engine rule)
    return mask


def _actor(seed: int = 20260920) -> SharedActor:
    torch.manual_seed(seed)
    actor = SharedActor(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden=64)
    actor.eval()
    return actor


def _independent_probs(raw_logits: np.ndarray, mask_vec: np.ndarray) -> np.ndarray:
    """Independent softmax over masked logits (does not call the actor)."""
    logits = np.asarray(raw_logits, dtype=np.float64).copy()
    logits[~np.asarray(mask_vec).astype(bool)] = -np.inf
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _frame(pre_obs, pre_mask, post_step_owner, actor):
    return build_prestep_onball_frame(
        actor,
        pre_obs,
        pre_mask,
        post_step_ball_owner_agent_idx=post_step_owner,
        action_names=ACTION_NAMES,
        extra={"seed": 42, "episode": 0, "tick": 0},
    )


# ---------------------------------------------------------------------------
# Case A-D: retention predicate
# ---------------------------------------------------------------------------
def test_case_a_pre_step_onball_post_step_not_agent0():
    """Case A: pre_step_obs[95] = 1, post_step_owner != 0 -> RETAIN = True."""
    actor = _actor()
    pre_obs = _obs("left")
    pre_mask = _mask(pass_legal=True, shot_legal=True)

    assert is_prestep_onball(pre_obs) is True
    frame = _frame(pre_obs, pre_mask, post_step_owner=2, actor=actor)

    assert frame is not None, "pre-step on-ball frame must be retained"
    assert frame["onball_source"] == ONBALL_SOURCE_PRESTEP
    assert frame["obs95"] == 1.0
    # The disagreeing post-step owner is recorded as a diagnostic only.
    assert frame["post_step_ball_owner_agent_idx"] == 2
    print("  PASS: Case A retained despite post-step owner != agent 0")


def test_case_b_pre_step_not_onball_post_step_is_agent0():
    """Case B: pre_step_obs[95] = 0, post_step_owner = 0 -> RETAIN = False.

    Critical case: the defective post-step-owner implementation retained exactly
    these frames.
    """
    actor = _actor()
    pre_obs = _obs("none")  # team-level ownership: no-one had the ball pre-step
    pre_mask = _mask(pass_legal=False, shot_legal=False)

    assert is_prestep_onball(pre_obs) is False
    frame = _frame(pre_obs, pre_mask, post_step_owner=0, actor=actor)

    assert frame is None, (
        "frame with pre-step obs[95] == 0 must NOT be retained even when the "
        "post-step ball owner becomes agent 0"
    )
    print("  PASS: Case B not retained (post-step owner could not force retention)")


def test_case_c_both_agree_agent0():
    """Case C: pre_step_obs[95] = 1, post_step_owner = 0 -> RETAIN = True."""
    actor = _actor()
    frame = _frame(
        _obs("left"),
        _mask(pass_legal=True, shot_legal=True),
        post_step_owner=0,
        actor=actor,
    )
    assert frame is not None
    assert frame["obs95"] == 1.0
    assert frame["post_step_ball_owner_agent_idx"] == 0
    print("  PASS: Case C retained (pre-step and post-step agree)")


def test_case_d_neither_indicates_agent0():
    """Case D: pre_step_obs[95] = 0, post_step_owner != 0 -> RETAIN = False."""
    actor = _actor()
    frame = _frame(
        _obs("right"),
        _mask(pass_legal=False, shot_legal=False),
        post_step_owner=255,
        actor=actor,
    )
    assert frame is None
    print("  PASS: Case D not retained")


def test_post_step_owner_cannot_override_pre_step_retention():
    """For every post-step owner value, retention follows pre-step obs[95]."""
    actor = _actor()
    onball_obs = _obs("left")
    offball_obs = _obs("none")

    for post_step_owner in (0, 1, 2, 255):
        assert _frame(onball_obs, _mask(True, True), post_step_owner, actor) is not None
        assert _frame(offball_obs, _mask(False, False), post_step_owner, actor) is None
    print("  PASS: post-step owner never overrides the pre-step predicate")


# ---------------------------------------------------------------------------
# Alignment of retained-frame contents
# ---------------------------------------------------------------------------
def test_retained_frame_uses_pre_step_mask():
    """The retained frame's legality/probabilities come from the PRE-step mask."""
    actor = _actor()
    pre_obs = _obs("left")

    mask_a = _mask(pass_legal=True, shot_legal=False)
    mask_b = _mask(pass_legal=False, shot_legal=True)

    frame_a = _frame(pre_obs, mask_a, post_step_owner=0, actor=actor)
    frame_b = _frame(pre_obs, mask_b, post_step_owner=0, actor=actor)
    assert frame_a is not None and frame_b is not None

    assert frame_a["action_mask"] == [int(v) for v in mask_a]
    assert frame_b["action_mask"] == [int(v) for v in mask_b]

    assert frame_a["mask_pass_legal"] == 1 and frame_a["mask_shot_legal"] == 0
    assert frame_b["mask_pass_legal"] == 0 and frame_b["mask_shot_legal"] == 1

    # Masked-out actions must carry exactly zero probability mass.
    assert frame_a["pi_pass"] > 0.0 and frame_a["pi_shot"] == 0.0
    assert frame_b["pi_pass"] == 0.0 and frame_b["pi_shot"] > 0.0
    print("  PASS: retained frame uses the pre-step mask")


def test_retained_frame_uses_pre_step_observation_and_logits():
    """Logits/probabilities in the retained frame are the PRE-step state's."""
    actor = _actor()
    pre_obs = _obs("left", marker=0.25)
    other_obs = _obs("left", marker=-0.75)  # a different state
    pre_mask = _mask(pass_legal=True, shot_legal=True)

    frame = _frame(pre_obs, pre_mask, post_step_owner=3, actor=actor)
    assert frame is not None

    assert frame["obs95"] == 1.0
    assert frame["obs_ball_ownership_slice"] == [0.0, 1.0, 0.0]

    with torch.no_grad():
        expected = (
            actor.net(torch.from_numpy(pre_obs).float().unsqueeze(0))
            .squeeze(0)
            .cpu()
            .numpy()
        )
        divergent = (
            actor.net(torch.from_numpy(other_obs).float().unsqueeze(0))
            .squeeze(0)
            .cpu()
            .numpy()
        )

    np.testing.assert_allclose(frame["raw_logits"], expected, rtol=0, atol=0)
    assert not np.allclose(frame["raw_logits"], divergent, atol=1e-6), (
        "frame logits must come from the pre-step observation, not another state"
    )

    # Probabilities must equal an independent softmax of the masked logits.
    np.testing.assert_allclose(
        frame["probs"], _independent_probs(expected, pre_mask), rtol=1e-6, atol=1e-9
    )
    assert abs(float(frame["probs"].sum()) - 1.0) < 1e-6
    print("  PASS: retained frame uses the pre-step observation/logits")


def test_deterministic_action_from_pre_step_policy():
    """The recorded action is the argmax of the PRE-step masked distribution."""
    actor = _actor()
    pre_obs = _obs("left", marker=0.1)
    pre_mask = _mask(pass_legal=True, shot_legal=True)

    frame = _frame(pre_obs, pre_mask, post_step_owner=0, actor=actor)
    assert frame is not None

    probs = np.asarray(frame["probs"], dtype=np.float64)
    legal = [i for i, v in enumerate(frame["action_mask"]) if int(v) == 1]

    assert frame["action_taken"] == int(np.argmax(probs))
    assert frame["action_taken"] in legal, "action must be legal under the pre-step mask"
    assert frame["action_taken_name"] == ACTION_NAMES[frame["action_taken"]]
    assert frame["deterministic_action_is_pass_shot"] == (
        frame["action_taken"] in PASS_SHOT_ACTION_IDS
    )
    print("  PASS: deterministic action derived from the pre-step policy")


def _summarize_probe(frames):
    return summarize_prestep_frames(
        frames,
        n_ticks=2550,
        seed=42,
        checkpoint_label="probe.pt",
        checkpoint_timesteps=49920,
        checkpoint="probe.pt",
        checkpoint_sha256="0" * 64,
        scenario="synthetic",
        base_seed=700000,
        num_episodes=50,
        deterministic=True,
        code_commit="test",
    )


def test_policy_probability_not_confused_with_action_frequency():
    """pi_PASS/pi_SHOT are policy probabilities, not selected-action frequency."""
    actor = _actor()
    frames = []
    for marker in (0.0, 0.4, -0.2):
        frame = _frame(
            _obs("left", marker=marker),
            _mask(pass_legal=True, shot_legal=True),
            post_step_owner=0,
            actor=actor,
        )
        assert frame is not None
        frames.append(frame)

    row = _summarize_probe(frames)

    expected_pi = float(np.mean([f["pi_pass_shot"] for f in frames]))
    assert row["mean_pi_pass_shot_all_onball"] is not None
    assert abs(row["mean_pi_pass_shot_all_onball"] - expected_pi) < 1e-12
    assert 0.0 <= row["mean_pi_pass_shot_all_onball"] <= 1.0

    # Behavioural statistic: hand-set the deterministic selection to non-PASS/SHOT
    # while leaving the policy probabilities untouched. The two families must not
    # move together.
    for f in frames:
        f["deterministic_action_is_pass_shot"] = False
    row2 = _summarize_probe(frames)

    assert row2["n_pass_shot_actions"] == 0
    assert row2["p_selected_pass_shot_given_onball"] == 0.0
    assert row2["mean_pi_pass_shot_all_onball"] == row["mean_pi_pass_shot_all_onball"]
    assert row2["mean_pi_pass_shot_all_onball"] > 0.0

    # Naming contract: no behavioural field may masquerade as a probability.
    for key in row:
        if key.startswith("pi_") or key.startswith("mean_pi_"):
            assert "selected" not in key and "frequency" not in key
    assert "p_selected_pass_shot_given_onball" in row
    print("  PASS: policy probabilities and behavioural frequency are separate")


def run_all_tests():
    """Run every synthetic regression test in this module."""
    tests = [
        test_case_a_pre_step_onball_post_step_not_agent0,
        test_case_b_pre_step_not_onball_post_step_is_agent0,
        test_case_c_both_agree_agent0,
        test_case_d_neither_indicates_agent0,
        test_post_step_owner_cannot_override_pre_step_retention,
        test_retained_frame_uses_pre_step_mask,
        test_retained_frame_uses_pre_step_observation_and_logits,
        test_deterministic_action_from_pre_step_policy,
        test_policy_probability_not_confused_with_action_frequency,
    ]
    print("=" * 70)
    print("SYNTHETIC REGRESSION TEST: PRE-STEP obs[95] TEMPORAL ALIGNMENT")
    print("=" * 70)
    for test in tests:
        print(f"\n[{test.__name__}]")
        test()
    print()
    print("=" * 70)
    print(f"ALL {len(tests)} PRE-STEP TEMPORAL ALIGNMENT TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
