"""
GMN-Football-3 — Pre-step on-ball measurement primitives (MEASUREMENT ONLY).

This module exists to close a temporal-alignment defect in the earlier on-ball
measurement pass (``training/eval_post_reweight_logits.py``), which retained a
tick using the POST-step individual ball owner

    ball_owner_agent_idx = getattr(env, "_last_ball_owner_agent_idx", 255)
    agent0_has_ball = (ball_owner_agent_idx == 0)

while reporting the action mask, actor logits and probabilities computed from
the PRE-step state. That produced a state/selection mismatch.

The corrected protocol retains a tick only when the PRE-step observation of
agent 0 indicates left-team ball ownership::

    retain  <=>  pre_step_obs[95] == 1.0

and derives *every* reported quantity (mask legality, raw logits, masked
logits, probabilities, deterministic action) from that same PRE-step state.

Mutation contract (do not violate): this module must never be used to change
reward, GAE, masks, actor/critic architecture, entropy, optimiser settings,
action taxonomy, environment semantics or checkpoint weights. It only reads
state that the evaluation loop already produced.

Semantics of ``obs[94:97]`` — see ``src/engine/ObservationEncoder.ts:33`` and
``:136-140``: one-hot ``[no-one, left, right]`` for BALL OWNERSHIP AT TEAM
LEVEL. Therefore ``obs[95] == 1.0`` means "the left team has the ball", i.e.
one of the left-team agents (0/1/2) is the individual ball owner. It is **not**
by itself a statement that agent 0 is the individual owner.

The per-agent action mask is the player-level possession signal: the engine
sets ``mask[9..12] = 1`` only when the acting player individually has
possession (``src/engine/ObservationEncoder.ts:394-424``). Both signals are
read from the same pre-step engine state, because the bridge emits
observations and masks in a single frame and the evaluation loop stacks them
before ``env.step()``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "OBS_L_TEAM_OWNERSHIP_INDEX",
    "OBS_BALL_OWNERSHIP_SLICE",
    "ONBALL_SOURCE_PRESTEP",
    "PASS_ACTION_IDS",
    "SHOT_ACTION_IDS",
    "PASS_SHOT_ACTION_IDS",
    "is_prestep_onball",
    "should_retain_prestep_onball_frame",
    "pass_legal_from_mask",
    "shot_legal_from_mask",
    "pass_or_shot_legal_from_mask",
    "action_group_prob",
    "masked_logits_and_probs",
    "build_prestep_onball_frame",
    "summarize_prestep_frames",
    "SUMMARY_COLUMNS",
    "OCCUPANCY_COLUMNS",
    "json_safe",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# obs[94], obs[95], obs[96] == [no-one, left, right] team-level ownership.
OBS_BALL_OWNERSHIP_SLICE = (94, 95, 96)
OBS_L_TEAM_OWNERSHIP_INDEX = 95

# Frame provenance tag. The literal value is part of the artifact schema.
ONBALL_SOURCE_PRESTEP = "obs[95]_pre_step"

# Action grouping already defined by the project (ACTION_NAMES in
# training/eval_post_reweight_logits.py; mask semantics in
# src/engine/ObservationEncoder.ts):
#   9 = LONG_PASS, 10 = HIGH_PASS, 11 = SHORT_PASS, 12 = SHOT
PASS_ACTION_IDS: Sequence[int] = (9, 10, 11)
SHOT_ACTION_IDS: Sequence[int] = (12,)
PASS_SHOT_ACTION_IDS: Sequence[int] = (9, 10, 11, 12)


# ---------------------------------------------------------------------------
# Retention predicate (the single source of truth for frame selection)
# ---------------------------------------------------------------------------

def _as_float_vector(vec: Any) -> np.ndarray:
    """Return ``vec`` as a flat float32 numpy array (never a view of inputs)."""
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def is_prestep_onball(obs_vec: Any) -> bool:
    """True when the PRE-step observation reports left-team ball ownership.

    The test is the exact mandated retention condition
    ``pre_step_obs[95] == 1.0`` (strict equality against the one-hot bit, not a
    rounded comparison).
    """
    if obs_vec is None:
        return False
    arr = _as_float_vector(obs_vec)
    if arr.shape[0] <= OBS_L_TEAM_OWNERSHIP_INDEX:
        return False
    return bool(float(arr[OBS_L_TEAM_OWNERSHIP_INDEX]) == 1.0)


def should_retain_prestep_onball_frame(
    pre_step_obs: Any,
    post_step_ball_owner_agent_idx: Optional[int] = None,
) -> bool:
    """Decide whether a decision tick is retained by the on-ball measurement.

    The ONLY retention condition is the PRE-step observation of agent 0::

        pre_step_obs[95] == 1.0

    ``post_step_ball_owner_agent_idx`` is accepted for call-site symmetry and
    explicit provenance but is **deliberately ignored**: post-step ownership
    must not determine whether the frame is retained (it is recorded in the
    detail artifact as a diagnostic field only).
    """
    return is_prestep_onball(pre_step_obs)


# ---------------------------------------------------------------------------
# Legality helpers (pre-step mask)
# ---------------------------------------------------------------------------

def _as_int_vector(vec: Any) -> np.ndarray:
    return np.asarray(vec).reshape(-1)


def pass_legal_from_mask(mask_vec: Any) -> bool:
    """PASS is legal when any of LONG_PASS/HIGH_PASS/SHORT_PASS (9/10/11) is on."""
    m = _as_int_vector(mask_vec)
    return bool(m[9] or m[10] or m[11])


def shot_legal_from_mask(mask_vec: Any) -> bool:
    """SHOT (12) is legal."""
    m = _as_int_vector(mask_vec)
    return bool(m[12])


def pass_or_shot_legal_from_mask(mask_vec: Any) -> bool:
    return pass_legal_from_mask(mask_vec) or shot_legal_from_mask(mask_vec)


def action_group_prob(probs: Any, action_ids: Iterable[int]) -> float:
    """Sum of policy probability mass over an action group."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    return float(sum(float(p[int(i)]) for i in action_ids))


# ---------------------------------------------------------------------------
# Frame construction (retention + pre-step derivation)
# ---------------------------------------------------------------------------

def build_prestep_onball_frame(
    actor: Any,
    pre_step_obs: Any,
    pre_step_mask: Any,
    *,
    post_step_ball_owner_agent_idx: Optional[int] = None,
    action_names: Optional[Sequence[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build the retained measurement frame for one decision tick.

    Returns ``None`` when the tick is not retained. Retention is decided
    exclusively from the PRE-step observation via
    :func:`should_retain_prestep_onball_frame`; every derived quantity is then
    computed from that *same* pre-step ``(obs, mask)`` pair, so the frame can
    never mix a pre-step observation with a post-step selection or vice versa.
    """
    retained = should_retain_prestep_onball_frame(
        pre_step_obs, post_step_ball_owner_agent_idx
    )
    if not retained:
        return None

    q = masked_logits_and_probs(actor, pre_step_obs, pre_step_mask)

    obs_arr = np.asarray(pre_step_obs, dtype=np.float32).reshape(-1)
    mask_arr = np.asarray(pre_step_mask).reshape(-1)
    action_taken = int(q["deterministic_action"])

    frame: Dict[str, Any] = {
        "onball_source": ONBALL_SOURCE_PRESTEP,
        "obs95": float(obs_arr[OBS_L_TEAM_OWNERSHIP_INDEX]),
        "obs_ball_ownership_slice": [
            float(obs_arr[i]) for i in OBS_BALL_OWNERSHIP_SLICE
        ],
        "action_mask": [int(v) for v in mask_arr],
        "mask_sum": int(q["mask_sum"]),
        "mask_pass_legal": int(q["pass_legal"]),
        "mask_shot_legal": int(q["shot_legal"]),
        "mask_pass_or_shot_legal": int(q["pass_or_shot_legal"]),
        "raw_logits": q["raw_logits"],
        "masked_logits": q["masked_logits"],
        "probs": q["probs"],
        "entropy": float(q["entropy"]),
        "pi_pass": float(q["pi_pass"]),
        "pi_shot": float(q["pi_shot"]),
        "pi_pass_shot": float(q["pi_pass_shot"]),
        "action_taken": action_taken,
        "action_taken_name": (
            action_names[action_taken]
            if action_names is not None and 0 <= action_taken < len(action_names)
            else str(action_taken)
        ),
        "deterministic_action_is_pass_shot": bool(
            q["deterministic_action_is_pass_shot"]
        ),
        # DIAGNOSTIC ONLY: never used for frame retention.
        "post_step_ball_owner_agent_idx": (
            None
            if post_step_ball_owner_agent_idx is None
            else int(post_step_ball_owner_agent_idx)
        ),
    }
    if extra:
        frame.update(extra)
    return frame


# ---------------------------------------------------------------------------
# Actor-derived (policy-distribution) quantities
# ---------------------------------------------------------------------------

def masked_logits_and_probs(actor: Any, obs_vec: Any, mask_vec: Any) -> Dict[str, Any]:
    """Compute policy quantities for one agent from one (pre-step) state.

    ``probs`` come from the actor's own ``forward(obs, action_mask)`` path, so
    the masking semantics are exactly the training-time semantics
    (``masked_fill(~mask, -inf)`` then ``Categorical(logits=...)``; see
    ``training/mappo_networks.py:42-60``). ``raw_logits`` are the pre-mask
    logits from the same forward pass. ``deterministic_action`` is the argmax
    of the masked logits (identical to the argmax of ``probs``).

    Raises whatever the actor raises when the supplied mask leaves no legal
    action: that is a protocol fault, not something to paper over.
    """
    import torch  # local import keeps this module importable without torch

    obs_arr = np.asarray(obs_vec, dtype=np.float32).reshape(-1)
    mask_arr = np.asarray(mask_vec).reshape(-1)
    mask_bool = mask_arr.astype(bool)

    with torch.no_grad():
        obs_t = torch.from_numpy(obs_arr).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask_bool).unsqueeze(0)

        raw_logits = actor.net(obs_t).squeeze(0).cpu().numpy()
        dist = actor(obs_t, mask_t)
        masked_logits = dist.logits.squeeze(0).cpu().numpy()
        probs = dist.probs.squeeze(0).cpu().numpy()
        entropy = float(dist.entropy().item())
        deterministic_action = int(torch.argmax(dist.logits, dim=-1).item())

    pi_pass = action_group_prob(probs, PASS_ACTION_IDS)
    pi_shot = action_group_prob(probs, SHOT_ACTION_IDS)
    pi_pass_shot = pi_pass + pi_shot

    return {
        "raw_logits": raw_logits,
        "masked_logits": masked_logits,
        "probs": probs,
        "entropy": entropy,
        "deterministic_action": deterministic_action,
        "deterministic_action_is_pass_shot": deterministic_action
        in set(int(i) for i in PASS_SHOT_ACTION_IDS),
        "pi_pass": pi_pass,
        "pi_shot": pi_shot,
        "pi_pass_shot": pi_pass_shot,
        "mask_sum": int(mask_bool.sum()),
        "pass_legal": pass_legal_from_mask(mask_arr),
        "shot_legal": shot_legal_from_mask(mask_arr),
        "pass_or_shot_legal": pass_or_shot_legal_from_mask(mask_arr),
    }


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------

def json_safe(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars and non-finite floats for JSON.

    Semantics of ``null`` in the emitted artifacts: the value was not finite.
    In ``masked_logits`` this is exactly the masked-out slot, i.e. the ``-inf``
    produced by the actor's ``masked_fill``. That is documented in the detail
    artifact's ``schema`` block. Using ``null`` avoids writing invalid JSON
    literals such as ``-Infinity`` / ``NaN``.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is None:
        return None
    try:
        # torch scalar tensors
        item = obj.item()  # type: ignore[attr-defined]
        if isinstance(item, (int, float, bool)):
            return json_safe(item)
    except Exception:
        pass
    return obj


# ---------------------------------------------------------------------------
# Aggregation (population definitions are part of the report schema)
# ---------------------------------------------------------------------------

# Column order of post_reweight_logit_prestep_summary.csv. Population labels:
#   * _all_onball        -> Population A: every retained pre-step on-ball frame
#   * _when_pass_legal   -> Population B: Population A AND mask_pass_legal == 1
#   * _when_shot_legal   -> Population C: Population A AND mask_shot_legal == 1
#   * _when_either_legal -> Population D: Population A AND mask_pass_or_shot_legal == 1
# mean_pi_* are POLICY probabilities (actor distribution). p_selected_pass_shot_given_onball
# is a BEHAVIOURAL frequency of the deterministic action and must never be labelled pi_*.
SUMMARY_COLUMNS: Sequence[str] = (
    "seed",
    "checkpoint_label",
    "checkpoint_timesteps",
    "checkpoint",
    "checkpoint_sha256",
    "scenario",
    "base_seed",
    "num_episodes",
    "deterministic",
    "n_ticks",
    "n_onball",
    "p_onball",
    "n_pass_legal",
    "pass_legal_rate",
    "n_shot_legal",
    "shot_legal_rate",
    "n_pass_or_shot_legal",
    "pass_or_shot_legal_rate",
    "mean_pi_pass_all_onball",
    "mean_pi_shot_all_onball",
    "mean_pi_pass_shot_all_onball",
    "mean_pi_pass_when_pass_legal",
    "mean_pi_shot_when_shot_legal",
    "mean_pi_pass_shot_when_either_legal",
    "n_pass_shot_actions",
    "p_selected_pass_shot_given_onball",
    "mask_sum_mean",
    "entropy_mean",
    "n_post_step_owner0",
    "n_onball_post_step_owner_not0",
    "code_commit",
)

OCCUPANCY_COLUMNS: Sequence[str] = (
    "seed",
    "checkpoint_label",
    "checkpoint_timesteps",
    "checkpoint",
    "checkpoint_sha256",
    "scenario",
    "base_seed",
    "num_episodes",
    "n_ticks",
    "n_onball",
    "p_onball",
    "code_commit",
)


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _rate_or_none(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def summarize_prestep_frames(
    frames: Sequence[Dict[str, Any]],
    *,
    n_ticks: int,
    seed: int,
    checkpoint_label: str,
    checkpoint_timesteps: int,
    checkpoint: str,
    checkpoint_sha256: str,
    scenario: str,
    base_seed: int,
    num_episodes: int,
    deterministic: bool = True,
    code_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate retained pre-step frames into one provenance-complete row.

    Every rate/probability carries the exact population size it was computed
    over, so sparse populations cannot be hidden behind a percentage. Empty
    populations yield ``None`` (empty CSV field / JSON null), never a
    fabricated ``0.0``.
    """
    n_onball = len(frames)

    pass_legal_frames = [f for f in frames if int(f["mask_pass_legal"]) == 1]
    shot_legal_frames = [f for f in frames if int(f["mask_shot_legal"]) == 1]
    either_legal_frames = [
        f for f in frames if int(f["mask_pass_or_shot_legal"]) == 1
    ]
    pass_shot_frames = [
        f for f in frames if bool(f["deterministic_action_is_pass_shot"])
    ]

    n_pass_legal = len(pass_legal_frames)
    n_shot_legal = len(shot_legal_frames)
    n_pass_or_shot_legal = len(either_legal_frames)
    n_pass_shot_actions = len(pass_shot_frames)
    n_post_step_owner0 = sum(
        1 for f in frames if f.get("post_step_ball_owner_agent_idx") == 0
    )

    return {
        "seed": int(seed),
        "checkpoint_label": checkpoint_label,
        "checkpoint_timesteps": int(checkpoint_timesteps),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "scenario": scenario,
        "base_seed": int(base_seed),
        "num_episodes": int(num_episodes),
        "deterministic": bool(deterministic),
        "n_ticks": int(n_ticks),
        "n_onball": int(n_onball),
        "p_onball": _rate_or_none(n_onball, n_ticks),
        "n_pass_legal": int(n_pass_legal),
        "pass_legal_rate": _rate_or_none(n_pass_legal, n_onball),
        "n_shot_legal": int(n_shot_legal),
        "shot_legal_rate": _rate_or_none(n_shot_legal, n_onball),
        "n_pass_or_shot_legal": int(n_pass_or_shot_legal),
        "pass_or_shot_legal_rate": _rate_or_none(n_pass_or_shot_legal, n_onball),
        "mean_pi_pass_all_onball": _mean_or_none([f["pi_pass"] for f in frames]),
        "mean_pi_shot_all_onball": _mean_or_none([f["pi_shot"] for f in frames]),
        "mean_pi_pass_shot_all_onball": _mean_or_none(
            [f["pi_pass_shot"] for f in frames]
        ),
        "mean_pi_pass_when_pass_legal": _mean_or_none(
            [f["pi_pass"] for f in pass_legal_frames]
        ),
        "mean_pi_shot_when_shot_legal": _mean_or_none(
            [f["pi_shot"] for f in shot_legal_frames]
        ),
        "mean_pi_pass_shot_when_either_legal": _mean_or_none(
            [f["pi_pass_shot"] for f in either_legal_frames]
        ),
        "n_pass_shot_actions": int(n_pass_shot_actions),
        "p_selected_pass_shot_given_onball": _rate_or_none(
            n_pass_shot_actions, n_onball
        ),
        "mask_sum_mean": _mean_or_none([f["mask_sum"] for f in frames]),
        "entropy_mean": _mean_or_none([f["entropy"] for f in frames]),
        "n_post_step_owner0": int(n_post_step_owner0),
        "n_onball_post_step_owner_not0": int(n_onball - n_post_step_owner0),
        "code_commit": code_commit,
    }
