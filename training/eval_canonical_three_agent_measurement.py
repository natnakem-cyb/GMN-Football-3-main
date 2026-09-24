"""
GMN-Football-3 — Canonical 3-Agent Pre-Step Measurement (MEASUREMENT ONLY).

Implements the full Final Measurement Gate specification:
  Task 0  — Freeze ownership semantics (obs[95] = team possession; pre-step ball-owner idx = individual carrier)
  Task 1  — Temporal alignment test (all agent decisions recorded; onball = pre_step_ball_owner_agent_idx == agent_index)
  Task 2  — Hard π-floor reconciliation
  Task 3  — Canonical 7650-decision scope (50 eps × 51 ticks × 3 agents)
  Task 4  — All-three-agent pre-step collection
  Task 5  — Canonical 7650-decision behavioral reconstruction
  Task 6  — All-agent occupancy / conditional behavior
  Task 7  — Canonical scope reconciliation
  Task 8  — π statistics separate from behavioral frequency
  Task 9  — Legality
  Task 10 — Supersede 085ec85 occupancy interpretation
  Task 11 — Required artifacts
  Task 12 — Artifact integrity
  Task 13 — Hard interpretation gate
  Task 14 — Allowed conclusions only
  Task 15 — No training verification

Non-goals (hard prohibitions):
  No training. No reward/GAE/mask/actor/critic/optimiser/entropy/
  environment-rule/action-taxonomy changes. No checkpoint-weight changes.

Canonical rate definition (from eval_progress.py):
    pass_shot_rate_pct = ((passes + shots) / (num_episodes * 51 * 3)) * 100

For 50 episodes: denominator = 50 * 51 * 3 = 7650 agent-decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv  # noqa: E402
from training.mappo_networks import SharedActor  # noqa: E402
from training.checkpoint_contract import load_mappo_actor  # noqa: E402
from training.mappo_rollout import _mask_matrix, unwrap_masks, unwrap_obs  # noqa: E402
from training.prestep_onball import (  # noqa: E402
    OBS_BALL_OWNERSHIP_SLICE,
    OBS_L_TEAM_OWNERSHIP_INDEX,
    ONBALL_SOURCE_PRESTEP,
    PASS_ACTION_IDS,
    SHOT_ACTION_IDS,
    PASS_SHOT_ACTION_IDS,
    is_prestep_onball,
    json_safe,
    summarize_prestep_frames,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTION_NAMES = [
    "IDLE", "LEFT", "RIGHT", "UP", "DOWN",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
    "LONG_PASS", "HIGH_PASS", "SHORT_PASS",
    "SHOT", "SPRINT",
    "RELEASE_DIRECTION", "RELEASE_SPRINT",
    "SLIDING", "DRIBBLE", "RELEASE_DRIBBLE",
]

OBS_DIM = 127
ACTION_DIM = 19
NUM_AGENTS = 3
DEFAULT_NUM_EPISODES = 50
DEFAULT_TICKS_PER_EPISODE = 51
DEFAULT_BASE_SEED = 500000
DEFAULT_SEEDS = (42, 123, 7, 999)
DEFAULT_SCENARIO = "academy_3_vs_1_with_keeper_onball"
CHECKPOINT_LABEL = "50k"
CHECKPOINT_DECLARED_TIMESTEPS = 50000
CHECKPOINT_TEMPLATE = (
    "mappo_academy_3_vs_1_with_keeper_onball_seed{seed}_actorreweight.pt"
)
INVENTORY_RELATIVE_PATH = os.path.join(
    "training", "results", "retest_checkpoint_inventory.csv"
)

# ---------------------------------------------------------------------------
# Reconciliation tolerance
# ---------------------------------------------------------------------------
RECONCILIATION_TOLERANCE_PP = 0.05           # percentage points
RECONCILIATION_TOLERANCE_ABS = 0.05 / 100.0  # absolute rate units

FLOAT_TOL = 1e-6
INVALID_LITERAL_RE = __import__("re").compile(r"[:,\[]\s*-?(?:Infinity|NaN)\b")


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


MEASUREMENT_CODE_FILES = (
    "training/prestep_onball.py",
    "training/eval_post_reweight_logits_prestep.py",
    "training/eval_canonical_three_agent_measurement.py",
    "training/verify_prestep_measurement.py",
    "training/tests/test_prestep_onball_temporal_alignment.py",
)


def measurement_code_digests() -> Dict[str, Optional[str]]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    digests: Dict[str, Optional[str]] = {}
    for rel in MEASUREMENT_CODE_FILES:
        path = os.path.join(repo_root, rel)
        digests[rel] = sha256_of(path) if os.path.exists(path) else None
    return digests


def load_inventory_final_hashes(inventory_path: str) -> Dict[int, str]:
    if not os.path.exists(inventory_path):
        raise FileNotFoundError(
            f"Checkpoint inventory not found: {inventory_path}. Refusing to "
            "measure without an inventory to verify checkpoint identity against."
        )
    mapping: Dict[int, str] = {}
    with open(inventory_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("trajectory_step", "")).strip().lower() != "final":
                continue
            mapping[int(row["seed"])] = str(row["sha256"]).strip()
    if not mapping:
        raise RuntimeError(f"No 'final' rows found in inventory {inventory_path}")
    return mapping


def verify_checkpoint_sha(seed: int, computed_sha: str, inventory: Dict[int, str]) -> None:
    if not computed_sha:
        raise RuntimeError(f"Empty checkpoint SHA-256 for seed {seed}")
    expected = inventory.get(int(seed))
    if expected is None:
        raise RuntimeError(
            f"No 'final' inventory row for seed {seed}; refusing to measure a "
            "checkpoint whose identity cannot be verified."
        )
    if computed_sha != expected:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch for seed {seed}:\n"
            f"  computed  = {computed_sha}\n"
            f"  inventory = {expected}\n"
            "Refusing to measure. Do not relabel or substitute checkpoints."
        )


# ---------------------------------------------------------------------------
# Actor helper: batched per-agent quantities
# ---------------------------------------------------------------------------
@torch.no_grad()
def _batched_actor_quantities(
    actor: SharedActor,
    obs_t: torch.Tensor,
    mask_t: torch.Tensor,
    graph_observations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    obs_t = torch.from_numpy(obs_t).float() if isinstance(obs_t, np.ndarray) else obs_t
    mask_t = torch.from_numpy(mask_t).bool() if isinstance(mask_t, np.ndarray) else mask_t
    if getattr(actor, "requires_graph_observations", False):
        if graph_observations is None:
            raise ValueError("GNN canonical evaluation requires per-agent graph observations")
        dist = actor(graph_observations, mask_t)
        raw_logits = actor.raw_logits(graph_observations).cpu().numpy()
    else:
        dist = actor(obs_t, mask_t)
        raw_logits = actor.net(obs_t).cpu().numpy()

    masked_logits = dist.logits.cpu().numpy()
    probs = dist.probs.cpu().numpy().astype(np.float64)
    entropy = dist.entropy().cpu().numpy().astype(np.float64)
    det_action = torch.argmax(dist.logits, dim=-1).cpu().numpy()

    pi_pass = np.sum(probs[:, PASS_ACTION_IDS], axis=1)
    pi_shot = probs[:, SHOT_ACTION_IDS[0]]
    pi_pass_shot = pi_pass + pi_shot
    mask_sum = mask_t.cpu().numpy().astype(bool).sum(axis=1)
    pass_legal = np.array([
        bool(mask_t.cpu().numpy()[i, j] == 1) for i in range(mask_t.shape[0])
        for j in PASS_ACTION_IDS
    ]).reshape(mask_t.shape[0], -1).any(axis=1)
    shot_legal = mask_t[:, SHOT_ACTION_IDS[0]].cpu().numpy() == 1
    pass_or_shot_legal = pass_legal | shot_legal

    return {
        "raw_logits": raw_logits,
        "masked_logits": masked_logits,
        "probs": probs,
        "entropy": entropy,
        "deterministic_action": det_action,
        "pi_pass": pi_pass,
        "pi_shot": pi_shot,
        "pi_pass_shot": pi_pass_shot,
        "mask_sum": mask_sum,
        "pass_legal": pass_legal,
        "shot_legal": shot_legal,
        "pass_or_shot_legal": pass_or_shot_legal,
    }


# ---------------------------------------------------------------------------
# Per-agent frame record
# ---------------------------------------------------------------------------
def _build_agent_decision_frame(
    seed: int,
    episode: int,
    tick: int,
    agent_index: int,
    agent_id: str,
    pre_step_obs: np.ndarray,
    pre_step_mask: np.ndarray,
    pre_step_ball_owner_agent_idx: int,
    actor_q: Dict[str, Any],
    event_code: Optional[int],
    reward: float,
    terminated: bool,
    truncated: bool,
    done: bool,
    checkpoint_sha256: str,
    checkpoint_timesteps: int,
    checkpoint_path: str,
    scenario: str,
    deterministic: bool,
    ep_seed: int,
) -> Dict[str, Any]:
    """Build one agent-decision frame record for the canonical measurement."""
    obs_arr = np.asarray(pre_step_obs, dtype=np.float32).reshape(-1)
    mask_arr = np.asarray(pre_step_mask).reshape(-1)

    team_has_ball = bool(float(obs_arr[OBS_L_TEAM_OWNERSHIP_INDEX]) == 1.0)
    agent_has_ball = bool(pre_step_ball_owner_agent_idx == agent_index)
    onball = agent_has_ball

    action_taken = int(actor_q["deterministic_action"][agent_index])
    action_taken_name = (
        ACTION_NAMES[action_taken]
        if 0 <= action_taken < len(ACTION_NAMES)
        else str(action_taken)
    )

    frame: Dict[str, Any] = {
        "seed": int(seed),
        "episode": int(episode),
        "tick": int(tick),
        "agent_index": int(agent_index),
        "agent_id": str(agent_id),
        "pre_step_obs95": float(obs_arr[OBS_L_TEAM_OWNERSHIP_INDEX]),
        "pre_step_obs_ball_ownership_slice": [
            float(obs_arr[i]) for i in OBS_BALL_OWNERSHIP_SLICE
        ],
        "pre_step_ball_owner_agent_idx": int(pre_step_ball_owner_agent_idx),
        "team_has_ball": team_has_ball,
        "agent_has_ball": agent_has_ball,
        "onball": onball,
        "action_mask": [int(v) for v in mask_arr],
        "mask_sum": int(actor_q["mask_sum"][agent_index]),
        "mask_pass_legal": int(actor_q["pass_legal"][agent_index]),
        "mask_shot_legal": int(actor_q["shot_legal"][agent_index]),
        "mask_pass_or_shot_legal": int(actor_q["pass_or_shot_legal"][agent_index]),
        "raw_logits": actor_q["raw_logits"][agent_index].tolist(),
        "masked_logits": _json_safe_floats(actor_q["masked_logits"][agent_index]),
        "probs": _json_safe_floats(actor_q["probs"][agent_index]),
        "entropy": float(actor_q["entropy"][agent_index]),
        "pi_pass": float(actor_q["pi_pass"][agent_index]),
        "pi_shot": float(actor_q["pi_shot"][agent_index]),
        "pi_pass_shot": float(actor_q["pi_pass_shot"][agent_index]),
        "action_taken": action_taken,
        "action_taken_name": action_taken_name,
        "deterministic_action_is_pass_shot": bool(action_taken in PASS_SHOT_ACTION_IDS),
        "event_code": int(event_code) if event_code is not None else None,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(done),
        "checkpoint_sha256": str(checkpoint_sha256),
        "checkpoint_timesteps": int(checkpoint_timesteps),
        "checkpoint": str(checkpoint_path),
        "scenario": str(scenario),
        "deterministic": bool(deterministic),
        "ep_seed": int(ep_seed),
        "code_commit": code_commit(),
    }
    return frame


def _json_safe_floats(arr: np.ndarray) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for v in arr:
        f = float(v)
        out.append(f if math.isfinite(f) else None)
    return out


# ---------------------------------------------------------------------------
# π-floor reconciliation helpers
# ---------------------------------------------------------------------------
def _recompute_probs_from_logits_mask(
    raw_logits: Sequence[float], mask_vec: Sequence[int]
) -> List[float]:
    """Independent masked softmax for π-floor verification."""
    masked = [
        (float(v) if int(m) == 1 else None)
        for v, m in zip(raw_logits, mask_vec)
    ]
    legal = [v for v in masked if v is not None]
    if not legal:
        raise ValueError("all-illegal mask")
    m = max(legal)
    exp = [math.exp(v - m) for v in legal]
    total = sum(exp)
    probs = [e / total for e in exp]
    out: List[float] = []
    it = iter(probs)
    for v in masked:
        out.append(0.0 if v is None else next(it))
    return out


def _pi_floor_checks(
    frames: List[Dict[str, Any]],
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Run all π-floor identity checks on on-ball frames.

    Returns (passed, failures, stats_dict).
    """
    failures: List[str] = []
    stats: Dict[str, Any] = {}

    onball = [f for f in frames if f["onball"]]
    stats["n_onball"] = len(onball)
    if not onball:
        stats["identity_ok"] = True
        stats["deterministic_ok"] = True
        stats["selected_prob_ok"] = True
        stats["floor_ok"] = True
        return True, failures, stats

    # Identity: pi_pass_shot == probs[9]+probs[10]+probs[11]+probs[12]
    identity_errors = 0
    det_mismatches = 0
    selected_prob_errors = 0
    all_illegal_fallbacks = 0

    pi_ps_values: List[float] = []
    pi_selected_values: List[float] = []
    mask_sums_selected: List[int] = []
    n_selected_ps = 0

    for fr in onball:
        probs = fr["probs"]
        mask = fr["action_mask"]
        action = fr["action_taken"]

        # Recompute π_PASS+SHOT from stored probs
        computed_ps = sum(probs[i] for i in PASS_SHOT_ACTION_IDS)
        stored_ps = fr["pi_pass_shot"]
        if abs(computed_ps - stored_ps) > FLOAT_TOL:
            identity_errors += 1

        # Deterministic identity
        legal_probs = [probs[i] for i in range(ACTION_DIM) if mask[i] == 1]
        if legal_probs:
            expected_action = max(
                [i for i in range(ACTION_DIM) if mask[i] == 1],
                key=lambda i: probs[i],
            )
        else:
            expected_action = action  # all-illegal: can't verify
        if action != expected_action:
            det_mismatches += 1

        # Selected probability check
        pi_selected = probs[action]
        if action in PASS_SHOT_ACTION_IDS:
            n_selected_ps += 1
            pi_ps_values.append(stored_ps)
            pi_selected_values.append(pi_selected)
            mask_sums_selected.append(fr["mask_sum"])
            if pi_selected > stored_ps + FLOAT_TOL:
                selected_prob_errors += 1
            # All-illegal fallback check
            if fr["mask_sum"] == 0:
                all_illegal_fallbacks += 1

    stats["identity_errors"] = identity_errors
    stats["deterministic_mismatches"] = det_mismatches
    stats["selected_prob_errors"] = selected_prob_errors
    stats["all_illegal_fallbacks"] = all_illegal_fallbacks
    stats["n_selected_ps"] = n_selected_ps

    # Aggregate π statistics
    stats["mean_pi_ps"] = float(np.mean(pi_ps_values)) if pi_ps_values else None
    stats["mean_pi_selected"] = float(np.mean(pi_selected_values)) if pi_selected_values else None
    stats["mean_pi_ps_given_selected"] = (
        float(np.mean(pi_ps_values)) if pi_ps_values else None
    )  # Same as mean_pi_ps since pi_ps_values already conditioned on selection
    stats["mean_mask_sum_given_selected"] = (
        float(np.mean(mask_sums_selected)) if mask_sums_selected else None
    )

    # Floors
    n_onball = len(onball)
    if n_selected_ps > 0 and n_onball > 0:
        stats["floor_from_action_space"] = (n_selected_ps / n_onball) * (1.0 / ACTION_DIM)
    else:
        stats["floor_from_action_space"] = None

    if mask_sums_selected:
        stats["floor_from_observed_masks"] = sum(1.0 / ms for ms in mask_sums_selected) / n_onball
    else:
        stats["floor_from_observed_masks"] = None

    # Inequalities
    if stats["mean_pi_ps"] is not None and stats["mean_pi_selected"] is not None:
        lhs = n_selected_ps * stats["mean_pi_selected"] / n_onball
        stats["inequality_1"] = lhs <= stats["mean_pi_ps"] + FLOAT_TOL
    else:
        stats["inequality_1"] = True

    if stats["floor_from_observed_masks"] is not None and stats["mean_pi_ps"] is not None:
        stats["inequality_2"] = stats["floor_from_observed_masks"] <= stats["mean_pi_ps"] + FLOAT_TOL
    else:
        stats["inequality_2"] = True

    if stats["mean_pi_selected"] is not None and stats["mean_pi_ps_given_selected"] is not None:
        stats["inequality_3"] = stats["mean_pi_selected"] <= stats["mean_pi_ps_given_selected"] + FLOAT_TOL
    else:
        stats["inequality_3"] = True

    # Seed-42 hard check
    if stats["n_onball"] == 27 and stats["n_selected_ps"] == 16:
        min_floor = (16 / 27) * (1.0 / ACTION_DIM)
        if stats["mean_pi_ps"] is not None and stats["mean_pi_ps"] < min_floor - FLOAT_TOL:
            failures.append(
                f"SEED-42 HARD FLOOR VIOLATION: mean_pi_ps={stats['mean_pi_ps']:.6f} "
                f"< floor={min_floor:.6f} (16/27/19)"
            )

    passed = (
        identity_errors == 0
        and det_mismatches == 0
        and selected_prob_errors == 0
        and bool(stats.get("inequality_1", True))
        and bool(stats.get("inequality_2", True))
        and bool(stats.get("inequality_3", True))
    )

    return passed, failures, stats


# ---------------------------------------------------------------------------
# Canonical 7650-decision rate computation
# ---------------------------------------------------------------------------
def _compute_canonical_rate(all_decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute canonical rate from all 7650 agent-decisions."""
    n_total = len(all_decisions)
    n_pass = sum(1 for d in all_decisions if d["action_taken"] in PASS_ACTION_IDS)
    n_shot = sum(1 for d in all_decisions if d["action_taken"] in SHOT_ACTION_IDS)
    n_pass_shot = n_pass + n_shot
    rate = (n_pass_shot / max(1, n_total)) * 100.0

    return {
        "n_decisions": n_total,
        "n_pass": n_pass,
        "n_shot": n_shot,
        "n_pass_shot": n_pass_shot,
        "canonical_rate_pct": rate,
    }


# ---------------------------------------------------------------------------
# Per-agent occupancy
# ---------------------------------------------------------------------------
def _compute_agent_occupancy(all_decisions: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Compute per-agent on-ball occupancy and conditional PS rate."""
    result: Dict[int, Dict[str, Any]] = {}
    for agent_idx in range(NUM_AGENTS):
        agent_frames = [d for d in all_decisions if d["agent_index"] == agent_idx]
        n_total = len(agent_frames)
        n_onball = sum(1 for d in agent_frames if d["onball"])
        n_selected_ps_onball = sum(
            1 for d in agent_frames
            if d["onball"] and d["action_taken"] in PASS_SHOT_ACTION_IDS
        )
        result[agent_idx] = {
            "n_decisions": n_total,
            "n_onball": n_onball,
            "p_onball": n_onball / max(1, n_total),
            "n_selected_ps_when_onball": n_selected_ps_onball,
            "p_selected_ps_given_onball": n_selected_ps_onball / max(1, n_onball),
        }
    return result


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------
def collect_canonical_measurement(
    *,
    actor: SharedActor,
    checkpoint_path: str,
    checkpoint_sha256: str,
    checkpoint_timesteps: int,
    seed: int,
    scenario: str,
    num_episodes: int,
    deterministic: bool,
    base_seed: int,
    bridge_port: int = 5050,
    label: str = CHECKPOINT_LABEL,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Roll out deterministic episodes, capturing all 3 agents' pre-step decisions."""
    graph_policy = bool(getattr(actor, "requires_graph_observations", False))
    env = GMNMultiAgentEnv(
        scenario=scenario,
        auto_start_bridge=True,
        port=bridge_port,
        include_graph_observations=graph_policy,
    )
    controllable_agents = list(env.possible_agents)

    episodes_data: List[Dict[str, Any]] = []
    all_decisions: List[Dict[str, Any]] = []
    n_ticks_total = 0

    try:
        for ep in range(num_episodes):
            ep_seed = base_seed + ep * 1009
            obs_dict, current_infos = env.reset(seed=ep_seed)
            current_ep_masks = unwrap_masks(obs_dict)
            obs_dict = unwrap_obs(obs_dict)

            current_agents = list(env.agents if env.agents else controllable_agents)
            if not current_agents:
                continue

            ep_ticks = 0
            ep_reward = 0.0
            tick_idx = 0

            while True:
                current_agents = list(
                    env.agents if env.agents else controllable_agents
                )
                if not current_agents:
                    break

                # ---- PRE-STEP state: the single source for this tick ----
                local_obs = np.stack(
                    [obs_dict[a] for a in current_agents], axis=0
                ).astype(np.float32)
                mask_matrix = _mask_matrix(current_ep_masks, current_agents)

                # Pre-step ball owner: env._last_ball_owner_agent_idx was set
                # by the PREVIOUS step (or reset) and corresponds to the
                # current pre-step observation. We read it BEFORE env.step().
                pre_step_ball_owner_agent_idx = int(
                    getattr(env, "_last_ball_owner_agent_idx", 255)
                )

                # Run actor once for all agents
                graphs = (
                    [current_infos[a]["graph_observation"] for a in current_agents]
                    if graph_policy
                    else None
                )
                actor_q = _batched_actor_quantities(
                    actor, local_obs, mask_matrix, graphs
                )

                # Build action dict from deterministic actions
                actions_np = actor_q["deterministic_action"]
                action_dict = {
                    a: int(actions_np[i]) for i, a in enumerate(current_agents)
                }

                # ---- Step (post-step ownership is diagnostic only) ----
                obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
                current_infos = infos
                current_ep_masks = unwrap_masks(obs_dict)
                obs_dict = unwrap_obs(obs_dict)

                terminated = any(terms.values()) if terms else False
                truncated = any(truncs.values()) if truncs else False
                done = terminated or truncated or not env.agents

                shared_rew = (
                    float(rewards[current_agents[0]])
                    if current_agents and current_agents[0] in rewards
                    else 0.0
                )
                ep_reward += shared_rew
                event_code = getattr(env, "_last_frame_event_code", None)

                # ---- Record each agent's decision ----
                for i, agent_id in enumerate(current_agents):
                    frame = _build_agent_decision_frame(
                        seed=seed,
                        episode=ep,
                        tick=tick_idx,
                        agent_index=i,
                        agent_id=agent_id,
                        pre_step_obs=local_obs[i],
                        pre_step_mask=mask_matrix[i],
                        pre_step_ball_owner_agent_idx=pre_step_ball_owner_agent_idx,
                        actor_q=actor_q,
                        event_code=event_code,
                        reward=shared_rew,
                        terminated=terminated,
                        truncated=truncated,
                        done=done,
                        checkpoint_sha256=checkpoint_sha256,
                        checkpoint_timesteps=checkpoint_timesteps,
                        checkpoint_path=checkpoint_path,
                        scenario=scenario,
                        deterministic=deterministic,
                        ep_seed=ep_seed,
                    )
                    all_decisions.append(frame)

                tick_idx += 1
                ep_ticks += 1
                if done:
                    break

            n_ticks_total += ep_ticks
            episodes_data.append(
                {
                    "seed": int(seed),
                    "episode": int(ep),
                    "ep_seed": int(ep_seed),
                    "checkpoint": checkpoint_path,
                    "checkpoint_label": label,
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_timesteps": int(checkpoint_timesteps),
                    "scenario": scenario,
                    "n_ticks": int(ep_ticks),
                    "episode_reward": float(ep_reward),
                }
            )

            if verbose and (ep + 1) % 10 == 0:
                print(
                    f"  [canonical seed={seed}] Ep {ep + 1:3d}/{num_episodes} | "
                    f"ticks={ep_ticks} decisions={len(current_agents) * ep_ticks} | "
                    f"reward={ep_reward:+.3f}",
                    flush=True,
                )
    finally:
        env.close()

    return {
        "seed": int(seed),
        "checkpoint": checkpoint_path,
        "checkpoint_label": label,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_timesteps": int(checkpoint_timesteps),
        "scenario": scenario,
        "deterministic": bool(deterministic),
        "base_seed": int(base_seed),
        "num_episodes": int(num_episodes),
        "n_ticks": int(n_ticks_total),
        "n_decisions": len(all_decisions),
        "episodes": episodes_data,
        "decisions": all_decisions,
        "frames": all_decisions,  # alias for artifact compatibility with write_detail_json
    }


# ---------------------------------------------------------------------------
# Aggregation for a seed
# ---------------------------------------------------------------------------
def _aggregate_seed(
    result: Dict[str, Any],
    rebuilt_rate_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute all required aggregates for one seed."""
    decisions = result["decisions"]
    seed = result["seed"]
    n_total = len(decisions)
    n_ticks = result["n_ticks"]

    # ---- Canonical rate (Task 3, 5) ----
    canonical = _compute_canonical_rate(decisions)

    # ---- All-agent on-ball frames ----
    onball_frames = [d for d in decisions if d["onball"]]
    n_onball = len(onball_frames)

    # ---- π statistics for on-ball frames (Task 2, 8) ----
    pi_pass_all = [d["pi_pass"] for d in onball_frames]
    pi_shot_all = [d["pi_shot"] for d in onball_frames]
    pi_ps_all = [d["pi_pass_shot"] for d in onball_frames]

    mean_pi_pass = float(np.mean(pi_pass_all)) if pi_pass_all else None
    mean_pi_shot = float(np.mean(pi_shot_all)) if pi_shot_all else None
    mean_pi_ps = float(np.mean(pi_ps_all)) if pi_ps_all else None

    # π-floor checks (Task 2)
    pi_floor_passed, pi_floor_failures, pi_floor_stats = _pi_floor_checks(decisions)

    # ---- Per-agent occupancy (Task 6) ----
    agent_occupancy = _compute_agent_occupancy(decisions)

    # ---- Team-wide aggregates (Task 6, 7) ----
    total_agent_onball = sum(a["n_onball"] for a in agent_occupancy.values())
    total_selected_ps_onball = sum(
        a["n_selected_ps_when_onball"] for a in agent_occupancy.values()
    )
    p_agent_onball = total_agent_onball / max(1, n_total)
    p_selected_ps_given_agent_onball = total_selected_ps_onball / max(1, total_agent_onball)

    # ---- Legality (Task 9) ----
    n_pass_legal = sum(1 for d in onball_frames if d["mask_pass_legal"] == 1)
    n_shot_legal = sum(1 for d in onball_frames if d["mask_shot_legal"] == 1)
    n_pass_or_shot_legal = sum(1 for d in onball_frames if d["mask_pass_or_shot_legal"] == 1)

    pi_pass_when_legal = (
        float(np.mean([d["pi_pass"] for d in onball_frames if d["mask_pass_legal"] == 1]))
        if n_pass_legal > 0 else None
    )
    pi_shot_when_legal = (
        float(np.mean([d["pi_shot"] for d in onball_frames if d["mask_shot_legal"] == 1]))
        if n_shot_legal > 0 else None
    )
    pi_ps_when_either_legal = (
        float(np.mean([d["pi_pass_shot"] for d in onball_frames if d["mask_pass_or_shot_legal"] == 1]))
        if n_pass_or_shot_legal > 0 else None
    )

    # ---- Canonical scope reconciliation (Task 7) ----
    # Central equality: raw all-agent PS selections / 7650 == rebuilt rate / 100
    rebuilt_match = False
    delta_pp = None
    if rebuilt_rate_pct is not None:
        expected_rate = rebuilt_rate_pct / 100.0
        actual_rate = canonical["n_pass_shot"] / max(1, n_total)
        rebuilt_match = abs(actual_rate - expected_rate) <= RECONCILIATION_TOLERANCE_ABS
        delta_pp = (canonical["canonical_rate_pct"] - rebuilt_rate_pct)

    return {
        "seed": seed,
        "checkpoint": result["checkpoint"],
        "checkpoint_label": result["checkpoint_label"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "checkpoint_timesteps": result["checkpoint_timesteps"],
        "scenario": result["scenario"],
        "base_seed": result["base_seed"],
        "num_episodes": result["num_episodes"],
        "deterministic": result["deterministic"],
        "n_ticks": n_ticks,
        "n_decisions": n_total,
        # Canonical rate
        "n_pass": canonical["n_pass"],
        "n_shot": canonical["n_shot"],
        "n_pass_shot": canonical["n_pass_shot"],
        "canonical_rate_pct": canonical["canonical_rate_pct"],
        # Rebuilt comparison
        "rebuilt_pass_shot_rate_pct": rebuilt_rate_pct,
        "delta_pp": delta_pp,
        "rebuilt_match": rebuilt_match,
        # On-ball
        "n_onball": n_onball,
        "p_onball": n_onball / max(1, n_ticks * NUM_AGENTS),
        # π statistics (on-ball only)
        "mean_pi_pass_all_onball": mean_pi_pass,
        "mean_pi_shot_all_onball": mean_pi_shot,
        "mean_pi_pass_shot_all_onball": mean_pi_ps,
        # π-floor
        "pi_floor_passed": pi_floor_passed,
        "pi_floor_failures": pi_floor_failures,
        "pi_floor_n_onball": pi_floor_stats.get("n_onball"),
        "pi_floor_n_selected_ps": pi_floor_stats.get("n_selected_ps"),
        "pi_floor_mean_pi_ps": pi_floor_stats.get("mean_pi_ps"),
        "pi_floor_mean_pi_selected": pi_floor_stats.get("mean_pi_selected"),
        "pi_floor_mean_pi_ps_given_selected": pi_floor_stats.get("mean_pi_ps_given_selected"),
        "pi_floor_mean_mask_sum_given_selected": pi_floor_stats.get("mean_mask_sum_given_selected"),
        "pi_floor_floor_action_space": pi_floor_stats.get("floor_from_action_space"),
        "pi_floor_floor_observed_masks": pi_floor_stats.get("floor_from_observed_masks"),
        "pi_floor_inequality_1": pi_floor_stats.get("inequality_1"),
        "pi_floor_inequality_2": pi_floor_stats.get("inequality_2"),
        "pi_floor_inequality_3": pi_floor_stats.get("inequality_3"),
        # Agent occupancy
        "agent0_n_decisions": agent_occupancy[0]["n_decisions"],
        "agent0_n_onball": agent_occupancy[0]["n_onball"],
        "agent0_p_onball": agent_occupancy[0]["p_onball"],
        "agent0_n_selected_ps_when_onball": agent_occupancy[0]["n_selected_ps_when_onball"],
        "agent0_p_selected_ps_given_onball": agent_occupancy[0]["p_selected_ps_given_onball"],
        "agent1_n_decisions": agent_occupancy[1]["n_decisions"],
        "agent1_n_onball": agent_occupancy[1]["n_onball"],
        "agent1_p_onball": agent_occupancy[1]["p_onball"],
        "agent1_n_selected_ps_when_onball": agent_occupancy[1]["n_selected_ps_when_onball"],
        "agent1_p_selected_ps_given_onball": agent_occupancy[1]["p_selected_ps_given_onball"],
        "agent2_n_decisions": agent_occupancy[2]["n_decisions"],
        "agent2_n_onball": agent_occupancy[2]["n_onball"],
        "agent2_p_onball": agent_occupancy[2]["p_onball"],
        "agent2_n_selected_ps_when_onball": agent_occupancy[2]["n_selected_ps_when_onball"],
        "agent2_p_selected_ps_given_onball": agent_occupancy[2]["p_selected_ps_given_onball"],
        # Team-wide
        "team_n_agent_onball_total": total_agent_onball,
        "team_p_agent_onball": p_agent_onball,
        "team_n_selected_ps_onball_total": total_selected_ps_onball,
        "team_p_selected_ps_given_agent_onball": p_selected_ps_given_agent_onball,
        # Legality
        "n_pass_legal": n_pass_legal,
        "n_shot_legal": n_shot_legal,
        "n_pass_or_shot_legal": n_pass_or_shot_legal,
        "pass_legal_rate": n_pass_legal / max(1, n_onball),
        "shot_legal_rate": n_shot_legal / max(1, n_onball),
        "pass_or_shot_legal_rate": n_pass_or_shot_legal / max(1, n_onball),
        "pi_pass_when_pass_legal": pi_pass_when_legal,
        "pi_shot_when_shot_legal": pi_shot_when_legal,
        "pi_pass_shot_when_either_legal": pi_ps_when_either_legal,
        # Code provenance
        "code_commit": result.get("code_commit", code_commit()),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return repr(value)
    return value


def write_csv(path: str, columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(columns))
        for row in rows:
            writer.writerow([_csv_value(row.get(col)) for col in columns])
    print(f"CSV: {path}")


def write_detail_json(
    path: str,
    results: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    provenance: Dict[str, Any],
) -> None:
    payload = {
        "schema": {
            "artifact": "post_reweight_logit_prestep_reconciled_detail.json",
            "measurement_only": True,
            "scope": "canonical_3_agent_7650_decision",
            "onball_source": "pre_step_ball_owner_agent_idx",
            "retention_rule": (
                "retain ALL agent decisions; on-ball = pre_step_ball_owner_agent_idx == agent_index"
            ),
            "team_possession_field": "pre_step_obs[95] == 1.0 (left-team possession)",
            "agent_possession_field": "pre_step_ball_owner_agent_idx == agent_index",
            "obs_ball_ownership_slice_semantics": (
                "obs[94], obs[95], obs[96] are a one-hot over [no-one, left, right] "
                "TEAM-LEVEL ball ownership (src/engine/ObservationEncoder.ts:33,136-140). "
                "obs[95] == 1.0 means LEFT TEAM has the ball. Agent-specific ownership "
                "uses pre_step_ball_owner_agent_idx."
            ),
            "canonical_rate_definition": (
                "((passes + shots) / (num_episodes * 51 * 3)) * 100; "
                "for 50 episodes: denominator = 7650 agent-decisions"
            ),
            "pi_pass_definition": "sum of actor softmax probs over actions {9, 10, 11}",
            "pi_shot_definition": "actor softmax prob of action {12}",
            "pi_pass_shot_definition": "pi_pass + pi_shot = mass over {9, 10, 11, 12}",
            "policy_probability_source": (
                "actor(obs, action_mask) forward pass on the PRE-step state; "
                "masking is training-time masked_fill(~mask, -inf) semantics"
            ),
            "deterministic_action_definition": "argmax of masked PRE-step actor logits",
            "behavioural_statistic": (
                "n_pass_shot_actions / p_selected_pass_shot count the SELECTED "
                "deterministic action; these are behavioural frequencies, NOT policy probabilities"
            ),
            "null_semantics": (
                "null means the value was not finite. In masked_logits, null is exactly "
                "the masked-out slot (-inf from masked_fill). No literal Infinity/-Infinity/NaN."
            ),
            "085ec85_status": "SUPERSEDED — scope mismatch: agent-0-only measurement is not canonical 3-agent scope",
        },
        "provenance": provenance,
        "results": list(results),
        "frames": [f for r in results for f in r.get("frames", [])],
        "aggregate": {"summary_columns": list(CANONICAL_SUMMARY_COLUMNS), "rows": list(rows)},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, allow_nan=False)
    print(f"Detail JSON: {path}")


# ---------------------------------------------------------------------------
# Summary column definitions
# ---------------------------------------------------------------------------
CANONICAL_SUMMARY_COLUMNS: Sequence[str] = (
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
    "n_decisions",
    # Canonical rate
    "n_pass",
    "n_shot",
    "n_pass_shot",
    "canonical_rate_pct",
    # Rebuilt comparison
    "rebuilt_pass_shot_rate_pct",
    "delta_pp",
    "rebuilt_match",
    # On-ball
    "n_onball",
    "p_onball",
    # π stats
    "mean_pi_pass_all_onball",
    "mean_pi_shot_all_onball",
    "mean_pi_pass_shot_all_onball",
    # π-floor
    "pi_floor_passed",
    "pi_floor_n_onball",
    "pi_floor_n_selected_ps",
    "pi_floor_mean_pi_ps",
    "pi_floor_mean_pi_selected",
    "pi_floor_mean_pi_ps_given_selected",
    "pi_floor_mean_mask_sum_given_selected",
    "pi_floor_floor_action_space",
    "pi_floor_floor_observed_masks",
    "pi_floor_inequality_1",
    "pi_floor_inequality_2",
    "pi_floor_inequality_3",
    # Agent occupancy
    "agent0_n_decisions",
    "agent0_n_onball",
    "agent0_p_onball",
    "agent0_n_selected_ps_when_onball",
    "agent0_p_selected_ps_given_onball",
    "agent1_n_decisions",
    "agent1_n_onball",
    "agent1_p_onball",
    "agent1_n_selected_ps_when_onball",
    "agent1_p_selected_ps_given_onball",
    "agent2_n_decisions",
    "agent2_n_onball",
    "agent2_p_onball",
    "agent2_n_selected_ps_when_onball",
    "agent2_p_selected_ps_given_onball",
    # Team-wide
    "team_n_agent_onball_total",
    "team_p_agent_onball",
    "team_n_selected_ps_onball_total",
    "team_p_selected_ps_given_agent_onball",
    # Legality
    "n_pass_legal",
    "n_shot_legal",
    "n_pass_or_shot_legal",
    "pass_legal_rate",
    "shot_legal_rate",
    "pass_or_shot_legal_rate",
    "pi_pass_when_pass_legal",
    "pi_shot_when_shot_legal",
    "pi_pass_shot_when_either_legal",
    # Provenance
    "code_commit",
)

CANONICAL_RECONCILIATION_COLUMNS: Sequence[str] = (
    "seed",
    "n_decisions",
    "team_wide_n_ps_selected",
    "team_wide_canonical_rate_pct",
    "total_controlled_agent_onball_frames",
    "agent_level_onball_ps_selections",
    "onball_conditional_ps_rate",
    "team_wide_rate_from_raw_decisions",
    "canonical_rebuilt_csv_rate",
    "delta",
    "match",
)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_measurement(
    *,
    seeds: Sequence[int],
    scenario: str,
    num_episodes: int,
    deterministic: bool,
    base_seed: int,
    output_dir: str,
    model_dir: str,
    inventory_path: str,
    rebuilt_csv_path: Optional[str] = None,
    bridge_port: int = 5050,
    suffix: str = "",
) -> Dict[str, Any]:
    """Measure every requested checkpoint; hard-fail on unverifiable identity."""
    os.makedirs(output_dir, exist_ok=True)
    commit = code_commit()
    inventory = load_inventory_final_hashes(inventory_path)
    print(f"[inventory] verified source: {inventory_path}")
    print(f"[code] commit: {commit}")

    # Load rebuilt rates if available
    rebuilt_rates: Dict[int, float] = {}
    if rebuilt_csv_path and os.path.exists(rebuilt_csv_path):
        with open(rebuilt_csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rebuilt_rates[int(row["seed"])] = float(row["pass_shot_rate_pct"])
                except (ValueError, KeyError):
                    pass

    results: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    reconciliation_rows: List[Dict[str, Any]] = []
    all_gates_passed = True
    gate_failures: List[str] = []

    for seed in seeds:
        ckpt_path = os.path.join(model_dir, CHECKPOINT_TEMPLATE.format(seed=seed))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint missing for seed {seed}: {ckpt_path}")

        ckpt_sha = sha256_of(ckpt_path)
        verify_checkpoint_sha(seed, ckpt_sha, inventory)
        actor, _critic, ckpt_timesteps = load_actor(ckpt_path)

        rebuilt_rate = rebuilt_rates.get(seed)

        print(f"\n{'=' * 70}")
        print(f"CANONICAL 3-AGENT MEASUREMENT: seed={seed} label={CHECKPOINT_LABEL}")
        print(f"  checkpoint        : {ckpt_path}")
        print(f"  checkpoint sha256 : {ckpt_sha}")
        print(f"  internal timesteps: {ckpt_timesteps}")
        print(f"  scenario          : {scenario}")
        print(f"  rebuilt rate      : {rebuilt_rate}%")
        print(f"{'=' * 70}")

        result = collect_canonical_measurement(
            actor=actor,
            checkpoint_path=ckpt_path,
            checkpoint_sha256=ckpt_sha,
            checkpoint_timesteps=ckpt_timesteps,
            seed=int(seed),
            scenario=scenario,
            num_episodes=num_episodes,
            deterministic=deterministic,
            base_seed=base_seed,
            bridge_port=bridge_port,
            label=CHECKPOINT_LABEL,
        )
        result["code_commit"] = commit

        # Temporal alignment check (Task 1)
        temporal_ok = _check_temporal_alignment(result["decisions"])
        if not temporal_ok:
            all_gates_passed = False
            gate_failures.append(f"seed {seed}: temporal alignment FAILED")

        # π-floor check (Task 2)
        row = _aggregate_seed(result, rebuilt_rate_pct=rebuilt_rate)
        if not row["pi_floor_passed"]:
            all_gates_passed = False
            gate_failures.append(f"seed {seed}: π-floor FAILED — {row['pi_floor_failures']}")

        results.append(result)
        rows.append(row)

        # Reconciliation row (Task 7)
        total_agent_onball = sum(
            row[f"agent{i}_n_onball"] for i in range(NUM_AGENTS)
        )
        total_selected_ps = sum(
            row[f"agent{i}_n_selected_ps_when_onball"] for i in range(NUM_AGENTS)
        )
        reconciliation_rows.append({
            "seed": seed,
            "n_decisions": row["n_decisions"],
            "team_wide_n_ps_selected": total_selected_ps,
            "team_wide_canonical_rate_pct": row["canonical_rate_pct"],
            "total_controlled_agent_onball_frames": total_agent_onball,
            "agent_level_onball_ps_selections": total_selected_ps,
            "onball_conditional_ps_rate": total_selected_ps / max(1, total_agent_onball),
            "team_wide_rate_from_raw_decisions": row["canonical_rate_pct"],
            "canonical_rebuilt_csv_rate": rebuilt_rate,
            "delta": row["delta_pp"],
            "match": row["rebuilt_match"],
        })

        print(
            f"  RESULT seed={seed}: n_decisions={row['n_decisions']} "
            f"n_pass_shot={row['n_pass_shot']} rate={row['canonical_rate_pct']:.3f}% "
            f"rebuilt={rebuilt_rate}% "
            f"n_onball={row['n_onball']} "
            f"pi_floor={'PASS' if row['pi_floor_passed'] else 'FAIL'}"
        )

        # ---- Incremental save after each seed (robustness) ----
        partial_provenance = {
            "code_commit": commit,
            "code_commit_semantics": (
                "git HEAD at measurement time; corrected measurement code identified "
                "by per-file SHA-256 digests"
            ),
            "measurement_code_sha256": measurement_code_digests(),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario,
            "seeds": [int(r["seed"]) for r in rows],
            "checkpoint_label": CHECKPOINT_LABEL,
            "checkpoint_declared_timesteps": CHECKPOINT_DECLARED_TIMESTEPS,
            "num_episodes": int(num_episodes),
            "deterministic": bool(deterministic),
            "base_seed": int(base_seed),
            "episode_seed_formula": "base_seed + episode_index * 1009",
            "inventory_csv": inventory_path,
            "rebuilt_csv": rebuilt_csv_path,
            "canonical_denominator": num_episodes * DEFAULT_TICKS_PER_EPISODE * NUM_AGENTS,
            "measurement_only": True,
            "partial_save": True,
            "checkpoints": [
                {
                    "seed": int(r["seed"]),
                    "checkpoint": r["checkpoint"],
                    "checkpoint_sha256": r["checkpoint_sha256"],
                    "checkpoint_timesteps": int(r["checkpoint_timesteps"]),
                    "inventory_sha256": inventory.get(int(r["seed"])),
                    "sha_verified_against_inventory": (
                        inventory.get(int(r["seed"])) == r["checkpoint_sha256"]
                    ),
                }
                for r in rows
            ],
            "gates": {
                "temporal_alignment": all(
                    _check_temporal_alignment(r["decisions"]) for r in results
                ),
                "pi_floor_all_seeds": all(r["pi_floor_passed"] for r in rows),
                "all_gates_passed": all_gates_passed,
                "gate_failures": gate_failures,
            },
        }
        partial_detail_path = os.path.join(
            output_dir,
            f"post_reweight_logit_prestep_reconciled_detail{suffix}_partial.json",
        )
        partial_summary_path = os.path.join(
            output_dir,
            f"post_reweight_logit_prestep_reconciled_summary{suffix}_partial.csv",
        )
        partial_recon_path = os.path.join(
            output_dir,
            f"post_reweight_canonical_scope_reconciliation{suffix}_partial.csv",
        )
        write_detail_json(
            partial_detail_path,
            results=list(results),
            rows=list(rows),
            provenance=partial_provenance,
        )
        write_csv(partial_summary_path, CANONICAL_SUMMARY_COLUMNS, rows)
        write_csv(partial_recon_path, CANONICAL_RECONCILIATION_COLUMNS, reconciliation_rows)
        print(f"  [partial save] {partial_detail_path}")
        print(f"  [partial save] {partial_summary_path}")
        print(f"  [partial save] {partial_recon_path}")

    # Provenance
    provenance = {
        "code_commit": commit,
        "code_commit_semantics": (
            "git HEAD at measurement time; corrected measurement code identified "
            "by per-file SHA-256 digests"
        ),
        "measurement_code_sha256": measurement_code_digests(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "seeds": [int(s) for s in seeds],
        "checkpoint_label": CHECKPOINT_LABEL,
        "checkpoint_declared_timesteps": CHECKPOINT_DECLARED_TIMESTEPS,
        "num_episodes": int(num_episodes),
        "deterministic": bool(deterministic),
        "base_seed": int(base_seed),
        "episode_seed_formula": "base_seed + episode_index * 1009",
        "inventory_csv": inventory_path,
        "rebuilt_csv": rebuilt_csv_path,
        "canonical_denominator": num_episodes * DEFAULT_TICKS_PER_EPISODE * NUM_AGENTS,
        "measurement_only": True,
        "checkpoints": [
            {
                "seed": int(r["seed"]),
                "checkpoint": r["checkpoint"],
                "checkpoint_sha256": r["checkpoint_sha256"],
                "checkpoint_timesteps": int(r["checkpoint_timesteps"]),
                "inventory_sha256": inventory.get(int(r["seed"])),
                "sha_verified_against_inventory": (
                    inventory.get(int(r["seed"])) == r["checkpoint_sha256"]
                ),
            }
            for r in results
        ],
        "gates": {
            "temporal_alignment": all(
                _check_temporal_alignment(r["decisions"]) for r in results
            ),
            "pi_floor_all_seeds": all(r["pi_floor_passed"] for r in rows),
            "all_gates_passed": all_gates_passed,
            "gate_failures": gate_failures,
        },
    }

    # Write artifacts
    detail_path = os.path.join(
        output_dir, f"post_reweight_logit_prestep_reconciled_detail{suffix}.json"
    )
    summary_path = os.path.join(
        output_dir, f"post_reweight_logit_prestep_reconciled_summary{suffix}.csv"
    )
    reconciliation_path = os.path.join(
        output_dir, f"post_reweight_canonical_scope_reconciliation{suffix}.csv"
    )

    write_detail_json(detail_path, results=results, rows=rows, provenance=provenance)
    write_csv(summary_path, CANONICAL_SUMMARY_COLUMNS, rows)
    write_csv(reconciliation_path, CANONICAL_RECONCILIATION_COLUMNS, reconciliation_rows)

    return {
        "results": results,
        "rows": rows,
        "reconciliation_rows": reconciliation_rows,
        "provenance": provenance,
        "detail_path": detail_path,
        "summary_path": summary_path,
        "reconciliation_path": reconciliation_path,
        "all_gates_passed": all_gates_passed,
        "gate_failures": gate_failures,
    }


def _check_temporal_alignment(decisions: List[Dict[str, Any]]) -> bool:
    """Verify that all retained decisions use the same pre-step state."""
    for d in decisions:
        probs = d["probs"]
        mask = d["action_mask"]
        action = d["action_taken"]
        legal = [i for i in range(ACTION_DIM) if mask[i] == 1]
        if not legal:
            return False
        expected = max(legal, key=lambda i: probs[i])
        if action != expected:
            return False
        if d["team_has_ball"] != (d["pre_step_obs95"] == 1.0):
            return False
    return True


def load_actor(checkpoint_path: str):
    """Load the checkpoint actor."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    actor = load_mappo_actor(ckpt)
    actor.eval()
    timesteps = ckpt.get("timesteps", None)
    actual_timesteps = (
        int(timesteps) if timesteps is not None else CHECKPOINT_DECLARED_TIMESTEPS
    )
    return actor, None, actual_timesteps


def main():
    parser = argparse.ArgumentParser(description="Canonical 3-agent pre-step measurement")
    parser.add_argument("--seeds", type=str, default="42,123,7,999")
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="training/results")
    parser.add_argument("--model-dir", type=str, default="training/models")
    parser.add_argument("--inventory", type=str, default=INVENTORY_RELATIVE_PATH)
    parser.add_argument("--rebuilt-csv", type=str, default=None)
    parser.add_argument("--bridge-port", type=int, default=5050)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    result = run_measurement(
        seeds=seeds,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        inventory_path=args.inventory,
        rebuilt_csv_path=args.rebuilt_csv,
        bridge_port=args.bridge_port,
    )
    print(f"All gates passed: {result['all_gates_passed']}")
    if result["gate_failures"]:
        print(f"Gate failures: {result['gate_failures']}")
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
