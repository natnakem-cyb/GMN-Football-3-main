"""
GMN-Football-3 — Corrected PRE-STEP On-Ball Measurement (MEASUREMENT ONLY).

Purpose
-------
Produce an audit-grade, temporally-correct measurement of agent 0 on-ball
occupancy and PASS/SHOT policy quantities for the four verified 50k
actor-reweight checkpoints.

Defect being corrected
----------------------
``training/eval_post_reweight_logits.py`` evaluated ``local_obs``,
``mask_matrix`` and the actor logits from the PRE-step state but decided frame
retention from the POST-step individual ball owner::

    ball_owner_agent_idx = getattr(env, "_last_ball_owner_agent_idx", 255)
    agent0_has_ball = (ball_owner_agent_idx == 0)      # post-step -> wrong

The state/selection mismatch is fixed here: retention and *every* reported
quantity come from the SAME pre-step ``(obs, mask)`` pair, and retention is
exactly::

    agent0_obs[95] == 1.0        # PRE-step observation of agent 0

Post-step ownership is retained only as a diagnostic field
(``post_step_ball_owner_agent_idx``) and never used for retention.

Outputs (default ``--output-dir training/results``)
--------------------------------------------------
* ``post_reweight_logit_prestep_detail.json``  — per-frame detail (JSON-safe)
* ``post_reweight_logit_prestep_summary.csv``  — per-seed aggregates + provenance
* ``onball_occupancy_prestep_summary.csv``     — occupancy table (Task 6)

Non-goals (hard prohibitions for this script): no training, no reward/GAE/mask/
actor/critic/optimiser/entropy/horizon changes, no environment or action-taxonomy
changes, no checkpoint-weight changes.

Known checkpoint identity semantics: the human label is ``50k`` while the
internal checkpoint timestep is whatever the checkpoint stores (49920 for these
files). Both are reported separately and never conflated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv  # noqa: E402
from training.mappo_networks import CentralizedCritic, SharedActor  # noqa: E402
from training.mappo_rollout import _mask_matrix, unwrap_masks, unwrap_obs  # noqa: E402
from training.prestep_onball import (  # noqa: E402
    OCCUPANCY_COLUMNS,
    SUMMARY_COLUMNS,
    build_prestep_onball_frame,
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

# Human label vs internal checkpoint timestep are reported separately.
CHECKPOINT_LABEL = "50k"
CHECKPOINT_DECLARED_TIMESTEPS = 50000
CHECKPOINT_TEMPLATE = (
    "mappo_academy_3_vs_1_with_keeper_onball_seed{seed}_actorreweight.pt"
)

DEFAULT_SCENARIO = "academy_3_vs_1_with_keeper_onball"
DEFAULT_SEEDS = (42, 123, 7, 999)
DEFAULT_NUM_EPISODES = 50
DEFAULT_BASE_SEED = 700000
DEFAULT_BRIDGE_PORT = 5050

INVENTORY_RELATIVE_PATH = os.path.join(
    "training", "results", "retest_checkpoint_inventory.csv"
)

DETAIL_SCHEMA: Dict[str, Any] = {
    "artifact": "post_reweight_logit_prestep_detail.json",
    "measurement_only": True,
    "onball_source": "obs[95]_pre_step",
    "retention_rule": "retain a decision tick iff pre-step agent-0 obs[95] == 1.0",
    "retention_is_pre_step": True,
    "post_step_ball_owner_agent_idx": "DIAGNOSTIC ONLY - never used for retention",
    "obs_ball_ownership_slice_semantics": (
        "obs[94], obs[95], obs[96] are a one-hot over [no-one, left, right] "
        "TEAM-LEVEL ball ownership (src/engine/ObservationEncoder.ts:33,136-140). "
        "obs[95] == 1.0 therefore means the LEFT TEAM has the ball (agent 0, 1 or "
        "2 may be the individual owner); it is not by itself an agent-0 possession "
        "flag. Per-agent possession is instead exposed by the action mask "
        "(mask[9..12] == 1 iff the acting player individually has possession)."
    ),
    "pi_pass_definition": "sum of actor softmax probs over actions {9, 10, 11}",
    "pi_shot_definition": "actor softmax prob of action {12}",
    "pi_pass_shot_definition": "pi_pass + pi_shot = mass over {9, 10, 11, 12}",
    "policy_probability_source": (
        "actor(obs, action_mask) forward pass on the PRE-step state; masking is "
        "the training-time masked_fill(~mask, -inf) semantics"
    ),
    "deterministic_action_definition": (
        "argmax of the masked PRE-step actor logits (same forward pass)"
    ),
    "behavioural_statistic": (
        "n_pass_shot_actions / p_selected_pass_shot_given_onball count the "
        "SELECTED deterministic action in {9, 10, 11, 12}; these are behavioural "
        "frequencies and are NOT policy probabilities"
    ),
    "populations": {
        "A_all_onball": "all retained pre-step on-ball frames",
        "B_pass_legal": "A and mask_pass_legal == 1",
        "C_shot_legal": "A and mask_shot_legal == 1",
        "D_pass_or_shot_legal": "A and mask_pass_or_shot_legal == 1",
    },
    "null_semantics": (
        "null means the value was not finite OR the population was empty. In "
        "masked_logits, null is exactly the masked-out slot (-inf from "
        "masked_fill). No literal Infinity/-Infinity/NaN is written."
    ),
}


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
    """HEAD commit of the repository the measurement ran from."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


# Files that constitute the corrected measurement code. Their digests are
# recorded in the artifact so the measurement stays auditable even when the
# working tree is not yet committed.
MEASUREMENT_CODE_FILES = (
    "training/prestep_onball.py",
    "training/eval_post_reweight_logits_prestep.py",
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


def load_actor(checkpoint_path: str):
    """Load the checkpoint actor (deterministic action selection uses it)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    obs_dim = int(ckpt.get("obs_dim", OBS_DIM))
    action_dim = int(ckpt.get("action_dim", ACTION_DIM))
    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    critic = None
    if "critic" in ckpt:
        critic = CentralizedCritic(obs_dim=obs_dim, hidden=64)
        critic.load_state_dict(ckpt["critic"])
        critic.eval()
    timesteps = ckpt.get("timesteps", None)
    actual_timesteps = (
        int(timesteps) if timesteps is not None else CHECKPOINT_DECLARED_TIMESTEPS
    )
    return actor, critic, actual_timesteps


def load_inventory_final_hashes(inventory_path: str) -> Dict[int, str]:
    """Map seed -> SHA-256 for the ``trajectory_step == 'final'`` inventory rows."""
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


def verify_checkpoint_sha(
    seed: int, computed_sha: str, inventory: Dict[int, str]
) -> None:
    """Hard-fail when a checkpoint does not match the committed inventory."""
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
# Episode collector (corrected, pre-step aligned)
# ---------------------------------------------------------------------------
def collect_measurement(
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
    bridge_port: int = DEFAULT_BRIDGE_PORT,
    label: str = CHECKPOINT_LABEL,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Roll out deterministic episodes, retaining PRE-STEP on-ball frames.

    The pre-step state is captured at the TOP of every decision tick, before
    ``env.step()``: ``local_obs``, ``mask_matrix``, ``agent0_obs``,
    ``agent0_mask`` and ``pre_step_onball``. Retention, legality, logits,
    probabilities and the deterministic action all come from that state.
    """
    env = GMNMultiAgentEnv(
        scenario=scenario, auto_start_bridge=True, port=bridge_port
    )
    controllable_agents = list(env.possible_agents)

    episodes_data: List[Dict[str, Any]] = []
    frames: List[Dict[str, Any]] = []
    n_ticks_total = 0

    try:
        for ep in range(num_episodes):
            ep_seed = base_seed + ep * 1009
            obs_dict, _ = env.reset(seed=ep_seed)
            current_ep_masks = unwrap_masks(obs_dict)
            obs_dict = unwrap_obs(obs_dict)

            current_agents = list(env.agents if env.agents else controllable_agents)
            if not current_agents:
                continue

            ep_frames = 0
            ep_reward = 0.0
            ep_ticks = 0
            ep_post_step_owner0 = 0
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
                agent0_obs = local_obs[0]
                agent0_mask = mask_matrix[0]
                pre_step_onball = bool(is_prestep_onball(agent0_obs))

                with torch.no_grad():
                    obs_tensor = torch.from_numpy(local_obs).float()
                    mask_tensor = torch.tensor(mask_matrix, dtype=torch.bool)
                    dist = actor(obs_tensor, mask_tensor)
                    if deterministic:
                        actions = dist.logits.argmax(dim=-1)
                    else:
                        actions = dist.sample()
                    actions_np = actions.cpu().numpy()

                action_dict = {
                    a: int(actions_np[i]) for i, a in enumerate(current_agents)
                }

                # ---- Step (post-step ownership is diagnostic only) ----
                obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
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
                post_step_ball_owner_agent_idx = int(
                    getattr(env, "_last_ball_owner_agent_idx", 255)
                )

                frame = build_prestep_onball_frame(
                    actor,
                    agent0_obs,
                    agent0_mask,
                    post_step_ball_owner_agent_idx=post_step_ball_owner_agent_idx,
                    action_names=ACTION_NAMES,
                    extra={
                        "seed": int(seed),
                        "episode": int(ep),
                        "tick": int(tick_idx),
                        "ep_seed": int(ep_seed),
                        "checkpoint": checkpoint_path,
                        "checkpoint_label": label,
                        "checkpoint_sha256": checkpoint_sha256,
                        "checkpoint_timesteps": int(checkpoint_timesteps),
                        "scenario": scenario,
                        "deterministic": bool(deterministic),
                        "event_code": event_code,
                        "reward": shared_rew,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "done": bool(done),
                    },
                )
                if frame is not None:
                    # The applied action must be the PRE-step argmax (alignment).
                    if int(frame["action_taken"]) != int(actions_np[0]):
                        raise RuntimeError(
                            "Temporal-alignment violation: frame action_taken="
                            f"{int(frame['action_taken'])} != applied action "
                            f"{int(actions_np[0])} at seed={seed} ep={ep} "
                            f"tick={tick_idx}"
                        )
                    if not pre_step_onball:
                        raise RuntimeError(
                            "Temporal-alignment violation: frame retained while "
                            "pre-step obs[95] != 1.0"
                        )
                    frames.append(frame)
                    ep_frames += 1
                    if post_step_ball_owner_agent_idx == 0:
                        ep_post_step_owner0 += 1

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
                    "n_onball_prestep": int(ep_frames),
                    "n_onball_and_post_step_owner0": int(ep_post_step_owner0),
                    "episode_reward": float(ep_reward),
                }
            )

            if verbose and (ep + 1) % 10 == 0:
                print(
                    f"  [prestep seed={seed}] Ep {ep + 1:3d}/{num_episodes} | "
                    f"ticks={ep_ticks} onball_prestep={ep_frames} | "
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
        "n_onball": len(frames),
        "episodes": episodes_data,
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _csv_value(value: Any) -> Any:
    """None -> empty field (documented); bool -> 1/0; floats -> repr (round-trip)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
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
    *,
    results: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    provenance: Dict[str, Any],
) -> None:
    """Write the machine-readable detail artifact (valid JSON, no NaN/Infinity)."""
    payload = {
        "schema": DETAIL_SCHEMA,
        "provenance": provenance,
        "results": list(results),
        "frames": [f for r in results for f in r["frames"]],
        "aggregate": {"summary_columns": list(SUMMARY_COLUMNS), "rows": list(rows)},
    }
    # allow_nan=False: json_safe must already have removed non-finite values.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, allow_nan=False)
    print(f"Detail JSON: {path}")


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
    bridge_port: int = DEFAULT_BRIDGE_PORT,
    suffix: str = "",
) -> Dict[str, Any]:
    """Measure every requested checkpoint; hard-fail on unverifiable identity."""
    os.makedirs(output_dir, exist_ok=True)
    commit = code_commit()
    inventory = load_inventory_final_hashes(inventory_path)
    print(f"[inventory] verified source: {inventory_path}")
    print(f"[code] commit: {commit}")

    results: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        ckpt_path = os.path.join(model_dir, CHECKPOINT_TEMPLATE.format(seed=seed))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint missing for seed {seed}: {ckpt_path}")

        ckpt_sha = sha256_of(ckpt_path)
        verify_checkpoint_sha(seed, ckpt_sha, inventory)
        actor, _critic, ckpt_timesteps = load_actor(ckpt_path)

        print(f"\n{'=' * 70}")
        print(f"PRE-STEP ON-BALL MEASUREMENT: seed={seed} label={CHECKPOINT_LABEL}")
        print(f"  checkpoint        : {ckpt_path}")
        print(f"  checkpoint sha256 : {ckpt_sha}")
        print(f"  internal timesteps: {ckpt_timesteps} (human label {CHECKPOINT_LABEL})")
        print(f"  scenario          : {scenario}")
        print(
            f"  episodes          : {num_episodes} deterministic, base_seed={base_seed}"
        )
        print(f"{'=' * 70}")

        result = collect_measurement(
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
        row = summarize_prestep_frames(
            result["frames"],
            n_ticks=result["n_ticks"],
            seed=int(seed),
            checkpoint_label=CHECKPOINT_LABEL,
            checkpoint_timesteps=ckpt_timesteps,
            checkpoint=ckpt_path,
            checkpoint_sha256=ckpt_sha,
            scenario=scenario,
            base_seed=base_seed,
            num_episodes=num_episodes,
            deterministic=deterministic,
            code_commit=commit,
        )
        results.append(result)
        rows.append(row)

        print(
            f"  RESULT seed={seed}: n_ticks={row['n_ticks']} n_onball={row['n_onball']} "
            f"P(on-ball)={row['p_onball']} n_pass_legal={row['n_pass_legal']} "
            f"n_shot_legal={row['n_shot_legal']} "
            f"n_pass_shot_actions={row['n_pass_shot_actions']}",
            flush=True,
        )

    provenance = {
        "code_commit": commit,
        "code_commit_semantics": (
            "git HEAD at measurement time; the corrected measurement code is "
            "additionally identified below by per-file SHA-256 digests"
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
        "n_ticks_total": int(sum(r["n_ticks"] for r in results)),
        "n_onball_total": int(sum(len(r["frames"]) for r in results)),
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
    }

    detail_path = os.path.join(
        output_dir, f"post_reweight_logit_prestep_detail{suffix}.json"
    )
    summary_path = os.path.join(
        output_dir, f"post_reweight_logit_prestep_summary{suffix}.csv"
    )
    occupancy_path = os.path.join(
        output_dir, f"onball_occupancy_prestep_summary{suffix}.csv"
    )

    write_detail_json(detail_path, results=results, rows=rows, provenance=provenance)
    write_csv(summary_path, SUMMARY_COLUMNS, rows)
    write_csv(occupancy_path, OCCUPANCY_COLUMNS, rows)

    return {
        "results": results,
        "rows": rows,
        "provenance": provenance,
        "detail_path": detail_path,
        "summary_path": summary_path,
        "occupancy_path": occupancy_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Corrected PRE-STEP obs[95] on-ball measurement (measurement only)"
        )
    )
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Deterministic argmax evaluation (default, canonical protocol).",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample actions instead (NOT the canonical protocol).",
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--bridge-port", type=int, default=DEFAULT_BRIDGE_PORT)
    parser.add_argument("--output-dir", type=str, default="training/results")
    parser.add_argument("--model-dir", type=str, default=os.path.join("training", "models"))
    parser.add_argument("--inventory", type=str, default=INVENTORY_RELATIVE_PATH)
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Appended to every artifact name (e.g. '_smoke' for a throwaway run).",
    )
    args = parser.parse_args(argv)

    run_measurement(
        seeds=args.seeds,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        inventory_path=args.inventory,
        bridge_port=args.bridge_port,
        suffix=args.output_suffix,
    )
    print("\nPre-step on-ball measurement complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
