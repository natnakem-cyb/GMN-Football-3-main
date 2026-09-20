"""
GMN-Football-3 — Post-Reweight Logit Snapshot

Measurement-only module. No training, no reward/GAE/mask/network/entropy changes.
Produces:
  - training/results/post_reweight_logit_summary.csv
  - training/results/post_reweight_logit_detail.json (optional, local/LFS if large)
  - training/results/POST_REWEIGHT_LOGIT_SNAPSHOT.md

Checkpoints: verified retest checkpoints from retest_checkpoint_inventory.csv
  - Phase 1 end (15k): mappo_academy_3_vs_1_with_keeper_onball_seed{N}_actorreweight_15000.pt
  - Phase 2 final (50k): mappo_academy_3_vs_1_with_keeper_onball_seed{N}_actorreweight.pt
  Seeds: 42, 123, 7, 999

Arms:
  - ONBALL-pi: pure policy, deterministic, 50 episodes per checkpoint
  - Frame filter: on-ball frames where PASS/SHOT are legal (mask)

Reuses patterns from eval_actor_action_forensics.py; no training changes.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor, CentralizedCritic
from training.mappo_rollout import unwrap_obs, unwrap_masks, _mask_matrix

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
MOVE_ACTION_IDS = frozenset(range(9))          # 0-8
PASS_ACTION_IDS = frozenset((9, 10, 11))       # LONG_PASS, HIGH_PASS, SHORT_PASS
SHOT_ACTION_IDS = frozenset((12,))             # SHOT
TACKLE_ACTION_ID = 16
DRIBBLE_ACTION_ID = 17
RELEASE_DRIBBLE_ACTION_ID = 18

OBS_DIM = 127
ACTION_DIM = 19
OBS_BALL_OWNERSHIP_SLICE = slice(94, 97)  # [no-one, left, right]

# Verified retest checkpoints (from retest_checkpoint_inventory.csv)
CHECKPOINT_SPECS = [
    # (label, filename_template, timesteps)
    ("actorreweight_15000", "mappo_academy_3_vs_1_with_keeper_onball_seed{seed}_actorreweight_15000.pt", 15000),
    ("actorreweight_final", "mappo_academy_3_vs_1_with_keeper_onball_seed{seed}_actorreweight.pt", 50000),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    obs_dim = ckpt.get("obs_dim", OBS_DIM)
    action_dim = ckpt.get("action_dim", ACTION_DIM)
    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    critic = None
    if "critic" in ckpt:
        critic = CentralizedCritic(obs_dim=obs_dim, hidden=64)
        critic.load_state_dict(ckpt["critic"])
        critic.eval()
    timesteps = ckpt.get("timesteps", None)
    return actor, critic, obs_dim, action_dim, timesteps


def _is_on_ball(obs_vec: np.ndarray) -> bool:
    """Return True if the left team ownership slot is active."""
    if obs_vec is None or len(obs_vec) < 97:
        return False
    return bool(obs_vec[95] == 1.0)


def _logits_and_probs(actor: SharedActor, obs_vec: np.ndarray, mask_vec: np.ndarray):
    """Return raw logits, masked logits, probs, entropy, and top-k for agent 0."""
    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0)  # (1, obs_dim)
        mask_tensor = torch.tensor(mask_vec, dtype=torch.bool).unsqueeze(0)  # (1, action_dim)

        # Raw logits (pre-mask) from the actor's underlying MLP
        raw_logits = actor.net(obs_tensor).squeeze(0).cpu().numpy()  # (action_dim,)

        # Apply mask exactly as training does
        masked_logits = raw_logits.copy()
        mask_np = mask_tensor.squeeze(0).cpu().numpy()
        masked_logits[~mask_np] = float("-inf")

        # Softmax probabilities on masked logits
        logits_tensor = torch.tensor(masked_logits, dtype=torch.float32)
        probs = torch.softmax(logits_tensor, dim=-1).numpy()

        # Entropy of the masked distribution
        entropy = Categorical(logits=logits_tensor).entropy().item()

        # Top-1 and top-3
        topk = torch.topk(logits_tensor, k=min(3, ACTION_DIM))
        top1_idx = int(topk.indices[0].item())
        top1_prob = float(probs[top1_idx])
        top3 = []
        for i in range(min(3, ACTION_DIM)):
            idx = int(topk.indices[i].item())
            top3.append({
                "action": idx,
                "name": ACTION_NAMES[idx] if idx < len(ACTION_NAMES) else str(idx),
                "logit": float(topk.values[i].item()),
                "prob": float(probs[idx]),
            })

    return raw_logits, masked_logits, probs, entropy, top1_idx, top1_prob, top3


def _action_group_prob(probs: np.ndarray, action_ids) -> float:
    return float(probs[list(action_ids)].sum())


def _best_logit_in_group(logits: np.ndarray, action_ids) -> float:
    return float(logits[list(action_ids)].max())


# ---------------------------------------------------------------------------
# Episode collector
# ---------------------------------------------------------------------------
def collect_forensics_episodes(
    actor: SharedActor,
    critic: CentralizedCritic,
    checkpoint_path: str,
    checkpoint_sha256: str,
    checkpoint_timesteps: int,
    seed: int,
    scenario: str,
    num_episodes: int,
    deterministic: bool,
    base_seed: int,
) -> Dict[str, Any]:
    env = GMNMultiAgentEnv(scenario=scenario, auto_start_bridge=True, port=5050)
    controllable_agents = list(env.possible_agents)

    episodes_data: List[Dict[str, Any]] = []
    all_frames: List[Dict[str, Any]] = []

    try:
        for ep in range(num_episodes):
            ep_seed = base_seed + ep * 1009
            obs_dict, _ = env.reset(seed=ep_seed)
            current_ep_masks = unwrap_masks(obs_dict)
            obs_dict = unwrap_obs(obs_dict)

            current_agents = list(env.agents if env.agents else controllable_agents)
            if not current_agents:
                continue

            ep_frames: List[Dict[str, Any]] = []
            ep_reward = 0.0
            ep_length = 0
            tick_idx = 0

            while True:
                current_agents = list(env.agents if env.agents else controllable_agents)
                if not current_agents:
                    break

                local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
                mask_matrix = _mask_matrix(current_ep_masks, current_agents)

                # Policy action selection
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(local_obs).float()
                    mask_tensor = torch.tensor(mask_matrix, dtype=torch.bool)
                    dist = actor(obs_tensor, mask_tensor)
                    if deterministic:
                        actions = dist.logits.argmax(dim=-1)
                    else:
                        actions = dist.sample()
                    actions_np = actions.cpu().numpy()

                action_dict = {a: int(actions_np[i]) for i, a in enumerate(current_agents)}

                # Step environment
                obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
                current_ep_masks = unwrap_masks(obs_dict)
                obs_dict = unwrap_obs(obs_dict)

                terminated = any(terms.values()) if terms else False
                truncated = any(truncs.values()) if truncs else False
                done = terminated or truncated or not env.agents

                shared_rew = float(rewards[current_agents[0]]) if current_agents and current_agents[0] in rewards else 0.0
                ep_reward += shared_rew
                ep_length += 1

                event_code = getattr(env, "_last_frame_event_code", None)
                ball_owner_agent_idx = getattr(env, "_last_ball_owner_agent_idx", 255)

                # Focus on agent 0 (first controllable agent)
                agent0_has_ball = (ball_owner_agent_idx == 0)
                agent0_obs = local_obs[0] if len(local_obs) > 0 else None
                agent0_mask = mask_matrix[0] if len(mask_matrix) > 0 else None

                if agent0_has_ball and agent0_obs is not None and agent0_mask is not None:
                    raw_logits, masked_logits, probs, entropy, top1_idx, top1_prob, top3 = _logits_and_probs(
                        actor, agent0_obs, agent0_mask
                    )

                    pass_legal = bool(agent0_mask[9] or agent0_mask[10] or agent0_mask[11])
                    shot_legal = bool(agent0_mask[12])
                    tackle_legal = bool(agent0_mask[16])
                    dribble_legal = bool(agent0_mask[17])

                    move_best = _best_logit_in_group(raw_logits, MOVE_ACTION_IDS)
                    pass_best = _best_logit_in_group(raw_logits, PASS_ACTION_IDS)
                    shot_logit = float(raw_logits[12])

                    frame = {
                        "seed": seed,
                        "episode": ep,
                        "tick": tick_idx,
                        "checkpoint": checkpoint_path,
                        "checkpoint_sha256": checkpoint_sha256,
                        "checkpoint_timesteps": checkpoint_timesteps,
                        "scenario": scenario,
                        "deterministic": deterministic,
                        "ep_seed": ep_seed,
                        "source": "pure_pi",
                        "raw_logits": raw_logits.tolist(),
                        "action_mask": agent0_mask.tolist(),
                        "mask_pass_legal": int(pass_legal),
                        "mask_shot_legal": int(shot_legal),
                        "mask_tackle_legal": int(tackle_legal),
                        "mask_dribble_legal": int(dribble_legal),
                        "mask_sum": int(agent0_mask.sum()),
                        "masked_logits": masked_logits.tolist(),
                        "probs": probs.tolist(),
                        "entropy": float(entropy),
                        "top1_action": int(top1_idx),
                        "top1_action_name": ACTION_NAMES[top1_idx] if top1_idx < len(ACTION_NAMES) else str(top1_idx),
                        "top1_prob": float(top1_prob),
                        "top3": top3,
                        "pi_move": _action_group_prob(probs, MOVE_ACTION_IDS),
                        "pi_pass": _action_group_prob(probs, PASS_ACTION_IDS),
                        "pi_shot": _action_group_prob(probs, SHOT_ACTION_IDS),
                        "pi_tackle": float(probs[TACKLE_ACTION_ID]),
                        "pi_dribble": float(probs[DRIBBLE_ACTION_ID]),
                        "delta_pass_move_logit": float(pass_best - move_best),
                        "delta_shot_move_logit": float(shot_logit - move_best),
                        "action_taken": int(actions_np[0]),
                        "action_taken_name": ACTION_NAMES[int(actions_np[0])] if int(actions_np[0]) < len(ACTION_NAMES) else str(int(actions_np[0])),
                        "has_ball": True,
                        "event_code": event_code,
                        "reward": shared_rew,
                        "terminated": terminated,
                        "truncated": truncated,
                        "done": done,
                    }
                    ep_frames.append(frame)
                    all_frames.append(frame)

                tick_idx += 1
                if done:
                    break

            episode_summary = {
                "seed": seed,
                "episode": ep,
                "checkpoint": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_timesteps": checkpoint_timesteps,
                "scenario": scenario,
                "ep_seed": ep_seed,
                "onball_frames": len(ep_frames),
                "episode_reward": ep_reward,
                "episode_length": ep_length,
            }
            episodes_data.append(episode_summary)

            if (ep + 1) % 10 == 0:
                print(f"  [POST-REWEIGHT seed={seed}] Ep {ep+1:3d}/{num_episodes} | "
                      f"onball_frames={len(ep_frames)} | Reward={ep_reward:+.3f} | Len={ep_length}")

    finally:
        env.close()

    return {
        "seed": seed,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_timesteps": checkpoint_timesteps,
        "scenario": scenario,
        "deterministic": deterministic,
        "base_seed": base_seed,
        "episodes": episodes_data,
        "frames": all_frames,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_frames(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups = defaultdict(list)
    for f in frames:
        key = (f["seed"], f["checkpoint_timesteps"], f["checkpoint"])
        groups[key].append(f)

    summary_rows = []
    for (seed, timesteps, ckpt), group in groups.items():
        probs_move = np.array([f["pi_move"] for f in group], dtype=np.float32)
        probs_pass = np.array([f["pi_pass"] for f in group], dtype=np.float32)
        probs_shot = np.array([f["pi_shot"] for f in group], dtype=np.float32)
        probs_tackle = np.array([f["pi_tackle"] for f in group], dtype=np.float32)
        probs_dribble = np.array([f["pi_dribble"] for f in group], dtype=np.float32)
        entropies = np.array([f["entropy"] for f in group], dtype=np.float32)
        delta_pass_move = np.array([f["delta_pass_move_logit"] for f in group], dtype=np.float32)
        delta_shot_move = np.array([f["delta_shot_move_logit"] for f in group], dtype=np.float32)
        mask_sums = np.array([f["mask_sum"] for f in group], dtype=np.float32)
        mask_pass_legal = np.array([f["mask_pass_legal"] for f in group], dtype=np.float32)
        mask_shot_legal = np.array([f["mask_shot_legal"] for f in group], dtype=np.float32)

        top1_probs = np.array([f["top1_prob"] for f in group], dtype=np.float32)

        summary_rows.append({
            "seed": seed,
            "checkpoint_label": os.path.basename(ckpt),
            "checkpoint_timesteps": timesteps,
            "checkpoint": ckpt,
            "checkpoint_sha256": "",
            "n_frames": len(group),
            "pi_move_mean": float(probs_move.mean()),
            "pi_move_std": float(probs_move.std()),
            "pi_pass_mean": float(probs_pass.mean()),
            "pi_pass_std": float(probs_pass.std()),
            "pi_shot_mean": float(probs_shot.mean()),
            "pi_shot_std": float(probs_shot.std()),
            "pi_tackle_mean": float(probs_tackle.mean()),
            "pi_tackle_std": float(probs_tackle.std()),
            "pi_dribble_mean": float(probs_dribble.mean()),
            "pi_dribble_std": float(probs_dribble.std()),
            "entropy_mean": float(entropies.mean()),
            "entropy_std": float(entropies.std()),
            "entropy_median": float(np.median(entropies)),
            "delta_pass_move_mean": float(delta_pass_move.mean()),
            "delta_pass_move_std": float(delta_pass_move.std()),
            "delta_shot_move_mean": float(delta_shot_move.mean()),
            "delta_shot_move_std": float(delta_shot_move.std()),
            "mask_sum_mean": float(mask_sums.mean()),
            "mask_pass_legal_rate": float(mask_pass_legal.mean()),
            "mask_shot_legal_rate": float(mask_shot_legal.mean()),
            "top1_prob_mean": float(top1_probs.mean()),
            "top1_prob_median": float(np.median(top1_probs)),
        })

    return {"per_group": summary_rows}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_snapshot(seeds, scenario, num_episodes, deterministic, base_seed, output_dir, checkpoint_specs):
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "training", "models")

    all_results = []
    all_frames = []

    for seed in seeds:
        for label, filename_template, timesteps in checkpoint_specs:
            filename = filename_template.format(seed=seed)
            ckpt_path = os.path.join(model_dir, filename)
            if not os.path.exists(ckpt_path):
                print(f"  SKIP: {label} seed={seed} — {ckpt_path} not found")
                continue

            print(f"\n{'='*70}")
            print(f"AUDIT: {label} seed={seed}")
            print(f"Checkpoint: {ckpt_path}")
            print(f"{'='*70}")

            ckpt_sha256 = sha256_of(ckpt_path)
            actor, critic, obs_dim, action_dim, ckpt_ts = load_checkpoint(ckpt_path)
            actual_timesteps = ckpt_ts if ckpt_ts is not None else timesteps

            result = collect_forensics_episodes(
                actor=actor,
                critic=critic,
                checkpoint_path=ckpt_path,
                checkpoint_sha256=ckpt_sha256,
                checkpoint_timesteps=actual_timesteps,
                seed=seed,
                scenario=scenario,
                num_episodes=num_episodes,
                deterministic=deterministic,
                base_seed=base_seed,
            )
            all_results.append(result)
            all_frames.extend(result["frames"])

    # Aggregate
    agg = aggregate_frames(all_frames)

    # Write detail JSON
    detail_path = os.path.join(output_dir, "post_reweight_logit_detail.json")
    with open(detail_path, "w") as f:
        json.dump({
            "results": all_results,
            "frames": all_frames,
            "aggregate": agg,
        }, f, indent=2, default=lambda o: None if o is None else float(o))
    print(f"\nDetail JSON: {detail_path}")

    # Write summary CSV
    csv_path = os.path.join(output_dir, "post_reweight_logit_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "checkpoint_label", "checkpoint_timesteps", "checkpoint", "checkpoint_sha256",
            "n_frames", "pi_move_mean", "pi_move_std",
            "pi_pass_mean", "pi_pass_std",
            "pi_shot_mean", "pi_shot_std",
            "pi_tackle_mean", "pi_tackle_std",
            "pi_dribble_mean", "pi_dribble_std",
            "entropy_mean", "entropy_std", "entropy_median",
            "delta_pass_move_mean", "delta_pass_move_std",
            "delta_shot_move_mean", "delta_shot_move_std",
            "mask_sum_mean", "mask_pass_legal_rate", "mask_shot_legal_rate",
            "top1_prob_mean", "top1_prob_median",
        ])
        for row in agg["per_group"]:
            writer.writerow([
                row["seed"], row["checkpoint_label"], row["checkpoint_timesteps"], row["checkpoint"], row.get("checkpoint_sha256", ""),
                row["n_frames"],
                row["pi_move_mean"], row["pi_move_std"],
                row["pi_pass_mean"], row["pi_pass_std"],
                row["pi_shot_mean"], row["pi_shot_std"],
                row["pi_tackle_mean"], row["pi_tackle_std"],
                row["pi_dribble_mean"], row["pi_dribble_std"],
                row["entropy_mean"], row["entropy_std"], row["entropy_median"],
                row["delta_pass_move_mean"], row["delta_pass_move_std"],
                row["delta_shot_move_mean"], row["delta_shot_move_std"],
                row["mask_sum_mean"], row["mask_pass_legal_rate"], row["mask_shot_legal_rate"],
                row["top1_prob_mean"], row["top1_prob_median"],
            ])
    print(f"Summary CSV: {csv_path}")

    return all_results, all_frames, agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Post-Reweight Logit Snapshot")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper_onball")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7, 999])
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--base-seed", type=int, default=700000)
    parser.add_argument("--output-dir", type=str, default="training/results")
    parser.add_argument("--checkpoint-specs", type=str, nargs="+", default=None,
                        help="Override checkpoint labels to audit (e.g. actorreweight_15000 actorreweight_final)")
    args = parser.parse_args()

    specs = CHECKPOINT_SPECS
    if args.checkpoint_specs:
        label_to_spec = {label: (label, template, ts) for label, template, ts in CHECKPOINT_SPECS}
        specs = [label_to_spec[l] for l in args.checkpoint_specs if l in label_to_spec]

    head_sha = "unknown"
    try:
        import subprocess
        head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        pass

    all_results, all_frames, agg = run_snapshot(
        seeds=args.seeds,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        checkpoint_specs=specs,
    )

    print("\nSnapshot complete.")
    print(f"Total frames: {len(all_frames)}")
    print(f"Aggregate groups: {len(agg['per_group'])}")


if __name__ == "__main__":
    main()
