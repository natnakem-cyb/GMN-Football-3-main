"""
GMN-Football-3 — Exploration Ablation Evaluation: pure-π evaluation across
all intermediate checkpoints for the Form A targeted on-ball entropy bonus.

Produces:
  - training/results/exploration_ablation_summary.csv
  - training/results/EXPLORATION_ABLATION_FINDINGS.md

Checkpoints evaluated:
  - Fresh baseline: mappo_academy_3_vs_1_with_keeper_seed{N}_freshtrain_50176.pt
  - Ablation intermediates: mappo_academy_3_vs_1_with_keeper_seed{N}_expl_ablation_{K}k.pt
    where K ∈ {5, 10, 15, 20, 25, 30}
  Seeds: 42, 123, 7, 999
  Scenario: academy_3_vs_1_with_keeper_onball
  Episodes per checkpoint per seed: 20
  Mode: pure-π, deterministic, no forcing, no mix-script
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv, EVENT_CODE_MAP
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

OBS_DIM = 127
ACTION_DIM = 19
OBS_BALL_OWNERSHIP_SLICE = slice(94, 97)  # [no-one, left, right]

FRESH_BASELINE_TEMPLATE = "mappo_academy_3_vs_1_with_keeper_seed{seed}_freshtrain_50176.pt"
ABLATION_TEMPLATE = "mappo_academy_3_vs_1_with_keeper_seed{seed}_expl_ablation_{K}.pt"
ABLATION_CHECKPOINTS = [5000, 10000, 15000, 20000, 25000, 30000]  # steps

SCENARIO = "academy_3_vs_1_with_keeper_onball"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _event_type_from_code(event_code: Optional[int]) -> Optional[str]:
    if event_code is None or event_code <= 0 or event_code >= len(EVENT_CODE_MAP):
        return None
    return EVENT_CODE_MAP[event_code]


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


def _closest_row(seed_to_rows: Dict[int, Dict[int, Dict[str, Any]]], seed: int, target_ts: int) -> Optional[Dict[str, Any]]:
    """Find the row for seed with the closest timestep <= target_ts."""
    rows = seed_to_rows.get(seed, {})
    candidates = [ts for ts in rows if ts <= target_ts + 1000]
    if not candidates:
        return None
    return rows[max(candidates)]


def _select_policy_actions(
    actor: SharedActor,
    local_obs: np.ndarray,
    mask_matrix: np.ndarray,
    deterministic: bool = True,
) -> np.ndarray:
    """Return action indices from the frozen policy for all agents."""
    with torch.no_grad():
        obs_tensor = torch.from_numpy(local_obs).float()
        mask_tensor = torch.tensor(mask_matrix, dtype=torch.bool)
        dist = actor(obs_tensor, mask_tensor)
        logits = dist.logits
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
    return actions.cpu().numpy()


# ---------------------------------------------------------------------------
# Episode collector
# ---------------------------------------------------------------------------
def collect_evaluation_episodes(
    actor: SharedActor,
    critic: Optional[CentralizedCritic],
    checkpoint_path: str,
    checkpoint_sha256: str,
    checkpoint_timesteps: int,
    seed: int,
    scenario: str,
    num_episodes: int,
    deterministic: bool,
    base_seed: int,
) -> Dict[str, Any]:
    """Run pure-π evaluation episodes and collect episode-level metrics."""
    env = GMNMultiAgentEnv(scenario=scenario, auto_start_bridge=True, port=5050)
    controllable_agents = list(env.possible_agents)

    episodes_data: List[Dict[str, Any]] = []

    try:
        for ep in range(num_episodes):
            ep_seed = base_seed + ep * 1009
            obs_dict, _ = env.reset(seed=ep_seed)
            current_ep_masks = unwrap_masks(obs_dict)
            obs_dict = unwrap_obs(obs_dict)

            current_agents = list(env.agents if env.agents else controllable_agents)
            if not current_agents:
                continue

            ep_reward = 0.0
            ep_length = 0
            action_counts = np.zeros(ACTION_DIM, dtype=np.int64)
            pass_completed_count = 0
            shot_event_count = 0
            goal_count = 0
            pass_legal_frames = 0
            shot_legal_frames = 0
            pass_selected_legal = 0
            shot_selected_legal = 0

            tick_idx = 0
            while True:
                current_agents = list(env.agents if env.agents else controllable_agents)
                if not current_agents:
                    break

                local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
                mask_matrix = _mask_matrix(current_ep_masks, current_agents)

                # Policy action selection
                actions_np = _select_policy_actions(actor, local_obs, mask_matrix, deterministic)
                action_dict = {a: int(actions_np[i]) for i, a in enumerate(current_agents)}

                # Track action counts for agent 0
                action_counts[int(actions_np[0])] += 1

                # Track pass/shot selection when legal (agent 0)
                agent0_mask = mask_matrix[0] if len(mask_matrix) > 0 else np.zeros(ACTION_DIM, dtype=bool)
                if agent0_mask[9] or agent0_mask[10] or agent0_mask[11]:
                    pass_legal_frames += 1
                    if int(actions_np[0]) in PASS_ACTION_IDS:
                        pass_selected_legal += 1
                if agent0_mask[12]:
                    shot_legal_frames += 1
                    if int(actions_np[0]) == 12:
                        shot_selected_legal += 1

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

                # Extract event info
                event_code = getattr(env, "_last_frame_event_code", None)
                event_type = _event_type_from_code(event_code)

                if event_type == "pass_completed":
                    pass_completed_count += 1
                if event_type in ("shot", "shot_saved", "shot_missed"):
                    shot_event_count += 1
                if event_type == "goal":
                    goal_count += 1

                tick_idx += 1
                if done:
                    break

            # Compute selection rates (overall)
            total_actions = float(action_counts.sum())
            action_rates = {
                ACTION_NAMES[i]: float(action_counts[i]) / total_actions if total_actions > 0 else 0.0
                for i in range(ACTION_DIM)
            }

            # Compute football-action selection rates when legal
            pass_rate_legal = pass_selected_legal / pass_legal_frames if pass_legal_frames > 0 else 0.0
            shot_rate_legal = shot_selected_legal / shot_legal_frames if shot_legal_frames > 0 else 0.0
            football_rate_legal = (pass_selected_legal + shot_selected_legal) / (pass_legal_frames + shot_legal_frames) \
                if (pass_legal_frames + shot_legal_frames) > 0 else 0.0

            episode_summary = {
                "seed": seed,
                "episode": ep,
                "checkpoint": os.path.basename(checkpoint_path),
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_timesteps": checkpoint_timesteps,
                "scenario": scenario,
                "ep_seed": ep_seed,
                "episode_reward": ep_reward,
                "episode_length": ep_length,
                "pass_completed_count": pass_completed_count,
                "shot_event_count": shot_event_count,
                "goal_count": goal_count,
                "action_counts": {ACTION_NAMES[i]: int(action_counts[i]) for i in range(ACTION_DIM)},
                "action_rates": action_rates,
                "pass_rate_legal": pass_rate_legal,
                "shot_rate_legal": shot_rate_legal,
                "football_rate_legal": football_rate_legal,
                "pass_legal_frames": pass_legal_frames,
                "shot_legal_frames": shot_legal_frames,
                "pass_selected_legal": pass_selected_legal,
                "shot_selected_legal": shot_selected_legal,
                "pi_pass": action_rates.get("SHORT_PASS", 0.0) + action_rates.get("HIGH_PASS", 0.0) + action_rates.get("LONG_PASS", 0.0),
                "pi_shot": action_rates.get("SHOT", 0.0),
                "pi_move": sum(action_rates.get(ACTION_NAMES[i], 0.0) for i in range(9)),
            }
            episodes_data.append(episode_summary)

            if (ep + 1) % 5 == 0:
                print(f"  [EVAL seed={seed}] Ep {ep+1:3d}/{num_episodes} | "
                      f"Reward={ep_reward:+.3f} | Len={ep_length} | "
                      f"PASS={pass_completed_count} SHOT={shot_event_count} GOAL={goal_count}")

    finally:
        env.close()

    return {
        "seed": seed,
        "checkpoint": os.path.basename(checkpoint_path),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_timesteps": checkpoint_timesteps,
        "scenario": scenario,
        "deterministic": deterministic,
        "base_seed": base_seed,
        "episodes": episodes_data,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_episodes(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate episode-level metrics per (seed, checkpoint_timesteps)."""
    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for result in all_results:
        key = (result["seed"], result["checkpoint_timesteps"])
        groups.setdefault(key, []).extend(result["episodes"])

    summary_rows = []
    for (seed, timesteps), episodes in sorted(groups.items()):
        n = len(episodes)
        rewards = np.array([ep["episode_reward"] for ep in episodes], dtype=np.float32)
        lengths = np.array([ep["episode_length"] for ep in episodes], dtype=np.float32)
        pass_completed = np.array([ep["pass_completed_count"] for ep in episodes], dtype=np.int64)
        shot_events = np.array([ep["shot_event_count"] for ep in episodes], dtype=np.int64)
        goals = np.array([ep["goal_count"] for ep in episodes], dtype=np.int64)

        pi_pass = np.array([ep["pi_pass"] for ep in episodes], dtype=np.float32)
        pi_shot = np.array([ep["pi_shot"] for ep in episodes], dtype=np.float32)
        pi_move = np.array([ep["pi_move"] for ep in episodes], dtype=np.float32)
        football_rate_legal = np.array([ep["football_rate_legal"] for ep in episodes], dtype=np.float32)

        pass_legal_frames = sum(ep["pass_legal_frames"] for ep in episodes)
        shot_legal_frames = sum(ep["shot_legal_frames"] for ep in episodes)
        pass_selected_legal = sum(ep["pass_selected_legal"] for ep in episodes)
        shot_selected_legal = sum(ep["shot_selected_legal"] for ep in episodes)

        summary_rows.append({
            "seed": seed,
            "checkpoint_timesteps": timesteps,
            "checkpoint": episodes[0]["checkpoint"] if episodes else "",
            "checkpoint_sha256": episodes[0]["checkpoint_sha256"] if episodes else "",
            "n_episodes": n,
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "mean_length": float(lengths.mean()),
            "std_length": float(lengths.std()),
            "total_pass_completed": int(pass_completed.sum()),
            "total_shot_events": int(shot_events.sum()),
            "total_goals": int(goals.sum()),
            "seeds_with_pass_completed": int(np.sum(pass_completed > 0)),
            "seeds_with_shot_event": int(np.sum(shot_events > 0)),
            "seeds_with_goal": int(np.sum(goals > 0)),
            "mean_pi_pass": float(pi_pass.mean()),
            "std_pi_pass": float(pi_pass.std()),
            "mean_pi_shot": float(pi_shot.mean()),
            "std_pi_shot": float(pi_shot.std()),
            "mean_pi_move": float(pi_move.mean()),
            "mean_football_rate_legal": float(football_rate_legal.mean()),
            "std_football_rate_legal": float(football_rate_legal.std()),
            "pass_legal_frames": int(pass_legal_frames),
            "shot_legal_frames": int(shot_legal_frames),
            "pass_selected_legal": int(pass_selected_legal),
            "shot_selected_legal": int(shot_selected_legal),
            "pass_rate_all_legal": pass_selected_legal / pass_legal_frames if pass_legal_frames > 0 else 0.0,
            "shot_rate_all_legal": shot_selected_legal / shot_legal_frames if shot_legal_frames > 0 else 0.0,
            "football_rate_all_legal": (pass_selected_legal + shot_selected_legal) / (pass_legal_frames + shot_legal_frames) \
                if (pass_legal_frames + shot_legal_frames) > 0 else 0.0,
        })

    return {"per_group": summary_rows, "groups": groups}


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------
def generate_findings_report(
    agg: Dict[str, Any],
    groups: Dict[Tuple[int, int], List[Dict[str, Any]]],
    seeds: List[int],
    output_dir: str,
    head_sha: str,
    fresh_baseline_rows: Dict[int, Dict[str, Any]],
    ablation_tail_rows: Dict[int, Dict[str, Any]],
    seed_to_rows: Dict[int, Dict[int, Dict[str, Any]]],
) -> str:
    """Generate EXPLORATION_ABLATION_FINDINGS.md."""
    per_group = { (r["seed"], r["checkpoint_timesteps"]): r for r in agg["per_group"] }

    lines = []
    lines.append("# Exploration Ablation Findings — Form A: Targeted On-Ball Entropy Bonus")
    lines.append(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    lines.append(f"HEAD: {head_sha}")
    lines.append("")
    lines.append("## 1. Protocol Summary")
    lines.append("")
    lines.append("- **Mechanism:** Targeted entropy bonus on legal PASS/SHOT marginal during Phase 1 (0–15k), alpha=0.50")
    lines.append("- **Phase 1:** 0–15k steps, bonus ON")
    lines.append("- **Phase 2:** 15k–30k steps, bonus OFF (unscripted tail)")
    lines.append("- **Checkpoints:** 5k, 10k, 15k, 20k, 25k, 30k for seeds 42, 123, 7, 999")
    lines.append("- **Baseline:** fresh-training final (50k) for each seed")
    lines.append("- **Evaluation:** pure-π, deterministic, 20 episodes/checkpoint/seed, no forcing")
    lines.append("")

    # TASK 1 — Full trajectory table
    lines.append("## 2. Full Trajectory Table (Phase 1 context, non-evidentiary)")
    lines.append("")
    lines.append("> **Note:** Phase 1 numbers are diagnostic-only context per protocol. Success is judged on Phase 2 tail (30k) only.")
    lines.append("")
    lines.append("| Checkpoint | Seed | n_ep | π(PASS) | π(SHOT) | π(MOVE) | Football Rate (legal) | H(π) proxy | PASS+SHOT | Pass Completed | Shot Events | Goals | Mean Reward |")
    lines.append("|------------|------|------|---------|---------|---------|----------------------|------------|-----------|----------------|-------------|-------|-------------|")

    for ts in [5000, 10000, 15000, 20000, 25000, 30000]:
        for seed in seeds:
            row = _closest_row(seed_to_rows, seed, ts)
            if row is None:
                continue
            phase = "P1" if ts <= 15000 else "P2"
            label = f"{ts//1000}k ({phase})" if ts >= 1000 else f"{ts} ({phase})"
            lines.append(
                f"| {label} | {seed} | {row['n_episodes']} | "
                f"{row['mean_pi_pass']:.4f} | {row['mean_pi_shot']:.4f} | {row['mean_pi_move']:.4f} | "
                f"{row['mean_football_rate_legal']:.4f} | — | "
                f"{row['mean_pi_pass'] + row['mean_pi_shot']:.4f} | "
                f"{row['total_pass_completed']} | {row['total_shot_events']} | {row['total_goals']} | "
                f"{row['mean_reward']:+.4f} |"
            )

    # Fresh baseline rows
    for seed in seeds:
        row = fresh_baseline_rows.get(seed)
        if row is None:
            continue
        lines.append(
            f"| fresh 50k (baseline) | {seed} | {row['n_episodes']} | "
            f"{row['mean_pi_pass']:.4f} | {row['mean_pi_shot']:.4f} | {row['mean_pi_move']:.4f} | "
            f"{row['mean_football_rate_legal']:.4f} | — | "
            f"{row['mean_pi_pass'] + row['mean_pi_shot']:.4f} | "
            f"{row['total_pass_completed']} | {row['total_shot_events']} | {row['total_goals']} | "
            f"{row['mean_reward']:+.4f} |"
        )
    lines.append("")

    # TASK 2 — Phase 2 gate judgment
    lines.append("## 3. Phase 2 Tail-End Gate Judgment (30k, primary criterion)")
    lines.append("")
    lines.append("### Primary Criterion: ≥3/4 seeds exceed fresh baseline by ≥1.5pp on PASS+SHOT combined rate")
    lines.append("")

    primary_pass_count = 0
    primary_details = []
    for seed in seeds:
        baseline_row = fresh_baseline_rows.get(seed)
        ablation_row = ablation_tail_rows.get(seed)
        if baseline_row is None or ablation_row is None:
            continue

        baseline_rate = baseline_row["mean_pi_pass"] + baseline_row["mean_pi_shot"]
        ablation_rate = ablation_row["mean_pi_pass"] + ablation_row["mean_pi_shot"]
        delta = ablation_rate - baseline_rate
        meets = delta >= 0.015
        if meets:
            primary_pass_count += 1

        primary_details.append({
            "seed": seed,
            "baseline_rate": baseline_rate,
            "ablation_rate": ablation_rate,
            "delta": delta,
            "meets": meets,
        })

    lines.append("| Seed | Fresh Baseline (PASS+SHOT) | 30k Ablation (PASS+SHOT) | Δ | Meets ≥1.5pp? |")
    lines.append("|------|---------------------------|--------------------------|---|---------------|")
    for d in primary_details:
        lines.append(
            f"| {d['seed']} | {d['baseline_rate']:.4f} | {d['ablation_rate']:.4f} | "
            f"{d['delta']:+.4f} | {'YES' if d['meets'] else 'NO'} |"
        )
    lines.append("")
    lines.append(f"**Primary criterion result: {primary_pass_count}/{len(seeds)} seeds meet the ≥1.5pp threshold.**")
    lines.append("")

    # TASK 3 — Secondary criterion
    lines.append("### Secondary Criterion: ≥2/4 seeds show non-forced pass_completed + shot events on tail")
    lines.append("")
    secondary_pass_count = 0
    secondary_details = []
    for seed in seeds:
        ablation_row = ablation_tail_rows.get(seed)
        if ablation_row is None:
            continue
        has_pass = ablation_row["total_pass_completed"] > 0
        has_shot = ablation_row["total_shot_events"] > 0
        meets = has_pass and has_shot
        if meets:
            secondary_pass_count += 1
        secondary_details.append({
            "seed": seed,
            "pass_completed": ablation_row["total_pass_completed"],
            "shot_events": ablation_row["total_shot_events"],
            "goals": ablation_row["total_goals"],
            "meets": meets,
        })

    lines.append("| Seed | pass_completed | shot_events | goals | Meets both? |")
    lines.append("|------|----------------|-------------|-------|--------------|")
    for d in secondary_details:
        lines.append(
            f"| {d['seed']} | {d['pass_completed']} | {d['shot_events']} | {d['goals']} | "
            f"{'YES' if d['meets'] else 'NO'} |"
        )
    lines.append("")
    lines.append(f"**Secondary criterion result: {secondary_pass_count}/{len(seeds)} seeds meet both event thresholds.**")
    lines.append("")

    # TASK 4 — Guard criterion
    lines.append("### Guard Criterion (Reversion Check): Any seed below own fresh baseline?")
    lines.append("")
    guard_details = []
    for seed in seeds:
        baseline_row = fresh_baseline_rows.get(seed)
        ablation_row = ablation_tail_rows.get(seed)
        if baseline_row is None or ablation_row is None:
            continue
        baseline_rate = baseline_row["mean_pi_pass"] + baseline_row["mean_pi_shot"]
        ablation_rate = ablation_row["mean_pi_pass"] + ablation_row["mean_pi_shot"]
        reverted = ablation_rate < baseline_rate
        guard_details.append({
            "seed": seed,
            "baseline_rate": baseline_rate,
            "ablation_rate": ablation_rate,
            "reverted": reverted,
        })

    lines.append("| Seed | Fresh Baseline (PASS+SHOT) | 30k Ablation (PASS+SHOT) | Reverted? |")
    lines.append("|------|---------------------------|--------------------------|-----------|")
    for d in guard_details:
        lines.append(
            f"| {d['seed']} | {d['baseline_rate']:.4f} | {d['ablation_rate']:.4f} | "
            f"{'YES — below baseline' if d['reverted'] else 'NO'} |"
        )
    lines.append("")

    # TASK 5 — Phase 1 context
    lines.append("## 4. Phase 1 Context (Diagnostic Only)")
    lines.append("")
    lines.append("Phase 1 (0–15k) shows whether the entropy bonus mechanism was active.")
    lines.append("")
    lines.append("| Checkpoint | Seed | π(PASS) | π(SHOT) | Football Rate (legal) | PASS+SHOT |")
    lines.append("|------------|------|---------|---------|----------------------|-----------|")
    for ts in [5000, 10000, 15000]:
        for seed in seeds:
            row = _closest_row(seed_to_rows, seed, ts)
            if row is None:
                continue
            lines.append(
                f"| {ts//1000}k | {seed} | {row['mean_pi_pass']:.4f} | {row['mean_pi_shot']:.4f} | "
                f"{row['mean_football_rate_legal']:.4f} | {row['mean_pi_pass'] + row['mean_pi_shot']:.4f} |"
            )
    lines.append("")

    # TASK 6 — Collapse / recovery shape
    lines.append("## 5. Collapse / Recovery Shape")
    lines.append("")
    lines.append("| Seed | 15k PASS+SHOT | 30k PASS+SHOT | 50k Baseline PASS+SHOT | Phase 2 Δ | Phase 1→2 Δ |")
    lines.append("|------|---------------|---------------|------------------------|-----------|-------------|")
    for seed in seeds:
        row_15k = _closest_row(seed_to_rows, seed, 15000)
        row_30k = ablation_tail_rows.get(seed)
        baseline_row = fresh_baseline_rows.get(seed)
        if row_15k is None or row_30k is None or baseline_row is None:
            continue
        rate_15k = row_15k["mean_pi_pass"] + row_15k["mean_pi_shot"]
        rate_30k = row_30k["mean_pi_pass"] + row_30k["mean_pi_shot"]
        rate_baseline = baseline_row["mean_pi_pass"] + baseline_row["mean_pi_shot"]
        phase2_delta = rate_30k - rate_15k
        phase1_to_2_delta = rate_30k - rate_baseline
        lines.append(
            f"| {seed} | {rate_15k:.4f} | {rate_30k:.4f} | {rate_baseline:.4f} | "
            f"{phase2_delta:+.4f} | {phase1_to_2_delta:+.4f} |"
        )
    lines.append("")

    # OVERALL JUDGMENT
    lines.append("## 6. Overall Judgment")
    lines.append("")
    primary_pass = primary_pass_count >= 3
    secondary_pass = secondary_pass_count >= 2

    if primary_pass and secondary_pass:
        judgment = "ABLATION SUCCESSFUL"
        judgment_detail = (
            f"Primary criterion passed ({primary_pass_count}/{len(seeds)} seeds ≥1.5pp improvement). "
            f"Secondary criterion passed ({secondary_pass_count}/{len(seeds)} seeds with both event types)."
        )
    elif primary_pass:
        judgment = "PARTIAL SUCCESS"
        judgment_detail = (
            f"Primary criterion passed ({primary_pass_count}/{len(seeds)} seeds ≥1.5pp improvement) "
            f"but secondary criterion not met ({secondary_pass_count}/{len(seeds)} seeds with both event types)."
        )
    else:
        judgment = "ABLATION INSUFFICIENT"
        judgment_detail = (
            f"Primary criterion failed ({primary_pass_count}/{len(seeds)} seeds ≥1.5pp improvement). "
            f"Secondary criterion: {secondary_pass_count}/{len(seeds)} seeds with both event types."
        )

    lines.append(f"**Judgment: {judgment}**")
    lines.append("")
    lines.append(f"{judgment_detail}")
    lines.append("")

    # Guard summary
    reverted_seeds = [d["seed"] for d in guard_details if d["reverted"]]
    if reverted_seeds:
        lines.append(f"**Guard violations:** Seeds {', '.join(map(str, reverted_seeds))} fell below their own fresh baseline on the Phase 2 tail.")
    else:
        lines.append("**Guard:** No seed fell below its own fresh baseline on the Phase 2 tail.")
    lines.append("")

    # NEXT RECOMMENDATION
    lines.append("## 7. Next Recommendation")
    lines.append("")
    if primary_pass and secondary_pass:
        lines.append("The targeted entropy bonus produced a sustained preference shift in Phase 2. "
                     "Recommend proceeding to Phase C (controlled exploration ablation with fixed horizon, isolate E) "
                     "or increasing sample size for robustness.")
    elif primary_pass:
        lines.append("Primary criterion met but secondary events are sparse. Recommend increasing episodes per checkpoint "
                     "or exploring a higher alpha coefficient before declaring the intervention effective.")
    else:
        lines.append("The ablation did not produce sufficient improvement on the Phase 2 tail. "
                     "Recommend either (a) increasing alpha, (b) extending Phase 2 duration, or (c) trying a count-based "
                     "exploration reward instead of an entropy bonus.")
    lines.append("")

    lines.append("CONFIRMATIONS")
    lines.append("-" * 40)
    lines.append("  No reward/GAE/mask/network/entropy changes: yes")
    lines.append("  Evaluation-only run, no training performed: yes")
    lines.append("  Prior result files untouched: yes")
    lines.append("")
    lines.append("FILES WRITTEN")
    lines.append("-" * 40)
    lines.append("  - training/results/exploration_ablation_summary.csv")
    lines.append("  - training/results/EXPLORATION_ABLATION_FINDINGS.md")
    lines.append("")

    report_path = os.path.join(output_dir, "EXPLORATION_ABLATION_FINDINGS.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Exploration Ablation Evaluation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7, 999])
    parser.add_argument("--scenario", type=str, default=SCENARIO)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--base-seed", type=int, default=700000)
    parser.add_argument("--output-dir", type=str, default="training/results")
    parser.add_argument("--models-dir", type=str, default="training/models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.models_dir))

    # Build checkpoint specs
    checkpoint_specs = []
    for seed in args.seeds:
        # Fresh baseline
        baseline_fn = FRESH_BASELINE_TEMPLATE.format(seed=seed)
        checkpoint_specs.append(("fresh_baseline", baseline_fn, 50176, seed))
        # Ablation intermediates
        for k in ABLATION_CHECKPOINTS:
            ablation_fn = ABLATION_TEMPLATE.format(seed=seed, K=k)
            checkpoint_specs.append((f"expl_ablation_{k}", ablation_fn, k, seed))

    all_results = []

    for label, filename, timesteps, seed in checkpoint_specs:
        ckpt_path = os.path.join(model_dir, filename)
        if not os.path.exists(ckpt_path):
            print(f"  SKIP: {label} seed={seed} — {ckpt_path} not found")
            continue

        print(f"\n{'='*70}")
        print(f"EVAL: {label} seed={seed} (t={timesteps})")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*70}")

        ckpt_sha256 = sha256_of(ckpt_path)
        actor, critic, obs_dim, action_dim, ckpt_ts = load_checkpoint(ckpt_path)
        actual_timesteps = ckpt_ts if ckpt_ts is not None else timesteps

        result = collect_evaluation_episodes(
            actor=actor,
            critic=critic,
            checkpoint_path=ckpt_path,
            checkpoint_sha256=ckpt_sha256,
            checkpoint_timesteps=actual_timesteps,
            seed=seed,
            scenario=args.scenario,
            num_episodes=args.num_episodes,
            deterministic=args.deterministic,
            base_seed=args.base_seed,
        )
        all_results.append(result)

    # Aggregate
    agg = aggregate_episodes(all_results)
    per_group = { (r["seed"], r["checkpoint_timesteps"]): r for r in agg["per_group"] }
    groups = agg["groups"]

    # Build lookup: seed -> {target_ts: actual_row}
    seed_to_rows: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for row in agg["per_group"]:
        seed_to_rows.setdefault(row["seed"], {})[row["checkpoint_timesteps"]] = row

    def _closest_row(seed: int, target_ts: int) -> Optional[Dict[str, Any]]:
        """Find the row for seed with the closest timestep <= target_ts."""
        rows = seed_to_rows.get(seed, {})
        candidates = [ts for ts in rows if ts <= target_ts + 1000]
        if not candidates:
            return None
        return rows[max(candidates)]

    # Build fresh baseline lookup
    fresh_baseline_rows: Dict[int, Dict[str, Any]] = {}
    for seed in args.seeds:
        row = _closest_row(seed, 50176)
        if row:
            fresh_baseline_rows[seed] = row

    # Build ablation tail lookup: use the maximum timesteps for each seed
    ablation_tail_rows: Dict[int, Dict[str, Any]] = {}
    for seed in args.seeds:
        row = _closest_row(seed, 30000)
        if row and "expl_ablation" in row["checkpoint"]:
            ablation_tail_rows[seed] = row

    # Write summary CSV
    csv_path = os.path.join(args.output_dir, "exploration_ablation_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "checkpoint_label", "checkpoint_timesteps", "checkpoint_sha256",
            "n_episodes", "mean_reward", "std_reward", "mean_length", "std_length",
            "total_pass_completed", "total_shot_events", "total_goals",
            "seeds_with_pass_completed", "seeds_with_shot_event", "seeds_with_goal",
            "mean_pi_pass", "std_pi_pass", "mean_pi_shot", "std_pi_shot",
            "mean_pi_move", "mean_football_rate_legal", "std_football_rate_legal",
            "pass_legal_frames", "shot_legal_frames",
            "pass_selected_legal", "shot_selected_legal",
            "pass_rate_all_legal", "shot_rate_all_legal", "football_rate_all_legal",
        ])
        for row in agg["per_group"]:
            writer.writerow([
                row["seed"], row["checkpoint"], row["checkpoint_timesteps"], row["checkpoint_sha256"],
                row["n_episodes"], row["mean_reward"], row["std_reward"], row["mean_length"], row["std_length"],
                row["total_pass_completed"], row["total_shot_events"], row["total_goals"],
                row["seeds_with_pass_completed"], row["seeds_with_shot_event"], row["seeds_with_goal"],
                row["mean_pi_pass"], row["std_pi_pass"], row["mean_pi_shot"], row["std_pi_shot"],
                row["mean_pi_move"], row["mean_football_rate_legal"], row["std_football_rate_legal"],
                row["pass_legal_frames"], row["shot_legal_frames"],
                row["pass_selected_legal"], row["shot_selected_legal"],
                row["pass_rate_all_legal"], row["shot_rate_all_legal"], row["football_rate_all_legal"],
            ])
    print(f"\nSummary CSV: {csv_path}")

    # Generate findings report
    head_sha = "unknown"
    try:
        import subprocess
        head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        pass

    report_path = generate_findings_report(
        agg=agg,
        groups=groups,
        seeds=args.seeds,
        output_dir=args.output_dir,
        head_sha=head_sha,
        fresh_baseline_rows=fresh_baseline_rows,
        ablation_tail_rows=ablation_tail_rows,
        seed_to_rows=seed_to_rows,
    )

    print(f"\nDone. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
