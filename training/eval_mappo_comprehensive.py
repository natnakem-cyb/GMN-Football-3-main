"""
GMN-Football-3 — Comprehensive MAPPO Checkpoint Evaluation
Evaluates trained MAPPO checkpoints with full behavioral metrics:
- Goal rate, win/draw/loss, goals conceded
- Shots, shots on target, shot accuracy
- Turnovers, possession, action distribution
- Episode length, reward breakdown
"""

import argparse
import hashlib
import os
import sys
import json
import math
import numpy as np
import torch
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor
from training.mappo_rollout import unwrap_obs
from training.football_metrics import FootballMetricsTracker, EpisodeMetrics, compute_distribution


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_evaluation_metadata(checkpoint_path: str, num_episodes: int, scenario: str) -> Dict[str, Any]:
    return {
        "evaluation_metadata": {
            "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode().strip(),
            "evaluator_version": "v2_ground_truth_bridge",
            "timestamp_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_file": checkpoint_path,
            "model_sha256": sha256_of(checkpoint_path),
            "episodes": num_episodes,
            "scenario": scenario,
        }
    }


# Action name mapping (matches ActionType in src/types/football.ts)
ACTION_NAMES = [
    "IDLE", "LEFT", "RIGHT", "UP", "DOWN",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
    "SHORT_PASS", "LONG_PASS", "HIGH_PASS",
    "SHOT",
    "SPRINT",
    "SLIDE_TACKLE", "INTERCEPT",
    "DIRECTIONAL_PASS", "DRIBBLE", "SKILL"
]

# Shot actions
SHOT_ACTIONS = {12}  # SHOT
PASS_ACTIONS = {9, 10, 11}  # SHORT_PASS, LONG_PASS, HIGH_PASS
SPRINT_ACTION = 13
DRIBBLE_ACTION = 17
TACKLE_ACTIONS = {14, 15}  # SLIDE_TACKLE, INTERCEPT


def extract_ball_from_obs(obs: np.ndarray) -> Dict[str, float]:
    """Extract ball position and ownership from observation vector."""
    # Offset 88 (len 3): Ball (x, y, z) position
    # Offset 91 (len 3): Ball (x, y, z) movement direction
    # Offset 94 (len 3): Ball ownership, one-hot: [no-one, left, right]
    ball_pos = {
        "x": float(obs[88]) if len(obs) > 88 else 0.0,
        "y": float(obs[89]) if len(obs) > 89 else 0.0,
        "z": float(obs[90]) if len(obs) > 90 else 0.0,
    }
    ownership = [0.0, 0.0, 0.0]
    if len(obs) > 96:
        ownership = [float(obs[94]), float(obs[95]), float(obs[96])]
    return ball_pos, ownership


def ownership_to_team(ownership: List[float]) -> Optional[str]:
    """Convert ownership one-hot to team string."""
    if len(ownership) != 3:
        return None
    max_idx = int(np.argmax(ownership))
    if max_idx == 0:
        return None  # no-one
    elif max_idx == 1:
        return "left"
    else:
        return "right"


def evaluate_checkpoint_comprehensive(
    checkpoint_path: str,
    scenario: str = "academy_3_vs_1_with_keeper",
    num_episodes: int = 50,
    deterministic: bool = True,
    base_seed: int = 500000,
    bridge_port: int = 5050,
    force: bool = False,
) -> Dict[str, Any]:
    """Run comprehensive evaluation on a MAPPO checkpoint."""

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 60)
    print("COMPREHENSIVE MAPPO EVALUATION")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Scenario   : {scenario}")
    print(f"Episodes   : {num_episodes}")
    print(f"Deterministic: {deterministic}")
    print(f"Base Seed  : {base_seed}")
    print("=" * 60)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    obs_dim = checkpoint.get("obs_dim", 127)
    action_dim = checkpoint.get("action_dim", 19)

    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    # Create environment
    env = GMNMultiAgentEnv(
        scenario=scenario,
        auto_start_bridge=True,
        port=bridge_port,
    )
    controllable_agents = list(env.possible_agents)

    tracker = FootballMetricsTracker()
    all_episode_metrics: List[EpisodeMetrics] = []

    rewards_list = []
    goals_list = []
    lengths_list = []
    shot_actions_list = []
    pass_actions_list = []
    tackle_actions_list = []

    # Ground-truth metrics from EPISODE_STATS WebSocket frame
    gt_possession_list = []
    gt_pass_accuracy_list = []
    gt_shot_accuracy_list = []
    gt_completed_passes_list = []
    gt_total_shots_list = []
    # Split by outcome so pass/shot accuracy is not conflated across
    # scoring vs non-scoring episodes.
    gt_possession_goal = []
    gt_pass_accuracy_goal = []
    gt_shot_accuracy_goal = []
    gt_possession_no_goal = []
    gt_pass_accuracy_no_goal = []
    gt_shot_accuracy_no_goal = []

    for ep in range(num_episodes):
        ep_seed = base_seed + ep * 1009
        obs_dict, _ = env.reset(seed=ep_seed)
        obs_dict = unwrap_obs(obs_dict)

        # Initialize episode tracking
        initial_obs = obs_dict[controllable_agents[0]]
        ball_pos, ownership = extract_ball_from_obs(initial_obs)
        tracker.start_episode(scenario=scenario, seed=ep_seed, ball_pos=ball_pos)

        ep_reward = 0.0
        ep_length = 0
        goal_scored = 0
        shot_actions = 0
        pass_actions = 0
        tackle_actions = 0
        last_info = {}
        episode_ground_truth = {}

        while True:
            current_agents = list(env.agents if env.agents else controllable_agents)
            local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)

            with torch.no_grad():
                dist = actor(torch.from_numpy(local_obs).float())
                if deterministic:
                    actions = dist.logits.argmax(dim=-1)
                else:
                    actions = dist.sample()

            action_dict = {}
            left_action_indices = []
            right_action_indices = []
            for i, a in enumerate(current_agents):
                act_int = int(actions[i].item())
                action_dict[a] = act_int
                if a.startswith("left_"):
                    left_action_indices.append(act_int)
                elif a.startswith("right_"):
                    right_action_indices.append(act_int)

                if act_int in SHOT_ACTIONS:
                    shot_actions += 1
                if act_int in PASS_ACTIONS:
                    pass_actions += 1
                if act_int in TACKLE_ACTIONS:
                    tackle_actions += 1

            obs_dict, rews, terms, truncs, infos = env.step(action_dict)
            obs_dict = unwrap_obs(obs_dict)
            ep_length += 1

            shared_rew = float(rews[current_agents[0]]) if current_agents and current_agents[0] in rews else 0.0
            ep_reward += shared_rew

            # Extract ball state from first agent's observation
            current_obs = obs_dict[current_agents[0]]
            ball_pos, ownership = extract_ball_from_obs(current_obs)
            owner_team = ownership_to_team(ownership)

            # Record engine events for ground-truth shot/pass accuracy
            event_type = None
            if infos:
                for inf in infos.values():
                    event = inf.get("event", {})
                    if isinstance(event, dict):
                        event_type = event.get("type")
                    break
            if event_type:
                tracker.record_event(event_type)

            tracker.record_tick(
                left_action_indices=left_action_indices,
                step_reward=shared_rew,
                ball_pos=ball_pos,
                owner_team=owner_team,
            )

            term = any(terms.values()) if terms else False
            trunc = any(truncs.values()) if truncs else False
            done = term or trunc or not env.agents

            if infos:
                for inf in infos.values():
                    last_info = inf
                    break

            if done:
                score_left = last_info.get("score", {}).get("left", 0)
                event = last_info.get("event", {})
                is_goal = score_left > 0 or (isinstance(event, dict) and event.get("type") == "goal")
                if is_goal:
                    goal_scored = 1
                # Capture ground-truth metrics from EPISODE_STATS frame
                for inf in infos.values():
                    if isinstance(inf, dict) and "ground_truth" in inf:
                        episode_ground_truth = inf["ground_truth"]
                        break
                break

        # End of episode - record metrics
        final_obs = obs_dict.get(controllable_agents[0], initial_obs)
        ball_pos, _ = extract_ball_from_obs(final_obs)
        final_score = last_info.get("score", {"left": 0, "right": 0})

        # Use ground-truth stats from FootballMetricsTracker
        ep_metrics = tracker.end_episode(
            final_score=final_score,
            final_stats={},
            final_ball_pos=ball_pos,
            max_ball_progress_x=0.0,
        )
        all_episode_metrics.append(ep_metrics)

        rewards_list.append(ep_reward)
        lengths_list.append(ep_length)
        goals_list.append(goal_scored)
        shot_actions_list.append(ep_metrics.shots_total)
        pass_actions_list.append(ep_metrics.passes_attempted)
        tackle_actions_list.append(tackle_actions)

        # Collect ground-truth metrics from EPISODE_STATS frame
        if episode_ground_truth:
            if episode_ground_truth.get("possession_left_pct") is not None:
                gt_possession_list.append(episode_ground_truth["possession_left_pct"])
            if episode_ground_truth.get("pass_accuracy") is not None:
                gt_pass_accuracy_list.append(episode_ground_truth["pass_accuracy"])
            if episode_ground_truth.get("shot_accuracy") is not None:
                gt_shot_accuracy_list.append(episode_ground_truth["shot_accuracy"])
            if episode_ground_truth.get("completed_passes_left") is not None:
                gt_completed_passes_list.append(episode_ground_truth["completed_passes_left"])
            if episode_ground_truth.get("total_shots_left") is not None:
                gt_total_shots_list.append(episode_ground_truth["total_shots_left"])
            # Split by outcome
            if goal_scored:
                if episode_ground_truth.get("possession_left_pct") is not None:
                    gt_possession_goal.append(episode_ground_truth["possession_left_pct"])
                if episode_ground_truth.get("pass_accuracy") is not None:
                    gt_pass_accuracy_goal.append(episode_ground_truth["pass_accuracy"])
                if episode_ground_truth.get("shot_accuracy") is not None:
                    gt_shot_accuracy_goal.append(episode_ground_truth["shot_accuracy"])
            else:
                if episode_ground_truth.get("possession_left_pct") is not None:
                    gt_possession_no_goal.append(episode_ground_truth["possession_left_pct"])
                if episode_ground_truth.get("pass_accuracy") is not None:
                    gt_pass_accuracy_no_goal.append(episode_ground_truth["pass_accuracy"])
                if episode_ground_truth.get("shot_accuracy") is not None:
                    gt_shot_accuracy_no_goal.append(episode_ground_truth["shot_accuracy"])

        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            print(
                f"   [Episode {ep + 1:3d}/{num_episodes}] "
                f"Mean Reward: {np.mean(rewards_list):+.4f} | "
                f"Goal Rate: {np.mean(goals_list) * 100.0:5.1f}% | "
                f"Mean Length: {np.mean(lengths_list):.1f}"
            )

    env.close()

    # Aggregate metrics
    agg = tracker.aggregate(policy_name=os.path.basename(checkpoint_path), scenario=scenario)

    # Override with ground-truth metrics from EPISODE_STATS WebSocket frame
    if gt_possession_list:
        agg["possession_rate_pct"] = {
            "mean": float(np.mean(gt_possession_list)),
            "std": float(np.std(gt_possession_list)),
            "median": float(np.median(gt_possession_list)),
            "min": float(np.min(gt_possession_list)),
            "max": float(np.max(gt_possession_list)),
            "ci95_low": float(np.percentile(gt_possession_list, 2.5)),
            "ci95_high": float(np.percentile(gt_possession_list, 97.5)),
        }
    if gt_pass_accuracy_list:
        agg["pass_completion_rate_pct"] = {
            "mean": float(np.mean(gt_pass_accuracy_list)) * 100.0,
            "std": float(np.std(gt_pass_accuracy_list)) * 100.0,
            "median": float(np.median(gt_pass_accuracy_list)) * 100.0,
            "min": float(np.min(gt_pass_accuracy_list)) * 100.0,
            "max": float(np.max(gt_pass_accuracy_list)) * 100.0,
            "ci95_low": float(np.percentile(gt_pass_accuracy_list, 2.5)) * 100.0,
            "ci95_high": float(np.percentile(gt_pass_accuracy_list, 97.5)) * 100.0,
        }
    if gt_shot_accuracy_list:
        agg["shot_accuracy_pct"] = {
            "mean": float(np.mean(gt_shot_accuracy_list)) * 100.0,
            "std": float(np.std(gt_shot_accuracy_list)) * 100.0,
            "median": float(np.median(gt_shot_accuracy_list)) * 100.0,
            "min": float(np.min(gt_shot_accuracy_list)) * 100.0,
            "max": float(np.max(gt_shot_accuracy_list)) * 100.0,
            "ci95_low": float(np.percentile(gt_shot_accuracy_list, 2.5)) * 100.0,
            "ci95_high": float(np.percentile(gt_shot_accuracy_list, 97.5)) * 100.0,
        }
    if gt_completed_passes_list:
        agg["passes_completed_per_episode"] = {
            "mean": float(np.mean(gt_completed_passes_list)),
            "std": float(np.std(gt_completed_passes_list)),
            "median": float(np.median(gt_completed_passes_list)),
            "min": float(np.min(gt_completed_passes_list)),
            "max": float(np.max(gt_completed_passes_list)),
            "ci95_low": float(np.percentile(gt_completed_passes_list, 2.5)),
            "ci95_high": float(np.percentile(gt_completed_passes_list, 97.5)),
        }
    if gt_total_shots_list:
        agg["shots_per_episode"] = {
            "mean": float(np.mean(gt_total_shots_list)),
            "std": float(np.std(gt_total_shots_list)),
            "median": float(np.median(gt_total_shots_list)),
            "min": float(np.min(gt_total_shots_list)),
            "max": float(np.max(gt_total_shots_list)),
            "ci95_low": float(np.percentile(gt_total_shots_list, 2.5)),
            "ci95_high": float(np.percentile(gt_total_shots_list, 97.5)),
        }

    # Add additional computed metrics from ground-truth tracker
    agg["shots_per_episode_mean"] = float(np.mean(shot_actions_list)) if shot_actions_list else 0.0
    agg["passes_per_episode_mean"] = float(np.mean(pass_actions_list)) if pass_actions_list else 0.0
    agg["tackles_per_episode_mean"] = float(np.mean(tackle_actions_list)) if tackle_actions_list else 0.0
    agg["shot_to_goal_pct"] = (
        (sum(goals_list) / max(1, sum(shot_actions_list))) * 100.0
        if sum(shot_actions_list) > 0
        else 0.0
    )
    agg["mean_episode_length"] = float(np.mean(lengths_list)) if lengths_list else 0.0
    agg["std_episode_length"] = float(np.std(lengths_list)) if lengths_list else 0.0
    agg["mean_episode_duration_seconds"] = float(np.mean(lengths_list)) / 60.0 if lengths_list else 0.0
    # Outcome-split pass/shot accuracy
    agg["pass_accuracy_goal_episodes"] = {
        "mean": float(np.mean(gt_pass_accuracy_goal)) * 100.0 if gt_pass_accuracy_goal else 0.0,
        "count": len(gt_pass_accuracy_goal),
    }
    agg["shot_accuracy_goal_episodes"] = {
        "mean": float(np.mean(gt_shot_accuracy_goal)) * 100.0 if gt_shot_accuracy_goal else 0.0,
        "count": len(gt_shot_accuracy_goal),
    }
    agg["pass_accuracy_no_goal_episodes"] = {
        "mean": float(np.mean(gt_pass_accuracy_no_goal)) * 100.0 if gt_pass_accuracy_no_goal else 0.0,
        "count": len(gt_pass_accuracy_no_goal),
    }
    agg["shot_accuracy_no_goal_episodes"] = {
        "mean": float(np.mean(gt_shot_accuracy_no_goal)) * 100.0 if gt_shot_accuracy_no_goal else 0.0,
        "count": len(gt_shot_accuracy_no_goal),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Episodes       : {num_episodes}")
    print(f"Goal Rate            : {agg['success_rate_pct']['mean']:.1f}% ({sum(goals_list)}/{num_episodes})")
    print(f"Win Rate             : {agg['win_rate_pct']['mean']:.1f}%")
    print(f"Draw Rate            : {agg['draw_rate_pct']['mean']:.1f}%")
    print(f"Loss Rate            : {agg['loss_rate_pct']['mean']:.1f}%")
    print(f"Goals Scored/Ep      : {agg['goals_scored_per_episode']['mean']:.2f} ± {agg['goals_scored_per_episode']['std']:.2f}")
    print(f"Goals Conceded/Ep    : {agg['goals_conceded_per_episode']['mean']:.2f} ± {agg['goals_conceded_per_episode']['std']:.2f}")
    print(f"Goal Difference/Ep   : {agg['goal_difference_per_episode']['mean']:.2f} ± {agg['goal_difference_per_episode']['std']:.2f}")
    print(f"Shots/Episode        : {agg['shots_per_episode_mean']:.2f}")
    print(f"Shot Accuracy        : {agg['shot_accuracy_pct']['mean']:.1f}%")
    print(f"Shot-to-Goal %       : {agg['shot_to_goal_pct']:.1f}%")
    print(f"Passes/Episode       : {agg['passes_per_episode_mean']:.2f}")
    print(f"Pass Completion      : {agg['pass_completion_rate_pct']['mean']:.1f}%")
    print(f"Tackles/Episode      : {agg['tackles_per_episode_mean']:.2f}")
    print(f"Turnovers Conceded/Ep: {agg['turnovers_conceded_per_episode']['mean']:.2f}")
    print(f"Possession %         : {agg['possession_rate_pct']['mean']:.1f}%")
    print(f"Episode Length       : {agg['mean_episode_length']:.1f} ± {agg['std_episode_length']:.1f} steps ({agg['mean_episode_duration_seconds']:.2f}s)")
    print(f"Mean Reward          : {agg['cumulative_reward']['mean']:.4f} ± {agg['cumulative_reward']['std']:.4f}")
    print(f"Pass Acc (goal eps)  : {agg['pass_accuracy_goal_episodes']['mean']:.1f}% ({agg['pass_accuracy_goal_episodes']['count']} eps)")
    print(f"Shot Acc (goal eps)  : {agg['shot_accuracy_goal_episodes']['mean']:.1f}% ({agg['shot_accuracy_goal_episodes']['count']} eps)")
    print(f"Pass Acc (no-goal eps): {agg['pass_accuracy_no_goal_episodes']['mean']:.1f}% ({agg['pass_accuracy_no_goal_episodes']['count']} eps)")
    print(f"Shot Acc (no-goal eps): {agg['shot_accuracy_no_goal_episodes']['mean']:.1f}% ({agg['shot_accuracy_no_goal_episodes']['count']} eps)")
    print("=" * 60)

    # Save detailed results
    results_path = os.path.join(
        os.path.dirname(checkpoint_path),
        f"comprehensive_eval_{os.path.splitext(os.path.basename(checkpoint_path))[0]}.json"
    )

    if os.path.exists(results_path) and not force:
        raise FileExistsError(
            f"Evaluation results already exist at {results_path}. "
            "Use --force to overwrite."
        )

    output_payload = build_evaluation_metadata(checkpoint_path, num_episodes, scenario)
    output_payload.update(agg)

    with open(results_path, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")

    return agg


def main():
    parser = argparse.ArgumentParser(description="Comprehensive MAPPO Checkpoint Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MAPPO checkpoint")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=500000)
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--force", action="store_true", help="Overwrite existing evaluation results")
    args = parser.parse_args()

    evaluate_checkpoint_comprehensive(
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        base_seed=args.seed,
        bridge_port=args.port,
        force=args.force,
    )


if __name__ == "__main__":
    main()
