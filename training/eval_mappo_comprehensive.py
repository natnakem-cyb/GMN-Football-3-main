"""
GMN-Football-3 — Comprehensive MAPPO Checkpoint Evaluation
Evaluates trained MAPPO checkpoints with full behavioral metrics:
- Goal rate, win/draw/loss, goals conceded
- Shots, shots on target, shot accuracy
- Turnovers, possession, action distribution
- Episode length, reward breakdown
"""

import argparse
import os
import sys
import json
import math
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor
from training.football_metrics import FootballMetricsTracker, EpisodeMetrics, compute_distribution


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

    for ep in range(num_episodes):
        ep_seed = base_seed + ep * 1009
        obs_dict, _ = env.reset(seed=ep_seed)

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
            for i, a in enumerate(current_agents):
                act_int = int(actions[i].item())
                action_dict[a] = act_int
                left_action_indices.append(act_int)

                if act_int in SHOT_ACTIONS:
                    shot_actions += 1
                if act_int in PASS_ACTIONS:
                    pass_actions += 1
                if act_int in TACKLE_ACTIONS:
                    tackle_actions += 1

            obs_dict, rews, terms, truncs, infos = env.step(action_dict)
            ep_length += 1

            shared_rew = float(rews[current_agents[0]]) if current_agents and current_agents[0] in rews else 0.0
            ep_reward += shared_rew

            # Extract ball state from first agent's observation
            current_obs = obs_dict[current_agents[0]]
            ball_pos, ownership = extract_ball_from_obs(current_obs)
            owner_team = ownership_to_team(ownership)

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
                break

        # End of episode - record metrics
        final_obs = obs_dict.get(controllable_agents[0], initial_obs)
        ball_pos, _ = extract_ball_from_obs(final_obs)
        final_score = last_info.get("score", {"left": 0, "right": 0})

        # Build final_stats from tracked actions since env doesn't expose full stats
        final_stats = {
            "passes": {"left": pass_actions},
            "completedPasses": {"left": pass_actions},  # Approximated
            "shots": {"left": shot_actions},
            "shotsOnTarget": {"left": shot_actions if goal_scored else 0},  # Approximated
            "tackles": {"left": tackle_actions},
            "interceptions": {"left": 0},
            "fouls": {"left": 0},
            "yellowCards": {"left": 0},
            "redCards": {"left": 0},
            "possession": {"left": 50.0},  # Will be computed by tracker
        }

        ep_metrics = tracker.end_episode(
            final_score=final_score,
            final_stats=final_stats,
            final_ball_pos=ball_pos,
            max_ball_progress_x=0.0,
        )
        all_episode_metrics.append(ep_metrics)

        rewards_list.append(ep_reward)
        lengths_list.append(ep_length)
        goals_list.append(goal_scored)
        shot_actions_list.append(shot_actions)
        pass_actions_list.append(pass_actions)
        tackle_actions_list.append(tackle_actions)

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

    # Add additional computed metrics
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
    print(f"Episode Length       : {agg['mean_episode_length']:.1f} ± {agg['std_episode_length']:.1f} steps")
    print(f"Mean Reward          : {agg['cumulative_reward']['mean']:.4f} ± {agg['cumulative_reward']['std']:.4f}")
    print("=" * 60)

    # Save detailed results
    results_path = os.path.join(
        os.path.dirname(checkpoint_path),
        f"comprehensive_eval_{os.path.splitext(os.path.basename(checkpoint_path))[0]}.json"
    )
    with open(results_path, "w") as f:
        json.dump(agg, f, indent=2)
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
    args = parser.parse_args()

    evaluate_checkpoint_comprehensive(
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        base_seed=args.seed,
        bridge_port=args.port,
    )


if __name__ == "__main__":
    main()
