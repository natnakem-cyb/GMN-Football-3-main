"""
GMN-Football-3 — Multi-Agent Policy Evaluation Script
Evaluates trained MAPPO model against the Multi-Agent environment.
"""

import argparse
import hashlib
import os
import sys
import numpy as np
import torch
from datetime import datetime, timezone

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor
from training.mappo_rollout import unwrap_obs
from training.episode_recorder import EpisodeRecorder


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_evaluation_metadata(checkpoint_path: str, num_episodes: int, scenario: str) -> dict:
    return {
        "evaluation_metadata": {
            "git_commit": __import__("subprocess").check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(__file__),
            ).decode().strip(),
            "evaluator_version": "v2_ground_truth_bridge",
            "timestamp_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_file": checkpoint_path,
            "model_sha256": sha256_of(checkpoint_path),
            "episodes": num_episodes,
            "scenario": scenario,
        }
    }


def evaluate_mappo(
    checkpoint_path: str = "training/models/mappo_academy_3_vs_1_with_keeper_best.pt",
    scenario: str = "academy_3_vs_1_with_keeper",
    num_episodes: int = 50,
    deterministic: bool = True,
    save_replay: bool = False,
    save_replay_episodes: int = 1,
    base_seed: int = 500000,
):
    print("==================================================")
    print("GMN-FOOTBALL-3 — MAPPO EVALUATION RUNNER")
    print(f"Model Checkpoint: {checkpoint_path}")
    print(f"Scenario: {scenario} | Episodes: {num_episodes} | Deterministic: {deterministic}")
    if save_replay:
        print(f"Replay Recording: ENABLED (Saving first {save_replay_episodes} episode traces to training/replays/)")
    print("==================================================")

    metadata = build_evaluation_metadata(checkpoint_path, num_episodes, scenario)
    print(f"[Lineage] git_commit={metadata['evaluation_metadata']['git_commit'][:12]} "
          f"evaluator_version={metadata['evaluation_metadata']['evaluator_version']} "
          f"timestamp={metadata['evaluation_metadata']['timestamp_iso']} "
          f"model_sha256={metadata['evaluation_metadata']['model_sha256'][:16]}...")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load model with dynamic obs_dim compatibility
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    inferred_obs_dim = checkpoint["actor"]["net.0.weight"].shape[1] if "net.0.weight" in checkpoint["actor"] else 127
    actor = SharedActor(obs_dim=inferred_obs_dim, action_dim=19, hidden=64)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = GMNMultiAgentEnv(scenario=scenario, auto_start_bridge=True)
    controllable_agents = list(env.possible_agents)

    rewards_list = []
    lengths_list = []
    goals_list = []
    ground_truth_possession = []
    ground_truth_pass_accuracy = []
    ground_truth_shot_accuracy = []
    ground_truth_completed_passes = []
    ground_truth_total_shots = []
    # Split by outcome so pass/shot accuracy is not conflated across
    # scoring vs non-scoring episodes.
    gt_possession_goal = []
    gt_pass_accuracy_goal = []
    gt_shot_accuracy_goal = []
    gt_possession_no_goal = []
    gt_pass_accuracy_no_goal = []
    gt_shot_accuracy_no_goal = []

    for ep in range(1, num_episodes + 1):
        ep_seed = base_seed + ep
        obs_dict, _ = env.reset(seed=ep_seed)
        obs_dict = unwrap_obs(obs_dict)
        ep_reward = 0.0
        ep_length = 0
        goal_scored = 0
        episode_ground_truth = {}

        recorder = None
        if save_replay and ep <= save_replay_episodes:
            recorder = EpisodeRecorder(
                scenario=scenario,
                seed=ep_seed,
                agent_ids=controllable_agents,
                checkpoint=os.path.basename(checkpoint_path),
            )

        while True:
            current_agents = list(env.agents if env.agents else controllable_agents)
            local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)

            with torch.no_grad():
                dist = actor(torch.from_numpy(local_obs).float())
                if deterministic:
                    # Argmax over action logits
                    actions = dist.logits.argmax(dim=-1)
                else:
                    actions = dist.sample()

            action_dict = {a: int(actions[i].item()) for i, a in enumerate(current_agents)}
            obs_dict, rewards, terminations, truncations, infos = env.step(action_dict)
            obs_dict = unwrap_obs(obs_dict)

            shared_rew = float(rewards[current_agents[0]]) if current_agents and current_agents[0] in rewards else 0.0
            ep_reward += shared_rew
            ep_length += 1

            term = any(terminations.values()) if terminations else False
            trunc = any(truncations.values()) if truncations else False
            done = term or trunc

            step_event = None
            step_score = {"left": 0, "right": 0}
            if infos:
                for agent_info in infos.values():
                    if "score" in agent_info:
                        step_score = agent_info["score"]
                    ev = agent_info.get("event")
                    if isinstance(ev, dict) and "type" in ev:
                        step_event = ev["type"]
                    elif isinstance(ev, str):
                        step_event = ev
                    # Capture ground-truth metrics on terminal step
                    if done and "ground_truth" in agent_info:
                        episode_ground_truth = agent_info["ground_truth"]
                    break

            if recorder is not None:
                recorder.record_step(
                    tick=ep_length - 1,
                    actions=action_dict,
                    observations=obs_dict,
                    reward=shared_rew,
                    terminated=term,
                    truncated=trunc,
                    score=step_score,
                    event=step_event,
                )

            if done:
                if step_score.get("left", 0) > 0:
                    goal_scored = 1
                break

        if recorder is not None:
            saved_path = recorder.close()
            print(f"   [Replay Saved] Episode {ep} -> {saved_path}")

        rewards_list.append(ep_reward)
        lengths_list.append(ep_length)
        goals_list.append(goal_scored)

        # Collect ground-truth metrics from episode end
        if episode_ground_truth:
            if episode_ground_truth.get("possession_left_pct") is not None:
                ground_truth_possession.append(episode_ground_truth["possession_left_pct"])
            if episode_ground_truth.get("pass_accuracy") is not None:
                ground_truth_pass_accuracy.append(episode_ground_truth["pass_accuracy"])
            if episode_ground_truth.get("shot_accuracy") is not None:
                ground_truth_shot_accuracy.append(episode_ground_truth["shot_accuracy"])
            if episode_ground_truth.get("completed_passes_left") is not None:
                ground_truth_completed_passes.append(episode_ground_truth["completed_passes_left"])
            if episode_ground_truth.get("total_shots_left") is not None:
                ground_truth_total_shots.append(episode_ground_truth["total_shots_left"])
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

        if ep % 10 == 0 or ep == num_episodes:
            print(
                f"   [Episode {ep:3d}/{num_episodes}] Mean Reward: {np.mean(rewards_list):+.4f} | "
                f"Goal Rate: {np.mean(goals_list)*100.0:5.1f}% | "
                f"Mean Length: {np.mean(lengths_list):.1f}"
            )

    mean_rew = float(np.mean(rewards_list))
    std_rew = float(np.std(rewards_list))
    goal_rate = float(np.mean(goals_list)) * 100.0
    mean_len = float(np.mean(lengths_list))

    # Compute ground-truth aggregates
    gt_possession = float(np.mean(ground_truth_possession)) if ground_truth_possession else None
    gt_pass_accuracy = float(np.mean(ground_truth_pass_accuracy)) if ground_truth_pass_accuracy else None
    gt_shot_accuracy = float(np.mean(ground_truth_shot_accuracy)) if ground_truth_shot_accuracy else None
    gt_completed_passes = float(np.mean(ground_truth_completed_passes)) if ground_truth_completed_passes else None
    gt_total_shots = float(np.mean(ground_truth_total_shots)) if ground_truth_total_shots else None
    # Split by outcome
    gt_pass_accuracy_goal_mean = float(np.mean(gt_pass_accuracy_goal)) if gt_pass_accuracy_goal else None
    gt_shot_accuracy_goal_mean = float(np.mean(gt_shot_accuracy_goal)) if gt_shot_accuracy_goal else None
    gt_pass_accuracy_no_goal_mean = float(np.mean(gt_pass_accuracy_no_goal)) if gt_pass_accuracy_no_goal else None
    gt_shot_accuracy_no_goal_mean = float(np.mean(gt_shot_accuracy_no_goal)) if gt_shot_accuracy_no_goal else None

    print("\n==================================================")
    print("MAPPO EVALUATION RESULTS SUMMARY")
    print("==================================================")
    print(f"Total Episodes Evaluated : {num_episodes}")
    print(f"Mean Episode Reward     : {mean_rew:+.4f} ± {std_rew:.4f}")
    print(f"Goal Conversion Rate    : {goal_rate:.1f}% ({sum(goals_list)}/{num_episodes} goals)")
    print(f"Mean Episode Length     : {mean_len:.1f} steps")
    if gt_possession is not None:
        print(f"Ground-Truth Possession : {gt_possession:.1f}% (left team)")
    if gt_pass_accuracy is not None:
        print(f"Ground-Truth Pass Acc   : {gt_pass_accuracy:.1f}% (all episodes)")
    if gt_shot_accuracy is not None:
        print(f"Ground-Truth Shot Acc   : {gt_shot_accuracy:.1f}% (all episodes)")
    if gt_pass_accuracy_goal_mean is not None:
        print(f"  Pass Acc (goal eps)   : {gt_pass_accuracy_goal_mean:.1f}% ({len(gt_pass_accuracy_goal)} eps)")
    if gt_shot_accuracy_goal_mean is not None:
        print(f"  Shot Acc (goal eps)   : {gt_shot_accuracy_goal_mean:.1f}% ({len(gt_shot_accuracy_goal)} eps)")
    if gt_pass_accuracy_no_goal_mean is not None:
        print(f"  Pass Acc (no-goal eps): {gt_pass_accuracy_no_goal_mean:.1f}% ({len(gt_pass_accuracy_no_goal)} eps)")
    if gt_shot_accuracy_no_goal_mean is not None:
        print(f"  Shot Acc (no-goal eps): {gt_shot_accuracy_no_goal_mean:.1f}% ({len(gt_shot_accuracy_no_goal)} eps)")
    if gt_completed_passes is not None:
        print(f"Completed Passes/Ep     : {gt_completed_passes:.1f}")
    if gt_total_shots is not None:
        print(f"Total Shots/Ep          : {gt_total_shots:.1f}")
    print("==================================================")

    return {
        "mean_reward": mean_rew,
        "std_reward": std_rew,
        "goal_rate": goal_rate,
        "mean_length": mean_len,
        "ground_truth_possession_left_pct": gt_possession,
        "ground_truth_pass_accuracy": gt_pass_accuracy,
        "ground_truth_shot_accuracy": gt_shot_accuracy,
        "ground_truth_completed_passes_ep": gt_completed_passes,
        "ground_truth_total_shots_ep": gt_total_shots,
        "pass_accuracy_goal_episodes": gt_pass_accuracy_goal_mean,
        "shot_accuracy_goal_episodes": gt_shot_accuracy_goal_mean,
        "pass_accuracy_no_goal_episodes": gt_pass_accuracy_no_goal_mean,
        "shot_accuracy_no_goal_episodes": gt_shot_accuracy_no_goal_mean,
        "goal_episodes_count": len(gt_pass_accuracy_goal),
        "no_goal_episodes_count": len(gt_pass_accuracy_no_goal),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="training/models/mappo_academy_3_vs_1_with_keeper_best.pt")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--save-replay", action="store_true", help="Record JSONL episode traces to training/replays/")
    parser.add_argument("--save-replay-episodes", type=int, default=1, help="Max number of replay episodes to record (default: 1)")
    parser.add_argument("--seed", type=int, default=500000, help="Base environment seed")
    args = parser.parse_args()

    evaluate_mappo(
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        save_replay=args.save_replay,
        save_replay_episodes=args.save_replay_episodes,
        base_seed=args.seed,
    )
