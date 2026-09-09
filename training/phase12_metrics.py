"""
GMN-Football-3 — Phase 12 Metrics Computer

Runs evaluation with debug_rewards=True to compute:
- pass_accuracy (ground truth)
- pass_attempts (from attempted_passes_left)
- shots/episode (from total_shots_left)
- goal_rate
- reward_variance (std/mean)
- action_entropy
- progress_reward_share
"""
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor


ACTION_NAMES = [
    "IDLE", "LEFT", "RIGHT", "UP", "DOWN",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
    "SHORT_PASS", "LONG_PASS", "HIGH_PASS",
    "SHOT",
    "SPRINT",
    "SLIDE_TACKLE", "INTERCEPT",
    "DIRECTIONAL_PASS", "DRIBBLE", "SKILL"
]

SHOT_ACTIONS = {12}
PASS_ACTIONS = {9, 10, 11}


def compute_entropy(action_counts: list) -> float:
    total = sum(action_counts)
    if total == 0:
        return 0.0
    probs = np.array(action_counts, dtype=np.float64) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def evaluate_with_debug_rewards(checkpoint_path, scenario, num_episodes=100, base_seed=500000, port=5050):
    print(f"\n{'='*60}")
    print(f"PHASE 12 METRICS: {checkpoint_path}")
    print(f"{'='*60}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    obs_dim = checkpoint.get("obs_dim", 127)
    action_dim = checkpoint.get("action_dim", 19)
    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = GMNMultiAgentEnv(
        scenario=scenario,
        auto_start_bridge=True,
        port=port,
        enable_reward_shaping=False,
        debug_rewards=True,
    )
    controllable_agents = list(env.possible_agents)

    rewards_list = []
    goals_list = []
    lengths_list = []
    action_counts_total = [0] * action_dim
    pass_attempts_list = []
    shots_list = []
    progress_rewards = []
    total_rewards = []

    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        obs_dict, _ = env.reset(seed=ep_seed)
        env.reward_components = []
        ep_reward = 0.0
        ep_length = 0
        goal_scored = 0
        ep_pass_attempts = 0
        ep_shots = 0

        while True:
            current_agents = list(env.agents if env.agents else controllable_agents)
            if not current_agents:
                break
            local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
            with torch.no_grad():
                dist = actor(torch.from_numpy(local_obs).float())
                actions = dist.logits.argmax(dim=-1)

            action_dict = {a: int(actions[i].item()) for i, a in enumerate(current_agents)}
            for act in action_dict.values():
                if 0 <= act < action_dim:
                    action_counts_total[act] += 1
                if act in PASS_ACTIONS:
                    ep_pass_attempts += 1
                if act in SHOT_ACTIONS:
                    ep_shots += 1

            obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
            ep_length += 1
            shared_rew = float(rewards[current_agents[0]]) if current_agents and current_agents[0] in rewards else 0.0
            ep_reward += shared_rew

            term = any(terms.values()) if terms else False
            trunc = any(truncs.values()) if truncs else False
            done = term or trunc or not env.agents

            if done:
                score_left = 0
                if infos:
                    for inf in infos.values():
                        if isinstance(inf, dict):
                            score_left = inf.get("score", {}).get("left", 0)
                            break
                if score_left > 0:
                    goal_scored = 1
                break

        # Collect reward components
        ep_components = env.reward_components
        if ep_components:
            ep_progress = sum(c.get("components", {}).get("progress", 0.0) for c in ep_components)
            ep_total = sum(c.get("components", {}).get("total", 0.0) for c in ep_components)
            progress_rewards.append(ep_progress)
            total_rewards.append(ep_total)

        rewards_list.append(ep_reward)
        goals_list.append(goal_scored)
        lengths_list.append(ep_length)
        pass_attempts_list.append(ep_pass_attempts)
        shots_list.append(ep_shots)

    env.close()

    # Compute metrics
    mean_rew = float(np.mean(rewards_list))
    std_rew = float(np.std(rewards_list))
    reward_variance = std_rew / abs(mean_rew) if mean_rew != 0 else float('inf')
    goal_rate = float(np.mean(goals_list)) * 100.0
    mean_length = float(np.mean(lengths_list))
    pass_attempts = float(np.mean(pass_attempts_list))
    shots = float(np.mean(shots_list))
    entropy = compute_entropy(action_counts_total)

    progress_share = 0.0
    if total_rewards and sum(total_rewards) != 0:
        progress_share = sum(progress_rewards) / sum(total_rewards) * 100.0

    print(f"Mean Episode Reward     : {mean_rew:+.4f} ± {std_rew:.4f}")
    print(f"Reward Variance (std/mean): {reward_variance:.4f}")
    print(f"Goal Conversion Rate    : {goal_rate:.1f}% ({sum(goals_list)}/{num_episodes} goals)")
    print(f"Mean Episode Length     : {mean_length:.1f} steps")
    print(f"Pass Attempts/Ep        : {pass_attempts:.2f}")
    print(f"Shots/Episode           : {shots:.2f}")
    print(f"Action Entropy          : {entropy:.4f}")
    print(f"Progress Reward Share   : {progress_share:.1f}%")
    print(f"Action Distribution     :")
    for i, count in enumerate(action_counts_total):
        pct = count / max(1, sum(action_counts_total)) * 100.0
        print(f"   {ACTION_NAMES[i]:20s}: {count:5d} ({pct:5.1f}%)")
    print(f"{'='*60}")

    return {
        "mean_reward": mean_rew,
        "std_reward": std_rew,
        "reward_variance": reward_variance,
        "goal_rate": goal_rate,
        "mean_length": mean_length,
        "pass_attempts_per_episode": pass_attempts,
        "shots_per_episode": shots,
        "action_entropy": entropy,
        "progress_reward_share": progress_share,
        "action_counts": action_counts_total,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=500000)
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    evaluate_with_debug_rewards(
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        num_episodes=args.episodes,
        base_seed=args.seed,
        port=args.port,
    )
