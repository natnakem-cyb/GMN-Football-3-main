"""
Actor-Optimization / Advantage-Consumption Audit — Task 1: Alignment Check

Performs fresh rollouts from a checkpoint, then:
1. Computes probe-style GAE advantages (shared, from first-agent reward)
2. Computes training-style GAE advantages (per-agent, from each agent's own reward)
3. Runs ppo_update and captures normalized advantages
4. Compares sign and magnitude per action family

Outputs:
  - training/results/actor_optimization_audit_task1.json
  - training/results/actor_optimization_audit_task1_summary.csv
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor, CentralizedCritic
from training.mappo_rollout import unwrap_obs, unwrap_masks, _mask_matrix, compute_gae
from training.mappo_update import ppo_update

ACTION_NAMES = [
    "IDLE", "LEFT", "RIGHT", "UP", "DOWN",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
    "LONG_PASS", "HIGH_PASS", "SHORT_PASS",
    "SHOT", "SPRINT",
    "RELEASE_DIRECTION", "RELEASE_SPRINT",
    "SLIDING", "DRIBBLE", "RELEASE_DRIBBLE",
]
TACKLE_ACTION = 16
SHOT_ACTIONS = {12}
PASS_ACTIONS = {9, 10, 11}
MOVE_ACTIONS = {1, 2, 3, 4, 5, 6, 7, 8}


def classify_action(action_idx: int) -> str:
    if action_idx == TACKLE_ACTION:
        return "TACKLE"
    if action_idx in SHOT_ACTIONS:
        return "SHOT"
    if action_idx in PASS_ACTIONS:
        return "PASS"
    if action_idx in MOVE_ACTIONS:
        return "MOVE"
    if action_idx == 0:
        return "IDLE"
    return "OTHER"


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    obs_dim = ckpt.get("obs_dim", 127)
    action_dim = ckpt.get("action_dim", 19)
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


def collect_episodes(checkpoint_path: str, num_episodes: int = 20, base_seed: int = 700000):
    actor, critic, obs_dim, action_dim, ckpt_timesteps = load_checkpoint(checkpoint_path)

    env = GMNMultiAgentEnv(
        scenario="academy_3_vs_1_with_keeper_onball",
        auto_start_bridge=True,
        port=5052,
    )
    controllable_agents = list(env.possible_agents)

    all_local_obs = []
    all_actions = []
    all_logprobs = []
    all_action_masks = []
    all_values = []
    all_shared_rewards = []
    all_per_agent_rewards = []
    all_dones = []
    all_tick_actions = []
    all_tick_event_types = []
    episode_lengths = []

    for ep_idx in range(num_episodes):
        ep_seed = base_seed + ep_idx * 137
        obs_dict, _ = env.reset(seed=ep_seed)
        current_ep_masks = unwrap_masks(obs_dict)
        obs_dict = unwrap_obs(obs_dict)

        ep_local_obs = []
        ep_actions = []
        ep_logprobs = []
        ep_action_masks = []
        ep_values = []
        ep_rewards = []
        ep_per_agent_rewards = []
        ep_dones = []
        ep_terminated = []
        tick_actions = []
        tick_event_types = []

        ep_reward = 0.0
        ep_length = 0

        while True:
            current_agents = list(env.agents if env.agents else controllable_agents)
            local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
            mask_matrix = _mask_matrix(current_ep_masks, current_agents)
            global_state = local_obs.flatten().astype(np.float32)

            with torch.no_grad():
                obs_tensor = torch.from_numpy(local_obs).float()
                mask_tensor = torch.tensor(mask_matrix, dtype=torch.bool)
                dist = actor(obs_tensor, mask_tensor)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                value = float(critic(torch.from_numpy(global_state).float().unsqueeze(0)).item())

            action_dict = {a: int(actions[i].item()) for i, a in enumerate(current_agents)}
            obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
            obs_dict = unwrap_obs(obs_dict)
            current_ep_masks = unwrap_masks(obs_dict)

            per_agent_rewards = np.array([rewards[a] for a in current_agents], dtype=np.float32)
            shared_reward = float(per_agent_rewards.mean())
            ep_reward += shared_reward
            ep_length += 1

            term = any(terms.values()) if terms else False
            trunc = any(truncs.values()) if truncs else False

            event_type = None
            if infos:
                for inf in infos.values():
                    event_type = inf.get("event", {}).get("type") if isinstance(inf.get("event"), dict) else None
                    break

            ep_local_obs.append(local_obs)
            ep_actions.append(np.array([action_dict[a] for a in current_agents], dtype=np.int64))
            ep_logprobs.append(log_probs.numpy())
            ep_action_masks.append(mask_matrix)
            ep_values.append(value)
            ep_rewards.append(shared_reward)
            ep_per_agent_rewards.append(per_agent_rewards)
            ep_dones.append(term)
            tick_actions.append(action_dict)
            tick_event_types.append(event_type)

            if term or trunc or not env.agents:
                break

        episode_lengths.append(ep_length)
        all_local_obs.extend(ep_local_obs)
        all_actions.extend(ep_actions)
        all_logprobs.extend(ep_logprobs)
        all_action_masks.extend(ep_action_masks)
        all_values.extend(ep_values)
        all_shared_rewards.extend(ep_rewards)
        all_per_agent_rewards.extend(ep_per_agent_rewards)
        all_dones.extend(ep_dones)
        all_tick_actions.extend(tick_actions)
        all_tick_event_types.extend(tick_event_types)

    env.close()

    T_total = len(all_values)
    num_agents = len(controllable_agents)
    local_obs_arr = np.stack(all_local_obs, axis=0).astype(np.float32)
    actions_arr = np.stack(all_actions, axis=0).astype(np.int64)
    logprobs_arr = np.stack(all_logprobs, axis=0).astype(np.float32)
    action_masks_arr = np.stack(all_action_masks, axis=0).astype(np.int8)
    values_arr = np.array(all_values, dtype=np.float32)
    shared_rewards_arr = np.array(all_shared_rewards, dtype=np.float32)
    per_agent_rewards_arr = np.stack(all_per_agent_rewards, axis=0).astype(np.float32)
    dones_arr = np.array(all_dones, dtype=bool)

    # Bootstrap for last step
    bootstrap_value = 0.0
    if not dones_arr[-1]:
        with torch.no_grad():
            bootstrap_value = float(critic(torch.from_numpy(local_obs_arr[-1]).float().unsqueeze(0)).item())

    # Probe-style GAE: shared rewards, shared advantages (matching stochastic_rollout_probe.py)
    advantages_shared, returns_shared = compute_gae(
        rewards=shared_rewards_arr,
        values=values_arr,
        dones=dones_arr,
        gamma=0.99,
        lam=0.95,
        bootstrap_value=bootstrap_value,
        next_local_obs=local_obs_arr[-1],
        critic=critic,
    )

    # Training-style GAE: per-agent rewards, per-agent advantages (matching train_mappo.py)
    advantages_per_agent, returns_per_agent = compute_gae(
        rewards=shared_rewards_arr,
        values=values_arr,
        dones=dones_arr,
        gamma=0.99,
        lam=0.95,
        bootstrap_value=bootstrap_value,
        next_local_obs=local_obs_arr[-1],
        critic=critic,
        per_agent_rewards=per_agent_rewards_arr,
    )

    return {
        "local_obs": local_obs_arr,
        "actions": actions_arr,
        "logprobs": logprobs_arr,
        "action_masks": action_masks_arr,
        "values": values_arr,
        "shared_rewards": shared_rewards_arr,
        "per_agent_rewards": per_agent_rewards_arr,
        "dones": dones_arr,
        "advantages_shared": advantages_shared,
        "returns_shared": returns_shared,
        "advantages_per_agent": advantages_per_agent,
        "returns_per_agent": returns_per_agent,
        "bootstrap_value": bootstrap_value,
        "tick_actions": all_tick_actions,
        "tick_event_types": all_tick_event_types,
        "T_total": T_total,
        "num_agents": num_agents,
        "episode_lengths": episode_lengths,
    }, actor, critic


def main():
    parser = argparse.ArgumentParser(description="Task 1: Alignment check")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="training/results")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=700000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Collecting {args.num_episodes} episodes from {args.checkpoint}...")
    data, actor, critic = collect_episodes(args.checkpoint, num_episodes=args.num_episodes, base_seed=args.base_seed)

    T_total = data["T_total"]
    num_agents = data["num_agents"]

    # Run ppo_update with PER-AGENT advantages (matching training exactly)
    buffer = {
        "local_obs": data["local_obs"],
        "actions": data["actions"],
        "logprobs": data["logprobs"],
        "action_masks": data["action_masks"],
        "values": data["values"],
        "rewards": data["shared_rewards"],
        "dones": data["dones"],
        "per_agent_rewards": data["per_agent_rewards"],
    }

    metrics = ppo_update(
        actor=actor,
        critic=critic,
        actor_opt=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic_opt=torch.optim.Adam(critic.parameters(), lr=3e-4),
        buffer=buffer,
        advantages=data["advantages_per_agent"],
        returns=data["returns_per_agent"],
        clip_range=0.15,
        n_epochs=1,
        batch_size=T_total,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        onball_football_entropy_bonus=0.0,
    )

    # Replicate normalization from ppo_update (per-agent path: flat_advantages = advantages.reshape(-1))
    flat_advantages_raw = data["advantages_per_agent"].reshape(-1)
    adv_mean = float(flat_advantages_raw.mean())
    adv_std = float(flat_advantages_raw.std())
    flat_advantages_normalized = (flat_advantages_raw - adv_mean) / (adv_std + 1e-8)

    # Build per-transition records
    records = []
    for t in range(T_total):
        for a in range(num_agents):
            idx = t * num_agents + a
            act_int = int(data["actions"][t, a])
            # Map per-agent advantage to this tick's shared event type
            event_type = data["tick_event_types"][t] if t < len(data["tick_event_types"]) else None
            records.append({
                "tick": t,
                "agent": a,
                "action_idx": act_int,
                "action_name": ACTION_NAMES[act_int],
                "action_family": classify_action(act_int),
                "event_type": event_type,
                "probe_A_shared": float(data["advantages_shared"][t]),
                "train_A_per_agent_raw": float(data["advantages_per_agent"][t, a]),
                "train_A_normalized": float(flat_advantages_normalized[idx]),
                "V": float(data["values"][t]),
                "reward_shared": float(data["shared_rewards"][t]),
                "reward_this_agent": float(data["per_agent_rewards"][t, a]),
                "done": bool(data["dones"][t]),
            })

    # Per-action-family summary
    family_stats = {}
    for rec in records:
        fam = rec["action_family"]
        if fam not in family_stats:
            family_stats[fam] = {
                "probe_A_shared": [],
                "train_A_per_agent_raw": [],
                "train_A_norm": [],
                "n": 0,
            }
        family_stats[fam]["probe_A_shared"].append(rec["probe_A_shared"])
        family_stats[fam]["train_A_per_agent_raw"].append(rec["train_A_per_agent_raw"])
        family_stats[fam]["train_A_norm"].append(rec["train_A_normalized"])
        family_stats[fam]["n"] += 1

    summary_rows = []
    for fam in ["TACKLE", "PASS", "SHOT", "MOVE", "IDLE", "OTHER"]:
        if fam not in family_stats:
            continue
        s = family_stats[fam]
        summary_rows.append({
            "action_family": fam,
            "n": s["n"],
            "probe_A_shared_mean": float(np.mean(s["probe_A_shared"])),
            "train_A_per_agent_mean": float(np.mean(s["train_A_per_agent_raw"])),
            "train_A_normalized_mean": float(np.mean(s["train_A_norm"])),
        })

    # Check for disconnect: sign mismatch between probe A and train A for TACKLE/PASS/SHOT
    disconnect_flags = []
    for fam in ["TACKLE", "PASS", "SHOT"]:
        if fam not in family_stats:
            continue
        s = family_stats[fam]
        probe_sign = float(np.sign(np.mean(s["probe_A_shared"])))
        train_sign = float(np.sign(np.mean(s["train_A_per_agent_raw"])))
        if probe_sign != train_sign and probe_sign != 0 and train_sign != 0:
            disconnect_flags.append(f"{fam}: probe sign={probe_sign:+.0f}, train sign={train_sign:+.0f}")

    # Also check if per-agent rewards differ from shared rewards (another potential disconnect)
    reward_diffs = np.abs(data["per_agent_rewards"].mean(axis=1) - data["shared_rewards"])
    max_reward_diff = float(reward_diffs.max())
    avg_reward_diff = float(reward_diffs.mean())

    result = {
        "timestamp_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoint": args.checkpoint,
        "num_episodes": args.num_episodes,
        "base_seed": args.base_seed,
        "total_ticks": T_total,
        "num_agents": num_agents,
        "total_transitions": len(records),
        "advantage_normalization": {
            "mean": adv_mean,
            "std": adv_std,
            "note": "ppo_update normalizes flat_advantages = (A - mean) / (std + 1e-8)"
        },
        "reward_alignment": {
            "max_abs_diff_shared_vs_per_agent_mean": max_reward_diff,
            "mean_abs_diff": avg_reward_diff,
            "note": "If >0, per-agent rewards differ from shared reward, causing probe/train advantage mismatch"
        },
        "disconnect_found": len(disconnect_flags) > 0,
        "disconnect_flags": disconnect_flags,
        "per_family_summary": summary_rows,
    }

    json_path = os.path.join(args.output_dir, "actor_optimization_audit_task1.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    csv_path = os.path.join(args.output_dir, "actor_optimization_audit_task1_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "action_family", "n", "probe_A_shared_mean", "train_A_per_agent_mean", "train_A_normalized_mean"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nTask 1 Alignment Check Complete")
    print(f"  Episodes: {args.num_episodes} | Total ticks: {T_total} | Agents: {num_agents}")
    print(f"  Total transitions: {len(records)}")
    print(f"  Advantage normalization: mean={adv_mean:+.6f}, std={adv_std:.6f}")
    print(f"  Reward diff (shared vs per-agent mean): max={max_reward_diff:.6f}, avg={avg_reward_diff:.6f}")
    print(f"  Disconnect flags: {disconnect_flags if disconnect_flags else 'None — probe A and train-consumed A are sign-aligned'}")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")

    print("\nPer-Action-Family Summary:")
    print(f"{'Family':<10} {'n':>6} {'Probe A':>12} {'Train A/agent':>14} {'Train A (norm)':>16}")
    for row in summary_rows:
        print(f"{row['action_family']:<10} {row['n']:>6} {row['probe_A_shared_mean']:>+12.4f} {row['train_A_per_agent_mean']:>+14.4f} {row['train_A_normalized_mean']:>+16.4f}")


if __name__ == "__main__":
    main()
