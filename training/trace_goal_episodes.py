"""
GMN-Football-3 — Episode-level goal tracing for deployed MAPPO checkpoint.
Runs deterministic episodes and logs detailed state for every goal scored.
"""

import sys
import os
import json
import hashlib
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor

# Deployed checkpoint
CHECKPOINT_PATH = "training/models/mappo_academy_3_vs_1_with_keeper_seed44_best.pt"
SCENARIO = "academy_3_vs_1_with_keeper"
NUM_EPISODES = 50
DETERMINISTIC = True
BASE_SEED = 500000

# Action mapping from ActionMapping.ts
ACTION_NAMES = [
    "action_idle",               # 0
    "action_left",               # 1
    "action_top_left",           # 2
    "action_top",                # 3
    "action_top_right",          # 4
    "action_right",              # 5
    "action_bottom_right",       # 6
    "action_bottom",             # 7
    "action_bottom_left",        # 8
    "action_long_pass",          # 9
    "action_high_pass",          # 10
    "action_short_pass",         # 11
    "action_shot",               # 12
    "action_sprint",             # 13
    "action_release_direction",  # 14
    "action_release_sprint",     # 15
    "action_sliding",            # 16
    "action_dribble",            # 17
    "action_release_dribble",    # 18
]

# Ball velocity scale factor (ObservationEncoder line 122)
BALL_VELOCITY_SCALE = 50.0

# Player velocity scale factor (ObservationEncoder lines 55, 94)
PLAYER_VELOCITY_SCALE = 50.0

# Observation layout from ObservationEncoder.ts:
# 0..21 (22): Left team player (x, y) positions, 11 players
# 22..43 (22): Left team player (x, y) movement direction * 50
# 44..65 (22): Right team player (x, y) positions, 11 players
# 66..87 (22): Right team player (x, y) movement direction * 50
# 88..90 (3): Ball (x, y, z) position
# 91..93 (3): Ball (x, y, z) movement direction * 50
# 94..96 (3): Ball ownership, one-hot: [no-one, left, right]
# 97..107 (11): Active player, one-hot over 11 players
# 108..114 (7): game_mode, one-hot
# 115..126 (12): Agent's assigned role one-hot


def decode_observation(obs):
    """Decode flat observation vector into structured state."""
    result = {
        "left_positions": [],
        "left_velocities": [],
        "right_positions": [],
        "right_velocities": [],
        "ball_position": [],
        "ball_velocity_raw": [],
        "ball_velocity_actual": [],
        "ball_ownership": [],
        "active_player": [],
        "game_mode": [],
        "role_onehot": [],
    }

    # Left positions (0..21)
    for i in range(11):
        result["left_positions"].append([float(obs[2 * i]), float(obs[2 * i + 1])])

    # Left velocities (22..43) - scaled by 50
    for i in range(11):
        raw_vx = float(obs[22 + 2 * i])
        raw_vy = float(obs[22 + 2 * i + 1])
        result["left_velocities"].append({
            "raw": [raw_vx, raw_vy],
            "actual": [raw_vx / PLAYER_VELOCITY_SCALE, raw_vy / PLAYER_VELOCITY_SCALE],
        })

    # Right positions (44..65)
    for i in range(11):
        result["right_positions"].append([float(obs[44 + 2 * i]), float(obs[44 + 2 * i + 1])])

    # Right velocities (66..87) - scaled by 50
    for i in range(11):
        raw_vx = float(obs[66 + 2 * i])
        raw_vy = float(obs[66 + 2 * i + 1])
        result["right_velocities"].append({
            "raw": [raw_vx, raw_vy],
            "actual": [raw_vx / PLAYER_VELOCITY_SCALE, raw_vy / PLAYER_VELOCITY_SCALE],
        })

    # Ball position (88..90)
    result["ball_position"] = [float(obs[88]), float(obs[89]), float(obs[90])]

    # Ball velocity (91..93) - scaled by 50
    raw_bvx = float(obs[91])
    raw_bvy = float(obs[92])
    raw_bvz = float(obs[93])
    result["ball_velocity_raw"] = [raw_bvx, raw_bvy, raw_bvz]
    result["ball_velocity_actual"] = [
        raw_bvx / BALL_VELOCITY_SCALE,
        raw_bvy / BALL_VELOCITY_SCALE,
        raw_bvz / BALL_VELOCITY_SCALE,
    ]

    # Ball ownership (94..96)
    result["ball_ownership"] = [float(obs[94]), float(obs[95]), float(obs[96])]

    # Active player (97..107)
    result["active_player"] = [float(x) for x in obs[97:108]]

    # Game mode (108..114)
    result["game_mode"] = [float(x) for x in obs[108:115]]

    # Role one-hot (115..126)
    result["role_onehot"] = [float(x) for x in obs[115:127]]

    return result


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    obs_dim = checkpoint.get("obs_dim", 127)
    action_dim = checkpoint.get("action_dim", 19)
    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor, obs_dim, action_dim


def run_traced_episodes():
    actor, obs_dim, action_dim = load_checkpoint(CHECKPOINT_PATH)
    env = GMNMultiAgentEnv(scenario=SCENARIO, auto_start_bridge=True)
    controllable_agents = list(env.possible_agents)

    goal_episodes = []
    total_goals = 0

    for ep in range(NUM_EPISODES):
        seed = BASE_SEED + ep * 1009
        obs_dict, _ = env.reset(seed=seed)
        episode_actions = []
        goal_scored = False
        episode_info = {
            "episode": ep,
            "seed": seed,
            "steps": 0,
            "reward": 0.0,
            "goal": False,
            "actions": [],
            "goal_step": None,
            "goal_event": None,
            "goal_score": None,
            "obs_at_goal": None,
            "decoded_at_goal": None,
            "obs_at_shot": [],
            "decoded_at_shot": [],
            "all_shot_actions": [],
            "all_dribble_actions": [],
            "all_pass_actions": [],
        }

        for step in range(600):
            current_agents = list(env.agents if env.agents else controllable_agents)
            local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)

            with torch.no_grad():
                dist = actor(torch.from_numpy(local_obs).float())
                if DETERMINISTIC:
                    actions_tensor = dist.logits.argmax(dim=-1)
                else:
                    actions_tensor = dist.sample()

            action_dict = {}
            for i, a in enumerate(current_agents):
                act_int = int(actions_tensor[i].item())
                action_dict[a] = act_int
                episode_actions.append({
                    "step": step,
                    "agent": a,
                    "action_idx": act_int,
                    "action_name": ACTION_NAMES[act_int],
                })
                # Track shot/dribble/pass actions
                if act_int == 12:
                    episode_info["all_shot_actions"].append({"step": step, "agent": a})
                    # Capture observation at shot moment
                    episode_info["obs_at_shot"].append(local_obs[0].copy().tolist())
                    episode_info["decoded_at_shot"].append(decode_observation(local_obs[0]))
                elif act_int == 17:
                    episode_info["all_dribble_actions"].append({"step": step, "agent": a})
                elif act_int in [9, 10, 11]:
                    episode_info["all_pass_actions"].append({"step": step, "agent": a})

            # Store observation before stepping
            obs_before_step = local_obs[0].copy()

            obs_dict, rews, terms, truncs, infos = env.step(action_dict)
            episode_info["steps"] = step + 1
            episode_info["reward"] += float(rews[current_agents[0]]) if current_agents and current_agents[0] in rews else 0.0

            term = any(terms.values()) if terms else False
            trunc = any(truncs.values()) if truncs else False
            done = term or trunc or not env.agents

            # Check for goal in infos
            if infos:
                for inf in infos.values():
                    score = inf.get("score", {})
                    event = inf.get("event", {})
                    if score.get("left", 0) > 0 or (isinstance(event, dict) and event.get("type") == "goal"):
                        goal_scored = True
                        episode_info["goal"] = True
                        episode_info["goal_step"] = step
                        episode_info["goal_event"] = event
                        episode_info["goal_score"] = score
                        episode_info["obs_at_goal"] = obs_before_step.tolist()
                        episode_info["decoded_at_goal"] = decode_observation(obs_before_step)
                        episode_info["actions"] = episode_actions.copy()
                        break
                if goal_scored:
                    break

            if done:
                break

        if goal_scored:
            total_goals += 1
            goal_episodes.append(episode_info)
            if len(goal_episodes) >= 10:
                break

    env.close()
    print(f"Total episodes scanned: {min(NUM_EPISODES, ep + 1)}")
    print(f"Goals found: {total_goals}")
    print(f"Goal episodes traced: {len(goal_episodes)}")
    return goal_episodes


def print_goal_trace(goal_episodes):
    print("\n" + "=" * 80)
    print("GOAL EPISODE TRACE — seed44_best deterministic checkpoint")
    print("=" * 80)

    for idx, ep in enumerate(goal_episodes):
        print(f"\n--- Goal {idx + 1} ---")
        print(f"Episode: {ep['episode']} | Seed: {ep['seed']} | Steps: {ep['steps']}")
        print(f"Final Reward: {ep['reward']:.4f}")
        print(f"Goal at step: {ep.get('goal_step', 'unknown')}")
        print(f"Goal event: {ep.get('goal_event', {})}")
        print(f"Score: {ep.get('goal_score', {})}")

        # Summary of all shot/dribble/pass actions in entire episode
        print(f"\nALL Shot actions in episode: {len(ep['all_shot_actions'])} "
              f"at steps {[a['step'] for a in ep['all_shot_actions']]}")
        print(f"ALL Dribble actions in episode: {len(ep['all_dribble_actions'])} "
              f"at steps {[a['step'] for a in ep['all_dribble_actions']]}")
        print(f"ALL Pass actions in episode: {len(ep['all_pass_actions'])} "
              f"at steps {[a['step'] for a in ep['all_pass_actions']]}")

        # Print last 15 actions before goal
        actions = ep.get("actions", [])
        if actions:
            print("\nAction sequence (last 15 before goal):")
            for a in actions[-15:]:
                print(f"  Step {a['step']:3d} | {a['agent']:12s} | {a['action_idx']:2d} | {a['action_name']}")

        # Print decoded observation at goal
        decoded = ep.get("decoded_at_goal")
        if decoded:
            print(f"\nState BEFORE goal step (observation at step {ep.get('goal_step', '?') - 1}):")
            print(f"  Ball Position: ({decoded['ball_position'][0]:.3f}, "
                  f"{decoded['ball_position'][1]:.3f}, {decoded['ball_position'][2]:.3f})")
            print(f"  Ball Velocity RAW (×{BALL_VELOCITY_SCALE}): "
                  f"({decoded['ball_velocity_raw'][0]:.4f}, "
                  f"{decoded['ball_velocity_raw'][1]:.4f}, {decoded['ball_velocity_raw'][2]:.4f})")
            print(f"  Ball Velocity ACTUAL: "
                  f"({decoded['ball_velocity_actual'][0]:.4f}, "
                  f"{decoded['ball_velocity_actual'][1]:.4f}, "
                  f"{decoded['ball_velocity_actual'][2]:.4f})")
            print(f"  Ball Ownership: {decoded['ball_ownership']} (no-one, left, right)")

            # Find active player
            active_idx = decoded["active_player"].index(1.0) if 1.0 in decoded["active_player"] else -1
            print(f"  Active Player Index: {active_idx}")

            # Print left team positions and velocities
            print("\n  Left Team (attacking right goal):")
            for i, (pos, vel) in enumerate(zip(decoded["left_positions"], decoded["left_velocities"])):
                if pos[0] >= 0:  # Not inactive (-1, -1)
                    print(f"    Player {i:2d}: Pos=({pos[0]:.3f}, {pos[1]:.3f}) | "
                          f"Vel=({vel['actual'][0]:.4f}, {vel['actual'][1]:.4f})")

            # Print right team positions (defenders + GK)
            print("\n  Right Team (defending):")
            for i, pos in enumerate(decoded["right_positions"]):
                if pos[0] >= 0:  # Not inactive
                    print(f"    Player {i:2d}: Pos=({pos[0]:.3f}, {pos[1]:.3f})")

        # Print observation at shot moment
        if ep.get("decoded_at_shot"):
            print(f"\nState at SHOT moment(s):")
            for i, (shot_action, decoded_shot) in enumerate(zip(ep["all_shot_actions"], ep["decoded_at_shot"])):
                print(f"\n  Shot {i+1} at step {shot_action['step']} by {shot_action['agent']}:")
                print(f"    Ball Position: ({decoded_shot['ball_position'][0]:.3f}, "
                      f"{decoded_shot['ball_position'][1]:.3f}, {decoded_shot['ball_position'][2]:.3f})")
                print(f"    Ball Velocity RAW (×{BALL_VELOCITY_SCALE}): "
                      f"({decoded_shot['ball_velocity_raw'][0]:.4f}, "
                      f"{decoded_shot['ball_velocity_raw'][1]:.4f}, "
                      f"{decoded_shot['ball_velocity_raw'][2]:.4f})")
                print(f"    Ball Velocity ACTUAL: "
                      f"({decoded_shot['ball_velocity_actual'][0]:.4f}, "
                      f"{decoded_shot['ball_velocity_actual'][1]:.4f}, "
                      f"{decoded_shot['ball_velocity_actual'][2]:.4f})")
                print(f"    Ball Ownership: {decoded_shot['ball_ownership']}")
                print(f"    Active Player Index: {decoded_shot['active_player'].index(1.0) if 1.0 in decoded_shot['active_player'] else -1}")

        # Check if goal likely came from shot vs dribble
        if ep["all_shot_actions"]:
            last_shot = ep["all_shot_actions"][-1]
            print(f"\n  >> Goal likely from SHOT action at step {last_shot['step']} by {last_shot['agent']}")
            print(f"     Time from shot to goal: {ep['goal_step'] - last_shot['step']} steps")
        elif ep["all_dribble_actions"] and not ep["all_shot_actions"]:
            print(f"\n  >> Goal likely from DRIBBLE sequence (no shot in episode)")
        elif not ep["all_shot_actions"] and not ep["all_dribble_actions"]:
            print(f"\n  >> WARNING: No shot or dribble actions in entire episode!")
        else:
            print(f"\n  >> Goal from other action sequence")

        print()


if __name__ == "__main__":
    goal_episodes = run_traced_episodes()
    print_goal_trace(goal_episodes)

    # Save raw trace for inspection
    with open("training/results/goal_trace_seed44_best.json", "w") as f:
        json.dump(goal_episodes, f, indent=2, default=str)
    print(f"\nTrace saved to: training/results/goal_trace_seed44_best.json")
