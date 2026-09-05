"""
Corrected physics/action trace for MAPPO 200k goal episodes.
Action names are taken EXACTLY from src/engine/ActionMapping.ts.
Ball velocity is de-scaled by /50 to show real physics units.
"""
import os
import sys
import json
import torch
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.abspath("."))

from training.gmn_pettingzoo import GMNMultiAgentEnv, OBSERVATION_DIM
from training.mappo_networks import SharedActor

CHECKPOINT = "training/models/mappo_academy_3_vs_1_with_keeper_best.pt"
SCENARIO = "academy_3_vs_1_with_keeper"

# EXACT mapping from src/engine/ActionMapping.ts lines 10-30
ACTION_NAMES = {
    0: "action_idle",
    1: "action_left",
    2: "action_top_left",
    3: "action_top",
    4: "action_top_right",
    5: "action_right",
    6: "action_bottom_right",
    7: "action_bottom",
    8: "action_bottom_left",
    9: "action_long_pass",
    10: "action_high_pass",
    11: "action_short_pass",
    12: "action_shot",
    13: "action_sprint",
    14: "action_release_direction",
    15: "action_release_sprint",
    16: "action_sliding",
    17: "action_dribble",
    18: "action_release_dribble",
}

# Real kick formula from src/engine/Physics.ts line 146:
#   kickSpeed = 0.9 + (player.stats.kickPower / 100) * 2.7 * min(1, max(0.1, power))
# Max possible = 0.9 + 1.0 * 2.7 * 1.0 = 3.6

def trace_goal_episodes(num_episodes=30, base_seed=500000):
    print(f"Loading checkpoint: {CHECKPOINT}")
    print(f"Action mapping source: src/engine/ActionMapping.ts")
    print(f"Physics formula: kickSpeed = 0.9 + (kickPower/100)*2.7*min(1,max(0.1,power))")
    print(f"Physics max kickSpeed: 3.6 units/step")
    print(f"Observation velocity scaling: x50 (real = obs_value / 50)")
    print()

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    actor = SharedActor(obs_dim=checkpoint.get("obs_dim", 127), action_dim=checkpoint.get("action_dim", 19), hidden=64)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = GMNMultiAgentEnv(scenario=SCENARIO, auto_start_bridge=True)
    controllable = list(env.possible_agents)

    goal_episodes = []
    all_shots = []
    all_speeds = []

    for ep in range(num_episodes):
        seed = base_seed + ep * 1009
        obs_dict, _ = env.reset(seed=seed)
        ep_steps = []
        steps = 0
        done = False
        last_info = {}

        while not done and steps < 600:
            current = list(env.agents if env.agents else controllable)
            local_obs = np.stack([obs_dict[a] for a in current], axis=0).astype(np.float32)

            with torch.no_grad():
                dist = actor(torch.from_numpy(local_obs).float())
                actions = dist.logits.argmax(dim=-1)

            action_dict = {}
            action_names_this_step = {}
            for i, a in enumerate(current):
                act_int = int(actions[i].item())
                action_dict[a] = act_int
                action_names_this_step[a] = ACTION_NAMES.get(act_int, f"UNKNOWN({act_int})")
                if act_int == 12:  # SHOT (real index from ActionMapping.ts)
                    all_shots.append({"episode": ep, "step": steps, "agent": a})

            obs_dict, rews, terms, truncs, infos = env.step(action_dict)
            steps += 1

            obs_vec = obs_dict[current[0]]
            # Observation-scale values (as stored in the 127-float vector)
            ball_x_obs = float(obs_vec[88])
            ball_y_obs = float(obs_vec[89])
            ball_z_obs = float(obs_vec[90])
            ball_vx_obs = float(obs_vec[91])
            ball_vy_obs = float(obs_vec[92])
            ball_vz_obs = float(obs_vec[93])
            ball_owned = int(np.argmax(obs_vec[94:97]))

            # REAL physics values (undo the x50 scaling from ObservationEncoder.ts lines 119-120)
            ball_x = ball_x_obs
            ball_y = ball_y_obs
            ball_z = ball_z_obs
            ball_vx = ball_vx_obs / 50.0
            ball_vy = ball_vy_obs / 50.0
            ball_vz = ball_vz_obs / 50.0
            ball_speed = np.sqrt(ball_vx**2 + ball_vy**2 + ball_vz**2)
            all_speeds.append(ball_speed)

            term = any(terms.values()) if terms else False
            trunc = any(truncs.values()) if truncs else False
            done = term or trunc or not env.agents

            event_type = None
            if infos:
                for inf in infos.values():
                    last_info = inf
                    break
            if last_info.get("event"):
                event_type = last_info["event"].get("type")

            ep_steps.append({
                "step": steps,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "ball_z": ball_z,
                "ball_vx": ball_vx,
                "ball_vy": ball_vy,
                "ball_vz": ball_vz,
                "ball_speed": ball_speed,
                "ball_owned": ball_owned,
                "actions": action_names_this_step,
                "action_indices": action_dict,
                "event": event_type,
                "reward": float(rews[current[0]]) if current and current[0] in rews else 0.0,
            })

        score_left = last_info.get("score", {}).get("left", 0)
        event = last_info.get("event", {})
        is_goal = score_left > 0 or (isinstance(event, dict) and event.get("type") == "goal")

        if is_goal:
            goal_episodes.append({
                "episode": ep,
                "seed": seed,
                "steps": steps,
                "score_left": score_left,
                "final_event": event_type,
                "ep_steps": ep_steps,
            })

    env.close()

    print(f"=== ANALYSIS RESULTS ({num_episodes} episodes) ===")
    print(f"Goals scored: {len(goal_episodes)} / {num_episodes}")
    print(f"Goal rate: {len(goal_episodes)/num_episodes*100:.1f}%")
    print(f"Total shots (action 12): {len(all_shots)}")
    print(f"Shots per episode: {len(all_shots)/num_episodes:.2f}")
    if all_speeds:
        print(f"Max ball speed observed: {max(all_speeds):.4f} units/step")
        print(f"Within Physics.ts max (3.6): {max(all_speeds) <= 3.6}")

    print(f"\n=== GOAL EPISODE SPOT-CHECK (first {min(5, len(goal_episodes))} goals) ===")
    for ge in goal_episodes[:5]:
        print(f"\n--- Episode {ge['episode']} (seed {ge['seed']}) ---")
        print(f"Total steps: {ge['steps']}, Final score: {ge['score_left']}")

        goal_step = None
        for s in reversed(ge['ep_steps']):
            if s['event'] == 'goal':
                goal_step = s
                break

        if goal_step:
            gs = goal_step
            print(f"Goal at step {gs['step']}:")
            print(f"  Ball pos: x={gs['ball_x']:.3f}, y={gs['ball_y']:.3f}, z={gs['ball_z']:.3f}")
            print(f"  Ball vel: vx={gs['ball_vx']:.4f}, vy={gs['ball_vy']:.4f}, vz={gs['ball_vz']:.4f}, speed={gs['ball_speed']:.4f}")
            print(f"  Ball owned by: {gs['ball_owned']}")
            print(f"  Actions: {gs['actions']}")
            print(f"  Action indices: {gs['action_indices']}")

        shots_in_ep = [s for s in ge['ep_steps'] if s['action_indices'] and any(v == 12 for v in s['action_indices'].values())]
        print(f"  SHOT actions (index 12) in episode: {len(shots_in_ep)}")
        for s in shots_in_ep[:5]:
            print(f"    Step {s['step']}: ball=({s['ball_x']:.3f},{s['ball_y']:.3f},{s['ball_z']:.3f}) speed={s['ball_speed']:.4f}, actions={s['actions']}, event={s['event']}")

        cnt = Counter([ACTION_NAMES.get(a, f"UNKNOWN({a})") for s in ge["ep_steps"] for a in s["action_indices"].values()])
        print(f"  Action distribution (top 5): {dict(cnt.most_common(5))}")

    out_path = "training/analysis_200k_goal_episodes_corrected.json"
    with open(out_path, "w") as f:
        json.dump({
            "source_checkpoint": CHECKPOINT,
            "action_mapping_source": "src/engine/ActionMapping.ts",
            "action_name_table": ACTION_NAMES,
            "physics_formula": "kickSpeed = 0.9 + (kickPower/100)*2.7*min(1,max(0.1,power))",
            "physics_max_kickSpeed": 3.6,
            "velocity_observation_scaling": 50,
            "num_episodes": num_episodes,
            "num_goals": len(goal_episodes),
            "goal_rate_pct": len(goal_episodes) / num_episodes * 100,
            "total_shots": len(all_shots),
            "shots_per_ep": len(all_shots) / num_episodes,
            "max_ball_speed_observed": float(max(all_speeds)) if all_speeds else 0.0,
            "within_physics_limits": bool(max(all_speeds) <= 3.6) if all_speeds else None,
            "goal_episodes": [
                {
                    "episode": g["episode"],
                    "seed": g["seed"],
                    "steps": g["steps"],
                    "score_left": g["score_left"],
                    "final_event": g["final_event"],
                    "goal_step": next((s for s in reversed(g["ep_steps"]) if s["event"] == "goal"), None),
                    "shot_steps": [s for s in g["ep_steps"] if s["action_indices"] and any(v == 12 for v in s["action_indices"].values())],
                    "action_distribution": dict(Counter([ACTION_NAMES.get(a, f"UNKNOWN({a})") for s in g["ep_steps"] for a in s["action_indices"].values()])),
                }
                for g in goal_episodes
            ],
        }, f, indent=2, default=str)
    print(f"\nCorrected analysis saved to: {out_path}")


if __name__ == "__main__":
    trace_goal_episodes()
