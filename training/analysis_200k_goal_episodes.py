"""
Detailed episode analysis for MAPPO 200k run.
Loads best checkpoint, runs deterministic episodes with full per-step logging,
and reports goal-scoring episode details.
"""
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from training.gmn_pettingzoo import GMNMultiAgentEnv, OBSERVATION_DIM
from training.mappo_networks import SharedActor

CHECKPOINT = "training/models/mappo_academy_3_vs_1_with_keeper_best.pt"
SCENARIO = "academy_3_vs_1_with_keeper"
NUM_EPISODES = 30
BASE_SEED = 500000

ACTION_NAMES = [
    "IDLE", "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT",
    "SPRINT", "TACKLE", "PASS", "HIGH_PASS", "SHOT",
    "INTERCEPT", "DEFEND", "SHOT_POWERED", "DRIBBLE",
    "LONG_PASS", "SLIDE_TACKLE", "PRESS", "CLEAR", "LOBBED_PASS"
]

def analyze_episodes():
    print(f"Loading checkpoint: {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    obs_dim = checkpoint.get("obs_dim", 127)
    action_dim = checkpoint.get("action_dim", 19)

    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = GMNMultiAgentEnv(scenario=SCENARIO, auto_start_bridge=True)
    controllable = list(env.possible_agents)

    goal_episodes = []
    all_shots = []

    for ep in range(NUM_EPISODES):
        seed = BASE_SEED + ep * 1009
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
            for i, a in enumerate(current):
                act_int = int(actions[i].item())
                action_dict[a] = act_int
                if act_int == 9:  # SHOT
                    all_shots.append({
                        "episode": ep,
                        "step": steps,
                        "agent": a,
                    })

            obs_dict, rews, terms, truncs, infos = env.step(action_dict)
            steps += 1

            # Extract observation features for the first controllable agent
            obs_vec = obs_dict[current[0]]
            ball_x = float(obs_vec[88])
            ball_y = float(obs_vec[89])
            ball_z = float(obs_vec[90])
            ball_owned = int(np.argmax(obs_vec[94:97]))  # 0=no-one, 1=left, 2=right

            # Right team (opponent) positions: indices 44-65
            right_positions = []
            for i in range(11):
                rx = float(obs_vec[44 + i * 2])
                ry = float(obs_vec[45 + i * 2])
                if rx >= -0.9:  # filter out inactive slots (-1)
                    right_positions.append((rx, ry))

                term = any(terms.values()) if terms else False
                trunc = any(truncs.values()) if truncs else False
            done = term or trunc or not env.agents

            if infos:
                for inf in infos.values():
                    last_info = inf
                    break

            event_type = None
            if last_info.get("event"):
                event_type = last_info["event"].get("type")

            ep_steps.append({
                "step": steps,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "ball_z": ball_z,
                "ball_owned": ball_owned,
                "right_positions": right_positions,
                "actions": {a: ACTION_NAMES[int(action_dict[a])] for a in current},
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

    print(f"\n=== ANALYSIS RESULTS ({NUM_EPISODES} episodes) ===")
    print(f"Goals scored: {len(goal_episodes)} / {NUM_EPISODES}")
    print(f"Goal rate: {len(goal_episodes)/NUM_EPISODES*100:.1f}%")
    print(f"Total shots: {len(all_shots)}")
    print(f"Shots per episode: {len(all_shots)/NUM_EPISODES:.2f}")

    print(f"\n=== GOAL EPISODE SPOT-CHECK (first {min(5, len(goal_episodes))} goals) ===")
    for ge in goal_episodes[:5]:
        print(f"\n--- Episode {ge['episode']} (seed {ge['seed']}) ---")
        print(f"Total steps: {ge['steps']}, Final score: {ge['score_left']}")
        print(f"Final event: {ge['final_event']}")

        # Find the goal event step
        goal_step = None
        for s in reversed(ge['ep_steps']):
            if s['event'] == 'goal':
                goal_step = s
                break

        if goal_step:
            gs = goal_step
            print(f"Goal at step {gs['step']}:")
            print(f"  Ball pos: x={gs['ball_x']:.3f}, y={gs['ball_y']:.3f}, z={gs['ball_z']:.3f}")
            print(f"  Ball owned by: {'Left' if gs['ball_owned']==1 else 'Right' if gs['ball_owned']==2 else 'No-one'}")
            print(f"  Nearby defenders (right team):")
            for rx, ry in gs['right_positions']:
                dist = np.sqrt((rx - gs['ball_x'])**2 + (ry - gs['ball_y'])**2)
                print(f"    x={rx:.3f}, y={ry:.3f}, dist_to_ball={dist:.3f}")
            print(f"  Actions this step: {gs['actions']}")

        # Show shot events in this episode
        shots_in_ep = [s for s in ge['ep_steps'] if s['event'] == 'shot' or 9 in [ACTION_NAMES.index(v) for v in s['actions'].values()]]
        print(f"  Shot events in episode: {len(shots_in_ep)}")
        for s in shots_in_ep[:5]:
            print(f"    Step {s['step']}: ball=({s['ball_x']:.2f},{s['ball_y']:.2f}), actions={s['actions']}, event={s['event']}")

    # Save detailed data
    out_path = "training/analysis_200k_goal_episodes.json"
    with open(out_path, "w") as f:
        json.dump({
            "num_episodes": NUM_EPISODES,
            "num_goals": len(goal_episodes),
            "goal_rate_pct": len(goal_episodes) / NUM_EPISODES * 100,
            "total_shots": len(all_shots),
            "shots_per_ep": len(all_shots) / NUM_EPISODES,
            "goal_episodes": [
                {
                    "episode": g["episode"],
                    "seed": g["seed"],
                    "steps": g["steps"],
                    "score_left": g["score_left"],
                    "final_event": g["final_event"],
                    "goal_step": next((s for s in reversed(g["ep_steps"]) if s["event"] == "goal"), None),
                    "shot_steps": [s for s in g["ep_steps"] if s["event"] == "shot"],
                }
                for g in goal_episodes
            ],
        }, f, indent=2, default=str)
    print(f"\nDetailed analysis saved to: {out_path}")


if __name__ == "__main__":
    analyze_episodes()
