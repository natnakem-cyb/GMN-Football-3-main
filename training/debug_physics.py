"""
Physics debug: check if ball moves on its own without player action,
and trace the exact goal event in Episode 0.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor

CHECKPOINT = "training/models/mappo_academy_3_vs_1_with_keeper_best.pt"
SCENARIO = "academy_3_vs_1_with_keeper"

def physics_debug():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    actor = SharedActor(obs_dim=checkpoint.get("obs_dim", 127), action_dim=checkpoint.get("action_dim", 19), hidden=64)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = GMNMultiAgentEnv(scenario=SCENARIO, auto_start_bridge=True)
    controllable = list(env.possible_agents)

    # Episode 0 is a goal episode
    seed = 500000
    obs_dict, _ = env.reset(seed=seed)

    print(f"=== Episode 0 (seed {seed}) - Physics Trace ===")
    print(f"Controllable agents: {controllable}")

    steps = 0
    done = False
    last_info = {}

    while not done and steps < 200:
        current = list(env.agents if env.agents else controllable)
        local_obs = np.stack([obs_dict[a] for a in current], axis=0).astype(np.float32)

        with torch.no_grad():
            dist = actor(torch.from_numpy(local_obs).float())
            actions = dist.logits.argmax(dim=-1)

        action_dict = {}
        for i, a in enumerate(current):
            action_dict[a] = int(actions[i].item())

        # Log every 10 steps, and all steps near goal
        obs_vec = obs_dict[current[0]]
        ball_x = float(obs_vec[88])
        ball_y = float(obs_vec[89])
        ball_z = float(obs_vec[90])
        ball_vx = float(obs_vec[91])
        ball_vy = float(obs_vec[92])
        ball_vz = float(obs_vec[93])
        ball_owned = int(np.argmax(obs_vec[94:97]))

        should_log = steps < 10 or steps > 140 or abs(ball_x) > 0.7

        if should_log:
            print(f"Step {steps:3d}: ball=({ball_x:+.3f},{ball_y:+.3f},{ball_z:+.3f}) vel=({ball_vx:+.3f},{ball_vy:+.3f},{ball_vz:+.3f}) owned={ball_owned} actions={action_dict}")

        obs_dict, rews, terms, truncs, infos = env.step(action_dict)
        steps += 1

        term = any(terms.values()) if terms else False
        trunc = any(truncs.values()) if truncs else False
        done = term or trunc or not env.agents

        if infos:
            for inf in infos.values():
                last_info = inf
                break

    score_left = last_info.get("score", {}).get("left", 0)
    event = last_info.get("event", {})
    print(f"\nFinal: steps={steps}, score_left={score_left}, event={event}")
    env.close()

if __name__ == "__main__":
    physics_debug()
