"""
Trace just the key steps around the goal to see keeper behavior.
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

checkpoint = torch.load(CHECKPOINT, map_location="cpu")
actor = SharedActor(obs_dim=checkpoint.get("obs_dim", 127), action_dim=checkpoint.get("action_dim", 19), hidden=64)
actor.load_state_dict(checkpoint["actor"])
actor.eval()

env = GMNMultiAgentEnv(scenario=SCENARIO, auto_start_bridge=True)
controllable = list(env.possible_agents)

seed = 500000
obs_dict, _ = env.reset(seed=seed)
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

    obs_dict, rews, terms, truncs, infos = env.step(action_dict)
    steps += 1

    obs_vec = obs_dict[current[0]]
    ball_x = float(obs_vec[88])
    ball_y = float(obs_vec[89])
    ball_z = float(obs_vec[90])
    ball_vx = float(obs_vec[91]) / 50.0
    ball_vy = float(obs_vec[92]) / 50.0
    ball_vz = float(obs_vec[93]) / 50.0
    ball_speed = np.sqrt(ball_vx**2 + ball_vy**2 + ball_vz**2)
    ball_owned = int(np.argmax(obs_vec[94:97]))

    term = any(terms.values()) if terms else False
    trunc = any(truncs.values()) if truncs else False
    done = term or trunc or not env.agents

    if infos:
        for inf in infos.values():
            last_info = inf
            break

    # Only print steps near the goal or where ball speed is high
    if ball_x > 0.6 or ball_z > 0.01 or steps < 5 or steps > 150:
        print(f"Step {steps:3d}: ball=({ball_x:+.3f},{ball_y:+.3f},{ball_z:+.3f}) vel=({ball_vx:+.4f},{ball_vy:+.4f},{ball_vz:+.4f}) speed={ball_speed:.4f} owned={ball_owned} actions={action_dict}")

    if last_info.get("event"):
        event_type = last_info["event"].get("type")
        if event_type:
            print(f"  *** EVENT: {event_type} ***")

score_left = last_info.get("score", {}).get("left", 0)
print(f"\nFinal: steps={steps}, score_left={score_left}")
env.close()
