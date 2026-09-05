"""
Very short trace to get clean debug output.
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

while not done and steps < 65:
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

    term = any(terms.values()) if terms else False
    trunc = any(truncs.values()) if truncs else False
    done = term or trunc or not env.agents

    if infos:
        for inf in infos.values():
            last_info = inf
            break

    if steps >= 48:
        print(f"Step {steps:3d}: ball=({ball_x:+.3f},{ball_y:+.3f},{ball_z:+.3f}) actions={action_dict}")

score_left = last_info.get("score", {}).get("left", 0)
print(f"\nFinal: steps={steps}, score_left={score_left}")
env.close()
