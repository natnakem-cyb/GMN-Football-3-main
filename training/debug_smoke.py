"""
Debug smoke test: print progress to see where it hangs.
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

print("Loading checkpoint...")
checkpoint = torch.load(CHECKPOINT, map_location="cpu")
actor = SharedActor(obs_dim=checkpoint.get("obs_dim", 127), action_dim=checkpoint.get("action_dim", 19), hidden=64)
actor.load_state_dict(checkpoint["actor"])
actor.eval()
print("Checkpoint loaded.")

print("Creating environment...")
env = GMNMultiAgentEnv(scenario=SCENARIO, auto_start_bridge=True)
controllable = list(env.possible_agents)
print(f"Controllable agents: {controllable}")

print("Resetting environment...")
obs_dict, _ = env.reset(seed=500000)
print(f"Reset complete. Agents: {env.agents}")

steps = 0
done = False
last_info = {}

while not done and steps < 10:
    print(f"Step {steps}...")
    current = list(env.agents if env.agents else controllable)
    local_obs = np.stack([obs_dict[a] for a in current], axis=0).astype(np.float32)

    with torch.no_grad():
        dist = actor(torch.from_numpy(local_obs).float())
        actions = dist.logits.argmax(dim=-1)

    action_dict = {}
    for i, a in enumerate(current):
        action_dict[a] = int(actions[i].item())

    print(f"  Actions: {action_dict}")
    obs_dict, rews, terms, truncs, infos = env.step(action_dict)
    steps += 1

    term = any(terms.values()) if terms else False
    trunc = any(truncs.values()) if truncs else False
    done = term or trunc or not env.agents

    if infos:
        for inf in infos.values():
            last_info = inf
            break

print(f"Done. Steps: {steps}, Done: {done}")
env.close()
