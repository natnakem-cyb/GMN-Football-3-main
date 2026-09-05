"""
Deep debug: track ball position and game events step-by-step for goal episodes.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from training.gmn_pettingzoo import GMNMultiAgentEnv, OBSERVATION_DIM
from training.mappo_networks import SharedActor

CHECKPOINT = "training/models/mappo_academy_3_vs_1_with_keeper_best.pt"
SCENARIO = "academy_3_vs_1_with_keeper"

ACTION_NAMES = [
    "IDLE", "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT",
    "SPRINT", "TACKLE", "PASS", "HIGH_PASS", "SHOT",
    "INTERCEPT", "DEFEND", "SHOT_POWERED", "DRIBBLE",
    "LONG_PASS", "SLIDE_TACKLE", "PRESS", "CLEAR", "LOBBED_PASS"
]

def deep_debug():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    actor = SharedActor(obs_dim=checkpoint.get("obs_dim", 127), action_dim=checkpoint.get("action_dim", 19), hidden=64)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = GMNMultiAgentEnv(scenario=SCENARIO, auto_start_bridge=True)
    controllable = list(env.possible_agents)

    # Run episodes and find goal episodes
    for ep in range(5):
        seed = 500000 + ep * 1009
        obs_dict, _ = env.reset(seed=seed)
        steps = 0
        done = False
        last_info = {}
        trajectory = []

        while not done and steps < 600:
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
            ball_owned = int(np.argmax(obs_vec[94:97]))

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

            trajectory.append({
                "step": steps,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "ball_z": ball_z,
                "ball_owned": ball_owned,
                "actions": action_dict,
                "event": event_type,
                "reward": float(rews[current[0]]) if current and current[0] in rews else 0.0,
            })

        score_left = last_info.get("score", {}).get("left", 0)
        print(f"\n=== Episode {ep} (seed {seed}) ===")
        print(f"Steps: {steps}, Score left: {score_left}")
        print(f"Action dist: ", end="")
        from collections import Counter
        all_acts = []
        for t in trajectory:
            for a in t["actions"].values():
                all_acts.append(a)
        cnt = Counter(all_acts)
        print({ACTION_NAMES[k]: v for k, v in cnt.most_common(10)})

        # Find key events
        for t in trajectory:
            if t["event"] in ("goal", "shot", "shot_saved", "shot_missed"):
                print(f"  EVENT at step {t['step']}: {t['event']}, ball=({t['ball_x']:.3f},{t['ball_y']:.3f},{t['ball_z']:.3f}), owned={t['ball_owned']}, actions={t['actions']}")
            if abs(t["ball_x"]) > 0.85 and t["ball_z"] < 0.1:
                print(f"  NEAR GOAL at step {t['step']}: ball=({t['ball_x']:.3f},{t['ball_y']:.3f},{t['ball_z']:.3f}), owned={t['ball_owned']}, actions={t['actions']}, event={t['event']}")

        if score_left > 0:
            print(f"  *** GOAL SCORED but no SHOT action detected ***")
            # Print last 10 steps
            for t in trajectory[-10:]:
                print(f"    Step {t['step']}: ball=({t['ball_x']:.3f},{t['ball_y']:.3f},{t['ball_z']:.3f}), owned={t['ball_owned']}, actions={t['actions']}, event={t['event']}")

    env.close()

if __name__ == "__main__":
    deep_debug()
