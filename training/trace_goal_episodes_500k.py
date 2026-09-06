"""
Trace specific goal episodes from the 500k MAPPO run.
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

ACTION_NAMES = {
    0: "IDLE", 1: "LEFT", 2: "TOP_LEFT", 3: "TOP", 4: "TOP_RIGHT",
    5: "RIGHT", 6: "BOTTOM_RIGHT", 7: "BOTTOM", 8: "BOTTOM_LEFT",
    9: "LONG_PASS", 10: "HIGH_PASS", 11: "SHORT_PASS", 12: "SHOT",
    13: "SPRINT", 14: "RELEASE_DIR", 15: "RELEASE_SPRINT",
    16: "SLIDING", 17: "DRIBBLE", 18: "RELEASE_DRIBBLE",
}

def trace_episode(episode_seed):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    actor = SharedActor(obs_dim=checkpoint.get("obs_dim", 127), action_dim=checkpoint.get("action_dim", 19), hidden=64)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = GMNMultiAgentEnv(scenario=SCENARIO, auto_start_bridge=True)
    controllable = list(env.possible_agents)

    obs_dict, _ = env.reset(seed=episode_seed)
    steps = 0
    done = False
    last_info = {}

    print(f"=== Episode seed {episode_seed} ===")
    print(f"{'Step':>4} {'BallX':>7} {'BallY':>7} {'BallZ':>7} {'Vx':>9} {'Vy':>9} {'Speed':>6} {'Owned':>6} {'Actions':<30}")
    print("-" * 110)

    while not done and steps < 600:
        current = list(env.agents if env.agents else controllable)
        local_obs = np.stack([obs_dict[a] for a in current], axis=0).astype(np.float32)

        with torch.no_grad():
            dist = actor(torch.from_numpy(local_obs).float())
            actions = dist.logits.argmax(dim=-1)

        action_dict = {}
        action_strs = []
        for i, a in enumerate(current):
            act_int = int(actions[i].item())
            action_dict[a] = act_int
            action_strs.append(f"{a[-2:]}={ACTION_NAMES[act_int][:4]}")

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

        event_type = None
        if last_info.get("event"):
            event_type = last_info["event"].get("type")

        event_str = str(event_type)[:12] if event_type else ""
        print(f"{steps:4d} {ball_x:7.3f} {ball_y:7.3f} {ball_z:7.3f} {ball_vx:9.4f} {ball_vy:9.4f} {ball_speed:6.4f} {ball_owned:6d} {','.join(action_strs):<30} {event_str:>12}")

        if event_type in ("goal", "shot", "shot_saved", "shot_missed"):
            print(f"  *** EVENT: {event_type} ***")

    score_left = last_info.get("score", {}).get("left", 0)
    event = last_info.get("event", {})
    print(f"\nFinal: steps={steps}, score_left={score_left}, event={event}")
    env.close()

if __name__ == "__main__":
    # Trace the two goal episodes from the 30-episode eval
    trace_episode(509081)  # Episode 9
    print("\n" + "="*80 + "\n")
    trace_episode(518162)  # Episode 18
