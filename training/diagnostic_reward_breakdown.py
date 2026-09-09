"""
GMN-Football-3 — Reward Breakdown Diagnostic for Phase 11

Runs policies C (SHOT spam) and D (DiagonalShot) for 10 episodes each
and prints a per-episode breakdown of:
  - goals scored
  - total checkpoint reward accumulated
  - total pass-completion reward accumulated (should be 0 for these policies)

This is used to diagnose why test_policy_c_shot_spam and test_policy_d_diagonal_shot
are failing with positive total reward.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from training.gmn_pettingzoo import GMNMultiAgentEnv

NUM_EPISODES = 10
SCENARIO = "academy_3_vs_1_with_keeper"


def _make_env():
    port = 5060 + hash("reward_breakdown_diag") % 1000
    env = GMNMultiAgentEnv(
        scenario=SCENARIO,
        auto_start_bridge=True,
        port=port,
        enable_reward_shaping=True,
        debug_rewards=True,
    )
    return env


def _run_policy_with_breakdown(env, policy_fn, num_episodes, policy_label="policy"):
    """Run policy and collect per-episode reward breakdown.

    Tracks the first episode with total_reward > 0 and prints every step's
    raw reward, checkpoint reward, event type, and action taken.
    """
    episodes = []
    first_positive_logged = False

    for ep in range(num_episodes):
        ep_goals = 0
        ep_checkpoint = 0.0
        ep_pass_reward = 0.0
        ep_total_reward = 0.0
        ep_steps = 0
        ep_crashed = False
        step_log = []

        try:
            obs_dict, info_dict = env.reset(seed=42 + ep)
            step_idx = 0

            while True:
                current_agents = list(env.agents if env.agents else obs_dict.keys())
                if not current_agents:
                    break

                action_dict = policy_fn(obs_dict, step_idx)
                obs_dict, rewards, terms, truncs, infos = env.step(action_dict)

                shared_reward = float(rewards[current_agents[0]]) if current_agents and current_agents[0] in rewards else 0.0
                ep_total_reward += shared_reward
                ep_steps += 1

                step_cp = 0.0
                step_event_type = None
                step_event_code = None
                step_score_l = None
                step_score_r = None
                step_frame_length = None
                for agent_id, info in infos.items():
                    cp = info.get("checkpointReward", 0.0)
                    step_cp += float(cp)
                    ep_checkpoint += float(cp)

                    for ev in info.get("step_events", []):
                        if ev.get("type") == "PASS_COMPLETED":
                            ep_pass_reward += 0.30

                    ev = info.get("event", {})
                    if isinstance(ev, dict):
                        ev_type = ev.get("type")
                    else:
                        ev_type = str(ev) if ev else None
                    if ev_type:
                        step_event_type = ev_type

                    if ev_type == "goal":
                        ep_goals += 1

                    step_event_code = info.get("eventCode")
                    score_dict = info.get("score", {})
                    step_score_l = score_dict.get("left")
                    step_score_r = score_dict.get("right")
                    step_frame_length = info.get("frame_length")

                step_log.append({
                    "step_idx": step_idx,
                    "raw_reward": shared_reward,
                    "checkpointReward": step_cp,
                    "event_type": step_event_type,
                    "event_code": step_event_code,
                    "score_l": step_score_l,
                    "score_r": step_score_r,
                    "frame_length": step_frame_length,
                    "action": action_dict,
                })

                if any(terms.values()) or any(truncs.values()):
                    break

                step_idx += 1
                if step_idx > 2000:
                    break

        except Exception:
            ep_crashed = True

        episodes.append({
            "episode": ep,
            "total_reward": ep_total_reward,
            "goals": ep_goals,
            "checkpoint_reward": ep_checkpoint,
            "pass_completion_reward": ep_pass_reward,
            "steps": ep_steps,
            "crashed": ep_crashed,
        })

        if not first_positive_logged and ep_total_reward > 0:
            first_positive_logged = True
            print(f"\n  --- [{policy_label}] First positive-reward episode: ep={ep} "
                  f"(total_reward={ep_total_reward:.4f}, goals={ep_goals}, "
                  f"checkpoint={ep_checkpoint:.4f}) ---")
            for entry in step_log:
                print(f"    step={entry['step_idx']:>4}  "
                      f"raw_reward={entry['raw_reward']:+.4f}  "
                      f"checkpointReward={entry['checkpointReward']:+.4f}  "
                      f"event_code={str(entry['event_code']):>6}  "
                      f"event={str(entry['event_type']):>20}  "
                      f"score_l={str(entry['score_l']):>4}  "
                      f"score_r={str(entry['score_r']):>4}  "
                      f"frame_len={str(entry['frame_length']):>6}  "
                      f"action={entry['action']}")
            print(f"    SUM raw_reward={sum(s['raw_reward'] for s in step_log):+.4f}  "
                  f"SUM checkpointReward={sum(s['checkpointReward'] for s in step_log):+.4f}")

    return episodes


def _print_table(episodes, title):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"  {'Ep':>3}  {'Total Reward':>12}  {'Goals':>6}  {'Checkpoint':>11}  {'PassReward':>11}  {'Steps':>6}  {'Crashed':>7}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*6}  {'-'*11}  {'-'*11}  {'-'*6}  {'-'*7}")
    total_reward = 0.0
    total_goals = 0
    total_checkpoint = 0.0
    total_pass = 0.0
    total_steps = 0
    total_crashes = 0
    for ep in episodes:
        print(f"  {ep['episode']:>3}  {ep['total_reward']:>12.4f}  {ep['goals']:>6}  {ep['checkpoint_reward']:>11.4f}  {ep['pass_completion_reward']:>11.4f}  {ep['steps']:>6}  {str(ep['crashed']):>7}")
        total_reward += ep['total_reward']
        total_goals += ep['goals']
        total_checkpoint += ep['checkpoint_reward']
        total_pass += ep['pass_completion_reward']
        total_steps += ep['steps']
        total_crashes += int(ep['crashed'])
    print(f"  {'-'*90}")
    print(f"  {'SUM':>3}  {total_reward:>12.4f}  {total_goals:>6}  {total_checkpoint:>11.4f}  {total_pass:>11.4f}  {total_steps:>6}  {total_crashes:>7}")
    print(f"{'='*90}")


def main():
    print("GMN-Football-3 — Reward Breakdown Diagnostic (Phase 11)")
    print("Scenario: academy_3_vs_1_with_keeper")

    # Policy C: SHOT spam
    env_c = _make_env()
    try:
        def policy_c(obs_dict, step_idx):
            return {a: 12 for a in obs_dict.keys()}  # SHOT

        episodes_c = _run_policy_with_breakdown(env_c, policy_c, NUM_EPISODES, policy_label="Policy C")
        _print_table(episodes_c, "Policy C: SHOT Spam (action 12 every step)")
    finally:
        env_c.close()

    # Policy D: DiagonalShot
    env_d = _make_env()
    try:
        def policy_d(obs_dict, step_idx):
            if step_idx % 20 == 0:
                return {a: 5 for a in obs_dict.keys()}  # UP_LEFT
            else:
                return {a: 12 for a in obs_dict.keys()}  # SHOT

        episodes_d = _run_policy_with_breakdown(env_d, policy_d, NUM_EPISODES, policy_label="Policy D")
        _print_table(episodes_d, "Policy D: DiagonalShot (UP_LEFT for 20 steps, then SHOT, repeat)")
    finally:
        env_d.close()


if __name__ == "__main__":
    main()
