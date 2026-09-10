"""
GMN-Football-3 — Shot-Spam Score Verification (Phase 11 diagnostic)

Runs ShotSpam and DiagonalShot policies and reports final info["score"]
for each episode, confirming whether score.left actually incremented to 1
on the +2.0-ish reward episodes.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from training.gmn_pettingzoo import GMNMultiAgentEnv

NUM_EPISODES = 10
SCENARIO = "academy_3_vs_1_with_keeper"


def _make_env():
    port = 5060 + hash("score_check") % 1000
    env = GMNMultiAgentEnv(
        scenario=SCENARIO,
        auto_start_bridge=True,
        port=port,
        enable_reward_shaping=True,
        debug_rewards=False,
    )
    return env


def _run_policy(env, policy_fn, num_episodes, label):
    results = []
    for ep in range(num_episodes):
        try:
            obs_dict, info_dict = env.reset(seed=42 + ep)
            ep_reward = 0.0
            step_idx = 0
            final_score = {"left": 0, "right": 0}

            while True:
                current_agents = list(env.agents if env.agents else obs_dict.keys())
                if not current_agents:
                    break

                action_dict = policy_fn(obs_dict, step_idx)
                obs_dict, rewards, terms, truncs, infos = env.step(action_dict)

                shared_reward = float(rewards[current_agents[0]]) if current_agents and current_agents[0] in rewards else 0.0
                ep_reward += shared_reward

                # Capture top-level score on terminal step
                if any(terms.values()) or any(truncs.values()):
                    for info in infos.values():
                        if isinstance(info, dict) and "score" in info:
                            final_score = info["score"]
                    break

                step_idx += 1
                if step_idx > 2000:
                    break

            results.append({
                "episode": ep,
                "total_reward": ep_reward,
                "final_score_left": final_score.get("left", 0),
                "final_score_right": final_score.get("right", 0),
            })
        except Exception as e:
            results.append({
                "episode": ep,
                "total_reward": None,
                "final_score_left": None,
                "final_score_right": None,
                "error": str(e),
            })

    print(f"\n{'='*60}")
    print(f"Policy: {label}")
    print(f"{'='*60}")
    positive_reward_eps = 0
    score_incremented_eps = 0
    for r in results:
        if r["total_reward"] is not None and r["total_reward"] > 0:
            positive_reward_eps += 1
        if r["final_score_left"] is not None and r["final_score_left"] > 0:
            score_incremented_eps += 1
        reward_str = f"{r['total_reward']:+.4f}" if r["total_reward"] is not None else "CRASH"
        print(
            f"  Ep {r['episode']+1:2d}: reward={reward_str:>10} | "
            f"score=({r['final_score_left']}-{r['final_score_right']})"
        )
    print(f"  Positive-reward episodes: {positive_reward_eps}/{num_episodes}")
    print(f"  score.left > 0 episodes  : {score_incremented_eps}/{num_episodes}")
    return results


def main():
    env = _make_env()
    try:
        def shot_spam(obs_dict, step_idx):
            return {a: 12 for a in obs_dict.keys()}

        def diagonal_shot(obs_dict, step_idx):
            if step_idx % 20 == 0:
                return {a: 5 for a in obs_dict.keys()}
            else:
                return {a: 12 for a in obs_dict.keys()}

        _run_policy(env, shot_spam, NUM_EPISODES, "ShotSpam (action 12 every step)")
        _run_policy(env, diagonal_shot, NUM_EPISODES, "DiagonalShot (UP_LEFT x19, SHOT x1)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
