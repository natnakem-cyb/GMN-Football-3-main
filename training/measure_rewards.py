"""
Phase 3: Instrumented reward contribution measurement.

Uses the bridge's debug=rewards mode to collect per-step reward components.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('training'))

from training.gmn_pettingzoo import GMNMultiAgentEnv


def measure_reward_contributions(scenario="academy_3_vs_1_with_keeper", num_episodes=5, seed=42):
    """
    Run episodes with debug_rewards=True and collect reward component breakdown.
    """
    port = 5400 + hash(scenario) % 1000
    env = GMNMultiAgentEnv(
        scenario=scenario,
        auto_start_bridge=True,
        port=port,
        enable_reward_shaping=False,
        debug_rewards=True,
    )
    
    all_components = []
    
    try:
        for ep in range(num_episodes):
            env.reset(seed=seed + ep)
            env.reward_components = []
            
            for _ in range(200):
                current_agents = list(env.agents if env.agents else env.possible_agents)
                if not current_agents:
                    break
                action_dict = {a: 0 for a in current_agents}
                obs, rewards, terms, truncs, infos = env.step(action_dict)
                
                if any(terms.values()) or any(truncs.values()) or not env.agents:
                    break
            
            # Collect reward components for this episode
            ep_components = env.reward_components
            all_components.extend(ep_components)
            
            # Print summary
            total_reward = sum(c["components"]["total"] for c in ep_components)
            goal_reward = sum(c["components"]["goal"] for c in ep_components)
            progress_reward = sum(c["components"]["progress"] for c in ep_components)
            shot_reward = sum(c["components"]["shot"] for c in ep_components)
            pass_shaper = sum(c["components"]["pass_shaper"] for c in ep_components)
            defensive = sum(c["components"]["defensive"] for c in ep_components)
            
            print(f"\n=== Episode {ep + 1} ===")
            print(f"  Steps: {len(ep_components)}")
            print(f"  Total reward: {total_reward:.4f}")
            print(f"  Goal reward: {goal_reward:.4f}")
            print(f"  Progress reward: {progress_reward:.4f}")
            print(f"  Shot reward: {shot_reward:.4f}")
            print(f"  Pass shaper: {pass_shaper:.4f}")
            print(f"  Defensive: {defensive:.4f}")
            
            if total_reward != 0:
                print(f"  Progress %: {(progress_reward / total_reward) * 100:.1f}%")
                print(f"  Goal %: {(goal_reward / total_reward) * 100:.1f}%")
                print(f"  Shot %: {(shot_reward / total_reward) * 100:.1f}%")
    finally:
        env.close()
    
    # Aggregate across all episodes
    if all_components:
        total = sum(c["components"]["total"] for c in all_components)
        goal = sum(c["components"]["goal"] for c in all_components)
        progress = sum(c["components"]["progress"] for c in all_components)
        shot = sum(c["components"]["shot"] for c in all_components)
        pass_s = sum(c["components"]["pass_shaper"] for c in all_components)
        defensive = sum(c["components"]["defensive"] for c in all_components)
        
        print(f"\n=== AGGREGATE ({num_episodes} episodes) ===")
        print(f"  Total reward: {total:.4f}")
        print(f"  Goal reward: {goal:.4f} ({goal/total*100:.1f}%)" if total != 0 else "  Goal reward: 0.0")
        print(f"  Progress reward: {progress:.4f} ({progress/total*100:.1f}%)" if total != 0 else "  Progress reward: 0.0")
        print(f"  Shot reward: {shot:.4f} ({shot/total*100:.1f}%)" if total != 0 else "  Shot reward: 0.0")
        print(f"  Pass shaper: {pass_s:.4f} ({pass_s/total*100:.1f}%)" if total != 0 else "  Pass shaper: 0.0")
        print(f"  Defensive: {defensive:.4f} ({defensive/total*100:.1f}%)" if total != 0 else "  Defensive: 0.0")


if __name__ == "__main__":
    measure_reward_contributions()
