"""
Phase 3: Reward contribution instrumentation.

Wraps GMNMultiAgentEnv to track per-component reward contributions.
Does not modify production code; uses delegation + event interception.
"""
import os
import sys
import numpy as np
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('training'))

from training.gmn_pettingzoo import GMNMultiAgentEnv, CooperativeRewardShaper


class RewardAuditor:
    """
    Tracks reward component contributions during environment interaction.
    Wraps an existing GMNMultiAgentEnv and records per-step reward breakdowns.
    """

    def __init__(self, env: GMNMultiAgentEnv):
        self.env = env
        self.reset_episode()

    def reset_episode(self) -> None:
        """Reset per-episode accumulators."""
        self.episode_rewards: List[float] = []
        self.episode_components: List[Dict[str, float]] = []
        self.total_reward = 0.0
        self.possession_reward = 0.0
        self.progress_reward = 0.0
        self.shot_reward = 0.0
        self.goal_reward = 0.0
        self.pass_reward = 0.0
        self.solitary_shot_penalty = 0.0
        self.ball_hogging_penalty = 0.0
        self.assisted_goal_bonus = 0.0
        self.defensive_reward = 0.0
        self.other_shaping_reward = 0.0
        self.step_count = 0
        self.passes_completed = 0
        self.shots_taken = 0
        self.goals_scored = 0
        self.turnovers = 0

    def _classify_reward(self, reward: float, info: Dict[str, Any]) -> Dict[str, float]:
        """
        Classify a reward into components based on available signals.
        Since the bridge only sends a scalar, we approximate components from
        event codes and the checkpointReward field.
        """
        components = {
            "total": reward,
            "possession": 0.0,
            "progress": 0.0,
            "shot": 0.0,
            "goal": 0.0,
            "pass": 0.0,
            "solitary_shot_penalty": 0.0,
            "ball_hogging_penalty": 0.0,
            "assisted_goal_bonus": 0.0,
            "defensive": 0.0,
            "other_shaping": 0.0,
        }

        # Extract checkpoint reward from info (sent by bridge)
        checkpoint_reward = info.get("checkpointReward", 0.0)
        if checkpoint_reward > 0:
            components["progress"] = checkpoint_reward

        # Extract event info
        event = info.get("event", {})
        event_type = event.get("type") if isinstance(event, dict) else None
        event_code = info.get("eventCode", 0)

        # Map event codes to reward components
        # Goal events
        if event_type == "goal" or event_code == 6:
            components["goal"] = reward - checkpoint_reward
            self.goals_scored += 1
            return components

        # Shot events
        if event_type == "shot" or event_code == 4:
            components["shot"] = reward - checkpoint_reward
            self.shots_taken += 1
            return components

        # Pass events
        if event_type == "pass" or event_code == 3:
            # Pass reward comes from shaper, not base reward
            # Base reward for a pass tick is typically 0 or checkpoint
            components["pass"] = reward - checkpoint_reward
            self.passes_completed += 1
            return components

        # Turnover events
        if event_type in ("interception", "tackle", "foul"):
            components["defensive"] = reward - checkpoint_reward
            self.turnovers += 1
            return components

        # Default: treat non-event reward as possession/progress
        if reward > 0:
            components["possession"] = reward

        return components

    def step(self, actions: Dict[str, int]) -> tuple:
        """
        Execute one environment step and track reward components.
        Returns the same tuple as env.step() but also updates internal accounting.
        """
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # Classify rewards for each agent
        step_components = {}
        for agent_id, reward in rewards.items():
            info = infos.get(agent_id, {})
            components = self._classify_reward(reward, info)
            step_components[agent_id] = components
            
            # Accumulate episode totals
            self.total_reward += reward
            for key, value in components.items():
                if key != "total" and value != 0:
                    attr_name = key
                    if hasattr(self, attr_name):
                        setattr(self, attr_name, getattr(self, attr_name) + value)

        self.episode_rewards.append(sum(rewards.values()))
        self.episode_components.append(step_components)
        self.step_count += 1

        return obs, rewards, terminations, truncations, infos

    def get_episode_summary(self) -> Dict[str, Any]:
        """Return a summary of the current episode's reward decomposition."""
        if self.step_count == 0:
            return {}

        return {
            "total_reward": self.total_reward,
            "possession_reward": self.possession_reward,
            "progress_reward": self.progress_reward,
            "shot_reward": self.shot_reward,
            "goal_reward": self.goal_reward,
            "pass_reward": self.pass_reward,
            "solitary_shot_penalty": self.solitary_shot_penalty,
            "ball_hogging_penalty": self.ball_hogging_penalty,
            "assisted_goal_bonus": self.assisted_goal_bonus,
            "defensive_reward": self.defensive_reward,
            "other_shaping_reward": self.other_shaping_reward,
            "step_count": self.step_count,
            "passes_completed": self.passes_completed,
            "shots_taken": self.shots_taken,
            "goals_scored": self.goals_scored,
            "turnovers": self.turnovers,
            "reward_per_step": self.total_reward / self.step_count if self.step_count > 0 else 0,
        }

    def get_percentage_breakdown(self) -> Dict[str, float]:
        """Return each component as a percentage of total reward."""
        summary = self.get_episode_summary()
        if summary.get("total_reward", 0) == 0:
            return {}

        pct = {}
        for key in [
            "possession_reward",
            "progress_reward",
            "shot_reward",
            "goal_reward",
            "pass_reward",
            "solitary_shot_penalty",
            "ball_hogging_penalty",
            "assisted_goal_bonus",
            "defensive_reward",
            "other_shaping_reward",
        ]:
            pct[key + "_pct"] = (abs(summary.get(key, 0)) / abs(summary["total_reward"])) * 100
        return pct


def run_audit(scenario: str = "academy_3_vs_1_with_keeper", num_episodes: int = 5, seed: int = 42):
    """
    Run a short audit episode and report reward decomposition.
    """
    port = 5300 + hash(scenario) % 1000
    env = GMNMultiAgentEnv(
        scenario=scenario,
        auto_start_bridge=True,
        port=port,
        enable_reward_shaping=False,
    )
    auditor = RewardAuditor(env)

    try:
        for ep in range(num_episodes):
            auditor.reset_episode()
            obs, info = env.reset(seed=seed + ep)
            
            for _ in range(200):
                current_agents = list(env.agents if env.agents else env.possible_agents)
                action_dict = {a: 0 for a in current_agents}
                obs, rewards, terms, truncs, infos = auditor.step(action_dict)
                
                if any(terms.values()) or any(truncs.values()) or not env.agents:
                    break
            
            summary = auditor.get_episode_summary()
            pct = auditor.get_percentage_breakdown()
            
            print(f"\n=== Episode {ep + 1} ===")
            print(f"  Total reward: {summary.get('total_reward', 0):.4f}")
            print(f"  Steps: {summary.get('step_count', 0)}")
            print(f"  Reward/step: {summary.get('reward_per_step', 0):.4f}")
            print(f"  Passes: {summary.get('passes_completed', 0)}")
            print(f"  Shots: {summary.get('shots_taken', 0)}")
            print(f"  Goals: {summary.get('goals_scored', 0)}")
            print(f"  Progress reward: {summary.get('progress_reward', 0):.4f}")
            print(f"  Goal reward: {summary.get('goal_reward', 0):.4f}")
            print(f"  Shot reward: {summary.get('shot_reward', 0):.4f}")
            print(f"  Pass reward: {summary.get('pass_reward', 0):.4f}")
            
            print("  Percentage breakdown:")
            for k, v in pct.items():
                print(f"    {k}: {v:.1f}%")
    finally:
        env.close()


if __name__ == "__main__":
    run_audit()
