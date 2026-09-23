"""Reproduction script for CooperativeRewardShaper double-payment bug.

Before fix (r_pass=0.30):
  With engine base_rewards containing +0.15 per agent (shared broadcast),
  a single PASS_COMPLETED event produces:
    - passer/receiver: 0.15 (engine) + 0.30 (adapter) = 0.45
    - other agents: 0.15 (engine only)
  Total team reward for one pass: 0.45 + 0.15 + 0.15 = 0.75
  This is a double-payment: engine and adapter both pay for the same pass.

After fix (r_pass=0.0):
  With engine base_rewards containing +0.15 per agent,
  a single PASS_COMPLETED event produces:
    - passer/receiver: 0.15 (engine) + 0.0 (adapter) = 0.15
    - other agents: 0.15 (engine only)
  Total team reward for one pass: 0.15 + 0.15 + 0.15 = 0.45
  Only the engine's base pass reward survives; adapter does not duplicate it.
"""

from training.gmn_pettingzoo import CooperativeRewardShaper


def reproduce():
    shaper = CooperativeRewardShaper()
    shaper.reset()

    # Simulate engine broadcast: all agents receive +0.15 shared reward
    base_rewards = {"left_0": 0.15, "left_1": 0.15, "left_2": 0.15}
    step_events = [{"type": "PASS_COMPLETED", "team": "left", "agent_id": "left_0"}]
    ground_truth = {"current_ball_owner": {"team": "left", "agent_id": "left_0"}}
    active_agents = ["left_0", "left_1", "left_2"]

    rewards = shaper.compute_shaped_rewards(
        base_rewards, step_events, ground_truth, active_agents
    )

    team_total = sum(rewards.values())
    print("=== CooperativeRewardShaper Double-Payment Reproduction ===")
    print(f"r_pass (adapter pass reward): {shaper.r_pass}")
    print(f"Engine base reward per agent:  0.15 (simulated)")
    print()
    for agent in active_agents:
        print(f"  {agent}: {rewards[agent]:.4f}")
    print()
    print(f"Team total reward: {team_total:.4f}")
    print()

    if shaper.r_pass > 0.0:
        print("BUG DETECTED: adapter pays r_pass ON TOP OF engine +0.15.")
        print(f"  Passer/receiver gets {0.15 + shaper.r_pass:.2f} (double-paid).")
        print("  Expected: only engine +0.15 should survive.")
    else:
        print("FIXED: adapter r_pass=0.0, engine +0.15 is sole pass reward.")


if __name__ == "__main__":
    reproduce()
