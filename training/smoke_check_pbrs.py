#!/usr/bin/env python3
"""Smoke check for PBRS implementation."""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'training')

from training.reward_adapters import AttackingDrillRewardAdapter


def main():
    ad = AttackingDrillRewardAdapter()
    ad.reset()

    # Build-up phase
    distances = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
    build_up_rewards = []

    print('=== Build-up phase ===')
    for i, dist in enumerate(distances):
        gt = {'current_ball_owner': {'team': 'left', 'agent_id': 'left_0'}, 
              'ball_distance_to_goal': dist}
        out = ad.compute_shaped_rewards({'left_0': 0.0}, [], gt, ['left_0'])
        r = out['left_0']
        build_up_rewards.append(r)
        print(f'Tick {i+1}: dist={dist:.1f}, reward={r:+.4f}')

    print()
    print(f'Build-up total: {sum(build_up_rewards):+.4f}')
    print(f'PBRS contribution: {sum(build_up_rewards) - 10 * (-0.005):+.4f}')

    # Goal phase
    print()
    print('=== Goal phase ===')
    gt_goal = {'current_ball_owner': {'team': 'left', 'agent_id': 'left_0'}, 
               'ball_distance_to_goal': 0.1}
    out_goal = ad.compute_shaped_rewards(
        {'left_0': 2.0}, 
        [{'type': 'GOAL_SCORED', 'team': 'left', 'agent_id': 'left_0'}], 
        gt_goal, ['left_0']
    )
    goal_val = out_goal['left_0']
    print(f'Goal tick reward: {goal_val:+.4f}')

    total = sum(build_up_rewards) + goal_val
    print()
    print(f'Total reward: {total:+.4f}')
    print(f'Goal reward dominates: {goal_val > sum(build_up_rewards)}')
    print('Smoke check: PASSED')
    
    # Verify no NaN/inf
    for r in build_up_rewards:
        assert r == r and abs(r) != float('inf'), f"NaN/inf in rewards: {r}"
    assert goal_val == goal_val and abs(goal_val) != float('inf'), f"NaN/inf in goal: {goal_val}"
    
    # PBRS magnitude sane
    pbrs_total = sum(build_up_rewards) - 10 * (-0.005)
    assert abs(pbrs_total) < 5.0, f"PBRS magnitude {pbrs_total} too large"
    assert total > 1.5, f"Total reward {total} should be dominated by goal (+2.0)"
    
    print('\nAll smoke checks passed!')
    return 0


if __name__ == '__main__':
    sys.exit(main())