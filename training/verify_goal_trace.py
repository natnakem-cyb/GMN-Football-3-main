import json

with open('training/results/goal_trace_seed44_best.json') as f:
    data = json.load(f)

print('=== GOAL TRACE VERIFICATION SUMMARY ===')
print(f'Total goal episodes: {len(data)}')
print()

shot_goal_pairs = []

for ep in data:
    shot_actions = ep['all_shot_actions']
    dribble_actions = ep['all_dribble_actions']
    pass_actions = ep['all_pass_actions']
    
    has_shot = len(shot_actions) > 0
    has_dribble = len(dribble_actions) > 0
    has_pass = len(pass_actions) > 0
    
    # Check for exploit signs
    exploit = False
    exploit_reasons = []
    
    # 1. No shot or dribble at all
    if not has_shot and not has_dribble:
        exploit = True
        exploit_reasons.append('NO_SHOT_OR_DRIBBLE')
    
    # 2. Ball velocity too high (unbounded momentum)
    decoded = ep.get('decoded_at_goal', {})
    ball_vel = decoded.get('ball_velocity_actual', [0, 0, 0])
    speed = (ball_vel[0]**2 + ball_vel[1]**2 + ball_vel[2]**2)**0.5
    if speed > 5.0:
        exploit = True
        exploit_reasons.append(f'HIGH_BALL_SPEED_{speed:.2f}')
    
    # 3. GK positioned incorrectly (should be near goal line x=1.0)
    right_positions = decoded.get('right_positions', [])
    if right_positions and right_positions[0][0] < 0.9:
        exploit = True
        exploit_reasons.append(f'GK_POSITION_{right_positions[0][0]:.3f}')
    
    shot_goal_pairs.append({
        'episode': ep['episode'],
        'has_shot': has_shot,
        'has_dribble': has_dribble,
        'has_pass': has_pass,
        'shot_steps': [s['step'] for s in shot_actions],
        'dribble_steps': [s['step'] for s in dribble_actions],
        'exploit': exploit,
        'exploit_reasons': exploit_reasons,
        'ball_speed': speed,
        'gk_x': right_positions[0][0] if right_positions else None,
    })

print('Per-episode breakdown:')
for s in shot_goal_pairs:
    status = 'EXPLOIT' if s['exploit'] else 'OK'
    print(f"  Ep {s['episode']:2d}: shot={s['has_shot']} dribble={s['has_dribble']} pass={s['has_pass']} | {status} | ball_speed={s['ball_speed']:.3f} | GK_x={s['gk_x']}")

print()
print('Aggregate stats:')
print(f"  All episodes have shot: {all(s['has_shot'] for s in shot_goal_pairs)}")
print(f"  All episodes have dribble or shot: {all(s['has_shot'] or s['has_dribble'] for s in shot_goal_pairs)}")
print(f"  Exploit flags: {sum(1 for s in shot_goal_pairs if s['exploit'])}")
print(f"  Unique shot steps: {len(set(step for s in shot_goal_pairs for step in s['shot_steps']))}")

# Verify shot timing makes sense
print()
print('Shot-to-goal timing:')
for ep_data, s in zip(data, shot_goal_pairs):
    if s['has_shot']:
        last_shot = max(s['shot_steps'])
        goal_step = ep_data.get('goal_step', '?')
        print(f"  Ep {s['episode']:2d}: shot at step {last_shot}, goal at step {goal_step}, delta = {goal_step - last_shot}")

# Verify ball trajectory physics
print()
print('Ball trajectory verification:')
for ep_data, s in zip(data, shot_goal_pairs):
    decoded = ep_data.get('decoded_at_goal', {})
    ball_pos = decoded.get('ball_position', [0, 0, 0])
    ball_vel = decoded.get('ball_velocity_actual', [0, 0, 0])
    print(f"  Ep {s['episode']:2d}: ball at ({ball_pos[0]:.3f}, {ball_pos[1]:.3f}, {ball_pos[2]:.3f}) "
          f"vel=({ball_vel[0]:.4f}, {ball_vel[1]:.4f}, {ball_vel[2]:.4f})")
