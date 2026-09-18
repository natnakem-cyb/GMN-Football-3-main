import json

base_seeds = [42, 123, 7, 999]
results = {}

for seed in base_seeds:
    path = f'training/results/f_act_ONBALL-pi_seed{seed}_mappo_academy_3_vs_1_with_keeper_seed{seed}_freshtrain_50176.json'
    with open(path) as f:
        data = json.load(f)
    
    total_ticks = 0
    pass_count = 0
    shot_count = 0
    tackle_count = 0
    dribble_count = 0
    move_count = 0
    idle_count = 0
    pass_events = 0
    pass_completed_events = 0
    shot_events = 0
    goal_events = 0
    
    for ep in data['episodes']:
        total_ticks += ep['episode_length']
        pass_count += ep['policy_pass_count']
        shot_count += ep['policy_shot_count']
        tackle_count += ep['policy_tackle_count']
        
        for tick in ep['tick_log']:
            ec = tick.get('event_code', 0)
            if ec == 10:
                pass_events += 1
            elif ec == 11:
                pass_completed_events += 1
            elif ec == 12:
                shot_events += 1
            elif ec == 1:
                goal_events += 1
    
    # DRIBBLE/MOVE/IDLE from tick_log action_selected (exclude event ticks to avoid double-count)
    # Actually, we need to count all ticks. Let me count from episode summary event_histogram
    # and tick_log. The episode summary doesn't have dribble/move/idle counts directly.
    # Let me count from tick_log for non-event ticks.
    
    for ep in data['episodes']:
        for tick in ep['tick_log']:
            ec = tick.get('event_code', 0)
            if ec == 0:
                act = tick.get('action_selected', -1)
                if act == 17:
                    dribble_count += 1
                elif 1 <= act <= 8:
                    move_count += 1
                elif act == 0:
                    idle_count += 1
    
    results[seed] = {
        'total_ticks': total_ticks,
        'pass': pass_count,
        'shot': shot_count,
        'tackle': tackle_count,
        'dribble': dribble_count,
        'move': move_count,
        'idle': idle_count,
        'pass_events': pass_events,
        'pass_completed_events': pass_completed_events,
        'shot_events': shot_events,
        'goal_events': goal_events,
    }

for seed, r in results.items():
    total = r['total_ticks']
    print(f"Seed {seed}: ticks={total}")
    print(f"  PASS={r['pass']} ({r['pass']/max(total,1)*100:.2f}%)")
    print(f"  SHOT={r['shot']} ({r['shot']/max(total,1)*100:.2f}%)")
    print(f"  TACKLE={r['tackle']} ({r['tackle']/max(total,1)*100:.2f}%)")
    print(f"  DRIBBLE={r['dribble']} ({r['dribble']/max(total,1)*100:.2f}%)")
    print(f"  MOVE={r['move']} ({r['move']/max(total,1)*100:.2f}%)")
    print(f"  IDLE={r['idle']} ({r['idle']/max(total,1)*100:.2f}%)")
    print(f"  Events: pass={r['pass_events']}, pass_completed={r['pass_completed_events']}, shot={r['shot_events']}, goal={r['goal_events']}")
    combined = r['pass'] + r['shot']
    print(f"  Combined PASS+SHOT rate: {combined}/{total} = {combined/max(total,1)*100:.2f}%")
    print()

# Pooled
all_r = list(results.values())
total_ticks = sum(r['total_ticks'] for r in all_r)
total_pass = sum(r['pass'] for r in all_r)
total_shot = sum(r['shot'] for r in all_r)
total_tackle = sum(r['tackle'] for r in all_r)
total_dribble = sum(r['dribble'] for r in all_r)
total_move = sum(r['move'] for r in all_r)
total_idle = sum(r['idle'] for r in all_r)
total_pass_events = sum(r['pass_events'] for r in all_r)
total_pass_completed = sum(r['pass_completed_events'] for r in all_r)
total_shot_events = sum(r['shot_events'] for r in all_r)
total_goal_events = sum(r['goal_events'] for r in all_r)

print(f"Pooled: ticks={total_ticks}")
print(f"  PASS={total_pass} ({total_pass/max(total_ticks,1)*100:.2f}%)")
print(f"  SHOT={total_shot} ({total_shot/max(total_ticks,1)*100:.2f}%)")
print(f"  TACKLE={total_tackle} ({total_tackle/max(total_ticks,1)*100:.2f}%)")
print(f"  DRIBBLE={total_dribble} ({total_dribble/max(total_ticks,1)*100:.2f}%)")
print(f"  MOVE={total_move} ({total_move/max(total_ticks,1)*100:.2f}%)")
print(f"  IDLE={total_idle} ({total_idle/max(total_ticks,1)*100:.2f}%)")
print(f"  Events: pass={total_pass_events}, pass_completed={total_pass_completed}, shot={total_shot_events}, goal={total_goal_events}")
combined = total_pass + total_shot
print(f"  Combined PASS+SHOT rate: {combined}/{total_ticks} = {combined/max(total_ticks,1)*100:.2f}%")
