import json
import csv
import os

base_seeds = [42, 123, 7, 999]
all_episodes = []

for seed in base_seeds:
    path = f'training/results/f_act_ONBALL-pi_seed{seed}_mappo_academy_3_vs_1_with_keeper_seed{seed}_freshtrain_50176.json'
    with open(path) as f:
        data = json.load(f)
    all_episodes.extend(data['episodes'])

# Aggregate
total_ticks = sum(ep['episode_length'] for ep in all_episodes)
total_pass = sum(ep['policy_pass_count'] for ep in all_episodes)
total_shot = sum(ep['policy_shot_count'] for ep in all_episodes)
total_tackle = sum(ep['policy_tackle_count'] for ep in all_episodes)
total_pass_events = sum(1 for ep in all_episodes for t in ep['tick_log'] if t.get('event_code') == 10)
total_pass_completed = sum(1 for ep in all_episodes for t in ep['tick_log'] if t.get('event_code') == 11)
total_shot_events = sum(1 for ep in all_episodes for t in ep['tick_log'] if t.get('event_code') == 12)
total_goal_events = sum(1 for ep in all_episodes for t in ep['tick_log'] if t.get('event_code') == 1)

# Write aggregate JSON
agg = {
    "run_id": "fresh_train_mu_onball_2026-09-18",
    "git_head": "05c95eb",
    "scenario": "academy_3_vs_1_with_keeper_onball",
    "arm": "ONBALL-pi",
    "deterministic": True,
    "base_seed": 42,
    "episodes_per_seed": 20,
    "seeds": base_seeds,
    "checkpoint_paths": {str(s): f"training/models/mappo_academy_3_vs_1_with_keeper_seed{s}_freshtrain_50176.pt" for s in base_seeds},
    "checkpoint_sha256": {},
    "checkpoint_timesteps": 50176,
    "aggregate": {
        "total_episodes": len(all_episodes),
        "valid_t0_count": sum(1 for ep in all_episodes if ep['valid_t0_possession']),
        "total_ticks": total_ticks,
        "total_policy_pass": total_pass,
        "total_policy_shot": total_shot,
        "total_policy_tackle": total_tackle,
        "total_pass_events": total_pass_events,
        "total_pass_completed_events": total_pass_completed,
        "total_shot_events": total_shot_events,
        "total_goal_events": total_goal_events
    },
    "per_seed": []
}

for seed in base_seeds:
    path = f'training/results/f_act_ONBALL-pi_seed{seed}_mappo_academy_3_vs_1_with_keeper_seed{seed}_freshtrain_50176.json'
    with open(path) as f:
        data = json.load(f)
    eps = data['episodes']
    agg["checkpoint_sha256"][str(seed)] = data['checkpoint_sha256']
    seed_total_ticks = sum(ep['episode_length'] for ep in eps)
    seed_pass = sum(ep['policy_pass_count'] for ep in eps)
    seed_shot = sum(ep['policy_shot_count'] for ep in eps)
    seed_tackle = sum(ep['policy_tackle_count'] for ep in eps)
    agg["per_seed"].append({
        "seed": seed,
        "checkpoint": data['checkpoint'],
        "checkpoint_sha256": data['checkpoint_sha256'],
        "episodes": 20,
        "valid_t0_count": sum(1 for ep in eps if ep['valid_t0_possession']),
        "total_ticks": seed_total_ticks,
        "policy_pass_count": seed_pass,
        "policy_shot_count": seed_shot,
        "policy_tackle_count": seed_tackle
    })

with open('training/results/f_act_fresh_train_postfix.json', 'w') as f:
    json.dump(agg, f, indent=2)

# Write summary CSV
csv_path = 'training/results/f_act_fresh_train_postfix_summary.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['seed', 'episodes', 'valid_t0_count', 'total_ticks', 'policy_pass_count', 'policy_shot_count', 'policy_tackle_count', 'pass_events', 'pass_completed_events', 'shot_events', 'goal_events'])
    for seed in base_seeds:
        path = f'training/results/f_act_ONBALL-pi_seed{seed}_mappo_academy_3_vs_1_with_keeper_seed{seed}_freshtrain_50176.json'
        with open(path) as f:
            data = json.load(f)
        eps = data['episodes']
        seed_total_ticks = sum(ep['episode_length'] for ep in eps)
        seed_pass = sum(ep['policy_pass_count'] for ep in eps)
        seed_shot = sum(ep['policy_shot_count'] for ep in eps)
        seed_tackle = sum(ep['policy_tackle_count'] for ep in eps)
        seed_pass_events = sum(1 for ep in eps for t in ep['tick_log'] if t.get('event_code') == 10)
        seed_pass_completed = sum(1 for ep in eps for t in ep['tick_log'] if t.get('event_code') == 11)
        seed_shot_events = sum(1 for ep in eps for t in ep['tick_log'] if t.get('event_code') == 12)
        seed_goal_events = sum(1 for ep in eps for t in ep['tick_log'] if t.get('event_code') == 1)
        writer.writerow([seed, 20, sum(1 for ep in eps if ep['valid_t0_possession']), seed_total_ticks, seed_pass, seed_shot, seed_tackle, seed_pass_events, seed_pass_completed, seed_shot_events, seed_goal_events])

print('Done writing aggregate files.')
