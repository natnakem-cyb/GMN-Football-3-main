"""Break down occupancy arithmetic into on-ball and off-ball contributions."""
import json
from collections import defaultdict

PASS_SHOT_ACTION_IDS = (9, 10, 11, 12)

with open('training/results/post_reweight_logit_prestep_reconciled_detail.json', 'r') as f:
    detail = json.load(f)

frames = detail['frames']
by_seed = defaultdict(list)
for fr in frames:
    by_seed[int(fr['seed'])].append(fr)

print('Occupancy arithmetic breakdown:')
print('Seed | n_onball | n_selected_ps_onball | n_pass_shot_total | n_offball | n_selected_ps_offball | implied_onball | implied_offball | total_implied | actual | match?')
rates = {42: 0.745098, 123: 1.673203, 7: 0.862745, 999: 1.032680}
for seed in [42, 123, 7, 999]:
    seed_frames = by_seed[seed]
    n_total = len(seed_frames)
    n_onball = sum(1 for f in seed_frames if f['onball'])
    n_offball = n_total - n_onball
    n_selected_ps_onball = sum(1 for f in seed_frames if f['onball'] and f['action_taken'] in PASS_SHOT_ACTION_IDS)
    n_pass_shot_total = sum(1 for f in seed_frames if f['action_taken'] in PASS_SHOT_ACTION_IDS)
    n_selected_ps_offball = n_pass_shot_total - n_selected_ps_onball
    
    implied_onball = n_selected_ps_onball / n_total * 100.0
    implied_offball = n_selected_ps_offball / n_total * 100.0
    total_implied = implied_onball + implied_offball
    actual = rates[seed]
    match = abs(total_implied - actual) < 0.001
    
    print(f'{seed:4d} | {n_onball:8d} | {n_selected_ps_onball:17d} | {n_pass_shot_total:16d} | {n_offball:8d} | {n_selected_ps_offball:17d} | {implied_onball:.6f}% | {implied_offball:.6f}% | {total_implied:.6f}% | {actual:.6f}% | {"YES" if match else "NO"}')

print()
print('Seed 42 disconnect analysis:')
print(f'  n_onball=13 ({13/7650*100:.3f}% of all decisions)')
print(f'  n_selected_ps_onball=8 (61.54% conditional rate)')
print(f'  n_pass_shot_total=57 (0.745% unconditional rate)')
print(f'  n_selected_ps_offball={57-8} ({49/7650*100:.3f}% of all decisions)')
print(f'  Off-ball PASS+SHOT selections: {49} out of 57 total (85.96%)')
print()
print('The seed-42 disconnect (low unconditional rate despite moderate conditional rate)')
print('is primarily explained by low on-ball occupancy (13/7650 = 0.17%), not by')
print('a failure to select PASS+SHOT when on-ball. When seed 42 has the ball, it')
print('selects PASS+SHOT at 61.54% (8/13), which is the second-lowest conditional')
print('rate among the four seeds. The low unconditional rate is driven by the tiny')
print('on-ball sample, with the majority of PASS+SHOT selections (49/57 = 85.96%)')
print('coming from off-ball situations where the policy is not constrained by the')
print('on-ball predicate.')
print()
print('Small-n caveat: With n_onball=13, a single frame changes the conditional')
print(f'rate by {1.0/13*100:.2f} percentage points. The observed 61.54% should be')
print('read as a descriptive statistic from a sparse sample, not a stable property.')
