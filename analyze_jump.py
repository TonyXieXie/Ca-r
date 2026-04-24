import json
import numpy as np

# The user is likely looking at one of these runs with 0.3-1.0 range
# Let's analyze the most recent full run: carracing_ppo_20260423_153611
path = r'runs/carracing_ppo_20260423_153611/metrics.jsonl'
records = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

rollout_rewards = [r.get('rollout_reward_mean') for r in records if r.get('rollout_reward_mean') is not None]

# Show consecutive differences
diffs = [rollout_rewards[i+1] - rollout_rewards[i] for i in range(len(rollout_rewards)-1)]
abs_diffs = [abs(d) for d in diffs]

print(f'rollout_reward_mean: [{min(rollout_rewards):.4f}, {max(rollout_rewards):.4f}]')
print(f'consecutive abs diff: mean={np.mean(abs_diffs):.4f}, max={np.max(abs_diffs):.4f}')
print(f'consecutive diff std: {np.std(diffs):.4f}')

# Show a sample of consecutive values to illustrate the jumping
print('\nSample of 40 consecutive values (around the middle):')
mid = len(rollout_rewards) // 2
for i in range(mid, min(mid+40, len(rollout_rewards))):
    diff = rollout_rewards[i] - rollout_rewards[i-1] if i > 0 else 0
    print(f'  update {i+1}: {rollout_rewards[i]:+.4f}  (Δ={diff:+.4f})')

# Also check: how many consecutive pairs have a big jump (>0.3)?
big_jumps = sum(1 for d in abs_diffs if d > 0.3)
print(f'\nConsecutive jumps > 0.3: {big_jumps}/{len(abs_diffs)} ({100*big_jumps/len(abs_diffs):.1f}%)')
big_jumps2 = sum(1 for d in abs_diffs if d > 0.2)
print(f'Consecutive jumps > 0.2: {big_jumps2}/{len(abs_diffs)} ({100*big_jumps2/len(abs_diffs):.1f}%)')

# Check avg_return_20 for the same run
avg_returns = [r.get('avg_return_20') for r in records if r.get('avg_return_20') is not None]
if avg_returns:
    print(f'\navg_return_20: [{min(avg_returns):.1f}, {max(avg_returns):.1f}]')
    avg_diffs = [abs(avg_returns[i+1] - avg_returns[i]) for i in range(len(avg_returns)-1)]
    print(f'consecutive diff: mean={np.mean(avg_diffs):.1f}, max={np.max(avg_diffs):.1f}')
