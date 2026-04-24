import json
import numpy as np

path = r'runs/carracing_ppo_20260423_103309/metrics.jsonl'
records = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

# Check the actual range the user mentioned: 0.3-1.0
rollout_rewards = [r.get('rollout_reward_mean') for r in records if r.get('rollout_reward_mean') is not None]

# The user said 0.3-1.0, but our data shows -0.33 to 0.31
# Let's check rollout_reward_std which might be what they're seeing
rollout_stds = [r.get('rollout_reward_std') for r in records if r.get('rollout_reward_std') is not None]
print(f'rollout_reward_std:')
print(f'  min: {min(rollout_stds):.4f}')
print(f'  max: {max(rollout_stds):.4f}')
print(f'  mean: {np.mean(rollout_stds):.4f}')

# Let's check the dashboard - maybe it shows something different
# Check avg_return_20 range
avg_returns = [r.get('avg_return_20') for r in records if r.get('avg_return_20') is not None]
print(f'\navg_return_20:')
print(f'  min: {min(avg_returns):.1f}')
print(f'  max: {max(avg_returns):.1f}')
print(f'  mean: {np.mean(avg_returns):.1f}')
print(f'  std: {np.std(avg_returns):.1f}')

# Show a window of consecutive values
print('\nSample of consecutive rollout_reward_mean (updates 100-130):')
for i in range(99, min(130, len(rollout_rewards))):
    print(f'  update {i+1}: {rollout_rewards[i]:.4f}')

# Key diagnostic: entropy is VERY negative, clipfrac is VERY high, KL is HUGE
print('\n=== CRITICAL DIAGNOSTICS ===')
print(f'entropy range: [{min([r.get("entropy",0) for r in records]):.2f}, {max([r.get("entropy",0) for r in records]):.2f}]')
print(f'clipfrac range: [{min([r.get("clipfrac",0) for r in records]):.2f}, {max([r.get("clipfrac",0) for r in records]):.2f}]')
print(f'approx_kl range: [{min([r.get("approx_kl",0) for r in records]):.4f}, {max([r.get("approx_kl",0) for r in records]):.4f}]')
print(f'approx_kl > 0.05 count: {sum(1 for r in records if r.get("approx_kl",0) > 0.05)}')
print(f'approx_kl > 1.0 count: {sum(1 for r in records if r.get("approx_kl",0) > 1.0)}')
print(f'clipfrac > 0.3 count: {sum(1 for r in records if r.get("clipfrac",0) > 0.3)}')

# Show early vs late entropy
entropies = [r.get('entropy') for r in records if r.get('entropy') is not None]
print(f'\nentropy first 5: {[f"{e:.4f}" for e in entropies[:5]]}')
print(f'entropy last 5: {[f"{e:.4f}" for e in entropies[-5:]]}')