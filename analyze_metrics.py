import json
import numpy as np

path = r'runs/carracing_ppo_20260423_103309/metrics.jsonl'
records = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f'Total records: {len(records)}')

rollout_rewards = [r.get('rollout_reward_mean') for r in records if r.get('rollout_reward_mean') is not None]
print(f'\nrollout_reward_mean count: {len(rollout_rewards)}')
if rollout_rewards:
    print(f'  min: {min(rollout_rewards):.4f}')
    print(f'  max: {max(rollout_rewards):.4f}')
    print(f'  mean: {np.mean(rollout_rewards):.4f}')
    print(f'  std: {np.std(rollout_rewards):.4f}')
    diffs = [abs(rollout_rewards[i+1] - rollout_rewards[i]) for i in range(len(rollout_rewards)-1)]
    print(f'  consecutive diff mean: {np.mean(diffs):.4f}')
    print(f'  consecutive diff max: {np.max(diffs):.4f}')

avg_returns = [r.get('avg_return_20') for r in records if r.get('avg_return_20') is not None]
print(f'\navg_return_20 count: {len(avg_returns)}')
if avg_returns:
    print(f'  min: {min(avg_returns):.1f}')
    print(f'  max: {max(avg_returns):.1f}')
    print(f'  mean: {np.mean(avg_returns):.1f}')

eval_returns = [r.get('eval_return_mean') for r in records if r.get('eval_return_mean') is not None]
print(f'\neval_return_mean count: {len(eval_returns)}')
if eval_returns:
    print(f'  min: {min(eval_returns):.1f}')
    print(f'  max: {max(eval_returns):.1f}')
    print(f'  mean: {np.mean(eval_returns):.1f}')

# Show first 30 records' rollout_reward_mean
print('\nFirst 30 rollout_reward_mean values:')
for i, v in enumerate(rollout_rewards[:30]):
    print(f'  update {i+1}: {v:.4f}')

# Check entropy and clipfrac
entropies = [r.get('entropy') for r in records if r.get('entropy') is not None]
print(f'\nentropy count: {len(entropies)}')
if entropies:
    print(f'  min: {min(entropies):.4f}')
    print(f'  max: {max(entropies):.4f}')
    print(f'  last 10: {[f"{e:.4f}" for e in entropies[-10:]]}')

clipfracs = [r.get('clipfrac') for r in records if r.get('clipfrac') is not None]
print(f'\nclipfrac count: {len(clipfracs)}')
if clipfracs:
    print(f'  min: {min(clipfracs):.4f}')
    print(f'  max: {max(clipfracs):.4f}')
    print(f'  mean: {np.mean(clipfracs):.4f}')

kls = [r.get('approx_kl') for r in records if r.get('approx_kl') is not None]
print(f'\napprox_kl count: {len(kls)}')
if kls:
    print(f'  min: {min(kls):.6f}')
    print(f'  max: {max(kls):.6f}')
    print(f'  mean: {np.mean(kls):.6f}')
