import json
import numpy as np
import os

# Check the most recent training runs
runs_dir = 'runs'
subdirs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))])
print(f'Available runs: {subdirs}')

for run_name in subdirs:
    metrics_path = os.path.join(runs_dir, run_name, 'metrics.jsonl')
    if not os.path.exists(metrics_path):
        continue
    records = []
    with open(metrics_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    rollout_rewards = [r.get('rollout_reward_mean') for r in records if r.get('rollout_reward_mean') is not None]
    if not rollout_rewards:
        continue
    
    print(f'\n=== {run_name} ({len(records)} records) ===')
    print(f'  rollout_reward_mean: [{min(rollout_rewards):.4f}, {max(rollout_rewards):.4f}]')
    
    # Check if values are in the 0.3-1.0 range
    in_range = sum(1 for v in rollout_rewards if 0.3 <= v <= 1.0)
    print(f'  values in [0.3, 1.0]: {in_range}/{len(rollout_rewards)}')
    
    entropies = [r.get('entropy') for r in records if r.get('entropy') is not None]
    if entropies:
        print(f'  entropy: [{min(entropies):.2f}, {max(entropies):.2f}]')
    
    clipfracs = [r.get('clipfrac') for r in records if r.get('clipfrac') is not None]
    if clipfracs:
        print(f'  clipfrac: [{min(clipfracs):.2f}, {max(clipfracs):.2f}]')
    
    kls = [r.get('approx_kl') for r in records if r.get('approx_kl') is not None]
    if kls:
        print(f'  approx_kl: [{min(kls):.4f}, {max(kls):.4f}]')
