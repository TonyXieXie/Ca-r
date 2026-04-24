import json
import numpy as np

path = r'runs/carracing_ppo_20260423_103309/metrics.jsonl'
records = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

# The user says rollout_reward_mean jumps between 0.3 and 1.0
# But our data shows it's in range [-0.33, 0.31]
# They might be looking at the dashboard which could normalize/scale the data
# Or they might be looking at a different metric

# Let's check if the dashboard normalizes the data
# Also let's check the actual raw values more carefully - maybe there are different phases

rollout_rewards = [r.get('rollout_reward_mean') for r in records if r.get('rollout_reward_mean') is not None]

# Check for any values in the 0.3-1.0 range
high_values = [(i, v) for i, v in enumerate(rollout_rewards) if v > 0.3]
print(f'Values > 0.3: {len(high_values)}')
if high_values:
    for i, v in high_values[:10]:
        print(f'  update {i+1}: {v:.4f}')

# Maybe the user is looking at the combined chart which normalizes to [0,1]?
# The dashboard svgChart function likely normalizes to 0-1 range for display
# Let's check what the normalization would look like

rmin = min(rollout_rewards)
rmax = max(rollout_rewards)
print(f'\nrollout_reward_mean raw range: [{rmin:.4f}, {rmax:.4f}]')
print(f'If normalized to [0,1]: value = (x - {rmin:.4f}) / {rmax - rmin:.4f}')

# Show some normalized values
print('\nNormalized values (what dashboard might show):')
for i in [0, 1, 2, 50, 100, 500, 1000, -1]:
    idx = i if i >= 0 else len(rollout_rewards) + i
    if 0 <= idx < len(rollout_rewards):
        raw = rollout_rewards[idx]
        norm = (raw - rmin) / (rmax - rmin) if rmax != rmin else 0
        print(f'  update {idx+1}: raw={raw:.4f}, normalized={norm:.4f}')

# But wait - the dashboard might show ALL metrics normalized together in the combined chart
# Let's look at what the user might actually be seeing

# Actually, re-reading the user's question: "0.3-1.0之间快速跳动"
# This could be the avg_return_20 normalized, or they might be reading the chart Y-axis wrong
# Or they could be looking at a different training run

# Let's check if there are other run directories
import os
runs_dir = 'runs'
if os.path.exists(runs_dir):
    subdirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    print(f'\nAll run directories: {subdirs}')
