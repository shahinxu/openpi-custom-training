#!/usr/bin/env python3
"""Visualize evaluation results and show episode instructions."""

import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Set up paths
eval_dir = Path("/home/rzh/zhenx/openpi/checkpoints/pi05_train_test/train34_test14_2000steps/eval_results")
episodes_file = Path("/home/rzh/zhenx/openpi/data/train_test_dataset/meta/episodes.jsonl")

# Read evaluation summary
summary = pd.read_csv(eval_dir / "evaluation_summary.csv")

# Read per-episode results for each step
per_episode_data = {}
for step in [200, 400, 600, 800]:
    df = pd.read_csv(eval_dir / f"eval_step_{step:06d}_per_episode.csv")
    per_episode_data[step] = df

# Extract instructions from episodes.jsonl
episode_instructions = {}
if episodes_file.exists():
    with open(episodes_file, 'r') as f:
        for line in f:
            episode = json.loads(line)
            ep_idx = episode['episode_index']
            if 34 <= ep_idx <= 47:
                # Extract task (remove any seg prefix)
                tasks = episode['tasks']
                if tasks:
                    task = tasks[0]
                    # Clean up task string
                    if '/' in task:
                        task = task.split('/')[0]
                    episode_instructions[ep_idx] = task
else:
    for ep_idx in range(34, 48):
        episode_instructions[ep_idx] = f'Test Episode {ep_idx}'

print("\n=== Episode Instructions ===")
for ep_idx in sorted(episode_instructions.keys()):
    print(f"Episode {ep_idx}: {episode_instructions[ep_idx]}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Evaluation Results: MSE across Training Steps', fontsize=16, fontweight='bold')

# Plot 1: Overall MSE trend
ax1 = axes[0, 0]
ax1.plot(summary['step'], summary['overall_mse'], marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax1.axhline(y=summary['overall_mse'].min(), color='red', linestyle='--', alpha=0.5, label='Best MSE')
ax1.set_xlabel('Training Step', fontsize=12)
ax1.set_ylabel('Overall MSE', fontsize=12)
ax1.set_title('Overall Test Set MSE', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
for i, row in summary.iterrows():
    ax1.annotate(f'{row["overall_mse"]:.4f}', 
                xy=(row['step'], row['overall_mse']),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9)

# Plot 2: Per-episode MSE at Step 800
ax2 = axes[0, 1]
step_800 = per_episode_data[800]
colors = ['#A23B72' if mse > 0.2 else '#F18F01' if mse > 0.15 else '#06A77D' 
          for mse in step_800['mse']]
bars = ax2.bar(step_800['episode_index'].astype(str), step_800['mse'], color=colors, alpha=0.8)
ax2.set_xlabel('Episode Index', fontsize=12)
ax2.set_ylabel('MSE', fontsize=12)
ax2.set_title('Per-Episode MSE at Step 800', fontsize=14, fontweight='bold')
ax2.axhline(y=step_800['mse'].mean(), color='red', linestyle='--', alpha=0.5, label='Mean MSE')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# Plot 3: MSE evolution for selected episodes
ax3 = axes[1, 0]
episodes_to_plot = [34, 39, 41, 45, 46]  # Best, worst, and interesting ones
for ep in episodes_to_plot:
    mse_values = [per_episode_data[step].loc[per_episode_data[step]['episode_index'] == ep, 'mse'].values[0] 
                  for step in [200, 400, 600, 800]]
    ax3.plot([200, 400, 600, 800], mse_values, marker='o', label=f'Ep {ep}', linewidth=2)
ax3.set_xlabel('Training Step', fontsize=12)
ax3.set_ylabel('MSE', fontsize=12)
ax3.set_title('MSE Evolution for Selected Episodes', fontsize=14, fontweight='bold')
ax3.legend(loc='best', ncol=2)
ax3.grid(True, alpha=0.3)

# Plot 4: Episode length vs MSE (Step 800)
ax4 = axes[1, 1]
scatter = ax4.scatter(step_800['num_frames'], step_800['mse'], 
                     s=100, alpha=0.6, c=step_800['mse'], 
                     cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
ax4.set_xlabel('Number of Frames', fontsize=12)
ax4.set_ylabel('MSE', fontsize=12)
ax4.set_title('Episode Length vs MSE (Step 800)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
# Add colorbar
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('MSE', fontsize=10)
# Annotate extreme points
for idx, row in step_800.iterrows():
    if row['mse'] > 0.24 or row['mse'] < 0.11:
        ax4.annotate(f"Ep {int(row['episode_index'])}", 
                    xy=(row['num_frames'], row['mse']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.tight_layout()
output_path = eval_dir / "evaluation_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")

# Create a summary table
print("\n=== Summary Statistics (Step 800) ===")
print(f"Best Episode: {step_800.loc[step_800['mse'].idxmin(), 'episode_index']:.0f} (MSE: {step_800['mse'].min():.4f})")
print(f"Worst Episode: {step_800.loc[step_800['mse'].idxmax(), 'episode_index']:.0f} (MSE: {step_800['mse'].max():.4f})")
print(f"Mean MSE: {step_800['mse'].mean():.4f}")
print(f"Std MSE: {step_800['mse'].std():.4f}")

# Print detailed per-episode table with instructions
print("\n=== Detailed Per-Episode Results (Step 800) ===")
print(f"{'Ep':<4} {'Frames':<7} {'MSE':<8} {'Instruction':<60}")
print("-" * 90)
for _, row in step_800.iterrows():
    ep_idx = int(row['episode_index'])
    instruction = episode_instructions.get(ep_idx, 'Unknown')
    print(f"{ep_idx:<4} {int(row['num_frames']):<7} {row['mse']:<8.4f} {instruction}")
