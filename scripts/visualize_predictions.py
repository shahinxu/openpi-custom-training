#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_episode(npz_file: Path, output_dir: Path):
    data = np.load(npz_file)
    predicted = data['predictions'][:, 0, :]
    ground_truth = data['ground_truth'][:, 0, :]
    
    _, action_dim = predicted.shape
    mse = np.mean((predicted - ground_truth) ** 2)
    per_dim_mse = np.mean((predicted - ground_truth) ** 2, axis=0)
    
    fig, axes = plt.subplots(action_dim, 1, figsize=(12, 2.5 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]
    
    for dim in range(action_dim):
        ax = axes[dim]
        ax.plot(ground_truth[:, dim], label='GT', linewidth=2, alpha=0.8)
        ax.plot(predicted[:, dim], label='Pred', linewidth=2, alpha=0.8, linestyle='--')
        ax.set_ylabel(f'Dim {dim}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Dim {dim} - MSE: {per_dim_mse[dim]:.6f}')
    
    axes[-1].set_xlabel('Frame')
    fig.suptitle(f'{npz_file.stem} - MSE: {mse:.6f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{npz_file.stem}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return mse


def plot_all(predictions_dir: Path, output_dir: Path, max_episodes: int = None):
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    npz_files = sorted(predictions_dir.glob('episode_*.npz'))
    if max_episodes:
        npz_files = npz_files[:max_episodes]
    
    all_mses = []
    for npz_file in npz_files:
        mse = plot_episode(npz_file, output_dir)
        all_mses.append(mse)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    episode_nums = [int(f.stem.split('_')[1]) for f in npz_files]
    ax.bar(range(len(episode_nums)), all_mses, alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('MSE')
    ax.set_xticks(range(len(episode_nums)))
    ax.set_xticklabels([f'{n}' for n in episode_nums], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    mean_mse = np.mean(all_mses)
    ax.axhline(mean_mse, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_mse:.6f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Mean MSE: {mean_mse:.6f}")


def compare_steps(eval_dir: Path, output_dir: Path, episode_id: int):
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_dirs = sorted(eval_dir.glob('predictions_step_*'))
    episode_file = f'episode_{episode_id:03d}.npz'
    steps_data = []
    
    for pred_dir in pred_dirs:
        npz_file = pred_dir / episode_file
        if npz_file.exists():
            data = np.load(npz_file)
            step_num = int(pred_dir.name.split('_')[-1])
            predicted = data['predictions'][:, 0, :]
            ground_truth = data['ground_truth'][:, 0, :]
            mse = np.mean((predicted - ground_truth) ** 2)
            steps_data.append({'step': step_num, 'predicted': predicted, 'ground_truth': ground_truth, 'mse': mse})
    
    steps_data = sorted(steps_data, key=lambda x: x['step'])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = [d['step'] for d in steps_data]
    mses = [d['mse'] for d in steps_data]
    ax.plot(steps, mses, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'ep{episode_id}_progress.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions_dir', type=str)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--max-episodes', type=int, default=None)
    parser.add_argument('--compare-steps', action='store_true')
    parser.add_argument('--episode-id', type=int, default=34)
    args = parser.parse_args()
    
    predictions_dir = Path(args.predictions_dir)
    
    if args.compare_steps:
        eval_dir = predictions_dir.parent if predictions_dir.name.startswith('predictions_') else predictions_dir
        output_dir = Path(args.output_dir) if args.output_dir else eval_dir / 'comparison'
        compare_steps(eval_dir, output_dir, args.episode_id)
    else:
        output_dir = Path(args.output_dir) if args.output_dir else predictions_dir / 'plots'
        plot_all(predictions_dir, output_dir, args.max_episodes)


if __name__ == '__main__':
    main()
