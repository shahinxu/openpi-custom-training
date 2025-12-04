#!/usr/bin/env python3
"""
Complete pipeline for converting H5 dataset and training Pi0.5 model with LoRA.

Usage:
    python pipeline_convert_and_train.py \
        --data_dir /path/to/h5/data \
        --output_dir datasets/my_dataset \
        --dataset_name my_dataset \
        --exp_name my_experiment

This script will:
1. Convert H5 files to LeRobot format
2. Split into train/test sets
3. Compute normalization statistics
4. Start training with Pi0.5 + LoRA
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


def load_h5_data(h5_path):
    """Load data from a single H5 file."""
    with h5py.File(h5_path, 'r') as f:
        images = f['observations']['images'][:]  # Shape: (T, H, W, C)
        states = f['observations']['states'][:]  # Shape: (T, state_dim)
        actions = f['actions'][:]  # Shape: (T, action_dim)
    return images, states, actions


def convert_dataset(data_dir: Path, output_dir: Path, dataset_name: str):
    """Convert H5 files to LeRobot format with train/test split."""
    print("=" * 80)
    print("Step 1: Converting H5 files to LeRobot format")
    print("=" * 80)
    
    h5_files = sorted(data_dir.glob("*.h5"))
    if not h5_files:
        raise ValueError(f"No H5 files found in {data_dir}")
    
    print(f"Found {len(h5_files)} H5 files")
    
    # Count episodes per task for train/test split
    task_episode_counts = {}
    task_files = {}
    
    for h5_file in h5_files:
        task_name = h5_file.parent.name
        if task_name not in task_episode_counts:
            task_episode_counts[task_name] = 0
            task_files[task_name] = []
        task_episode_counts[task_name] += 1
        task_files[task_name].append(h5_file)
    
    # Determine train/test split (single-episode tasks -> test)
    train_files = []
    test_files = []
    
    for task_name, count in task_episode_counts.items():
        if count == 1:
            test_files.extend(task_files[task_name])
        else:
            train_files.extend(task_files[task_name])
    
    print(f"\nTrain/Test Split:")
    print(f"  Training episodes: {len(train_files)}")
    print(f"  Test episodes: {len(test_files)}")
    
    # Convert to LeRobot format
    all_episodes = []
    split_info = {}
    
    for episode_idx, h5_file in enumerate(tqdm(h5_files, desc="Converting episodes")):
        images, states, actions = load_h5_data(h5_file)
        task_name = h5_file.stem.replace('_', ' ').title()
        
        # Determine split
        split = "train" if h5_file in train_files else "test"
        split_info[episode_idx] = split
        
        episode_data = {
            "observation.images.cam_0": images,
            "observation.state": states,
            "action": actions,
            "task": task_name,
            "episode_index": episode_idx,
        }
        all_episodes.append(episode_data)
    
    # Create LeRobot dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = LeRobotDataset.create(
        repo_id=dataset_name,
        fps=30,
        root=str(output_dir.parent),
        robot_type="custom",
        keys_to_record=["observation.images.cam_0", "observation.state", "action", "task"],
        image_writer_threads=4,
    )
    
    for episode_data in all_episodes:
        for i in range(len(episode_data["action"])):
            frame = {
                "observation.images.cam_0": episode_data["observation.images.cam_0"][i],
                "observation.state": episode_data["observation.state"][i],
                "action": episode_data["action"][i],
                "task": episode_data["task"],
                "episode_index": episode_data["episode_index"],
                "frame_index": i,
                "timestamp": i / 30.0,
                "next.done": i == len(episode_data["action"]) - 1,
            }
            dataset.add_frame(frame)
        
        dataset.save_episode(episode_data["task"])
    
    dataset.consolidate()
    
    # Save split info
    split_file = output_dir / "split_info.json"
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✓ Dataset saved to: {output_dir}")
    print(f"✓ Split info saved to: {split_file}")
    
    return output_dir


def compute_norm_stats(dataset_name: str):
    """Compute normalization statistics."""
    print("\n" + "=" * 80)
    print("Step 2: Computing normalization statistics")
    print("=" * 80)
    
    cmd = [
        "uv", "run", "scripts/compute_norm_stats.py",
        "--config-name", "pi05_custom",
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        raise RuntimeError("Failed to compute normalization statistics")
    
    print("\n✓ Normalization statistics computed")


def start_training(exp_name: str, gpu_id: int = 0):
    """Start training with Pi0.5 + LoRA."""
    print("\n" + "=" * 80)
    print("Step 3: Starting training")
    print("=" * 80)
    
    print(f"\nTraining configuration:")
    print(f"  Config: pi05_custom")
    print(f"  Experiment: {exp_name}")
    print(f"  GPU: {gpu_id}")
    print(f"  Model: Pi0.5 with LoRA (gemma_2b_lora + gemma_300m_lora)")
    print(f"  Action dim: 32 (padded from dataset's native dimension)")
    print(f"  Batch size: 32")
    print(f"  Training steps: 10,000")
    print(f"  Learning rate: 3e-4 (cosine decay)")
    
    cmd = [
        "uv", "run", "scripts/train.py",
        "pi05_custom",
        "--exp_name", exp_name,
    ]
    
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    
    print(f"\n🚀 Starting training (this will take several hours)...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Check Wandb for live metrics: https://wandb.ai/")
    print(f"   Logs: training_{exp_name}.log")
    
    with open(f"training_{exp_name}.log", 'w') as log_file:
        subprocess.run(cmd, env={**subprocess.os.environ, **env}, stdout=log_file, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description="Complete pipeline for H5 -> LeRobot -> Training")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing H5 files")
    parser.add_argument("--output_dir", type=str, default="datasets/custom_dataset", help="Output directory for converted dataset")
    parser.add_argument("--dataset_name", type=str, default="custom_dataset", help="Dataset name (used as repo_id)")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name for training")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for training")
    parser.add_argument("--skip_convert", action="store_true", help="Skip data conversion if already done")
    parser.add_argument("--skip_norm_stats", action="store_true", help="Skip norm stats computation if already done")
    parser.add_argument("--train_only", action="store_true", help="Only run training (skip conversion and norm stats)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not args.train_only:
        # Step 1: Convert dataset
        if not args.skip_convert:
            convert_dataset(data_dir, output_dir, args.dataset_name)
        else:
            print(f"Skipping conversion, using existing dataset at {output_dir}")
        
        # Step 2: Compute normalization stats
        if not args.skip_norm_stats:
            compute_norm_stats(args.dataset_name)
        else:
            print("Skipping norm stats computation")
    
    # Step 3: Start training
    start_training(args.exp_name, args.gpu_id)
    
    print("\n" + "=" * 80)
    print("✓ Pipeline complete!")
    print("=" * 80)
    print(f"\nCheckpoints saved to: checkpoints/pi05_custom/{args.exp_name}/")
    print(f"View training progress: https://wandb.ai/")


if __name__ == "__main__":
    main()
