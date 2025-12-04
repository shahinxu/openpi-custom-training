#!/usr/bin/env python3
"""Compute normalization statistics for LeRobot dataset."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def compute_norm_stats(dataset_dir):
    """Compute action and state normalization statistics."""
    dataset_path = Path(dataset_dir)
    train_dir = dataset_path / "train" / "data"
    
    if not train_dir.exists():
        raise ValueError(f"Train data not found: {train_dir}")
    
    all_actions, all_states = [], []
    
    print("Loading training data...")
    for npz_file in sorted(train_dir.glob("episode_*.npz")):
        data = np.load(npz_file)
        all_actions.append(data["action"])
        all_states.append(data["state"])
    
    actions = np.concatenate(all_actions, axis=0)
    states = np.concatenate(all_states, axis=0)
    
    stats = {
        "action": {
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
        },
        "state": {
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
            "min": states.min(axis=0).tolist(),
            "max": states.max(axis=0).tolist(),
        },
    }
    
    # Save stats
    stats_file = dataset_path / "norm_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Normalization stats saved to {stats_file}")
    print(f"  Actions: shape={actions.shape}, mean={actions.mean():.3f}, std={actions.std():.3f}")
    print(f"  States: shape={states.shape}, mean={states.mean():.3f}, std={states.std():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Dataset directory")
    args = parser.parse_args()
    
    compute_norm_stats(args.dataset_dir)
