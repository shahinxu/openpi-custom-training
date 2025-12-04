#!/usr/bin/env python3
"""Start training with proper environment setup."""

import argparse
import os
import subprocess
import sys


def start_training(config_name, exp_name, gpu_id=0):
    """Launch training process with single GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = ["uv", "run", "scripts/train.py", config_name, "--exp_name", exp_name]
    
    print(f"Starting training: {config_name}")
    print(f"Experiment name: {exp_name}")
    print(f"GPU: {gpu_id}")
    print(f"Command: {' '.join(cmd)}")
    print("\nMonitor at: https://wandb.ai")
    print("-" * 60)
    
    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_custom", help="Config name")
    parser.add_argument("--exp_name", required=True, help="Experiment name")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    args = parser.parse_args()
    
    start_training(args.config, args.exp_name, args.gpu)
