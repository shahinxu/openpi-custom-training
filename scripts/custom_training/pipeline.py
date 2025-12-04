#!/usr/bin/env python3
"""Complete pipeline: convert H5 data → compute stats → train."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, desc):
    """Run command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Step: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"✗ Failed: {desc}")
        sys.exit(1)
    print(f"✓ Completed: {desc}")


def main():
    parser = argparse.ArgumentParser(description="Full training pipeline")
    parser.add_argument("--data_dir", required=True, help="Source H5 directory")
    parser.add_argument("--output_dir", default="datasets/my_dataset", help="Dataset output")
    parser.add_argument("--exp_name", required=True, help="Experiment name")
    parser.add_argument("--config", default="pi05_custom", help="Training config")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--skip_convert", action="store_true", help="Skip data conversion")
    parser.add_argument("--skip_stats", action="store_true", help="Skip norm stats")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    # Step 1: Convert data
    if not args.skip_convert:
        run_command(
            ["python", str(script_dir / "convert_data.py"),
             "--data_dir", args.data_dir,
             "--output_dir", args.output_dir],
            "Data conversion"
        )
    
    # Step 2: Compute stats
    if not args.skip_stats:
        run_command(
            ["python", str(script_dir / "compute_stats.py"),
             "--dataset_dir", args.output_dir],
            "Compute normalization stats"
        )
    
    # Step 3: Train
    run_command(
        ["python", str(script_dir / "run_training.py"),
         "--config", args.config,
         "--exp_name", args.exp_name,
         "--gpu", str(args.gpu)],
        "Training"
    )


if __name__ == "__main__":
    main()
