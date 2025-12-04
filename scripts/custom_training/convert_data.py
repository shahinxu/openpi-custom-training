#!/usr/bin/env python3
"""Convert H5 dataset to LeRobot format with train/test split."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_h5_data(h5_path):
    """Load data from H5 file."""
    with h5py.File(h5_path, "r") as f:
        images = np.array(f["observations"]["images"])
        states = np.array(f["observations"]["states"])
        actions = np.array(f["actions"])
    return {"images": images, "states": states, "actions": actions}


def convert_dataset(source_dir, output_dir, split_mode="auto"):
    """Convert H5 files to LeRobot format.
    
    Args:
        source_dir: Directory containing H5 files
        output_dir: Output directory for converted dataset
        split_mode: "auto" (single-episode as test) or "all_train"
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    h5_files = sorted(source_path.glob("*.h5"))
    if not h5_files:
        raise ValueError(f"No H5 files found in {source_dir}")
    
    print(f"Found {len(h5_files)} H5 files")
    
    # Determine split
    train_files, test_files = [], []
    if split_mode == "auto":
        for h5_file in h5_files:
            data = load_h5_data(h5_file)
            if len(data["images"]) == 1:
                test_files.append(h5_file)
            else:
                train_files.append(h5_file)
    else:
        train_files = h5_files
    
    print(f"Train: {len(train_files)} files, Test: {len(test_files)} files")
    
    # Create output structure
    for split in ["train", "test"]:
        (output_path / split / "data").mkdir(parents=True, exist_ok=True)
        (output_path / split / "videos").mkdir(parents=True, exist_ok=True)
    
    # Convert files
    def process_files(files, split):
        episode_data = []
        frame_idx = 0
        
        for ep_idx, h5_file in enumerate(tqdm(files, desc=f"Converting {split}")):
            data = load_h5_data(h5_file)
            n_frames = len(data["images"])
            
            episode_info = {
                "episode_index": ep_idx,
                "tasks": [h5_file.stem],
                "length": n_frames,
            }
            episode_data.append(episode_info)
            
            # Save data
            data_file = output_path / split / "data" / f"episode_{ep_idx:06d}.npz"
            np.savez_compressed(
                data_file,
                action=data["actions"],
                state=data["states"],
                frame_index=np.arange(frame_idx, frame_idx + n_frames),
            )
            
            # Save images
            for i, img in enumerate(data["images"]):
                img_path = output_path / split / "videos" / f"episode_{ep_idx:06d}_frame_{i:06d}.png"
                Image.fromarray(img).save(img_path)
            
            frame_idx += n_frames
        
        # Save metadata
        meta = {
            "episodes": episode_data,
            "total_frames": frame_idx,
            "total_episodes": len(files),
        }
        meta_file = output_path / split / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)
    
    process_files(train_files, "train")
    if test_files:
        process_files(test_files, "test")
    
    # Save split info
    split_info = {
        "train_tasks": sorted([f.stem for f in train_files]),
        "test_tasks": sorted([f.stem for f in test_files]),
    }
    with open(output_path / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✓ Dataset converted to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Source H5 directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--split_mode", default="auto", choices=["auto", "all_train"])
    args = parser.parse_args()
    
    convert_dataset(args.data_dir, args.output_dir, args.split_mode)
