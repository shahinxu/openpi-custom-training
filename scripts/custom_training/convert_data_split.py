#!/usr/bin/env python3
"""Convert separate train and test directories to LeRobot format."""

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
    
    # Convert images to uint8 if needed
    if images.dtype != np.uint8:
        # Assume images are in [0, 1] range and convert to [0, 255]
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8)
    
    # Ensure images have correct shape (remove extra dimensions)
    if images.ndim == 5 and images.shape[1] == 1 and images.shape[2] == 1:
        # Shape is (n, 1, 1, h, w, c), squeeze to (n, h, w, c)
        images = images.squeeze(axis=(1, 2))
    
    return {"images": images, "states": states, "actions": actions}


def load_directory_data(dir_path):
    """Load data from directory with subdirectories containing H5 files."""
    dir_path = Path(dir_path)
    all_data = []
    
    # Find all subdirectories
    subdirs = sorted([d for d in dir_path.iterdir() if d.is_dir()])
    
    for subdir in subdirs:
        # Find ALL H5 files in this subdirectory
        h5_files = sorted(subdir.glob("*.h5"))
        for h5_file in h5_files:
            data = load_h5_data(h5_file)
            # Use a unique name combining subdir name and file name
            episode_name = f"{subdir.name}/{h5_file.stem}"
            all_data.append({
                "name": episode_name,
                "data": data,
                "path": h5_file
            })
    
    return all_data


def convert_split(data_list, output_path, split_name):
    """Convert a list of data to LeRobot format."""
    print(f"\nConverting {split_name} set: {len(data_list)} episodes")
    
    # Create output structure
    split_path = output_path / split_name
    (split_path / "data").mkdir(parents=True, exist_ok=True)
    (split_path / "videos").mkdir(parents=True, exist_ok=True)
    
    episode_data = []
    frame_idx = 0
    
    for ep_idx, item in enumerate(tqdm(data_list, desc=f"Converting {split_name}")):
        data = item["data"]
        n_frames = len(data["images"])
        
        episode_info = {
            "episode_index": ep_idx,
            "tasks": [item["name"]],
            "length": n_frames,
        }
        episode_data.append(episode_info)
        
        # Save data
        data_file = split_path / "data" / f"episode_{ep_idx:06d}.npz"
        np.savez_compressed(
            data_file,
            action=data["actions"],
            state=data["states"],
            frame_index=np.arange(frame_idx, frame_idx + n_frames),
        )
        
        # Save images
        for i, img in enumerate(data["images"]):
            img_path = split_path / "videos" / f"episode_{ep_idx:06d}_frame_{i:06d}.png"
            Image.fromarray(img).save(img_path)
        
        frame_idx += n_frames
    
    # Save metadata
    meta = {
        "episodes": episode_data,
        "total_frames": frame_idx,
        "total_episodes": len(data_list),
    }
    meta_file = split_path / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"✓ {split_name} set: {len(data_list)} episodes, {frame_idx} frames")


def convert_dataset(train_dir, test_dir, output_dir):
    """Convert separate train and test directories to LeRobot format.
    
    Args:
        train_dir: Directory with subdirectories containing training H5 files
        test_dir: Directory with test H5 files
        output_dir: Output directory for converted dataset
    """
    output_path = Path(output_dir)
    
    # Load training data from subdirectories
    print(f"Loading training data from {train_dir}")
    train_data = load_directory_data(train_dir)
    if not train_data:
        raise ValueError(f"No training data found in {train_dir}")
    
    # Load test data from H5 files
    print(f"Loading test data from {test_dir}")
    test_path = Path(test_dir)
    test_h5_files = sorted(test_path.glob("*.h5"))
    if not test_h5_files:
        raise ValueError(f"No H5 files found in {test_dir}")
    
    test_data = []
    for h5_file in test_h5_files:
        data = load_h5_data(h5_file)
        test_data.append({
            "name": h5_file.stem,
            "data": data,
            "path": h5_file
        })
    
    print(f"\nDataset summary:")
    print(f"  Training episodes: {len(train_data)}")
    print(f"  Test episodes: {len(test_data)}")
    
    # Convert both splits
    convert_split(train_data, output_path, "train")
    convert_split(test_data, output_path, "test")
    
    # Save split info
    split_info = {
        "train_tasks": sorted([item["name"] for item in train_data]),
        "test_tasks": sorted([item["name"] for item in test_data]),
        "train_count": len(train_data),
        "test_count": len(test_data),
    }
    with open(output_path / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✓ Dataset converted to {output_dir}")
    print(f"  Train: {len(train_data)} episodes")
    print(f"  Test: {len(test_data)} episodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert separate train/test data to LeRobot format")
    parser.add_argument("--train_dir", required=True, help="Training data directory")
    parser.add_argument("--test_dir", required=True, help="Test data directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    convert_dataset(args.train_dir, args.test_dir, args.output_dir)
