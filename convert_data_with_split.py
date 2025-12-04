"""
Convert state_segmented_data to LeRobot format with train/test split.
Single-episode tasks → test set
Multi-episode tasks → train set

Usage:
python convert_data_with_split.py
"""

import glob
import os
import shutil
from pathlib import Path
from collections import defaultdict

import h5py
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME


# Configuration
DATA_DIR = "/home/rzh/zhenx/openpi/state_segmented_data"
REPO_ID = "rzh/openpi_segmented_data"
FPS = 30  # Frames per second


def main():
    print("🚀 开始数据转换 (带训练/测试集划分)...")
    
    # Clean up existing dataset
    output_path = HF_LEROBOT_HOME / REPO_ID
    if output_path.exists():
        print(f"🧹 清理已存在的数据集: {output_path}")
        shutil.rmtree(output_path)
    
    # Find all h5 files and group by task
    h5_files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.h5"), recursive=True))
    print(f"📦 找到 {len(h5_files)} 个 h5 文件")
    
    if len(h5_files) == 0:
        raise ValueError(f"在 {DATA_DIR} 中没有找到 h5 文件")
    
    # Group files by task directory
    task_groups = defaultdict(list)
    for h5_path in h5_files:
        task_dir = os.path.basename(os.path.dirname(h5_path))
        task_groups[task_dir].append(h5_path)
    
    # Split into train and test
    train_files = []
    test_files = []
    
    for task_dir, files in task_groups.items():
        if len(files) == 1:
            test_files.extend(files)
        else:
            train_files.extend(files)
    
    print(f"\n📊 数据划分:")
    print(f"  训练集: {len(train_files)} episodes")
    print(f"  测试集: {len(test_files)} episodes")
    
    # Load first file to determine dimensions
    with h5py.File(h5_files[0], 'r') as f:
        sample_action = f['actions'][0]
        sample_state = f['observations']['states'][0]
        sample_image = f['observations']['images'][0]
        
        action_dim = sample_action.shape[0]
        state_dim = sample_state.shape[0]
        h, w, c = sample_image.shape
    
    print(f"\n📐 数据维度:")
    print(f"  动作 (action): {action_dim}")
    print(f"  状态 (state): {state_dim}")
    print(f"  图像 (image): ({h}, {w}, {c})")
    
    # Define features
    features = {
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": None
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": None
        },
        "observation.images.cam_0": {
            "dtype": "video",
            "shape": (c, h, w),
            "names": ["channel", "height", "width"]
        },
    }
    
    # Create dataset
    print(f"\n🏗️  创建 LeRobot 数据集 (FPS={FPS})...")
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        features=features,
        fps=FPS,
        robot_type="custom",
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # Process files
    episode_splits = {}  # Track which episode belongs to which split
    current_episode_idx = 0
    
    def process_files(files, split_name):
        nonlocal current_episode_idx
        print(f"\n{'='*60}")
        print(f"处理 {split_name} 数据")
        print(f"{'='*60}")
        
        total_frames = 0
        for episode_idx, h5_path in enumerate(files):
            filename = os.path.basename(h5_path)
            task_dir = os.path.basename(os.path.dirname(h5_path))
            
            # Extract task name from filename
            task_name = "_".join(filename.split("_")[1:]).replace(".h5", "")
            
            # Load data from h5 file
            with h5py.File(h5_path, 'r') as f:
                actions = f['actions'][:]
                states = f['observations']['states'][:]
                images = f['observations']['images'][:]
            
            num_frames = actions.shape[0]
            total_frames += num_frames
            
            print(f"📝 [{episode_idx+1}/{len(files)}] {task_dir}/{filename}")
            print(f"    任务: {task_name}, 帧数: {num_frames}")
            
            # Add each frame
            for frame_idx in range(num_frames):
                # Convert image from (H, W, C) to (C, H, W) and to torch tensor
                img_tensor = torch.from_numpy(images[frame_idx]).permute(2, 0, 1)
                
                frame = {
                    "action": torch.from_numpy(actions[frame_idx]).float(),
                    "observation.state": torch.from_numpy(states[frame_idx]).float(),
                    "observation.images.cam_0": img_tensor,
                    "task": task_name
                }
                
                dataset.add_frame(frame)
            
            # Save episode
            dataset.save_episode()
            episode_splits[current_episode_idx] = split_name
            current_episode_idx += 1
            
            if (episode_idx + 1) % 5 == 0:
                print(f"  ✅ 已处理 {episode_idx + 1}/{len(files)} episodes")
        
        return total_frames
    
    # Process train files
    train_frames = process_files(train_files, "train")
    
    # Process test files
    test_frames = process_files(test_files, "test")
    
    # Save split information
    import json
    split_info_path = output_path / "split_info.json"
    with open(split_info_path, 'w') as f:
        json.dump({
            "episode_splits": episode_splits,
            "train_episodes": len(train_files),
            "test_episodes": len(test_files),
        }, f, indent=2)
    print(f"\n💾 Split信息已保存到: {split_info_path}")
    
    print(f"\n{'='*60}")
    print(f"🎉 转换完成!")
    print(f"{'='*60}")
    print(f"\n📊 最终统计:")
    print(f"  训练集:")
    print(f"    Episodes: {len(train_files)}")
    print(f"    总帧数: {train_frames}")
    print(f"  测试集:")
    print(f"    Episodes: {len(test_files)}")
    print(f"    总帧数: {test_frames}")
    print(f"  总计:")
    print(f"    Episodes: {len(train_files) + len(test_files)}")
    print(f"    总帧数: {train_frames + test_frames}")
    print(f"\n💾 数据集保存位置: {output_path}")
    print(f"📝 数据集 ID: {REPO_ID}")
    print(f"\n🚀 训练命令示例:")
    print(f"  python scripts/train.py --dataset.repo_id {REPO_ID}")


if __name__ == "__main__":
    main()
