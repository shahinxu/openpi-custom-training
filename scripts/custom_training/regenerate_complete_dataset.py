import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import subprocess
import tempfile


def load_h5_data(h5_path):
    """从H5文件加载数据"""
    with h5py.File(h5_path, "r") as f:
        images = np.array(f["observations"]["images"])
        states = np.array(f["observations"]["states"])
        actions = np.array(f["actions"])
    
    # 转换图像为uint8
    if images.dtype != np.uint8:
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8)
    
    # 移除多余维度
    if images.ndim == 5 and images.shape[1] == 1 and images.shape[2] == 1:
        images = images.squeeze(axis=(1, 2))
    
    return {"images": images, "states": states, "actions": actions}


def load_all_data(train_dir, test_dir):
    """加载所有训练和测试数据"""
    all_data = []
    
    # 加载训练数据（从子目录）
    train_path = Path(train_dir)
    subdirs = sorted([d for d in train_path.iterdir() if d.is_dir()])
    
    for subdir in subdirs:
        h5_files = sorted(subdir.glob("*.h5"))
        for h5_file in h5_files:
            data = load_h5_data(h5_file)
            episode_name = f"{subdir.name}/{h5_file.stem}"
            all_data.append({
                "name": episode_name,
                "data": data,
                "split": "train"
            })
    
    # 加载测试数据
    test_path = Path(test_dir)
    test_h5_files = sorted(test_path.glob("*.h5"))
    
    for h5_file in test_h5_files:
        data = load_h5_data(h5_file)
        all_data.append({
            "name": h5_file.stem,
            "data": data,
            "split": "test"
        })
    
    return all_data


def generate_dataset(all_data, output_dir):
    """生成完整的LeRobot格式数据集"""
    output_path = Path(output_dir)
    
    # 创建目录 - 使用chunk格式
    chunks_size = 1000
    data_dir = output_path / "data" / "chunk-000"
    video_dir = output_path / "videos" / "chunk-000" / "observation.images.cam_0"
    meta_dir = output_path / "meta"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集统计信息
    all_actions = []
    all_states = []
    tasks = []
    episodes = []
    episodes_stats = []
    
    global_frame_idx = 0
    fps = 30.0
    
    print(f"\n生成数据集: {len(all_data)} episodes")
    
    for ep_idx, item in enumerate(tqdm(all_data, desc="转换episodes")):
        data = item["data"]
        n_frames = len(data["images"])
        
        # 收集统计信息
        all_actions.extend(data["actions"].tolist())
        all_states.extend(data["states"].tolist())
        
        # 任务信息
        task_info = {
            "task_index": ep_idx,
            "task": item["name"]
        }
        tasks.append(task_info)
        
        # Episode信息
        episode_info = {
            "episode_index": ep_idx,
            "tasks": [item["name"]],
            "length": n_frames
        }
        episodes.append(episode_info)
        
        # Episode统计 - 完整格式匹配LeRobot要求
        # 图像数据需要归一化到[0,1]
        images_normalized = data["images"].astype(np.float32) / 255.0
        
        episode_stats = {
            "episode_index": ep_idx,
            "stats": {
                "action": {
                    "min": data["actions"].min(axis=0).tolist(),
                    "max": data["actions"].max(axis=0).tolist(),
                    "mean": data["actions"].mean(axis=0).tolist(),
                    "std": data["actions"].std(axis=0).tolist(),
                    "count": [n_frames]
                },
                "observation.state": {
                    "min": data["states"].min(axis=0).tolist(),
                    "max": data["states"].max(axis=0).tolist(),
                    "mean": data["states"].mean(axis=0).tolist(),
                    "std": data["states"].std(axis=0).tolist(),
                    "count": [n_frames]
                },
                "observation.images.cam_0": {
                    "min": [[[float(x)]] for x in images_normalized.min(axis=(0, 1, 2))],
                    "max": [[[float(x)]] for x in images_normalized.max(axis=(0, 1, 2))],
                    "mean": [[[float(x)]] for x in images_normalized.mean(axis=(0, 1, 2))],
                    "std": [[[float(x)]] for x in images_normalized.std(axis=(0, 1, 2))],
                    "count": [n_frames]
                },
                "timestamp": {
                    "min": [0.0],
                    "max": [(n_frames - 1) / fps],
                    "mean": [(n_frames - 1) / (2 * fps)],
                    "std": [float(np.arange(n_frames).std() / fps)],
                    "count": [n_frames]
                },
                "frame_index": {
                    "min": [0],
                    "max": [n_frames - 1],
                    "mean": [float((n_frames - 1) / 2)],
                    "std": [float(np.arange(n_frames).std())],
                    "count": [n_frames]
                },
                "episode_index": {
                    "min": [ep_idx],
                    "max": [ep_idx],
                    "mean": [float(ep_idx)],
                    "std": [0.0],
                    "count": [n_frames]
                },
                "index": {
                    "min": [global_frame_idx],
                    "max": [global_frame_idx + n_frames - 1],
                    "mean": [float(global_frame_idx + (n_frames - 1) / 2)],
                    "std": [float(np.arange(n_frames).std())],
                    "count": [n_frames]
                },
                "task_index": {
                    "min": [ep_idx],
                    "max": [ep_idx],
                    "mean": [float(ep_idx)],
                    "std": [0.0],
                    "count": [n_frames]
                }
            }
        }
        episodes_stats.append(episode_stats)
        
        # 生成parquet数据 - 关键：timestamp从0开始！
        # 注意：不包含图像列，图像从视频文件加载
        df_data = {
            "action": list(data["actions"]),
            "observation.state": list(data["states"]),
            "timestamp": list(np.arange(n_frames) / fps),  # 每个episode从0开始
            "frame_index": list(range(n_frames)),  # 每个episode从0开始
            "episode_index": [ep_idx] * n_frames,
            "index": list(range(global_frame_idx, global_frame_idx + n_frames)),  # 全局索引
            "task_index": [ep_idx] * n_frames,
        }
        df = pd.DataFrame(df_data)
        df.to_parquet(data_dir / f"episode_{ep_idx:06d}.parquet", engine='pyarrow')
        
        # 生成视频 - 使用临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # 保存图像为PNG
            for frame_idx, img in enumerate(data["images"]):
                img_path = tmpdir_path / f"frame_{frame_idx:06d}.png"
                Image.fromarray(img).save(img_path)
            
            # 使用ffmpeg生成视频
            video_path = video_dir / f"episode_{ep_idx:06d}.mp4"
            cmd = [
                "ffmpeg", "-y", "-framerate", "30",
                "-i", str(tmpdir_path / "frame_%06d.png"),
                "-vframes", str(n_frames),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "slow",
                "-crf", "18",
                str(video_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
        
        global_frame_idx += n_frames
    
    # 计算normalization统计
    actions_np = np.array(all_actions)
    states_np = np.array(all_states)
    
    norm_stats = {
        "norm_stats": {
            "action": {
                "mean": actions_np.mean(axis=0).tolist(),
                "std": actions_np.std(axis=0).tolist(),
                "min": actions_np.min(axis=0).tolist(),
                "max": actions_np.max(axis=0).tolist(),
                "q01": np.quantile(actions_np, 0.01, axis=0).tolist(),
                "q99": np.quantile(actions_np, 0.99, axis=0).tolist()
            },
            "state": {
                "mean": states_np.mean(axis=0).tolist(),
                "std": states_np.std(axis=0).tolist(),
                "min": states_np.min(axis=0).tolist(),
                "max": states_np.max(axis=0).tolist(),
                "q01": np.quantile(states_np, 0.01, axis=0).tolist(),
                "q99": np.quantile(states_np, 0.99, axis=0).tolist()
            }
        }
    }
    
    # 保存metadata
    with open(output_path / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for episode in episodes:
            f.write(json.dumps(episode) + "\n")
    
    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for stats in episodes_stats:
            f.write(json.dumps(stats) + "\n")
    
    # 计算唯一任务数量
    unique_tasks = len(tasks)
    
    # info.json - 完全匹配参考格式
    info = {
        "codebase_version": "v2.1",
        "robot_type": "custom",
        "total_episodes": len(all_data),
        "total_frames": global_frame_idx,
        "total_tasks": unique_tasks,
        "total_videos": len(all_data),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": int(fps),
        "splits": {
            "train": f"0:{sum(1 for x in all_data if x['split'] == 'train')}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32", 
                "shape": [actions_np.shape[1]], 
                "names": None
            },
            "observation.state": {
                "dtype": "float32", 
                "shape": [states_np.shape[1]], 
                "names": None
            },
            "observation.images.cam_0": {
                "dtype": "video", 
                "shape": [3, 224, 224], 
                "names": ["channel", "height", "width"],
                "info": {
                    "video.height": 224,
                    "video.width": 224,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": int(fps),
                    "video.channels": 3,
                    "has_audio": False
                }
            },
            "timestamp": {
                "dtype": "float32", 
                "shape": [1], 
                "names": None
            },
            "frame_index": {
                "dtype": "int64", 
                "shape": [1], 
                "names": None
            },
            "episode_index": {
                "dtype": "int64", 
                "shape": [1], 
                "names": None
            },
            "index": {
                "dtype": "int64", 
                "shape": [1], 
                "names": None
            },
            "task_index": {
                "dtype": "int64", 
                "shape": [1], 
                "names": None
            }
        }
    }
    
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ 数据集生成完成:")
    print(f"  总episodes: {len(all_data)}")
    print(f"  训练episodes: {sum(1 for x in all_data if x['split'] == 'train')}")
    print(f"  测试episodes: {sum(1 for x in all_data if x['split'] == 'test')}")
    print(f"  总帧数: {global_frame_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="/home/rzh/zhenx/openpi/dataset_train")
    parser.add_argument("--test_dir", default="/home/rzh/zhenx/openpi/dataset_test")
    parser.add_argument("--output_dir", default="/home/rzh/zhenx/openpi/data/train_test_dataset")
    args = parser.parse_args()
    
    # 加载所有数据
    all_data = load_all_data(args.train_dir, args.test_dir)
    
    # 生成数据集
    generate_dataset(all_data, args.output_dir)
