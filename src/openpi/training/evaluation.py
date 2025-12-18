"""Test set evaluation utilities for training."""

import csv
import json
import logging

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils


def evaluate_on_test_set(
    train_state: training_utils.TrainState,
    config: _config.TrainConfig,
    step: int,
    checkpoint_dir: epath.Path,
    mesh: jax.sharding.Mesh,
    rng: at.KeyArrayLike,
    data_loader: _data_loader.DataLoader,
) -> dict[str, float]:
    """Evaluate model on test set. Compute MSE for overall test set and per-episode."""
    logging.info(f"Starting test set evaluation at step {step}...")
    
    try:
        # Create evaluation directory
        eval_dir = checkpoint_dir / "eval_results"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Get metadata to determine test episodes
        info_path = epath.Path(config.data.local_dir) / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        
        # Parse train split
        train_split = info["splits"]["train"]
        train_start, train_end = map(int, train_split.split(":"))
        total_episodes = info["total_episodes"]
        test_episode_indices = list(range(train_end, total_episodes))
        
        logging.info(f"Test episodes: {train_end}-{total_episodes-1} ({len(test_episode_indices)} episodes)")
        
        # Reconstruct model
        model = nnx.merge(train_state.model_def, train_state.params)
        
        # Load dataset
        dataset = LeRobotDataset(
            repo_id=config.data.repo_id,
            root=config.data.local_dir,
        )
        
        # Group frames by episode
        logging.info("Grouping test data by episode...")
        episode_frames = {}
        for idx in range(len(dataset)):
            sample = dataset[idx]
            ep_idx = int(sample['episode_index'].item() if hasattr(sample['episode_index'], 'item') else sample['episode_index'])
            if ep_idx not in test_episode_indices:
                continue
            if ep_idx not in episode_frames:
                episode_frames[ep_idx] = []
            episode_frames[ep_idx].append(idx)
        
        logging.info(f"Found {len(episode_frames)} test episodes")
        
        # Collect MSE per episode (not storing predictions to save memory)
        episode_data = {}
        for ep_idx in sorted(episode_frames.keys()):
            episode_data[ep_idx] = {
                'episode_index': ep_idx,
                'mses': [],
                'num_frames': len(episode_frames[ep_idx]),
            }
        
        # Apply same transforms as training data
        logging.info("Creating transformed dataset for evaluation...")
        
        # Create torch dataset with same config
        torch_dataset = _data_loader.create_torch_dataset(
            config.data.create(config.assets_dirs, config.model),
            action_horizon=config.model.action_horizon,
            model_config=config.model
        )
        
        # Apply transforms
        transformed_dataset = _data_loader.transform_dataset(
            torch_dataset,
            config.data.create(config.assets_dirs, config.model),
            skip_norm_stats=False
        )
        
        # Process each test episode with model inference
        logging.info("Running model inference on test episodes...")
        for ep_idx in sorted(episode_frames.keys()):
            logging.info(f"  Evaluating episode {ep_idx} ({episode_data[ep_idx]['num_frames']} frames)...")
            frame_indices = episode_frames[ep_idx]
            
            # Process in batches for efficiency (reduced batch size to save memory)
            batch_size = 8
            for i in range(0, len(frame_indices), batch_size):
                batch_frame_indices = frame_indices[i:i+batch_size]
                
                # Get transformed samples
                batch_samples = [transformed_dataset[idx] for idx in batch_frame_indices]
                
                # Stack into batch
                batch_obs_dict = {}
                batch_actions_gt = []
                
                for sample in batch_samples:
                    # Extract action
                    batch_actions_gt.append(sample['actions'])
                    
                    # Extract observations (all keys except 'actions')
                    for key, value in sample.items():
                        if key != 'actions':
                            if key not in batch_obs_dict:
                                batch_obs_dict[key] = []
                            batch_obs_dict[key].append(value)
                
                # Stack tensors - handle nested dicts (e.g., 'image' contains multiple camera views)
                def stack_nested(values):
                    """Recursively stack values that might be nested dicts or tensors."""
                    if not values:
                        return None
                    first = values[0]
                    if isinstance(first, dict):
                        # Recursively stack nested dict
                        return {k: stack_nested([v[k] for v in values]) for k in first.keys()}
                    elif isinstance(first, torch.Tensor):
                        return torch.stack(values)
                    else:
                        # Convert to tensor, handling numpy types properly
                        tensor_values = []
                        for x in values:
                            if isinstance(x, torch.Tensor):
                                tensor_values.append(x)
                            elif isinstance(x, (np.ndarray, np.generic)):
                                # np.generic covers all numpy scalar types (np.bool_, np.int64, etc.)
                                tensor_values.append(torch.from_numpy(np.asarray(x)))
                            else:
                                tensor_values.append(torch.tensor(x))
                        return torch.stack(tensor_values)
                
                for key in batch_obs_dict:
                    batch_obs_dict[key] = stack_nested(batch_obs_dict[key])
                
                # Convert actions to tensors if needed
                batch_actions_gt = torch.stack([
                    torch.tensor(x) if not isinstance(x, torch.Tensor) else x 
                    for x in batch_actions_gt
                ])
                
                # Convert to JAX/numpy format
                batch_obs_jax = jax.tree.map(
                    lambda x: jnp.array(x.numpy()) if isinstance(x, torch.Tensor) else jnp.array(x),
                    batch_obs_dict
                )
                
                # Create Observation object from dict
                batch_obs_model = _model.Observation.from_dict(batch_obs_jax)
                
                # Run model prediction
                with sharding.set_mesh(mesh):
                    pred_rng = jax.random.fold_in(rng, ep_idx * 100000 + i)
                    actions_pred = model.sample_actions(pred_rng, batch_obs_model)
                
                # Convert to numpy
                actions_pred_np = np.array(jax.device_get(actions_pred))
                actions_gt_np = batch_actions_gt.numpy()
                
                # Compute MSE for each frame in batch (don't store predictions to save memory)
                for j in range(len(batch_samples)):
                    pred = actions_pred_np[j]
                    gt = actions_gt_np[j]
                    
                    # Only store MSE, not full predictions (saves memory)
                    mse = float(np.mean((pred - gt) ** 2))
                    episode_data[ep_idx]['mses'].append(mse)
                
                # Clear JAX cache to free memory
                jax.clear_caches()
        
        # Compute per-episode MSE
        for ep_idx, ep_data in episode_data.items():
            if ep_data['mses']:
                ep_data['mse'] = float(np.mean(ep_data['mses']))
                ep_data['num_frames'] = len(ep_data['mses'])
            else:
                ep_data['mse'] = 0.0
                ep_data['num_frames'] = 0
        
        # Compute overall MSE
        all_mses = []
        for ep_data in episode_data.values():
            all_mses.extend(ep_data['mses'])
        overall_mse = float(np.mean(all_mses)) if all_mses else 0.0
        
        logging.info(f"Evaluation complete: Overall MSE = {overall_mse:.6f}")
        
        # Save results (without predictions to save disk space)
        eval_summary = {
            "step": step,
            "overall_mse": overall_mse,
            "num_episodes": len(episode_data),
            "num_frames": len(all_mses),
            "per_episode_results": [
                {
                    'episode_index': ep_idx,
                    'num_frames': ep_data['num_frames'],
                    'mse': ep_data['mse'],

                }
                for ep_idx, ep_data in sorted(episode_data.items())
            ]
        }
        
        eval_file = eval_dir / f"eval_step_{step:06d}.json"
        with open(eval_file, "w") as f:
            json.dump(eval_summary, f, indent=2)
        
        # Save per-episode CSV
        csv_file = eval_dir / f"eval_step_{step:06d}_per_episode.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode_index", "num_frames", "mse"])
            writer.writeheader()
            for ep_idx, ep_data in sorted(episode_data.items()):
                writer.writerow({
                    "episode_index": ep_idx,
                    "num_frames": ep_data['num_frames'],
                    "mse": f"{ep_data['mse']:.6f}",
                })
        
        # Append to summary CSV
        summary_csv = eval_dir / "evaluation_summary.csv"
        file_exists = summary_csv.exists()
        with open(summary_csv, "a", newline="") as f:
            fieldnames = ["step", "overall_mse", "num_episodes", "num_frames"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "step": step,
                "overall_mse": f"{overall_mse:.6f}",
                "num_episodes": len(episode_data),
                "num_frames": len(all_mses),
            })
        
        # Log results
        logging.info(f"\n{'='*70}")
        logging.info(f"Test Set Evaluation - Step {step}")
        logging.info(f"{'='*70}")
        logging.info(f"  Test episodes: {len(episode_data)}")
        logging.info(f"  Total frames: {len(all_mses)}")
        logging.info(f"  Overall MSE: {overall_mse:.6f}")
        logging.info(f"\n  Per-Episode MSE:")
        for ep_idx, ep_data in sorted(episode_data.items()):
            logging.info(f"    Episode {ep_idx:2d}: {ep_data['mse']:.6f} ({ep_data['num_frames']} frames)")
        logging.info(f"{'='*70}\n")
        
        return {
            "test/mse": overall_mse,
            "test/num_episodes": len(episode_data),
            "test/num_frames": len(all_mses),
        }
        
    except Exception as e:
        logging.error(f"Evaluation failed at step {step}: {e}")
        import traceback
        traceback.print_exc()
        return {}
