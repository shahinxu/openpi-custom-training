
import csv
import json
import logging
from collections import defaultdict

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
    test_loader: _data_loader.DataLoader | None = None,
    save_predictions: bool = True,  # Save prediction sequences
) -> dict[str, float]:
    """Evaluate model on test set. Compute MSE for overall test set and per-episode.
    
    Args:
        test_loader: Optional pre-created test set DataLoader for faster evaluation.
                     If None, will load data from scratch (slower but more flexible).
        save_predictions: If True, save predicted and ground truth action sequences to disk.
    """
    logging.info(f"Starting test set evaluation at step {step}...")
    
    try:
        eval_dir = checkpoint_dir / "eval_results"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        model = nnx.merge(train_state.model_def, train_state.params)
        model.eval()  # Set to eval mode
        
        if test_loader is not None:
            return _evaluate_with_dataloader(
                model=model,
                test_loader=test_loader,
                step=step,
                eval_dir=eval_dir,
                mesh=mesh,
                rng=rng,
                save_predictions=save_predictions,
            )
        
        return _evaluate_from_scratch(
            model=model,
            config=config,
            step=step,
            eval_dir=eval_dir,
            mesh=mesh,
            rng=rng,
            save_predictions=save_predictions,
        )
        
    except Exception as e:
        logging.error(f"Evaluation failed at step {step}: {e}")
        import traceback
        traceback.print_exc()
        return {}
def _evaluate_with_dataloader(
    model: _model.BaseModel,
    test_loader: _data_loader.DataLoader,
    step: int,
    eval_dir: epath.Path,
    mesh: jax.sharding.Mesh,
    rng: at.KeyArrayLike,
    save_predictions: bool = True,
) -> dict[str, float]:
    """Fast evaluation using pre-created test DataLoader."""
    logging.info("Using fast evaluation with DataLoader...")
    
    dataset = test_loader._torch_dataset.hf_dataset
    episode_indices = dataset['episode_index']
    
    episode_data = defaultdict(lambda: {'mses': [], 'episode_index': 0})
    
    all_mses = []
    batch_count = 0
    
    for batch_obs, batch_actions_gt in test_loader:
        batch_count += 1
        if batch_count % 10 == 0:
            logging.info(f"  Processed {batch_count} batches...")
        
        batch_episode_indices = episode_indices[
            batch_count * test_loader.batch_size : (batch_count + 1) * test_loader.batch_size
        ] if hasattr(episode_indices, '__getitem__') else None
        
        with sharding.set_mesh(mesh):
            pred_rng = jax.random.fold_in(rng, batch_count)
            actions_pred = model.sample_actions(pred_rng, batch_obs)
        
        actions_pred_np = np.array(jax.device_get(actions_pred))
        actions_gt_np = np.array(jax.device_get(batch_actions_gt))
        
        for i in range(len(actions_pred_np)):
            mse = float(np.mean((actions_pred_np[i] - actions_gt_np[i]) ** 2))
            all_mses.append(mse)
            
            if batch_episode_indices is not None:
                ep_idx = int(batch_episode_indices[i])
                episode_data[ep_idx]['mses'].append(mse)
                episode_data[ep_idx]['episode_index'] = ep_idx
    
    overall_mse = float(np.mean(all_mses)) if all_mses else 0.0
    
    for ep_idx, ep_data in episode_data.items():
        if ep_data['mses']:
            ep_data['mse'] = float(np.mean(ep_data['mses']))
            ep_data['num_frames'] = len(ep_data['mses'])
        else:
            ep_data['mse'] = 0.0
            ep_data['num_frames'] = 0
    
    logging.info(f"Fast evaluation complete: Overall MSE = {overall_mse:.6f}")
    
    _save_eval_results(episode_data, overall_mse, all_mses, step, eval_dir)
    
    return {
        "test/mse": overall_mse,
        "test/num_episodes": len(episode_data) if episode_data else 0,
        "test/num_frames": len(all_mses),
    }
def _evaluate_from_scratch(
    model: _model.BaseModel,
    config: _config.TrainConfig,
    step: int,
    eval_dir: epath.Path,
    mesh: jax.sharding.Mesh,
    rng: at.KeyArrayLike,
    save_predictions: bool = True,
) -> dict[str, float]:
    """Slow evaluation path - loads data from scratch."""
    logging.info("Using slow evaluation (loading from scratch)...")
    
    info_path = epath.Path(config.data.local_dir) / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    
    train_split = info["splits"]["train"]
    train_start, train_end = map(int, train_split.split(":"))
    total_episodes = info["total_episodes"]
    test_episode_indices = list(range(train_end, total_episodes))
    
    logging.info(f"Test episodes: {train_end}-{total_episodes-1} ({len(test_episode_indices)} episodes)")
    
    dataset = LeRobotDataset(
        repo_id=config.data.repo_id,
        root=config.data.local_dir,
    )
    
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
    
    episode_data = {}
    for ep_idx in sorted(episode_frames.keys()):
        episode_data[ep_idx] = {
            'episode_index': ep_idx,
            'mses': [],
            'num_frames': len(episode_frames[ep_idx]),
            'predictions': [] if save_predictions else None,
            'ground_truth': [] if save_predictions else None,
        }
    
    logging.info("Creating transformed dataset for evaluation...")
    
    torch_dataset = _data_loader.create_torch_dataset(
        config.data.create(config.assets_dirs, config.model),
        action_horizon=config.model.action_horizon,
        model_config=config.model
    )
    
    transformed_dataset = _data_loader.transform_dataset(
        torch_dataset,
        config.data.create(config.assets_dirs, config.model),
        skip_norm_stats=False
    )
    
    logging.info("Running model inference on test episodes...")
    for ep_idx in sorted(episode_frames.keys()):
        logging.info(f"  Evaluating episode {ep_idx} ({episode_data[ep_idx]['num_frames']} frames)...")
        frame_indices = episode_frames[ep_idx]
        
        batch_size = 16
        for i in range(0, len(frame_indices), batch_size):
            batch_frame_indices = frame_indices[i:i+batch_size]
            
            batch_samples = [transformed_dataset[idx] for idx in batch_frame_indices]
            
            batch_obs_dict = {}
            batch_actions_gt = []
            
            for sample in batch_samples:
                batch_actions_gt.append(sample['actions'])
                
                for key, value in sample.items():
                    if key != 'actions':
                        if key not in batch_obs_dict:
                            batch_obs_dict[key] = []
                        batch_obs_dict[key].append(value)
            
            def stack_nested(values):
                if not values:
                    return None
                first = values[0]
                if isinstance(first, dict):
                    return {k: stack_nested([v[k] for v in values]) for k in first.keys()}
                elif isinstance(first, torch.Tensor):
                    return torch.stack(values)
                else:
                    tensor_values = []
                    for x in values:
                        if isinstance(x, torch.Tensor):
                            tensor_values.append(x)
                        elif isinstance(x, (np.ndarray, np.generic)):
                            tensor_values.append(torch.from_numpy(np.asarray(x)))
                        else:
                            tensor_values.append(torch.tensor(x))
                    return torch.stack(tensor_values)
            
            for key in batch_obs_dict:
                batch_obs_dict[key] = stack_nested(batch_obs_dict[key])
            
            batch_actions_gt = torch.stack([
                torch.tensor(x) if not isinstance(x, torch.Tensor) else x 
                for x in batch_actions_gt
            ])
            
            batch_obs_jax = jax.tree.map(
                lambda x: jnp.array(x.numpy()) if isinstance(x, torch.Tensor) else jnp.array(x),
                batch_obs_dict
            )
            
            batch_obs_model = _model.Observation.from_dict(batch_obs_jax)
            
            with sharding.set_mesh(mesh):
                pred_rng = jax.random.fold_in(rng, ep_idx * 100000 + i)
                actions_pred = model.sample_actions(pred_rng, batch_obs_model)
            
            actions_pred_np = np.array(jax.device_get(actions_pred))
            actions_gt_np = batch_actions_gt.numpy()
            
            for j in range(len(batch_samples)):
                pred = actions_pred_np[j]
                gt = actions_gt_np[j]
                
                if save_predictions:
                    episode_data[ep_idx]['predictions'].append(pred.tolist())
                    episode_data[ep_idx]['ground_truth'].append(gt.tolist())
                mse = float(np.mean((pred - gt) ** 2))
                episode_data[ep_idx]['mses'].append(mse)
            
            jax.clear_caches()
        
    for ep_idx, ep_data in episode_data.items():
        if ep_data['mses']:
            ep_data['mse'] = float(np.mean(ep_data['mses']))
            ep_data['num_frames'] = len(ep_data['mses'])
        else:
            ep_data['mse'] = 0.0
            ep_data['num_frames'] = 0
    
    all_mses = []
    for ep_data in episode_data.values():
        all_mses.extend(ep_data['mses'])
    overall_mse = float(np.mean(all_mses)) if all_mses else 0.0
    
    logging.info(f"Evaluation complete: Overall MSE = {overall_mse:.6f}")
    
    _save_eval_results(episode_data, overall_mse, all_mses, step, eval_dir, save_predictions)
    
    return {
        "test/mse": overall_mse,
        "test/num_episodes": len(episode_data),
        "test/num_frames": len(all_mses),
    }
def _save_eval_results(episode_data, overall_mse, all_mses, step, eval_dir, save_predictions=False):
    """Save evaluation results to disk."""
    per_episode_results = []
    for ep_idx, ep_data in sorted(episode_data.items()):
        result = {
            'episode_index': ep_idx,
            'num_frames': ep_data['num_frames'],
            'mse': ep_data['mse'],
        }
        if save_predictions and ep_data.get('predictions') is not None:
            result['predictions'] = ep_data['predictions']
            result['ground_truth'] = ep_data['ground_truth']
        per_episode_results.append(result)
    
    eval_summary = {
        "step": step,
        "overall_mse": overall_mse,
        "num_episodes": len(episode_data),
        "num_frames": len(all_mses),
        "per_episode_results": per_episode_results,
    }
    
    eval_file = eval_dir / f"eval_step_{step:06d}.json"
    with open(eval_file, "w") as f:
        json.dump(eval_summary, f, indent=2)
    
    if save_predictions:
        predictions_dir = eval_dir / f"predictions_step_{step:06d}"
        predictions_dir.mkdir(exist_ok=True)
        
        for ep_idx, ep_data in episode_data.items():
            if ep_data.get('predictions') is not None:
                np.savez(
                    predictions_dir / f"episode_{ep_idx:03d}.npz",
                    predictions=np.array(ep_data['predictions']),
                    ground_truth=np.array(ep_data['ground_truth']),
                    mse=ep_data['mse'],
                    episode_index=ep_idx,
                )
        logging.info(f"✓ Saved predictions to: {predictions_dir}")
    
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
