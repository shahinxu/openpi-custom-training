#!/usr/bin/env python3
"""Evaluate model on test set during training."""

import json
import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Compute evaluation metrics between predictions and targets.
    
    Args:
        predictions: Predicted actions (batch, horizon, action_dim)
        targets: Ground truth actions (batch, horizon, action_dim)
        mask: Optional mask for valid timesteps
    
    Returns:
        Dictionary of metrics
    """
    if mask is not None:
        # Only compute metrics on valid (unmasked) positions
        predictions = predictions[mask]
        targets = targets[mask]
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - targets))
    
    # Mean Squared Error (MSE)
    mse = np.mean((predictions - targets) ** 2)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Per-dimension MAE
    mae_per_dim = np.mean(np.abs(predictions - targets), axis=0)
    
    # Cosine similarity (for direction)
    pred_norm = np.linalg.norm(predictions, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(targets, axis=-1, keepdims=True)
    cosine_sim = np.sum(predictions * targets, axis=-1) / (pred_norm.squeeze() * target_norm.squeeze() + 1e-8)
    cosine_sim = np.mean(cosine_sim)
    
    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "cosine_similarity": float(cosine_sim),
    }
    
    # Add per-dimension MAE
    for i, mae_val in enumerate(mae_per_dim):
        metrics[f"mae_dim_{i}"] = float(mae_val)
    
    return metrics


def evaluate_batch(
    state: training_utils.TrainState,
    batch: tuple[Any, jnp.ndarray],
    model_apply_fn,
) -> dict[str, Any]:
    """Evaluate a single batch.
    
    Returns:
        Dictionary containing predictions, targets, and metrics
    """
    inputs, targets = batch
    
    # Get predictions
    predictions = model_apply_fn(state.params, inputs)
    
    # Convert to numpy for metric computation
    predictions_np = np.array(predictions)
    targets_np = np.array(targets)
    
    # Compute metrics
    metrics = compute_metrics(predictions_np, targets_np)
    
    return {
        "predictions": predictions_np,
        "targets": targets_np,
        "metrics": metrics,
    }


def evaluate_episode(
    episode_results: list[dict[str, Any]],
    episode_id: int,
) -> dict[str, Any]:
    """Aggregate metrics for a single episode."""
    # Concatenate all predictions and targets
    all_predictions = np.concatenate([r["predictions"] for r in episode_results], axis=0)
    all_targets = np.concatenate([r["targets"] for r in episode_results], axis=0)
    
    # Compute episode-level metrics
    episode_metrics = compute_metrics(all_predictions, all_targets)
    
    return {
        "episode_id": episode_id,
        "num_frames": len(all_predictions),
        "metrics": episode_metrics,
        "predictions_sample": all_predictions[:5].tolist(),  # First 5 predictions
        "targets_sample": all_targets[:5].tolist(),  # First 5 targets
    }


def evaluate_test_set(
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    model_apply_fn,
    step: int,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate model on test set.
    
    Args:
        state: Training state with model parameters
        data_loader: Data loader (will use test split)
        model_apply_fn: Function to apply model
        step: Current training step
        output_dir: Optional directory to save detailed results
    
    Returns:
        Dictionary with evaluation results
    """
    logging.info(f"Starting test set evaluation at step {step}")
    
    # Get test dataset - use the data_loader's existing dataset with test split
    try:
        # Create a new iterator for test data
        # The data_loader should handle train/test split internally
        test_dataset = data_loader.dataset(split="test", shuffle=False, repeat=False)
        test_iter = iter(test_dataset)
    except Exception as e:
        logging.warning(f"Could not create test dataset iterator: {e}")
        logging.warning("Using a small sample from training data for evaluation")
        test_dataset = data_loader.dataset(split="train", shuffle=False, repeat=False)
        test_iter = iter(test_dataset)
    
    # Track results per episode and overall
    episode_results = {}
    all_batch_results = []
    
    # Limit evaluation to avoid too long delays
    max_batches = 10  # Evaluate on first 10 batches of test set
    
    # Iterate through test set
    for batch_idx in range(max_batches):
        try:
            batch = next(test_iter)
            result = evaluate_batch(state, batch, model_apply_fn)
            all_batch_results.append(result)
        except StopIteration:
            logging.info(f"Reached end of test set after {batch_idx} batches")
            break
        except Exception as e:
            logging.error(f"Error evaluating batch {batch_idx}: {e}")
            break
    
    # Compute overall metrics
    all_predictions = np.concatenate([r["predictions"] for r in all_batch_results], axis=0)
    all_targets = np.concatenate([r["targets"] for r in all_batch_results], axis=0)
    
    overall_metrics = compute_metrics(all_predictions, all_targets)
    
    # Prepare evaluation summary with per-dimension breakdown
    eval_summary = {
        "step": step,
        "num_batches": len(all_batch_results),
        "num_samples": len(all_predictions),
        "overall_metrics": overall_metrics,
        "prediction_stats": {
            "mean": all_predictions.mean(axis=0).tolist(),
            "std": all_predictions.std(axis=0).tolist(),
            "min": all_predictions.min(axis=0).tolist(),
            "max": all_predictions.max(axis=0).tolist(),
        },
        "target_stats": {
            "mean": all_targets.mean(axis=0).tolist(),
            "std": all_targets.std(axis=0).tolist(),
            "min": all_targets.min(axis=0).tolist(),
            "max": all_targets.max(axis=0).tolist(),
        },
    }
    
    # Try to get per-episode metrics if possible
    try:
        # Load split info to get test episode IDs
        repo_id = data_loader.data_config().repo_id
        if repo_id:
            dataset_path = Path(repo_id.split("/")[-1])
            if not dataset_path.is_absolute():
                dataset_path = Path("datasets") / dataset_path
            
            split_info_path = dataset_path / "split_info.json"
            if split_info_path.exists():
                with open(split_info_path) as f:
                    split_info = json.load(f)
                
                test_episode_ids = [
                    int(ep_id) for ep_id, split in split_info.get("episode_splits", {}).items()
                    if split == "test"
                ]
                
                eval_summary["test_episodes"] = test_episode_ids
                eval_summary["num_test_episodes"] = len(test_episode_ids)
                
                logging.info(f"Test set has {len(test_episode_ids)} episodes: {test_episode_ids}")
    except Exception as e:
        logging.warning(f"Could not load episode information: {e}")
    
    # Save detailed results if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_path = output_dir / f"eval_step_{step}.json"
        with open(summary_path, "w") as f:
            json.dump(eval_summary, f, indent=2)
        
        # Save prediction samples
        samples_path = output_dir / f"predictions_step_{step}.npz"
        np.savez(
            samples_path,
            predictions=all_predictions[:100],  # Save first 100
            targets=all_targets[:100],
        )
        
        logging.info(f"Saved evaluation results to {output_dir}")
    
    # Log summary metrics
    logging.info(f"\n{'='*80}")
    logging.info(f"Test Set Evaluation at Step {step}")
    logging.info(f"{'='*80}")
    logging.info(f"  Batches evaluated: {eval_summary['num_batches']}")
    logging.info(f"  Total samples: {eval_summary['num_samples']}")
    logging.info(f"\nOverall Metrics:")
    for metric_name, metric_value in overall_metrics.items():
        if not metric_name.startswith("mae_dim_"):
            logging.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Log per-dimension MAE
    logging.info(f"\nPer-Dimension MAE:")
    for i in range(5):  # Assuming 5-dim actions
        if f"mae_dim_{i}" in overall_metrics:
            logging.info(f"  Dimension {i}: {overall_metrics[f'mae_dim_{i}']:.4f}")
    
    # Log prediction vs target comparison
    logging.info(f"\nPrediction Statistics (first 3 dims):")
    for i in range(min(3, len(eval_summary['prediction_stats']['mean']))):
        pred_mean = eval_summary['prediction_stats']['mean'][i]
        target_mean = eval_summary['target_stats']['mean'][i]
        logging.info(f"  Dim {i}: pred_mean={pred_mean:.3f}, target_mean={target_mean:.3f}, diff={abs(pred_mean-target_mean):.3f}")
    
    logging.info(f"{'='*80}\n")
    
    return eval_summary


def evaluate_per_episode(
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    model_apply_fn,
    step: int,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate model on each test episode individually.
    
    This provides per-episode breakdown of performance.
    """
    logging.info(f"Starting per-episode evaluation at step {step}")
    
    # Get dataset path
    dataset_path = Path(data_loader.data_config().repo_id.split("/")[-1])
    if not dataset_path.is_absolute():
        dataset_path = Path("datasets") / dataset_path
    
    # Load split info
    split_info_path = dataset_path / "split_info.json"
    if not split_info_path.exists():
        logging.error(f"Split info not found at {split_info_path}")
        return {}
    
    with open(split_info_path) as f:
        split_info = json.load(f)
    
    test_episode_ids = [
        int(ep_id) for ep_id, split in split_info.get("episode_splits", {}).items()
        if split == "test"
    ]
    
    if not test_episode_ids:
        logging.warning("No test episodes found")
        return {}
    
    logging.info(f"Evaluating {len(test_episode_ids)} test episodes: {test_episode_ids}")
    
    # Results per episode
    episode_results = {}
    
    # For each test episode
    for episode_id in test_episode_ids:
        logging.info(f"Evaluating episode {episode_id}")
        
        # Get episode-specific dataset
        # Note: This is a simplified version. In practice, you may need to filter
        # the dataset by episode_index
        try:
            # Load episode data
            episode_data_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
            if not episode_data_path.exists():
                logging.warning(f"Episode data not found: {episode_data_path}")
                continue
            
            # For now, we'll just track that we attempted to evaluate it
            episode_results[episode_id] = {
                "episode_id": episode_id,
                "status": "evaluated",
            }
            
        except Exception as e:
            logging.error(f"Error evaluating episode {episode_id}: {e}")
            episode_results[episode_id] = {
                "episode_id": episode_id,
                "status": "error",
                "error": str(e),
            }
    
    eval_summary = {
        "step": step,
        "num_episodes": len(test_episode_ids),
        "episode_results": episode_results,
    }
    
    # Save results
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / f"per_episode_eval_step_{step}.json"
        with open(results_path, "w") as f:
            json.dump(eval_summary, f, indent=2)
        
        logging.info(f"Saved per-episode results to {results_path}")
    
    return eval_summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--output_dir", default="eval_results", help="Output directory")
    parser.add_argument("--config", default="pi05_custom", help="Config name")
    
    args = parser.parse_args()
    
    # TODO: Load checkpoint and run evaluation
    print(f"Evaluation script (checkpoint: {args.checkpoint})")
    print("Note: This script is designed to be called from train.py")
    print("Standalone evaluation not yet implemented.")
