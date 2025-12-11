import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def evaluate_on_test_set(
    train_state: training_utils.TrainState,
    config: _config.TrainConfig,
    train_step_fn,
    step: int,
    checkpoint_dir: epath.Path,
    mesh: jax.sharding.Mesh,
    data_iter,  # Use the training data iterator
    rng: at.KeyArrayLike,  # Need random key for train_step
    max_eval_batches: int = 20,
) -> dict[str, float]:
    """Evaluate model on test set and return metrics including action prediction errors."""
    import json
    
    logging.info(f"Starting evaluation at step {step} (evaluating next {max_eval_batches} batches)...")
    
    try:
        # Create evaluation directory
        eval_dir = checkpoint_dir / "eval_results"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect action predictions
        action_errors = []  # Store per-sample action errors
        all_predictions = []  # Store all predictions for detailed analysis
        all_ground_truth = []  # Store all ground truth actions
        
        # Reconstruct model from train_state
        model = nnx.merge(train_state.model_def, train_state.params)
        
        for batch_count in range(max_eval_batches):
            try:
                # Get next batch from training iterator
                batch = next(data_iter)
                observation, actions_gt = batch
                
                # Sample actions from the model for action prediction metrics
                with sharding.set_mesh(mesh):
                    pred_rng = jax.random.fold_in(rng, batch_count)
                    actions_pred = model.sample_actions(pred_rng, observation)
                
                # Convert to numpy for metrics computation
                actions_pred_np = jax.device_get(actions_pred)
                actions_gt_np = jax.device_get(actions_gt)
                
                # Compute per-sample errors
                batch_size = actions_pred_np.shape[0]
                for i in range(batch_size):
                    pred = actions_pred_np[i]  # Shape: (horizon, action_dim)
                    gt = actions_gt_np[i]      # Shape: (horizon, action_dim)
                    
                    # Compute MSE and MAE for this sample
                    sample_mse = np.mean((pred - gt) ** 2)
                    sample_mae = np.mean(np.abs(pred - gt))
                    
                    # Compute per-dimension errors (averaged over horizon)
                    per_dim_mse = np.mean((pred - gt) ** 2, axis=0)
                    per_dim_mae = np.mean(np.abs(pred - gt), axis=0)
                    
                    action_errors.append({
                        "batch": batch_count,
                        "sample_idx": i,
                        "mse": float(sample_mse),
                        "mae": float(sample_mae),
                        "per_dim_mse": per_dim_mse.tolist(),
                        "per_dim_mae": per_dim_mae.tolist(),
                    })
                    
                    # Store predictions and ground truth for detailed comparison
                    all_predictions.append(pred.tolist())
                    all_ground_truth.append(gt.tolist())
                
                if (batch_count + 1) % 5 == 0:
                    logging.info(f"  Evaluated {batch_count + 1}/{max_eval_batches} batches")
                
            except StopIteration:
                logging.info(f"Data iterator exhausted after {batch_count} batches")
                break
            except Exception as e:
                logging.warning(f"Error in eval batch {batch_count}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        if not action_errors:
            logging.warning("No test batches evaluated")
            return {}
        
        # Compute overall metrics
        all_mses = [e["mse"] for e in action_errors]
        all_maes = [e["mae"] for e in action_errors]
        
        metrics = {
            "num_test_batches": len(action_errors) // 32,  # Approximate number of batches
            "num_samples": len(action_errors),
            # Action prediction metrics
            "action_mse": float(np.mean(all_mses)),
            "action_mse_std": float(np.std(all_mses)),
            "action_mae": float(np.mean(all_maes)),
            "action_mae_std": float(np.std(all_maes)),
        }
        
        # Compute per-dimension statistics
        num_dims = len(action_errors[0]["per_dim_mse"])
        per_dim_stats = {}
        for dim_idx in range(num_dims):
            dim_mses = [e["per_dim_mse"][dim_idx] for e in action_errors]
            dim_maes = [e["per_dim_mae"][dim_idx] for e in action_errors]
            per_dim_stats[f"dim_{dim_idx}_mse"] = float(np.mean(dim_mses))
            per_dim_stats[f"dim_{dim_idx}_mae"] = float(np.mean(dim_maes))
        
        metrics.update(per_dim_stats)
        
        # Save detailed evaluation results
        eval_summary = {
            "step": step,
            "metrics": metrics,
            "per_sample_errors": action_errors,  # Detailed per-sample errors
            "sample_predictions_vs_groundtruth": [
                {
                    "sample_idx": i,
                    "predicted": all_predictions[i],
                    "ground_truth": all_ground_truth[i],
                }
                for i in range(min(20, len(all_predictions)))  # Save first 20 samples for inspection
            ]
        }
        
        eval_file = eval_dir / f"eval_step_{step:06d}.json"
        with open(eval_file, "w") as f:
            json.dump(eval_summary, f, indent=2)
        
        # Save per-sample errors as CSV table for easy analysis
        import csv
        csv_file = eval_dir / f"eval_step_{step:06d}_per_sample.csv"
        with open(csv_file, "w", newline="") as f:
            if action_errors:
                # Create header with all dimension columns
                num_dims = len(action_errors[0]["per_dim_mse"])
                fieldnames = ["sample_idx", "batch", "mse", "mae"]
                for dim_idx in range(num_dims):
                    fieldnames.extend([f"dim_{dim_idx}_mse", f"dim_{dim_idx}_mae"])
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for err in action_errors:
                    row = {
                        "sample_idx": err["sample_idx"],
                        "batch": err["batch"],
                        "mse": f"{err['mse']:.6f}",
                        "mae": f"{err['mae']:.6f}",
                    }
                    for dim_idx in range(num_dims):
                        row[f"dim_{dim_idx}_mse"] = f"{err['per_dim_mse'][dim_idx]:.6f}"
                        row[f"dim_{dim_idx}_mae"] = f"{err['per_dim_mae'][dim_idx]:.6f}"
                    writer.writerow(row)
        
        # Append summary metrics to a cumulative CSV file across all evaluations
        summary_csv = eval_dir / "evaluation_summary.csv"
        file_exists = summary_csv.exists()
        with open(summary_csv, "a", newline="") as f:
            fieldnames = ["step"] + list(metrics.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            row = {"step": step}
            row.update({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in metrics.items()})
            writer.writerow(row)
        
        # Log to console
        logging.info(f"\n{'='*70}")
        logging.info(f"Test Set Evaluation - Step {step}")
        logging.info(f"{'='*70}")
        logging.info(f"  Samples evaluated: {metrics['num_samples']}")
        logging.info(f"")
        logging.info(f"  Action Prediction Metrics:")
        logging.info(f"    MSE: {metrics['action_mse']:.6f} ± {metrics['action_mse_std']:.6f}")
        logging.info(f"    MAE: {metrics['action_mae']:.6f} ± {metrics['action_mae_std']:.6f}")
        logging.info(f"")
        logging.info(f"  Per-Dimension MSE:")
        for dim_idx in range(num_dims):
            logging.info(f"    Dim {dim_idx}: {metrics[f'dim_{dim_idx}_mse']:.6f} (MAE: {metrics[f'dim_{dim_idx}_mae']:.6f})")
        logging.info(f"{'='*70}\n")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Evaluation failed at step {step}: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            
            # Run evaluation on test set
            if step > 0:
                try:
                    eval_metrics = evaluate_on_test_set(
                        train_state=train_state,
                        config=config,
                        train_step_fn=functools.partial(train_step, config),
                        step=step,
                        checkpoint_dir=checkpoint_manager.directory,
                        mesh=mesh,
                        data_iter=data_iter,
                        rng=train_rng,  # Pass the training random key
                        max_eval_batches=20,
                    )
                    
                    # Log to wandb with eval/ prefix
                    if eval_metrics:
                        wandb_eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        wandb.log(wandb_eval_metrics, step=step)
                except Exception as e:
                    logging.error(f"Evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
