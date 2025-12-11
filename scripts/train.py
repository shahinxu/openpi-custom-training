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
    step: int,
    checkpoint_dir: epath.Path,
    mesh: jax.sharding.Mesh,
    rng: at.KeyArrayLike,
    data_loader: _data_loader.DataLoader,
) -> dict[str, float]:
    """Evaluate model on test set. Compute MSE for overall test set and per-episode."""
    import json
    import csv
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    
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
        from openpi.training.data_loader import transform_dataset, create_torch_dataset
        
        # Create torch dataset with same config
        torch_dataset = create_torch_dataset(
            config.data.create(config.assets_dirs, config.model),
            action_horizon=config.model.action_horizon,
            model_config=config.model
        )
        
        # Apply transforms
        transformed_dataset = transform_dataset(
            torch_dataset,
            config.data.create(config.assets_dirs, config.model),
            skip_norm_stats=False
        )
        
        # Process each test episode with model inference
        logging.info("Running model inference on test episodes...")
        import torch
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
                # Sample format after transform: dict with observation keys + 'actions' key
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
                from openpi.models import model as model_lib
                batch_obs_model = model_lib.Observation.from_dict(batch_obs_jax)
                
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
                        step=step,
                        checkpoint_dir=checkpoint_manager.directory,
                        mesh=mesh,
                        rng=train_rng,
                        data_loader=data_loader,
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
