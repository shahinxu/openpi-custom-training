"""
Training script for pi0.5 LoRA fine-tuning on custom segmented data.

Usage:
uv run scripts/train.py pi05_custom

This will use the custom config defined in this file.
"""

import dataclasses
from pathlib import Path
from typing_extensions import override

# Add custom config to the training configs
from openpi.training import config as _config
from openpi.models import pi0_config, model as _model
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from openpi.transforms import Group as TransformsGroup, RepackTransform
from openpi.policies import libero_policy


# Create a custom data config factory
@dataclasses.dataclass(frozen=True)
class LeRobotCustomDataConfig(_config.DataConfigFactory):
    """Configuration for custom segmented dataset."""
    
    @override
    def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        # Repack transform to match dataset keys to inference keys
        repack_transform = TransformsGroup(
            inputs=[
                RepackTransform({
                    "observation.images.cam_0": "image",  # Use image instead of cam_0
                    "observation.state": "state",
                    "action": "actions",  # Note: dataset uses 'action', but model expects 'actions'
                })
            ]
        )
        
        # Use similar data transforms as Libero
        data_transforms = TransformsGroup(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        
        # Model transforms
        model_transforms = _config.ModelTransformFactory()(model_config)
        
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


# Custom training config
PI05_CUSTOM_CONFIG = _config.TrainConfig(
    name="pi05_custom",
    
    # Pi0.5 model with LoRA enabled
    model=pi0_config.Pi0Config(
        pi05=True,  # Use pi0.5
        paligemma_variant="gemma_2b_lora",  # Enable LoRA on main model
        action_expert_variant="gemma_300m_lora",  # Enable LoRA on action expert
        action_dim=5,  # Match your data (5D action space)
        action_horizon=10,  # Predict 10 steps ahead
        discrete_state_input=False,  # Continuous state
    ),
    
    # Dataset configuration
    data=LeRobotCustomDataConfig(
        repo_id="rzh/openpi_segmented_data",
    ),
    
    # Training hyperparameters
    batch_size=32,  # Adjust based on your GPU memory
    num_train_steps=10_000,  # Number of training steps
    
    # Learning rate schedule
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=3e-4,  # Higher LR for LoRA
        decay_steps=10_000,
        decay_lr=1e-5,
    ),
    
    # Optimizer
    optimizer=_optimizer.AdamW(
        clip_gradient_norm=1.0,
        weight_decay=0.01,
    ),
    
    # Load pre-trained pi0.5 base weights
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    
    # LoRA: freeze all parameters except LoRA adapters
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=5,
        action_horizon=10,
    ).get_freeze_filter(),
    
    # Disable EMA for LoRA fine-tuning
    ema_decay=None,
    
    # Checkpoint and logging
    log_interval=50,
    save_interval=1000,
    keep_period=5000,
)


# Add the config to the global registry
_config._CONFIGS.append(PI05_CUSTOM_CONFIG)


if __name__ == "__main__":
    print("=" * 80)
    print("Custom Pi0.5 LoRA Training Configuration")
    print("=" * 80)
    print("\n📝 To run training, use:")
    print(f"    uv run scripts/train.py pi05_custom")
    print("\n📝 Configuration:")
    print(f"  Dataset: {PI05_CUSTOM_CONFIG.data.repo_id}")
    print(f"  Model: Pi0.5 with LoRA")
    print(f"  Action dim: {PI05_CUSTOM_CONFIG.model.action_dim}")
    print(f"  Action horizon: {PI05_CUSTOM_CONFIG.model.action_horizon}")
    print(f"  Batch size: {PI05_CUSTOM_CONFIG.batch_size}")
    print(f"  Training steps: {PI05_CUSTOM_CONFIG.num_train_steps}")
    print(f"  Peak LR: {PI05_CUSTOM_CONFIG.lr_schedule.peak_lr}")
    print("=" * 80)
