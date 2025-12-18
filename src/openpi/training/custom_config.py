"""
Custom training configuration for pi0.5 LoRA fine-tuning.
This file will be automatically imported by the training system.
"""

import dataclasses
from pathlib import Path
from typing_extensions import override

from openpi.training import config as _config
from openpi.models import pi0_config, model as _model
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from openpi.transforms import Group as TransformsGroup, RepackTransform
from openpi.policies import libero_policy
@dataclasses.dataclass(frozen=True)
class LeRobotCustomDataConfig(_config.DataConfigFactory):
    """Configuration for custom segmented dataset."""
    
    @override
    def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        repack_transform = TransformsGroup(
            inputs=[
                RepackTransform({
                    "observation.images.cam_0": "image",
                    "observation.state": "state",
                    "action": "actions",
                })
            ]
        )
        
        data_transforms = TransformsGroup(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        
        model_transforms = _config.ModelTransformFactory()(model_config)
        
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
PI05_CUSTOM_CONFIG = _config.TrainConfig(
    name="pi05_custom",
    
    model=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=5,
        action_horizon=10,
        discrete_state_input=False,
    ),
    
    data=LeRobotCustomDataConfig(
        repo_id="rzh/openpi_segmented_data",
    ),
    
    batch_size=32,
    num_train_steps=10_000,
    
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=3e-4,
        decay_steps=10_000,
        decay_lr=1e-5,
    ),
    
    optimizer=_optimizer.AdamW(
        clip_gradient_norm=1.0,
        weight_decay=0.01,
    ),
    
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=5,
        action_horizon=10,
    ).get_freeze_filter(),
    
    ema_decay=None,
    
    log_interval=50,
    save_interval=1000,
    keep_period=5000,
)
_config._CONFIGS.append(PI05_CUSTOM_CONFIG)
