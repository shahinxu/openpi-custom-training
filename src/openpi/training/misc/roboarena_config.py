
from typing import TypeAlias

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.droid_policy as droid_policy
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
def get_roboarena_configs():
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import SimpleDataConfig
    from openpi.training.config import TrainConfig

    return [
        #
        #
        TrainConfig(
            name="paligemma_binning_droid",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=15,
                max_token_len=400,
                fast_model_tokenizer=_tokenizer.BinningTokenizer,
            ),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            name="paligemma_fast_droid",
            model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=15),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            name="paligemma_fast_specialist_droid",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=15,
                fast_model_tokenizer=_tokenizer.FASTTokenizer,
                fast_model_tokenizer_kwargs={"fast_tokenizer_path": "KarlP/fast_droid_specialist"},
            ),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            name="paligemma_vq_droid",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=15,
                fast_model_tokenizer=_tokenizer.FSQTokenizer,
                fast_model_tokenizer_kwargs={"fsq_tokenizer_path": "gs://openpi-assets/tokenizers/droid_fsq_tokenizer"},
            ),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            name="paligemma_diffusion_droid",
            model=pi0_config.Pi0Config(action_horizon=10, action_dim=8),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
    ]
