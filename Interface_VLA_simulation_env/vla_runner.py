from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi.models import model as model_lib
from openpi.models import pi0_config, tokenizer as tokenizer_lib
from openpi.shared import normalize as norm_lib
from openpi.transforms import Normalize, Unnormalize

from .config import ActionMappingConfig, VLACheckpointConfig


@dataclass
class _ObservationBundle:
    images: dict[str, np.ndarray]
    state: np.ndarray


class VLACheckpointRunner:
    """封装 Pi0.5 checkpoint 推理，负责处理观测、指令以及动作映射。"""

    _IMAGE_KEYS = model_lib.IMAGE_KEYS
    _IMAGE_SHAPE = model_lib.IMAGE_RESOLUTION
    _STATE_CANDIDATE_KEYS = ("state", "proprio", "agent_state", "robot_state", "qpos", "eef_pose")
    _IMAGE_CANDIDATE_KEYS = ("image", "images", "rgb", "camera", "cam")

    def __init__(self, cfg: VLACheckpointConfig):
        self._cfg = cfg
        self._rng = jax.random.key(cfg.rng_seed)
        self._tokenizer = tokenizer_lib.PaligemmaTokenizer(max_len=cfg.max_token_len)
        self._model = self._load_model(cfg)
        self._state_norm, self._action_norm = self._load_norm_stats(cfg)
        self._state_normalizer = (
            Normalize({"state": self._state_norm}, use_quantiles=cfg.use_quantile_stats) if self._state_norm else None
        )
        self._action_denorm = (
            Unnormalize({"action": self._action_norm}, use_quantiles=cfg.use_quantile_stats) if self._action_norm else None
        )

    def predict(
        self,
        observation: Mapping[str, Any] | None,
        instruction: str | None = None,
        *,
        robot_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """使用 checkpoint 计算单步动作。"""

        prompt = instruction or self._cfg.instruction
        obs_bundle = self._prepare_model_inputs(observation, robot_state)
        model_obs = self._to_model_observation(obs_bundle, prompt)
        self._rng, sample_key = jax.random.split(self._rng)
        action_traj = self._model.sample_actions(
            sample_key,
            model_obs,
            num_steps=self._cfg.num_denoising_steps,
        )
        action_np = np.asarray(action_traj)[0, 0]
        if self._action_denorm is not None:
            action_np = self._denormalize_action(action_np)
        return self._map_to_simulator(action_np)

    def reset(self):
        """重置内部随机种子（可选）。"""

        self._rng = jax.random.key(self._cfg.rng_seed)

    def _load_model(self, cfg: VLACheckpointConfig) -> model_lib.BaseModel:
        model_cfg = pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant=cfg.paligemma_variant,
            action_expert_variant=cfg.action_expert_variant,
            action_dim=cfg.action_dim,
            action_horizon=cfg.action_horizon,
            max_token_len=cfg.max_token_len,
            discrete_state_input=cfg.discrete_state_input,
        )
        params = model_lib.restore_params(cfg.checkpoint_dir / "train_state")
        return model_cfg.load(params)

    def _load_norm_stats(
        self, cfg: VLACheckpointConfig
    ) -> tuple[norm_lib.NormStats | None, norm_lib.NormStats | None]:
        stats_path = cfg.norm_stats_dir
        if not stats_path.exists():
            return None, None
        stats = norm_lib.load(stats_path)
        return stats.get("state"), stats.get("action")

    def _prepare_model_inputs(
        self,
        observation: Mapping[str, Any] | None,
        robot_state: np.ndarray | None,
    ) -> _ObservationBundle:
        images = self._extract_images(observation)
        state = self._extract_state(observation)
        if state is None:
            state = robot_state
        if state is None:
            raise ValueError("无法从观测或 robot_state 中提取状态信息")
        processed_images = self._prepare_images(images)
        processed_state = self._prepare_state(state)
        return _ObservationBundle(images=processed_images, state=processed_state)

    def _to_model_observation(self, bundle: _ObservationBundle, prompt: str) -> model_lib.Observation:
        image_tensors = {}
        mask_tensors = {}
        for key in self._IMAGE_KEYS:
            image_tensors[key] = jnp.asarray(bundle.images[key][None, ...], dtype=jnp.float32)
            mask_tensors[key] = jnp.ones((1,), dtype=jnp.bool_)
        tokenized_prompt, prompt_mask = self._tokenizer.tokenize(
            prompt,
            bundle.state if self._cfg.discrete_state_input else None,
        )
        state = bundle.state[None, ...].astype(np.float32)
        return model_lib.Observation(
            images=image_tensors,
            image_masks=mask_tensors,
            state=jnp.asarray(state),
            tokenized_prompt=jnp.asarray(tokenized_prompt[None, ...], dtype=jnp.int32),
            tokenized_prompt_mask=jnp.asarray(prompt_mask[None, ...], dtype=jnp.bool_),
        )

    def _prepare_images(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not images:
            raise ValueError("观测中缺少图像数据")
        processed = {}
        fallback = next(iter(images.values()))
        for key in self._IMAGE_KEYS:
            source = images.get(key, fallback)
            processed[key] = self._format_image(source)
        return processed

    def _prepare_state(self, state: np.ndarray | Mapping[str, Any]) -> np.ndarray:
        vec = np.asarray(state, dtype=np.float32).reshape(-1)
        if vec.size < self._cfg.action_dim:
            padded = np.zeros(self._cfg.action_dim, dtype=np.float32)
            padded[: vec.size] = vec
            vec = padded
        elif vec.size > self._cfg.action_dim:
            vec = vec[: self._cfg.action_dim]
        if self._state_normalizer is not None:
            normalized = self._state_normalizer({"state": vec[None, ...]})["state"][0]
            vec = normalized.astype(np.float32)
        return vec

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        unnorm = self._action_denorm({"action": action[None, ...]})
        return np.asarray(unnorm["action"][0], dtype=np.float32)

    def _map_to_simulator(self, action: np.ndarray) -> np.ndarray:
        mapping: ActionMappingConfig = self._cfg.action_mapping
        result = np.zeros(mapping.sim_action_dim, dtype=np.float32)
        for src, dst in zip(mapping.policy_indices, mapping.sim_indices):
            if src >= action.shape[0] or dst >= mapping.sim_action_dim:
                continue
            result[dst] = action[src]
        return result

    def _extract_images(self, observation: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
        if observation is None or not isinstance(observation, Mapping):
            return {}
        stack: list[Mapping[str, Any]] = [observation]
        while stack:
            node = stack.pop()
            for key in self._IMAGE_CANDIDATE_KEYS:
                if key not in node:
                    continue
                value = node[key]
                if isinstance(value, Mapping):
                    return {
                        sub_key: np.asarray(sub_value)
                        for sub_key, sub_value in value.items()
                        if isinstance(sub_value, np.ndarray)
                    }
                if isinstance(value, np.ndarray):
                    return {key: np.asarray(value)}
            for value in node.values():
                if isinstance(value, Mapping):
                    stack.append(value)
        return {}

    def _extract_state(self, observation: Mapping[str, Any] | None) -> np.ndarray | None:
        if observation is None or not isinstance(observation, Mapping):
            return None
        stack: list[Mapping[str, Any]] = [observation]
        while stack:
            node = stack.pop()
            for key, value in node.items():
                if key in self._STATE_CANDIDATE_KEYS and isinstance(value, np.ndarray):
                    return np.asarray(value)
                if isinstance(value, Mapping):
                    stack.append(value)
        return None

    def _format_image(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        # convert to [0, 255] if necessary
        if arr.max() > 1.5 or arr.min() < -0.5:
            arr = np.clip(arr, 0.0, 255.0)
            arr = arr / 255.0 * 2.0 - 1.0
        arr = np.clip(arr, -1.0, 1.0)
        return _resize_with_pad(arr, *self._IMAGE_SHAPE)


def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """简单的等比例缩放+填充，输入输出范围维持在 [-1, 1]。"""

    image_uint8 = np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image_uint8)
    pil_img.thumbnail((width, height), Image.BICUBIC)
    canvas = Image.new("RGB", (width, height))
    paste_x = (width - pil_img.width) // 2
    paste_y = (height - pil_img.height) // 2
    canvas.paste(pil_img, (paste_x, paste_y))
    arr = np.asarray(canvas).astype(np.float32) / 127.5 - 1.0
    return arr
