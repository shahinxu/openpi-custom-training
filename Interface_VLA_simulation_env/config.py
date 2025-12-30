from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "train_test_dataset" / "data"
RESULT_DIR = PROJECT_ROOT / "Interface_VLA_simulation_env" / "result" / "simulation_output"


@dataclass(frozen=True)
class ActionSequenceConfig:
	parquet_path: Path
	action_column: str = "action"
	input_dof_dim: Sequence[int] = (2, 3, 4)
	sim_dof_dim: Sequence[int] = (4, 5, 6)
	sim_action_dim: int | None = None
	frame_start: int = 0
	frame_end: int | None = None


ACTION_SEQUENCE = ActionSequenceConfig(
	parquet_path=DATA_ROOT / "chunk-000" / "episode_000000.parquet",
	sim_action_dim=7,
)


@dataclass(frozen=True)
class ActionMappingConfig:
	policy_indices: Sequence[int]
	sim_indices: Sequence[int]
	sim_action_dim: int


@dataclass(frozen=True)
class VLACheckpointConfig:
	checkpoint_dir: Path
	norm_stats_dir: Path
	instruction: str
	paligemma_variant: str = "gemma_2b_lora"
	action_expert_variant: str = "gemma_300m_lora"
	action_dim: int = 5
	action_horizon: int = 10
	max_token_len: int = 48
	discrete_state_input: bool = False
	num_denoising_steps: int = 10
	rng_seed: int = 0
	use_quantile_stats: bool = True
	action_mapping: ActionMappingConfig = field(
		default_factory=lambda: ActionMappingConfig(
			policy_indices=(2, 3, 4),
			sim_indices=(4, 5, 6),
			sim_action_dim=7,
		)
	)

TASK_NAME = "google_robot_pick_coke_can"
OUTPUT_VIDEO_PATH = RESULT_DIR / "demo.mp4"

USE_VLA_POLICY = True
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "pi05_train_test" / "my_training_run_2" / "1200"
NORM_STATS_DIR = PROJECT_ROOT / "data" / "train_test_dataset"
DEFAULT_INSTRUCTION = "Pick up the coke can and hold it steady."

VLA_POLICY = VLACheckpointConfig(
	checkpoint_dir=CHECKPOINT_DIR,
	norm_stats_dir=NORM_STATS_DIR,
	instruction=DEFAULT_INSTRUCTION,
)