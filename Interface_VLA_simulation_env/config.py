from __future__ import annotations

from dataclasses import dataclass
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

TASK_NAME = "google_robot_pick_coke_can"
OUTPUT_VIDEO_PATH = RESULT_DIR / "demo.mp4"