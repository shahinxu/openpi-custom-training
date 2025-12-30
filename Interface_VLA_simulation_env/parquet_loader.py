from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def load_parquet_table(path: Path | str) -> pd.DataFrame:
	parquet_path = Path(path)
	if not parquet_path.exists():
		raise FileNotFoundError(f"Parquet 文件不存在: {parquet_path}")
	df = pd.read_parquet(parquet_path)
	if "action" not in df.columns:
		cols = ", ".join(df.columns)
		raise KeyError(f"缺少 action 列，现有列: {cols}")
	return df


def extract_action_matrix(df: pd.DataFrame, column: str = "action") -> np.ndarray:
	"""Convert the per-row action list into a dense float32 matrix."""

	series = df[column].to_numpy()
	if len(series) == 0:
		return np.zeros((0, 0), dtype=np.float32)
	matrix = np.stack(series).astype(np.float32)
	return matrix


def select_action_dims(action_matrix: np.ndarray, indices: Sequence[int]) -> np.ndarray:
	"""Pick a subset of action dimensions (input_dof_dim)."""

	if action_matrix.ndim != 2:
		raise ValueError("action_matrix 必须是 2D")
	if not indices:
		return np.zeros((action_matrix.shape[0], 0), dtype=action_matrix.dtype)
	return action_matrix[:, indices]


def remap_actions(
	action_matrix: np.ndarray,
	input_indices: Sequence[int],
	sim_indices: Sequence[int],
	sim_action_dim: int | None = None,
) -> np.ndarray:
	"""Copy selected dims into simulator action space order."""

	if len(input_indices) != len(sim_indices):
		raise ValueError("input_indices 与 sim_indices 长度不一致")
	total_dim = sim_action_dim or action_matrix.shape[1]
	if total_dim <= max(sim_indices, default=-1):
		raise ValueError("sim_action_dim 过小，无法容纳对应索引")
	mapped = np.zeros((action_matrix.shape[0], total_dim), dtype=action_matrix.dtype)
	if input_indices:
		mapped[:, sim_indices] = action_matrix[:, input_indices]
	return mapped


def load_and_remap_actions(
	path: Path | str,
	input_indices: Sequence[int],
	sim_indices: Sequence[int],
	sim_action_dim: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	df = load_parquet_table(path)
	raw = extract_action_matrix(df)
	subset = select_action_dims(raw, input_indices)
	mapped = remap_actions(raw, input_indices, sim_indices, sim_action_dim)
	return raw, subset, mapped

