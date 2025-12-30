"""Command-line entry for replaying dataset actions inside SIMPLER."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

try:
    from .config import ACTION_SEQUENCE, OUTPUT_VIDEO_PATH, TASK_NAME, ActionSequenceConfig
    from .sim_runner import run_action_sequence
except ImportError:  # direct execution fallback
    from config import ACTION_SEQUENCE, OUTPUT_VIDEO_PATH, TASK_NAME, ActionSequenceConfig
    from sim_runner import run_action_sequence


def _parse_int_list(value: str | None, default: Sequence[int]):
    if value is None:
        return default
    cleaned = [item.strip() for item in value.split(",") if item.strip()]
    if not cleaned:
        return default
    return tuple(int(item) for item in cleaned)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay action sequence and export MP4")
    parser.add_argument("--parquet", type=str, default=None, help="Path to the parquet episode")
    parser.add_argument("--task", type=str, default=None, help="SIMPLER task name, defaults to config")
    parser.add_argument("--video", type=str, default=None, help="Output video path")
    parser.add_argument("--frame-start", type=int, default=None, help="Start frame index")
    parser.add_argument("--frame-end", type=int, default=None, help="End frame index (exclusive)")
    parser.add_argument("--input-dofs", type=str, default=None, help="Comma separated dataset DOF indices")
    parser.add_argument("--sim-dofs", type=str, default=None, help="Comma separated simulator DOF indices")
    parser.add_argument("--sim-action-dim", type=int, default=None, help="Full simulator action dimension")
    parser.add_argument("--action-repeat", type=int, default=None, help="Number of sim steps per dataset frame")
    parser.add_argument("--video-fps", type=int, default=None, help="FPS for exported video")
    return parser


def main(argv: Sequence[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    action_cfg: ActionSequenceConfig = ACTION_SEQUENCE
    if args.parquet is not None:
        action_cfg = replace(action_cfg, parquet_path=Path(args.parquet).expanduser().resolve())
    if args.frame_start is not None:
        action_cfg = replace(action_cfg, frame_start=args.frame_start)
    if args.frame_end is not None:
        action_cfg = replace(action_cfg, frame_end=args.frame_end)
    if args.input_dofs is not None:
        action_cfg = replace(action_cfg, input_dof_dim=_parse_int_list(args.input_dofs, action_cfg.input_dof_dim))
    if args.sim_dofs is not None:
        action_cfg = replace(action_cfg, sim_dof_dim=_parse_int_list(args.sim_dofs, action_cfg.sim_dof_dim))
    if args.sim_action_dim is not None:
        action_cfg = replace(action_cfg, sim_action_dim=args.sim_action_dim)

    task_name = args.task or TASK_NAME
    video_path = Path(args.video).expanduser().resolve() if args.video else OUTPUT_VIDEO_PATH
    extra_kwargs = {}
    if args.action_repeat is not None:
        extra_kwargs["action_repeat"] = args.action_repeat
    if args.video_fps is not None:
        extra_kwargs["video_fps"] = args.video_fps

    run_action_sequence(
        task_name=task_name,
        action_cfg=action_cfg,
        output_video_path=video_path,
        **extra_kwargs,
    )


if __name__ == "__main__":
    main()
