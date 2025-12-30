from __future__ import annotations

import os
def _configure_env():
    if "DISPLAY" in os.environ:
        del os.environ["DISPLAY"]
    os.environ["SVULKAN2_HEADLESS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["SAP_DEVICE_ID"] = "0"

_configure_env()

import math
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np

import simpler_env
from mani_skill2_real2sim.utils.sapien_utils import look_at

try:
    from .config import ACTION_SEQUENCE, OUTPUT_VIDEO_PATH, TASK_NAME
    from .parquet_loader import load_and_remap_actions
except ImportError:
    from config import ACTION_SEQUENCE, OUTPUT_VIDEO_PATH, TASK_NAME
    from parquet_loader import load_and_remap_actions
CAMERA_EYE = [0.05, -0.20, 3.40]
CAMERA_TARGET = [0.55, 0.15, 0.10]
CAMERA_FOV = 0.70
ACTION_REPEAT = 6
VIDEO_FPS = 15
CAPTURE_VIDEO = True
DESIRED_WRIST_POSE = {
    "forearm_roll": 0.0,
    "wrist_angle": -math.pi / 2,
    "wrist_rotate": 0.0,
}


def _build_render_config():
    pose = look_at(CAMERA_EYE, CAMERA_TARGET)
    return {
        "render_camera": {
            "p": pose.p.tolist(),
            "q": pose.q.tolist(),
            "fov": CAMERA_FOV,
        }
    }


def _make_env(task_name: str):
    return simpler_env.make(
        task_name,
        render_mode="rgb_array",
        render_camera_cfgs=_build_render_config(),
    )


def _align_wrist_pose(env, joint_targets):
    if not joint_targets:
        return
    controller = env.unwrapped.agent.controller
    arm_ctrl = controller.controllers.get("arm")
    if arm_ctrl is None:
        return
    if hasattr(arm_ctrl, "joint_names"):
        names = arm_ctrl.joint_names
    else:
        names = [joint.name for joint in getattr(arm_ctrl, "joints", [])]
    name_to_idx = {name: idx for name, idx in zip(names, arm_ctrl.joint_indices)}
    robot = env.unwrapped.agent.robot
    qpos = robot.get_qpos()
    changed = False
    for joint_name, target in joint_targets.items():
        joint_idx = name_to_idx.get(joint_name)
        if joint_idx is None:
            continue
        qpos[joint_idx] = target
        changed = True
    if changed:
        robot.set_qpos(qpos)


def _render_rgb(env):
    frame = env.render()
    if isinstance(frame, list):
        frame = frame[0]
    return frame


def _save_video(frames: Iterable[np.ndarray], output_path: Path, fps: int):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, list(frames), fps=fps)


def _load_mapped_actions(env, sequence_cfg):
    sim_dim = env.action_space.shape[0]
    requested_dim = sequence_cfg.sim_action_dim or sim_dim
    target_dim = max(sim_dim, requested_dim)
    _, _, mapped = load_and_remap_actions(
        sequence_cfg.parquet_path,
        sequence_cfg.input_dof_dim,
        sequence_cfg.sim_dof_dim,
        sim_action_dim=target_dim,
    )
    start = max(sequence_cfg.frame_start, 0)
    end = sequence_cfg.frame_end
    mapped = mapped[start:end]
    if mapped.shape[0] == 0:
        raise ValueError("未从动作序列中截取到有效帧")
    if target_dim > sim_dim:
        mapped = mapped[:, :sim_dim]
    elif target_dim < sim_dim:
        padded = np.zeros((mapped.shape[0], sim_dim), dtype=mapped.dtype)
        padded[:, :target_dim] = mapped
        mapped = padded
    return mapped


def _replay_actions(env, actions: np.ndarray, action_repeat: int, frames: list[np.ndarray]):
    obs, _ = env.reset()
    _align_wrist_pose(env, DESIRED_WRIST_POSE)
    for step_idx, action in enumerate(actions):
        for _ in range(action_repeat):
            obs, reward, terminated, truncated, info = env.step(action)
            if CAPTURE_VIDEO:
                frames.append(_render_rgb(env))
            if terminated or truncated:
                obs, _ = env.reset()
                _align_wrist_pose(env, DESIRED_WRIST_POSE)
        if step_idx % 10 == 0:
            print(f"已执行 {step_idx + 1}/{actions.shape[0]} 帧")


def run_action_sequence(
    task_name: str = TASK_NAME,
    action_cfg=ACTION_SEQUENCE,
    output_video_path: Path = OUTPUT_VIDEO_PATH,
    action_repeat: int = ACTION_REPEAT,
    video_fps: int = VIDEO_FPS,
):
    print(f"[Info] 创建环境: {task_name}")
    env = _make_env(task_name)
    frames: list[np.ndarray] = []
    try:
        actions = _load_mapped_actions(env, action_cfg)
        print(f"[Info] 动作帧数: {actions.shape[0]}, 动作维度: {actions.shape[1]}")
        _replay_actions(env, actions, action_repeat, frames)
    finally:
        env.close()
    if CAPTURE_VIDEO and frames:
        _save_video(frames, output_video_path, video_fps)
        print(f"[Info] 视频已保存到 {output_video_path}")
    elif CAPTURE_VIDEO:
        print("[Warn] 未捕获到帧，视频未生成。")
    return output_video_path


if __name__ == "__main__":
    run_action_sequence()