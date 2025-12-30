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
import sys
from pathlib import Path
from typing import Iterable, Mapping, Any
import imageio.v2 as imageio
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import simpler_env
from mani_skill2_real2sim.utils.sapien_utils import look_at

_VLA_IMPORT_ERROR: Exception | None = None

try:
    from .config import ACTION_SEQUENCE, OUTPUT_VIDEO_PATH, TASK_NAME, USE_VLA_POLICY, VLA_POLICY
    from .parquet_loader import load_and_remap_actions
    from .vla_runner import VLACheckpointRunner
except ImportError:
    from Interface_VLA_simulation_env.config import (
        ACTION_SEQUENCE,
        OUTPUT_VIDEO_PATH,
        TASK_NAME,
        USE_VLA_POLICY,
        VLA_POLICY,
    )
    from Interface_VLA_simulation_env.parquet_loader import load_and_remap_actions

    try:
        from Interface_VLA_simulation_env.vla_runner import VLACheckpointRunner
    except ImportError as err:  # pragma: no cover - runner 可选
        _VLA_IMPORT_ERROR = err
        VLACheckpointRunner = None  # type: ignore
CAMERA_EYE = [0.25, -0.05, 3.40]
CAMERA_TARGET = [0.40, 0.05, 0.15]
CAMERA_UP = [0.0, 1.0, 0.25]
CAMERA_FOV = 0.70
ACTION_REPEAT = 6
VIDEO_FPS = 15
CAPTURE_VIDEO = True
DESIRED_WRIST_POSE = {
    "forearm_roll": 0.0,
    "wrist_angle": -math.pi / 2,
    "wrist_rotate": 0.0,
}
MAX_POLICY_STEPS = 300


def _build_render_config():
    pose = look_at(CAMERA_EYE, CAMERA_TARGET, up=CAMERA_UP)
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


def _reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def _robot_state(env) -> np.ndarray | None:
    try:
        robot = env.unwrapped.agent.robot
        return np.asarray(robot.get_qpos(), dtype=np.float32)
    except Exception:
        return None


def _clone_observation(obs: Any) -> Mapping[str, Any]:
    if isinstance(obs, dict):
        cloned = dict(obs)
        if isinstance(obs.get("image"), dict):
            cloned["image"] = dict(obs["image"])
        return cloned
    return {}


def _policy_observation(obs: Any, env) -> Mapping[str, Any]:
    cloned = _clone_observation(obs)
    frame = _render_rgb(env)
    image_block = cloned.get("image")
    if not isinstance(image_block, dict):
        image_block = {}
    image_block["base_0_rgb"] = frame
    cloned["image"] = image_block
    return cloned


def _replay_actions(env, actions: np.ndarray, action_repeat: int, frames: list[np.ndarray]):
    env.reset()
    _align_wrist_pose(env, DESIRED_WRIST_POSE)
    finished = False
    for step_idx, action in enumerate(actions):
        if finished:
            break
        for _ in range(action_repeat):
            _, _, terminated, truncated, _ = env.step(action)
            if CAPTURE_VIDEO:
                frames.append(_render_rgb(env))
            if terminated or truncated:
                print(
                    f"[Info] Episode ended early at frame {step_idx + 1} (terminated={terminated}, truncated={truncated})"
                )
                finished = True
                break
        if step_idx % 10 == 0:
            print(f"已执行 {step_idx + 1}/{actions.shape[0]} 帧")


def _run_policy_episode(
    env,
    runner: VLACheckpointRunner,
    instruction: str,
    action_repeat: int,
    frames: list[np.ndarray],
    max_steps: int = MAX_POLICY_STEPS,
):
    obs = _reset_env(env)
    _align_wrist_pose(env, DESIRED_WRIST_POSE)
    finished = False
    step_count = 0
    while not finished and step_count < max_steps:
        robot_state = _robot_state(env)
        policy_obs = _policy_observation(obs, env)
        action = runner.predict(policy_obs, instruction, robot_state=robot_state)
        for _ in range(action_repeat):
            obs, _, terminated, truncated, _ = env.step(action)
            if CAPTURE_VIDEO:
                frames.append(_render_rgb(env))
            if terminated or truncated:
                finished = True
                break
        step_count += 1
        if step_count % 5 == 0:
            print(f"[Info] 闭环推理已执行 {step_count} 步")
    if not finished:
        print(f"[Warn] 达到最大步数 {max_steps}，提前结束。")


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
        if USE_VLA_POLICY and VLA_POLICY is not None:
            if VLACheckpointRunner is None:
                msg = "[Warn] 未找到 VLACheckpointRunner，回退到离线动作重放。"
                if _VLA_IMPORT_ERROR is not None:
                    msg += f" 原因: {_VLA_IMPORT_ERROR}"
                print(msg)
            else:
                print("[Info] 使用 VLA checkpoint 执行闭环仿真")
                runner = VLACheckpointRunner(VLA_POLICY)
                _run_policy_episode(env, runner, VLA_POLICY.instruction, action_repeat, frames)
                return output_video_path
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