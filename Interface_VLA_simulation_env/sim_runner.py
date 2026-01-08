from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd

import robosuite as suite

_VLA_IMPORT_ERROR: Exception | None = None

try:
    from .config import ACTION_SEQUENCE, OUTPUT_VIDEO_PATH, TASK_NAME, USE_VLA_POLICY, VLA_POLICY
except ImportError:
    from Interface_VLA_simulation_env.config import (
        ACTION_SEQUENCE,
        OUTPUT_VIDEO_PATH,
        TASK_NAME,
        USE_VLA_POLICY,
        VLA_POLICY,
    )

try:
    from .vla_runner import VLACheckpointRunner
except ImportError:
    try:
        from Interface_VLA_simulation_env.vla_runner import VLACheckpointRunner
    except ImportError as err:
        _VLA_IMPORT_ERROR = err
        VLACheckpointRunner = None  # type: ignore

if TYPE_CHECKING:
    from .vla_runner import VLACheckpointRunner as VLACheckpointRunnerType
else:
    VLACheckpointRunnerType = Any

ROBOSUITE_TASK_MAP = {
    "google_robot_pick_coke_can": {
        "env_name": "Lift",
        "robots": "Hannes",
    },
}

DEFAULT_CAMERA_NAME = "birdview"
DEFAULT_CAMERA_WIDTH = 256
DEFAULT_CAMERA_HEIGHT = 256
DEFAULT_CONTROL_FREQ = 20


def _make_robosuite_env(task_name: str, use_camera: bool = True):
    config = ROBOSUITE_TASK_MAP.get(task_name)
    if config is None:
        raise ValueError(f"未找到任务 {task_name} 的配置，请在 ROBOSUITE_TASK_MAP 中添加")
    
    env = suite.make(
        env_name=config["env_name"],
        robots=config["robots"],
        has_renderer=False,
        has_offscreen_renderer=use_camera,
        use_camera_obs=False,
        camera_names=DEFAULT_CAMERA_NAME if use_camera else None,
        camera_heights=DEFAULT_CAMERA_HEIGHT if use_camera else None,
        camera_widths=DEFAULT_CAMERA_WIDTH if use_camera else None,
        control_freq=DEFAULT_CONTROL_FREQ,
    )
    
    if config["robots"] == "Hannes":
        base_body_id = env.sim.model.body_name2id('robot0_base')
        env.sim.model.body_pos[base_body_id][0] = 0.0
        env.sim.model.body_quat[base_body_id] = [0, 0, 0, 1]
        cam_id = env.sim.model.camera_name2id(DEFAULT_CAMERA_NAME)
        env.sim.model.cam_pos[cam_id] = [-1.0, 0.0, 1.35]
        env.sim.model.cam_quat[cam_id] = [0.43, 0.56, -0.56, 0.43]
    
    return env


def _render_frame(env) -> np.ndarray:
    cam_id = env.sim.model.camera_name2id("frontview")
    env.sim.model.cam_pos[cam_id][:] = [0.0, 1.0, 1.5]
    env.sim.model.cam_quat[cam_id][:] = [0.924, -0.383, 0, 0]
    env.sim.forward()
    
    frame = env.sim.render(
        camera_name="frontview",
        width=DEFAULT_CAMERA_WIDTH,
        height=DEFAULT_CAMERA_HEIGHT,
        depth=False,
    )
    frame = np.asarray(frame)
    if frame.ndim == 3:
        return np.flipud(frame)
    
    raise RuntimeError("无法从环境渲染图像")


def _save_video(frames: list[np.ndarray], output_path: Path, fps: int):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        imageio.mimsave(output_path, frames, fps=fps)
    else:
        print("[Warn] 未捕获到任何帧，不会生成视频。")

def _load_mapped_actions(env, sequence_cfg) -> np.ndarray:
    parquet_path = Path(sequence_cfg.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet 文件不存在: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    if "action" not in df.columns:
        cols = ", ".join(df.columns)
        raise KeyError(f"缺少 action 列，现有列: {cols}")
    action_series = df["action"].to_numpy()
    if len(action_series) == 0:
        raise ValueError("动作数据为空")
    raw_actions = np.stack(action_series).astype(np.float32)
    low, high = env.action_spec
    sim_dim = len(low)
    
    requested_dim = sequence_cfg.sim_action_dim or sim_dim
    target_dim = max(sim_dim, requested_dim)
    
    input_indices = sequence_cfg.input_dof_dim
    sim_indices = sequence_cfg.sim_dof_dim
    
    if len(input_indices) != len(sim_indices):
        raise ValueError("input_dof_dim 与 sim_dof_dim 长度不一致")
    
    if target_dim <= max(sim_indices, default=-1):
        raise ValueError("sim_action_dim 过小，无法容纳对应索引")
    
    mapped = np.zeros((raw_actions.shape[0], target_dim), dtype=np.float32)
    if input_indices:
        mapped[:, sim_indices] = raw_actions[:, input_indices]
    
    start = max(sequence_cfg.frame_start, 0)
    end = sequence_cfg.frame_end
    mapped = mapped[start:end]
    
    if mapped.shape[0] == 0:
        raise ValueError("未从动作序列中截取到有效帧")
    if target_dim > sim_dim:
        mapped = mapped[:, :sim_dim]
    elif target_dim < sim_dim:
        padded = np.zeros((mapped.shape[0], sim_dim), dtype=np.float32)
        padded[:, :target_dim] = mapped
        mapped = padded
    
    return mapped


def _replay_actions(env, actions: np.ndarray, action_repeat: int, frames: list[np.ndarray]):
    env.reset()
    frames.append(_render_frame(env))
    
    low, high = env.action_spec
    finished = False
    
    for step_idx, action in enumerate(actions):
        if finished:
            break
        
        clipped_action = np.clip(action, low, high)
        
        for _ in range(action_repeat):
            obs, reward, done, info = env.step(clipped_action)
            frames.append(_render_frame(env))
            if done:
                print(f"[Info] Episode ended early at frame {step_idx + 1}")
                finished = True
                break
        
        if step_idx % 10 == 0:
            print(f"已执行 {step_idx + 1}/{actions.shape[0]} 帧")


def _run_policy_episode(
    env,
    runner: VLACheckpointRunnerType,
    instruction: str,
    action_repeat: int,
    frames: list[np.ndarray],
    max_steps: int = 20,
):
    """使用 VLA 策略在 robosuite 环境中运行闭环控制"""
    env.reset()
    frames.append(_render_frame(env))
    
    low, high = env.action_spec
    finished = False
    step_count = 0
    
    while not finished and step_count < max_steps:
        frame = _render_frame(env)
        policy_obs = {
            "state": np.zeros(0, dtype=np.float32),
            "image": {"base_0_rgb": frame},
        }
        
        vla_action = runner.predict(policy_obs, instruction)
        if not np.all(np.isfinite(vla_action)):
            raise ValueError(f"策略输出了非法动作: {vla_action}")
        
        action = np.array([
            vla_action[2],
            vla_action[3],
            vla_action[4],
            vla_action[4],
            vla_action[4],
            vla_action[4],
        ])
        
        clipped_action = np.clip(action, low, high)
        
        if step_count < 3:
            print(f"[Debug] 第 {step_count + 1} 步动作: {np.array2string(clipped_action, precision=4)}")
        
        for _ in range(action_repeat):
            obs, reward, done, info = env.step(clipped_action)
            frames.append(_render_frame(env))
            if done:
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
    action_repeat: int = 6,
    video_fps: int = 15,
    gpu_id: int | None = None,
    max_steps: int = 20,
):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[Info] 使用 GPU {gpu_id} 进行 VLA 推理")
    
    print(f"[Info] 创建 robosuite 环境: {task_name}")
    env = _make_robosuite_env(task_name, use_camera=True)
    frames: list[np.ndarray] = []
    
    try:
        used_vla_policy = False
        if USE_VLA_POLICY and VLA_POLICY is not None:
            if VLACheckpointRunner is None:
                msg = "[Warn] 未找到 VLACheckpointRunner，回退到离线动作重放。"
                if _VLA_IMPORT_ERROR is not None:
                    msg += f" 原因: {_VLA_IMPORT_ERROR}"
                print(msg)
            else:
                print("[Info] 使用 VLA checkpoint 执行闭环仿真")
                runner = VLACheckpointRunner(VLA_POLICY)
                _run_policy_episode(env, runner, VLA_POLICY.instruction, action_repeat, frames, max_steps=max_steps)
                used_vla_policy = True
        
        if not used_vla_policy:
            actions = _load_mapped_actions(env, action_cfg)
            print(f"[Info] 动作帧数: {actions.shape[0]}, 动作维度: {actions.shape[1]}")
            _replay_actions(env, actions, action_repeat, frames)
    finally:
        env.close()
    
    if frames:
        _save_video(frames, output_video_path, video_fps)
        print(f"[Info] 视频已保存到 {output_video_path}")
    else:
        print("[Warn] 未捕获到帧，视频未生成。")
    
    return output_video_path


if __name__ == "__main__":
    run_action_sequence()
