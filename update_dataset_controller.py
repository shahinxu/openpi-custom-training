import h5py
import json

dataset_path = 'hannes_demonstrations_robomimic/hannes_Lift_source_20260126_234824.hdf5'

# OSC controller配置
controller_config = {
    "type": "BASIC",
    "body_parts": {
        "right": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "input_type": "delta",
            "input_ref_frame": "base",
            "interpolation": None,
            "ramp_ratio": 0.2,
            "gripper": {
                "type": "GRIP"
            }
        }
    }
}

with h5py.File(dataset_path, 'r+') as f:
    # 读取当前env_args
    env_args = json.loads(f['data'].attrs['env_args'])
    print("当前 env_args:")
    print(json.dumps(env_args, indent=2))
    
    # 添加controller_configs到env_kwargs中
    env_args['env_kwargs']['controller_configs'] = controller_config
    
    # 更新数据集
    f['data'].attrs['env_args'] = json.dumps(env_args)
    
    print("\n更新后的 env_args:")
    print(json.dumps(env_args, indent=2))

print('\n✅ 成功更新数据集中的controller配置')
