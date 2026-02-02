import h5py
import numpy as np

dataset_path = 'hannes_demonstrations_robomimic/hannes_Lift_source_20260126_234824.hdf5'

with h5py.File(dataset_path, 'r+') as f:
    demo_keys = list(f['data'].keys())
    
    for demo_key in demo_keys:
        demo_grp = f['data'][demo_key]
        datagen_info = demo_grp['datagen_info']
        
        # 获取trajectory长度
        traj_len = len(demo_grp['actions'])
        
        # 为Lift任务创建子任务终止信号组
        # 删除旧的（如果存在）
        if 'subtask_term_signals' in datagen_info:
            del datagen_info['subtask_term_signals']
        
        # 创建subtask_term_signals组
        subtask_grp = datagen_info.create_group('subtask_term_signals')
        
        # Lift是单一子任务，创建全0信号（表示没有中间终止点）
        # 使用int32类型而不是float32，因为这是布尔信号
        subtask_1_signal = np.zeros(traj_len, dtype=np.int32)
        subtask_grp.create_dataset('subtask_1', data=subtask_1_signal)
        
        print(f'{demo_key}: 添加了 subtask_term_signals/subtask_1 (shape: {subtask_1_signal.shape})')

print('\n所有demonstrations都已添加subtask_term_signals')
