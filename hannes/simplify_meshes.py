#!/usr/bin/env python3
"""简化STL网格到MuJoCo可以处理的大小（<200k faces）"""

import trimesh

# 需要简化的文件和目标面数
files_to_simplify = {
    'palm.stl': 150000,  # 从318k降到150k
    'ringfinger.stl': 100000,  # 从158k降到100k
}

# 其他文件保持不变（已经小于200k）
files_to_copy = [
    'forefinger.stl',
    'midfinger.stl', 
    'littlefinger.stl',
]

import os
import shutil

source_dir = '/playpen-shared/zhenx/openpi/hannes'
target_dir = '/home/rzh/conda-environment/openpi_mujoco/lib/python3.10/site-packages/robosuite/models/assets/robots/hannes'

print("=== 简化网格 ===")
for filename, target_faces in files_to_simplify.items():
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, filename)
    
    print(f"\n处理 {filename}...")
    mesh = trimesh.load(source_path)
    original_faces = len(mesh.faces)
    print(f"  原始面数: {original_faces:,}")
    
    # 计算简化比例
    target_ratio = target_faces / original_faces
    print(f"  目标比例: {target_ratio:.2%}")
    
    # 使用quadric decimation简化
    simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
    final_faces = len(simplified.faces)
    print(f"  简化后面数: {final_faces:,} ({final_faces/original_faces*100:.1f}%)")
    
    # 保存
    simplified.export(target_path)
    print(f"  已保存到: {target_path}")

print("\n=== 复制其他文件 ===")
for filename in files_to_copy:
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, filename)
    shutil.copy2(source_path, target_path)
    
    mesh = trimesh.load(source_path)
    print(f"{filename:20s}: {len(mesh.faces):,} faces (无需简化)")

print("\n完成！所有网格文件已准备好。")
