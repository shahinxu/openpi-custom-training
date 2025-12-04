# 训练说明

## 数据集信息

- **位置**: `datasets/openpi_segmented_data/`
- **数据集 ID**: `rzh/openpi_segmented_data`
- **总 Episodes**: 34 (训练集: 31, 测试集: 3)
- **总 Frames**: 1962
- **划分信息**: `datasets/openpi_segmented_data/split_info.json`

## 开始训练

### 方法 1: 使用启动脚本（推荐）

```bash
./start_training.sh my_experiment_name
```

### 方法 2: 直接使用 uv

```bash
# 首先导入配置
python train_custom_lora.py

# 然后运行训练
uv run scripts/train.py pi05_custom --exp_name my_experiment_name
```

## 训练配置

配置文件: `train_custom_lora.py`

关键参数:
- **模型**: Pi0.5 with LoRA
- **Action维度**: 5
- **Action horizon**: 10
- **Batch size**: 32
- **训练步数**: 10,000
- **学习率**: 3e-4 (peak)
- **优化器**: AdamW
- **LoRA**: 在 PaliGemma (2B) 和 Action Expert (300M) 上启用

## 监控训练

训练会自动在 Wandb 上记录：
- Project: `openpi`
- 查看 loss, gradient norm, 等指标

## Checkpoints

- **保存位置**: `checkpoints/pi05_custom/{experiment_name}/`
- **保存频率**: 每 1000 步
- **永久保存**: 每 5000 步的checkpoint不会被删除

## 继续训练

如果训练中断，可以使用 `--resume` 标志继续：

```bash
uv run scripts/train.py pi05_custom --exp_name my_experiment_name --resume
```

## 调整配置

修改 `train_custom_lora.py` 中的 `PI05_CUSTOM_CONFIG` 来调整：
- Batch size（根据GPU内存）
- 学习率
- 训练步数
- 等等

修改后，重新运行 `python train_custom_lora.py` 更新配置。
