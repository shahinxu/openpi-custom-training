# Complete Training Pipeline Guide

This guide provides a complete, reproducible pipeline for training Pi0.5 with LoRA on custom H5 datasets.

## Prerequisites

- H5 dataset with structure:
  ```
  data_dir/
    task1/
      episode1.h5
      episode2.h5
    task2/
      episode1.h5
  ```
- Each H5 file contains:
  - `observations/images`: (T, H, W, C) uint8 images
  - `observations/states`: (T, state_dim) float32 states
  - `actions`: (T, action_dim) float32 actions

## Quick Start

### Option 1: One-Command Pipeline

```bash
python pipeline_convert_and_train.py \
    --data_dir /path/to/your/h5/data \
    --output_dir datasets/my_dataset \
    --dataset_name my_dataset \
    --exp_name my_first_run \
    --gpu_id 0
```

This single command will:
1. Convert H5 files to LeRobot format
2. Automatically split into train/test sets
3. Compute normalization statistics
4. Start training

### Option 2: Step-by-Step

#### Step 1: Data Conversion

```bash
python pipeline_convert_and_train.py \
    --data_dir /path/to/your/h5/data \
    --output_dir datasets/my_dataset \
    --dataset_name my_dataset \
    --exp_name placeholder \
    --train_only false \
    --skip_norm_stats
```

#### Step 2: Compute Normalization Statistics

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_custom
```

#### Step 3: Start Training

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py pi05_custom --exp_name my_experiment
```

## Configuration Details

The pipeline uses the `pi05_custom` configuration defined in `src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi05_custom",
    model=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",  # LoRA for PaliGemma
        action_expert_variant="gemma_300m_lora",  # LoRA for Action Expert
        action_dim=32,  # Actions padded to match pretrained model
        action_horizon=10,
        discrete_state_input=False,
    ),
    data=LeRobotCustomDataConfig(
        repo_id="your_dataset_name",
    ),
    batch_size=32,
    num_train_steps=10_000,
    lr_schedule=CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=3e-4,
        decay_steps=10_000,
        decay_lr=1e-5,
    ),
    weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
)
```

### Key Features

- **LoRA Fine-tuning**: Only trains LoRA adapters (~1% of parameters)
- **Action Padding**: Automatically pads actions to 32-dim to match pretrained model
- **Pre-trained Weights**: Loads pi05_base checkpoint for transfer learning
- **Train/Test Split**: Automatically uses single-episode tasks as test set

## Dataset Format

### Input: H5 Files

Each H5 file represents one episode:

```python
episode.h5
├── observations/
│   ├── images: (T, 224, 224, 3) uint8
│   └── states: (T, state_dim) float32
└── actions: (T, action_dim) float32
```

### Output: LeRobot Format

```
datasets/my_dataset/
├── data/
│   ├── chunk-000/
│   │   └── episode_000000.parquet
│   ├── chunk-001/
│   └── ...
├── videos/
│   ├── chunk-000/
│   │   └── episode_000000/
│   │       └── observation.images.cam_0.mp4
│   └── ...
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   └── tasks.jsonl
└── split_info.json  # Train/test split mapping
```

## Monitoring Training

### Wandb Dashboard

Training metrics are automatically logged to Wandb:
- https://wandb.ai/your_username/openpi

Metrics include:
- `train/loss`: Total training loss
- `train/action_loss`: Action prediction loss
- `train/learning_rate`: Current learning rate
- `train/grad_norm`: Gradient norm (should stay < 1.0)

### Local Logs

```bash
# View real-time logs
tail -f training_my_experiment.log

# Check training progress
grep "Progress" training_my_experiment.log
```

### Checkpoints

Checkpoints are saved every 1000 steps to:
```
checkpoints/pi05_custom/my_experiment/
├── 1000/
├── 2000/
├── ...
└── 10000/
```

Each checkpoint contains:
- `params/`: Model parameters
- `train_state/`: Optimizer state
- `assets/`: Normalization statistics
- `metrics.json`: Training metrics

## Customization

### Modify Training Hyperparameters

Edit `src/openpi/training/config.py` and update the `pi05_custom` TrainConfig:

```python
TrainConfig(
    name="pi05_custom",
    batch_size=16,  # Reduce for smaller GPU
    num_train_steps=20_000,  # Train longer
    lr_schedule=CosineDecaySchedule(
        peak_lr=1e-4,  # Lower learning rate
    ),
    save_interval=500,  # Save more frequently
)
```

### Change Action Dimension

If your dataset has different action dimensions, update `action_dim`:

```python
model=pi0_config.Pi0Config(
    action_dim=7,  # Set to your action dimension
)
```

Note: Actions will be automatically padded to 32-dim during training.

### Use Different GPU

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py pi05_custom --exp_name exp1

# Different GPU
CUDA_VISIBLE_DEVICES=3 uv run scripts/train.py pi05_custom --exp_name exp2
```

### Resume Training

```bash
uv run scripts/train.py pi05_custom --exp_name my_experiment --resume
```

## Troubleshooting

### Out of Memory

Reduce batch size in config:
```python
batch_size=16  # or 8
```

### Slow Training

- Use single GPU (avoid multi-GPU overhead for small datasets)
- Reduce `num_workers` in config
- Check GPU utilization: `nvidia-smi`

### Data Loading Errors

Verify dataset structure:
```bash
ls datasets/my_dataset/
# Should contain: data/, videos/, meta/

# Check first episode
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; \
           ds = LeRobotDataset('my_dataset', root='datasets'); \
           print(ds[0])"
```

### Norm Stats Not Found

Recompute normalization statistics:
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_custom
```

## Expected Training Time

On a single NVIDIA A100/H100:
- **10,000 steps**: ~12 hours
- **Per step**: ~4 seconds
- **Per epoch** (34 episodes, batch_size=32): ~100 steps

## Evaluation

After training, evaluate on test set:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load dataset with split info
dataset = LeRobotDataset("my_dataset", root="datasets")

# Get test episodes
import json
with open("datasets/my_dataset/split_info.json") as f:
    split_info = json.load(f)

test_episodes = [int(k) for k, v in split_info.items() if v == "test"]
print(f"Test episodes: {test_episodes}")
```

## Next Steps

1. **Monitor training**: Check Wandb for loss curves
2. **Adjust hyperparameters**: Based on validation performance
3. **Deploy model**: Use trained checkpoint for inference
4. **Collect more data**: If performance is insufficient

## References

- Main training script: `scripts/train.py`
- Data loader: `src/openpi/training/data_loader.py`
- Model config: `src/openpi/models/pi0_config.py`
- Training config: `src/openpi/training/config.py`
