# Training Guide

## Dataset Overview

**Location**: `/home/rzh/zhenx/openpi/datasets/openpi_segmented_data/`

### Dataset Statistics
- **Total Episodes**: 34 (31 train, 3 test)
- **Total Frames**: 1962
- **Format**: LeRobot (Parquet + PNG images)
- **Split Strategy**: Single-episode tasks → test set, Multi-episode tasks → train set

### Data Structure
```
datasets/openpi_segmented_data/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  # Episode metadata
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.cam_0/
│       │   ├── episode_000000/
│       │   │   ├── frame_000000.png
│       │   │   └── ...
│       │   └── ...
├── meta/
│   └── info.json                   # Dataset metadata
├── split_info.json                 # Train/test split
└── norm_stats.json                 # Normalization statistics
```

### Episode Content
Each episode contains:
- **Images**: `observation.images.cam_0` (224×224×3 RGB)
- **State**: `observation.state` (5-dim vector)
- **Action**: `action` (5-dim vector)
- **Task**: `task` (text description)

### Normalization Statistics
Computed from training set:
```python
state: {
    mean: [0.0, 50.0, 64.4, 37.1, 57.3],
    std: [0.0, 0.0, 26.4, 16.5, 33.7]
}
action: {
    mean: [0.0, 0.0, 0.04, 0.08, 0.09],
    std: [0.0, 0.0, 0.77, 0.66, 1.54]
}
```

## Training Process

### Model Architecture

**Pi0.5 with LoRA Fine-tuning**

```
Input Image (224×224×3)
    ↓
Vision Encoder (SigLIP ViT-400M)
    ↓
PaliGemma 2B + LoRA (rank=16)
    ├─ Base weights: Frozen (bfloat16)
    └─ LoRA adapters: Trainable (float32)
    ↓
Action Expert 300M + LoRA (rank=32)
    ├─ Base weights: Frozen (bfloat16)
    └─ LoRA adapters: Trainable (float32)
    ↓
Action Projection (5→32→5 dims)
    └─ Trainable (float32)
    ↓
Output Action (5-dim effective, 32-dim padded)
```

### Trainable Parameters

Only LoRA adapters and projection layers are trained:
- `PaliGemma/llm/layers/*/lora_a` - PaliGemma LoRA low-rank matrices
- `PaliGemma/llm/layers/*/lora_b` - PaliGemma LoRA projection
- `PaliGemma/llm/layers/mlp_1/*_lora_*` - Action Expert LoRA
- `action_in_proj` - Action input projection (5→1024)
- `action_out_proj` - Action output projection (1024→32)
- `time_mlp_in`, `time_mlp_out` - Timestep embedding

**Total Parameters**: ~2.3B (base) + ~50M (LoRA trainable)

### Data Pipeline

1. **Load from LeRobot Dataset**
   ```python
   episode_data = load_episode(episode_id)
   # Keys: observation.images.cam_0, observation.state, action, task
   ```

2. **Repack Transform** - Map dataset keys to policy format
   ```python
   {
       "observation/image": "observation.images.cam_0",
       "observation/wrist_image": "observation.images.cam_0",
       "observation/state": "observation.state",
       "actions": "action",  # Note: singular → plural
       "prompt": "task"
   }
   ```

3. **Libero Data Transform** - Multi-view processing
   ```python
   # Generate 3 camera views (base, left_wrist, right_wrist)
   # Extract robot state
   ```

4. **Model Transform**
   ```python
   # Resize images to 224×224
   # Tokenize prompt text (max 200 tokens)
   # Normalize state and actions (z-score or quantile)
   # Pad actions from 5-dim to 32-dim
   ```

5. **Batching**
   ```python
   batch_size = 32
   # Create action sequences with horizon=10
   ```

### Training Configuration

**Config Name**: `pi05_custom`

```python
model:
  type: Pi0.5
  paligemma_variant: gemma_2b_lora
  action_expert_variant: gemma_300m_lora
  action_dim: 32  # Padded from 5
  action_horizon: 10
  discrete_state_input: False

data:
  repo_id: rzh/openpi_segmented_data
  batch_size: 32
  use_quantile_norm: True

training:
  num_train_steps: 10,000
  learning_rate:
    warmup_steps: 500
    peak_lr: 3e-4
    decay_steps: 10,000
    final_lr: 1e-5
    schedule: cosine
  optimizer:
    type: AdamW
    weight_decay: 0.01
    gradient_clip_norm: 1.0

checkpointing:
  save_interval: 1000  # Save every 1000 steps
  keep_period: 5000    # Keep checkpoint every 5000 steps
  max_to_keep: 1       # Keep only latest checkpoint
  
pretrained_weights:
  source: gs://openpi-assets/checkpoints/pi05_base/params
  strategy: Load base model, freeze non-LoRA weights
```

## Running Training

### Quick Start

```bash
# Start training with default config
CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py pi05_custom --exp_name my_experiment

# Or use the custom training script
python scripts/custom_training/run_training.py \
  --config pi05_custom \
  --exp_name my_experiment \
  --gpu 0
```

### Full Pipeline (Convert + Train)

If you have new H5 data:

```bash
python scripts/custom_training/pipeline.py \
  --data_dir /path/to/h5_files \
  --output_dir datasets/my_dataset \
  --exp_name my_experiment
```

### Resume Training

Training automatically resumes from the latest checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py pi05_custom --exp_name existing_experiment
```

## Monitoring

### Wandb Dashboard

Training metrics are logged to Wandb:
- **Project**: `openpi`
- **URL**: https://wandb.ai/shahinjohome/openpi

**Tracked Metrics**:
- `loss` - Training loss
- `grad_norm` - Gradient norm (for stability monitoring)
- `param_norm` - Parameter norm
- `learning_rate` - Current learning rate
- `step_time` - Time per training step

### Local Logs

Logs are saved to:
```
wandb/run-<timestamp>-<run_id>/
checkpoints/pi05_custom/<exp_name>/
  ├── 1/              # Checkpoint at step 1000
  │   ├── params/
  │   ├── metrics/
  │   └── assets/
  ├── 5000/           # Checkpoint at step 5000
  └── wandb_id.txt    # Wandb run ID for resumption
```

## Performance Expectations

### Training Speed
- **Steps per second**: ~0.073 (13.6s per step)
- **Total time (10k steps)**: ~37 hours on single GPU
- **GPU memory**: ~23 GB (LoRA fine-tuning)

### Convergence
- **Initial loss**: ~0.12-0.15
- **Expected final loss**: ~0.02-0.05 (dataset dependent)
- **Warmup**: 500 steps (loss may spike initially)
- **Convergence**: Usually by 5000-8000 steps

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size in config.py
batch_size: 16  # Default is 32
```

### Slow Training
```bash
# Check GPU utilization
nvidia-smi

# Use multiple GPUs (requires code modification)
CUDA_VISIBLE_DEVICES=0,1 uv run scripts/train.py pi05_custom --exp_name my_exp
```

### Data Loading Issues
```bash
# Verify dataset structure
ls datasets/openpi_segmented_data/

# Check normalization stats exist
cat datasets/openpi_segmented_data/norm_stats.json

# Regenerate if needed
python scripts/custom_training/compute_stats.py \
  --dataset_dir datasets/openpi_segmented_data
```

## Next Steps

### Evaluation

After training, evaluate on test set:
```bash
# TODO: Add evaluation script
python scripts/evaluate.py \
  --checkpoint checkpoints/pi05_custom/my_experiment/10000 \
  --dataset datasets/openpi_segmented_data
```

### Deployment

Deploy trained model for inference:
```bash
# TODO: Add inference script
python scripts/inference.py \
  --checkpoint checkpoints/pi05_custom/my_experiment/10000 \
  --task "your task description"
```

## Configuration Reference

Full config location: `src/openpi/training/config.py:1004`

Key parameters to tune:
- `batch_size` - Larger = faster but more memory
- `peak_lr` - Higher = faster convergence but less stable
- `num_train_steps` - More steps = better convergence
- `action_horizon` - Longer = smoother actions
- `warmup_steps` - Adjust based on dataset size

## Citation

Based on Physical Intelligence's OpenPI:
```
@misc{openpi2024,
  title={π₀.5: Improved Open-World Generalization with Knowledge Insulation},
  author={Physical Intelligence Team},
  year={2024},
  url={https://github.com/Physical-Intelligence/openpi}
}
```
