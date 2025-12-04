# OpenPI Custom Training Repository

This repository contains a customized setup for training Pi0.5 models with LoRA on custom H5 datasets.

## Quick Start

### 1. Complete Pipeline (Recommended)

Train from H5 data to final model in one command:

```bash
python pipeline_convert_and_train.py \
    --data_dir /path/to/h5/data \
    --output_dir datasets/my_dataset \
    --dataset_name my_dataset \
    --exp_name my_first_run \
    --gpu_id 0
```

### 2. Step-by-Step

#### Data Conversion

```bash
# Convert H5 files to LeRobot format
python pipeline_convert_and_train.py \
    --data_dir /path/to/h5/data \
    --output_dir datasets/my_dataset \
    --dataset_name my_dataset \
    --exp_name placeholder \
    --skip_norm_stats
```

#### Compute Normalization Stats

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_custom
```

#### Start Training

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py pi05_custom --exp_name my_experiment
```

## Repository Structure

```
.
├── pipeline_convert_and_train.py  # Complete training pipeline
├── PIPELINE_GUIDE.md              # Detailed documentation
├── src/openpi/
│   ├── training/
│   │   └── config.py              # Contains pi05_custom config
│   └── ...
├── scripts/
│   ├── train.py                   # Main training script
│   ├── compute_norm_stats.py      # Compute dataset statistics
│   └── ...
├── datasets/                      # Your converted datasets
│   └── openpi_segmented_data/     # Example dataset
└── checkpoints/                   # Training checkpoints
    └── pi05_custom/
        └── experiment_name/
```

## Key Features

- **Pi0.5 Model**: State-of-the-art vision-language-action model
- **LoRA Fine-tuning**: Memory-efficient training (~1% of parameters)
- **Automatic Action Padding**: Works with any action dimension
- **Pre-trained Weights**: Leverages pi05_base checkpoint
- **Train/Test Split**: Automatic splitting based on task structure
- **Wandb Integration**: Real-time training monitoring

## Configuration

The main training configuration is in `src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi05_custom",
    model=Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=32,  # Actions padded to 32-dim
        action_horizon=10,
    ),
    batch_size=32,
    num_train_steps=10_000,
    lr_schedule=CosineDecaySchedule(peak_lr=3e-4),
)
```

## Requirements

### H5 Dataset Format

Each H5 file should contain:

```python
observations/
  images: (T, H, W, C) uint8  # Images
  states: (T, state_dim) float32  # State observations
actions: (T, action_dim) float32  # Action commands
```

### System Requirements

- NVIDIA GPU (tested on A100/H100)
- Python 3.11+
- CUDA 11.8+
- ~25GB GPU memory for batch_size=32

## Training Details

- **Duration**: ~12 hours for 10,000 steps (single A100)
- **Checkpoints**: Saved every 1,000 steps
- **Logging**: Every 50 steps to Wandb
- **Metrics**: Loss, learning rate, gradient norms

## Monitoring

### Wandb Dashboard

View training metrics at: https://wandb.ai/your_username/openpi

### Local Logs

```bash
tail -f training_my_experiment.log
```

## Troubleshooting

### Out of Memory

Reduce batch size in `src/openpi/training/config.py`:

```python
batch_size=16  # or 8
```

### Slow Training

- Use single GPU with `CUDA_VISIBLE_DEVICES=0`
- Check GPU utilization with `nvidia-smi`
- Verify data loading isn't the bottleneck

### Data Errors

Verify dataset structure:

```bash
ls datasets/my_dataset/
# Should show: data/, videos/, meta/, split_info.json
```

## Documentation

- **Complete Guide**: See `PIPELINE_GUIDE.md` for detailed documentation
- **Original OpenPI**: Based on Physical Intelligence's OpenPI framework
- **Model Details**: Pi0.5 architecture with PaliGemma vision encoder

## Citation

This repository is based on OpenPI by Physical Intelligence:

```bibtex
@misc{openpi2024,
  title={OpenPI: Open Physical Intelligence},
  author={Physical Intelligence},
  year={2024},
  url={https://github.com/physical-intelligence/openpi}
}
```

## License

See `LICENSE` and `LICENSE_GEMMA.txt` for details.

## Support

For issues and questions:
1. Check `PIPELINE_GUIDE.md` for detailed documentation
2. Review training logs and Wandb metrics
3. Verify dataset format and structure
