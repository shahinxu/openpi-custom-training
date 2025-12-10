# Critical Files for Training

## Essential Files (Cannot Delete)

### 1. Model & Core Training Code
```
src/openpi/
├── models/
│   ├── pi0.py                    # Pi0 model implementation
│   ├── pi0_config.py             # Model configuration
│   ├── gemma.py                  # Gemma language model
│   ├── gemma_fast.py             # Optimized Gemma
│   ├── siglip.py                 # Vision encoder
│   ├── vit.py                    # Vision Transformer
│   ├── lora.py                   # LoRA implementation
│   └── tokenizer.py              # Text tokenizer
├── training/
│   ├── config.py                 # **CRITICAL** Training configs (pi05_custom)
│   ├── data_loader.py            # Data loading pipeline
│   ├── checkpoints.py            # Checkpoint management
│   └── optimizer.py              # Optimizer configuration
├── policies/
│   └── libero_policy.py          # Libero input/output transforms
├── transforms.py                 # Data transformations
└── shared/
    └── normalize.py              # Normalization utilities
```

### 2. Training Scripts
```
scripts/
├── train.py                      # **CRITICAL** Main training script
├── compute_norm_stats.py         # Compute normalization statistics
└── custom_training/
    ├── pipeline.py               # Full training pipeline
    ├── convert_data.py           # H5 → LeRobot conversion
    ├── compute_stats.py          # Stats computation
    └── run_training.py           # Training launcher
```

### 3. Dataset Files
```
datasets/openpi_segmented_data/
├── data/chunk-000/
│   └── episode_*.parquet         # **CRITICAL** Episode data
├── videos/chunk-000/
│   └── observation.images.cam_0/
│       └── episode_*/
│           └── frame_*.png       # **CRITICAL** Image frames
├── meta/
│   └── info.json                 # Dataset metadata
├── split_info.json               # Train/test split
└── norm_stats.json               # **CRITICAL** Normalization stats
```

### 4. Configuration Files
```
pyproject.toml                    # **CRITICAL** Python dependencies
.python-version                   # Python version spec
uv.lock                          # Locked dependency versions
```

### 5. Checkpoint Directory
```
checkpoints/pi05_custom/
└── <exp_name>/
    ├── <step>/
    │   ├── params/              # **CRITICAL** Model weights
    │   ├── metrics/             # Training metrics
    │   └── assets/              # Normalization stats copy
    └── wandb_id.txt             # Wandb run ID
```

## Important (Recommended to Keep)

### Documentation
```
README.md                        # Project overview
TRAINING_GUIDE.md               # Training documentation
LICENSE                         # License file
```

### Assets
```
assets/pi05_custom/
└── rzh/openpi_segmented_data/
    └── norm_stats.json         # Cached normalization stats
```

## Can Be Regenerated

### Temporary Files
```
.pytest_cache/                  # Test cache
__pycache__/                    # Python bytecode
*.pyc                           # Compiled Python
wandb/                          # Wandb local logs (synced to cloud)
.venv/                          # Virtual environment (can reinstall)
```

### Git Files
```
.git/                           # Git history (on GitHub)
.gitignore                      # Git ignore rules
```

## Files You Can Safely Delete

### Old/Unused Files
```
state_segmented_data/           # Original H5 files (if converted)
assets/pi05_custom/test_run*/   # Old experiment assets
checkpoints/pi05_custom/old_*/  # Old experiments
```

## Dependency Chain

```
Training Command (train.py)
    ↓
Training Config (config.py::pi05_custom)
    ↓
Model Definition (pi0.py, pi0_config.py)
    ├→ Vision Encoder (siglip.py, vit.py)
    ├→ Language Model (gemma.py, gemma_fast.py)
    └→ LoRA Adapters (lora.py)
    ↓
Data Loader (data_loader.py)
    ↓
Dataset (LeRobot format in datasets/)
    ├→ Episode Data (.parquet files)
    ├→ Images (.png files)
    └→ Norm Stats (norm_stats.json)
    ↓
Data Transforms (transforms.py)
    ├→ Repack (key mapping)
    ├→ Libero Transforms (libero_policy.py)
    └→ Normalization (normalize.py)
    ↓
Model Training Loop
    ├→ Optimizer (optimizer.py)
    ├→ Checkpoint Saving (checkpoints.py)
    └→ Wandb Logging
```

## Critical Path for Training

**Minimal required structure:**

```
openpi/
├── src/openpi/              # All Python source code
├── scripts/train.py         # Training entry point
├── datasets/                # Your training data
├── pyproject.toml          # Dependencies
└── .venv/                  # Python environment
```

**If any of these are deleted, training will fail immediately.**

## Size Breakdown

Approximate sizes:
```
src/openpi/                 ~50 MB   (code)
datasets/                   ~300 MB  (34 episodes with images)
checkpoints/                ~12 GB   (per checkpoint)
.venv/                     ~8 GB    (Python packages)
wandb/                     ~100 MB  (logs)
```

## Backup Priority

**Priority 1 (Must backup):**
- `datasets/openpi_segmented_data/` - Your training data
- `checkpoints/pi05_custom/<exp_name>/` - Trained model weights
- `src/openpi/training/config.py` - Your custom config

**Priority 2 (Good to backup):**
- `scripts/custom_training/` - Your custom scripts
- `TRAINING_GUIDE.md` - Documentation

**Priority 3 (Can recreate):**
- `.venv/` - Reinstall with `uv sync`
- `wandb/` - Synced to cloud
- `assets/` - Regenerated from datasets

## How to Verify Integrity

Check all critical files exist:

```bash
# Model code
test -f src/openpi/models/pi0.py && echo "✓ Model code OK"

# Training script
test -f scripts/train.py && echo "✓ Training script OK"

# Config with pi05_custom
grep -q "pi05_custom" src/openpi/training/config.py && echo "✓ Config OK"

# Dataset
test -d datasets/openpi_segmented_data/data && echo "✓ Dataset OK"

# Norm stats
test -f datasets/openpi_segmented_data/norm_stats.json && echo "✓ Norm stats OK"

# Dependencies
test -f pyproject.toml && echo "✓ Dependencies OK"
```

## Recovery Scenarios

### Lost dataset files
```bash
# Reconvert from H5 files
python scripts/custom_training/convert_data.py \
  --data_dir /path/to/h5_backups \
  --output_dir datasets/openpi_segmented_data
```

### Lost norm_stats.json
```bash
# Recompute from dataset
python scripts/custom_training/compute_stats.py \
  --dataset_dir datasets/openpi_segmented_data
```

### Lost .venv
```bash
# Reinstall environment
uv sync
```

### Lost config changes
```bash
# Check git history
git log -p src/openpi/training/config.py
git checkout <commit> -- src/openpi/training/config.py
```

### Lost checkpoint
- Cannot recover (must retrain)
- Check if synced to wandb artifacts
- Restore from your backup system

## Conclusion

**The 3 most critical components:**
1. **Source code** (`src/openpi/`) - The training logic
2. **Dataset** (`datasets/`) - Your training data
3. **Config** (`config.py::pi05_custom`) - Your training setup

Everything else can be regenerated, but these three cannot be recreated without significant effort.
