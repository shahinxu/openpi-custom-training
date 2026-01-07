XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
# GIT_LFS_SKIP_SMUDGE=1 \
CUDA_VISIBLE_DEVICES=3 \
uv run scripts/train.py pi05_train_test \
--exp-name pi05_train_test_debug \
--data.local-dir /playpen-shared/zhenx/openpi/data/train_test_dataset