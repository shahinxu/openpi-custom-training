#!/bin/bash
# Training launcher script for pi0.5 LoRA fine-tuning

set -e  # Exit on error

echo "================================================================================"
echo "🚀 Starting Pi0.5 LoRA Fine-Tuning on Custom Dataset"
echo "================================================================================"

# Training arguments
EXPERIMENT_NAME="${1:-custom_lora_$(date +%Y%m%d_%H%M%S)}"

echo ""
echo "📝 Training Configuration:"
echo "  Config: pi05_custom"
echo "  Experiment name: ${EXPERIMENT_NAME}"
echo "  Dataset: rzh/openpi_segmented_data"
echo "  Model: Pi0.5 with LoRA"
echo "================================================================================"
echo ""

# Run training
uv run scripts/train.py pi05_custom \
    --exp_name "${EXPERIMENT_NAME}" \
    "${@:2}"

echo ""
echo "================================================================================"
echo "✅ Training completed or interrupted"
echo "================================================================================"
