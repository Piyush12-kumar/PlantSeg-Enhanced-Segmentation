#!/bin/bash
# Experiment 1: Lion Optimizer
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate plantseg_san

# Install lion-pytorch if not available
pip install lion-pytorch 2>/dev/null || true

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/PlantSeg
pip install -e . --no-deps 2>/dev/null || pip install -e . --no-deps

echo "============================================================"
echo " Experiment 1: Lion Optimizer"
echo " GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

python tools/train.py \
    experiments/exp1_lion/config.py \
    --work-dir work_dirs/exp1_lion \
    --resume
