#!/bin/bash
# Experiment 2: SAM (Sharpness-Aware Minimization) Optimizer
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate plantseg_san

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/PlantSeg
pip install -e . --no-deps 2>/dev/null || pip install -e . --no-deps

echo "============================================================"
echo " Experiment 2: SAM Optimizer"
echo " GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

python tools/train.py \
    experiments/exp2_sam/config.py \
    --work-dir work_dirs/exp2_sam \
    --resume
