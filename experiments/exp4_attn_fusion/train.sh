#!/bin/bash
# Experiment 4: Attention-Based CLIP Fusion
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate plantseg_san

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/PlantSeg
pip install -e . --no-deps 2>/dev/null || pip install -e . --no-deps

echo "============================================================"
echo " Experiment 4: Attention-Based CLIP Fusion"
echo " GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

python tools/train.py \
    experiments/exp4_attn_fusion/config.py \
    --work-dir work_dirs/exp4_attn_fusion \
    --resume
