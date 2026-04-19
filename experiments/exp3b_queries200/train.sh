#!/bin/bash
# Experiment 3b: 200 Queries
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate plantseg_san

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/PlantSeg
pip install -e . --no-deps 2>/dev/null || pip install -e . --no-deps

echo "============================================================"
echo " Experiment 3b: 200 Queries"
echo " GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

python tools/train.py \
    experiments/exp3b_queries200/config.py \
    --work-dir work_dirs/exp3b_queries200 \
    --resume
