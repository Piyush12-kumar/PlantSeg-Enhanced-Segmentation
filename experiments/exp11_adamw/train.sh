#!/bin/bash
# AdamW baseline
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate plantseg_san

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/PlantSeg
pip install -e . --no-deps 2>/dev/null || pip install -e . --no-deps

python tools/train.py \
    experiments/exp11_adamw/config.py \
    --work-dir work_dirs/exp11_adamw \
    --resume
