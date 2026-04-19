#!/bin/bash
# Adafactor
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate plantseg_san

pip install transformers 2>/dev/null || true

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/PlantSeg
pip install -e . --no-deps 2>/dev/null || pip install -e . --no-deps

python tools/train.py \
    experiments/exp10_adafactor/config.py \
    --work-dir work_dirs/exp10_adafactor \
    --resume
