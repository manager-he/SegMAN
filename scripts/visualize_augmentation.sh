#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/visualize_augmentation_compare.py \
  --img-dir segmentation/data/wound/foot/images/validation \
  --out-dir segmentation/work_dirs/paper_figures/data_augmentation \
  --num-samples 3 \
  --num-images 6
