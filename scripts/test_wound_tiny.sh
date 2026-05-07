#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="segmentation/work_dirs/wound_tiny_ablation"
mkdir -p "$OUT_ROOT"

CONFIGS=(
  # Format: "<config_path>|<ckpt_filename>"
  # "segmentation/local_configs/segman/tiny/segman_t_wound_bce_basic_aug.py|latest.pth"
  "segmentation/local_configs/segman/tiny/segman_t_wound_bce_complex_aug.py|iter_12000.pth"
  "segmentation/local_configs/segman/tiny/segman_t_wound_bce_dice_focal_complex_aug.py|iter_32000.pth"
  # "segmentation/local_configs/segman/tiny/segman_t_wound_bce_complex_aug_boundary.py|latest.pth"
  "segmentation/local_configs/segman/tiny/segman_t_wound_bce_dice_focal_complex_aug_boundary.py|iter_120000.pth"
)

for item in "${CONFIGS[@]}"; do

  IFS='|' read -r cfg ckpt_name <<< "$item"

  exp_name="$(basename "${cfg%.py}")"
  work_dir="$OUT_ROOT/$exp_name"
  eval_dir="$work_dir/eval"
  mkdir -p "$work_dir" "$eval_dir"

  ckpt="$work_dir/$ckpt_name"

  echo "[Eval] $exp_name"
  python segmentation/tools/test.py "$cfg" \
    --checkpoint "$ckpt" \
    --eval mIoU mDice mFscore \
    --work-dir "$eval_dir"

done

