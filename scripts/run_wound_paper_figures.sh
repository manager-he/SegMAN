#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="segmentation/work_dirs/paper_figures"
mkdir -p "$OUT_ROOT"

# Method format: "name|config_path|checkpoint_path"
METHODS=(
  # "FUSegNet|segmentation/local_configs/fuseg/fuseg_wound.py|segmentation/work_dirs/fuseg_wound/latest.pth"
  # "SegMAN-Base|segmentation/local_configs/segman/tiny/segman_t_wound_bce_dice_focal_complex_aug.py|segmentation/work_dirs/wound_tiny_ablation/segman_t_wound_bce_dice_focal_complex_aug/iter_32000.pth"
  "SegMAN-Boundary(ours)|segmentation/local_configs/segman/tiny/segman_t_wound_bce_dice_focal_complex_aug_boundary.py|segmentation/work_dirs/wound_tiny_ablation/segman_t_wound_bce_dice_focal_complex_aug_boundary/iter_120000.pth"
)

CMD=(
  python scripts/visualize_wound_paper_figures.py
  --primary-config segmentation/local_configs/segman/tiny/segman_t_wound_bce_dice_focal_complex_aug_boundary.py
  --focus-method "SegMAN-Boundary(ours)"
  --out-dir "$OUT_ROOT"
  --device "${DEVICE:-cuda:0}"
  --num-success 5
  --num-failure 5
  --num-boundary 3
  --skip-missing-checkpoint
)

for item in "${METHODS[@]}"; do
  CMD+=(--method "$item")
done

echo "[Run] Generating paper figures..."
"${CMD[@]}"
echo "[Done] Figures are saved to: $OUT_ROOT"
