#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CFG="segmentation/local_configs/segman/tiny/segman_t_wound_bce_basic_aug.py"
WORK_DIR="segmentation/work_dirs/smoke_t_wound_bce_basic"

echo "[Smoke] Training dry run start"
python segmentation/tools/train.py "$CFG" \
  --work-dir "$WORK_DIR" \
  --no-validate \
  --cfg-options \
    runner.max_iters=20 \
    checkpoint_config.interval=20 \
    log_config.interval=5 \
    data.samples_per_gpu=2

CKPT="$WORK_DIR/iter_20.pth"
if [[ ! -f "$CKPT" ]]; then
  echo "[Smoke] Expected checkpoint not found: $CKPT"
  exit 1
fi

echo "[Smoke] Quick evaluation"
python segmentation/tools/test.py "$CFG" \
  --checkpoint "$CKPT" \
  --eval mIoU mDice mFscore \
  --work-dir "$WORK_DIR/eval"

echo "[Smoke] Done: train + eval pipeline works"
