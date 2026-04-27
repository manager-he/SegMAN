#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="segmentation/work_dirs/wound_tiny_ablation"
mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary_metrics.csv"

# Fast ablation defaults (override with env vars if needed)
MAX_ITERS="${MAX_ITERS:-24000}"
CKPT_INTERVAL="${CKPT_INTERVAL:-4000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-4000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"

echo "config,work_dir,checkpoint,mIoU,mDice,mFscore,aAcc" > "$SUMMARY_CSV"

CONFIGS=(
  # "segmentation/local_configs/segman/tiny/segman_t_wound_bce_basic_aug.py"
  # "segmentation/local_configs/segman/tiny/segman_t_wound_bce_complex_aug.py"
  # "segmentation/local_configs/segman/tiny/segman_t_wound_bce_dice_focal_complex_aug.py"
  # "segmentation/local_configs/segman/tiny/segman_t_wound_bce_complex_aug_boundary.py",
  "segmentation/local_configs/segman/tiny/segman_t_wound_bce_dice_focal_complex_aug_boundary.py"  
)

for cfg in "${CONFIGS[@]}"; do
  exp_name="$(basename "${cfg%.py}")"
  work_dir="$OUT_ROOT/$exp_name"
  eval_dir="$work_dir/eval"
  mkdir -p "$work_dir" "$eval_dir"

  echo "[Train] $exp_name"
  CUDA_LAUNCH_BLOCKING=1 python segmentation/tools/train.py "$cfg" \
    --work-dir "$work_dir" \
    --cfg-options \
      runner.max_iters="$MAX_ITERS" \
      checkpoint_config.interval="$CKPT_INTERVAL" \
      evaluation.interval="$EVAL_INTERVAL" \
      log_config.interval="$LOG_INTERVAL"

  ckpt=""
  if [[ -f "$work_dir/latest.pth" ]]; then
    ckpt="$work_dir/latest.pth"
  else
    ckpt="$(ls -1 "$work_dir"/iter_*.pth 2>/dev/null | tail -n 1 || true)"
  fi

  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    echo "[Error] checkpoint not found for $exp_name"
    exit 1
  fi

  echo "[Eval] $exp_name"
  python segmentation/tools/test.py "$cfg" \
    --checkpoint "$ckpt" \
    --eval mIoU mDice mFscore \
    --work-dir "$eval_dir"

  latest_json="$(ls -1t "$eval_dir"/eval_single_scale_*.json | head -n 1)"
  metrics_line="$(python - "$latest_json" <<'PY'
import json
import sys
p = sys.argv[1]
with open(p, 'r', encoding='utf-8') as f:
    d = json.load(f)
print(','.join(str(d.get(k, 'NA')) for k in ['mIoU', 'mDice', 'mFscore', 'aAcc']))
PY
)"

  echo "$cfg,$work_dir,$ckpt,$metrics_line" >> "$SUMMARY_CSV"

done

echo "[Done] All runs complete."
echo "[Done] Summary: $SUMMARY_CSV"
echo "[Done] Overrides: max_iters=$MAX_ITERS, ckpt_interval=$CKPT_INTERVAL, eval_interval=$EVAL_INTERVAL, log_interval=$LOG_INTERVAL"
