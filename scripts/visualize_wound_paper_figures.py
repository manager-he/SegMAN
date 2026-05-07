#!/usr/bin/env python3
import argparse
import csv
import os
import os.path as osp
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.datasets import build_dataset


@dataclass
class MethodSpec:
    name: str
    config: str
    checkpoint: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready wound segmentation visualizations.")
    parser.add_argument(
        "--primary-config",
        required=True,
        help="Config used to build test dataset.")
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method spec in format: name|config_path|checkpoint_path")
    parser.add_argument(
        "--focus-method",
        default=None,
        help="Method name used for failure/success and bucket analysis. Default: last method.")
    parser.add_argument("--out-dir", required=True, help="Output root directory.")
    parser.add_argument("--device", default="cuda:0", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--class-id", type=int, default=1, help="Foreground class id")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all test samples")
    parser.add_argument("--num-success", type=int, default=5)
    parser.add_argument("--num-failure", type=int, default=5)
    parser.add_argument("--num-boundary", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.9,
        help="All methods dsc >= threshold are treated as success candidates.")
    parser.add_argument(
        "--skip-missing-checkpoint",
        action="store_true",
        help="Skip methods with missing checkpoint files instead of raising error.")
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "savefig.dpi": 300,
        "figure.dpi": 120,
    })


def parse_methods(raw_methods: List[str]) -> List[MethodSpec]:
    methods: List[MethodSpec] = []
    for item in raw_methods:
        parts = item.split("|")
        if len(parts) != 3:
            raise ValueError(f"Invalid --method format: {item}")
        name, config, checkpoint = [p.strip() for p in parts]
        methods.append(MethodSpec(name=name, config=config, checkpoint=checkpoint))
    if not methods:
        raise ValueError("At least one --method is required")
    return methods


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dice_iou_pr_rc(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum(pred_bin & gt_bin))
    fp = int(np.sum(pred_bin & (~gt_bin)))
    fn = int(np.sum((~pred_bin) & gt_bin))

    union = tp + fp + fn
    iou = tp / union if union > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    dsc = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "dsc": float(dsc),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def mask_to_boundary(mask: np.ndarray, width: int = 3) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if np.sum(mask_u8) == 0:
        return np.zeros_like(mask_u8, dtype=bool)
    kernel = np.ones((width, width), dtype=np.uint8)
    dil = cv2.dilate(mask_u8, kernel, iterations=1)
    ero = cv2.erode(mask_u8, kernel, iterations=1)
    boundary = (dil - ero) > 0
    return boundary


def boundary_f1(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    pb = mask_to_boundary(pred_bin)
    gb = mask_to_boundary(gt_bin)
    tp = np.sum(pb & gb)
    fp = np.sum(pb & (~gb))
    fn = np.sum((~pb) & gb)
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return float((2 * tp) / denom)


def wound_ratio(gt_bin: np.ndarray) -> float:
    return float(np.mean(gt_bin) * 100.0)


def classify_failure_reason(img_bgr: np.ndarray, gt_bin: np.ndarray, dsc: float) -> str:
    if np.mean(gt_bin) * 100.0 < 0.1:
        return "Tiny lesion area"

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    illum = float(np.mean(gray))
    if illum < 55 or illum > 205:
        return "Extreme illumination"

    wound_vals = gray[gt_bin]
    bg_vals = gray[~gt_bin]
    if wound_vals.size > 30 and bg_vals.size > 30:
        contrast = abs(float(np.mean(wound_vals) - np.mean(bg_vals)))
        if contrast < 10:
            return "Low lesion-skin contrast"

    gt_boundary = mask_to_boundary(gt_bin)
    if np.sum(gt_boundary) > 0:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        bgrad = float(np.mean(grad[gt_boundary]))
        if bgrad < 10:
            return "Blurred/ambiguous boundary"

    if dsc < 0.5:
        return "Severe under/over segmentation"
    return "Mixed factors"


def load_dataset(config_path: str):
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    return cfg, dataset


def get_image_path(dataset, idx: int) -> str:
    rel = dataset.img_infos[idx]["filename"]
    return osp.join(dataset.img_dir, rel)


def resize_mask_like(mask: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if mask.shape == ref.shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)


def overlay_mask(img_bgr: np.ndarray, mask_bin: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = img_bgr.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    region = mask_bin.astype(bool)
    out[region] = (1.0 - alpha) * out[region] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_boundary(img_bgr: np.ndarray, mask_bin: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    out = img_bgr.copy()
    contours, _ = cv2.findContours(mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def find_crop_bbox(gt_bin: np.ndarray, margin: int = 32) -> Tuple[int, int, int, int]:
    ys, xs = np.where(gt_bin)
    h, w = gt_bin.shape
    if ys.size == 0:
        return 0, h, 0, w
    y1 = max(0, int(np.min(ys)) - margin)
    y2 = min(h, int(np.max(ys)) + margin + 1)
    x1 = max(0, int(np.min(xs)) - margin)
    x2 = min(w, int(np.max(xs)) + margin + 1)
    return y1, y2, x1, x2


def save_case_panel(
    out_path: str,
    img_bgr: np.ndarray,
    gt_bin: np.ndarray,
    per_method_pred: Dict[str, np.ndarray],
    per_method_metric: Dict[str, Dict[str, float]],
    title: str,
) -> None:
    names = list(per_method_pred.keys())
    n_cols = 2 + len(names)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7))

    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")

    gt_overlay = overlay_mask(img_bgr, gt_bin, (0, 255, 0), alpha=0.4)
    gt_overlay = draw_boundary(gt_overlay, gt_bin, (0, 255, 0), thickness=2)
    axes[0, 1].imshow(cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("GT")
    axes[0, 1].axis("off")

    y1, y2, x1, x2 = find_crop_bbox(gt_bin, margin=40)

    for i, name in enumerate(names):
        pred = per_method_pred[name]
        met = per_method_metric[name]
        overlay = overlay_mask(img_bgr, pred, (255, 80, 0), alpha=0.4)
        overlay = draw_boundary(overlay, gt_bin, (0, 255, 0), thickness=2)
        overlay = draw_boundary(overlay, pred, (255, 60, 0), thickness=2)

        axes[0, i + 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0, i + 2].set_title(f"{name}\nDSC={met['dsc']:.3f} IoU={met['iou']:.3f}")
        axes[0, i + 2].axis("off")

        crop = overlay[y1:y2, x1:x2]
        axes[1, i + 2].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[1, i + 2].set_title("Boundary Zoom")
        axes[1, i + 2].axis("off")

    gt_crop = gt_overlay[y1:y2, x1:x2]
    axes[1, 0].imshow(cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Image Zoom")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(cv2.cvtColor(gt_crop, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("GT Zoom")
    axes[1, 1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_case_tile(
    img_bgr: np.ndarray,
    gt_bin: np.ndarray,
    pred_bin: np.ndarray,
    label: str,
    subtitle: str,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(2.9, 5.2))

    overlay = overlay_mask(img_bgr, gt_bin, (0, 255, 0), alpha=0.25)
    overlay = overlay_mask(overlay, pred_bin, (255, 60, 0), alpha=0.25)
    overlay = draw_boundary(overlay, gt_bin, (0, 255, 0), thickness=2)
    overlay = draw_boundary(overlay, pred_bin, (255, 60, 0), thickness=2)

    y1, y2, x1, x2 = find_crop_bbox(gt_bin, margin=36)
    zoom = overlay[y1:y2, x1:x2]

    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title(label)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB))
    axes[1].set_title(subtitle)
    axes[1].axis("off")

    plt.tight_layout(pad=0.4)
    return fig


def render_dsc_bucket_tile(
    img_bgr: np.ndarray,
    gt_bin: np.ndarray,
    pred_bin: np.ndarray,
    title: str,
) -> plt.Figure:
    overlay = img_bgr.copy()
    overlay = overlay_mask(overlay, gt_bin, (0, 255, 0), alpha=0.25)
    overlay = overlay_mask(overlay, pred_bin, (255, 60, 0), alpha=0.25)
    overlay = draw_boundary(overlay, gt_bin, (0, 255, 0), thickness=2)
    overlay = draw_boundary(overlay, pred_bin, (255, 60, 0), thickness=2)

    tp = pred_bin & gt_bin
    fp = pred_bin & (~gt_bin)
    fn = (~pred_bin) & gt_bin
    map_img = np.zeros_like(img_bgr)
    map_img[tp] = (0, 200, 0)
    map_img[fp] = (0, 0, 255)
    map_img[fn] = (255, 0, 0)

    fig, axes = plt.subplots(2, 1, figsize=(2.9, 5.2))
    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title(title)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("TP/FP/FN")
    axes[1].axis("off")

    plt.tight_layout(pad=0.4)
    return fig


def figure_to_image(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    rgb = image[:, :, :3].copy()
    plt.close(fig)
    return rgb


def save_two_row_overview(
    top_images: List[np.ndarray],
    bottom_images: List[np.ndarray],
    top_titles: List[str],
    bottom_titles: List[str],
    out_path: str,
    title: str,
    n_cols: int,
    title_fontsize: int = 10,
) -> None:
    n_items = len(top_images)
    if n_items == 0:
        return

    n_groups = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(2 * n_groups, n_cols, figsize=(3.2 * n_cols, 5.4 * n_groups))
    axes = np.array(axes).reshape(2 * n_groups, n_cols)

    for group_idx in range(n_groups):
        start = group_idx * n_cols
        end = min(start + n_cols, n_items)
        for col in range(n_cols):
            top_ax = axes[2 * group_idx, col]
            bottom_ax = axes[2 * group_idx + 1, col]
            item_idx = start + col

            if item_idx < end:
                top_ax.imshow(top_images[item_idx])
                top_ax.set_title(top_titles[item_idx], fontsize=title_fontsize)
                bottom_ax.imshow(bottom_images[item_idx])
                bottom_ax.set_title(bottom_titles[item_idx], fontsize=title_fontsize)

            top_ax.axis("off")
            bottom_ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_image_grid(
    images: List[np.ndarray],
    labels: List[str],
    out_path: str,
    n_rows: int,
    n_cols: int,
    title: str,
    panel_fontsize: int = 11,
) -> None:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 5.6 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            ax.imshow(images[idx])
            ax.set_title(labels[idx], fontsize=panel_fontsize)
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_boundary_comparison(
    out_path: str,
    img_bgr: np.ndarray,
    gt_bin: np.ndarray,
    method_preds: Dict[str, np.ndarray],
    focus_name: str,
) -> None:
    names = list(method_preds.keys())
    n_cols = 1 + len(names)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.3 * n_cols, 4.8))

    y1, y2, x1, x2 = find_crop_bbox(gt_bin, margin=48)

    gt_view = draw_boundary(img_bgr, gt_bin, (0, 255, 0), thickness=2)
    axes[0].imshow(cv2.cvtColor(gt_view[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    axes[0].set_title("GT (zoom)")
    axes[0].axis("off")

    focus = method_preds[focus_name]

    for i, name in enumerate(names):
        pred = method_preds[name]
        view = draw_boundary(img_bgr, gt_bin, (0, 255, 0), thickness=2)
        view = draw_boundary(view, pred, (255, 80, 0), thickness=2)

        if name != focus_name:
            improve = ((focus == gt_bin) & (pred != gt_bin) & mask_to_boundary(gt_bin, width=5))
            ys, xs = np.where(improve)
            if ys.size > 0:
                y, x = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
                cv2.arrowedLine(view, (max(0, x - 35), max(0, y - 20)), (x, y), (0, 255, 255), 2, tipLength=0.3)

        axes[i + 1].imshow(cv2.cvtColor(view[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(name)
        axes[i + 1].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_dsc_bucket_figures(
    out_dir: str,
    sample_key: str,
    img_bgr: np.ndarray,
    gt_bin: np.ndarray,
    pred_bin: np.ndarray,
) -> None:
    # Figure A: boundary + translucent masks
    overlay = img_bgr.copy()
    overlay = overlay_mask(overlay, gt_bin, (0, 255, 0), alpha=0.25)
    overlay = overlay_mask(overlay, pred_bin, (255, 60, 0), alpha=0.25)
    overlay = draw_boundary(overlay, gt_bin, (0, 255, 0), thickness=2)
    overlay = draw_boundary(overlay, pred_bin, (255, 60, 0), thickness=2)

    # Figure B: opaque TP/FP/FN map
    tp = pred_bin & gt_bin
    fp = pred_bin & (~gt_bin)
    fn = (~pred_bin) & gt_bin
    map_img = np.zeros_like(img_bgr)
    map_img[tp] = (0, 200, 0)
    map_img[fp] = (0, 0, 255)
    map_img[fn] = (255, 0, 0)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Boundary Overlay (GT=green, Pred=orange)")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("TP/FP/FN (TP=green, FP=red, FN=blue)")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(osp.join(out_dir, f"{sample_key}.png"), bbox_inches="tight")
    plt.close(fig)


def quantile_bins(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, q)
    # enforce strictly increasing edges for digitize
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-8
    return edges


def fixed_ratio_bucket_specs() -> Tuple[List[Tuple[float, float]], List[str]]:
    # 用户指定的10个区间（单位%）：0, 0-0.1, 0.1-0.3, 0.3-0.5, 0.5-0.7, 0.7-1, 1-1.5, 1.5-3, 3-6, >=6
    bins = [
        (0.0, 0.0),      # 0
        (0.0, 0.1),      # 0-0.1
        (0.1, 0.3),      # 0.1-0.3
        (0.3, 0.5),      # 0.3-0.5
        (0.5, 0.7),      # 0.5-0.7
        (0.7, 1.0),      # 0.7-1
        (1.0, 1.5),      # 1-1.5
        (1.5, 3.0),      # 1.5-3
        (3.0, 6.0),      # 3-6
        (6.0, np.inf),   # >=6
    ]
    labels = [
        "0",
        "0-0.1",
        "0.1-0.3",
        "0.3-0.5",
        "0.5-0.7",
        "0.7-1",
        "1-1.5",
        "1.5-3",
        "3-6",
        ">=6",
    ]
    return bins, labels


def assign_fixed_ratio_bucket(ratio: float, bins: List[Tuple[float, float]]) -> int:
    # 0单独一类
    if ratio == 0:
        return 0
    for idx in range(1, len(bins)):
        lower, upper = bins[idx]
        if lower <= ratio < upper:
            return idx
    return len(bins) - 1


def main() -> None:
    args = parse_args()
    set_style()
    random.seed(args.seed)
    np.random.seed(args.seed)

    ensure_dir(args.out_dir)
    ensure_dir(osp.join(args.out_dir, "cases", "success"))
    ensure_dir(osp.join(args.out_dir, "cases", "failure"))
    ensure_dir(osp.join(args.out_dir, "cases", "boundary_improvement"))
    ensure_dir(osp.join(args.out_dir, "dsc_buckets"))
    ensure_dir(osp.join(args.out_dir, "stats"))

    methods = parse_methods(args.method)
    if args.focus_method is None:
        focus_name = methods[-1].name
    else:
        focus_name = args.focus_method

    valid_methods: List[MethodSpec] = []
    for m in methods:
        if osp.isfile(m.checkpoint):
            valid_methods.append(m)
            continue
        if args.skip_missing_checkpoint:
            print(f"[Warn] Skip method due to missing checkpoint: {m.name} -> {m.checkpoint}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {m.checkpoint}")

    if not valid_methods:
        raise RuntimeError("No valid methods to evaluate.")

    method_names = [m.name for m in valid_methods]
    if focus_name not in method_names:
        focus_name = method_names[-1]
        print(f"[Warn] focus method not found. Fallback to: {focus_name}")

    cfg, dataset = load_dataset(args.primary_config)

    total_n = len(dataset)
    indices = list(range(total_n))
    if args.max_samples > 0:
        indices = indices[: args.max_samples]

    # Cache GT and image path once
    sample_meta: Dict[int, Dict[str, object]] = {}
    for idx in indices:
        gt = dataset.get_gt_seg_map_by_idx(idx)
        gt_bin = (gt == args.class_id)
        sample_meta[idx] = {
            "img_path": get_image_path(dataset, idx),
            "gt_bin": gt_bin,
            "wound_ratio": wound_ratio(gt_bin),
        }

    per_image_records: List[Dict[str, object]] = []
    preds_by_method: Dict[str, Dict[int, np.ndarray]] = {m.name: {} for m in valid_methods}

    for m in valid_methods:
        print(f"[Infer] {m.name}")
        model = init_segmentor(
            m.config,
            checkpoint=m.checkpoint,
            device=args.device,
            CLASSES=getattr(dataset, "CLASSES", None),
            PALETTE=getattr(dataset, "PALETTE", None),
        )

        for idx in indices:
            img_path = sample_meta[idx]["img_path"]
            gt_bin = sample_meta[idx]["gt_bin"]
            pred = inference_segmentor(model, img_path)[0]
            pred = resize_mask_like(pred, gt_bin.astype(np.uint8))
            pred_bin = (pred == args.class_id)

            preds_by_method[m.name][idx] = pred_bin
            mets = dice_iou_pr_rc(pred_bin, gt_bin)
            b_f1 = boundary_f1(pred_bin, gt_bin)

            per_image_records.append({
                "index": idx,
                "method": m.name,
                "img_path": img_path,
                "wound_ratio": float(sample_meta[idx]["wound_ratio"]),
                "iou": mets["iou"],
                "precision": mets["precision"],
                "recall": mets["recall"],
                "dsc": mets["dsc"],
                "boundary_f1": b_f1,
                "tp": mets["tp"],
                "fp": mets["fp"],
                "fn": mets["fn"],
            })

        del model

    # Save raw metrics
    metrics_csv = osp.join(args.out_dir, "stats", "per_image_metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_records[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_records)

    # Re-index by image and method
    records_by_img: Dict[int, Dict[str, Dict[str, float]]] = {idx: {} for idx in indices}
    for r in per_image_records:
        records_by_img[int(r["index"])][str(r["method"])] = r

    # Success cases: all methods good, focus has strong boundary F1
    success_candidates = []
    for idx in indices:
        row = records_by_img[idx]
        if any(name not in row for name in method_names):
            continue
        if all(float(row[name]["dsc"]) >= args.success_threshold for name in method_names):
            score = float(row[focus_name]["boundary_f1"])
            success_candidates.append((idx, score))
    success_candidates = sorted(success_candidates, key=lambda x: x[1], reverse=True)[: args.num_success]
    success_top_images: List[np.ndarray] = []
    success_bottom_images: List[np.ndarray] = []
    success_top_titles: List[str] = []
    success_bottom_titles: List[str] = []

    for rank, (idx, _) in enumerate(success_candidates, start=1):
        img = mmcv.imread(sample_meta[idx]["img_path"])
        gt_bin = sample_meta[idx]["gt_bin"]
        pred_dict = {name: preds_by_method[name][idx] for name in method_names}
        met_dict = {name: records_by_img[idx][name] for name in method_names}
        save_case_panel(
            out_path=osp.join(args.out_dir, "cases", "success", f"{rank:02d}_idx{idx}.png"),
            img_bgr=img,
            gt_bin=gt_bin,
            per_method_pred=pred_dict,
            per_method_metric=met_dict,
            title=f"Success Case #{rank} (idx={idx})",
        )
        focus_metric = records_by_img[idx][focus_name]
        overlay = overlay_mask(img, gt_bin, (0, 255, 0), alpha=0.25)
        overlay = overlay_mask(overlay, preds_by_method[focus_name][idx], (255, 60, 0), alpha=0.25)
        overlay = draw_boundary(overlay, gt_bin, (0, 255, 0), thickness=2)
        overlay = draw_boundary(overlay, preds_by_method[focus_name][idx], (255, 60, 0), thickness=2)
        y1, y2, x1, x2 = find_crop_bbox(gt_bin, margin=36)
        success_top_images.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        success_bottom_images.append(cv2.cvtColor(overlay[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        success_top_titles.append(f"Case {rank} | idx={idx}")
        success_bottom_titles.append(f"{focus_name} | DSC={float(focus_metric['dsc']):.3f}")

    # 固定2行5列
    if success_top_images:
        n_cols = 5
        n_items = len(success_top_images)
        # 补空白
        while len(success_top_images) < 10:
            blank = np.ones_like(success_top_images[0]) * 255
            success_top_images.append(blank)
            success_bottom_images.append(blank)
            success_top_titles.append("")
            success_bottom_titles.append("")
        save_two_row_overview(
            success_top_images[:10],
            success_bottom_images[:10],
            success_top_titles[:10],
            success_bottom_titles[:10],
            osp.join(args.out_dir, "cases", "success_overview.png"),
            title="Successful Cases Overview",
            n_cols=n_cols,
        )

    # Failure cases: focus method low DSC
    focus_rows = [r for r in per_image_records if r["method"] == focus_name]
    focus_rows = sorted(focus_rows, key=lambda x: float(x["dsc"]))
    chosen_failure = focus_rows[: args.num_failure]
    failure_top_images: List[np.ndarray] = []
    failure_bottom_images: List[np.ndarray] = []
    failure_top_titles: List[str] = []
    failure_bottom_titles: List[str] = []

    for rank, row in enumerate(chosen_failure, start=1):
        idx = int(row["index"])
        img = mmcv.imread(sample_meta[idx]["img_path"])
        gt_bin = sample_meta[idx]["gt_bin"]
        reason = classify_failure_reason(img, gt_bin, float(row["dsc"]))

        pred_dict = {name: preds_by_method[name][idx] for name in method_names}
        met_dict = {name: records_by_img[idx][name] for name in method_names}
        save_case_panel(
            out_path=osp.join(args.out_dir, "cases", "failure", f"{rank:02d}_idx{idx}.png"),
            img_bgr=img,
            gt_bin=gt_bin,
            per_method_pred=pred_dict,
            per_method_metric=met_dict,
            title=f"Failure Case #{rank} (idx={idx}) | reason: {reason}",
        )
        overlay = overlay_mask(img, gt_bin, (0, 255, 0), alpha=0.25)
        overlay = overlay_mask(overlay, preds_by_method[focus_name][idx], (255, 60, 0), alpha=0.25)
        overlay = draw_boundary(overlay, gt_bin, (0, 255, 0), thickness=2)
        overlay = draw_boundary(overlay, preds_by_method[focus_name][idx], (255, 60, 0), thickness=2)
        y1, y2, x1, x2 = find_crop_bbox(gt_bin, margin=36)
        failure_top_images.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        failure_bottom_images.append(cv2.cvtColor(overlay[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        failure_top_titles.append(f"Case {rank} | idx={idx}")
        failure_bottom_titles.append(f"DSC={float(row['dsc']):.3f} | {reason}")

    if failure_top_images:
        n_cols = 5
        n_items = len(failure_top_images)
        while len(failure_top_images) < 10:
            blank = np.ones_like(failure_top_images[0]) * 255
            failure_top_images.append(blank)
            failure_bottom_images.append(blank)
            failure_top_titles.append("")
            failure_bottom_titles.append("")
        save_two_row_overview(
            failure_top_images[:10],
            failure_bottom_images[:10],
            failure_top_titles[:10],
            failure_bottom_titles[:10],
            osp.join(args.out_dir, "cases", "failure_overview.png"),
            title="Failure Cases Overview",
            n_cols=n_cols,
        )

    # Boundary improvement cases
    boundary_candidates = []
    baseline_names = [n for n in method_names if n != focus_name]
    for idx in indices:
        if not baseline_names:
            continue
        focus_bf1 = float(records_by_img[idx][focus_name]["boundary_f1"])
        base_bf1 = max(float(records_by_img[idx][n]["boundary_f1"]) for n in baseline_names)
        gain = focus_bf1 - base_bf1
        boundary_candidates.append((idx, gain))

    # 保证有输出
    boundary_candidates = sorted(boundary_candidates, key=lambda x: x[1], reverse=True)
    selected = boundary_candidates[: args.num_boundary]
    if not selected:
        fallback = sorted([(idx, float(records_by_img[idx][focus_name]["boundary_f1"])) for idx in indices], key=lambda x: x[1], reverse=True)[: args.num_boundary]
        selected = fallback
    for rank, (idx, gain) in enumerate(selected, start=1):
        img = mmcv.imread(sample_meta[idx]["img_path"])
        gt_bin = sample_meta[idx]["gt_bin"]
        pred_dict = {name: preds_by_method[name][idx] for name in method_names}
        out_name = f"{rank:02d}_idx{idx}_gain{gain:.3f}.png"
        save_boundary_comparison(
            out_path=osp.join(args.out_dir, "cases", "boundary_improvement", out_name),
            img_bgr=img,
            gt_bin=gt_bin,
            method_preds=pred_dict,
            focus_name=focus_name,
        )

    # Size-bucket analysis for focus method
    ratios = np.array([float(sample_meta[idx]["wound_ratio"]) for idx in indices], dtype=np.float32)
    ratio_bins, ratio_bin_labels = fixed_ratio_bucket_specs()
    focus_rows_by_idx = {int(r["index"]): r for r in per_image_records if r["method"] == focus_name}

    bucket_metrics = {i: {"iou": [], "precision": [], "recall": [], "dsc": []} for i in range(10)}
    for idx in indices:
        ratio = float(sample_meta[idx]["wound_ratio"])
        b = assign_fixed_ratio_bucket(ratio, ratio_bins)
        r = focus_rows_by_idx[idx]
        bucket_metrics[b]["iou"].append(float(r["iou"]))
        bucket_metrics[b]["precision"].append(float(r["precision"]))
        bucket_metrics[b]["recall"].append(float(r["recall"]))
        bucket_metrics[b]["dsc"].append(float(r["dsc"]))

    # Boxplot for IoU/P/R/DSC across 10 bins
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics_show = ["iou", "precision", "recall", "dsc"]
    titles = ["IoU", "Precision", "Recall", "DSC"]
    for ax, m, t in zip(axes.flatten(), metrics_show, titles):
        data = [bucket_metrics[i][m] if bucket_metrics[i][m] else [np.nan] for i in range(10)]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set(facecolor="#dae8f6", edgecolor="#2c5d8a")
        for med in bp["medians"]:
            med.set(color="#d33f49", linewidth=1.8)
        ax.set_title(f"{t} by GT-size bucket")
        ax.set_xlabel("Fixed wound-ratio bucket")
        ax.set_ylabel(t)
        ax.set_xticklabels(ratio_bin_labels, rotation=35, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(osp.join(args.out_dir, "stats", "size_bucket_boxplots.png"), bbox_inches="tight")
    plt.close(fig)

    # Pie chart: sample count by bucket
    counts = [len(bucket_metrics[i]["dsc"]) for i in range(10)]
    labels = ratio_bin_labels
    fig = plt.figure(figsize=(7.5, 7.5))
    plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Sample distribution by fixed wound-ratio bucket")
    plt.tight_layout()
    plt.savefig(osp.join(args.out_dir, "stats", "size_bucket_pie.png"), bbox_inches="tight")
    plt.close(fig)

    # Histogram: wound pixel ratio distribution
    fig = plt.figure(figsize=(10, 5.2))
    n_hist_bins = min(20, max(5, int(np.sqrt(len(ratios)))))
    plt.hist(ratios, bins=n_hist_bins, color="#5b8e7d", edgecolor="white", alpha=0.9)
    plt.xlabel("Wound pixel ratio in GT (%)")
    plt.ylabel("Number of samples")
    plt.title("Distribution of wound pixel ratio")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(osp.join(args.out_dir, "stats", "wound_ratio_histogram.png"), bbox_inches="tight")
    plt.close(fig)

    # Bar chart: sample count in each quantile bucket
    fig = plt.figure(figsize=(11, 5.2))
    bucket_names = ratio_bin_labels
    bar_colors = ["#8fbcd4" if i % 2 == 0 else "#d8a47f" for i in range(10)]
    plt.bar(bucket_names, counts, color=bar_colors, edgecolor="#4a4a4a", linewidth=0.8)
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts) * 0.015 if max(counts) > 0 else 0.02, str(count), ha="center", va="bottom", fontsize=9)
    plt.xlabel("Fixed bucket by wound pixel ratio (%)")
    plt.ylabel("Number of samples")
    plt.title("Sample count per fixed wound-ratio bucket")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(osp.join(args.out_dir, "stats", "wound_ratio_bucket_bar.png"), bbox_inches="tight")
    plt.close(fig)

    # DSC line chart by bucket (median)
    dsc_med = [float(np.nanmedian(bucket_metrics[i]["dsc"])) if bucket_metrics[i]["dsc"] else np.nan for i in range(10)]
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, 11), dsc_med, marker="o", color="#1f7a8c", linewidth=2)
    plt.xticks(np.arange(1, 11), ratio_bin_labels, rotation=35, ha="right")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Fixed wound-ratio bucket")
    plt.ylabel("Median DSC")
    plt.title("DSC trend across fixed wound-ratio buckets")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(osp.join(args.out_dir, "stats", "size_bucket_dsc_line.png"), bbox_inches="tight")
    plt.close(fig)

    # Export bucket boundaries and summary
    with open(osp.join(args.out_dir, "stats", "size_bucket_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bucket", "bucket_label", "ratio_min", "ratio_max", "count", "mean_iou", "mean_precision", "mean_recall", "mean_dsc"])
        for i in range(10):
            vals_iou = bucket_metrics[i]["iou"]
            vals_p = bucket_metrics[i]["precision"]
            vals_r = bucket_metrics[i]["recall"]
            vals_d = bucket_metrics[i]["dsc"]
            writer.writerow([
                i + 1,
            ratio_bin_labels[i],
            float(ratio_bins[i][0]),
            "inf" if np.isinf(ratio_bins[i][1]) else float(ratio_bins[i][1]),
                len(vals_d),
                float(np.mean(vals_iou)) if vals_iou else np.nan,
                float(np.mean(vals_p)) if vals_p else np.nan,
                float(np.mean(vals_r)) if vals_r else np.nan,
                float(np.mean(vals_d)) if vals_d else np.nan,
            ])

    # DSC bucket visualization: <80, [80,90), [90,95), >=95
    dsc_buckets = {
        "A_lt80": [],
        "B_80_90": [],
        "C_90_95": [],
        "D_ge95": [],
    }
    for idx in indices:
        d = float(focus_rows_by_idx[idx]["dsc"])
        if d < 0.80:
            dsc_buckets["A_lt80"].append((idx, d))
        elif d < 0.90:
            dsc_buckets["B_80_90"].append((idx, d))
        elif d < 0.95:
            dsc_buckets["C_90_95"].append((idx, d))
        else:
            dsc_buckets["D_ge95"].append((idx, d))

    dsc_bucket_overview_images: List[np.ndarray] = []
    dsc_bucket_overview_labels: List[str] = []
    for bucket_name, items in dsc_buckets.items():
        if not items:
            continue
        items = sorted(items, key=lambda x: x[1])
        pick = [items[len(items) // 3], items[(2 * len(items)) // 3]] if len(items) >= 2 else [items[0]]
        for j, (idx, dsc_val) in enumerate(pick, start=1):
            img = mmcv.imread(sample_meta[idx]["img_path"])
            gt_bin = sample_meta[idx]["gt_bin"]
            pred_bin = preds_by_method[focus_name][idx]
            sample_key = f"{bucket_name}_sample{j}_idx{idx}_dsc{dsc_val:.3f}"
            save_dsc_bucket_figures(osp.join(args.out_dir, "dsc_buckets"), sample_key, img, gt_bin, pred_bin)
            tile = render_dsc_bucket_tile(
                img,
                gt_bin,
                pred_bin,
                title=f"{bucket_name} | S{j} | DSC={dsc_val:.3f}",
            )
            dsc_bucket_overview_images.append(figure_to_image(tile))
            dsc_bucket_overview_labels.append(f"idx={idx}")

    if dsc_bucket_overview_images:
        n_items = len(dsc_bucket_overview_images)
        if n_items <= 4:
            n_rows, n_cols = 2, 2
        elif n_items <= 8:
            n_rows, n_cols = 2, 4
        else:
            n_rows = 2
            n_cols = int(np.ceil(n_items / n_rows))
        save_image_grid(
            dsc_bucket_overview_images,
            dsc_bucket_overview_labels,
            osp.join(args.out_dir, "dsc_buckets", "dsc_bucket_overview.png"),
            n_rows=n_rows,
            n_cols=n_cols,
            title="DSC Bucket Overview",
            panel_fontsize=10,
        )

    # Boundary-head effectiveness figure: mean boundary F1 + 95% CI
    fig = plt.figure(figsize=(9.5, 5))
    means = []
    cis = []
    for name in method_names:
        arr = np.array([float(records_by_img[idx][name]["boundary_f1"]) for idx in indices], dtype=np.float32)
        means.append(float(np.mean(arr)))
        se = float(np.std(arr, ddof=1) / np.sqrt(max(len(arr), 1))) if len(arr) > 1 else 0.0
        cis.append(1.96 * se)

    xs = np.arange(len(method_names))
    plt.bar(xs, means, yerr=cis, capsize=4, color=["#8fbcd4", "#7aa08d", "#d8a47f", "#c76d6d"][: len(method_names)])
    plt.xticks(xs, method_names, rotation=15)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Boundary F1")
    plt.title("Boundary quality comparison (mean +/- 95% CI)")
    plt.tight_layout()
    plt.savefig(osp.join(args.out_dir, "stats", "boundary_effectiveness_bar.png"), bbox_inches="tight")
    plt.close(fig)

    # Method-level summary
    with open(osp.join(args.out_dir, "stats", "method_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "mean_iou", "mean_precision", "mean_recall", "mean_dsc", "mean_boundary_f1"])
        for name in method_names:
            rows = [records_by_img[idx][name] for idx in indices]
            writer.writerow([
                name,
                float(np.mean([float(r["iou"]) for r in rows])),
                float(np.mean([float(r["precision"]) for r in rows])),
                float(np.mean([float(r["recall"]) for r in rows])),
                float(np.mean([float(r["dsc"]) for r in rows])),
                float(np.mean([float(r["boundary_f1"]) for r in rows])),
            ])

    print("[Done] Visualization package created.")
    print(f"[Done] Output root: {args.out_dir}")
    print(f"[Done] Raw metrics: {metrics_csv}")


if __name__ == "__main__":
    main()
