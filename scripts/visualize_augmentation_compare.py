#!/usr/bin/env python3
"""Generate paper-style augmentation comparison figures.

This script compares two hand-crafted augmentation policies on the same images:
1) feature-preserving complexity augmentation
2) destructive CutOut + CLAHE augmentation

Default output directory:
    work_dirs/paper_figures/data_augmentation
"""

import argparse
import os
import os.path as osp
import random
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize two augmentation schemes for paper figures")
    parser.add_argument(
        "--img-dir",
        default="segmentation/data/wound/foot/images/validation",
        help="Input image directory",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/paper_figures/data_augmentation",
        help="Output directory for figures",
    )
    parser.add_argument("--num-images", type=int, default=6, help="How many source images to visualize")
    parser.add_argument("--num-samples", type=int, default=3, help="Samples per policy per image")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Longest side for visualization to control memory (0 disables resize)",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_images(img_dir: str) -> List[str]:
    files = []
    for name in os.listdir(img_dir):
        if name.lower().endswith(VALID_EXTS):
            files.append(osp.join(img_dir, name))
    files.sort()
    return files


def resize_for_vis(img: np.ndarray, max_size: int) -> np.ndarray:
    if max_size <= 0:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return img
    scale = float(max_size) / float(longest)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def feature_preserving_aug(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Increase appearance complexity while preserving lesion structure."""
    out = img_bgr.copy()

    # Mild scale jitter around 1.0, then center-crop/pad back.
    scale = rng.uniform(0.92, 1.08)
    h, w = out.shape[:2]
    nh, nw = max(16, int(h * scale)), max(16, int(w * scale))
    out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_LINEAR)

    if nh >= h and nw >= w:
        y0 = (nh - h) // 2
        x0 = (nw - w) // 2
        out = out[y0:y0 + h, x0:x0 + w]
    else:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        y0 = (h - nh) // 2
        x0 = (w - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = out
        out = canvas

    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.2:
        out = cv2.flip(out, 0)

    angle = rng.uniform(-10.0, 10.0)
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    out = cv2.warpAffine(out, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Mild brightness/contrast changes.
    alpha = rng.uniform(0.9, 1.15)
    beta = rng.uniform(-12.0, 12.0)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Mild saturation jitter in HSV space.
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= rng.uniform(0.9, 1.15)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out


def destructive_cutout_clahe_aug(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Intentionally destructive augmentation policy explored previously."""
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # Strong CLAHE can over-amplify local contrast and noise.
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clip = rng.uniform(4.0, 8.0)
    grid = rng.choice([6, 8, 10, 12])
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Aggressive cutout blocks may occlude lesion regions.
    n_holes = rng.randint(2, 5)
    for _ in range(n_holes):
        rh = rng.uniform(0.08, 0.2)
        rw = rng.uniform(0.08, 0.2)
        ch = max(4, int(h * rh))
        cw = max(4, int(w * rw))
        cy = rng.randint(0, max(0, h - 1))
        cx = rng.randint(0, max(0, w - 1))
        y1 = max(0, cy - ch // 2)
        y2 = min(h, y1 + ch)
        x1 = max(0, cx - cw // 2)
        x2 = min(w, x1 + cw)
        fill_mode = rng.choice(["black", "gray", "noise"])
        if fill_mode == "black":
            out[y1:y2, x1:x2] = 0
        elif fill_mode == "gray":
            out[y1:y2, x1:x2] = 127
        else:
            noise = rng.randint(0, 255)
            out[y1:y2, x1:x2] = noise

    # Extra contrast stretch to exaggerate appearance shifts.
    alpha = rng.uniform(1.1, 1.35)
    beta = rng.uniform(-20.0, 20.0)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return out


def save_comparison_panel(
    out_path: str,
    image_name: str,
    src_bgr: np.ndarray,
    preserve_imgs: List[np.ndarray],
    destructive_imgs: List[np.ndarray],
) -> None:
    n = len(preserve_imgs)
    cols = 1 + n  # Adjusted to only include original + n samples per row
    fig, axes = plt.subplots(2, cols, figsize=(3.2 * cols, 6.0))

    for r in range(2):
        for c in range(cols):
            axes[r, c].axis("off")

    src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(src_rgb)
    axes[0, 0].set_title("Original")
    axes[1, 0].imshow(src_rgb)
    axes[1, 0].set_title("Original")

    for i, im in enumerate(preserve_imgs):
        axes[0, 1 + i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[0, 1 + i].set_title(f"Preserve+Complex #{i + 1}")

    for i, im in enumerate(destructive_imgs):
        axes[1, 1 + i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[1, 1 + i].set_title(f"CutOut+CLAHE #{i + 1}")

    fig.suptitle(f"Augmentation Comparison - {image_name}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    ensure_dir(args.out_dir)

    if not osp.isdir(args.img_dir):
        raise FileNotFoundError(f"Image directory not found: {args.img_dir}")

    image_paths = list_images(args.img_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {args.img_dir}")

    if args.num_images > 0:
        rng.shuffle(image_paths)
        image_paths = image_paths[: args.num_images]

    for p in image_paths:
        name = osp.basename(p)
        src = cv2.imread(p, cv2.IMREAD_COLOR)
        if src is None:
            print(f"[Warn] Skip unreadable image: {p}")
            continue

        src = resize_for_vis(src, args.max_size)

        preserve = [feature_preserving_aug(src, rng) for _ in range(args.num_samples)]
        destructive = [destructive_cutout_clahe_aug(src, rng) for _ in range(args.num_samples)]

        out_name = osp.splitext(name)[0] + "_aug_compare.png"
        out_path = osp.join(args.out_dir, out_name)
        save_comparison_panel(out_path, name, src, preserve, destructive)
        print(f"[Save] {out_path}")

    print("[Done] Data augmentation visualization completed.")
    print(f"[Done] Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
