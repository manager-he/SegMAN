"""Batch visualization for wound/foot validation samples.

Selects the first N images in the validation set, runs inference with a
checkpoint, overlays a green mask on the wound region, and saves a 2-row grid
containing originals (top) and masked results (bottom).
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

from mmseg.apis import inference_segmentor, init_segmentor


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Batch visualize segmentation results")
    parser.add_argument(
        "config",
        help="Config file path (e.g., local_configs/segman/base/segman_b_wound.py)",
    )
    parser.add_argument(
        "checkpoint",
        help="Checkpoint file path (e.g., outputs/b_wound_full/iter_104000.pth)",
    )
    parser.add_argument(
        "--images-dir",
        default="data/wound/foot/images/validation",
        help="Directory that contains validation images",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=6,
        help="Number of validation images to visualize (sorted by name)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for inference (e.g., cuda:0 or cpu)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Opacity for the green overlay in [0, 1]",
    )
    parser.add_argument(
        "--output",
        default="outputs/eval/validation_grid.png",
        help="Output path for the combined grid image",
    )
    return parser


def overlay_mask(
    image: Image.Image,
    seg_map: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.9,
) -> Image.Image:
    """Overlay a single-class mask on top of the image.

    Args:
        image: PIL image in RGB.
        seg_map: 2D numpy array with class indices.
        color: RGB color for the mask.
        alpha: Blending factor for the mask.
    """

    if seg_map.shape != (image.height, image.width):
        seg_map = np.array(
            Image.fromarray(seg_map.astype(np.uint8)).resize(
                image.size, resample=Image.NEAREST
            )
        )

    img_arr = np.array(image).astype(np.float32)
    mask = seg_map > 0
    if not mask.any():
        return image.copy()

    result = img_arr.copy()
    color_arr = np.array(color, dtype=np.float32)
    result[mask] = img_arr[mask] * (1.0 - alpha) + color_arr * alpha
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def make_grid(row1: Sequence[Image.Image], row2: Sequence[Image.Image]) -> Image.Image:
    assert len(row1) == len(row2), "Row lengths must match"
    width, height = row1[0].size
    grid = Image.new("RGB", (width * len(row1), height * 2))
    for idx, img in enumerate(row1):
        grid.paste(img, (idx * width, 0))
    for idx, img in enumerate(row2):
        grid.paste(img, (idx * width, height))
    return grid


def load_images(images_dir: Path, num: int) -> List[Path]:
    candidates = sorted(images_dir.glob("*.png"))
    if len(candidates) < num:
        raise ValueError(f"Requested {num} images but found {len(candidates)} in {images_dir}")
    return candidates[:num]


def main():
    args = parse_args().parse_args()
    images_dir = Path(args.images_dir)
    out_path = Path(args.output)

    img_paths = load_images(images_dir, args.num)

    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    originals: List[Image.Image] = []
    overlays: List[Image.Image] = []

    target_size = None
    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        if target_size is None:
            target_size = image.size
        elif image.size != target_size:
            image = image.resize(target_size, resample=Image.BILINEAR)

        seg_output = inference_segmentor(model, str(img_path))
        seg_map = seg_output[0] if isinstance(seg_output, (list, tuple)) else seg_output

        masked = overlay_mask(image, np.array(seg_map), alpha=args.alpha)
        originals.append(image)
        overlays.append(masked)

    grid = make_grid(originals, overlays)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"Saved grid to {out_path}")


if __name__ == "__main__":
    main()