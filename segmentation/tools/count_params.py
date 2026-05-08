#!/usr/bin/env python3
from __future__ import annotations

"""Count model parameters for MMSegmentation configs.

Example:
    python segmentation/tools/count_params.py \
        --config segmentation/local_configs/segman/xxx.py
"""

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count model parameters")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path to load before counting",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used to build/load model",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Show top-k largest parameter tensors",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Hide per-tensor top-k details",
    )
    return parser.parse_args()


def safe_disable_pretrained(cfg: Config) -> None:
    """Disable pretrained loading to avoid side effects while counting."""
    if "model" not in cfg:
        return

    cfg.model.pretrained = None

    if "backbone" in cfg.model and isinstance(cfg.model.backbone, dict):
        if "pretrained" in cfg.model.backbone:
            cfg.model.backbone.pretrained = None
        if "init_cfg" in cfg.model.backbone:
            cfg.model.backbone.init_cfg = None


def build_model(config_path: str, checkpoint: str, device: str) -> torch.nn.Module:
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmseg.models import build_segmentor

    cfg = Config.fromfile(config_path)
    safe_disable_pretrained(cfg)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )

    if checkpoint:
        load_checkpoint(model, checkpoint, map_location="cpu")

    model.to(torch.device(device))
    model.eval()
    return model


def count_params(model: torch.nn.Module) -> Tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def group_by_top_module(
    model: torch.nn.Module,
) -> List[Tuple[str, int, int]]:
    grouped_total: Dict[str, int] = defaultdict(int)
    grouped_trainable: Dict[str, int] = defaultdict(int)

    for name, p in model.named_parameters():
        top = name.split(".", 1)[0]
        grouped_total[top] += p.numel()
        if p.requires_grad:
            grouped_trainable[top] += p.numel()

    rows = []
    for k in grouped_total:
        t = grouped_total[k]
        tr = grouped_trainable[k]
        rows.append((k, t, tr))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def topk_tensors(model: torch.nn.Module, k: int) -> List[Tuple[str, int, bool]]:
    rows = [(n, p.numel(), p.requires_grad) for n, p in model.named_parameters()]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:k]


def human_million(x: int) -> str:
    return f"{x / 1e6:.3f}M"


def main() -> None:
    args = parse_args()

    # Delay heavy imports so `--help` works even when the runtime env is not ready.
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(args.config, args.checkpoint, device)

    total, trainable, frozen = count_params(model)
    grouped = group_by_top_module(model)

    print("=" * 88)
    print("Parameter Summary")
    print("=" * 88)
    print(f"Total params     : {total:,} ({human_million(total)})")
    print(f"Trainable params : {trainable:,} ({human_million(trainable)})")
    print(f"Frozen params    : {frozen:,} ({human_million(frozen)})")

    print("\nBy top-level module")
    print("-" * 88)
    print(f"{'Module':<24}{'Total':>18}{'Trainable':>18}{'Trainable%':>14}")
    print("-" * 88)
    for module_name, module_total, module_trainable in grouped:
        ratio = 100.0 * module_trainable / module_total if module_total > 0 else 0.0
        print(
            f"{module_name:<24}"
            f"{module_total:>18,}"
            f"{module_trainable:>18,}"
            f"{ratio:>13.2f}%"
        )

    if not args.no_details:
        print("\nTop parameter tensors")
        print("-" * 88)
        print(f"{'Name':<58}{'Params':>18}{'Trainable':>12}")
        print("-" * 88)
        for name, numel, req_grad in topk_tensors(model, args.topk):
            flag = "yes" if req_grad else "no"
            short_name = name if len(name) <= 56 else name[:53] + "..."
            print(f"{short_name:<58}{numel:>18,}{flag:>12}")

    print("=" * 88)


if __name__ == "__main__":
    main()
