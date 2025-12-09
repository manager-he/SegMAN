#!/usr/bin/env python3
"""Convert binary/black-white masks to single-channel class-index masks.

Usage:
    python scripts/convert_masks_wound.py --src segmentation/data/wound/foot_raw --dst segmentation/data/wound/foot
"""
import argparse
import os
from PIL import Image
import numpy as np


def convert_split(src_split_dir, dst_split_dir):
    os.makedirs(dst_split_dir, exist_ok=True)
    for fname in os.listdir(src_split_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            continue
        src_p = os.path.join(src_split_dir, fname)
        dst_p = os.path.join(dst_split_dir, fname)
        im = Image.open(src_p).convert('L')
        arr = np.array(im)
        # Map white (>=128) to 1, black to 0
        mask = (arr >= 128).astype('uint8')
        out = Image.fromarray(mask)
        out.save(dst_p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='source root where raw masks are stored (should contain training/validation)')
    parser.add_argument('--dst', required=True, help='destination root (will create annotation/training and annotation/validation)')
    args = parser.parse_args()

    for split in ['training', 'validation']:
        src_dir = os.path.join(args.src, split)
        dst_dir = os.path.join(args.dst, 'annotation', split)
        if not os.path.isdir(src_dir):
            print(f'Source split dir not found: {src_dir}, skipping')
            continue
        os.makedirs(dst_dir, exist_ok=True)
        convert_split(src_dir, dst_dir)
        print(f'Converted masks from {src_dir} -> {dst_dir}')


if __name__ == '__main__':
    main()
