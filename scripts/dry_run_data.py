#!/usr/bin/env python3
"""Dry-run for dataset and pipeline: instantiate dataset from config and fetch samples.

Usage:
    python scripts/dry_run_data.py segmentation/local_configs/segman/base/segman_b_wound.py
"""
import sys
import traceback
from mmcv import Config
from mmseg.datasets import build_dataset


def main(cfg_path):
    try:
        cfg = Config.fromfile(cfg_path)
        print('Loaded config:', cfg_path)
        # Build dataset (train)
        dataset_cfg = cfg.data.train
        print('Building dataset from cfg.data.train ...')
        dataset = build_dataset(dataset_cfg)
        print('Dataset built. Length:', len(dataset))
        # Try to get a few samples
        n = min(3, len(dataset))
        for i in range(n):
            item = dataset[i]
            print(f'Item {i} keys:', list(item.keys()))
            if 'img' in item:
                img = item['img']
                print('  img type:', type(img), 'shape/len:', getattr(img, 'shape', getattr(img, '__len__', None)))
            if 'gt_semantic_seg' in item:
                gt = item['gt_semantic_seg']
                print('  gt type:', type(gt), 'shape/len:', getattr(gt, 'shape', getattr(gt, '__len__', None)))
        print('Dry-run data load successful.')
    except Exception as e:
        print('Error during dry-run:')
        traceback.print_exc()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/dry_run_data.py <config_path>')
        sys.exit(1)
    main(sys.argv[1])
