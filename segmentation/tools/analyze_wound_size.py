
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmcv.utils import DictAction

from mmseg.apis import single_gpu_test, multi_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes

def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze wound segmentation performance by wound size')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None, type=str)
    parser.add_argument('--work-dir', help='directory to save results')
    parser.add_argument('--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    return parser.parse_args()

def calculate_metrics(pred_mask, gt_mask, class_id=1):
    """
    Calculate IoU, Precision, Recall, DSC for a specific class.
    """
    pred_bin = (pred_mask == class_id)
    gt_bin = (gt_mask == class_id)

    tp = np.sum(pred_bin & gt_bin)
    fp = np.sum(pred_bin & ~gt_bin)
    fn = np.sum(~pred_bin & gt_bin)
    
    union = tp + fp + fn
    iou = tp / union if union > 0 else (1.0 if np.sum(gt_bin) == 0 else 0.0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if np.sum(gt_bin) == 0 and np.sum(pred_bin) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if np.sum(gt_bin) == 0 else 0.0)
    
    dsc = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else (1.0 if np.sum(gt_bin) == 0 else 0.0)
    
    return iou, precision, recall, dsc

def get_category(gt_mask, class_id=1):
    """
    Determine the category based on %GT area.
    """
    total_pixels = gt_mask.size
    assert total_pixels == 512*512
    wound_pixels = np.sum(gt_mask == class_id)
    percentage = (wound_pixels / total_pixels) * 100
    
    if percentage == 0:
        return 1, percentage
    elif percentage < 0.1:
        return 2, percentage
    elif percentage < 0.2:
        return 3, percentage
    elif percentage < 0.4:
        return 4, percentage
    elif percentage < 0.7:
        return 5, percentage
    elif percentage < 1:
        return 6, percentage
    elif percentage < 1.5:
        return 7, percentage
    elif percentage < 2:
        return 8, percentage
    elif percentage < 3:
        return 9, percentage
    else:
        return 10, percentage

def main():
    args = parse_args()
    
    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    setup_multi_processes(cfg)
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # Work directory
    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0], 'wound_size_analysis')
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    # Build dataset and dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # Build model
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    if args.checkpoint:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        else:
            model.PALETTE = dataset.PALETTE

    model = build_dp(model, get_device(), device_ids=cfg.gpu_ids)
    model.eval()

    results_by_category = {i: {'iou': [], 'p': [], 'r': [], 'dsc': []} for i in range(1, 11)}
    
    print("Starting inference and analysis...")
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        
        pred_mask = result[0]
        gt_mask = dataset.get_gt_seg_map_by_idx(i)
        
        # Resize pred_mask if necessary
        if pred_mask.shape != gt_mask.shape:
            # Use mmcv.imresize or similar, but masks are categorical.
            # Nearest neighbor interpolation is needed.
            # pred_mask is (H, W)
            import cv2
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Assuming class 1 is wound
        category, percentage = get_category(gt_mask, class_id=1)
        iou, p, r, dsc = calculate_metrics(pred_mask, gt_mask, class_id=1)
        
        results_by_category[category]['iou'].append(iou)
        results_by_category[category]['p'].append(p)
        results_by_category[category]['r'].append(r)
        results_by_category[category]['dsc'].append(dsc)
        
        prog_bar.update()

    print("\nAnalysis complete. Generating plots...")
    
    # 1. Line plot of DSC vs Category (2-10)
    categories = list(range(2, 11))
    mean_dsc = [np.mean(results_by_category[c]['dsc']) if results_by_category[c]['dsc'] else 0 for c in categories]
    
    plt.figure(figsize=(10, 6))
    plt.plot(categories, mean_dsc, marker='o', linestyle='-')
    plt.title('DSC vs Wound Size Category')
    plt.xlabel('Category (Wound Size)')
    plt.ylabel('Mean DSC')
    plt.xticks(categories)
    plt.grid(True)
    plt.savefig(osp.join(args.work_dir, 'dsc_line_plot.png'))
    plt.close()
    
    # 2. Category count table figure
    category_counts = {cat: len(results_by_category[cat]['dsc']) for cat in range(1, 11)}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    table_data = [[cat, category_counts[cat]] for cat in range(1, 11)]
    table = ax.table(cellText=table_data, colLabels=['Category', 'Image Count'], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    plt.title('Image Count per Category')
    plt.savefig(osp.join(args.work_dir, 'category_counts_table.png'), bbox_inches='tight')
    plt.close()

    # 3. Box plots per category (2 rows x 5 columns), empty fill
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    metrics_order = ['iou', 'p', 'r', 'dsc']
    metric_labels = ['IoU', 'P', 'R', 'DSC']
    for idx, cat in enumerate(range(1, 11)):
        ax = axes[idx // 5][idx % 5]
        data_lists = [results_by_category[cat][m] for m in metrics_order]
        if all(len(lst) == 0 for lst in data_lists):
            ax.text(0.5, 0.5, 'No samples', ha='center', va='center')
            ax.set_title(f'Category {cat}')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Use matplotlib boxplot to control facecolor
        bp = ax.boxplot(
            data_lists,
            tick_labels=metric_labels,
            patch_artist=True,
            showfliers=False
        )
        for box in bp['boxes']:
            box.set(facecolor='none', edgecolor='#1f77b4')
        for median in bp['medians']:
            median.set(color='#d62728')
        for whisker in bp['whiskers']:
            whisker.set(color='#1f77b4')
        for cap in bp['caps']:
            cap.set(color='#1f77b4')

        ax.set_title(f'Category {cat}')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(osp.join(args.work_dir, 'metrics_boxplots_grid.png'))
    plt.close()
    
    print(f"Plots saved to {args.work_dir}")

if __name__ == '__main__':
    main()
