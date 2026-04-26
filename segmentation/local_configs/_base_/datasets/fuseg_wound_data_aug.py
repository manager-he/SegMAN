dataset_type = 'CustomDataset'
data_root = '/root/autodl-tmp/SegMAN/segmentation/data/wound/foot'

classes = ('background', 'wound')
palette = [[0, 0, 0], [255, 255, 255]]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# FUSeg-friendly crop size: preserve local lesion boundaries while keeping context.
crop_size = (512, 512)

# Complex augmentation for wound data with illumination/device/domain shifts.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.7, 1.5)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.85),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.2, direction='vertical'),
    dict(type='RandomRotate', prob=0.3, degree=20, pad_val=0, seg_pad_val=255),
    dict(type='PhotoMetricDistortion',
         brightness_delta=20,
         contrast_range=(0.8, 1.2),
         saturation_range=(0.8, 1.2),
         hue_delta=10),
    dict(type='CLAHE', clip_limit=4.0, tile_grid_size=(8, 8)),
    dict(type='RandomCutOut',
         prob=0.2,
         n_holes=(1, 2),
         cutout_ratio=[(0.08, 0.08), (0.12, 0.12)],
         fill_in=(0, 0, 0),
         seg_fill_in=255),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=train_pipeline,
        gt_seg_map_loader_cfg=dict(imdecode_backend='cv2')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline,
        gt_seg_map_loader_cfg=dict(imdecode_backend='cv2')),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline,
        gt_seg_map_loader_cfg=dict(imdecode_backend='cv2')))
