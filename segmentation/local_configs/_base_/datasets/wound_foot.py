dataset_type = 'CustomDataset'
data_root = '/root/autodl-tmp/SegMAN/segmentation/data/wound/foot'
# define dataset classes and palette (required for testing/validation)
# model uses num_classes=2, so provide two class names
classes = ('background', 'wound')
# simple palette: background=black, wound=white
palette = [[0, 0, 0], [255, 255, 255]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
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
        gt_seg_map_loader_cfg=dict(imdecode_backend='cv2'))
)
