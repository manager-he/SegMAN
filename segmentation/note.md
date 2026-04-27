
# 主要代码结构
segmentation模块是做分割的

| 代码的核心部分是open-mmlab的mmsegmentation框架，SegMAN的作者仅仅是修改了local_configs，以及mmseg/models目录下增加了backbone的segman_encoder和decode_heads的segman_decoder。注意，在最顶层的segman_encoder就是对encoder整理了一遍，仅作文本，无代码价值

local_configs原本包含了base目录和众多模型单独的目录，作者删去了其余的模型并增加了一个segman目录
segmentation/local_configs/segman/
├── base
│   ├── segman_b_ade.py
│   ├── segman_b_cityscapes.py
│   ├── segman_b_coco.py
│   ├── segman_b_wound.py
│   └── segman_b_wound_smoke.py
├── large
├── small
└── tiny
该目录下的cfg文件不仅仅是对模型的config，还包含了数据集，训练策略等
字段有__base__:model,dataset,runtime,schedules配置文件的地址，norm_cfg, model, optimizer, lr_config, data, evaluation

cfg文件会通过mmseg框架构建mmseg/models下定义的模型，Segman对应了segman_encoder和segman_decoder

# tools/train 数据流

接受命令行参数+配置文件
调用mmseg.models.build_segmentor构建模型，包括backbone/neck/decode head/aux head
构建数据集mmseg.datasets.build_dataset
构建DataLoader

mmseg/models/segmentors 是训练的基类



# segman_base + wound_cfg

**1. 数据流（从原始图像到模型输入）**
数据集定义在 wound_foot.py。

训练集 flow：
1. 读取图像与标注：LoadImageFromFile + LoadAnnotations  
2. 几何增强：Resize 到基准尺度 1024x1024，比例随机在 0.5 到 2.0  
3. 随机裁剪：RandomCrop 到 512x512，cat_max_ratio=0.75  
4. 随机翻转：RandomFlip，prob=0.5  
5. 光照/颜色增强：PhotoMetricDistortion  
6. 归一化：Normalize(mean/std, to_rgb=True)  
7. Padding：Pad 到 512x512，标注填充值 255（ignore）  
8. 打包与收集：DefaultFormatBundle + Collect(img, gt_semantic_seg)

验证/测试 flow：
1. LoadImageFromFile  
2. MultiScaleFlipAug，img_scale=1024x1024，flip=False  
3. 内部 transform：Resize(keep_ratio=True) + RandomFlip + Normalize + ImageToTensor + Collect(img)

结论：
1. 原始图像在训练时明确经过了数据增强（尺度、裁剪、翻转、光度扰动）。  
2. 验证/测试阶段主要是确定性预处理（归一化等）；flip=False 下通常不会真正做翻转测试增强。  

**2. 模型构建后的关键参数**
模型相关来自：
- 覆盖配置：segman_b_wound.py
- 基线模板：segman.py

最终重点：
1. 框架类型：EncoderDecoder  
2. Backbone：SegMANEncoder_b  
3. 预训练权重：../pretrained/SegMAN_Encoder_b.pth.tar  
4. Decode head：SegMANDecoder  
5. 多尺度输入通道：in_channels=[96,160,364,560]，in_index=[0,1,2,3]  
6. 解码通道：channels=180，feat_proj_dim=320，dropout_ratio=0.1  
7. 类别数：num_classes=2（与数据集 classes 背景/伤口一致）  
8. 归一化：SyncBN  
9. 测试方式：slide 推理，crop_size=(1024,1024)，stride=(768,768)

补充：
1. 基线模型里的 num_classes=19 和通道设置已被当前文件覆盖为 wound 任务配置。  
2. slide test 会提升大图推理稳定性，但速度相对 whole mode 更慢。  

**3. 训练阶段关键参数**
训练策略来自：
- 覆盖配置：segman_b_wound.py
- 基线 schedule：schedule_160k_adamw.py
- runtime：default_runtime.py

最终重点：
1. Optimizer：AdamW，lr=6e-5，betas=(0.9,0.999)，weight_decay=0.01  
2. 参数分组策略：  
- pos_block 不做 weight decay（decay_mult=0）  
- norm 不做 weight decay（decay_mult=0）  
- head 学习率放大 10 倍（lr_mult=10）  
3. LR 策略：poly，power=1.0，warmup=linear，warmup_iters=1500，warmup_ratio=1e-6，min_lr=0  
4. Runner：IterBasedRunner，max_iters=160000（160k iteration 训练）  
5. Batch 相关：samples_per_gpu=2，workers_per_gpu=2  
6. 评估：每 4000 iter，指标 mIoU/mDice/mFscore  
7. Checkpoint：每 4000 iter 保存（来自 base schedule）  
8. 运行时：cudnn_benchmark=True，workflow=[('train',1)]


# segman不同大小的模型参数比较

可以直接按官方表看，SegMAN 各规模参数量对比如下（来自 README.md）：

| 模型 | Params | 相对 T 的倍数 |
|---|---:|---:|
| SegMAN-T | 6.4M | 1.0x |
| SegMAN-S | 29.4M | 4.59x |
| SegMAN-B | 51.8M | 8.09x |
| SegMAN-L | 92.6M | 14.47x |

导致 SegMAN 不同规模参数量差异的“代码根因”，主要是这两块：

Backbone 变宽 + 变深（主因，贡献最大）
Decode Head 跟着 backbone 通道数一起变大（次主因）

其中最关键的是：

* embed_dims（每 stage 通道宽度）
* depths（每 stage block 数量）
* mlp_ratios（FFN 隐层比例）
* num_heads（影响注意力分头与相对位置偏置参数）

构造器里会按 depth 循环堆叠 block（深度直接乘参数量），见 segman_encoder.py:761；并按 embed_dims 建立 stage 间下采样卷积，见 segman_encoder.py:953。

每个 block 里最吃参数的是这些层：

* 注意力 qkv/proj 1x1 卷积，见 segman_encoder.py:529, segman_encoder.py:531
* FFN 的 fc1/fc2，见 segman_encoder.py:639, segman_encoder.py:642

所以通道 C 增大时，很多项近似按 C^2 增长；再乘以 depth，参数会明显跳升。

* linear_c2/c3/c4 依赖 in_channels 与 feat_proj_dim，见 segman_decoder.py:860
* linear_fuse 依赖 feat_proj_dim 和 channels，见 segman_decoder.py:864
* cat/proj_out/reduce_channels 也依赖 channels 与 feat_proj_dim，见 segman_decoder.py:873, segman_decoder.py:891, segman_decoder.py:899

对照一下四种规模的核心超参（backbone）：

T: embed_dims [32,64,144,192], depths [2,2,4,2]
S: embed_dims [64,144,288,512], depths [2,2,10,4]
B: embed_dims [96,160,364,560], depths [4,4,18,4]
L: embed_dims [96,192,432,640], depths [4,4,28,4]

# Data augmentation

# Complex augmentation for wound data with illumination/device/domain shifts.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# Decoder-boundery

解码头新增参数：boundary_enabled、boundary_loss_weight、boundary_kernel_size。
新增轻量边界头：DepthwiseSeparableConvModule + 1x1 conv，输出单通道边界 logit。
if self.boundary_enabled:
    self.boundary_head = nn.Sequential(
        DepthwiseSeparableConvModule(
            self.embed_dim,
            self.embed_dim,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU')),
        nn.Conv2d(self.embed_dim, 1, kernel_size=1))

边界 GT 形态学生成与分辨率对齐
* segman_decoder.py:998 _build_boundary_target
* segman_decoder.py:1008 _loss_boundary
用 max_pool/min_pool 的形态学梯度生成边界目标。对 ignore 区域及其邻域做 loss mask，避免噪声监督。
边界 logit 在计算损失前 resize 到 GT 尺寸，保证分辨率严格对齐。

复写基类的losses函数，使得计算时加上边界损失
总损失中新增 loss_boundary，按 boundary_loss_weight 加权。

# out channels 修复

我记录在某个地方了

# focal loss 修复

Focal loss 的算法会用 out_channels 等价 num_classes，参与 one-hot 计算，就导致one-hot只对一个类别进行编码

因此增加对out_channels为1的特殊判断
```python
# Binary segmentation with out_channels=1 provides labels in {0, 1}
# and should be treated as dense binary targets instead of class
# indices, otherwise one_hot(num_classes=1) will fail on label=1.
if num_classes == 1 and target.dim() == 1:
    target = target.type_as(pred).view(-1, 1)
    one_hot_target = target
    calculate_loss_func = py_sigmoid_focal_loss
```

# 对比实验

segman-base basic: 160k iters
+------------+-------+-------+-------+--------+-----------+--------+
|   Class    |  IoU  |  Acc  |  Dice | Fscore | Precision | Recall |
+------------+-------+-------+-------+--------+-----------+--------+
| background | 99.81 | 99.89 | 99.91 | 99.91  |   99.92   | 99.89  |
|   wound    | 86.12 | 93.82 | 92.54 | 92.54  |    91.3   | 93.82  |
+------------+-------+-------+-------+--------+-----------+--------+


segman-tiny basic: 24k iters
+------------+-------+-------+-------+--------+-----------+--------+
|   Class    |  IoU  |  Acc  |  Dice | Fscore | Precision | Recall |
+------------+-------+-------+-------+--------+-----------+--------+
| background | 99.81 | 99.93 | 99.91 | 99.91  |   99.89   | 99.93  |
|   wound    | 85.64 | 90.77 | 92.26 | 92.26  |    93.8   | 90.77  |
+------------+-------+-------+-------+--------+-----------+--------+

data agumentation: 效果更差了
+------------+-------+-------+-------+--------+-----------+--------+
|   Class    |  IoU  |  Acc  |  Dice | Fscore | Precision | Recall |
+------------+-------+-------+-------+--------+-----------+--------+
| background | 99.39 | 99.98 | 99.69 | 99.69  |   99.41   | 99.98  |
|   wound    | 51.09 |  52.0 | 67.63 | 67.63  |    96.7   |  52.0  |
+------------+-------+-------+-------+--------+-----------+--------+
data agumentation 2: 24k iters
+------------+-------+-------+-------+--------+-----------+--------+
|   Class    |  IoU  |  Acc  |  Dice | Fscore | Precision | Recall |
+------------+-------+-------+-------+--------+-----------+--------+
| background |  99.8 | 99.92 |  99.9 |  99.9  |   99.88   | 99.92  |
|   wound    | 84.96 |  90.3 | 91.87 | 91.87  |    93.5   |  90.3  |
+------------+-------+-------+-------+--------+-----------+--------+
和basic配置持平，确定为数据增强方案了


segman-tiny combined loss: 24k
+------------+-------+-------+-------+--------+-----------+--------+
|   Class    |  IoU  |  Acc  |  Dice | Fscore | Precision | Recall |
+------------+-------+-------+-------+--------+-----------+--------+
| background | 99.79 | 99.86 |  99.9 |  99.9  |   99.93   | 99.86  |
|   wound    | 84.98 | 94.76 | 91.88 | 91.88  |   89.18   | 94.76  |
+------------+-------+-------+-------+--------+-----------+--------+

segman-tiny basic + boundary: 24k
+------------+-------+-------+-------+--------+-----------+--------+
|   Class    |  IoU  |  Acc  |  Dice | Fscore | Precision | Recall |
+------------+-------+-------+-------+--------+-----------+--------+
| background | 99.81 | 99.92 |  99.9 |  99.9  |   99.88   | 99.92  |
|   wound    | 85.25 | 90.42 | 92.04 | 92.04  |   93.72   | 90.42  |
+------------+-------+-------+-------+--------+-----------+--------+