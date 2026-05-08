# 训练配置
| 所有信息来源segmentation/work_dirs/wound_tiny_ablation/segman_t_wound_bce_dice_focal_complex_aug_boundary/20260506_175707.log

### 运行环境与硬件
见 20260506_175707.log:8, 20260506_175707.log:10, 20260506_175707.log:12, 20260506_175707.log:13, 20260506_175707.log:30
包括 Linux、Python 3.10.20、RTX 4090、CUDA 11.8、PyTorch 2.1.2+cu118、MMSeg 0.30.0 等。

### 模型配置
见 20260506_175707.log:41 到 20260506_175707.log:82
可提取：

框架：EncoderDecoder
Backbone：SegMANEncoder_t（加载预训练）
Head：SegMANDecoder，in_channels=[32,64,144,192]，channels=128
类别数：2（background/wound）
损失组合：BCE(0.5)+Dice(1.0)+Focal(0.5)
边界分支开启，boundary_loss_weight=0.5，kernel_size=5
推理模式：slide，crop_size=(1024,1024)，stride=(768,768)

### 数据配置与增强
见 20260506_175707.log:85 到 20260506_175707.log:212
可提取：

数据集类型：CustomDataset
数据根目录：wound/foot
类别与调色板
Train pipeline：Resize(0.5-2.0), RandomCrop(512), 双向Flip, PhotoMetricDistortion, Normalize, Pad
Test pipeline：MultiScaleFlipAug（此处 flip=False）
batch 与 loader：samples_per_gpu=4，workers_per_gpu=2

### 训练策略与超参数
见 20260506_175707.log:222 到 20260506_175707.log:244
可提取：

Optimizer：AdamW，lr=1e-4，weight_decay=0.01
Paramwise 策略：head lr_mult=10，norm/pos_block decay_mult=0
LR policy：poly + linear warmup(1500)
训练长度：IterBasedRunner，max_iters=160000
Eval 间隔：8000 iter，指标 mIoU/mDice/mFscore
随机种子：1015420629（非 deterministic）

### 数据量与训练状态
见 20260506_175707.log:522, 20260506_175707.log:523, 20260506_175707.log:525
可提取：

训练集加载 810 张，验证加载 200 张
本次是续训：从 epoch 199, iter 119999 恢复
资源占用与训练速度
见 20260506_175707.log:577
可提取：显存约 19180 MB，单 iter 时间与 data_time（可用于效率讨论）。



## 模型总参数量（Params）
```bash
python segmentation/tools/count_params.py --config segmentation/local_configs/segman/tiny/segman_t_wound_bce_dice_focal_complex_aug_boundary.py --checkpoint segmentation/work_dirs/wound_tiny_ablation/segman_t_wound_bce_dice_focal_complex_aug_boundary/iter_120000.pth --device cpu
```
推理 FPS 和 FLOPs 没有直接打印。

========================================================================================
Parameter Summary
========================================================================================
Total params     : 6,427,128 (6.427M)
Trainable params : 6,427,128 (6.427M)
Frozen params    : 0 (0.000M)

By top-level module
----------------------------------------------------------------------------------------
Module                               Total         Trainable    Trainable%
----------------------------------------------------------------------------------------
decode_head                      4,145,794         4,145,794       100.00%
backbone                         2,281,334         2,281,334       100.00%

Top parameter tensors
----------------------------------------------------------------------------------------
Name                                                                  Params   Trainable
----------------------------------------------------------------------------------------
decode_head.conv_downsample_4.conv.weight                          1,638,400         yes
decode_head.vssm.mlp.fc1.weight                                      589,824         yes
decode_head.vssm.mlp.fc2.weight                                      589,824         yes
decode_head.conv_downsample_2.conv.weight                            294,912         yes
decode_head.reduce_channels.2.conv.weight                            262,144         yes
backbone.layers.5.0.weight                                           248,832         yes
decode_head.cat.conv.weight                                          155,648         yes
decode_head.reduce_channels.1.conv.weight                            131,072         yes
backbone.layers.6.blocks.0.token_mixer.qkv.weight                    110,592         yes
backbone.layers.6.blocks.0.mlp.fc1.weight                            110,592         yes
backbone.layers.6.blocks.0.mlp.fc2.weight                            110,592         yes
backbone.layers.6.blocks.1.token_mixer.qkv.weight                    110,592         yes
backbone.layers.6.blocks.1.mlp.fc1.weight                            110,592         yes
backbone.layers.6.blocks.1.mlp.fc2.weight                            110,592         yes
decode_head.linear_fuse.conv.weight                                   98,304         yes
backbone.layers.3.0.weight                                            82,944         yes
decode_head.proj_out.conv.weight                                      73,728         yes
decode_head.reduce_channels.0.conv.weight                             65,536         yes
backbone.layers.4.blocks.0.token_mixer.qkv.weight                     62,208         yes
backbone.layers.4.blocks.0.mlp.fc1.weight                             62,208         yes
========================================================================================