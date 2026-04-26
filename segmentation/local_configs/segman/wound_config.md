
# segman_base + wound_cfg示例：segman_b_wound.py

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

# fuseg_wound_data_aug

| 修改数据增强策略后的数据层配置文件

这份配置相对原来的 wound 数据集配置，主要强化了更贴近伤口场景的域扰动和形态扰动：

* 缩放范围从原来的更宽泛随机缩放，收窄为 0.7 到 1.5，减少过激变形。
* 裁剪约束更强，cat_max_ratio 提到 0.85，尽量避免裁到几乎全背景。
* 增加了纵向翻转，适配拍摄方向不固定的情况。
* 增加了随机旋转，模拟手机或临床拍摄角度变化。
* 光照增强参数做了收敛，偏向真实设备成像波动而不是强烈颜色扰动。
* 增加了 CLAHE ，用来增强局部对比度，适合边界和组织纹理不明显的伤口图像。
* 增加了 RandomCutOut ，模拟局部遮挡、反光、纱布边缘或成像不完整。

# tiny/segman_t_wound_...

| tiny模型层的消融对比实验

对比数据增强方案和损失函数优化方案的效果

数据增强方案见上
函数优化方案比对BCE和BCE+DICE+FOCAL平权


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


# 数据增强与预处理支持的功能 Pipeline

几何增强
* Resize: transforms.py:297
* AlignedResize: transforms.py:12
* ResizeToMultiple: transforms.py:240
* RandomCrop: transforms.py:811
* RandomFlip: transforms.py:552
* RandomRotate: transforms.py:884
* RandomMosaic: transforms.py:1299

颜色与强度类增强
* PhotoMetricDistortion: transforms.py:1088
* CLAHE: transforms.py:766
* AdjustGamma: transforms.py:1022
* RGB2Gray: transforms.py:967
* Rerange: transforms.py:720

遮挡与标签相关
* RandomCutOut: transforms.py:1207
* SegRescale: transforms.py:1058
* Pad: transforms.py:607

常规读取与测试时增强
* LoadImageFromFile: loading.py:11
* LoadAnnotations: loading.py:91
* MultiScaleFlipAug: test_time_aug.py:11
* Normalize 归一化: transforms.py:678
