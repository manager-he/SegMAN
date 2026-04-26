
## create virtual environment

python=3.10

```bash
python --version
python -m venv .venv
source .venv/bin/activate
```

## torch

```
nvidia-smi: 显卡驱动支持的最高CUDA
nvcc -v: 当前系统环境变量正在使用的CUDA toolkit
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu # cpu only
# ROCM 5.6 (Linux only)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/rocm5.6
# CUDA 11.8
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```


## MMSeg

```bash
pip install -U openmim
mim install mmcv-full
cd segmentation
pip install -v -e .
```

numpy degrade
```bash
pip install "numpy<2.0"
```

modify mmcv to support torch>2.1.0
replace 75 line of site-packages/mmcv/parallel/_functions.py
```python
if version.parse(torch.__version__) >= version.parse('2.1.0'):
    streams = [_get_stream(torch.device("cuda", device)) for device in target_gpus]
else:
    streams = [_get_stream(device) for device in target_gpus]
```

## Natten
从该网址下载对应的whl文件:https://shi-labs.com/natten/wheels/
```bash
cd asserts
wget http://shi-labs.com/natten/wheels/cu121/torch2.2.0/natten-0.17.3%2Btorch220cu121-cp310-cp310-linux_x86_64.whl
pip install natten-0.17.3+torch230cpu-cp310-cp310-linux_x86_64.whl
cd ..
```

selective Scan 2D: CUDA加速内核,虚拟机安装不了
```bash
cd kernels/selective_scan && pip install .
pip install -r requirements.txt
```


# Data Config
```bash
mkdir segmentation/data
```

data的结构需要和segmentation/local_configs/_base_/dataset中的配置文件一致

data_root: data全局地址
data.img_dir/ann_dir: 内部结构

按照当前配置，数据集应该是这个结构
segmentation/data/wound/foot
    - annotations
        - training
        - validation
    - images
        - training
        - validation

segmentation/outputs
    - b_wound_full
        - iter_104000.pth

This is not the structure when clone from FUSegData
You should upload the zip file from ML/Medical_segment/wound_data.zip to segmentation/

# Pretrain

Download the ImageNet-1k pretrained weights here and put them in a folder pretrained/
https://drive.google.com/drive/folders/1QYU7nhpe0ddH7bPxI7VH4drc__07uEHs?usp=sharing