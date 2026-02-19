# Gaussian Shading 环境配置完整指南

本指南将从零开始，详细介绍如何使用 Anaconda 配置 Gaussian Shading 项目的完整运行环境。

---

## 目录

1. [系统要求](#1-系统要求)
2. [安装 Anaconda / Miniconda](#2-安装-anaconda--miniconda)
3. [创建 Conda 虚拟环境](#3-创建-conda-虚拟环境)
4. [安装 PyTorch (GPU/CPU)](#4-安装-pytorch)
5. [安装项目依赖](#5-安装项目依赖)
6. [验证安装](#6-验证安装)
7. [下载预训练模型](#7-下载预训练模型)
8. [运行项目](#8-运行项目)
9. [常见问题排查](#9-常见问题排查)
10. [一键安装](#10-一键安装)

---

## 1. 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Ubuntu 18.04+ / CentOS 7+ / macOS | Ubuntu 20.04+ |
| Python | 3.8 | 3.8 |
| GPU | NVIDIA GPU (CUDA 11.6+) | NVIDIA A100 / RTX 3090+ |
| 显存 | 8GB+ | 16GB+ |
| 内存 | 16GB+ | 32GB+ |
| 磁盘空间 | 20GB+ (含模型权重) | 50GB+ |

> **注意**: 项目支持 CPU 模式运行，但速度会非常慢，建议使用 NVIDIA GPU。

---

## 2. 安装 Anaconda / Miniconda

### 方案 A: 安装 Miniconda（推荐，体积小）

```bash
# 下载 Miniconda 安装脚本 (Linux x86_64)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# 运行安装
bash miniconda.sh -b -p $HOME/miniconda3

# 初始化 conda (写入 shell 配置)
$HOME/miniconda3/bin/conda init bash

# 使配置生效
source ~/.bashrc

# 验证安装
conda --version
```

### 方案 B: 安装完整版 Anaconda

```bash
# 下载 Anaconda 安装脚本 (Linux x86_64)
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O anaconda.sh

# 运行安装
bash anaconda.sh -b -p $HOME/anaconda3

# 初始化 conda
$HOME/anaconda3/bin/conda init bash

# 使配置生效
source ~/.bashrc

# 验证安装
conda --version
```

### macOS 用户

```bash
# Intel Mac
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh

# Apple Silicon (M1/M2/M3)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -O miniconda.sh

bash miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init zsh  # macOS 默认用 zsh
source ~/.zshrc
```

---

## 3. 创建 Conda 虚拟环境

```bash
# 创建名为 "gs" 的环境，指定 Python 3.8
conda create -n gs python=3.8 -y

# 激活环境
conda activate gs

# 验证 Python 版本
python --version
# 应输出: Python 3.8.x
```

> **为什么用 Python 3.8?**
> 本项目的依赖版本（特别是 `torch==1.13.0` 和 `diffusers==0.11.1`）与 Python 3.8 兼容性最好。

---

## 4. 安装 PyTorch

PyTorch 的安装需要根据你的 CUDA 版本选择对应的安装命令。

### 检查 CUDA 版本

```bash
# 查看 NVIDIA 驱动和 CUDA 版本
nvidia-smi
```

### 方案 A: CUDA 11.7（推荐）

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

### 方案 B: CUDA 11.6

```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116
```

### 方案 C: CPU 版本（无 GPU）

```bash
pip install torch==1.13.0+cpu torchvision==0.14.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

### 验证 PyTorch 安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 5. 安装项目依赖

### 方案 A: 使用 requirements.txt（逐步安装）

```bash
# 确保已激活 gs 环境
conda activate gs

# 安装核心依赖（排除 torch 和 torchvision，因为已经安装了）
pip install diffusers==0.11.1
pip install transformers==4.34.0
pip install huggingface_hub==0.22.2
pip install datasets==2.18.0

# 安装图像处理库
pip install Pillow==10.3.0 albumentations==1.4.3 scikit-image kornia==0.6.4

# 安装科学计算库
pip install numpy==1.24.4 scipy==1.13.0

# 安装加密库 (ChaCha20 支持)
pip install pycryptodome==3.20.0

# 安装其他依赖
pip install pytorch_lightning==2.2.1 timm==0.5.4 einops==0.4.1
pip install omegaconf==2.3.0 PyYAML==6.0.1
pip install regex==2023.12.25 ftfy==6.2.0
pip install tqdm==4.66.2 Requests==2.31.0
pip install more_itertools==10.2.0 natsort==8.4.0 typing_extensions==4.11.0
pip install matplotlib==3.7.5 academictorrents==2.3.3

# 安装 OpenCLIP (用于 CLIP Score 计算)
pip install open_clip_torch==2.20.0
```

### 方案 B: 使用 environment.yml（一键安装）

```bash
# 从项目根目录执行
conda env create -f environment.yml

# 激活环境
conda activate gs
```

### 关于 requirements.txt 中几个特殊依赖的说明

| 依赖包 | 说明 |
|--------|------|
| `clip==0.2.0` | 项目已自带 `open_clip/` 目录，无需额外安装 CLIP 包 |
| `skimage==0.0` | 这是旧的包名，实际应安装 `scikit-image` |
| `horovod==0.28.1` | 分布式训练框架，单机运行时**不需要安装**，安装较复杂 |

---

## 6. 验证安装

运行以下 Python 脚本来验证所有依赖是否正确安装：

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')

import diffusers
print(f'Diffusers: {diffusers.__version__}')

import transformers
print(f'Transformers: {transformers.__version__}')

from PIL import Image
print(f'Pillow: {Image.__version__}')

from Crypto.Cipher import ChaCha20
print('PyCryptoDome (ChaCha20): OK')

import numpy, scipy, kornia
print(f'NumPy: {numpy.__version__}, SciPy: {scipy.__version__}, Kornia: {kornia.__version__}')

print('所有依赖验证通过!')
"
```

---

## 7. 下载预训练模型

项目默认使用 Stable Diffusion 2.1 base 模型，首次运行时会自动从 HuggingFace 下载。

### 自动下载（推荐）

直接运行代码，模型会自动下载到 `~/.cache/huggingface/` 目录：

```bash
# 首次运行会自动下载模型 (~5GB)
python run_gaussian_shading.py --num 1 --fpr 0.000001 --channel_copy 1 --hw_copy 8
```

### 手动下载（网络不佳时）

如果网络不好，可以手动下载模型：

```bash
# 安装 huggingface-cli
pip install huggingface_hub

# 下载 Stable Diffusion 2.1 base
huggingface-cli download stabilityai/stable-diffusion-2-1-base --local-dir ./stable-diffusion-2-1-base
```

然后运行时指定本地路径：

```bash
python run_gaussian_shading.py \
    --model_path ./stable-diffusion-2-1-base \
    --num 1 --fpr 0.000001 --channel_copy 1 --hw_copy 8
```

### 中国大陆用户加速下载

如果在中国大陆，建议使用 HuggingFace 镜像：

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 然后正常运行即可，模型会从镜像下载
python run_gaussian_shading.py --num 1 --fpr 0.000001 --channel_copy 1 --hw_copy 8
```

也可以写入 shell 配置使其永久生效：

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

---

## 8. 运行项目

### 8.1 测试 TPR（真阳性率）和比特准确率

```bash
# 无失真情况下的测试
python run_gaussian_shading.py \
    --fpr 0.000001 \
    --channel_copy 1 \
    --hw_copy 8 \
    --chacha \
    --num 1000
```

### 8.2 噪声扰动下的鲁棒性测试

```bash
# JPEG 压缩 (QF=25)
python run_gaussian_shading.py --jpeg_ratio 25

# 随机裁剪 (60%)
python run_gaussian_shading.py --random_crop_ratio 0.6

# 高斯模糊
python run_gaussian_shading.py --gaussian_blur_r 4

# 高斯噪声
python run_gaussian_shading.py --gaussian_std 0.05

# 中值滤波
python run_gaussian_shading.py --median_blur_k 7

# 椒盐噪声
python run_gaussian_shading.py --sp_prob 0.05

# 缩放
python run_gaussian_shading.py --resize_ratio 0.25

# 亮度调整
python run_gaussian_shading.py --brightness_factor 6

# 随机丢弃
python run_gaussian_shading.py --random_drop_ratio 0.8
```

### 8.3 计算 CLIP Score

```bash
python run_gaussian_shading.py \
    --fpr 0.000001 \
    --channel_copy 1 \
    --hw_copy 8 \
    --chacha \
    --num 1000 \
    --reference_model ViT-g-14 \
    --reference_model_pretrain laion2b_s12b_b42k
```

### 8.4 计算 FID

需要先下载 COCO 数据集的 ground truth 图片（5000 张），下载地址见 README。

```bash
python gaussian_shading_fid.py \
    --channel_copy 1 \
    --hw_copy 8 \
    --chacha \
    --num 5000
```

---

## 9. 常见问题排查

### Q1: `conda activate gs` 报错 "CommandNotFoundError"

```bash
# 需要先初始化 conda
conda init bash
source ~/.bashrc
```

### Q2: PyTorch 安装后 `torch.cuda.is_available()` 返回 False

可能的原因：
1. 没有安装 NVIDIA 驱动：运行 `nvidia-smi` 检查
2. 安装了 CPU 版本的 PyTorch：重新安装带 CUDA 的版本
3. CUDA 版本不匹配：确保驱动支持的 CUDA 版本 >= PyTorch 要求的版本

```bash
# 检查驱动
nvidia-smi

# 重新安装正确版本
pip uninstall torch torchvision -y
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

### Q3: `diffusers==0.11.1` 安装失败

```bash
# 尝试不指定版本先安装，再降级
pip install diffusers
pip install diffusers==0.11.1
```

### Q4: `pycryptodome` 安装失败

```bash
# Ubuntu/Debian 需要安装编译依赖
sudo apt-get install build-essential libssl-dev libffi-dev python3-dev

# 然后重新安装
pip install pycryptodome==3.20.0
```

### Q5: 内存不足 (OOM)

- 减少 `--num` 参数值
- 使用 `torch.float16` 精度（代码已默认使用）
- 减少 batch size

### Q6: HuggingFace 下载模型超时

```bash
# 方法 1: 使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方法 2: 使用代理
export https_proxy=http://your-proxy:port
export http_proxy=http://your-proxy:port
```

### Q7: `horovod` 安装失败

`horovod` 是分布式训练框架，单机运行时**不需要安装**。如果确实需要：

```bash
# 先安装系统依赖
sudo apt-get install -y cmake g++ openmpi-bin libopenmpi-dev

# 安装 horovod
HOROVOD_WITH_PYTORCH=1 pip install horovod==0.28.1
```

---

## 10. 一键安装

我们提供了自动配置脚本，可以一键完成所有步骤：

```bash
# 赋予执行权限
chmod +x setup_environment.sh

# 运行安装脚本
./setup_environment.sh
```

脚本会自动完成以下操作：
1. 检查并安装 Miniconda
2. 创建 Python 3.8 虚拟环境
3. 根据系统自动选择 GPU/CPU 版本的 PyTorch
4. 安装所有项目依赖
5. 验证安装结果

---

## 快速参考

```bash
# 激活环境
conda activate gs

# 退出环境
conda deactivate

# 查看已安装的包
conda list

# 查看环境列表
conda env list

# 删除环境（需要时）
conda env remove -n gs
```
