#!/bin/bash
# ============================================================
# Gaussian Shading 环境自动配置脚本
# 适用系统: Ubuntu / CentOS / macOS (需要 NVIDIA GPU 支持 CUDA)
# ============================================================

set -e

echo "=========================================="
echo " Gaussian Shading 环境配置脚本"
echo "=========================================="

# ---- 颜色定义 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ---- 函数: 打印信息 ----
info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================
# 第一步: 检查并安装 Anaconda/Miniconda
# ============================================================
echo ""
info "第一步: 检查 Conda 安装状态..."

if command -v conda &> /dev/null; then
    info "Conda 已安装: $(conda --version)"
else
    warn "未检测到 Conda，开始安装 Miniconda..."

    # 检测系统架构
    ARCH=$(uname -m)
    OS=$(uname -s)

    if [ "$OS" = "Linux" ]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh"
    elif [ "$OS" = "Darwin" ]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-${ARCH}.sh"
    else
        error "不支持的操作系统: $OS"
        exit 1
    fi

    info "下载 Miniconda: $MINICONDA_URL"
    wget -q "$MINICONDA_URL" -O /tmp/miniconda.sh

    info "安装 Miniconda..."
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh

    # 初始化 conda
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash

    info "Miniconda 安装完成!"
    warn "请运行 'source ~/.bashrc' 或重新打开终端后再次运行此脚本"
fi

# ============================================================
# 第二步: 创建 Conda 虚拟环境
# ============================================================
echo ""
info "第二步: 创建 Conda 虚拟环境 (Python 3.8)..."

ENV_NAME="gs"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    warn "环境 '${ENV_NAME}' 已存在"
    read -p "是否删除并重新创建? (y/N): " choice
    if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
        conda env remove -n "$ENV_NAME" -y
        info "已删除旧环境"
    else
        info "使用已有环境"
    fi
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
    info "创建新的 Conda 环境: ${ENV_NAME} (Python 3.8)..."
    conda create -n "$ENV_NAME" python=3.8 -y
    info "环境创建完成!"
fi

# ============================================================
# 第三步: 激活环境并安装依赖
# ============================================================
echo ""
info "第三步: 激活环境并安装依赖..."

# 激活环境
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

info "当前 Python 版本: $(python --version)"
info "当前 pip 版本: $(pip --version)"

# ============================================================
# 第四步: 安装 PyTorch (根据 CUDA 版本)
# ============================================================
echo ""
info "第四步: 安装 PyTorch..."

# 检测 CUDA 可用性
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    info "检测到 CUDA 版本: $CUDA_VERSION"

    # 根据 CUDA 版本选择合适的 PyTorch
    # torch 1.13.0 支持 CUDA 11.6 和 11.7
    info "安装 PyTorch 1.13.0 (CUDA 11.7)..."
    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117
else
    warn "未检测到 NVIDIA GPU，安装 CPU 版本的 PyTorch"
    warn "注意: CPU 模式下运行将非常缓慢，建议使用 GPU"
    pip install torch==1.13.0+cpu torchvision==0.14.0+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu
fi

# ============================================================
# 第五步: 安装核心依赖
# ============================================================
echo ""
info "第五步: 安装核心依赖包..."

# Diffusion 模型相关
info "安装 Diffusers 和 Transformers..."
pip install diffusers==0.11.1 transformers==4.34.0

# HuggingFace 相关
info "安装 HuggingFace 组件..."
pip install huggingface_hub==0.22.2 datasets==2.18.0

# 图像处理
info "安装图像处理库..."
pip install Pillow==10.3.0 albumentations==1.4.3 scikit-image kornia==0.6.4

# 数学和科学计算
info "安装科学计算库..."
pip install numpy==1.24.4 scipy==1.10.1

# 加密库
info "安装加密库..."
pip install pycryptodome==3.20.0

# 其他依赖
info "安装其他依赖..."
pip install \
    pytorch_lightning==2.2.1 \
    timm==0.5.4 \
    einops==0.4.1 \
    omegaconf==2.3.0 \
    PyYAML==6.0.1 \
    regex==2023.12.25 \
    ftfy==6.2.0 \
    tqdm==4.66.2 \
    Requests==2.31.0 \
    more_itertools==10.2.0 \
    natsort==8.4.0 \
    typing_extensions==4.11.0 \
    matplotlib==3.7.5 \
    academictorrents==2.3.3

# 安装 open_clip (项目自带了 open_clip 模块，但某些依赖可能需要)
info "安装 open_clip_torch..."
pip install open_clip_torch==2.20.0

# ============================================================
# 第六步: 验证安装
# ============================================================
echo ""
info "第六步: 验证安装..."

python -c "
import sys
print(f'Python 版本: {sys.version}')

import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 设备: {torch.cuda.get_device_name(0)}')

import diffusers
print(f'Diffusers 版本: {diffusers.__version__}')

import transformers
print(f'Transformers 版本: {transformers.__version__}')

from PIL import Image
print(f'Pillow 版本: {Image.__version__}')

from Crypto.Cipher import ChaCha20
print('PyCryptoDome (ChaCha20): OK')

import numpy as np
print(f'NumPy 版本: {np.__version__}')

import scipy
print(f'SciPy 版本: {scipy.__version__}')

print()
print('========================================')
print(' 所有核心依赖验证通过!')
print('========================================')
"

# ============================================================
# 完成
# ============================================================
echo ""
echo "=========================================="
info "环境配置完成!"
echo "=========================================="
echo ""
info "使用以下命令激活环境:"
echo "  conda activate ${ENV_NAME}"
echo ""
info "运行示例 (无失真测试):"
echo "  python run_gaussian_shading.py --fpr 0.000001 --channel_copy 1 --hw_copy 8 --chacha --num 1000"
echo ""
info "更多测试用例请参考: scripts/run.sh"
echo ""
