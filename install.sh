#!/usr/bin/env bash
# OrcaManipulation v5 环境安装（对应 docs/快速开始.md 第 0 步）。
# 在仓库根目录 OrcaManipulationv5/ 下运行：bash install.sh
set -e

ENV_NAME=orcalab_lerobot

# 确保 conda activate 可用（对未初始化 conda 的 shell 友好）
if ! command -v conda &> /dev/null; then
    echo "Error: conda 未找到，请先安装 Anaconda/Miniconda。"
    exit 1
fi

# 尝试加载 conda.sh，解决 "Run 'conda init' before 'conda activate'" 错误
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"

pip install orca-lab
pip install -r requirements.txt
pip install "orca-gym[all]"
pip install numpy==2.2.6


# 推理才需要：安装 openpi 客户端
pip install -e third_party/openpi/packages/openpi-client
pip install lerobot

echo "✓ 安装完成。验证: conda activate $ENV_NAME && python -c 'import orca_gym, lerobot; print(\"OK\")'"
