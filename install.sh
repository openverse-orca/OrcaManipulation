#!/usr/bin/env bash
# OrcaManipulation v5 环境安装（对应 docs/快速开始.md 第 0 步）。
# 在仓库根目录 OrcaManipulationv5/ 下运行：bash install.sh
set -e

ENV_NAME=orcalab_lerobot

conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"

pip install orca-lab
pip install -r requirements.txt
pip install "orca-gym[all]"
pip install numpy==2.2.6
pip install lerobot

# 推理才需要：安装 openpi 客户端
pip install -e third_party/openpi/packages/openpi-client

echo "✓ 安装完成。验证: conda activate $ENV_NAME && python -c 'import orca_gym, lerobot; print(\"OK\")'"
