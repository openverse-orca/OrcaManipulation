#!/usr/bin/env bash
# 一键安装运行依赖。
# lerobot 官方包强依赖 wandb/diffusers 等训练组件，数采用不到，
# 且很多镜像源缺 wandb 会导致解析失败，因此用 --no-deps 单独安装。
set -euo pipefail

cd "$(dirname "$0")"

python -m pip install -r requirements.txt
python -m pip install "lerobot==0.3.3" --no-deps

python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('LeRobotDataset OK')"
