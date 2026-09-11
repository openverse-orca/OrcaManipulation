# 随附运行时组件

本目录包含 LeRobot 数据写入与在线推理所需的固定版本组件，由 `docs/lerobot/install_runtime.sh` 安装。不要与根目录 `requirements.txt`（HDF5 旧入口）混装。说明见 [docs/lerobot/](../docs/lerobot/README.md)。

| 目录 | 用途 |
| --- | --- |
| `lerobot/` | 数据集读写与视频处理（`0.3.4+orca.1`，含 `save_episode_data_only`） |
| `openpi-client/` | 与 OpenPI 策略服务通信的客户端（`0.1.0+orca.1`） |

```bash
conda env create -f docs/lerobot/environment.yml
conda activate orcalab_lerobot
bash docs/lerobot/install_runtime.sh
```

策略服务、模型和 checkpoint 不包含在本仓库中。
