# 随附运行时组件

本目录包含 LeRobot 数据写入与在线推理所需的固定版本组件。

| 目录 | 用途 |
| --- | --- |
| `lerobot/` | 数据集读写与视频处理（`0.3.4+orca.1`，含 `save_episode_data_only`） |
| `openpi-client/` | 与 OpenPI 策略服务通信的客户端（`0.1.0+orca.1`） |

安装方式：

```bash
python -m pip install --no-deps --no-build-isolation ./third_party/lerobot
python -m pip install --no-deps --no-build-isolation ./third_party/openpi-client
```

策略服务、模型和 checkpoint 不包含在本仓库中。
