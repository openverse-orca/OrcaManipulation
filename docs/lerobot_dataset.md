# LeRobot 数据集约定

本仓库用 `LeRobotDatasetWriter` 把仿真采集写成 [LeRobot v2.1](https://github.com/huggingface/lerobot) 数据集。训练与推理共用同一套 `PolicySchema` 定义的 state / action。

## 数据链路

```
观测 obs_callback
    → PolicySchema.build_state / build_action
    → LeRobotSimSyncMixin 缓冲帧
    → LeRobotDatasetWriter.flush_episode
    → HF_LEROBOT_HOME/<repo_id>/{data,videos,meta}
```

写入层在 `src/dataStorage/lerobot_writer.py`。对上游 LeRobot 的 `_get_image_file_path` / `_save_image` 替换集中在 `patch_lerobot_dataset_for_nvenc()`，不要在业务脚本里再替换。

`save_episode_data_only()` 是随仓携带的 `third_party/lerobot`（`0.3.4+orca.1`）新增 API，上游官方包没有。

## 目录与元数据

| 路径 | 内容 |
|------|------|
| `data/chunk-*/episode_*.parquet` | 本体状态与动作 |
| `videos/` | 按相机键切分的 MP4 |
| `meta/info.json` | fps、features、机器人类型 |
| `meta/tasks.jsonl` | 每集 `task` 文本（与推理 `--prompt` 对齐） |

`task` 是语言描述，不是场景文件名。同一数据集内不同 episode 可以有不同 prompt，但推理时必须与训练时写入的文本一致。

## Schema 职责

`PolicySchema`（`src/policy/schema.py`）同时定义：

- `state_dim` / `state_names` / `build_state(obs)`
- `parse_action(raw)`：把策略向量还原成控制器可用的位姿与夹爪绝对值

正逆变换必须成对写在同一个 Schema 里。Storage 只负责观测采集与落盘，不再自己实现 `build_state`。

已有实现：

- `policy/dual_arm_schema.py`：G1 OmniPicker 与 Unitree G1 Pick OSC，18 维
- `policy/openloong_schema.py`、`policy/tiangong2_schema.py`

## 安装

```bash
pip install --no-deps --no-build-isolation ./third_party/lerobot
```

相机端口与环境相机名属于机器人/部署配置，见各 `conf/*.py` 的 `camera_map()`，以及 [cameras_and_video.md](cameras_and_video.md)。
