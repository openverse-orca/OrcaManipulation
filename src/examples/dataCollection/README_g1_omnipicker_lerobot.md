# G1 OmniPicker LeRobot 遥操作数据采集

VR 手柄（Pico）遥操作 G1 OmniPicker，数据以 [LeRobot v2.1](https://github.com/huggingface/lerobot) 格式（parquet + mp4）保存。视频编码使用 av1_nvenc GPU 流式编码，无 PNG 磁盘往返。

## 前置条件

1. `conda activate orcalab_lerobot`
2. OrcaStudio 仿真服务已启动（默认 `localhost:50051`）
3. 相机端口：头部=7090，左腕=7070，右腕=7080
4. Pico 通过 USB 连接并执行 `adb reverse tcp:8001 tcp:8001`
5. Fork 版 lerobot（`/home/dht/OrcaGym/lerobot`）中 `compute_stats.py` 的 `sample_images` 已支持 numpy 数组输入（NVENC 编码必需）

## 运行命令

```bash
cd src/examples/dataCollection

python g1_omnipicker_collection_tele_lerobot.py \
  --level default \
  --task_config example.yaml \
  --lerobot_out /path/to/output_dataset \
  --repo_id local/g1_omnipicker \
  --fps 20 \
  --clock wall
```

断点续采加 `--resume`；仅用右腕相机加 `--cameras wrist_r`。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lerobot_out` | *必填* | 数据集输出目录 |
| `--repo_id` | `local/g1_omnipicker` | LeRobot repo_id |
| `--level` | `default` | 场景名称 |
| `--task_config` | `example.yaml` | 场景配置 YAML |
| `--task` | `g1 omnipicker teleoperation` | 任务语言描述 |
| `--fps` | `20` | 采集帧率 |
| `--clock` | `wall` | 时钟源：`wall`（墙钟）或 `sim`（仿真时间） |
| `--cameras` | `head,wrist_l,wrist_r` | 启用的相机 |
| `--cam_resolution` | `480x640` | 分辨率 HxW |
| `--camera_source` | `websocket` | `websocket` 或 `mp4` |
| `--resume` | `false` | 追加到已有数据集 |
| `--orcagym_addr` | `localhost:50051` | OrcaGym 地址 |

## 按键操作

| 按键 | 功能 |
|------|------|
| 左臂 | **已锁定**，全程静止 |
| 右手柄位姿 | 右臂末端运动 |
| X / Y / 左扳机 | 左夹爪开合 |
| A / B / 右扳机 | 右夹爪开合 |
| 左 Grip（第1次） | 开始采集本集 |
| 左 Grip（第2次） | 保存本集（强制保存，无论成功与否） |
| 右 Grip | 丢弃本集，重置场景，继续采集 |
| 左右 Grip 同按 | 结束全部采集，等待编码完成后退出 |

开始采集前机械臂保持静止，不响应手柄；首次检测到手柄连接时屏幕显示提示。

## 数据格式

```
<lerobot_out>/
├── data/chunk-000/episode_XXXXXX.parquet   # state/action（18维）
├── videos/chunk-000/
│   ├── observation.images.cam_head/episode_XXXXXX.mp4
│   ├── observation.images.cam_wrist_l/episode_XXXXXX.mp4
│   └── observation.images.cam_wrist_r/episode_XXXXXX.mp4
└── meta/
    ├── info.json
    └── episodes.jsonl
```

State/action 向量（18维）：`l_pos(3) + l_quat_xyzw(4) + r_pos(3) + r_quat_xyzw(4) + l_grip_inner + l_grip_outer + r_grip_inner + r_grip_outer`

`action[i] = state[i+1]`（next-step 移位约定）。

## 示例数据集

`lerobot_datasets/g1_omni01/` 包含若干条采集示例，可直接用 LeRobotDataset 加载：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="local/g1_omni01",
                    root="src/examples/dataCollection/lerobot_datasets/g1_omni01",
                    download_videos=False)
print(ds.num_episodes, ds.num_frames)
```
