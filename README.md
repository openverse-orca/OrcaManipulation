# OrcaManipulation — G1 OmniPicker 数据采集

基于 OrcaLab/OrcaGym 的 G1 OmniPicker 数据采集工具包，支持 **四色按按钮** 与 **工具整理** 两个场景，覆盖 Pico VR 遥操作采集、脚本化自动采集、LeRobot 数据回放和 OpenPI 策略在线推理。

## 兼容环境

| 组件 | 版本 / 要求 |
|------|-------------|
| OrcaLab | **6.3** |
| Python | **3.12 及以上** |
| `orca-gym` | 推荐使用与 OrcaLab 6.3 配套的 **26.6.x** |
| `lerobot` | **0.3.x**，数据集格式为 LeRobot v2.1 |
| GPU 视频编码 | NVIDIA GPU；驱动及 PyAV/FFmpeg 需要支持 `av1_nvenc` |

> 依赖见 `requirements.txt`。在线推理还需 `openpi_client`（或通过 `OPENPI_CLIENT_SRC` 指定源码路径）。

## 目录结构

```text
OrcaManipulation/
├── README.md
├── USAGE.md                          # 按键映射与运行命令参考
├── requirements.txt
├── pyproject.toml
└── src/
    ├── conf/
    │   └── g1_omnipicker_conf.py     # G1 机器人配置（关节/夹爪/相机）
    ├── controllers/                  # OSC 臂控制器 / 夹爪控制器 / 任务状态
    ├── dataCollectionManager/        # 数据采集主控
    ├── dataStorage/                  # LeRobot v2.1 存储（Parquet + MP4）
    ├── devices/                      # 输入设备抽象
    ├── envs/dataCollection/          # MuJoCo 环境封装
    ├── scene/                        # 场景管理
    ├── task/                         # 任务基类
    └── examples/dataCollection/      # 数据采集、推理和回放入口
```

## 入口脚本

入口脚本位于 `src/examples/dataCollection/`：

| 脚本 | 用途 |
|------|------|
| `g1_omnipicker_collection_scripted_tool_lerobot.py` | 工具整理脚本化自动采集 |
| `g1_omnipicker_collection_scripted_button_lerobot.py` | 四色按钮脚本化自动采集 |
| `g1_omnipicker_collection_tele_lerobot.py` | Pico VR 遥操作采集（左臂锁定） |
| `g1_omnipicker_replay_lerobot.py` | LeRobot Parquet 数据集回放 |
| `eval_g1_omnipicker_lerobot.py` | 连接 OpenPI 策略服务器进行在线推理 |
| `data_collection_scripted.py` | 脚本轨迹插值工具库 |

## 环境安装

```bash
conda activate orcalab_lerobot
pip install -r requirements.txt
```

推理脚本如需 `openpi_client`：

```bash
export OPENPI_CLIENT_SRC=~/openpi/packages/openpi-client/src
```

## 场景加载与相机配置

### 1. 加载场景

1. 启动 OrcaLab 6.3。
2. 按任务加载对应布局：
   - **工具整理**：`src/examples/dataCollection/g1_tool.json`
   - **四色按按钮**：`src/examples/dataCollection/g1_button.json`
3. 确认 G1 OmniPicker 与场景物体已正确加载。
4. 确认场景名称与 `example.yaml` 中的 `level_name: "example"` 一致。

### 2. 手动配置三路相机

打开场景布局后，必须在 OrcaLab 中重新配置相机（两个场景相同）：

| 相机 | 代码中的相机名称 | Color Port |
|------|------------------|------------|
| Left Arm | `camera_wrist_l_color` | **7070** |
| Right Arm | `camera_wrist_r_color` | **7080** |
| Head | `camera_head_color` | **7090** |

对每一路相机执行以下配置：

1. 勾选 **UseNvEnc**。
2. 勾选 **Color Camera**。
3. 将 **Color Port** 修改为上表中的对应端口。
4. 完成三路相机的端口及其他配置后，再勾选 **Recording**。

> [!CAUTION]
> - 必须先修改完三路相机的全部配置，最后再勾选 **Recording**。
> - 勾选 **Recording** 后不要再取消勾选，取消时容易导致 OrcaLab 崩溃。
> - 相机配置完成后，**不要保存到场景布局 `.json` 文件中**。
> - 每次重新打开布局文件，都需要再次手动配置 UseNvEnc、Color Camera、Color Port 和 Recording。

### 3. 启动仿真

完成相机配置后，点击 OrcaLab 的运行按钮启动仿真，并等待 OrcaGym gRPC 服务就绪。默认地址为：

```text
localhost:50051
```

数据采集和带图像的推理需要三路相机端口正常监听。数据回放本身不依赖 WebSocket 相机流。

## 运行准备

以下命令默认从入口脚本目录执行：

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection
```

---

## 运行方式一：工具整理脚本化采集

先在 OrcaLab 加载 `g1_tool.json`，再运行：

```bash
python g1_omnipicker_collection_scripted_tool_lerobot.py \
    --lerobot_out /path/to/out_dataset \
    --num_episodes 20
```

断点续采时追加 `--resume`。

---

## 运行方式二：四色按钮脚本化采集

先在 OrcaLab 加载 `g1_button.json`，再运行：

```bash
python g1_omnipicker_collection_scripted_button_lerobot.py \
    --task_config example.yaml \
    --lerobot_out ~/datasets/g1_button_scripted \
    --repo_id local/g1_omnipicker_button \
    --counts 25,25,25,25 \
    --fps 20 \
    --clock wall
```

说明：

- `--counts` 顺序为红、绿、黄、蓝，例如 `25,25,25,25` 表示每种颜色采集 25 个 episode。
- 交互式终端会询问各颜色采集数量；非交互环境使用 `--counts`。
- 默认候选位姿文件为 `pose_g1_button_candidates.yaml`。
- 在已有数据集后继续采集时增加 `--resume`。

---

## 运行方式三：Pico VR 遥操作采集

适用于已加载的场景（按按钮或工具整理均可，按实际任务填写 `--task`）。

采集前新开终端，将主机 `8001` 端口反向映射到 Pico：

```bash
adb reverse tcp:8001 tcp:8001
```

然后启动：

```bash
python g1_omnipicker_collection_tele_lerobot.py \
    --task_config example.yaml \
    --lerobot_out ~/datasets/g1_tele \
    --repo_id local/g1_omnipicker \
    --task "整理工具" \
    --fps 20 \
    --clock wall \
    --camera_source websocket
```

常用操作：

- 同时按下左右摇杆，触发手柄连接
- 左 Grip 单击：开始当前 episode；再次单击：结束并保存
- 右 Grip 单击：放弃当前 episode 并重置
- 左右 Grip 同时按下：结束全部采集
- 左臂锁定，右手柄 6DOF 控制右臂末端
- 续采时增加 `--resume`

更详细的按键映射见 [`USAGE.md`](USAGE.md)。

---

## 数据回放

```bash
python g1_omnipicker_replay_lerobot.py \
    --dataset_dir /path/to/lerobot_dataset \
    --task_config example.yaml \
    --episode 1 \
    --steps_per_frame 3 \
    --render_every 5
```

- `--episode` 从 **1** 开始；省略则顺序播完全部
- `--steps_per_frame` 越小回放越快
- `--loop` 可循环回放（`Ctrl+C` 退出）
- 回放前应加载与采集时一致的场景布局和 `example.yaml`

---

## 在线推理

先在策略侧启动服务器：

```bash
python serve_policy.py \
    --port 8010 \
    --checkpoint /path/to/checkpoint
```

再在本项目环境中运行：

```bash
export OPENPI_CLIENT_SRC=~/openpi/packages/openpi-client/src

python eval_g1_omnipicker_lerobot.py \
    --task_config example.yaml \
    --host localhost \
    --port 8010 \
    --prompt "整理工具" \
    --max_steps 500 \
    --action_repeat 1 \
    --episodes 3
```

`--prompt` 须与训练数据中的任务描述一致。

---

## 示例数据与 PI0.5 训练参考

### 示例 VR 自采数据

- 链接：<https://pan.baidu.com/s/1Q0Zoakl4eUYLNwWpzjqajw>
- 提取码：`5hne`

### PI0.5 LoRA 训练参考（四色按钮示例）

请参考 OpenPI 官方 PI0.5 流程。下列配置可作为 G1 OmniPicker 四色按钮任务的参考：

```python
TrainConfig(
    name="pi05_g1_omnipicker_button_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
        discrete_state_input=True,
    ),
    data=LeRobotG1OmnipickerDataConfig(
        repo_id="hangzhou2026/g1_omnipicker_button",
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=32,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2e-4,
        decay_steps=10_000,
        decay_lr=2e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,
    weight_loader=weight_loaders.CheckpointWeightLoader(
        ".cache/openpi/openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=10_000,
    log_interval=100,
    save_interval=5_000,
    keep_period=5_000,
    num_workers=2,
)
```

训练前请确认：已下载 `pi05_base` 权重、已注册 `LeRobotG1OmnipickerDataConfig`、`repo_id` 与数据一致、`prompt_from_task` 与推理 `--prompt` 对齐、`action_dim` / `action_horizon` 与数据一致。

---

## 配置文件

| 文件 | 说明 |
|------|------|
| `example.yaml` | 采集、回放和推理共用任务配置，`level_name: "example"` |
| `g1_tool.json` | OrcaLab 工具整理场景布局 |
| `g1_button.json` | OrcaLab 四色按钮场景布局 |
| `my_waypoint_tool1.yaml` … `my_waypoint_tool5.yaml` | 工具整理采集路点 |
| `my_slot_waypoints.yaml` | 工具整理采集辅助路点 |
| `pose_g1_button_candidates.yaml` | 四色按钮脚本化采集候选位姿 |

## 数据集格式

数据保存为 LeRobot v2.1：

```text
<dataset_root>/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
├── data/chunk-000/
│   └── episode_XXXXXX.parquet
└── videos/chunk-000/
    ├── observation.images.cam_head/
    ├── observation.images.cam_wrist_l/
    └── observation.images.cam_wrist_r/
```

默认相机 480×640，默认 20 FPS，视频为 MP4（`av1_nvenc`）。

`action` 与 `observation.state` 均为 **18 维**：

| 维度 | 含义 |
|------|------|
| `[0:3]` | 左臂末端位置 `l_pos`（base 系，米） |
| `[3:7]` | 左臂末端四元数 `l_quat_xyzw` |
| `[7:10]` | 右臂末端位置 `r_pos` |
| `[10:14]` | 右臂末端四元数 `r_quat_xyzw` |
| `[14]` | 左夹爪内侧归一化值 |
| `[15]` | 左夹爪外侧归一化值 |
| `[16]` | 右夹爪内侧归一化值 |
| `[17]` | 右夹爪外侧归一化值 |

夹爪归一化：`norm = (motor_val + 1) / 3`（电机量程 `[-1, 2]`）。  
base frame：`g1_omnipicker_body_link1`。训练采用 `action[t] = state[t+1]`。

## 启动前检查

- OrcaLab 6.3，且 `orca-gym` 版本匹配
- 已加载对应场景布局（`g1_tool.json` 或 `g1_button.json`）
- `example.yaml` 中 `level_name` 与场景名称一致
- OrcaGym gRPC 在 `localhost:50051` 就绪
- 三路相机端口 7070 / 7080 / 7090，UseNvEnc、Color Camera、Recording 已按要求启用
- NVIDIA 驱动与 PyAV/FFmpeg 支持 `av1_nvenc`
- 输出目录可写且磁盘空间充足

更详细的 Pico 按键映射见 [`USAGE.md`](USAGE.md)。
