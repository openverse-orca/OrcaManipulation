# OrcaManipulation — G1 OmniPicker 按按钮场景

基于 OrcaLab/OrcaGym 仿真平台的 G1 OmniPicker 四色按钮操作工具包，覆盖 Pico VR 遥操作数据采集、脚本化自动采集、LeRobot 数据回放和 OpenPI 策略在线推理。

## 兼容环境

以下为本项目使用和验证的主要环境：

| 组件 | 版本 / 要求 |
|------|-------------|
| OrcaLab | **6.3** |
| Python | **3.12 及以上** |
| `orca-gym` | 推荐使用与 OrcaLab 6.3 配套的 **26.6.x** |
| `lerobot` | **0.3.x**，数据集格式为 LeRobot v2.1 |
| GPU 视频编码 | NVIDIA GPU；驱动及 PyAV/FFmpeg 需要支持 `av1_nvenc` |

> `requirements.txt` 提供基础依赖。运行环境还需要包含 `numpy`、`scipy`、`gymnasium`、`opencv-python`、`av`、`pyarrow`、`lerobot` 和 `mujoco`；在线推理还需要 `openpi_client`。

## 目录结构

```text
OrcaManipulation/
├── README.md
├── USAGE.md                          # 更详细的按键和参数说明
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
| `g1_omnipicker_collection_tele_lerobot.py` | Pico VR 遥操作采集，左臂锁定，控制右臂和双夹爪 |
| `g1_omnipicker_collection_scripted_button_lerobot.py` | 四色按钮脚本化自动采集 |
| `g1_omnipicker_collection_scripted_tool_lerobot.py` | 工具整理脚本化自动采集（右臂依次抓取 5 个工具放入工具箱） |
| `g1_omnipicker_replay_lerobot.py` | LeRobot Parquet 数据集回放 |
| `eval_g1_omnipicker_lerobot.py` | 连接 OpenPI 策略服务器进行在线推理 |
| `data_collection_scripted.py` | 脚本轨迹插值工具库 |

## 环境安装

```bash
conda activate orcalab_lerobot
pip install -r requirements.txt
```

确认当前环境还安装了项目实际使用的依赖：

```bash
pip install numpy scipy gymnasium opencv-python av pyarrow mujoco
```

LeRobot 和 OrcaGym 建议安装与验证环境一致的版本。推理脚本还需要安装 `openpi_client`，或通过环境变量指定其源码目录：

```bash
export OPENPI_CLIENT_SRC=~/openpi/packages/openpi-client/src
```

## 场景加载与相机配置

### 1. 加载场景

1. 启动 OrcaLab 6.3。
2. 加载 `src/examples/dataCollection/g1_button.json`。
3. 确认 G1 OmniPicker 和带四色按钮的电柜均已正确加载。
4. 确认场景名称与 `example.yaml` 中的 `level_name: "example"` 一致。

### 2. 手动配置三路相机

打开场景布局后，必须在 OrcaLab 中重新配置相机。三路相机的配置如下：

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

## 运行方式一：Pico VR 遥操作采集

在运行采集脚本前，新开一个终端，执行以下命令，将主机的 `8001` 端口反向映射到 Pico 设备：

```bash
adb reverse tcp:8001 tcp:8001
```

然后在项目终端中启动遥操作采集：

```bash
conda activate orcalab_lerobot
python g1_omnipicker_collection_tele_lerobot.py \
    --task_config example.yaml \
    --lerobot_out ~/datasets/g1_button_tele \
    --repo_id local/g1_omnipicker \
    --task "按红色按钮" \
    --fps 20 \
    --clock wall \
    --camera_source websocket
```

常用操作：
- 同时按下左右摇杆，触发手柄连接
- 左 Grip 单击：开始当前 episode；再次单击：结束并保存。
- 右 Grip 单击：放弃当前 episode 并重置。
- 左右 Grip 同时按下：结束全部采集。
- 左臂在本场景中保持锁定，右手柄 6DOF 控制右臂末端。
- 在已有数据集后继续采集时增加 `--resume`。
- 默认采集头部、左腕和右腕三路相机；可用 `--cameras head,wrist_l,wrist_r` 调整。

操作流程：
1.终端运行命令g1_omnipicker_collection_tele_lerobot.py 
2.修改prompt按回车键确认
3.同时按下左右摇杆连接手柄（可能需要多次尝试直到控制台输出已连接客户端）
4.按下左grip键开始录制
5.丢弃数据按右grip
6.左grip保存当前数据
7.下一条数据采集签需要执行步骤2
8.同时按下左右grip结束所有数据录制

## 运行方式二：脚本化自动采集

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

- `--counts` 顺序为红、绿、黄、蓝，例如 `25,25,25,25` 表示每种颜色采集 25 个 episode。可以在终端重新修改
- 在交互式终端中，程序启动后会询问各颜色采集数量；非交互环境使用 `--counts`。
- 默认候选位姿文件为 `pose_g1_button_candidates.yaml`。
- 可用 `--pose_candidates /path/to/candidates.yaml` 指定其他候选位姿。
- 使用 `--shuffle_seed` 可固定颜色序列的随机打乱结果。
- 在已有数据集后继续采集时增加 `--resume`。

## 运行方式三：工具整理脚本化采集

```bash
python g1_omnipicker_collection_scripted_tool_lerobot.py \
    --task_config example.yaml \
    --lerobot_out /path/to/out_dataset \
    --repo_id local/g1_omnipicker_tool \
    --num_episodes 1 \
    --fps 20 \
    --clock wall
```

说明：

- 右臂从左到右依次抓取 5 个工具放入工具箱，全程单条 episode，左臂全程锁定。
- 每个工具的路点由独立 YAML 文件指定，默认读取同目录 `my_waypoint_tool1.yaml` … `my_waypoint_tool5.yaml`。
- 可用 `--waypoint_files` 指定自定义路点文件（逗号分隔，顺序即抓取顺序）。
- `--safe_z` 控制高位过渡安全高度（base 系 z，单位米，默认 0.50）。
- `--kp` 调整 OSC 阻抗刚度（默认 220，范围 0～300），越大末端跟踪越紧。
- 路点文件可使用 `record_g1_waypoints.py` 遥操记录生成。
- 在已有数据集后继续采集时增加 `--resume`。

## 示例 VR 自采数据与 PI0.5 训练

### 示例 VR 自采数据

可通过百度网盘下载本项目的 VR 遥操作自采示例数据：

- 链接：<https://pan.baidu.com/s/1Q0Zoakl4eUYLNwWpzjqajw>
- 提取码：`5hne`

> 数据由百度网盘超级会员 v6 分享。

### PI0.5 LoRA 训练参考

请参考 OpenPI 官方的 PI0.5 训练流程准备数据、注册数据配置并启动训练。可使用以下配置作为 G1 OmniPicker 四色按钮任务的参考：

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

训练前请确认：

- 已按照 OpenPI 的 PI0.5 官方流程下载基础权重，且 `pi05_base/params` 路径与本机实际路径一致。
- 已在 OpenPI 中实现并注册 `LeRobotG1OmnipickerDataConfig`。
- `repo_id` 指向实际训练数据集；使用本地数据时，应按 OpenPI 数据配置方式修改对应路径或仓库标识。
- 数据集中的任务描述可被 `prompt_from_task=True` 正确读取，并与推理阶段的 `--prompt` 保持一致。
- 模型配置中的 `action_dim=32`、`action_horizon=50` 和数据预处理输出保持一致。

## 在线推理

首先在 OpenPI 或训练工程所在环境中启动策略服务器，例如：

```bash
python serve_policy.py \
    --port 8010 \
    --checkpoint /path/to/checkpoint
```

然后在本项目环境中运行推理：

```bash
export OPENPI_CLIENT_SRC=~/openpi/packages/openpi-client/src

python eval_g1_omnipicker_lerobot.py \
    --task_config example.yaml \
    --host localhost \
    --port 8010 \
    --prompt "按红色按钮" \
    --max_steps 500 \
    --action_repeat 1 \
    --episodes 3
```

注意：

- `--prompt` 应与训练数据中的任务语言描述保持一致。
- 默认使用三路相机图像和机器人状态进行推理。
- 仅状态策略可使用 `--no_images` 跳过图像采集。
- 不需要实时相机预览时可使用 `--no_preview`。
- `--sleep` 会按照环境的 `real_time_step` 节奏执行。

## 数据回放

```bash
python g1_omnipicker_replay_lerobot.py \
    --dataset_dir ~/datasets/g1_button_tele \
    --task_config example.yaml \
    --episode 1 \
    --steps_per_frame 3 \
    --render_every 5
```

说明：

- `--episode` 从 **1** 开始计数；省略时依次回放全部 episode。
- `--steps_per_frame` 控制每个 action 保持的仿真步数，数值越小回放越快。
- `--render_every 0` 可关闭渲染。
- 增加 `--loop` 可循环回放，使用 `Ctrl+C` 退出。
- 回放前应加载与数据采集时一致的场景布局和 `example.yaml`。

## 配置文件

| 文件 | 说明 |
|------|------|
| `example.yaml` | 采集、回放和推理共用的任务配置，`level_name: "example"` |
| `g1_button.json` | OrcaLab 按钮场景布局，包含已标定的 G1 底盘位姿 |
| `g1_tool.json` | OrcaLab 工具整理场景布局 |
| `pose_g1_button_candidates.yaml` | 四色按钮候选接触位姿，按钮脚本化采集使用 |
| `pose_g1_button_targets.yaml` | 四色按钮聚合目标位姿，参考使用 |
| `my_waypoint_tool1.yaml` … `my_waypoint_tool5.yaml` | 5 个工具的 4 点位路点（接近位 / 抓取闭爪 / 箱上方 / 箱上松开），工具整理采集使用 |

## 数据集格式

数据保存为 LeRobot v2.1 格式：

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

默认相机分辨率为 480×640，默认采集帧率为 20 FPS。当前流式视频编码器使用 `av1_nvenc`，视频封装为 MP4。

`action` 和 `observation.state` 均为 **18 维**：

| 维度 | 含义 |
|------|------|
| `[0:3]` | 左臂末端位置 `l_pos`，base 坐标系，单位米 |
| `[3:7]` | 左臂末端四元数 `l_quat_xyzw` |
| `[7:10]` | 右臂末端位置 `r_pos` |
| `[10:14]` | 右臂末端四元数 `r_quat_xyzw` |
| `[14]` | 左夹爪内侧归一化值 |
| `[15]` | 左夹爪外侧归一化值 |
| `[16]` | 右夹爪内侧归一化值 |
| `[17]` | 右夹爪外侧归一化值 |

夹爪归一化公式：

```text
norm = (motor_val + 1) / 3
```

电机量程为 `[-1, 2]`，base frame 参考体为 `g1_omnipicker_body_link1`。训练数据采用 `action[t] = state[t+1]` 的 next-state 动作定义。

## 启动前检查

- OrcaLab 版本为 6.3，OrcaGym Python 包与其版本匹配。
- `example.yaml` 中的 `level_name` 与 OrcaLab 当前场景名称一致。
- OrcaGym gRPC 服务已在 `localhost:50051` 启动。
- 三路相机已按 7070、7080、7090 配置，且端口未被其他程序占用。
- UseNvEnc、Color Camera 和 Recording 已按要求启用。
- NVIDIA 驱动和 PyAV/FFmpeg 支持 `av1_nvenc`。
- 推理时策略服务器已启动，主机、端口和 prompt 配置正确。
- 输出目录具有写权限和足够磁盘空间。

更详细的 Pico 按键映射和参数解释见 [`USAGE.md`](USAGE.md)。
