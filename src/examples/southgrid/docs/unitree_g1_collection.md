# 宇树 G1 · 数据采集与回放

本文说明宇树 G1 的相机配置、Pico 遥操作、工具/按钮脚本化采集、回放与按键。通用数据链路见 [lerobot_dataset.md](../../../../docs/lerobot_dataset.md) 与 [cameras_and_video.md](../../../../docs/cameras_and_video.md)。姿态约束在 `g1_pick_constraints.py`，采集 / 回放 / 推理入口都调用它。在线推理见 [examples/inference/README.md](../../inference/README.md)。

---

## 场景与相机

### 加载场景

1. 请在运行本项目的主机上启动 OrcaLab。
2. 请在 OrcaLab 的加载布局对话框中选择任务对应的布局文件：
   - 工具与电柜场景：`src/examples/southgrid/unitree_g1/g1_pick_tools.json`
   - 按钮场景：`src/examples/southgrid/unitree_g1/g1_pick_buttons.json`
3. 请确认宇树 G1 与场景物体已正确加载。
4. 请确认 `src/examples/southgrid/unitree_g1/example.yaml` 中的 `level_name` 与 OrcaLab 场景名称一致，默认值为 `"example"`。

两个布局当前保存的机器人名称相同：

| 布局文件 | 布局中的机器人名称 | `--agent_name` |
|----------|--------------------|----------------|
| `g1_pick_tools.json` | `g1_pick` | `g1_pick` |
| `g1_pick_buttons.json` | `g1_pick` | `g1_pick` |

本文的示例命令均显式传入：

```text
--agent_name g1_pick
```

采集、回放和推理入口的默认 `--agent_name` 都是 `g1_pick`。示例仍显式传入该参数，便于在使用自定义布局时检查机器人名称是否一致。

### 配置相机

当前 Unitree G1 采集链路使用头部和右腕两路彩色相机：

| 相机位置 | 布局内相机实体 | 代码中的相机名称 | LeRobot 数据键 | Color Port |
|----------|----------------|------------------|------------------|------------|
| 右腕 | `camera_right` | `camera_wrist_r_color` | `cam_wrist_r` | 7080 |
| 头部 | `head_cam` | `camera_head_color` | `cam_head` | 7090 |

以上端口同时由两个布局与代码确认：

- `g1_pick_tools.json` 和 `g1_pick_buttons.json` 均为右腕相机明确设置 `ColorPort: 7080`，为头部相机明确设置 `ColorPort: 7090`，并启用 `ColorCamera`、`UseNvEnc` 和 `Enable`。
- `g1_pick_buttons.json` 还为两路相机显式设置了 `IsRecording: true`。
- `conf.g1_pick_osc_conf.camera_map()` 使用相同的端口：右腕 `7080`、头部 `7090`。

加载布局后，请在 OrcaLab 中检查两路相机：

1. `Color Camera` 已启用。
2. `UseNvEnc` 已启用。
3. 相机组件处于启用状态。
4. 右腕和头部 `Color Port` 分别为 `7080` 和 `7090`。
5. 启动仿真后没有其它程序占用这两个端口。

共享相机代码还保留左腕 `camera_wrist_l_color:7070`，但当前 Unitree G1 遥操作和脚本化采集入口的 `--cameras` 只支持 `head` 与 `wrist_r`。本文示例不启用左腕相机。

### 启动仿真

完成场景和相机检查后，请点击 OrcaLab 的运行按钮启动仿真，并等待 OrcaGym 服务就绪。默认服务地址为：

```text
localhost:50051
```

---

## 运行准备

请先按仓库根目录 [README](../../../../README.md) 完成环境安装，并确认已在 OrcaLab 资产库中订阅 `SouthGrid_Competition_2026` 与 `g1_pick`。

后续命令均在运行 OrcaLab 的主机上执行：

```bash
conda activate orcalab_lerobot
cd src/examples/southgrid/unitree_g1
```

示例命令中的数据集统一写入 `$HOME/southgrid_datasets`。可以换成其它可写目录，但不要在同一目录中混入不同机器人、不同 state/action schema 或不同相机组合的数据。

---

## Pico 遥操作采集

遥操作脚本为 `g1_pick_osc_collection_tele_lerobot.py`。右臂使用 OSC 跟随 Pico 右手柄位姿，右夹爪使用 Pico 按键或扳机控制，采集结果写为 LeRobot v2.1 数据集。

### Pico 端口转发

连接 Pico 后，请先确认 ADB 能发现设备，并把主机的 `8001` 端口反向转发到头显：

```bash
adb devices
adb reverse tcp:8001 tcp:8001
```

每次重新连接或重启 Pico 后，建议重新执行端口转发命令。

### 启动命令

请先加载 `g1_pick_tools.json` 并启动仿真，再执行：

```bash
python g1_pick_osc_collection_tele_lerobot.py \
    --task_config example.yaml \
    --agent_name g1_pick \
    --lerobot_out $HOME/southgrid_datasets/g1_osc \
    --repo_id local/g1_pick \
    --task "按红色按钮" \
    --fps 20 \
    --clock wall \
    --cameras head,wrist_r \
    --camera_source websocket
```

`g1_pick` 的 OSC 默认 `--dls_lambda 0.23 --dls_sigma_th 0.12`。脚本会调用 `pin_all_joints` 钉住浮动基座、腰和左臂。

| 参数 | 含义 | 脚本默认值 | 示例值或使用建议 |
|------|------|------------|------------------|
| `--task_config` | 场景任务配置文件 | `example.yaml` | 一般无需修改 |
| `--agent_name` | OrcaLab 布局中的机器人名称 | `g1_pick` | 与工具、按钮两个布局一致 |
| `--lerobot_out` | LeRobot 数据集输出目录 | 无，必须指定 | 每个数据集使用独立目录 |
| `--repo_id` | 写入数据集元信息的仓库名 | `local/g1_pick` | 可按任务修改 |
| `--task` | 写入数据集的语言指令 | `g1 pick teleoperation` | 应与实际任务和训练指令一致 |
| `--fps` | 数据采集帧率 | `20` | 遥操作推荐 20 |
| `--clock` | 采帧时钟：`wall` 或 `sim` | `wall` | 遥操作推荐 `wall` |
| `--resume` | 追加到已有数据集 | 未启用 | 断点续采时追加 |
| `--cameras` | 启用的相机，可选 `head`、`wrist_r` | `head,wrist_r` | 默认使用两路相机 |
| `--camera_source` | `websocket` 流式采集或 `mp4` 集末提取 | `websocket` | 推荐 `websocket` |
| `--orcagym_addr` | OrcaGym 服务地址 | `localhost:50051` | 服务地址变化时修改 |

### 按键映射

采集操作集中在右臂和右夹爪；左臂由 `pin_all_joints` 钉在停靠姿态，不跟随 Pico。

| 功能 | 操作 | 说明 |
|------|------|------|
| 右臂末端位姿 | 移动 Pico 右手柄 | 右手柄 6DOF 位姿驱动右臂 OSC |
| 右夹爪张开 | 右手柄 A | 离散张开 |
| 右夹爪闭合 | 右手柄 B | 离散闭合 |
| 右夹爪连续开合 | 右扳机 | 按扳机量连续控制 |
| 开始当前集 | 第一次按左 Grip 侧握键 | 开始后右臂才响应手柄并开始记录 |
| 结束并保存 | 第二次按左 Grip 侧握键 | 无论任务是否成功，均保存当前集 |
| 放弃当前集 | 单按右 Grip 侧握键 | 丢弃当前集并重置场景 |
| 终止全部采集 | 左右 Grip 同时按下 | 丢弃未保存集，等待视频编码完成后退出 |
| 强制中断 | 主机终端按 `Ctrl+C` | 中止采集并执行退出清理 |

脚本连接到 Pico 后并不会立即驱动机器人。场景重置后必须先按一次左 Grip 进入 `RUNNING` 状态，右臂和右夹爪才会响应手柄。

### 断点续采

向原数据集追加 episode 时，在原命令末尾加入：

```text
--resume
```

启动后应看到已加载的数据集 episode 数和帧数。续采只允许写入与当前 state/action schema 和相机特征一致的数据集；若元信息不一致，脚本会拒绝续写。

---

## 路点从哪来

路点由通用标点入口写出，默认保存到 `examples/southgrid/waypoint/g1_pick/`。在仓库 `src/examples/dataCollection` 下、OrcaLab 已加载对应布局后：

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection

# → waypoint/g1_pick/my_waypoint_tool.yaml
python record_waypoints.py --agent_name g1_pick --task tool \
  --task_config ../southgrid/unitree_g1/example.yaml --guide tool6

# → waypoint/g1_pick/my_waypoint_button1.yaml
python record_waypoints.py --agent_name g1_pick --task button --color red \
  --task_config ../southgrid/unitree_g1/example.yaml
```

按键：双 Grip 记一个点，单右 Grip 重置当前任务，Ctrl+C 写出 YAML。宇树按钮文件带 `button_color` / `task` 头。`--task slot` 不支持 `g1_pick`。

---

## 脚本化数据采集

`g1_pick_osc_collection_scripted_lerobot.py` 读取**一个**路点 YAML，将各段插值为连续轨迹，自动控制右臂和右夹爪。相对路径以脚本目录为基准。

### 工具任务示例

仓库自带路点在 `waypoint_tool/`。新录的路点默认在 `../waypoint/g1_pick/`。

```bash
python g1_pick_osc_collection_scripted_lerobot.py \
    --task_config example.yaml \
    --agent_name g1_pick \
    --waypoint waypoint_tool/my_waypoint_tool1.yaml \
    --lerobot_out $HOME/southgrid_datasets/g1_osc_tools \
    --repo_id local/g1_pick_scripted \
    --max_episodes 1 \
    --fps 20 \
    --clock sim
```

### 按钮任务示例

请先在 OrcaLab 中加载 `g1_pick_buttons.json` 并启动仿真。每次运行读一个按钮路点文件：

```bash
python g1_pick_osc_collection_scripted_lerobot.py \
    --task_config example.yaml \
    --agent_name g1_pick \
    --waypoint my_waypoint_button/my_waypoint_button1.yaml \
    --lerobot_out $HOME/southgrid_datasets/g1_osc_buttons \
    --repo_id local/g1_pick_scripted \
    --max_episodes 1 \
    --fps 20 \
    --clock sim
```

四个 `my_waypoint_button` 文件分别定义一条按钮轨迹，YAML 顶层带 `button_color` 和 `task`：

| 路点文件 | 按钮颜色 | 写入的 task prompt |
|----------|----------|--------------------|
| `my_waypoint_button1.yaml` | 红色 | `按红色按钮` |
| `my_waypoint_button2.yaml` | 绿色 | `按绿色按钮` |
| `my_waypoint_button3.yaml` | 黄色 | `按黄色按钮` |
| `my_waypoint_button4.yaml` | 蓝色 | `按蓝色按钮` |

| 参数 | 含义 | 默认值 | 使用建议 |
|------|------|--------|----------|
| `--waypoint` | 路点 YAML | `my_waypoint_button/my_waypoint_button1.yaml` | 工具任务改成 `waypoint_tool/...` 或 `../waypoint/g1_pick/...` |
| `--lerobot_out` | LeRobot 数据集输出目录 | 无，必须指定 | 每个数据集使用独立目录 |
| `--repo_id` | 数据集仓库名 | `local/g1_pick_scripted` | 可按任务修改 |
| `--max_episodes` | 采集轮数 | `1` | 需要多集时 |
| `--fps` | 数据采集帧率 | `20` | 一般保持 20 |
| `--clock` | `sim` 或 `wall` | `sim` | 脚本化采集推荐 `sim` |
| `--cameras` | 启用的相机 | `head,wrist_r` | 默认两路 |
| `--track_ki` | 末端位置外环积分增益 | `0.02` | 需要关闭时设为 `0` |
| `--track_clamp` | 积分补偿限幅，单位米 | `0.08` | 仅在 `track_ki > 0` 时生效 |

---

## 数据回放

`g1_pick_osc_replay_lerobot.py` 从 LeRobot 数据集的 parquet 文件中读取 18 维 action，只驱动右臂 OSC 和右夹爪进行回放。回放不读取数据集视频，也不需要连接相机。

请加载与采集时相同的布局并启动仿真，再执行：

```bash
python g1_pick_osc_replay_lerobot.py \
    --lerobot_out $HOME/southgrid_datasets/g1_osc_scripted \
    --task_config example.yaml \
    --agent_name g1_pick \
    --episode_index 0
```

| 参数 | 含义 | 默认值 | 使用建议 |
|------|------|--------|----------|
| `--lerobot_out` | 待回放的 LeRobot 数据集根目录 | 无，必须指定 | 目录下应存在 `data/chunk-*` |
| `--episode_index` | 回放第几集，从 0 开始 | `0` | 回放其它集时 |
| `--steps_per_frame` | 每个 parquet 帧重复执行的控制步数 | `0`（按 fps 推算） | 需要加快或放慢时 |
| `--orcagym_addr` | OrcaGym 服务地址 | `localhost:50051` | 服务地址变化时修改 |

---

## OSC 与物理参数

遥操作、脚本化采集和回放应使用一致的机器人名称、任务模型配置和物理步长。

| 参数 | 含义 | 推荐或示例值 |
|------|------|--------------|
| `--dls_lambda` | DLS 最大阻尼系数；设为 0 使用原始伪逆 | `g1_pick` 默认 `0.23` |
| `--dls_sigma_th` | 最小奇异值触发阈值；0 表示固定阻尼 | `0.12` |
| `--null_kp` | 零空间关节复原增益 | `10` |
| `--kp` | OSC 阻抗刚度覆盖值 | `0` 表示沿用控制器配置 |

入口脚本的 `frame_skip` 固定为 `5`。回放 `--steps_per_frame 0` 时按数据集 fps 与控制周期推算。

---

## example.yaml 关键字段

Unitree G1 任务配置位于 `src/examples/southgrid/unitree_g1/example.yaml`：

```yaml
level_name: "example"
type: "pick_and_place"
data_collection:
  agent_joint_prefix: "g1_pick_"
```

运行时，三个入口脚本都会根据 `--agent_name` 覆盖 `agent_joint_prefix`。因此最重要的是保证命令中的 `--agent_name` 与当前布局中的机器人名称完全一致。

---

## 配置与入口文件

| 文件 | 说明 |
|------|------|
| `src/examples/southgrid/unitree_g1/example.yaml` | 场景和数据采集配置 |
| `src/examples/southgrid/unitree_g1/g1_pick_tools.json` | 工具与电柜场景；显式配置头部 7090、右腕 7080 |
| `src/examples/southgrid/unitree_g1/g1_pick_buttons.json` | 按钮场景；显式配置头部 7090、右腕 7080 |
| `src/examples/southgrid/unitree_g1/g1_pick_osc_collection_tele_lerobot.py` | Pico 遥操作和 LeRobot 数据采集 |
| `src/examples/southgrid/unitree_g1/g1_pick_osc_collection_scripted_lerobot.py` | 路点插值与脚本化数据采集 |
| `src/examples/southgrid/unitree_g1/g1_pick_osc_replay_lerobot.py` | LeRobot parquet 数据回放 |
| `src/examples/southgrid/unitree_g1/waypoint_tool/*.yaml` | 工具任务脚本化路点 |
| `src/examples/southgrid/unitree_g1/my_waypoint_button/*.yaml` | 按钮任务脚本化路点 |
| `src/sensor/camera_stream.py 与 conf.g1_pick_osc_conf.camera_map()` | 相机名称、端口和 WebSocket 连接实现 |
| `src/dataStorage/g1_pick_osc_data_storage.py` | Unitree G1 的 18 维 state/action 定义 |

入口会根据命令行参数加载相应的任务模型配置，无需单独运行辅助模块。

---

## 数据集格式

### 目录结构

采集结果采用 LeRobot v2.1 格式：

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
    └── observation.images.cam_wrist_r/
```

默认采集分辨率为 480×640，默认帧率为 20 FPS。WebSocket 模式下，相机帧由 NVENC 流式编码为 MP4。

### state 与 action

`observation.state` 和 `action` 均为 18 维：

```text
[左末端位置 3,
 左末端四元数 xyzw 4,
 右末端位置 3,
 右末端四元数 xyzw 4,
 左夹爪归一化控制量 2,
 右夹爪归一化控制量 2]
```

默认 `action[i]` 为下一采样时刻的绝对 state，即 `state[i+1]`。回放脚本读取其中的右末端位置、右末端四元数和右夹爪控制量，只驱动右臂与右夹爪。

训练、分析或系统集成时，请以数据集内 `meta/info.json` 的 feature 定义为准。

---

## 在线推理

宇树推理走通用入口，wrapper 注入 `--agent_name g1_pick`、`--pin_joints` 和 `pin_all_joints`。需先启动 OpenPI 策略服务，再：

```bash
conda activate orcalab_lerobot
cd src/examples/southgrid/inference/unitree_g1
python eval_g1_pick_lerobot.py --host 127.0.0.1 --port 8010 --prompt "整理工具"
```

通用参数见 [examples/inference/README.md](../../inference/README.md)。

---

## 启动前检查

- 已安装仓库要求的运行环境，并激活 `orcalab_lerobot`。
- 已订阅 `SouthGrid_Competition_2026` 与 `g1_pick` 资产。
- 已加载与任务对应的布局并启动仿真。
- 命令中的 `--agent_name` 与布局机器人名称完全一致。
- `example.yaml` 的 `level_name` 与 OrcaLab 场景名称一致。
- OrcaGym 服务 `localhost:50051` 已就绪。
- 头部相机端口为 `7090`，右腕相机端口为 `7080`。
- 两路相机均已启用 `Color Camera`、`UseNvEnc` 和相机组件。
- Pico 已被 `adb devices` 识别，并已执行 `adb reverse tcp:8001 tcp:8001`。
- GPU、NVIDIA 驱动和 PyAV/FFmpeg 支持 `av1_nvenc`。
- 数据集输出目录可写且磁盘空间充足。
- 仓库自带路点在 `waypoint_tool/` 与 `my_waypoint_button/`；新录路点在 `examples/southgrid/waypoint/g1_pick/`。

---

## 故障排查

**现象**：脚本找不到机器人或初始化失败。 **处理**：当前工具和按钮布局的机器人名称都是 `g1_pick`；使用自定义布局时，请检查 `AgentList` 并将 `--agent_name` 改为实际名称。

**现象**：相机端口 `7080` 或 `7090` 超时。 **处理**：当前两个布局都显式配置了右腕 `7080`、头部 `7090`；请确认仿真已运行，并检查 `Color Camera`、`UseNvEnc` 和相机启用状态。

**现象**：Pico 显示已连接，但机器人不动。 **处理**：连接成功后还要第一次按下左 Grip 才会开始当前集并解除采集前冻结。

**现象**：Pico 没有输入。 **处理**：执行 `adb devices`，确认设备已授权，再重新执行 `adb reverse tcp:8001 tcp:8001`，并确认 Pico 端应用已启动。

**现象**：脚本化采集找不到路点文件。 **处理**：相对路径以 `src/examples/southgrid/unitree_g1` 为基准。仓库自带文件用 `waypoint_tool/文件名.yaml` 或 `my_waypoint_button/文件名.yaml`；新录文件用 `../waypoint/g1_pick/文件名.yaml`。

**现象**：使用 `--resume` 时拒绝续写。 **处理**：检查旧数据集的 state/action feature 和相机键是否与当前命令一致。不要向不同 schema 或不同相机组合的数据集续写。

**现象**：视频编码失败或报找不到 `av1_nvenc`。 **处理**：确认 NVIDIA GPU 和驱动支持 AV1 NVENC，并使用仓库安装脚本配置的 PyAV/FFmpeg 环境。

**现象**：脚本报模块找不到。 **处理**：确认已激活 `orcalab_lerobot`，并在仓库根目录重新执行 `bash scripts/install_runtime.sh`。
