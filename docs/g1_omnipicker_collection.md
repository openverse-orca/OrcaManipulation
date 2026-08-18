# 智元 G1 OmniPicker · 数据采集

本文说明智元 G1 OmniPicker 的场景准备、脚本化/遥操作采集、回放与数据格式。在线推理见 [g1_omnipicker_inference.md](g1_omnipicker_inference.md)。

---

## 场景与相机

### 加载场景

1. 请在运行本项目的主机上启动 OrcaLab 7.1。
2. 请按任务在 OrcaLab 的加载布局对话框中选择对应的布局文件：
   - 四色按按钮任务使用 `src/examples/dataCollection/g1_omnipicker/g1_button.json`
   - 工具整理任务使用 `src/examples/dataCollection/g1_omnipicker/g1_tool.json`
3. 请确认 G1 OmniPicker 与场景物体已正确加载。
4. 请确认 `src/examples/dataCollection/common/example.yaml` 中的 `level_name` 与 OrcaLab 里的场景名称一致（默认均为 `"example"`）。

### 配置三路相机

打开布局后，请在 OrcaLab 中手动配置以下三路相机（两个场景配置相同）：

| 相机位置 | OrcaLab 中的相机名称 | Color Port |
|----------|----------------------|------------|
| 左腕 | `camera_wrist_l_color` | 7070 |
| 右腕 | `camera_wrist_r_color` | 7080 |
| 头部 | `camera_head_color` | 7090 |

对每路相机执行以下操作：

1. 请勾选 **UseNvEnc**。
2. 请勾选 **Color Camera**。
3. 请将 **Color Port** 改为上表中对应的端口号。
4. 三路全部配完后，再统一勾选 **Recording**。

### 启动仿真

完成相机配置后，请点击 OrcaLab 的运行按钮启动仿真，并等待 OrcaLab 界面显示仿真已运行。默认的 OrcaGym 服务地址为：

```
localhost:50051
```

---

## 运行准备

后续所有命令均在运行本项目的主机上执行。请先激活 conda 环境，再从仓库根目录进入脚本所在目录：

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection/g1_omnipicker
```

---

## 路点标点（脚本化采集前置）

脚本化采集使用遥操录好的右臂末端路点，而不是在线自动算接触点。请先在 OrcaLab 中加载 `g1_tool.json`，再在上一节的工作目录中录点。

### 工具整理主路点

左右 Grip 同时按下记录一点。推荐 6 点（`--guide tool6`）：接近、抓取闭爪、放箱经由 1、放箱经由 2、箱上方、箱上松开。

```bash
python record_g1_waypoints.py \
    --task_config ../common/example.yaml \
    --output my_waypoint_tool3.yaml \
    --guide tool6
```

每个工具一份 YAML，默认文件名为 `my_waypoint_tool1.yaml` … `my_waypoint_tool5.yaml`。

### 跨槽位补录

当工具不在原生槽位时，用下面脚本补录该 `(槽位, 工具)` 的接近点 / 抓取点（wp0 / wp1）：

```bash
python record_slot_waypoints.py \
    --task_config ../common/example.yaml \
    --output my_slot_waypoints.yaml
```

脚本化采集会读取这些 YAML：主路点决定抓取与放箱轨迹，`my_slot_waypoints.yaml`（或 `*_reloc.yaml`）覆盖跨槽位抓取位姿。

---

## 工具整理脚本化采集

请先在 OrcaLab 中加载 `src/examples/dataCollection/g1_omnipicker/g1_tool.json`，再在上一节的工作目录中运行：

```bash
python g1_omnipicker_collection_scripted_tool_lerobot.py \
    --task_config ../common/example.yaml \
    --lerobot_out ~/datasets/g1_tool_scripted \
    --repo_id local/g1_omnipicker_tool \
    --num_episodes 20 \
    --fps 20
```

只抓指定槽位、或给放箱轨迹加扰动时，用同目录的另外两个入口：

```bash
python g1_omnipicker_collection_scripted_tool_slots_lerobot.py \
    --task_config ../common/example.yaml \
    --lerobot_out ~/datasets/g1_tool_slot01 \
    --slots 0,1 --num_episodes 20 --fps 20

python g1_omnipicker_collection_scripted_tool_perturb_lerobot.py \
    --task_config ../common/example.yaml \
    --lerobot_out ~/datasets/g1_tool_perturb \
    --perturb --num_episodes 20 --fps 20
```

断点续采时请追加 `--resume`。

| 参数 | 含义 | 默认值 | 何时需要改 |
|------|------|--------|------------|
| `--task_config` | 任务配置文件路径 | `../common/example.yaml` | 使用其它任务配置时 |
| `--lerobot_out` | 数据集输出目录 | 无默认值，必须指定 | 每次采集都需要指定 |
| `--repo_id` | 数据集仓库名 | `local/g1_omnipicker_tool` | 需要区分不同数据集时 |
| `--num_episodes` | 采集轮数 | `1` | 需要采集多轮时 |
| `--task` | 写入数据集的语言指令 | `整理工具` | 需要与训练时的指令对齐时 |
| `--fps` | 采集帧率 | `20` | 需要更改帧率时 |
| `--clock` | 采帧时钟源：`sim` 为仿真时间，`wall` 为墙钟 | `sim` | 需要与遥操作采集对齐时钟时改为 `wall` |
| `--resume` | 追加到已有数据集 | 未启用 | 断点续采时追加该参数 |
| `--orcagym_addr` | OrcaGym 服务地址 | `localhost:50051` | 服务不在本机默认端口时 |
| `--randomize` | 每个 episode 随机排列槽位 | 未启用 | 需要随机化抓取顺序时追加该参数 |
| `--seed` | 随机化基础种子 | 未设置 | 需要可复现的随机顺序时 |
| `--num_tools` | 本集实际抓取的工具数量 | `5` | 调试时只抓前几把工具时 |
| `--tools` | 要抓的工具编号，从 1 开始，逗号分隔 | 空，按 `--num_tools` 取前 N 把 | 只抓指定编号的工具时 |
| `--speed` | 轨迹整体提速倍率 | `1.0` | 需要加快或放慢脚本化轨迹时 |

路点与分段步数等冷门参数见文末「高级调参参数」。

---

## 四色按钮脚本化采集

请先在 OrcaLab 中加载 `src/examples/dataCollection/g1_omnipicker/g1_button.json`，再在上一节的工作目录中运行：

```bash
python g1_omnipicker_collection_scripted_button_lerobot.py \
    --task_config ../common/example.yaml \
    --lerobot_out ~/datasets/g1_button_scripted \
    --repo_id local/g1_omnipicker_button \
    --counts 25,25,25,25 \
    --fps 20 \
    --clock wall
```

说明：

- `--counts` 参数的顺序为红、绿、黄、蓝，例如 `25,25,25,25` 表示每种颜色各采 25 集。
- 交互式终端会由脚本询问各颜色集数。非交互终端若不传入 `--counts`，脚本将按红、绿、黄、蓝各 5 集执行。
- 候选位姿文件默认为同目录的 `pose_g1_button_candidates.yaml`，可通过 `--pose_candidates` 指定其它文件。
- 断点续采时请追加 `--resume`。

| 参数 | 含义 | 默认值 | 何时需要改 |
|------|------|--------|------------|
| `--task_config` | 任务配置文件路径 | `../common/example.yaml` | 使用其它任务配置时 |
| `--lerobot_out` | 数据集输出目录 | 无默认值，必须指定 | 每次采集都需要指定 |
| `--repo_id` | 数据集仓库名 | `local/g1_omnipicker_button` | 需要区分不同数据集时 |
| `--counts` | 红、绿、黄、蓝各采集集数，逗号分隔 | 未传入时：交互式终端询问，非交互终端按各色 5 集执行 | 需要指定各颜色集数时 |
| `--fps` | 采集帧率 | `20` | 需要更改帧率时 |
| `--clock` | 采帧时钟源：`sim` 为仿真时间，`wall` 为墙钟 | `wall` | 需要改用仿真时钟时 |
| `--resume` | 追加到已有数据集 | 未启用 | 断点续采时追加该参数 |
| `--orcagym_addr` | OrcaGym 服务地址 | `localhost:50051` | 服务不在本机默认端口时 |
| `--pose_candidates` | 候选位姿文件路径 | 同目录 `pose_g1_button_candidates.yaml` | 使用其它候选位姿文件时 |
| `--shuffle_seed` | 随机打乱种子 | 未设置 | 需要可复现的颜色顺序时 |

接近、前推、保压、后撤等分段步数见文末「高级调参参数」。

---

## Pico 遥操作采集

智元 G1 OmniPicker 的遥操作采集使用 Pico 头显自带的手柄，主机与头显之间通过 `8001` 端口通信。

### 前置步骤

请先在 OrcaLab 中加载与任务对应的布局文件。随后请在主机上打开另一个终端，在该终端中执行下列命令，把主机的 8001 端口转发到头显，以便头显上的手柄能够访问本机上的遥操作服务：

```bash
adb reverse tcp:8001 tcp:8001
```

### 启动命令

```bash
python g1_omnipicker_collection_tele_lerobot.py \
    --task_config ../common/example.yaml \
    --lerobot_out ~/datasets/g1_tele \
    --repo_id local/g1_omnipicker \
    --task "按红色按钮" \
    --fps 20 \
    --clock wall \
    --cameras head,wrist_r \
    --cam_resolution 480x640 \
    --camera_source websocket
```

`--task` 用于写入数据集的语言指令。按钮任务可填写 `按红色按钮`、`按绿色按钮`、`按黄色按钮` 或 `按蓝色按钮`；工具任务可填写 `整理工具`。若不传入该参数，脚本将使用默认值 `g1 omnipicker teleoperation`。断点续采时请追加 `--resume`。

| 参数 | 含义 | 默认值 | 何时需要改 |
|------|------|--------|------------|
| `--level` | 场景名称 | `default` | 场景名称不是 `default` 时 |
| `--task_config` | 任务配置文件路径 | `../common/example.yaml` | 使用其它任务配置时 |
| `--lerobot_out` | 数据集输出目录 | 无默认值，必须指定 | 每次采集都需要指定 |
| `--repo_id` | 数据集仓库名 | `local/g1_omnipicker` | 需要区分不同数据集时 |
| `--task` | 写入数据集的语言指令 | `g1 omnipicker teleoperation` | 需要与训练时的指令对齐时 |
| `--fps` | 采集帧率 | `20` | 需要更改帧率时 |
| `--clock` | 采帧时钟源：`sim` 为仿真时间，`wall` 为墙钟 | `wall` | 需要改用仿真时钟时 |
| `--resume` | 追加到已有数据集 | 未启用 | 断点续采时追加该参数 |
| `--orcagym_addr` | OrcaGym 服务地址 | `localhost:50051` | 服务不在本机默认端口时 |
| `--cameras` | 启用的相机列表，逗号分隔，可选 `head` / `wrist_r` | `head,wrist_r` | 只启用其中一路相机时 |
| `--cam_resolution` | 采集帧分辨率，格式为高×宽 | `480x640` | 需要更改分辨率时 |
| `--camera_source` | 相机数据来源：`websocket` 为内存流，`mp4` 为集末从服务端提取 | `websocket` | 需要改用集末提取时 |

### 按键映射

设备为 **Pico VR 手柄**。

#### 机器人控制功能

| 功能 | 按键 | 说明 |
|------|------|------|
| 右臂末端位姿 | 右手柄 6DOF | 右手柄带动右臂末端跟随 |
| 左夹爪 | 左 **X** / **Y** 或左扳机 | X 张开，Y 闭合，扳机连续控制开合 |
| 右夹爪 | 右 **A** / **B** 或右扳机 | A 张开，B 闭合，扳机连续控制开合 |

#### 采集会话控制

| 功能 | 操作 | 说明 |
|------|------|------|
| 开始当前集 | 第一次按**左侧握键** | 进入采集中；开始后机器人才会跟随手柄 |
| 结束并保存 | 第二次按**左侧握键** | 结束并保存当前集，不论任务是否成功 |
| 放弃本集 | 按**右侧握键**（仅右，不含左） | 丢弃当前集并重置场景 |
| 终止全部采集 | **左侧握键与右侧握键同时按下** | 丢弃未保存集，等待视频编码后退出 |
| 强制退出 | 在运行采集脚本的主机终端中按 `Ctrl+C` | 中断采集 |

未开始采集时，机械臂与夹爪均不响应手柄，保持静止；仅左侧握键有效。脚本连接成功并不等于已开始采集，请再按一次左侧握键后机器人才会跟随手柄。

OrcaLab 场景内会显示操作提示：`第一次按左侧握键=开始 第二次按左侧握键=保存 右侧握键=丢弃重置 左右同按=退出`

> 握持手柄时请避免左右侧握同时按下，以免误触「终止全部采集」。

### 续采说明

断点续采时请在启动命令中追加 `--resume`。续采启动后，终端应打印已加载的集数与帧数（如 `[resume] 已加载 N 集 / M 帧`），随后周期性出现「正在采集第 … 集」。若终端只出现「手柄已连接」等连接提示，而没有上述续采与采集信息，说明数据集加载失败，请停止脚本并检查终端报错。

续采成功后，仍须再按**左侧握键**开始新的一集；仅看到「手柄已连接」并不代表已经开始采集。

---

## 数据回放

请先在 OrcaLab 中加载与采集时一致的布局文件，再在上一节的工作目录中执行：

```bash
python g1_omnipicker_replay_lerobot.py \
    --dataset_dir /path/to/lerobot_dataset \
    --task_config ../common/example.yaml \
    --episode 1 \
    --steps_per_frame 10 \
    --render_every 5
```

- `--episode` 指定要回放的集号，集号从 1 开始。传入 `--episode 1` 时只回放第 1 集。不传入该参数时，脚本按文件名顺序回放数据集中的全部集。
- `--steps_per_frame` 表示每个数据帧重复执行的控制步数，数值越小回放速度越快（默认 10）。
- 需要循环播放时请追加 `--loop`。循环播放时，请在运行回放脚本的主机终端中按 `Ctrl+C` 退出。

| 参数 | 含义 | 默认值 | 何时需要改 |
|------|------|--------|------------|
| `--dataset_dir` | 待回放的数据集目录 | 无默认值，必须指定 | 每次回放都需要指定 |
| `--task_config` | 任务配置文件路径 | 无默认值，必须指定 | 每次回放都需要指定 |
| `--episode` | 只回放指定集号；集号从 1 开始 | 未传入则按文件名顺序回放全部集 | 只回放其中一集时 |
| `--loop` | 全部播完后从头循环 | 未启用 | 需要循环播放时追加该参数 |
| `--steps_per_frame` | 每个数据帧重复执行的控制步数 | `10` | 需要加快或放慢回放时 |
| `--render_every` | 每隔多少控制步渲染一次 | `5` | 需要调整画面刷新频率时 |
| `--orcagym_addr` | OrcaGym 服务地址 | `localhost:50051` | 服务不在本机默认端口时 |

回放跟踪刚度与近桌补偿等冷门参数见文末「高级调参参数」。

---

## example.yaml 关键字段

任务配置文件位于 `src/examples/dataCollection/common/example.yaml`，其中与采集相关的字段如下：

```yaml
level_name: "example"          # 与 OrcaLab 中加载的场景名称对应
type: "pick_and_place"         # 任务类型（按按钮场景沿用此类型）
data_collection:
  agent_joint_prefix: "g1_omnipicker_"   # 只记录该前缀的机器人关节，不记录场景中其他物体的关节
```

---

## 配置文件说明

| 文件 | 说明 |
|------|------|
| `src/examples/dataCollection/common/example.yaml` | 采集、回放和推理共用的任务配置，`level_name: "example"` |
| `src/examples/dataCollection/g1_omnipicker/g1_tool.json` | OrcaLab 工具整理场景布局 |
| `src/examples/dataCollection/g1_omnipicker/g1_button.json` | OrcaLab 四色按按钮场景布局 |
| `src/examples/dataCollection/g1_omnipicker/my_waypoint_tool1.yaml` … `my_waypoint_tool5.yaml` | 工具整理采集路点 |
| `src/examples/dataCollection/g1_omnipicker/my_slot_waypoints.yaml` | 工具整理辅助路点 |
| `src/examples/dataCollection/g1_omnipicker/pose_g1_button_candidates.yaml` | 四色按钮脚本化采集候选位姿 |

---

## 数据集格式

### 目录结构

LeRobot v2.1 格式如下：

```text
<dataset_root>/
├── meta/
│   ├── info.json               # 数据集元信息（fps / 维度 / 相机键等）
│   ├── episodes.jsonl          # 每集的 index / length / task
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl             # 语言指令列表
├── data/chunk-000/
│   └── episode_XXXXXX.parquet  # action / observation.state / timestamp
└── videos/chunk-000/
    ├── observation.images.cam_head/
    ├── observation.images.cam_wrist_l/
    └── observation.images.cam_wrist_r/
```

默认相机分辨率为 480×640，默认帧率为 20 FPS，视频格式为 MP4（`av1_nvenc`）。

### action 与 observation.state 维度（18 维）

| 维度范围 | 字段名 | 含义 |
|----------|--------|------|
| `[0:3]` | `l_pos_x/y/z` | 左臂末端位置（base 坐标系，单位米） |
| `[3:7]` | `l_quat_x/y/z/w` | 左臂末端四元数（xyzw） |
| `[7:10]` | `r_pos_x/y/z` | 右臂末端位置 |
| `[10:14]` | `r_quat_x/y/z/w` | 右臂末端四元数 |
| `[14]` | `l_gripper_inner_norm` | 左夹爪内侧归一化值 |
| `[15]` | `l_gripper_outer_norm` | 左夹爪外侧归一化值 |
| `[16]` | `r_gripper_inner_norm` | 右夹爪内侧归一化值 |
| `[17]` | `r_gripper_outer_norm` | 右夹爪外侧归一化值 |

夹爪归一化公式：`norm = (电机值 + 1) / 3`（电机量程 `[-1, 2]`）。

基座坐标系为 `g1_omnipicker_body_link1`。

---

## 示例数据与 PI0.5 训练参考

### 示例 VR 自采数据

- 链接：<https://pan.baidu.com/s/1Q0Zoakl4eUYLNwWpzjqajw>
- 提取码：`5hne`

### PI0.5 LoRA 训练参考（四色按钮示例）

请参考 OpenPI 官方 PI0.5 流程。下列配置可以作为 G1 OmniPicker 四色按钮任务的训练配置参考：

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

训练前请确认：已下载 `pi05_base` 权重；`repo_id` 与数据集一致；`prompt_from_task` 与推理时的 `--prompt` 对齐；`action_dim` / `action_horizon` 与数据集一致。

---

## 启动前检查

- OrcaLab 版本为 7.1，且 `orca-gym` 版本为 26.6.3。
- 已在 OrcaLab 中加载对应场景布局（`src/examples/dataCollection/g1_omnipicker/g1_tool.json` 或 `src/examples/dataCollection/g1_omnipicker/g1_button.json`）。
- `src/examples/dataCollection/common/example.yaml` 中的 `level_name` 与场景名称一致。
- OrcaGym 服务在 `localhost:50051` 已就绪。
- 三路相机端口（`7070` / `7080` / `7090`）均已配置，UseNvEnc、Color Camera、Recording 已按要求启用。
- NVIDIA 驱动与 PyAV/FFmpeg 支持 `av1_nvenc`。
- 输出目录可写且磁盘空间充足。

---

## 故障排查

**现象**：相机超时或连接失败。**原因**：端口未按上表配置，或 Recording 未勾选。**处理**：请重新配置三路相机，确认端口号与推流状态。

**现象**：续采后机器人不动。**原因**：手柄已连接但尚未开始采集。**处理**：请再按一次左侧握键进入采集。

**现象**：续采报错，或终端只见「手柄已连接」等连接提示，而无「正在采集」日志。**原因**：数据集未能成功加载。**处理**：请停止脚本并检查终端报错；或去掉 `--resume` 后重新采集。

**现象**：脚本报模块找不到。**原因**：未激活正确的 conda 环境，或依赖未安装。**处理**：请确认已激活 `orcalab_lerobot`，并在仓库根目录重新执行 `bash scripts/install_runtime.sh`。

---

## 高级调参参数

以下参数一般保持默认即可。需要调整脚本化轨迹节奏或回放跟踪时，再按表修改。

### 工具整理脚本化采集

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--waypoint_files` | 5 个路点 YAML 路径，逗号分隔，顺序即抓取顺序 | 同目录 `my_waypoint_tool1.yaml` … `my_waypoint_tool5.yaml` |
| `--extra_slot_waypoints` | 槽位路点 YAML；传空字符串可禁用 | 同目录 `my_slot_waypoints.yaml` |
| `--safe_z` | 安全过渡高度，基座坐标系 z，单位米 | `0.50` |
| `--steps_transit` | 高位过渡帧数 | `50` |
| `--steps_descend` | 垂直下降帧数 | `30` |
| `--steps_grasp` | 移到抓取点帧数 | `35` |
| `--steps_settle` | 抓取点沉降与闭爪驻留帧数 | `30` |
| `--steps_lift` | 抬升帧数 | `35` |
| `--steps_place_via` | 放箱经由段各自帧数 | `45` |
| `--steps_to_box` | 移到工具箱上方帧数 | `50` |
| `--steps_release` | 闭爪逼近松开位帧数 | `45` |
| `--steps_release_settle` | 松开位沉降与张开驻留帧数 | `20` |
| `--steps_lift_after` | 放置后抬升帧数 | `15` |
| `--kp` | 阻抗刚度 | `220.0` |
| `--check_box_bottom` | 该工具路点结束后检查是否入箱 | 启用 |
| `--place_check_timeout_s` | 出箱后等待接触的最长时间，单位秒 | `2.5` |
| `--max_place_retries` | 单集入箱失败最大重试次数 | `8` |

### 四色按钮脚本化采集

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--steps_approach` | 接近段步数 | `250` |
| `--steps_push` | 前推接触段步数 | `120` |
| `--steps_hold` | 保压段步数 | `40` |
| `--steps_retract` | 后撤段步数 | `150` |

### 数据回放

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--kp` | 阻抗刚度 | `200.0` |
| `--cmd_bias_x` | 右臂目标在基座坐标系 X 方向的偏移，单位米 | `0.0` |
| `--cmd_bias_y` | 右臂目标在基座坐标系 Y 方向的偏移，单位米 | `0.0` |
| `--cmd_bias_z` | 右臂目标在基座坐标系 Z 方向的偏移，单位米；`0` 表示关闭 | `-0.030` |
| `--cmd_bias_z_below` | 仅当原始右臂 z 不超过该值时施加前馈，单位米 | `0.25` |
| `--sync_nullspace` | 施加前馈时同步零空间 | 启用 |
| `--grasp_integral` | 开启近桌外环积分 | 未启用 |
| `--grasp_integral_ki` | 外环积分增益 | `0.3` |
| `--grasp_integral_max` | 外环积分偏置限幅，单位米 | `0.04` |
| `--grasp_integral_axes` | 外环积分生效轴，如 `z` / `xy` / `xyz` | `z` |
| `--grasp_integral_log_every` | 积分日志节流：每 N 控制步打印一次；`0` 表示关闭 | `10` |
| `--grasp_integral_z_below` | 仅当原始右臂 z 不超过该值时启用外环积分，单位米 | `0.25` |
