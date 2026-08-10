# 智元 G1 OmniPicker · 数据采集

本文说明智元 G1 OmniPicker 的场景准备、脚本化/遥操作采集、回放与数据格式。在线推理见 [g1_omnipicker_inference.md](g1_omnipicker_inference.md)。

---

## 场景与相机

### 加载场景

1. 请启动 OrcaLab 6.3。
2. 请按任务选择布局文件并加载：
   - 四色按按钮任务使用 `g1_button.json`
   - 工具整理任务使用 `g1_tool.json`
3. 请确认 G1 OmniPicker 与场景物体已正确加载。
4. 请确认 `example.yaml` 中的 `level_name` 与 OrcaLab 里的场景名称一致（默认均为 `"example"`）。

### 配置三路相机

打开布局后，请在 OrcaLab 中手动配置以下三路相机（两个场景配置相同）：

| 相机位置 | 代码中的相机名称 | Color Port |
|----------|------------------|------------|
| 左腕 | `camera_wrist_l_color` | 7070 |
| 右腕 | `camera_wrist_r_color` | 7080 |
| 头部 | `camera_head_color` | 7090 |

对每路相机执行以下操作：

1. 请勾选 **UseNvEnc**。
2. 请勾选 **Color Camera**。
3. 请将 **Color Port** 改为上表中对应的端口号。
4. 三路全部配完后，再统一勾选 **Recording**。

> [!CAUTION]
> 请务必先完成三路相机的全部配置，最后再勾选 Recording。Recording 勾选后请勿取消，取消时容易导致 OrcaLab 崩溃。相机配置完成后，Recording=True 不要保存到布局文件；每次重新打开布局都需要重新配置。

### 启动仿真

完成相机配置后，请点击 OrcaLab 的运行按钮启动仿真，等待 OrcaGym gRPC 服务就绪。默认地址为：

```
localhost:50051
```

---

## 运行准备

后续所有命令均在以下工作目录执行：

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection/g1_omnipicker
```

---

## 工具整理脚本化采集

请先在 OrcaLab 加载 `g1_tool.json`，再运行：

```bash
python g1_omnipicker_collection_scripted_tool_lerobot.py \
    --task_config ../common/example.yaml \
    --lerobot_out ~/datasets/g1_tool_scripted \
    --repo_id local/g1_omnipicker_tool \
    --num_episodes 20 \
    --fps 20
```

断点续采时请追加 `--resume`。

---

## 四色按钮脚本化采集

请先在 OrcaLab 加载 `g1_button.json`，再运行：

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
- 交互式终端会由脚本询问各颜色集数；非交互环境请通过 `--counts` 直接传入。
- 候选位姿文件默认为同目录的 `pose_g1_button_candidates.yaml`，可通过 `--pose_candidates` 覆盖。
- 断点续采时请追加 `--resume`。

---

## Pico 遥操作采集

智元遥操作走 Pico 原生手柄链路，通信端口为 `8001`。

### 前置步骤

请加载与任务对应的布局文件，再新开一个终端执行端口转发：

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
    --camera_source websocket
```

`--task` 可填写 `按红色按钮` / `按绿色按钮` / `按黄色按钮` / `按蓝色按钮`，或工具任务描述。断点续采时请追加 `--resume`。

### 按键映射

设备为 **Pico VR 手柄**，通过 `PicoJoystick` / `PicoJoystickDevice` 接入。

#### 硬件 ↔ 代码键名对照

| 物理按键（Pico） | 左手 | 右手 | 代码枚举 `PicoJoystickKey` | 底层字段 |
|------------------|------|------|---------------------------|----------|
| 6DOF 位姿 | ✓ | ✓ | `L_TRANSFORM` / `R_TRANSFORM` | `position`, `rotation` |
| 侧握 Grip | ✓ | ✓ | `L_GRIPBUTTON` / `R_GRIPBUTTON` | `gripButtonPressed` |
| 扳机 Trigger | ✓ | ✓ | `L_TRIGGER` / `R_TRIGGER` | `triggerValue` ∈ [0, 1] |
| 主键 | **X** | **A** | `X` / `A` | `primaryButtonPressed` |
| 副键 | **Y** | **B** | `Y` / `B` | `secondaryButtonPressed` |
| 摇杆 | ✓ | ✓ | `L_JOYSTICK_POSITION` / `R_JOYSTICK_POSITION` | `joystickPosition` [x, y] |

#### 机器人控制功能

| 功能 | 按键 | 说明 |
|------|------|------|
| 右臂末端位姿 | 右手柄 6DOF（`R_TRANSFORM`） | OSC 跟踪；Unity 左手系 → MuJoCo 右手系自动转换 |
| 左臂 | — | **全程锁定**，不响应左手柄位姿 |
| 左夹爪 | 左 **X** / **Y** 或左扳机 | X=张开，Y=闭合，扳机=按指数曲线连续闭合 |
| 右夹爪 | 右 **A** / **B** 或右扳机 | A=张开，B=闭合，扳机=按指数曲线连续闭合 |
| 底盘摇杆 | — | **本场景已关闭**，摇杆无效 |

> 夹爪逻辑见 `controller_2f85_reverse.py`：主键（X/A）增大开度，副键（Y/B）减小开度，扳机值按指数曲线驱动连续闭合。

#### 采集会话控制

| 功能 | 操作 | 说明 |
|------|------|------|
| 开始当前集 | 轻按**左 Grip** ×1 | 状态从 `NOT_STARTED` 进入 `RUNNING`；防抖 ≥ 0.2 s |
| 结束并保存 | 再次轻按**左 Grip** ×1 | 状态到 `END`；强制保存，不论任务是否成功 |
| 放弃本集 | 轻按**右 Grip**（仅右，不含左） | 丢弃当前集并重置场景；防抖 0.3 s |
| 终止全部采集 | **左 + 右 Grip 同时按下** | 丢弃未保存集，等待视频编码后退出；防抖 0.3 s |
| 强制退出 | 终端 `Ctrl+C` | 中断采集 |

**采集前保护逻辑**：未开始采集时（状态非 `RUNNING`），机械臂、夹爪和右臂位姿均不响应，保持静止；仅左 Grip 有效。脚本连接成功并不等于已开始采集，请再按一次左 Grip 后机器人才会跟随手柄。

OrcaLab 场景内会显示操作提示：`左Grip×1=开始 左Grip×2=保存 右Grip=丢弃重置 左右Grip同按=退出`

> 握持手柄时请避免左右侧握同时按下，以免误触「终止全部采集」。

#### 预留功能（当前场景未启用）

| 功能 | 绑定按键 | 代码位置 |
|------|----------|----------|
| 差速底盘 | 单摇杆 | `add_differential_drive_pico_controller` |
| 转向 + 油门 | 左摇杆转向 / 右摇杆油门 | `add_steering_drive_pico_controller` |

### 续采说明

断点续采时请在启动命令中追加 `--resume`。续采启动后，终端应打印已加载的集数与帧数（如 `[resume] 已加载 N 集 / M 帧`），随后周期性出现「正在采集第 … 集」。若只有连接类提示而没有上述信息，说明数据集加载失败，请停止并检查终端报错。

续采成功后，仍须再按**左 Grip** 开始新的一集；仅看到「手柄已连接」并不代表已经开始采集。

异常退出可能导致 meta / parquet / mp4 不一致；此时请用备份覆盖数据集后再续采，或不加 `--resume` 覆写重采。

---

## 数据回放

请先加载与采集时一致的布局文件，再执行：

```bash
python g1_omnipicker_replay_lerobot.py \
    --dataset_dir /path/to/lerobot_dataset \
    --task_config ../common/example.yaml \
    --episode 1 \
    --steps_per_frame 3 \
    --render_every 5
```

- 集号从 1 开始；省略则顺序播完全部。
- `--steps_per_frame` 越小回放速度越快（默认 3）。
- 需要循环播放时请追加 `--loop`（按 `Ctrl+C` 退出）。

---

## example.yaml 关键字段

```yaml
level_name: "example"          # 与 OrcaLab 中加载的场景名称对应
type: "pick_and_place"         # 任务类型（按按钮场景沿用此类型）
data_collection:
  agent_joint_prefix: "g1_omnipicker_"   # 只记录该前缀的关节，忽略场景随机 GUID actor
```

---

## 配置文件说明

| 文件 | 说明 |
|------|------|
| `example.yaml` | 采集、回放和推理共用的任务配置，`level_name: "example"` |
| `g1_tool.json` | OrcaLab 工具整理场景布局 |
| `g1_button.json` | OrcaLab 四色按按钮场景布局 |
| `my_waypoint_tool1.yaml` … `my_waypoint_tool5.yaml` | 工具整理采集路点 |
| `my_slot_waypoints.yaml` | 工具整理辅助路点 |
| `pose_g1_button_candidates.yaml` | 四色按钮脚本化采集候选位姿 |

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

基座坐标系为 `g1_omnipicker_body_link1`。训练时动作标签为 `action[t] = state[t+1]`。

---

## 示例数据与 PI0.5 训练参考

### 示例 VR 自采数据

- 链接：<https://pan.baidu.com/s/1Q0Zoakl4eUYLNwWpzjqajw>
- 提取码：`5hne`

### PI0.5 LoRA 训练参考（四色按钮示例）

请参考 OpenPI 官方 PI0.5 流程。下列配置可作为 G1 OmniPicker 四色按钮任务的起始参考：

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

- OrcaLab 版本为 6.3，且 `orca-gym` 版本与之匹配。
- 已加载对应场景布局（`g1_tool.json` 或 `g1_button.json`）。
- `example.yaml` 中的 `level_name` 与场景名称一致。
- OrcaGym gRPC 服务在 `localhost:50051` 已就绪。
- 三路相机端口（`7070` / `7080` / `7090`）均已配置，UseNvEnc、Color Camera、Recording 已按要求启用。
- NVIDIA 驱动与 PyAV/FFmpeg 支持 `av1_nvenc`。
- 输出目录可写且磁盘空间充足。

---

## 故障排查

**现象**：相机超时或连接失败。**原因**：端口未按上表配置，或 Recording 未勾选。**处理**：请重新配置三路相机，确认端口号与推流状态。

**现象**：续采后机器人不动。**原因**：手柄已连接但尚未开始采集。**处理**：请再按一次左 Grip 进入采集。

**现象**：续采报错或只见连接提示，无「正在采集」日志。**原因**：上次异常退出导致 meta / parquet / mp4 不一致。**处理**：请用备份覆盖数据集后再续采，或不加 `--resume` 覆写重采。

**现象**：脚本报模块找不到。**原因**：未激活正确的 conda 环境，或依赖未安装。**处理**：请确认已激活 `orcalab_lerobot`，并在仓库根目录重新执行 `bash scripts/install_runtime.sh`。
