 运行命令参考

所有脚本均在 `src/examples/dataCollection/` 目录下运行，`sys.path` 会自动将 `src/` 加入搜索路径。

```bash
conda activate orca_lerobot
cd src/examples/dataCollection
```

---

## VR 按键映射表

设备：**Pico VR 手柄**（通过 `PicoJoystick` / `PicoJoystickDevice` 接入 OrcaGym）。  
左臂在 G1 场景中**全程锁死**，不绑定 `L_TRANSFORM`；底盘摇杆在本项目中**未绑定**（空闲）。

### 硬件 ↔ 代码键名

| 物理按键（Pico） | 左手 | 右手 | 代码枚举 `PicoJoystickKey` | 底层字段 |
|------------------|------|------|---------------------------|----------|
| 6DOF 位姿 | ✓ | ✓ | `L_TRANSFORM` / `R_TRANSFORM` | `position`, `rotation` |
| 侧握 Grip | ✓ | ✓ | `L_GRIPBUTTON` / `R_GRIPBUTTON` | `gripButtonPressed` |
| 扳机 Trigger | ✓ | ✓ | `L_TRIGGER` / `R_TRIGGER` | `triggerValue` ∈ [0, 1] |
| 主键 | **X** | **A** | `X` / `A` | `primaryButtonPressed` |
| 副键 | **Y** | **B** | `Y` / `B` | `secondaryButtonPressed` |
| 摇杆 | ✓ | ✓ | `L_JOYSTICK_POSITION` / `R_JOYSTICK_POSITION` | `joystickPosition` [x, y] |

### 机器人控制（两脚本共用部分）

| 功能 | 按键 | 说明 |
|------|------|------|
| 右臂末端位姿 | 右手柄 6DOF（`R_TRANSFORM`） | OSC 跟踪；Unity 左手系 → MuJoCo 右手系自动转换 |
| 左臂 | — | **锁定**，不响应左手柄位姿 |
| 左夹爪 | 左 **X** / **Y** 或 左扳机 | `Controller2F85Reverse`：X=张开，Y=闭合，扳机=模拟闭合 |
| 右夹爪 | 右 **A** / **B** 或 右扳机 | A=张开，B=闭合，扳机=模拟闭合 |
| 底盘转向 / 驱动 | 左/右摇杆 | **本场景已关闭**，摇杆无作用 |

> 夹爪逻辑见 `controller_2f85_reverse.py`：主键（X/A）增大开度，副键（Y/B）减小开度，扳机按指数曲线连续闭合。

### 脚本 A：VR 遥操作采集（`g1_omnipicker_collection_tele_lerobot.py`）

| 功能 | 按键 | 状态 / 备注 |
|------|------|-------------|
| 开始采集 | 轻按 **左 Grip** ×1 | `NOT_STARTED` → `RUNNING`；防抖 ≥ 0.2 s |
| 结束并保存 | 再轻按 **左 Grip** ×1 | `RUNNING` → `END`；**强制保存**，不论任务成败 |
| 放弃本集 | 轻按 **右 Grip**（仅右，不含左） | 丢弃当前集、重置场景、继续下一集；防抖 0.3 s |
| 终止全部采集 | **左 + 右 Grip 同时** | 丢弃未保存集，等待视频编码后退出；防抖 0.3 s |
| 强制退出 | 终端 `Ctrl+C` | 中断采集 |

**采集前门控**（`TaskStatus != RUNNING` 时）：

- 机械臂、夹爪、右臂位姿**均不响应**，保持静止。
- 仅 **左 Grip** 有效（用于开始 / 保存）。
- 进入 `RUNNING` 后，除左臂外的全部按键生效；左臂仍锁定。

**UI 提示**（OrcaStudio 场景内）：`左Grip×1=开始 左Grip×2=保存 右Grip=丢弃重置 左右Grip同按=退出`

### 脚本 B：路点录制（`record_g1_waypoints.py`）

| 功能 | 按键 | 备注 |
|------|------|------|
| 记录路点 | **左 + 右 Grip 同时**（边沿触发） | 记录右臂 base 系位姿 + 右夹爪状态；默认防抖 0.5 s |
| 退出并写 YAML | 终端 `Ctrl+C` | 自动输出 `--output` 指定文件 |

路点模式下：仅绑定右臂 + 右夹爪；左夹爪不绑按键；左臂锁死。

### 预留（代码支持，G1 按按钮场景未启用）

| 功能 | 按键 | 代码位置 |
|------|------|----------|
| 差速底盘 | 单摇杆 | `add_differential_drive_pico_controller` |
| 转向 + 油门 | 左摇杆转向 / 右摇杆油门 | `add_steering_drive_pico_controller` |

---

## 1. 路点录制（首次部署或场景变更后）

遥操右臂靠近按钮，同时按下左右 Grip 键触发记录，Ctrl+C 写出文件：

```bash
python record_g1_waypoints.py \
    --task_config example.yaml \
    --output my_waypoints_new.yaml
```

---

## 2. VR 遥操作采集

```bash
python g1_omnipicker_collection_tele_lerobot.py \
    --task_config example.yaml \
    --task "按红色按钮" \
    --lerobot_out /path/to/output_dataset \
    --fps 20
```

- 支持语言指令：`按红色按钮` / `按绿色按钮` / `按黄色按钮` / `按蓝色按钮`
- 断点续采：追加 `--resume`
- `--lerobot_out` 支持 `~` 路径展开

---

## 3. 脚本化自动采集

```bash
python g1_omnipicker_collection_scripted_button_lerobot.py \
    --task_config example.yaml \
    --lerobot_out /path/to/output_dataset \
    --fps 20 \
    --clock wall
```

启动后终端会交互式询问四色各采集集数，或通过 `--counts 25,25,25,25` 非交互传入。

候选位姿文件默认为同目录 `pose_g1_button_candidates.yaml`，可通过 `--pose_candidates` 覆盖。

---

## 4. 数据回放验证

```bash
python g1_omnipicker_replay_lerobot.py \
    --dataset_dir /path/to/lerobot_dataset \
    --task_config example.yaml \
    --episode 1 \
    --steps_per_frame 3 \
    --render_every 5
```

- `--episode`：指定集号（1-indexed），省略则顺序播完全部
- `--steps_per_frame`：每帧保持步数，越小速度越快（默认 3）
- `--loop`：循环回放（Ctrl+C 退出）

---

## 5. 推理评估

先在另一终端启动策略服务器（端口 8010）：

```bash
# 外部训练仓
python serve_policy.py --port 8010 --checkpoint /path/to/checkpoint
```

然后运行推理脚本：

```bash
# 如果 openpi_client 未安装到环境，先设置源码路径
export OPENPI_CLIENT_SRC=~/openpi/packages/openpi-client/src

python eval_g1_omnipicker_lerobot.py \
    --task_config example.yaml \
    --host localhost \
    --port 8010 \
    --prompt "按红色按钮" \
    --max_steps 500 \
    --episodes 3
```

---

## 6. 场景与配置说明

### OrcaStudio 场景加载顺序

1. 使用 **OrcaLab 6.3** 打开 OrcaStudio，并加载 `g1_button.json`（此文件含已标定的 G1 底盘世界位姿）。
2. 手动配置三路相机：Left Arm=`7070`、Right Arm=`7080`、Head=`7090`。
3. 每路相机勾选 **UseNvEnc** 和 **Color Camera**，完成全部端口及其他配置后，再勾选 **Recording**。
4. 勾选 Recording 后不要取消勾选，取消时容易导致 OrcaStudio 崩溃。
5. 不要把相机配置保存到布局 `.json`；每次重新打开布局都需要重新配置。
6. 确认 G1 机器人和电柜就位后点击「运行」。
7. 所有脚本统一使用 `--task_config example.yaml`（`level_name: "example"`）。

### example.yaml 关键字段

```yaml
level_name: "example"          # 与 OrcaStudio 中场景 level 名称对应
type: "pick_and_place"         # 任务类型（按按钮场景沿用此类型）
data_collection:
  agent_joint_prefix: "g1_omnipicker_"   # 只记录机器人关节，忽略随机 GUID 场景 actor
```

---

## 7. 数据集格式（训练对接）

LeRobot v2.1 目录结构：

```
<dataset_root>/
├── meta/
│   ├── info.json            # 数据集元信息（fps/维度/相机键等）
│   ├── episodes.jsonl       # 每集 index/length/task
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl          # 语言指令列表
├── data/chunk-000/
│   └── episode_XXXXXX.parquet   # action / observation.state / timestamp
└── videos/chunk-000/
    ├── observation.images.cam_head/
    ├── observation.images.cam_wrist_l/
    └── observation.images.cam_wrist_r/
```

相机分辨率：480×640，视频编码：MP4 封装（当前流式编码器为 `av1_nvenc`）  
语言指令列（parquet 列名）：`language_instruction` / `task`  
帧率：20 fps（`--fps 20`）