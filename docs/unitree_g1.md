# 宇树 G1

本文说明宇树 G1 四色按按钮任务的 VR 遥操作采集。宇树侧通过 TeleVuer 接收 Pico 手柄位姿，端口为 `8012`；智元侧流程见 [g1_omnipicker.md](g1_omnipicker.md)，两套机器人请勿混用配置。

---

## 场景与相机

### 加载场景

1. 请启动 OrcaLab 6.3。
2. 请加载布局文件 `unitree_button.json`。
3. 请确认场景中的机器人名称为 `unitree_humanoid_robot_1`。

### 配置两路相机

| 相机位置 | Color Port |
|----------|------------|
| 头部 `head_cam` | 7070 |
| 右腕 `wrist_cam` | 7080 |

每路相机请勾选 **UseNvEnc**、**Color Camera**、**Enable**，并设置上表端口。两路全部配完后，再统一勾选 **Recording**。Recording 勾选后请勿取消，也不要写回布局文件；每次重新打开布局都需要重新配置。

配置完成后，请点击运行，等待 OrcaGym gRPC 服务就绪，默认地址为：

```
localhost:50051
```

---

## 连接头显与启动采集

宇树遥操作通过 TeleVuer 读取 Pico 手柄位姿（端口 `8012`）。请严格按下列顺序操作。

### 1. 连接头显并做端口转发

1. 请用 USB 连接 Pico 头显，确认 `adb devices` 能看到设备。
2. 请执行端口转发：

```bash
adb reverse tcp:8012 tcp:8012
```

### 2. 启动数采脚本

请激活环境并进入脚本目录：

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection
```

启动遥操作采集脚本（请将输出目录与 `repo_id` 换成你自己的路径与名称）：

```bash
python -u g1_pick_collection_tele_lerobot.py \
    --level default \
    --task_config example.yaml \
    --scene_json unitree_button.json \
    --task "按红色按钮" \
    --lerobot_out <数据集输出目录> \
    --repo_id <数据集仓库名> \
    --fps 20 \
    --clock wall \
    --cameras head,wrist_r \
    --orcagym_addr localhost:50051 \
    --agent_name unitree_humanoid_robot_1 \
    --tv_no_tls \
    --tv_goal_mode rebased_tv \
    --tv_ee_dx 0.03
```

请等到终端打印出访问地址后再继续。关闭加密传输（`--tv_no_tls`）时，地址形如：

```text
http://127.0.0.1:8012/
```

脚本还会打印操作提示。正常进入采集后，终端会周期性出现「正在采集第 N 集」。若只有连接类提示而没有该信息，请停止并检查终端报错。

断点续采时请在命令中追加 `--resume`。

### 3. 打开头显页面并开始采集

1. 请另开一个终端，用 adb 在头显浏览器中打开上述地址：

```bash
adb reverse tcp:8012 tcp:8012
adb shell am start -a android.intent.action.VIEW \
  -d 'http://127.0.0.1:8012/'
```

2. 请在头显页面中点击 **Virtual Reality**。
3. 请轻按**左 squeeze（侧握键）**开始数采。

每次重新运行数采前：请先关闭头显当前浏览器页面，再按本节顺序重新执行（先启脚本，再 adb 打开页面）。
## 按键映射

宇树遥操作使用 **Pico VR 手柄**，通过 TeleVuer 接入（端口 `8012`）。双臂均可跟随手柄；灵巧手通过扳机控制开合；腿部与腰部保持锁定。

### 硬件 ↔ 代码键名对照

| 物理按键（Pico） | 左手 | 右手 | 代码枚举 `PicoJoystickKey` | 底层字段 |
|------------------|------|------|---------------------------|----------|
| 6DOF 位姿 | ✓ | ✓ | `L_TRANSFORM` / `R_TRANSFORM` | `position`, `rotation` |
| 侧握 Grip | ✓ | ✓ | `L_GRIPBUTTON` / `R_GRIPBUTTON` | `gripButtonPressed` |
| 扳机 Trigger | ✓ | ✓ | `L_TRIGGER` / `R_TRIGGER` | `triggerValue` ∈ [0, 1] |

### 机器人控制功能

| 功能 | 操作 | 说明 |
|------|------|------|
| 双臂末端位姿跟随 | 左右手柄 6DOF | 双臂均响应 |
| 左手灵巧手 | 左扳机 | 连续控制手指开合 |
| 右手灵巧手 | 右扳机 | 连续控制手指开合 |
| 腿部 / 腰部 | — | **全程锁定** |

### 采集会话控制

| 功能 | 操作 | 说明 |
|------|------|------|
| 开始当前集 | 轻按**左 Grip** ×1 | 防抖 ≥ 0.2 s；开始后机器人才会跟随手柄 |
| 结束并保存 | 再次轻按**左 Grip** ×1 | 强制保存当前集 |
| 放弃本集 | 轻按**右 Grip**（仅右，不含左） | 丢弃当前集并重置场景；防抖 0.3 s |
| 终止全部采集 | **左 + 右 Grip 同时按下** | 丢弃未保存集，等待视频编码后退出 |
| 强制退出 | 终端 `Ctrl+C` | 中断采集 |

未开始采集时，双臂保持静止。脚本连接成功并不等于已开始采集，请再按一次左 Grip 后机器人才会跟随手柄。握持时请避免左右侧握同时按下，以免误触「终止全部采集」。

---

## 续采说明

断点续采时请在启动命令中追加 `--resume`。续采启动后，终端应打印已加载的集数与帧数（如 `[resume] 已加载 N 集 / M 帧`），随后周期性出现「正在采集第 … 集」。若只有连接提示而没有上述信息，说明数据集加载失败，请查看终端报错。

异常退出可能导致 meta / parquet / mp4 不一致；此时请用备份覆盖数据集后再续采，或不加 `--resume` 覆写重采。

---

## 数据集格式

### 目录结构

与智元数据集结构相同，采用 LeRobot v2.1 格式：

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

### action 与 observation.state 维度（28 维）

宇树 G1 配备灵巧手，状态维度与智元不同（智元为 18 维二指夹爪，宇树为 28 维多指灵巧手）：

| 维度范围 | 字段名 | 含义 |
|----------|--------|------|
| `[0:3]` | `l_pos_x/y/z` | 左臂末端位置（base 坐标系，单位米） |
| `[3:7]` | `l_quat_x/y/z/w` | 左臂末端四元数（xyzw） |
| `[7:10]` | `r_pos_x/y/z` | 右臂末端位置 |
| `[10:14]` | `r_quat_x/y/z/w` | 右臂末端四元数 |
| `[14:21]` | `l_<joint>_norm` × 7 | 左手 7 个关节归一化值（拇指 3 + 中指 2 + 食指 2） |
| `[21:28]` | `r_<joint>_norm` × 7 | 右手 7 个关节归一化值（拇指 3 + 中指 2 + 食指 2） |

左右手各 7 个关节（按代码顺序）：

| 索引（相对每手） | 关节名 |
|------------------|--------|
| 0 | `thumb_0_joint` |
| 1 | `thumb_1_joint` |
| 2 | `thumb_2_joint` |
| 3 | `middle_0_joint` |
| 4 | `middle_1_joint` |
| 5 | `index_0_joint` |
| 6 | `index_1_joint` |

关节归一化公式：`norm = clamp(电机值 / 关节量程, 0, 1)`。

---

## 故障排查

**现象**：头显打不开页面或持续重连。**原因**：端口转发未建立，或误用了加密地址。**处理**：请重执行 `adb reverse tcp:8012 tcp:8012`，并按脚本打印的地址打开页面。

**现象**：相机超时。**原因**：头部或右腕端口未分别设为 `7070` / `7080`，或推流未启用。**处理**：请按上表重新配置相机，确认 Recording 已勾选。

**现象**：关节名报错（如 `KeyError: 'g1_pick_left_hand_thumb_0_joint'`）。**原因**：机器人前缀与场景不一致。**处理**：请确认布局中的机器人名称，并在启动命令中将 `--agent_name` 设为相同值（默认为 `unitree_humanoid_robot_1`）。

**现象**：手柄已连接但机器人不动。**原因**：尚未开始当前集。**处理**：请轻按左手柄侧握键开始采集。
