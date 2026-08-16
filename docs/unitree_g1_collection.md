# 宇树 G1 · 数据采集

本文说明宇树 G1 四色按按钮任务的场景准备、TeleVuer 遥操作采集、脚本化采集与数据格式。在线推理见 [unitree_g1_inference.md](unitree_g1_inference.md)。

---

## 环境

请先在运行本项目的主机上，按仓库根目录 README 的「环境安装」一节创建全新环境并执行：

```bash
conda activate orcalab_lerobot
bash scripts/install_runtime.sh
```

该脚本会安装仓库内的 TeleVuer，并验证 Conda 提供的 Pinocchio 3.9.0、CasADi 3.7.2。请勿自行安装 pin 或 pinocchio。

G1 的 IK 模型（`src/examples/dataCollection/assets/g1/`）随仓库提供，无需另外下载。
首次启动会在 `src/examples/dataCollection/_ik_cache/` 生成模型缓存，耗时数秒。

---

## 场景与相机

### 加载场景

1. 请在运行本项目的主机上启动 OrcaLab 7.1。
2. 请在 OrcaLab 的加载布局对话框中选择布局文件 `src/examples/dataCollection/unitree_g1/unitree_button.json`。
3. 请确认场景中的机器人名称为 `unitree_humanoid_robot_1`。

### 配置两路相机

| 相机位置 | Color Port |
|----------|------------|
| 头部 `head_cam` | 7070 |
| 右腕 `wrist_cam` | 7080 |

每路相机请勾选 **UseNvEnc**、**Color Camera**、**Enable**，并设置上表端口。两路全部配完后，再统一勾选 **Recording**。

配置完成后，请点击 OrcaLab 的运行按钮启动仿真，并等待 OrcaLab 界面显示仿真已运行。默认的 OrcaGym 服务地址为：

```
localhost:50051
```

---

## 连接头显与启动采集

宇树遥操作通过 TeleVuer 读取 Pico 手柄位姿（端口 `8012`）。请严格按下列顺序操作。

### 1. 连接头显并做端口转发

1. 请用 USB 连接 Pico 头显，确认 `adb devices` 能看到设备。
2. **第一次遥操作之前**（每台 Pico 只需做一次；恢复出厂后需重做）：请禁用房间标定向导。否则进入 VR 时会强制弹出标定流程，WebXR 无法启动：

```bash
# 将 <序列号> 替换为 adb devices 列出的设备序列号
adb -s <序列号> shell pm disable-user --user 0 com.pvr.roomcapture
```

3. 请执行端口转发：

```bash
adb reverse tcp:8012 tcp:8012
```

### 2. 启动数采脚本

请在运行本项目的主机上激活环境，再从仓库根目录进入脚本所在目录：

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection/unitree_g1
```

启动遥操作采集脚本（请将输出目录与 `repo_id` 替换为实际的路径和名称）：

```bash
python -u g1_pick_collection_tele_lerobot.py \
    --level default \
    --task_config ../common/example.yaml \
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

请等到终端打印出访问地址后再继续。USB 与 `adb reverse` 模式下，地址形如：

```text
http://127.0.0.1:8012/
```

如需改走无线连接（头显直接访问本机 IP），必须同时传入
`--tv_host <本机IP>`、`--tv_cert_file <cert.pem>` 与 `--tv_key_file <key.pem>`。
非回环地址必须启用 TLS，才能在头显中进入沉浸式会话。

脚本还会打印操作提示。正常进入采集后，终端会周期性出现「正在采集第 N 集」。若终端只出现连接提示而没有该信息，请停止脚本并检查终端报错。

断点续采时请在命令中追加 `--resume`。

### 3. 打开头显页面并开始采集

1. 请在主机上打开另一个终端，在该主机终端中执行下列 adb 命令，以便在头显浏览器中打开下述地址：

```bash
adb reverse tcp:8012 tcp:8012
adb shell am start -a android.intent.action.VIEW \
  -d 'http://127.0.0.1:8012/'
```

2. 请在头显页面中点击 **Virtual Reality**。
3. 请轻按**左 squeeze（侧握键）**开始数采。

每次重新运行数采前，请先关闭头显当前浏览器页面，再按本节顺序重新执行：先启动数采脚本，再在主机终端中用 adb 打开页面。

---

## 脚本化四色按钮采集

请先在 OrcaLab 中加载 `src/examples/dataCollection/unitree_g1/unitree_button.json`，再在运行本项目的主机上执行脚本化四色按钮采集：

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection/unitree_g1

python -u g1_pick_collection_scripted_button_lerobot.py \
    --level default \
    --task_config ../common/example.yaml \
    --scene_json unitree_button.json \
    --agent_name unitree_humanoid_robot_1 \
    --pose_candidates pose_g1_pick_button \
    --lerobot_out <数据集输出目录> \
    --repo_id <数据集仓库名> \
    --counts 5,5,5,5 \
    --fps 20 --clock wall --cameras head,wrist_r \
    --cam_resolution 480x640
```

`--pose_candidates` 默认使用目录 `pose_g1_pick_button/`（每种颜色对应一个 YAML）。也可指定单文件 `pose_g1_pick_button_candidates.yaml`。臂部刚度、到位检测等控制参数已内置默认值；如需调整，请通过脚本 `--help` 查看对应选项。


---

## 按键映射

宇树遥操作使用 **Pico VR 手柄**，通过 TeleVuer 接入（端口 `8012`）。双臂均可跟随手柄；灵巧手通过扳机控制开合；腿部与腰部保持锁定。

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
| 开始当前集 | 第一次按**左 squeeze** | 进入采集中；开始后机器人才会跟随手柄 |
| 结束并保存 | 第二次按**左 squeeze** | 结束并保存当前集 |
| 放弃本集 | 按**右 squeeze**（仅右，不含左） | 丢弃当前集并重置场景 |
| 终止全部采集 | **左 squeeze 与右 squeeze 同时按下** | 丢弃未保存集，等待视频编码后退出 |
| 强制退出 | 在运行采集脚本的主机终端中按 `Ctrl+C` | 中断采集 |

未开始采集时，双臂保持静止。脚本连接成功并不等于已开始采集，请再按一次左 squeeze 后机器人才会跟随手柄。握持时请避免左右侧握同时按下，以免误触「终止全部采集」。

---

## 续采说明

断点续采时请在启动命令中追加 `--resume`。续采启动后，终端应打印已加载的集数与帧数（如 `[resume] 已加载 N 集 / M 帧`），随后周期性出现「正在采集第 … 集」。若终端只出现连接提示而没有上述信息，说明数据集加载失败，请查看终端报错。

---

## 数据集格式

### 目录结构

采用 LeRobot v2.1 格式：

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

| 维度范围 | 含义 |
|----------|------|
| `[0:7]` | 左臂 7 关节角（rad） |
| `[7:14]` | 右臂 7 关节角 |
| `[14:21]` | 左手 7 关节角 |
| `[21:28]` | 右手 7 关节角 |

各维度均为对应关节的角度，单位为弧度。

---

## 故障排查

**现象**：头显打不开页面或持续重连。**原因**：端口转发未建立、先开了旧浏览器页、或误用了加密地址。**处理**：请关闭头显当前页面 → 确认数采脚本已启动 → 重执行 `adb reverse` 与 `adb shell am start ...` 打开 `http://127.0.0.1:8012/`（`--tv_no_tls` 时用 http）。

**现象**：相机超时。**原因**：头部或右腕端口未分别设为 `7070` / `7080`，或推流未启用。**处理**：请按上表重新配置相机，确认 Recording 已勾选。

**现象**：关节名报错。**原因**：机器人前缀与场景不一致。**处理**：请确认布局中的机器人名称，并在启动命令中将 `--agent_name` 设为相同值（默认为 `unitree_humanoid_robot_1`）。

**现象**：手柄已连接但机器人不动。**原因**：尚未开始当前集。**处理**：请轻按左手柄 squeeze 开始采集。

**现象**：进入头显后弹出「房间标定」向导，无法进入 VR 页面。**原因**：Pico 系统的 `com.pvr.roomcapture` 未禁用。**处理**：请在头显中退出房间标定界面，然后在电脑终端执行（每台 Pico 只需做一次，恢复出厂后需重做）：

```bash
adb shell pm disable-user --user 0 com.pvr.roomcapture
```

禁用后重新进入头显浏览器，按正常流程打开 `http://127.0.0.1:8012/`。

**现象**：按左手柄 squeeze 毫无反应，终端也无任何日志。**原因**：8012 端口已被其它进程占用，TeleVuer 服务未能启动。**处理**：请先释放端口，再重新启动数采脚本：

```bash
fuser -k 8012/tcp
```
