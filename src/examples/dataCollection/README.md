# 数据采集示例（dataCollection）

在 OrcaLab / OrcaStudio 仿真中采集机器人演示数据。本目录包含 **HDF5** 与 **LeRobot v2.1** 两套脚本。

## 前置条件

1. **OrcaLab 已启动**，场景已加载，仿真 gRPC 可达（默认 `localhost:50051`）
2. **VR 遥操作**还需：
   - Pico 安装 OrcaGymCtrl App
   - USB 连接 PC，执行端口转发：
     ```bash
     adb reverse tcp:8001 tcp:8001
     ```
3. **LeRobot 脚本**还需相机推流就绪（OrcaStudio 配置 7070/7080/7090 后，脚本会调用 `begin_save_video` 触发）

## 环境安装

本机实测环境：`orcalab_lerobot`（Python 3.12.13，`orca-gym==26.6.3`，`lerobot==0.3.4` editable）。

### 方式 A：已有 orcalab_lerobot 环境

```bash
conda activate orcalab_lerobot

# 在仓库根目录
pip install -r requirements.txt
pip install -e <你的 OrcaGym 仓库>/lerobot    # 本机示例：/home/dht/OrcaGym/lerobot
pip install -r src/examples/dataCollection/requirements-lerobot.txt
```

### 方式 B：从配置文件新建

```bash
# 在仓库根目录
conda env create -f src/examples/dataCollection/environment.yml
conda activate orcalab_lerobot
pip install -e <你的 OrcaGym 仓库>/lerobot
```

> `requirements-lerobot.txt` 中的版本号与 `orcalab_lerobot` 环境实测对齐，非宽松范围。
> HDF5 采集（`data_collection_tele.py`）只需 `requirements.txt`。
> `eval_lerobot.py` 还需安装 `openpi-client`（见 `requirements-lerobot.txt` 末尾注释）。

## 进入目录

所有命令均在本目录执行：

```bash
cd src/examples/dataCollection
```

## 脚本一览

| 脚本 | 格式 | 输入设备 | 典型用途 |
|------|------|----------|----------|
| `data_collection_tele.py` | HDF5 | Pico VR | 遥操作采集（原版） |
| `data_collection_tele_lerobot.py` | LeRobot | Pico VR | 遥操作采集（训练用 parquet+视频） |
| `data_collection_scripted.py` | HDF5 | 脚本轨迹 | 自动轨迹采集 |
| `data_collection_scripted_lerobot.py` | LeRobot | 脚本轨迹 | 自动轨迹采集（LeRobot） |
| `data_collection_replay_lerobot.py` | — | — | LeRobot 数据集 manager 模式回放 |
| `replay_lerobot.py` | — | — | LeRobot 数据集轻量回放验证 |
| `eval_lerobot.py` | — | OpenPI 策略 | 远程策略推理评估 |

共用参数（部分脚本适用）：

| 参数 | 适用脚本 | 说明 |
|------|----------|------|
| `--level` | 采集类 | 场景标签，用于区分数据子目录 |
| `--agent_name` | `tele` / `tele_lerobot` / `tele.py` / `replay` | `openloong` 或 `tiangong2` |
| `--task_config` | 全部 | 场景 YAML（如 `example.yaml`、`scripted-example.yaml`） |
| `--orcagym_addr` | 全部 | gRPC 地址，默认 `localhost:50051` |

## 快速开始

### 1. HDF5 遥操作采集

```bash
python data_collection_tele.py \
    --level shop_tele \
    --agent_name openloong \
    --task_config example.yaml
```

数据输出：

```
dataset/<agent_name>/<level>/<episode_uuid>/record/proprio_stats.hdf5
dataset/<agent_name>/<level>/<episode_uuid>/video/
```

### 2. LeRobot 遥操作采集（推荐）

```bash
python data_collection_tele_lerobot.py \
    --level shop_tele \
    --agent_name openloong \
    --task_config example.yaml \
    --lerobot_out /path/to/lerobot_data/shop_openloong \
    --repo_id local/shop_openloong \
    --task "robot arm pick and place" \
    --fps 20 \
    --clock wall
```

数据输出：直接写入 `--lerobot_out`（parquet + 编码视频 + meta）。

断点续采：加 `--resume`。

WebSocket 端口不可用时，改用服务端 MP4 按集录制：

```bash
python data_collection_tele_lerobot.py ... --camera_source mp4
```

### 3. LeRobot 脚本化采集

`data_collection_scripted_lerobot.py` 无需 `--agent_name`（内部固定 openloong）：

```bash
python data_collection_scripted_lerobot.py \
    --level scripted \
    --task_config scripted-example.yaml \
    --lerobot_out /path/to/lerobot_data/scripted \
    --repo_id local/scripted_openloong \
    --episodes 10 \
    --pose_file pose.yaml \
    --fps 30
```

### 4. 策略评估（OpenPI）

先启动策略服务，再执行：

```bash
python eval_lerobot.py \
    --task_config scripted-example.yaml \
    --host localhost --port 8010 \
    --prompt "robot arm pick and place" \
    --max_steps 500 --episodes 3
```

> `--prompt` 必须与训练时一致。

### 5. LeRobot 回放验证

#### 完整 manager 模式回放（`data_collection_replay_lerobot.py`，推荐）

走 `manager.run()` 事件驱动机制，与采集流程完全对称，适用于验证场景还原效果。固定 openloong，无需 `--agent_name`。

```bash
# 回放全部集
python data_collection_replay_lerobot.py \
    --dataset_dir /path/to/lerobot_data/shop_openloong \
    --task_config example.yaml

# 仅回放第 1 集（1-indexed）
python data_collection_replay_lerobot.py \
    --dataset_dir /path/to/lerobot_data/shop_openloong \
    --task_config example.yaml \
    --episode 1

# 循环回放（Ctrl+C 停止）
python data_collection_replay_lerobot.py \
    --dataset_dir /path/to/lerobot_data/shop_openloong \
    --task_config example.yaml \
    --loop
```

关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_dir` | — | LeRobot 数据集根目录（必填） |
| `--task_config` | — | 场景配置 YAML（必填） |
| `--episode` | 全部 | 仅回放指定集（1-indexed） |
| `--loop` | 否 | 循环回放，播完全部后从头再来 |
| `--steps_per_frame` | 10 | 每帧保持的 5ms 控制步数，10≈50ms（约 1.5x 实时），传 7 接近实时 |

#### 轻量回放（`replay_lerobot.py`）

直接读 parquet、无 manager 事件循环，适合快速核查轨迹数据。支持 openloong / tiangong2。

```bash
python replay_lerobot.py \
    --dataset_dir /path/to/lerobot_data/shop_openloong \
    --episode 1 \
    --agent_name openloong \
    --task_config example.yaml
```

## VR 操作（遥操作脚本）

| 操作 | 按键 |
|------|------|
| 开始 / 结束采集 | 左手柄 **Grip 侧握键**（中指握住侧面那颗键） |
| 左臂移动 | 左手柄 Transform（持握激活） |
| 右臂移动 | 右手柄 Transform（持握激活） |
| 左夹爪 | X / Y / 左扳机 |
| 右夹爪 | A / B / 右扳机 |
| 停止程序 | 终端 `Ctrl+C` |

任务成功 → 自动保存；失败 → 丢弃本集。

## 相机说明

- **默认（websocket）**：`begin_save_video` 触发 OrcaStudio 推流，脚本通过 7070/7080/7090 取帧
- **备用（mp4）**：`--camera_source mp4`，每集由 OrcaStudio 录 MP4，集末批量提取
- 减少路数：`--cameras head`（默认 `head,wrist_l,wrist_r`）
- 客户端缩放：`--cam_resolution 480x640`（不改变 OrcaStudio 渲染分辨率）

## 目录结构

```
dataCollection/
├── README.md                          # 本文件
├── requirements-lerobot.txt           # LeRobot 依赖（版本对齐 orcalab_lerobot）
├── environment.yml                    # conda 环境配置
├── example.yaml                       # 超市场景任务配置
├── scripted-example.yaml              # 脚本化场景配置
├── dataset/                           # HDF5 采集输出（自动生成）
├── _lerobot_scratch/                  # LeRobot 暂存（bench、mp4 等）
├── logs/                              # 运行日志
├── data_collection_tele.py              # HDF5 遥操作
├── data_collection_tele_lerobot.py    # LeRobot 遥操作
├── data_collection_scripted_lerobot.py
├── data_collection_replay_lerobot.py  # LeRobot manager 模式回放
├── eval_lerobot.py
└── replay_lerobot.py                  # LeRobot 轻量回放验证
```

## 常见问题

**Pico 显示「无客户端连接」**
→ 检查 `adb reverse tcp:8001 tcp:8001` 是否执行，App 是否已启动。

**相机端口连接失败**
→ 确认 OrcaStudio 已渲染且推流端口 7070/7080/7090 在监听；或改用 `--camera_source mp4`。

**采了很多集只保存少量**
→ 仅任务成功的 episode 会写入数据集，失败集自动丢弃。

**机械臂动作偏快 / 视频被压缩**
→ VR 遥操作使用 `--clock wall --fps 20`；若日志提示欠采，降低 `--fps`。
