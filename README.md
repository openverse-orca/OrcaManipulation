# OrcaManipulation

本仓库基于 OrcaLab / OrcaGym，提供人形机器人数据采集、回放与在线推理工具。当前交付覆盖两类机器人：

| 机器人 | 支持场景 | 采集文档 | 推理文档 |
|--------|----------|----------|----------|
| 智元 G1 OmniPicker | 四色按按钮、工具整理（脚本化 + Pico 遥操作） | [docs/g1_omnipicker_collection.md](docs/g1_omnipicker_collection.md) | [docs/g1_omnipicker_inference.md](docs/g1_omnipicker_inference.md) |
| 宇树 G1 | 四色按按钮（TeleVuer 遥操作 / 脚本化） | [docs/unitree_g1_collection.md](docs/unitree_g1_collection.md) | [docs/unitree_g1_inference.md](docs/unitree_g1_inference.md) |

请按上表进入对应文档查看完整流程。下文说明两套机器人共用的环境配置与数据格式。

---

## 兼容环境

| 组件 | 版本 / 要求 |
|------|-------------|
| OrcaLab | **6.3** |
| Python | **3.12 及以上** |
| `orca-gym` | 推荐与 OrcaLab 6.3 配套的 **26.6.x** |
| `lerobot` | **0.3.4**（更低版本缺少所需的数据集接口） |
| `gymnasium` | **1.2.1**（与 `orca-gym` 一致） |
| GPU 视频编码 | NVIDIA GPU，驱动及 PyAV/FFmpeg 需支持 `av1_nvenc` |

---

## 环境安装

```bash
conda activate orcalab_lerobot
pip install -r requirements.txt
```

安装完成后若出现以下类似提示，可以忽略：

```
lerobot 0.3.4 requires gymnasium<1.0.0, but you have gymnasium 1.2.1
```

本项目只使用 lerobot 的数据集读写能力，不依赖其 gymnasium 集成；运行时以 `orca-gym` 所需版本为准。`requirements.txt` 已显式锁定 `gymnasium==1.2.1` 与 `lerobot==0.3.4`，防止 pip 自动降级。若个别环境仍出现降级，请执行：

```bash
pip install lerobot==0.3.4 --no-deps
pip install gymnasium==1.2.1
```

在线推理额外依赖 `openpi_client`，请在运行推理脚本前设置：

```bash
export OPENPI_CLIENT_SRC=<openpi-client源码路径>/src
```

---

## 目录结构

```text
OrcaManipulation/
├── README.md
├── docs/
│   ├── g1_omnipicker_collection.md   # 智元 · 采集
│   ├── g1_omnipicker_inference.md    # 智元 · 推理
│   ├── unitree_g1_collection.md      # 宇树 · 采集
│   └── unitree_g1_inference.md       # 宇树 · 推理
├── requirements.txt
├── pyproject.toml
└── src/
    ├── conf/
    ├── controllers/
    ├── dataCollectionManager/
    ├── dataStorage/
    ├── devices/
    ├── envs/dataCollection/
    ├── scene/
    ├── task/
    └── examples/
        ├── dataCollection/
        │   ├── common/                 # 共用 example.yaml / scripted 基座
        │   ├── g1_omnipicker/          # 智元采集入口与布局
        │   └── unitree_g1/             # 宇树采集入口与布局
        └── inference/
            ├── g1_omnipicker/          # 智元推理入口
            └── unitree_g1/             # 宇树推理入口
```

---

## 入口脚本

### 数据采集（`src/examples/dataCollection/`）

| 脚本 | 机器人 | 用途 |
|------|--------|------|
| `g1_omnipicker/g1_omnipicker_collection_scripted_tool_lerobot.py` | 智元 | 工具整理脚本化自动采集 |
| `g1_omnipicker/g1_omnipicker_collection_scripted_button_lerobot.py` | 智元 | 四色按钮脚本化自动采集 |
| `g1_omnipicker/g1_omnipicker_collection_tele_lerobot.py` | 智元 | Pico VR 遥操作采集（左臂锁定） |
| `g1_omnipicker/g1_omnipicker_replay_lerobot.py` | 智元 | LeRobot Parquet 数据集回放 |
| `unitree_g1/g1_pick_collection_tele_lerobot.py` | 宇树 | TeleVuer 遥操作采集（双臂） |
| `unitree_g1/g1_pick_collection_scripted_button_lerobot.py` | 宇树 | 四色按钮脚本化自动采集 |

### 在线推理（`src/examples/inference/`）

| 脚本 | 机器人 | 用途 |
|------|--------|------|
| `g1_omnipicker/eval_g1_omnipicker_lerobot.py` | 智元 | 按钮任务在线推理（OpenPI） |
| `g1_omnipicker/eval_g1_omnipicker_tool_lerobot.py` | 智元 | 工具任务在线推理（OpenPI） |
| `unitree_g1/eval_g1_pick_lerobot.py` | 宇树 | 按钮任务在线推理（OpenPI） |

---

## 数据集格式

采集结果保存为 LeRobot v2.1 格式：

```text
<dataset_root>/
├── meta/
│   ├── info.json              # 数据集元信息（fps / 维度 / 相机键等）
│   ├── episodes.jsonl         # 每集的 index / length / task
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl            # 语言指令列表
├── data/chunk-000/
│   └── episode_XXXXXX.parquet # action / observation.state / timestamp
└── videos/chunk-000/
    ├── observation.images.cam_head/
    ├── observation.images.cam_wrist_l/    # 智元专有
    └── observation.images.cam_wrist_r/
```

- 默认相机分辨率为 480×640，默认帧率为 20 FPS。
- 视频编码为 MP4（`av1_nvenc`）。
- `action` 与 `observation.state` 的维度及字段含义因机器人而异，详见各机器人文档。
