# OrcaManipulation

本仓库基于 OrcaLab / OrcaGym，提供人形机器人数据采集、回放与在线推理工具，支持以下两类机器人：

| 机器人 | 支持场景 | 采集文档 | 推理文档 |
|--------|----------|----------|----------|
| 智元 G1 OmniPicker | 四色按按钮、工具整理（脚本化 + Pico 遥操作） | [docs/g1_omnipicker_collection.md](docs/g1_omnipicker_collection.md) | [docs/g1_omnipicker_inference.md](docs/g1_omnipicker_inference.md) |
| 宇树 G1 | 四色按按钮（TeleVuer 遥操作 / 脚本化） | [docs/unitree_g1_collection.md](docs/unitree_g1_collection.md) | [docs/unitree_g1_inference.md](docs/unitree_g1_inference.md) |

请按上表进入对应文档查看完整流程。下文说明两套机器人共用的环境配置与数据格式。

如需基于本仓库采集的数据进行策略训练或部署推理服务，请参阅 [docs/openpi_deployment.md](docs/openpi_deployment.md)（RTC 异步推理为其中可选章节）。

---

## 前置：系统依赖与资产订阅

**1. 安装系统依赖**

```bash
sudo apt install -y libvdpau1
```

OrcaLab 首次启动时会自行补齐它缺少的 Qt 平台库，其中 `libxcb-cursor0` 会以普通用户权限下载并放到 PySide6 目录旁，无需 root；但 `libvdpau1` 走的是 `sudo apt install`，先装好可以避免首次启动卡在密码提示上。

**2. 订阅场景资产**

OrcaLab 7.1 本身无需单独下载安装包，它会随下文的 Conda 环境一起装好（见「环境安装」）。环境装好后启动 `orcalab`，登录 OrcaLab 资产库并订阅以下三个资产包（每个资产在打开场景时会自动加载）：

- `SouthGrid_Competition_2026`
- `G1_omnipicker`
- `G1_Picker_SouthGrid`

若场景无法加载或物体不显示，请先检查以上资产包是否均已订阅。

---

## 获取代码

```bash
git clone -b g1_lerobot https://github.com/openverse-orca/OrcaManipulation.git
cd OrcaManipulation
```

克隆后，请确认仓库根目录下存在以下文件和目录，以核对本仓库已完整获取：

```
environment-unitree.yml
requirements.txt
scripts/
third_party/
src/
```

---

## 兼容环境

以下为经过完整验证的运行环境。请确认运行本项目的主机满足下表全部要求后再继续安装。

| 组件 | 版本 / 要求 |
|------|-------------|
| OrcaLab / OrcaGym | **26.7.1 / 26.7.1**（即 OrcaLab 7.1；两者版本号必须一致，由 `install_runtime.sh` 安装） |
| Python | **3.12.13** |
| NumPy / SciPy | 2.4.6 / 1.15.3（由 Conda 管理，不可用 pip 覆盖） |
| Pinocchio / CasADi | 3.9.0 / 3.7.2（由 Conda 管理，不可用 pip 覆盖） |
| Gymnasium / MuJoCo | 1.2.1 / 3.7.0 |
| PyTorch | 2.7.1+cpu，torchvision 0.22.1+cpu |
| LeRobot / TeleVuer / OpenPI client | 仓库内 `third_party/` 固定源码 |
| 宇树 IK 模型 | 仓库内 `src/examples/dataCollection/assets/g1/` |
| GPU 视频编码 | **NVIDIA Ada-Lovelace 架构（RTX 40 系）及以上**，需安装对应驱动；RTX 30 系及更早显卡不支持 AV1 NVENC，无法使用视频录制功能 |
| ADB（仅 TeleVuer 遥操作） | 用于 Pico 头显端口转发；安装方式：`sudo apt install adb`（Debian/Ubuntu），或从 [Android Platform Tools](https://developer.android.com/tools/releases/platform-tools) 下载 |

---

## 环境安装

请在运行本项目的主机上新建 Conda 环境；请勿在已有环境上执行 `conda env update`，也不要将运行本项目的主机上其他环境的 `site-packages` 目录复制过来：

```bash
cd /path/to/OrcaManipulation
conda env create -f environment-unitree.yml
conda activate orcalab_lerobot
bash scripts/install_runtime.sh
```

`install_runtime.sh` 会先验证 Conda 管理的数值库版本，再按哈希锁安装 pip 依赖；随后安装 `orca-gym==26.7.1` 和 `orca-lab==26.7.1`；最后从仓库内 `third_party/` 安装 LeRobot、TeleVuer 和 OpenPI client，并执行数据集写入、编码器、GUI 与资产路径自检。

OrcaLab 桌面端就是 `orca-lab` 这个 Python 包，因此上述命令跑完即安装完毕，无需再从官网下载安装包。启动方式：

```bash
conda activate orcalab_lerobot
orcalab
```

**必须先 `conda activate` 再运行 `orcalab`**，不要用 `/path/to/envs/orcalab_lerobot/bin/orcalab` 这类绝对路径直接调用。OrcaLab 的 PySide6 补丁逻辑会通过 `PATH` 里的 `python3` 定位 Qt 库，绕过 activate 会让它找到系统 Python 并在启动时报错。

以下命令**禁止单独执行**：

```bash
# 禁止：
pip install pin
pip install pinocchio
pip install casadi
pip install -r requirements.txt   # 不要单独运行，应通过 install_runtime.sh 调用
```

安装完成后，请在仓库根目录运行统一复检命令，确认环境正常：

```bash
python scripts/verify_environment.py
```

该命令会验证所有关键包版本、Pinocchio 来源、OrcaLab 与 OrcaGym 的版本一致性、相机库导入链路、AV1 NVENC GPU 编码能力以及 Unitree G1 IK 模型完整性。全部通过后打印 `Environment verification OK`。

宇树 IK 所需的 G1 描述文件（`g1_body29_hand14.urdf` 与它引用的 49 个 STL）
已随仓库提供，来源为 Unitree `unitree_ros` 的 `g1_description`（BSD-3-Clause，
见该目录下的 `LICENSE`）。该目录中的文件请勿删减或改名。

---

## 目录结构

```text
OrcaManipulation/
├── README.md
├── requirements.txt              # 含哈希的 pip 锁（安装时使用）
├── requirements.in               # pip 顶层输入（维护锁时使用，安装时无需关注）
├── constraints.txt               # 传递依赖锁定输入（维护锁时使用）
├── environment-unitree.yml       # Conda 数值库精确构建
├── scripts/                      # 安装与环境验证
├── third_party/
│   ├── lerobot/                  # LeRobot 数据集运行时
│   ├── televuer/                 # TeleVuer XR 运行时
│   ├── openpi-client/            # OpenPI 推理客户端
│   └── openpi-rtc/               # OpenPI RTC 异步推理扩展
├── docs/
│   ├── g1_omnipicker_collection.md   # 智元 · 采集
│   ├── g1_omnipicker_inference.md    # 智元 · 推理
│   ├── unitree_g1_collection.md      # 宇树 · 采集
│   ├── unitree_g1_inference.md       # 宇树 · 推理
│   └── openpi_deployment.md          # OpenPI 完整部署（训练 / 推理服务 / 可选 RTC）
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
    ├── utils/
    └── examples/
        ├── dataCollection/
        │   ├── assets/g1/              # 宇树 G1 URDF + STL（IK 必需）
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
| `g1_omnipicker/g1_omnipicker_collection_tele_lerobot.py` | 智元 | Pico VR 手柄遥操作采集 |
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
    ├── observation.images.cam_wrist_l/    # 部分任务会包含左腕相机
    └── observation.images.cam_wrist_r/
```

- 默认相机分辨率为 480×640，默认帧率为 20 FPS。
- 视频编码为 MP4（`av1_nvenc`），需要 Ada-Lovelace 架构 GPU。
- `action` 与 `observation.state` 的维度及字段含义因机器人而异，详见各机器人文档。
