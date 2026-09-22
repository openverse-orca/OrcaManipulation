# Cloth Robot P23c 运行指南

本文档说明如何在 **OrcaManipulation 仓库内**运行 布料遥操（Cloth Robot P23c 三进程联调）场景。

## 目录

- [1. 目录结构与介绍](#1-目录结构与介绍)
- [2. 运行流程](#2-运行流程)
  - [2.1 前置准备 克隆依赖仓库](#21-前置准备-克隆依赖仓库)
  - [2.2 启动 OrcaStudio / OrcaLab 并进入 Play](#22-启动-orcastudio--orcalab-并进入-play)
  - [2.3 激活 conda 环境](#23-激活-conda-环境)
  - [2.4 安装依赖](#24-安装依赖)
  - [2.5 运行三进程联调](#25-运行三进程联调)
- [3. 调整参数参考](#3-调整参数参考)
  - [3.1 运行参数（环境变量）](#31-运行参数环境变量)
  - [3.2 布料 / 耦合参数（cloth_sim_config）](#32-布料--耦合参数cloth_sim_config)

---

## 1. 目录结构与介绍

| 文件 | 说明 |
|------|------|
| `data_collection_cloth_tele.py` | **主入口**（布料遥操 / 轨迹回放）；`--cloth-coupling` 挂载布料耦合 |
| `RunCloth/run_cloth_robot_p23c.sh` | **三进程联调一键脚本**（refresh session → export scene → 起 tele → 自动拉起 OrcaLink + XPBD） |
| `data_collection_fluid_tele.py` | 流体遥操入口（SPH 链路，另一条，不参与布料） |
| `generate_pico_replay_data.py` | 轨迹关键帧 → Pico JSON（回放数据） |
| `generate_cloth_robot_replay_data.py` | 生成 cloth robot 回放数据 |
| `cloth_replay_paths.py` | 回放路径配置 |
| `RunCloth/` | 联调辅助脚本（config 解析 / XPBD 构建 / 运行时宿主探测 / 相位校验） |
| `analyze/` | 联调分析脚本（debug session / 夹爪闭合 / SBT 旋转离线诊断） |
| `Cloth_Robot_DATA_FLOW.md` | 数据传输链路文档（三通道 + 频率分层） |
| `logs/` | 运行时日志输出（tele / orcalink / xpbd） |

`RunCloth/run_cloth_robot_p23c.sh` 内部自动完成四段：

1. **解析环境 / 配置**：LEVEL / AGENT / MJC_PREFIX，以及 cloth_sim_config（`resolve_cloth_config.sh`，按 level+agent 解析）。
2. **进程 / 端口就绪**：清旧进程、确保 Studio/OrcaLab、等 OrcaGym / PBDRender 端口。
3. **同步 XPBD session**：从 Studio MJCF refresh session + export scene。
4. **启动遥操**：起 `data_collection_cloth_tele.py --cloth-coupling`（自动拉起 OrcaLink + XPBD + ClothOrcaLinkBridge）。

> 数据流（三条独立通道）：**A 刚体物理** MJC → OrcaLink ch20 → XPBD；**B 布料显示** XPBD → Studio PBDRender `:50261`；**C 刚体显示** MJC → Studio OrcaGym `:50051`。详见 `Cloth_Robot_DATA_FLOW.md`。

---

## 2. 运行流程

```
2.1 克隆依赖仓库到同级目录（仅一次）
        ↓
2.2 启动 OrcaStudio / OrcaLab 并进入 Play（手动，仅一次）
        ↓
2.3 激活 conda 环境
        ↓
2.4 安装依赖
        ↓
2.5 运行 bash RunCloth/run_cloth_robot_p23c.sh
```

> 注意：**必须先启动 Studio / OrcaLab 并 Play**，否则 2.5 会卡在
> 「等待 OrcaGym :50051 / PBDRender :50261」超时。

### 2.1 前置准备 克隆依赖仓库

> 一次性操作。联调需要 **OrcaGym / OrcaLink / OrcaPlayground / XPBD** 与 OrcaManipulation 放在**同一级目录**（即 OrcaManipulation 的上一级目录下）：

```
<REPO_ROOT>/                 # OrcaManipulation 的上一级目录（如 ~/Development）
├── OrcaGym/                 # ← 克隆到这里（同级）
├── OrcaLink/
├── OrcaPlayground/
├── XPBD/
└── OrcaManipulation/
```

```bash
cd <REPO_ROOT>                                          # OrcaManipulation 的上一级目录
# OrcaGym 走 LFS skip（GitHub LFS 配额问题，见 2026-09-21 排查），其余正常 clone
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:openverse-orca/OrcaGym.git && (cd OrcaGym && git checkout dev)
git clone git@github.com:openverse-orca/OrcaLink.git && (cd OrcaLink && git checkout Dev)
git clone git@github.com:openverse-orca/OrcaPlayground.git && (cd OrcaPlayground && git checkout dev)
git clone git@github.com:openverse-orca/XPBD.git && (cd XPBD && git checkout XPBD_mujoco312)
```

> `<REPO_ROOT>` 即 `run_cloth_robot_p23c.sh` 里 `$(dirname "$0")/../../../../..` 定位的目录，也可用环境变量 `REPO_ROOT` 覆盖。

### 2.2 启动 OrcaStudio / OrcaLab 并进入 Play

#### 方式 A：启动 OrcaLab

```bash
cd <REPO_ROOT>/OrcaLab_2409
./OrcaLab.sh --scene NursingHome
```

**OrcaLab 场景准备（G1 遥操布料）**

- 机器人资产实例命名为 **`g1_omnipicker_usda`**
- 将该实例拖到左侧大纲的 **`Group`** 下，再 **Play**

#### 方式 B：启动 OrcaStudio

```bash
cd <REPO_ROOT>/OrcaStudio_2409/build
./OrcaEditor
```

在界面中打开关卡（如 `NursingHome`），点击 **Play**。

确认以下端口已监听：

```bash
ss -tlnp | grep -E "50051|50261"
# OrcaGym  :50051
# PBDRender:50261
```

> 布料 gRPC 是 PBDRender Gem **:50261**（不是 MultiPhysicsRender 的 SoftBody :50263）。没点 Play 端口不会起来。

### 2.3 激活 conda 环境

联调 Python 必须是 conda **orca-apr24**（OrcaGym 依赖，`resolve_orca_conda_python.sh` 会定位）：

```bash
# 如果 conda 未初始化，先 source
source ~/miniconda3/etc/profile.d/conda.sh   # 或 /opt/conda/etc/profile.d/conda.sh

# 创建环境（如果不存在）
CONDA_ENV_NAME="orca-apr24"
if ! conda env list | grep -qE "^\s*${CONDA_ENV_NAME}\s"; then
    conda create -n ${CONDA_ENV_NAME} python=3.12 -y
fi

# 激活环境
conda activate ${CONDA_ENV_NAME}
```

### 2.4 安装依赖

```bash
# 确认 conda 环境已激活
conda activate ${CONDA_ENV_NAME}

pip install -r src/examples/dataCollection_cloth/requirements.txt

# 验证安装
pip list | grep -E "orca-gym|orca-link|orca-xpbd"
```

预期输出：

```text
orca-gym            26.x.x
orca-link           26.9.1.2
orca-xpbd           26.9.1.2
```

> `requirements.txt` 已内置两个源（官方 PyPI 主源 + test PyPI 次源，`orca-*` 包发布在 test.pypi.org），并通过 `-r ../../../requirements.txt` 引用顶层通用依赖。
>
> 若用 XPBD 源码二进制（推荐联调时，`ORCAXPBD_USE_PIP_PACKAGE=0`），`run_cloth_robot_p23c.sh` 会经 `ensure_xpbd_pip.py` 自动准备/编译 XPBD，不依赖 pip 包。

### 2.5 运行三进程联调

```bash
# 确认 conda 环境已激活
conda activate ${CONDA_ENV_NAME}

# 运行联调脚本（自动完成：编译 XPBD → 刷新 session → 生成 replay → 启动 tele）
cd <REPO_ROOT>/OrcaManipulation/src/examples/dataCollection_cloth/RunCloth
bash run_cloth_robot_p23c.sh
```

#### 2.5.1 完整命令

```bash
cd <REPO_ROOT>/OrcaManipulation/src/examples/dataCollection_cloth/RunCloth

# 回放（第一次先跑）
REPLAY=1 CLOTH_NO_REALTIME=1 XPBD_UI=1 MAX_MACRO_FRAMES=800 \
LEVEL=NursingHome AGENT=g1_omnipicker MJC_PREFIX=g1_omnipicker_usda \
ORCAXPBD_USE_PIP_PACKAGE=0 \
bash run_cloth_robot_p23c.sh

# PICO 真遥操
REPLAY=0 CLOTH_NO_REALTIME=0 MAX_MACRO_FRAMES=20000 \
LEVEL=NursingHome AGENT=g1_omnipicker MJC_PREFIX=g1_omnipicker_usda \
ORCAXPBD_USE_PIP_PACKAGE=0 \
bash run_cloth_robot_p23c.sh
```

#### 2.5.2 运行参数（环境变量）

运行参数通过**环境变量**传入（也可写成 `export` 后跑脚本）。常用参数如下：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LEVEL` | `NursingHome` | 关卡名（省略时由 `detect_studio_level.sh` 自动解析） |
| `AGENT` | `openloong` | Agent 名称：`openloong` / `g1_omnipicker` / `tiangong2` |
| `MJC_PREFIX` | `openloong_gripper_2f85_fix_base_usda` | MJCF agent 前缀（g1_omnipicker 用 `g1_omnipicker_usda`） |
| `DEBUG` | `0` | `1` 开启 debug 模式（CSV + 采集 + 自动分析） |
| `XPBD_RELEASE_BUILD` | `1` | `0` 用 Debug 编译的 XPBD |
| `CLOTH_DEBUG` | `0` | `1` 开启 cloth debug CSV 输出 |
| `COLLECT_DATA` | `0` | `1` 采集数据集 / HDF5 |
| `REPLAY` | `1` | `1` 回放；`0` 听 PICO 手柄 |
| `CLOTH_NO_REALTIME` | `1` | `1` 尽快跑完不等实时；`0` 按墙钟跟手 |
| `XPBD_UI` | `1` | `0` 关闭 XPBD OpenGL 窗口 |
| `CLOTH_SYNC_STUDIO_VIS` | `1` | `0` 关闭 Studio 刚体跟随（replay 推 qpos） |
| `MAX_MACRO_FRAMES` | `800` | 单回合宏步上限 |
| `MAX_SEC` | `120` | 单回合时长上限（秒） |
| `PBD_GRPC_SBT_ROTATION` | `from_quat` | XPBD→Studio 刚体旋转传递方式（`zup_yflip` / `from_quat`） |
| `ORCAXPBD_USE_PIP_PACKAGE` | `0` | `1` 用 pip 包而非本地编译 XPBD |
| `CLOTH_HOST` | `auto` | 运行时宿主：`auto` / `studio` / `orcalab` |
| `AUTO_START_STUDIO` | `0` | `1` 先自动启动 Studio 再联调 |
| `PBD_GRPC_PORT` | `50261` | PBDRender gRPC 端口 |
| `ORCALINK_PORT` | `50361` | OrcaLink Server 端口 |
| `PICO_PORT` | `8001` | Pico 手柄 TCP 端口 |
| `WAIT_SEC` | `180` | 等端口就绪超时（秒） |
| `KILL_STALE` | `1` | 是否清理陈旧联调进程；`SKIP_STALE_KILL=1` 跳过 |

> 端口约定：`8001`(PICO→PicoJoystick)、`50051`(OrcaGym↔Studio)、`50361`(OrcaLink)、`50261`(XPBD→PBDRender)。

> 回放 vs 真遥操关键差异：`REPLAY`（`1` 回放 / `0` 听手柄）、`CLOTH_NO_REALTIME`（`1` 不等实时 / `0` 跟手）、`MAX_MACRO_FRAMES`（回放 `800` / 遥操 `20000`）。

---

## 3. 调整参数参考

参数分两类：**运行参数**（环境变量，见 2.5.2）、**布料 / 耦合参数**（`cloth_sim_config.*.json`，物理与刚性映射）。

### 3.1 运行参数（环境变量）

见 [2.5.2](#252-运行参数环境变量)。调「回放 vs 遥操」「XPBD 窗口显隐」「实时同步」都在这里。

### 3.2 布料 / 耦合参数（cloth_sim_config）

布料物理与 MjcPBD 耦合参数在 `cloth_sim_config.*.json`（`OrcaPlayground/examples/cloth_3d/` 下），由 `resolve_cloth_config.sh` 按 `LEVEL` + `AGENT` 解析（默认 `cloth_sim_config.orcagym_e2e.json`，继承 `dual_gripper_cross_full.json`）。

| 段 | 内容 | 说明 |
|---|---|---|
| `mujoco` | `env_id` / `frame_skip` / `timestep` / `macro_dt` | MuJoCo 子步与宏步对齐（`--frame-skip` 须与 `mujoco.frame_skip` 一致） |
| `frame_count` | `mujoco_substeps_per_macro_frame` / `xpbd_substeps_per_macro_frame` | 两侧宏步子步对齐 |
| `orcalink` | `client.session`（session 102、`expected_clients: 2`）/ `channels` | OrcaLink 中继与 sync 握手 |
| `particle_render` | `enabled` / `grpc_address`（默认 `localhost:50261`） | 布料 gRPC 推顶点开关与地址 |
| `rigid_body_map` | 7 刚体 body 映射（base + 双爪掌/指） | XPBD 刚体跟踪的 body 名单 |
| `cloth` | 布料网格 / 物理参数 | 布料形变与接触求解 |

> 更多数据流细节（三通道、频率分层、单宏步时序）见 `Cloth_Robot_DATA_FLOW.md`。