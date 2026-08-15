# Cloth Robot P23c 运行指南（Python 编排）

本文档说明如何在 **OrcaManipulation 仓库内**运行软体三进程联调（P23c）。

## 目录

- [1. 目录结构与介绍](#目录结构与介绍)
- [2. 运行流程](#运行流程)
  - [2.1 启动 OrcaStudio 并进入 Play](#启动-orcastudio-并进入-play)
  - [2.2 激活 conda 环境](#激活-conda-环境)
  - [2.3 安装依赖](#安装依赖)
  - [2.4 运行三进程联调](#运行三进程联调)
    - [2.4.1 完整命令（默认 G1 Cook 布料遥操）](#完整命令默认-g1-cook-布料遥操)
    - [2.4.2 常用环境变量](#常用环境变量)

---

## 1. 目录结构与介绍

| 文件 | 说明 |
|------|------|
| `RunSim.py` | 三进程联调编排入口（解析配置 → 进程就绪 → session → 启动 tele） |
| `data_collection_cloth_tele.py` | 布料遥操/回放主程序（由 `RunSim.py` 自动拉起） |
| `requirements.txt` | 本示例依赖声明（引用顶层 + 固定 orca-xpbd/orca-link 版本） |
| `xpbd_scene.conf` | 遗留文件：XPBD target/版本集中配置。当前 `RunSim.py` **不读取**它，target/版本默认值在 `envs/softbody/ProcessXPBD.py`（`DEFAULT_TARGET` / `DEFAULT_VERSION`），可用 `XPBD_BUILD_TARGET` 环境变量覆盖 |
| `logs/` | 运行时日志输出（tele / orcalink / xpbd） |
| `dataset/` | 采集产物（`COLLECT_DATA=1` 时写入，gitignore） |

`RunSim.py` 自动完成四段：

1. **解析环境/配置**：LEVEL / AGENT / MJC_PREFIX / CFG 与 XPBD 二进制；
   AGENT 与 MJC_PREFIX 会先从最新 Studio MJCF 自动扫描（`scan_tele_layout_from_mjcf`），
   未显式设置时用扫描结果覆盖默认值。
2. **进程/端口就绪**：清旧进程、确保 Studio、等 OrcaGym / PBDRender 端口。
3. **同步 XPBD session**：从 Studio MJCF refresh session + export scene。
4. **启动 `data_collection_cloth_tele.py`**（OrcaLink + XPBD + bridge 遥操/回放）。

---

## 2. 运行流程

```
2.1 启动 OrcaStudio 并进入 Play（手动，仅一次）
        ↓
2.2 激活 conda 环境
        ↓
2.3 安装依赖（requirements.txt）
        ↓
2.4 运行 python RunSim.py
```

> 注意：**必须先启动 Studio 并 Play**，否则 2.4 会卡在
> 「等待 OrcaGym :50051 / PBDRender :50263」超时。

### 2.1 启动 OrcaStudio 并进入 Play

在宿主机启动 OrcaStudio（OrcaEditor），打开关卡（如 `Life_Kitchen_01_with_utree`），点击 **Play**。

确认以下端口已监听：

```bash
ss -tlnp | grep -E "50051|50263"
# OrcaGym  :50051
# PBDRender:50263
```

### 2.2 激活 conda 环境

`RunSim.py` 默认用当前解释器（`sys.executable`），在已激活的 conda 环境下直接运行即可。
也可用 `PYTHON=/path/to/python` 显式指定（须为单个可执行文件路径）。

```bash
conda activate <你的环境名>
```

### 2.3 安装依赖

```bash
# 在本示例目录下安装依赖（引用顶层 requirements + 声明 orca-xpbd/orca-link 版本）
pip install -r src/examples/dataCollection_cook/requirements.txt \
    --index-url https://pypi.org/simple/ \
    --extra-index-url https://test.pypi.org/simple/
```

`requirements.txt` 固定两个发布包版本（`orca-xpbd` / `orca-link`），
版本号与 `envs/softbody/ProcessXPBD.py` 的 `DEFAULT_VERSION` 一致。

### 2.4 运行三进程联调

```bash
cd <REPO_ROOT>/OrcaManipulation/src/examples/dataCollection_cook

python RunSim.py
```

#### 2.4.1 完整命令（默认 G1 Cook 布料遥操）

先设置环境变量、再调用 `RunSim.py`。

```bash
cd OrcaManipulation/src/examples/dataCollection_cook

export DEBUG=0
export CLOTH_DEBUG=0
export COLLECT_DATA=0
export CLOTH_NO_REALTIME=0
export XPBD_UI=1
export MAX_MACRO_FRAMES=20000
export PBD_GRPC_SBT_ROTATION=zup_yflip
# bunny.vtk ≈3.07 单位 vs O3DE 场景 bunny ≈0.03：缩 XPBD 布到 O3DE 尺寸
export MJC_PBD_CLOTH_SCALE=0.05
export LEVEL=Life_Kitchen_01_with_utree
export AGENT=g1_omnipicker
export MJC_PREFIX=g1_cook2_usda

python3 RunSim.py
```

`CLOTH_CONFIG` 为可选：仅当 G1 Cook 配置
（`PriWaic/examples/Kitchen_Test0729/configs/cloth_sim_config.g1_cook.json`）存在时
才由脚本拷入并导出，否则走 `RunSim.py` 按 level+agent 的默认解析。

#### 2.4.2 常用环境变量

> 下表为 `RunSim.py` 内部默认值；P23c 实际联调会由 上方「完整命令」显式覆盖

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LEVEL` / `ORCA_LEVEL_NAME` | 自动检测 | 关卡名（如 `Life_Kitchen_01_with_utree`） |
| `AGENT` | `openloong` | 机器人种类：`openloong` / `tiangong2` / `g1_omnipicker` |
| `MJC_PREFIX` | `openloong_gripper_2f85_fix_base_usda` | MuJoCo agent 前缀 |
| `CFG` / `CLOTH_CONFIG` | 按 level+agent 解析 | cloth_sim JSON 路径 |
| `DEBUG` | `0` | 联动下面几个 debug 默认值 |
| `CLOTH_DEBUG` | `0` | 全量 CSV + XPBD 刚体/布料跟踪 |
| `COLLECT_DATA` | `0` | 设为 `1` 采集数据集/HDF5 |
| `XPBD_UI` / `SHOW_UI` | `1` | 设为 `0` 关闭 XPBD OpenGL 窗口 |
| `CLOTH_SYNC_STUDIO_VIS` | `1` | 推送 qpos 到 Studio 视口 |
| `CLOTH_NO_REALTIME` | `1` | 不按 macro_dt 墙钟 sleep，尽快跑完 |
| `MAX_MACRO_FRAMES` / `MAX_SEC` | `800` / `120` | 单回合时长上限 |
| `ORCAGYM_PORT` | `50051` | OrcaGym gRPC 端口 |
| `PBD_GRPC_PORT` | `50263` | PBDRender gRPC 端口 |
| `ORCALINK_PORT` | `50361` | OrcaLink Server 端口 |
| `PICO_PORT` | `8001` | Pico 手柄 TCP 端口 |
| `WAIT_SEC` | `180` | 等端口就绪超时（秒） |
| `KILL_STALE` / `SKIP_STALE_KILL` | `1` / `0` | 是否清理陈旧联调进程 |
| `AUTO_START_STUDIO` | `0` | 是否自动拉起 Studio |
| `XPBD_AUTO_BUILD` / `XPBD_BUILD_TARGET` | `1` / `dual_gripper_g1_cook2` | XPBD 二进制准备与 target |
| `PBD_GRPC_SBT_ROTATION` | `zup_yflip` | XPBD→Studio 布顶点 gRPC 的 SBT 旋转（透传给 XPBD） |
| `MJC_PBD_CLOTH_SCALE` | `0.05` | XPBD 布缩放：bunny.vtk≈3.07 vs O3DE≈0.03（透传给 XPBD） |
| `MUJOCO_VIEWER` / `GUI` | `0` | 设为 `1` 打开 MuJoCo 原生 viewer |
| `BENCH_JSON` | 空 | 宏步逐帧计时输出 JSON |
