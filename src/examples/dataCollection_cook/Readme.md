# 机器人抓面包场景 运行指南

本文档说明如何在 **OrcaManipulation 仓库内**运行 机器人抓面包场景。

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
| `RunCollection.py` | 三进程联调入口（组装 `P23cParams` → 调 `envs.softbody.attach_coupling.run_p23c`） |
| `Config.json` | **单配置**：布料 sim + 场景资产路径规则 + XPBD target/版本（`xpbd_default_target` / `xpbd_default_version`） |
| `requirements.txt` | 本示例依赖声明（固定 orca-xpbd / orca-link 版本） |
| `logs/` | 运行时日志输出（session config / tele / orcalink / xpbd） |

`RunCollection.py` 内部 `run_p23c` 自动完成四段：

1. **解析环境/配置**：LEVEL / AGENT / MJC_PREFIX 与 XPBD target/版本（target/版本从 `Config.json` 读，环境变量可覆盖）。
2. **进程/端口就绪**：清旧进程、确保 Studio、等 OrcaGym / PBDRender 端口。
3. **同步 XPBD session**：从 Studio MJCF refresh session + export scene。
4. **启动遥操**（OrcaLink + XPBD + bridge 遥操/回放）。

---

## 2. 运行流程

```
2.1 启动 OrcaStudio 并进入 Play（手动，仅一次）
        ↓
2.2 激活 conda 环境
        ↓
2.3 安装依赖（requirements.txt）
        ↓
2.4 运行 python RunCollection.py
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

`RunCollection.py` 默认用当前解释器（`sys.executable`），在已激活的 conda 环境下直接运行即可。

```bash
conda activate <你的环境名>
```

### 2.3 安装依赖

```bash
pip install -r src/examples/dataCollection_cook/requirements.txt \
    --index-url https://pypi.org/simple/ \
    --extra-index-url https://test.pypi.org/simple/
```

`requirements.txt` 固定两个发布包版本（`orca-xpbd` / `orca-link`），
XPBD target/版本号与 `Config.json` 的 `xpbd_default_target` / `xpbd_default_version` 一致。

### 2.4 运行三进程联调

```bash
cd <REPO_ROOT>/OrcaManipulation/src/examples/dataCollection_cook

python RunCollection.py
```

#### 2.4.1 完整命令（默认 G1 Cook 布料遥操）

```bash
cd OrcaManipulation/src/examples/dataCollection_cook

export DEBUG=0
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

python3 RunCollection.py
```

#### 2.4.2 常用环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LEVEL` / `ORCA_LEVEL_NAME` | 自动检测 | 关卡名（如 `Life_Kitchen_01_with_utree`） |
| `AGENT` | `openloong` | 机器人种类：`openloong` / `g1_omnipicker` |
| `MJC_PREFIX` | 自动检测 | MuJoCo agent 前缀（如 `g1_cook2_usda`） |
| `CFG` / `CLOTH_CONFIG` | `Config.json` | 显式指定 cloth config 路径（默认走 `dataCollection_cook/Config.json`） |
| `DEBUG` | `0` | 联动下面几个 debug 默认值 |
| `COLLECT_DATA` | `0` | 设为 `1` 采集数据集/HDF5 |
| `XPBD_UI` / `SHOW_UI` | `1` | 设为 `0` 关闭 XPBD OpenGL 窗口 |
| `CLOTH_SYNC_STUDIO_VIS` | `1` | 推送 qpos 到 Studio 视口 |
| `CLOTH_NO_REALTIME` | `1` | 不按 macro_dt 墙钟 sleep，尽快跑完 |
| `MAX_MACRO_FRAMES` / `MAX_SEC` | `20000` / `120` | 单回合时长上限 |
| `ORCAGYM_PORT` | `50051` | OrcaGym gRPC 端口 |
| `PBD_GRPC_PORT` | `50263` | PBDRender gRPC 端口 |
| `ORCALINK_PORT` | `50361` | OrcaLink Server 端口 |
| `PICO_PORT` | `8001` | Pico 手柄 TCP 端口 |
| `WAIT_SEC` | `180` | 等端口就绪超时（秒） |
| `KILL_STALE` / `SKIP_STALE_KILL` | `1` / `0` | 是否清理陈旧联调进程 |
| `AUTO_START_STUDIO` | `0` | 是否自动拉起 Studio |
| `XPBD_AUTO_BUILD` | `1` | 是否自动准备 XPBD 二进制 |
| `XPBD_BUILD_TARGET` | `Config.json 的 xpbd_default_target` | XPBD target 二进制名（如 `dual_gripper_g1_cook2`） |
| `ORCA_XPBD_VERSION` | `Config.json 的 xpbd_default_version` | orca-xpbd pip 版本 |
| `PBD_GRPC_SBT_ROTATION` | `zup_yflip` | XPBD→Studio 布顶点 gRPC 的 SBT 旋转 |
| `MJC_PBD_CLOTH_SCALE` | `0.05` | XPBD 布缩放：bunny.vtk≈3.07 vs O3DE≈0.03 |
| `MUJOCO_VIEWER` / `GUI` | `0` | 设为 `1` 打开 MuJoCo 原生 viewer |
| `BENCH_JSON` | 空 | 宏步逐帧计时输出 JSON |
