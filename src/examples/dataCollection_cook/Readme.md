# 机器人抓面包场景 运行指南

本文档说明如何在 **OrcaManipulation 仓库内**运行 机器人抓面包场景。

## 目录

- [1. 目录结构与介绍](#目录结构与介绍)
- [2. 运行流程](#运行流程)
  - [2.1 启动 OrcaStudio 并进入 Play](#启动-orcastudio-并进入-play)
  - [2.2 激活 conda 环境](#激活-conda-环境)
  - [2.3 安装依赖](#安装依赖)
  - [2.4 运行三进程联调](#运行三进程联调)
    - [2.4.1 完整命令](#完整命令)
    - [2.4.2 运行参数（Config.json run 段）](#运行参数configjson-run-段)

---

## 1. 目录结构与介绍

| 文件 | 说明 |
|------|------|
| `RunCollection.py` | 三进程联调入口（组装 `P23cParams` → 调 `envs.softbody.Start`） |
| `Config.json` | **单配置**：布料 sim + 场景资产路径规则 + XPBD target/版本（`xpbd_default_target` / `xpbd_default_version`） |
| `requirements.txt` | 本示例依赖声明（固定 orca-xpbd / orca-link 版本） |
| `logs/` | 运行时日志输出（session config / tele / orcalink / xpbd） |

`RunCollection.py` 内部 `Start` 自动完成四段：

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

#### 2.4.1 完整命令

```bash
cd OrcaManipulation/src/examples/dataCollection_cook

python3 RunCollection.py
```

#### 2.4.2 运行参数（Config.json `run` 段）

运行参数统一在 `Config.json` 的 `run` 段配置，无需再设环境变量。参数含义如下：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `level` | `Life_Kitchen_01_with_utree` | 关卡名 |
| `agent` | `g1_omnipicker` | 机器人种类：`openloong` / `g1_omnipicker` / `tiangong2` |
| `mjc_prefix` | `g1_cook2_usda` | MuJoCo agent 前缀（MJCF body 前缀） |
| `collect_data` | `false` | 是否采集数据集 / HDF5 |
| `xpbd_ui` | `true` | 是否显示 XPBD OpenGL 窗口 |
| `cloth_sync_studio_vis` | `true` | 是否推送 qpos 到 Studio 视口 |
| `cloth_no_realtime` | `false` | 是否不按 macro_dt 墙钟 sleep，尽快跑完 |
| `max_macro_frames` | `20000` | 单回合宏步上限 |
| `max_sec` | `120` | 单回合时长上限（秒） |
| `pbd_grpc_sbt_rotation` | `zup_yflip` | XPBD→Studio 布顶点 gRPC 的 SBT 旋转 |
| `mjc_pbd_cloth_scale` | `0.05` | XPBD 布缩放（bunny.vtk≈3.07 vs O3DE≈0.03） |
| `orcagym_port` | `50051` | OrcaGym gRPC 端口 |
| `pbd_grpc_port` | `50263` | PBDRender gRPC 端口 |
| `orcalink_port` | `50361` | OrcaLink Server 端口 |
| `pico_port` | `8001` | Pico 手柄 TCP 端口 |
| `wait_sec` | `180` | 等端口就绪超时（秒） |
| `kill_stale` | `"1"` | 是否清理陈旧联调进程（`"1"`/`"0"`） |
| `skip_stale_kill` | `false` | 是否跳过陈旧进程清理 |
| `auto_start_studio` | `false` | 是否自动拉起 Studio |
| `xpbd_auto_build` | `true` | 是否自动准备 XPBD 二进制 |
| `xpbd_build_target` | `""` | XPBD target 二进制名；空则用顶层 `xpbd_default_target` |
| `mujoco_viewer` | `"0"` | 是否打开 MuJoCo 原生 viewer（`"1"`/`"0"`） |
| `gui` | `"0"` | 同 `mujoco_viewer` 的别名 |
| `bench_json` | `""` | 宏步逐帧计时输出 JSON 路径（空则关闭） |

> 顶层还有两个 XPBD 默认字段（不在 `run` 段）：`xpbd_default_target`（默认 target 二进制名）、`xpbd_default_version`（orca-xpbd pip 版本）。

> 如需临时覆盖而不改 Config.json，仍可用同名环境变量（如 `AGENT=openloong`），优先级：env > Config.json `run` 段。`CFG` / `CLOTH_CONFIG` 仍为仅环境变量参数（指定 config 文件路径）。
