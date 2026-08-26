# 机器人抓面包场景 运行指南

本文档说明如何在 **OrcaManipulation 仓库内**运行 机器人抓面包场景。

## 目录

- [1. 目录结构与介绍](#目录结构与介绍)
- [2. 运行流程](#运行流程)
  - [2.0 前置准备 克隆 OrcaGym](#前置准备-克隆-orcagym)
  - [2.1 启动 OrcaStudio 并进入 Play](#启动-orcastudio-并进入-play)
  - [2.2 激活 conda 环境](#激活-conda-环境)
  - [2.3 安装依赖](#安装依赖)
  - [2.4 运行三进程联调](#运行三进程联调)
    - [2.4.1 完整命令](#完整命令)
    - [2.4.2 运行参数（Config.json run 段）](#运行参数configjson-run-段)
- [3. 机器人参数（Agent-utree.json）](#机器人参数agent-utreejson)
  - [3.1 Agent-utree.json 结构](#agent-utreejson-结构)
  - [3.2 新增一个机器人](#新增一个机器人)
  - [3.3 手指弯曲调优参数](#33-手指弯曲调优参数)
  - [3.4 抓取锁定参数](#34-抓取锁定参数)

---

## 1. 目录结构与介绍

| 文件 | 说明 |
|------|------|
| `RunCollection.py` | 三进程联调入口（组装 `P23cParams` → 调 `envs.softbody.Start`） |
| `Config.json` | **场景/任务配置**：布料 sim + 场景资产路径规则 + XPBD target/版本；通过 `agent_file` 引用机器人参数 |
| `Agent-utree.json` | **机器人配置**：关节/夹爪/body 匹配（`robot` / `grip_bind` / `xpbd_auto_discover` / `orcagym_rigid_body_map` / `run.agent` / `run.mjc_prefix` 等） |
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
2.0 克隆 OrcaGym 到同级目录（仅一次）
        ↓
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

### 2.0 前置准备 克隆 OrcaGym

> 一次性操作。`RunCollection.py` 启动时会把 OrcaGym 加进 `sys.path`，因此 **OrcaGym 必须与 OrcaManipulation 放在同一级目录**（即 OrcaManipulation 的上一级目录下）：

```
<REPO_ROOT>/                 # OrcaManipulation 的上一级目录（如 ~/Development）
├── OrcaGym/                 # ← 克隆到这里（同级）
└── OrcaManipulation/
```

```bash
cd <REPO_ROOT>                                          # OrcaManipulation 的上一级目录
git clone git@github.com:openverse-orca/OrcaGym.git
cd OrcaGym
git checkout cloth_dev                                  # 切到 cloth_dev 分支
```

> `<REPO_ROOT>` 即 `RunCollection.py` 里 `Path(__file__).resolve().parents[4]` 定位的目录（`.../OrcaManipulation/src/examples/dataCollection_cook/RunCollection.py` 往上 4 层）。也可用环境变量 `REPO_ROOT` 覆盖。

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
| `agent` | `g1_omnipicker` | 机器人种类（**在 `Agent-utree.json` 的 `run` 段**）：`openloong` / `g1_omnipicker` / `tiangong2` |
| `mjc_prefix` | `g1_cook2_usda` | MuJoCo agent 前缀（**在 `Agent-utree.json` 的 `run` 段**） |
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
| `xpbd_auto_build` | `true` | 是否自动准备 XPBD 二进制 |
| `xpbd_build_target` | `""` | XPBD target 二进制名；空则用顶层 `xpbd_default_target` |
| `mujoco_viewer` | `"0"` | 是否打开 MuJoCo 原生 viewer（`"1"`/`"0"`） |
| `gui` | `"0"` | 同 `mujoco_viewer` 的别名 |
| `bench_json` | `""` | 宏步逐帧计时输出 JSON 路径（空则关闭） |

> 顶层还有两个 XPBD 默认字段（不在 `run` 段）：`xpbd_default_target`（默认 target 二进制名）、`xpbd_default_version`（orca-xpbd pip 版本）。

> 布料网格（也不在 `run` 段）：`cloth.mesh`（网格文件名，默认 `croissant_Low-copy.vtk`）+ `orcastudio.assets_folder`（资产目录，容器内路径如 `OrcaStudio_2409/Assets/Life_Kitchen_01`）。XPBD 按 `assets_folder + "/" + mesh` 加载（写进 session 的 `cloth.mesh_abs_path`）。注意 `assets_folder` 必须是 XPBD 进程所在环境能访问的绝对路径。

> 如需临时覆盖而不改 Config.json，仍可用同名环境变量（如 `AGENT=openloong`），优先级：env > Config.json `run` 段。`CFG` / `CLOTH_CONFIG` 仍为仅环境变量参数（指定 config 文件路径）。

---

## 3. 机器人参数（Agent-utree.json）

机器人相关参数已从 `Config.json` 抽离到 `Agent-utree.json`，由 `Config.json` 的 `agent_file` 字段引用：

```json
{ "agent_file": "Agent-utree.json" }
```

加载时（`RunCollection._load_config` 与 `envs.softbody.load_cloth_config`）会把 `Agent-utree.json` 深合并进 `Config.json`；两者字段不相交，合并后等价于原来的单配置。

### 3.1 Agent-utree.json 结构

| 段 | 内容 |
|---|---|
| `run.agent` / `run.mjc_prefix` | 机器人型号 + MuJoCo body 前缀 |
| `orcagym` | `mjc_agent_prefix` / `rigid_body_map_key` / `sync_mocap_from_gripper` |
| `robot` | 关节/夹爪/执行器配置（`l_arm` / `r_arm` / `gripper_l` / `gripper_r` / `base_body` / `motors_group` / `positions_group` / `grip_trigger_scale`），对应原 `conf/*_conf.py` |
| `grip_bind` | 左右掌/指 body 名匹配（legacy + suffix） |
| `xpbd_auto_discover` | body 匹配白名单 / 黑名单 / follow_mode |
| `orcagym_rigid_body_map` | robot holder 刚体表 |

### 3.2 新增一个机器人

1. 新建 `Agent_<robot>.json`，按上表填好该机器人的关节/夹爪/body 匹配。
2. 在 `Config.json` 里把 `agent_file` 指向它（或按机器人名切换）。
3. 若 `robot` 段已在 Agent-utree.json 提供，`_assemble_agent` 会优先用它（无需再写 `conf/*_conf.py`）；`data_storage` / `obs_callback` 仍按 `agent_name` 分支。

### 3.3 手指弯曲调优参数

手指弯曲幅度由两个参数控制（都在 `Agent-utree.json` 的 `robot` 段），公式：

```
手指 ctrl = init_ctrl × (1 + grip_trigger_scale × 扳机值)
```

| 参数 | 默认值 | 作用 |
|---|---|---|
| `robot.gripper_l / gripper_r.init_ctrl` | 左 `[0.4, -0.2, 0.3, -0.3, -0.3, -0.3, -0.3]`、右取镜像 | **复位后**（扳机=0）的基础弯度；顺序 = `[thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1]` |
| `robot.grip_trigger_scale` | `0.3` | **扣扳机**时在基础弯度上的合拢倍率 |

调节方式：

- **复位后太弯** → 调小 `init_ctrl` 绝对值（如 middle/index ±0.3 → ±0.2）。
- **扣扳机太弯** → 调小 `grip_trigger_scale`（如 0.3 → 0.2）。
- **想要更弯** → 反向调大。

### 3.4 抓取锁定参数

抓取时面包哪些粒子被「锁」（跟随指尖），由 `Config.json` 的 `cloth` 段两个参数控制：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `cloth.lock_radius_m` | `0.05` | 锁粒半径：指尖中点球内、距离 < 此值的粒子才被锁 |
| `cloth.max_lock_approach_m` | `0.25` | 最大接近距离：指尖到最近粒子 > 此值就拒绝锁（防隔空抓取） |

锁粒判断（`dg_lock_grip_hand`）：

```
指尖到最近粒子距离 < max_lock_approach_m（否则拒绝锁）
  且
粒子到指尖距离 < lock_radius_m（锁这些粒子）
```

调节方式：

- **锁太多（整块僵直、穿平台）** → 调小 `lock_radius_m`（如 10.0 → 0.05），只锁指尖附近，其余粒子保持动态、会与平台碰撞。
- **隔空也能抓** → 调大 `max_lock_approach_m`（如 100.0）；**要靠近才抓** → 调小（如 0.25）。
- **抓取点从「指尖」换成「手心」** → 需改 XPBD C++（`dg_lock_grip_hand` 的锁中心 + `dg_track_grip_hand` 的跟踪），见 [[20260826抓取锁定后穿平台排查.md]]。

> 注意：锁定的粒子 `inv_mass=0`（运动学），不参与碰撞——锁全部粒子会导致面包穿平台。所以当前默认只锁指尖附近一小撮。
