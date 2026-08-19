# 机器人抓面包场景 运行指南

本文档说明如何在 **OrcaManipulation 仓库内**运行 机器人抓面包场景。

## 目录

- [架构设计](#架构设计)
---

## 架构设计

### 两个目录的定位

| 目录 | 定位 |
|------|------|
| `src/envs/softbody` | 布料 MjcPBD 耦合的**核心库（引擎）**：编排器 + 子进程生命周期 + 领域工具 + 中性基础设施。只含逻辑，不含具体场景/关卡参数。 |
| `src/examples/dataCollection_cook` | **薄启动器 + 数据目录（入口）**。`RunCollection.py` 只从环境变量组装 `P23cParams` 并调用 `run_p23c`；同时承载**单配置文件 `Config.json`**（布料 sim + 场景资产路径 + XPBD target/版本）与运行时产物（`cloth_sim_session_*.json`、`logs/`）。 |

### softbody 分层

softbody 内按职责分四层 + 一个脚本，方向是「上层调用下层、下层不感知上层」：

1. **编排器** `attach_coupling.py`（主进程）
   - 顶层入口 `run_p23c(params)`：session 准备 → 场景/资产解析 → config 组装 → 建 env → 挂耦合 → 遥操循环。
   - `P23cParams` dataclass 是唯一参数契约。
   - 场景/资产解析、config 改写等「主进程」职责都在这里。

2. **子进程生命周期** `Process*.py`
   - `ProcessOrcaGym` / `ProcessOrcaLink` / `ProcessXPBD` / `ProcessStudio` / `ProcessPico`。
   - 各自只负责对应子进程的启动/清理/探测，**零相互依赖**，只通过编排器交互。

3. **领域工具 / 数据契约** `domain/`
   - `body_map.py`：刚体表 + 锚点 SITE + 碰撞 AABB（`BodyMapEntry` 等）。
   - `anchor_frame.py` / `anchor_tetrahedron.py` / `mjc_coords.py`：锚点帧 / 四面体 / 坐标转换。
   - 被编排器、Process* 复用；可依赖 `base/`。

4. **中性基础设施** `base/`
   - `paths.py`：纯路径解析（`PROJECT_ROOT` / `ORCA_REPO_ROOT` / `SOFTBODY_DIR` 等）。
   - `process_utils.py`：`ProcessManager` 子进程登记与统一清理。
   - `masked_vtk.py`：掩码 VTK 三件套路径/文件名变换。
   - 不引入领域语义与重依赖（mujoco 等）。

### 进程拓扑

- 主进程：`RunCollection.py` → `run_p23c`
- OrcaLink Server：端口 50361
- XPBD：原生二进制子进程
- Pico 手柄：TCP 客户端（端口 8001），自身无监听端口

### 职责边界规则

- Process* 零相互依赖，只通过编排器交互。
- 子进程模块不导入主进程模块（编排器用函数传参，而非 import）。
- 依赖单向：`domain/` 可依赖 `base/`，`base/` 不得依赖 `domain/`（中性层不感知领域）。
- 日志目录由编排器显式传入（`P23cParams.log_dir`），不硬编码外部路径。
