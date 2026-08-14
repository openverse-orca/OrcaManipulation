# Cloth Robot P23c 运行指南（PyPI 模式）

本文档说明如何在 **OrcaManipulation 仓库内**以 PyPI 包方式运行布料三进程联调（P23c），
不依赖外部编排脚本。

PyPI 模式直接 `pip install` 发布包（`orca-xpbd` / `orca-link`），不编译源码。

---

## 运行流程总览

```
Step 0: 启动 OrcaStudio 并进入 Play（手动，仅一次）
        ↓
Step 1: 进入容器，激活 conda 环境
        ↓
Step 2: 安装依赖（requirements.txt）
        ↓
Step 3: 运行 run_cloth_robot_p23c.sh
```

> 注意：**必须先启动 Studio 并 Play**，否则 Step 3 会卡在
> 「等待 OrcaGym :50051 / PBDRender :50263」超时。

---

## Step 0: 启动 OrcaStudio 并进入 Play

在宿主机启动 OrcaStudio（OrcaEditor），打开关卡（如 `Life_Kitchen_01_with_utree`），点击 **Play**。

确认以下端口已监听：

```bash
ss -tlnp | grep -E "50051|50263"
# OrcaGym  :50051
# PBDRender:50263
```

---

## Step 1: 激活 conda 环境

```bash
conda activate <你的环境名>
```

`run_cloth_robot_p23c.sh` 会通过 `PrepareConda.py`
自动使用当前激活的非 base conda 环境，无需额外配置。

---

## Step 2: 安装依赖

```bash
# 在本示例目录下安装依赖（引用顶层 requirements + 声明 orca-xpbd/orca-link 版本）
pip install -r src/examples/dataCollection_cook/requirements.txt \
    --index-url https://pypi.org/simple/ \
    --extra-index-url https://test.pypi.org/simple/
```

`requirements.txt` 已声明两个发布包的固定版本：

```txt
orca-xpbd==<版本>
orca-link==<版本>
```

版本号与 `xpbd_scene.conf` 的 `XPBD_DEFAULT_VERSION` 保持一致，
由发布流程自动同步，无需手动维护。

---

## Step 3: 运行三进程联调

```bash
cd <REPO_ROOT>/OrcaManipulation/src/examples/dataCollection_cook

# PyPI 模式：使用 pip 安装的 XPBD 二进制，而非源码编译
ORCAXPBD_USE_PIP_PACKAGE=1 bash run_cloth_robot_p23c.sh
```

`run_cloth_robot_p23c.sh` 会自动：

1. 读取 `xpbd_scene.conf`（target + 版本）
2. PyPI 模式走 `EnsureXPBD.py`，从 pip 包同步 `dual_gripper_g1_cook2`
3. 刷新 session + 导出 scene
4. 启动 `data_collection_cloth_tele.py`（OrcaLink + XPBD + bridge）

### 常用环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LEVEL` | 自动检测 | 关卡名（如 `Life_Kitchen_01_with_utree`） |
| `AGENT` | `openloong` | Agent 名称（如 `g1_omnipicker`） |
| `XPBD_UI` | `1` | 设为 `0` 关闭 XPBD OpenGL 窗口 |
| `COLLECT_DATA` | `0` | 设为 `1` 采集数据集/HDF5 |
| `ORCAXPBD_USE_PIP_PACKAGE` | `0` | 设为 `1` 走 pip 包；`0` 走源码编译 |

---

## 相关文件（均在 OrcaManipulation 仓库内）

| 文件 | 说明 |
|------|------|
| `xpbd_scene.conf` | XPBD 场景集中配置（target + 版本），新增场景只改这里 |
| `run_cloth_robot_p23c.sh` | 三进程联调入口 |
| `EnsureXPBD.py` | 确保 XPBD 二进制就位（pip 同步 或 源码编译，二选一） |
| `data_collection_cloth_tele.py` | 布料遥操/回放主程序 |
| `requirements.txt` | 本示例依赖声明 |
