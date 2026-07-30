# Cloth Robot P23c 运行指南

## Step 1: 启动 OrcaLab（或 OrcaStudio）并进入 Play

### 方式 A：启动 OrcaLab

```bash
cd <REPO_ROOT>/OrcaLab_2409
./OrcaLab.sh --scene NursingHome
```

### 方式 B：启动 OrcaStudio

```bash
cd <REPO_ROOT>/OrcaStudio_2409/build
./OrcaEditor
```

在界面中打开关卡（如 NursingHome），点击 **Play**。

确认以下端口已监听：
```bash
# 检查 OrcaGym (50051) 和 PBDRender (50261)
ss -tlnp | grep -E "50051|50261"
```

---

## Step 2: 创建 conda 环境

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

---

## Step 3: 安装 Python 依赖

```bash
# 确认 conda 环境已激活
conda activate ${CONDA_ENV_NAME}

# 安装 OrcaManipulation 依赖（如果存在 requirements.txt）
pip install -r <REPO_ROOT>/OrcaManipulation/requirements.txt

# 安装 orca-link
pip install orca-link==26.7.1.2 \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/

# 安装 orca-xpbd
pip install orca-xpbd==26.7.1.9 \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/

# 验证安装
pip list | grep -E "orca-link|orca-xpbd"
```

预期输出：
```
orca-link            26.7.1.2
orca-xpbd            26.7.1.9
```

---


## Step 4: 运行 Cloth P23c 三进程联调

```bash
# 确认 conda 环境已激活
conda activate ${CONDA_ENV_NAME}

# 运行联调脚本（自动完成：编译 XPBD → 刷新 session → 生成 replay → 启动 tele）
cd <REPO_ROOT>/OrcaManipulation/src/examples/dataCollection_cloth/RunCloth
bash run_cloth_robot_p23c.sh
```

**常用环境变量**


```bash
# 可直接复制到终端运行
export LEVEL="NursingHome"
export AGENT="g1_omnipicker"
export MJC_PREFIX="g1_omnipicker_usda"
export DEBUG=0
export XPBD_RELEASE_BUILD=1
export CLOTH_DEBUG=0
export COLLECT_DATA=0
export REPLAY=0
export CLOTH_NO_REALTIME=0
export XPBD_UI=1
export MAX_MACRO_FRAMES=20000
export PBD_GRPC_SBT_ROTATION="from_quat"
export ORCAXPBD_USE_PIP_PACKAGE=0
```

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LEVEL` | `NursingHome` | 关卡名 |
| `AGENT` | `g1_omnipicker` | Agent 名称 |
| `MJC_PREFIX` | `g1_omnipicker_usda` | MJCF agent 前缀 |
| `DEBUG` | `0` | 设为 `1` 开启 debug 模式（含自动分析） |
| `XPBD_RELEASE_BUILD` | `1` | 设为 `0` 使用 Debug 编译的 XPBD |
| `CLOTH_DEBUG` | `0` | 设为 `1` 开启 cloth debug CSV 输出 |
| `COLLECT_DATA` | `0` | 设为 `1` 采集数据集/HDF5 |
| `REPLAY` | `0` | 设为 `1` 开启 replay 模式 |
| `CLOTH_NO_REALTIME` | `0` | 设为 `1` 尽快跑完不等实时 |
| `XPBD_UI` | `1` | 设为 `0` 关闭 XPBD OpenGL 窗口 |
| `MAX_MACRO_FRAMES` | `20000` | 最大宏步数 |
| `PBD_GRPC_SBT_ROTATION` | `from_quat` | 刚体旋转传递方式 |
| `ORCAXPBD_USE_PIP_PACKAGE` | `0` | 设为 `1` 使用 pip 包而非本地编译的 XPBD |

---
