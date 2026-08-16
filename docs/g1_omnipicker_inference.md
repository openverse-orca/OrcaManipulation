# 智元 G1 OmniPicker · 在线推理

本文说明按钮/工具任务的 OpenPI 在线推理。数据采集见 [g1_omnipicker_collection.md](g1_omnipicker_collection.md)。

---

## 架构概览

推理由两个独立进程组成，通过 WebSocket 通信：

```
┌──────────────────────────────────────────┐      WebSocket :8010
│  策略服务器（openpi uv 环境）              │ ◄──────────────────────
│  serve_policy.py / openpi-rtc-serve      │
│  需要 GPU（≥8 GB 显存）                   │ ──────────────────────►
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  eval 推理脚本（orcalab_lerobot Conda 环境）│
│  eval_g1_omnipicker_lerobot.py           │
│  连接 OrcaLab 仿真 + 策略服务器           │
└──────────────────────────────────────────┘
```

两个进程所用的 Python 环境**完全独立**：eval 脚本用本仓库的 `orcalab_lerobot`（Conda），策略服务器用 openpi 的 `uv` 环境。

---

## 前置条件

1. 请在运行本项目的主机上启动 OrcaLab 7.1，并在 OrcaLab 的加载布局对话框中选择与任务对应的布局文件（`src/examples/dataCollection/g1_omnipicker/g1_button.json` 或 `src/examples/dataCollection/g1_omnipicker/g1_tool.json`）。
2. 请按采集文档配置相机端口并启动仿真（`localhost:50051`）。
3. 请确认已按仓库根目录 README 的「环境安装」一节执行 `bash scripts/install_runtime.sh`。
4. 策略服务器需要独立的 **openpi uv 环境**。如尚未完成 openpi 安装与训练，请先阅读 [openpi_deployment.md](openpi_deployment.md) 第 1–8 节，完成环境配置、数据准备与训练。完成后再回到本文执行推理。

---

## 场景一：本地推理

策略服务器与 eval 脚本运行在同一台机器上。该机器需要同时具备 GPU 和已安装的 OrcaLab。

### 1. 启动策略服务器

请在运行本项目的主机上打开一个独立终端，进入 openpi 工作目录（完整说明见 [openpi_deployment.md § 8](openpi_deployment.md#8-启动推理服务)）：

```bash
cd /path/to/openpi

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run scripts/serve_policy.py \
    --port 8010 \
    policy:checkpoint \
    --policy.config=<your_config_name> \
    --policy.dir=checkpoints/<your_config_name>/<exp_name>/<step>
```

请等待策略服务器终端打印 `server listening on 0.0.0.0:8010` 后，再继续下一步。

### 2. 运行 eval 脚本

请在运行本项目的主机上打开另一个终端，从仓库根目录进入推理脚本所在目录。本地推理时，请将 `--host` 设为 `localhost`（该参数的默认值即为 `localhost`）：

**按钮任务：**

```bash
cd src/examples/inference/g1_omnipicker
python eval_g1_omnipicker_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host localhost \
    --port 8010 \
    --prompt "按红色按钮" \
    --max_steps 500 \
    --action_repeat 1 \
    --episodes 3
```

**工具任务：**

```bash
cd src/examples/inference/g1_omnipicker
python eval_g1_omnipicker_tool_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host localhost \
    --port 8010 \
    --prompt "整理工具" \
    --max_steps 10000 \
    --episodes 1
```

---

## 场景二：远程服务器推理

策略服务器运行在远程 GPU 服务器上，eval 脚本在本地（OrcaLab 所在机器）运行。

### 1. 在 GPU 服务器上启动策略服务

请先通过 SSH 登录远程 GPU 服务器，再在该服务器的终端中进入 openpi 工作目录（参考 [openpi_deployment.md § 8](openpi_deployment.md#8-启动推理服务)）：

```bash
cd /path/to/openpi

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run scripts/serve_policy.py \
    --port 8010 \
    policy:checkpoint \
    --policy.config=<your_config_name> \
    --policy.dir=checkpoints/<your_config_name>/<exp_name>/<step>
```

> **网络要求**：服务器的 8010 端口需对本地机器可达。若两台机器不在同一内网，可通过 SSH 隧道转发：
> ```bash
> # 在本地机器执行，将本地 8010 映射到远端服务器 8010
> ssh -L 8010:localhost:8010 user@<server_ip>
> ```
> 使用 SSH 隧道时，eval 脚本仍填 `--host localhost --port 8010`。

### 2. 在本地运行 eval 脚本

请在运行 OrcaLab 的本地机器上打开终端，从仓库根目录进入推理脚本所在目录。不使用 SSH 隧道时，请将 `--host` 替换为服务器 IP 或主机名：

**按钮任务：**

```bash
cd src/examples/inference/g1_omnipicker
python eval_g1_omnipicker_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host <server_ip_or_hostname> \
    --port 8010 \
    --prompt "按红色按钮" \
    --max_steps 500 \
    --action_repeat 1 \
    --episodes 3
```

**工具任务：**

```bash
cd src/examples/inference/g1_omnipicker
python eval_g1_omnipicker_tool_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host <server_ip_or_hostname> \
    --port 8010 \
    --prompt "整理工具" \
    --max_steps 10000 \
    --episodes 1
```

---

## 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--task_config` | 任务配置文件路径 | `../../dataCollection/common/example.yaml` |
| `--orcagym_addr` | OrcaGym 服务地址 | `localhost:50051` |
| `--host` | 策略服务器主机（本地填 `localhost`，远程填 IP 或主机名） | `localhost` |
| `--port` | 策略服务器 WebSocket 端口 | `8010` |
| `--prompt` | 任务语言指令，须与训练数据中的描述完全一致 | 按钮任务为 `按红色按钮`，工具任务为 `整理工具` |
| `--max_steps` | 每集最大控制步数 | 按钮 500，工具 10000 |
| `--action_repeat` | 每个动作块重复执行次数 | `1` |
| `--episodes` | 评估集数 | `1` |
| `--camera_warmup_steps` | 每集推理前相机预热步数 | `10` |
| `--sleep` | 按实时步长节奏运行 | 未启用 |
| `--no_images` | 跳过相机采图，发送空图 | 未启用 |
| `--no_preview` | 按钮任务：不显示相机实时预览小窗口 | 未启用（按钮任务默认显示预览） |
| `--preview` | 工具任务：显示相机实时预览 | 未启用（工具任务默认不显示预览） |
| `--kp` | 工具任务：阻抗刚度 | `150.0` |

工具任务的近桌外环积分等冷门参数见下方「高级调参参数」。

---

## 故障排查

**现象**：找不到 `openpi_client`。**处理**：请在仓库根目录重新执行 `bash scripts/install_runtime.sh`，不要添加外部源码路径。

**现象**：相机超时。**处理**：请按采集文档重新配置相机端口，并确认 Recording 已勾选。

**现象**：WebSocket 连接失败（远程场景）。**处理**：确认服务器防火墙已放行 8010 端口，或改用 SSH 隧道方案。

**现象**：策略服务器 OOM 或响应慢。**处理**：参见 [openpi_deployment.md § 10](openpi_deployment.md#10-故障排查)。

---

## 高级调参参数

以下参数仅出现在工具任务推理脚本中，一般保持默认即可。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--grasp_integral` | 开启近桌外环积分 | 未启用 |
| `--grasp_integral_ki` | 外环积分增益 | `0.2` |
| `--grasp_integral_max` | 外环积分偏置限幅，单位米 | `0.010` |
| `--grasp_integral_axes` | 外环积分生效轴，如 `z` / `xy` / `xyz` | `z` |
| `--grasp_integral_log_every` | 积分日志节流：每 N 控制步打印一次；`0` 表示关闭 | `10` |
| `--grasp_integral_z_below` | 仅当策略右臂目标 z 不超过该值时启用外环积分，单位米 | `0.25` |
