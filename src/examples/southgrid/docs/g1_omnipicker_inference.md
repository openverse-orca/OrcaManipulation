# 智元 G1 OmniPicker · 在线推理

本文说明按钮/工具任务的 OpenPI 在线推理。通用接口见 [policy_inference.md](../../../../docs/lerobot/policy_inference.md) 与 [openpi_deployment.md](../../../../docs/lerobot/openpi_deployment.md)。数据采集见 [g1_omnipicker_collection.md](g1_omnipicker_collection.md)。通用入口用法见 [examples/inference/README.md](../../inference/README.md)。

---

## 架构概览

推理由两个独立进程组成，通过 WebSocket 通信：

```
┌──────────────────────────────────────────┐      WebSocket :8010
│  策略服务器（openpi uv 环境）              │ ◄──────────────────────
│  serve_policy.py                          │
│  需要 GPU（≥8 GB 显存）                   │ ──────────────────────►
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  eval 推理脚本（orcalab_lerobot Conda 环境）│
│  eval_g1_omnipicker_button_lerobot.py     │
│  连接 OrcaLab 仿真 + 策略服务器           │
└──────────────────────────────────────────┘
```

两个进程所用的 Python 环境**完全独立**：eval 脚本用本仓库的 `orcalab_lerobot`（Conda），策略服务器用 openpi 的 `uv` 环境。

按钮/工具脚本只注入机型、任务配置和中文 prompt，装配走 `examples/inference/infer_lerobot.py`。

---

## 前置条件

1. 请在运行本项目的主机上启动 OrcaLab 7.3，并在 OrcaLab 的加载布局对话框中选择与任务对应的布局文件（`src/examples/southgrid/g1_omnipicker/g1_button.json` 或 `src/examples/southgrid/g1_omnipicker/g1_tool.json`）。
2. 请按采集文档配置相机端口并启动仿真（`localhost:50051`）。
3. 请确认已按 [docs/lerobot/](../../../../docs/lerobot/README.md) 执行 `conda env create -f docs/lerobot/environment.yml` 与 `bash docs/lerobot/install_runtime.sh`，并激活 `orcalab_lerobot`。
4. 策略服务器需要独立的 **openpi uv 环境**。请先按 [策略服务部署](../../../../docs/lerobot/openpi_deployment.md) 创建独立的 OpenPI 环境，并使用对应的策略配置和 checkpoint 启动服务。

---

## 场景一：本地推理

策略服务器与 eval 脚本运行在同一台机器上。该机器需要同时具备 GPU 和已安装的 OrcaLab。

### 1. 启动策略服务器

请在运行本项目的主机上打开一个独立终端，进入 openpi 工作目录（完整说明见 [openpi_deployment.md § 8](../../../../docs/lerobot/openpi_deployment.md#8-启动推理服务)）：

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

请打开另一个终端，激活 `orcalab_lerobot`，从仓库根目录进入推理脚本所在目录。本地推理时请把 `--host` / `--port` 指到策略服务（通用入口默认 `127.0.0.1:8000`，openpi 示例用 `8010`）：

**按钮任务：**

```bash
conda activate orcalab_lerobot
cd src/examples/southgrid/inference/g1_omnipicker
python eval_g1_omnipicker_button_lerobot.py \
    --host localhost \
    --port 8010 \
    --prompt "按红色按钮" \
    --max_steps 500 \
    --action_repeat 1 \
    --episodes 3
```

**工具任务：**

```bash
conda activate orcalab_lerobot
cd src/examples/southgrid/inference/g1_omnipicker
python eval_g1_omnipicker_tool_lerobot.py \
    --host localhost \
    --port 8010 \
    --prompt "整理工具" \
    --max_steps 10000 \
    --episodes 1
```

工具脚本已注入 `--kp 220`。需要近桌外环积分时再加 `--grasp_integral`。

---

## 场景二：远程服务器推理

策略服务器运行在远程 GPU 服务器上，eval 脚本在本地（OrcaLab 所在机器）运行。

### 1. 在 GPU 服务器上启动策略服务

请先通过 SSH 登录远程 GPU 服务器，再在该服务器的终端中进入 openpi 工作目录（参考 [openpi_deployment.md § 8](../../../../docs/lerobot/openpi_deployment.md#8-启动推理服务)）：

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

请在运行 OrcaLab 的本地机器上打开终端。不使用 SSH 隧道时，请将 `--host` 替换为服务器 IP 或主机名：

**按钮任务：**

```bash
conda activate orcalab_lerobot
cd src/examples/southgrid/inference/g1_omnipicker
python eval_g1_omnipicker_button_lerobot.py \
    --host <server_ip_or_hostname> \
    --port 8010 \
    --prompt "按红色按钮" \
    --max_steps 500 \
    --action_repeat 1 \
    --episodes 3
```

**工具任务：**

```bash
conda activate orcalab_lerobot
cd src/examples/southgrid/inference/g1_omnipicker
python eval_g1_omnipicker_tool_lerobot.py \
    --host <server_ip_or_hostname> \
    --port 8010 \
    --prompt "整理工具" \
    --max_steps 10000 \
    --episodes 1
```

---

## 参数说明

下列参数由通用入口 `examples/inference/infer_lerobot.py` 提供。按钮/工具脚本会预填 `--agent_name`、`--level`、`--task_config`、`--prompt`（工具另预填 `--kp 220`），命令行后续参数可覆盖。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--agent_name` | 机型，按钮/工具脚本固定为 `g1_omnipicker` | 脚本注入 |
| `--level` | 场景档位；可用环境变量 `LEVEL` 覆盖 | `default` |
| `--task_config` | 任务配置文件路径 | `examples/southgrid/configs/example.yaml` |
| `--orcagym_addr` | OrcaGym 服务地址 | `localhost:50051` |
| `--host` | 策略服务器主机 | `127.0.0.1` |
| `--port` | 策略服务器 WebSocket 端口 | `8000` |
| `--prompt` | 任务语言指令，须与训练数据中的描述完全一致 | 按钮 `按红色按钮`，工具 `整理工具` |
| `--cameras` | 参与推理的相机别名 | `head,wrist_r` |
| `--max_episodes` / `--episodes` | 评估集数 | `1` |
| `--max_steps` | 每集最大控制步数 | `500` |
| `--action_repeat` | 每个动作连续下发的控制步数 | `1` |
| `--camera_warmup_steps` | 每集推理前拉相机帧的次数 | `10` |
| `--no_images` | 跳过相机采图，向策略发送空图 | 未启用 |
| `--preview` | 弹出相机实时预览窗口 | 未启用 |
| `--no_realtime` | 关闭按 `real_time_step` 补齐的控制周期 | 未启用 |
| `--pin_joints` | `env.reset` 后调用机型 `constraints_hook` | 未启用（本机型脚本不注入） |
| `--kp` | OSC 阻抗刚度 | `220`（`g1_omnipicker`） |
| `--grasp_integral` | 近桌时对右臂末端位置做外环积分 | 未启用 |

`--sleep` / `--no_preview` 已不存在：实时节奏默认开启，预览默认关闭。

---

## 故障排查

**现象**：找不到 `openpi_client`。**处理**：请激活 `orcalab_lerobot`，并在仓库根目录重新执行 `bash docs/lerobot/install_runtime.sh`。

**现象**：相机超时。**处理**：请按采集文档重新配置相机端口，并确认 IsRecording 已勾选。

**现象**：WebSocket 连接失败（远程场景）。**处理**：确认服务器防火墙已放行策略端口，或改用 SSH 隧道方案。

**现象**：策略服务器 OOM 或响应慢。**处理**：确认所用 GPU 满足策略服务的显存要求，并检查是否有其他进程占用显存。
