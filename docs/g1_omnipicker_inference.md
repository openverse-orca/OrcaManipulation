# 智元 G1 OmniPicker · 在线推理

本文说明按钮/工具任务的 OpenPI 在线推理。数据采集见 [g1_omnipicker_collection.md](g1_omnipicker_collection.md)。

---

## 前置条件

1. 请启动 OrcaLab 6.3，并加载与任务对应的布局（`g1_button.json` 或 `g1_tool.json`）。
2. 请按采集文档配置相机端口并启动仿真（`localhost:50051`）。
3. 请确认已按仓库根目录 README 执行 `bash scripts/install_runtime.sh`。
4. 策略服务器需要独立的 **openpi uv 环境**（与本仓库的 `orcalab_lerobot` Conda 环境相互独立）。如尚未完成 openpi 安装与训练，请先阅读 [openpi_deployment.md](openpi_deployment.md) 第 1–8 节完成环境配置、数据准备与训练，再回到本文执行推理。

---

## 在线推理

### 启动策略服务

请在 **openpi 工作目录**的独立终端中启动服务（以端口 `8010` 为例），完整命令见 [openpi_deployment.md § 8](openpi_deployment.md#8-启动推理服务)：

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

### 按钮任务推理

```bash
python eval_g1_omnipicker_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host localhost \
    --port 8010 \
    --prompt "按红色按钮" \
    --max_steps 500 \
    --action_repeat 1 \
    --episodes 3
```

### 工具任务推理

```bash
python eval_g1_omnipicker_tool_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host localhost \
    --port 8010 \
    --prompt "整理工具" \
    --max_steps 10000 \
    --episodes 1
```

`--prompt` 须与训练数据中的任务描述保持一致。

---


## 故障排查

**现象**：找不到 `openpi_client`。**处理**：请在仓库根目录重新执行 `bash scripts/install_runtime.sh`，不要添加外部源码路径。

**现象**：相机超时。**处理**：请按采集文档重新配置相机端口与 Recording。
