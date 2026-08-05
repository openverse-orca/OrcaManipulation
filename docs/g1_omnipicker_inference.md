# 智元 G1 OmniPicker · 在线推理

本文说明按钮/工具任务的 OpenPI 在线推理。数据采集见 [g1_omnipicker_collection.md](g1_omnipicker_collection.md)。

---

## 前置条件

1. 请启动 OrcaLab 6.3，并加载与任务对应的布局（`g1_button.json` 或 `g1_tool.json`）。
2. 请按采集文档配置相机端口并启动仿真（`localhost:50051`）。
3. 请设置 `openpi_client` 路径：

```bash
export OPENPI_CLIENT_SRC=<openpi-client源码路径>/src
```

---

## 在线推理

### 启动策略服务

请在策略侧的独立终端启动服务（以端口 `8010` 为例）：

```bash
python serve_policy.py \
    --port 8010 \
    --checkpoint /path/to/checkpoint
```

### 按钮任务推理

```bash
export OPENPI_CLIENT_SRC=~/openpi/packages/openpi-client/src

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
export OPENPI_CLIENT_SRC=~/openpi/packages/openpi-client/src

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

**现象**：找不到 `openpi_client`。**处理**：请确认已设置 `OPENPI_CLIENT_SRC` 且路径下存在可导入的包。

**现象**：相机超时。**处理**：请按采集文档重新配置相机端口与 Recording。
