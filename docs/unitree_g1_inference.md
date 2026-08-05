# 宇树 G1 · 在线推理

本文说明宇树 G1 四色按按钮任务的 OpenPI 在线推理。数据采集见 [unitree_g1_collection.md](unitree_g1_collection.md)。

---

## 前置条件

1. 请启动 OrcaLab 6.3，并加载布局 `unitree_button.json`。
2. 请按采集文档配置两路相机（头 `7070`、右腕 `7080`）并启动仿真。
3. 请设置 `openpi_client` 路径：

```bash
export OPENPI_CLIENT_SRC=<openpi-client源码路径>/src
```

4. 请先启动策略服务（端口与下方 `--port` 一致，默认 `8010`）。

---

## 启动推理

```bash
conda activate orcalab_lerobot
cd src/examples/inference/unitree_g1

python -u eval_g1_pick_lerobot.py \
    --task_config ../../dataCollection/common/example.yaml \
    --host localhost --port 8010 \
    --prompt "按红色按钮" \
    --agent_name unitree_humanoid_robot_1 \
    --max_steps 500 --episodes 3
```

说明：

- 观测为 28 维实测关节角 `q`，策略输出 `Δq`，控制器做开环积分后下发给臂/手 position 执行器。
- 相机键为 `cam_head` / `cam_wrist_r`（与采集一致）。

---

## 故障排查

**现象**：找不到 `openpi_client`。**处理**：请确认已设置 `OPENPI_CLIENT_SRC`。

**现象**：相机超时。**处理**：请按采集文档重新配置相机端口与 Recording。

**现象**：机械臂下垂或跟踪发散。**处理**：请确认策略输出的是 `Δq`，且评估脚本使用开环积分（勿每步用实测重基）。
