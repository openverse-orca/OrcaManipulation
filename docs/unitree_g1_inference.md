# 宇树 G1 · 在线推理

本文说明宇树 G1 四色按按钮任务的 OpenPI 在线推理。数据采集见 [unitree_g1_collection.md](unitree_g1_collection.md)。

---

## 前置条件

1. 请在运行本项目的主机上启动 OrcaLab 7.1，并在 OrcaLab 的加载布局对话框中选择 `src/examples/dataCollection/unitree_g1/unitree_button.json`。
2. 请按采集文档配置两路相机（头 `7070`、右腕 `7080`）并启动仿真。
3. 请确认已按仓库根目录 README 的「环境安装」一节执行 `bash scripts/install_runtime.sh`。
4. 请先启动策略服务（端口与下方 `--port` 一致，默认 `8010`）。

---

## 启动推理

请在运行本项目的主机上激活环境，再从仓库根目录进入推理脚本所在目录：

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

- 观测为 28 维关节角，相机键为 `cam_head` / `cam_wrist_r`（与采集一致）。

---

## 故障排查

**现象**：找不到 `openpi_client`。**处理**：请在仓库根目录重新执行 `bash scripts/install_runtime.sh`，不要添加外部源码路径。

**现象**：相机超时。**处理**：请按采集文档重新配置相机端口与 Recording。

**现象**：机械臂下垂或跟踪发散。**处理**：请确认评估脚本与训练时使用同一套动作约定，并检查策略服务是否已加载正确的 checkpoint。
