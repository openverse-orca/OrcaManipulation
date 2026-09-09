# SouthGrid 示例

南网大赛任务留在本目录：四色按钮、工具整理、槽位/候选位姿、中文 prompt、布局 JSON 与相机端口表。通用采集、LeRobot 写入、策略接口在仓库 `src/` 与 `docs/`。

## 目录

| 路径 | 内容 |
|------|------|
| `g1_omnipicker/` | 智元 G1 OmniPicker 遥操作 / 脚本化采集 / 回放 |
| `unitree_g1/` | Unitree G1 OSC 采集 / 回放；`g1_pick_constraints.py` 为共用姿态约束 |
| `inference/g1_omnipicker/` | OpenPI 在线推理（按钮 / 工具） |
| `configs/example.yaml` | 采集与推理共用任务配置 |
| `docs/` | 比赛场景与按键说明 |

框架用法见：

- [docs/lerobot_dataset.md](../../../docs/lerobot_dataset.md)
- [docs/policy_inference.md](../../../docs/policy_inference.md)
- [docs/cameras_and_video.md](../../../docs/cameras_and_video.md)
- [docs/openpi_deployment.md](../../../docs/openpi_deployment.md)

## 入口脚本

在对应子目录运行（需先启动 OrcaLab）：

```bash
# OmniPicker 遥操作采集
python g1_omnipicker_collection_tele_lerobot.py --lerobot_out ~/datasets/g1 --task_config ../configs/example.yaml

# Unitree G1 遥操作采集
python g1_pick_osc_collection_tele_lerobot.py --lerobot_out ~/datasets/g1_pick --task_config example.yaml

# 按钮任务推理
python eval_g1_omnipicker_button_lerobot.py --prompt "按红色按钮"
```

相机与 writer 由 `manager.run()` 通过 storage 的 `open_capture_session` 管理。也可以走通用入口：

```bash
python data_collection_tele_lerobot.py --agent_name g1_omnipicker \
  --lerobot_out ~/datasets/g1 --task_config ../southgrid/configs/example.yaml --level default
```
