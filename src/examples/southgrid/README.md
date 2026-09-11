# SouthGrid 示例

南网大赛任务留在本目录：四色按钮、工具整理、槽位/候选位姿、中文 prompt、布局 JSON 与相机端口表。通用采集、LeRobot 写入、策略接口在仓库 `src/` 与 `docs/`。

## 目录

| 路径 | 内容 |
|------|------|
| `g1_omnipicker/` | 智元 G1 OmniPicker 遥操作 / 脚本化采集 / 回放 |
| `unitree_g1/` | Unitree G1 OSC 采集 / 回放；`g1_pick_constraints.py` 为共用姿态约束 |
| `inference/g1_omnipicker/` | OpenPI 在线推理（按钮 / 工具），装配走 `examples/inference/` |
| `inference/unitree_g1/` | Unitree G1 在线推理（`--pin_joints` + `pin_all_joints`） |
| `configs/example.yaml` | 采集与推理共用任务配置 |
| `waypoint/` | 标点脚本默认保存目录（按机型分子目录） |
| `docs/` | 比赛场景与按键说明 |

框架用法见：

- [docs/lerobot_dataset.md](../../../docs/lerobot_dataset.md)
- [docs/policy_inference.md](../../../docs/policy_inference.md)
- [examples/inference/README.md](../inference/README.md)
- [docs/cameras_and_video.md](../../../docs/cameras_and_video.md)
- [docs/openpi_deployment.md](../../../docs/openpi_deployment.md)

## 入口脚本

在对应子目录运行（需先启动 OrcaLab）：

```bash
# OmniPicker 遥操作采集（cd src/examples/southgrid/g1_omnipicker）
python g1_omnipicker_collection_tele_lerobot.py --lerobot_out ~/datasets/g1 --task_config ../configs/example.yaml

# Unitree G1 遥操作采集（cd src/examples/southgrid/unitree_g1）
python g1_pick_osc_collection_tele_lerobot.py --lerobot_out ~/datasets/g1_pick --task_config example.yaml

# 按钮任务推理（cd src/examples/southgrid/inference/g1_omnipicker）
python eval_g1_omnipicker_button_lerobot.py --prompt "按红色按钮"

# 宇树推理（cd src/examples/southgrid/inference/unitree_g1）
python eval_g1_pick_lerobot.py --prompt "整理工具"
```

脚本化路点用通用标点入口（在 `src/examples/dataCollection` 下运行，需 Pico）。默认写到 `examples/southgrid/waypoint/<机型>/`：

```bash
# OmniPicker 工具 6 点 → waypoint/g1_omnipicker/my_waypoint_tool.yaml
python record_waypoints.py --agent_name g1_omnipicker --task tool \
  --task_config ../southgrid/configs/example.yaml --guide tool6

# OmniPicker 按钮候选（一色一次，--resume 保留其它颜色）
python record_waypoints.py --agent_name g1_omnipicker --task button --color red \
  --task_config ../southgrid/configs/example.yaml --resume

# OmniPicker 槽位补录
python record_waypoints.py --agent_name g1_omnipicker --task slot \
  --task_config ../southgrid/configs/example.yaml

# 宇树工具 / 按钮
python record_waypoints.py --agent_name g1_pick --task tool \
  --task_config ../southgrid/unitree_g1/example.yaml
python record_waypoints.py --agent_name g1_pick --task button --color red \
  --task_config ../southgrid/unitree_g1/example.yaml
```

双 Grip 记点，单右 Grip 重置当前任务，Ctrl+C 写出 YAML。`--task slot` 仅支持 OmniPicker。

相机与 writer 由 `manager.run()` 通过 storage 的 `open_capture_session` 管理。也可以走通用入口（在 `src/examples/dataCollection` 下）：

```bash
# 遥操作
python data_collection_tele_lerobot.py --agent_name g1_omnipicker \
  --lerobot_out ~/datasets/g1 --task_config ../southgrid/configs/example.yaml --level default

# 脚本化采集（工具路点）
python data_collection_scripted_lerobot.py \
  --agent_name g1_omnipicker --level default \
  --task_config ../southgrid/configs/example.yaml \
  --pose_file ../southgrid/waypoint/g1_omnipicker/my_waypoint_tool.yaml \
  --lerobot_out ~/datasets/g1_tool --fps 20 --kp 220 --grasp_integral

# 回放
python data_collection_replay_lerobot.py \
  --agent_name g1_omnipicker --level default \
  --task_config ../southgrid/configs/example.yaml \
  --lerobot_out ~/datasets/g1_tool --episode_index 0 --kp 220 --grasp_integral
```

OSC 参数：`g1_omnipicker` 默认 `--kp 220`；`g1_pick` 默认 `--dls_lambda 0.23 --dls_sigma_th 0.12 --track_ki 0.02`。工具脚本化采集和回放默认加 `--grasp_integral`。
