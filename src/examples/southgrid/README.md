# SouthGrid 示例

南网大赛任务留在本目录：四色按钮、工具整理、槽位/候选位姿、中文 prompt、布局 JSON 与相机端口表。通用采集、LeRobot 写入、策略接口在仓库 `src/` 与 [docs/lerobot/](../../../docs/lerobot/README.md)。运行环境用 `docs/lerobot/environment.yml` 与 `docs/lerobot/install_runtime.sh`，不要装根目录 `requirements.txt`。

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

- [docs/lerobot/README.md](../../../docs/lerobot/README.md)
- [docs/lerobot/dataset.md](../../../docs/lerobot/dataset.md)
- [docs/lerobot/policy_inference.md](../../../docs/lerobot/policy_inference.md)
- [examples/inference/README.md](../inference/README.md)
- [docs/lerobot/cameras_and_video.md](../../../docs/lerobot/cameras_and_video.md)
- [docs/lerobot/openpi_deployment.md](../../../docs/lerobot/openpi_deployment.md)

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

# 脚本化采集：上午自己标的、已经是控制段的路点（默认 none，原样播）
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

## 脚本化编段（`build_segments`）

通用入口 `data_collection_scripted_lerobot.py` 只播控制段。任务怎么走近目标、何时开合爪，由 `--segment_builder` / `--segment_config` 决定，不写进采集内核。

| builder | 输入 | 编出 |
|---------|------|------|
| `none`（默认） | YAML `segments` | 原样直线播 |
| `tool` | 每文件 4 或 6 个业务路点 | 高位过渡 → 垂直落下 → 对准再闭爪 → 抬升放箱 |
| `button` | `buttons.*.candidates` 或单个接触点 | 后退接近 → 前推 → 保压 → 撤回 |

接口：`trajectory.segment_builder.SegmentBuilder.build(waypoints) -> segments`。新任务实现一个 builder 并在 `get_segment_builder` 注册即可。

配置文件只写任务参数（`builder`、`safe_z`、各段步数等）：

- 工具：`configs/segment_builder_tool.yaml`
- 按钮：`configs/segment_builder_button.yaml`

不传 `--segment_config` 时，`tool` / `button` 用文件里同一套默认值。逗号分隔的多个工具 YAML **按文件分别编段再拼接**。

### 工具（SouthGrid 交付路点）

OrcaLab 加载 `g1_tool.json`。`my_waypoint_tool1.yaml` 这类 6 点文件必须加 `tool`，否则会从初始位直线扑向抓取高度。

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
WP=/home/dht/orca_m/SouthGrid/src/examples/dataCollection/g1_omnipicker

python data_collection_scripted_lerobot.py \
  --agent_name g1_omnipicker --level example \
  --task_config ../southgrid/configs/example.yaml \
  --segment_builder tool \
  --segment_config ../southgrid/configs/segment_builder_tool.yaml \
  --pose_file $WP/my_waypoint_tool1.yaml,$WP/my_waypoint_tool2.yaml,$WP/my_waypoint_tool3.yaml,$WP/my_waypoint_tool4.yaml,$WP/my_waypoint_tool5.yaml \
  --lerobot_out ~/datasets/sg_omni_tool \
  --repo_id local/g1_omnipicker_tool \
  --task "整理工具" --fps 20 --clock sim --kp 220 \
  --max_episodes 5
```

### 按钮（候选接触点）

OrcaLab 加载 `g1_button.json`。选色 / `--counts` / 每集抽候选在 **按钮脚本**里，不进通用入口。编段用 `ButtonSegmentBuilder`（后退接近 → 前推 → 保压 → 撤回）。

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/southgrid/g1_omnipicker

python g1_omnipicker_collection_scripted_button_lerobot.py \
  --task_config ../configs/example.yaml \
  --pose_file pose_g1_button_candidates.yaml \
  --segment_config ../configs/segment_builder_button.yaml \
  --lerobot_out ~/datasets/sg_omni_button_scripted \
  --repo_id local/g1_omnipicker_button \
  --counts 1,1,1,1 \
  --fps 20 --clock wall --kp 220
```

`--counts` 顺序是红,绿,黄,蓝；每集在该颜色候选里随机抽一个，语言指令用 YAML 里的「按红色按钮」等。

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/southgrid/g1_omnipicker

python g1_omnipicker_replay_lerobot.py \
  --task_config ../configs/example.yaml \
  --lerobot_out ~/datasets/sg_omni_button_scripted \
  --episode_index 0 --kp 220
```
