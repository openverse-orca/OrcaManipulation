# 南网链路测试：两机型 × 两任务

测试目标：确认 **通用数采脚本**（`src/examples/dataCollection/`）能够驱动 southgrid 两机型两任务的完整链路。  
环境：`orcalab_lerobot`。每换布局在 OrcaLab 重新加载对应 JSON 再点运行。

```bash
conda activate orcalab_lerobot
adb reverse tcp:8001 tcp:8001    # 遥操/标点前执行
```

相机：右腕 `7080`、头部 `7090`；勾选 UseNvEnc · Color Camera · IsRecording。  
仿真地址默认 `localhost:50051`。

| 机型 | 任务 | 布局 JSON |
|------|------|-----------|
| g1_omnipicker | 按钮 | `src/examples/southgrid/g1_omnipicker/g1_button.json` |
| g1_omnipicker | 工具 | `src/examples/southgrid/g1_omnipicker/g1_tool.json` |
| g1_pick | 按钮 | `src/examples/southgrid/unitree_g1/g1_pick_buttons.json` |
| g1_pick | 工具 | `src/examples/southgrid/unitree_g1/g1_pick_tools.json` |

**所有数采/回放脚本均在 `src/examples/dataCollection/` 下执行，`--task_config` 路径相对该目录。**  
遥操按键：左 Grip 开始 / 再按保存；右 Grip 单按丢弃重置；左右同按退出。  
标点按键：双 Grip 记点；单右 Grip 重置当前任务；Ctrl+C 写 YAML。

通过条件：`data/chunk-000/*.parquet`、`videos/chunk-000/**/*.mp4`、`meta/tasks.jsonl` 写出；回放末端按轨迹运动不崩溃；推理能连上策略并持续下发动作。

---

## 1. OmniPicker · 按钮

OrcaLab 加载 `g1_button.json`。

### 1.1 遥操采集

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python data_collection_tele_lerobot.py \
  --agent_name g1_omnipicker --level example \
  --task_config ../southgrid/configs/example.yaml \
  --lerobot_out ~/datasets/sg_omni_button_tele \
  --repo_id local/g1_omnipicker_button \
  --task "按红色按钮" --fps 20 --clock wall
```

### 1.2 遥操回放

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python data_collection_replay_lerobot.py \
  --agent_name g1_omnipicker --level example \
  --task_config ../southgrid/configs/example.yaml \
  --lerobot_out ~/datasets/sg_omni_button_tele \
  --episode_index 0 --kp 220
```

### 1.3 标点

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python record_waypoints.py --agent_name g1_omnipicker --task button --color red \
  --task_config ../southgrid/configs/example.yaml --resume
```

写出 `src/examples/southgrid/waypoint/g1_omnipicker/pose_g1_button_candidates.yaml`。  
绿/黄/蓝按钮改 `--color green/yellow/blue`，均加 `--resume`。  
仓库自带候选位姿：`../southgrid/g1_omnipicker/pose_g1_button_candidates.yaml`。

### 1.4 脚本化采集

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

### 1.5 脚本化回放

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/southgrid/g1_omnipicker
python g1_omnipicker_replay_lerobot.py \
  --task_config ../configs/example.yaml \
  --lerobot_out ~/datasets/sg_omni_button_scripted \
  --episode_index 0 --kp 220
```

### 1.6 推理

另开终端启动 OpenPI 策略服务（见 `docs/lerobot/openpi_deployment.md` § 8）。

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/inference
python infer_lerobot.py \
  --agent_name g1_omnipicker --level example \
  --task_config ../southgrid/configs/example.yaml \
  --host 127.0.0.1 --port 8010 --prompt "按红色按钮"
```

---

## 2. OmniPicker · 工具

OrcaLab 加载 `g1_tool.json`。

### 2.1 遥操采集

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python data_collection_tele_lerobot.py \
  --agent_name g1_omnipicker --level example \
  --task_config ../southgrid/configs/example.yaml \
  --lerobot_out ~/datasets/sg_omni_tool_tele \
  --repo_id local/g1_omnipicker_tool \
  --task "整理工具" --fps 20 --clock wall
```

### 2.2 遥操回放

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python data_collection_replay_lerobot.py \
  --agent_name g1_omnipicker --level example \
  --task_config ../southgrid/configs/example.yaml \
  --lerobot_out ~/datasets/sg_omni_tool_tele \
  --episode_index 0 --kp 220 --grasp_integral
```

### 2.3 标点

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python record_waypoints.py --agent_name g1_omnipicker --task tool \
  --task_config ../southgrid/configs/example.yaml --guide tool6
```

写出路点 YAML。仓库自带路点：`../southgrid/g1_omnipicker/my_waypoint_tool1.yaml`（螺丝刀）。

### 2.4 脚本化采集

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python data_collection_scripted_lerobot.py \
  --agent_name g1_omnipicker --level example \
  --task_config ../southgrid/configs/example.yaml \
  --segment_builder tool \
  --segment_config ../southgrid/configs/segment_builder_tool.yaml \
  --pose_file ../southgrid/g1_omnipicker/my_waypoint_tool1.yaml \
  --lerobot_out ~/datasets/sg_omni_tool_scripted \
  --repo_id local/g1_omnipicker_tool \
  --task "整理工具" --fps 20 --clock sim --kp 220
```

### 2.5 脚本化回放

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python data_collection_replay_lerobot.py \
  --agent_name g1_omnipicker --level example \
  --task_config ../southgrid/configs/example.yaml \
  --lerobot_out ~/datasets/sg_omni_tool_scripted \
  --episode_index 0 --kp 220 --grasp_integral
```

### 2.6 推理

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/inference
python infer_lerobot.py \
  --agent_name g1_omnipicker --level example \
  --task_config ../southgrid/configs/example.yaml \
  --host 127.0.0.1 --port 8010 --prompt "整理工具"
```

---

## 3. Unitree G1 · 按钮

OrcaLab 加载 `g1_pick_buttons.json`。DataCollectionManager 会通过 `g1_pick_osc_conf` 钉浮动基座 / 腰 / 左臂。

### 3.1 遥操采集

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python data_collection_tele_lerobot.py \
  --agent_name g1_pick --level example \
  --task_config ../southgrid/unitree_g1/example.yaml \
  --lerobot_out ~/datasets/sg_g1_button_tele \
  --repo_id local/g1_pick_button \
  --task "按红色按钮" --fps 20 --clock wall
```

### 3.2 遥操回放

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python data_collection_replay_lerobot.py \
  --agent_name g1_pick --level example \
  --task_config ../southgrid/unitree_g1/example.yaml \
  --lerobot_out ~/datasets/sg_g1_button_tele \
  --episode_index 0
```

### 3.3 标点

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python record_waypoints.py --agent_name g1_pick --task button --color red \
  --task_config ../southgrid/unitree_g1/example.yaml --resume
```

仓库自带路点：`../southgrid/unitree_g1/my_waypoint_button1.yaml`（4 个按钮各一个）。

### 3.4 脚本化采集

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python data_collection_scripted_lerobot.py \
  --agent_name g1_pick --level example \
  --task_config ../southgrid/unitree_g1/example.yaml \
  --pose_file ../southgrid/unitree_g1/my_waypoint_button1.yaml \
  --lerobot_out ~/datasets/sg_g1_button_scripted \
  --repo_id local/g1_pick_button \
  --task "按红色按钮" --fps 20 --clock sim \
  --dls_lambda 0.23 --dls_sigma_th 0.12 --track_ki 0.02
```

### 3.5 脚本化回放

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python data_collection_replay_lerobot.py \
  --agent_name g1_pick --level example \
  --task_config ../southgrid/unitree_g1/example.yaml \
  --lerobot_out ~/datasets/sg_g1_button_scripted \
  --episode_index 0 --dls_lambda 0.23 --dls_sigma_th 0.12
```

### 3.6 推理

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/inference
python infer_lerobot.py \
  --agent_name g1_pick --level example \
  --task_config ../southgrid/unitree_g1/example.yaml \
  --host 127.0.0.1 --port 8010 --prompt "按红色按钮"
```

---

## 4. Unitree G1 · 工具

OrcaLab 加载 `g1_pick_tools.json`。

### 4.1 遥操采集

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python data_collection_tele_lerobot.py \
  --agent_name g1_pick --level example \
  --task_config ../southgrid/unitree_g1/example.yaml \
  --lerobot_out ~/datasets/sg_g1_tool_tele \
  --repo_id local/g1_pick_tool \
  --task "整理工具" --fps 20 --clock wall
```

### 4.2 遥操回放

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python data_collection_replay_lerobot.py \
  --agent_name g1_pick --level example \
  --task_config ../southgrid/unitree_g1/example.yaml \
  --lerobot_out ~/datasets/sg_g1_tool_tele \
  --episode_index 0 --dls_lambda 0.23 --dls_sigma_th 0.12
```

### 4.3 标点

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python record_waypoints.py --agent_name g1_pick --task tool \
  --task_config ../southgrid/unitree_g1/example.yaml --guide tool6
```

仓库自带路点：`../southgrid/unitree_g1/my_waypoint_tool1.yaml`。

### 4.4 脚本化采集

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python data_collection_scripted_lerobot.py \
  --agent_name g1_pick --level example \
  --task_config ../southgrid/unitree_g1/example.yaml \
  --pose_file ../southgrid/unitree_g1/my_waypoint_tool1.yaml \
  --lerobot_out ~/datasets/sg_g1_tool_scripted \
  --repo_id local/g1_pick_tool \
  --task "整理工具" --fps 20 --clock sim \
  --dls_lambda 0.23 --dls_sigma_th 0.12 --track_ki 0.02 --grasp_integral
```

### 4.5 脚本化回放

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/dataCollection
python data_collection_replay_lerobot.py \
  --agent_name g1_pick --level example \
  --task_config ../southgrid/unitree_g1/example.yaml \
  --lerobot_out ~/datasets/sg_g1_tool_scripted \
  --episode_index 0 --dls_lambda 0.23 --dls_sigma_th 0.12 --grasp_integral
```

### 4.6 推理

```bash
cd /home/dht/hebing/OrcaManipulation/src/examples/inference
python infer_lerobot.py \
  --agent_name g1_pick --level example \
  --task_config ../southgrid/unitree_g1/example.yaml \
  --host 127.0.0.1 --port 8010 --prompt "整理工具"
```

---

## 已知注意事项

| # | 问题 | 影响 |
|---|------|------|
| 1 | `data_collection_replay_lerobot.py` line 102：`SceneManager("localhost:50051", ...)` 硬编码，不读 `--orcagym_addr` | 当前布局在 50051，无影响；换端口需手动改代码 |
| 2 | `data_collection_tele_lerobot.py` 遥操采集 g1_pick 时 `pico_transform=None`，使用默认 Pico 坐标映射（SouthGrid 参考脚本有 g1_omnipicker 专用变换） | g1_omnipicker 已有专用变换；g1_pick 遥操精度可能略低，需实测调整 |
| 3 | `--save_policy` 默认 `on_success`；工具任务的 `ToolPlaceTask` 未在通用脚本中使用（用 `EmptyTask` 替代），任务状态自动通过 `add_task_status_autostart_controller` 开启 | 脚本化采集一轮完成后自动保存，不影响功能 |

---

## 进度清单

| # | 组合 | 遥操采集 | 遥操回放 | 标点 | 脚本化采集 | 脚本化回放 | 推理 |
|---|------|:--------:|:--------:|:----:|:----------:|:----------:|:----:|
| 1 | OmniPicker 按钮 | | | | | | |
| 2 | OmniPicker 工具 | | | | | | |
| 3 | Unitree G1 按钮 | | | | | | |
| 4 | Unitree G1 工具 | | | | | | |

不同任务请用不同 `--lerobot_out`。推理前策略服务 checkpoint 须与机型 schema / `--prompt` 一致。
