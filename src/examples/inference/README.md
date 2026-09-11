# 通用在线推理

本目录接收用 LeRobot 数据集训练出来的策略，把观测送到策略服务，再把动作接到仿真控制器。比赛任务的 prompt、布局和机型约束留在 `examples/southgrid/inference/`。

```
obs → PolicySchema.build_state
    → PolicyClient.infer_action_chunk
    → PolicySchema.parse_action
    → PolicyDevice.update → Controller
```

| 路径 | 内容 |
|------|------|
| `agents.py` | 机器人接口注册表：conf / storage / schema / 夹爪类型 |
| `infer_lerobot.py` | 通用装配入口 |

核心接口在 `src/policy/` 与 `src/devices/policy_device.py`，说明见 [docs/policy_inference.md](../../../docs/policy_inference.md)。

## 已注册机型

| `--agent_name` | schema | storage | 反向夹爪 |
|----------------|--------|---------|----------|
| `openloong` | `OpenLoongPolicySchema` | `OpenLoongLeRobotStorage` | 否 |
| `tiangong2` | `Tiangong2PolicySchema` | `Tiangong2LeRobotStorage` | 否 |
| `g1_omnipicker` | `g1_omnipicker_schema`（18 维） | `G1OmniPickerLeRobotStorage` | 是 |
| `g1_pick` | `g1_pick_osc_schema`（18 维） | `G1PickOscLeRobotStorage` | 是 |

新增机型：在 `agents.py` 的 `AGENTS` 加一条；若需要钉关节，由业务 wrapper 传入 `constraints_hook`，不要让本目录 import `examples/southgrid`。

## 直接运行

需先启动 OrcaLab，并另开终端启动 OpenPI 策略服务（见 [openpi_deployment.md](../../../docs/openpi_deployment.md)）。

```bash
conda activate orcalab_lerobot
cd src/examples/inference

python infer_lerobot.py \
  --agent_name g1_omnipicker \
  --level default \
  --task_config ../southgrid/configs/example.yaml \
  --host 127.0.0.1 \
  --port 8010 \
  --prompt "按红色按钮" \
  --max_steps 500 \
  --episodes 1
```

宇树机型需要钉浮动基座、腰和左臂。通用入口只提供 `--pin_joints` 开关，具体 pin 由 wrapper 传入：

```bash
cd src/examples/southgrid/inference/unitree_g1
python eval_g1_pick_lerobot.py --host 127.0.0.1 --port 8010 --prompt "整理工具"
```

## 参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--agent_name` | 注册表中的机型 | 必填 |
| `--level` | 场景档位 | 必填 |
| `--task_config` | 任务 YAML | 必填 |
| `--host` | 策略服务器主机 | `127.0.0.1` |
| `--port` | 策略服务器端口 | `8000` |
| `--prompt` | 语言指令，须与数据集 `meta/tasks.jsonl` 一致 | `robot manipulation` |
| `--cameras` | 相机别名，逗号分隔 | `head,wrist_r` |
| `--orcagym_addr` | OrcaGym 地址 | `localhost:50051` |
| `--max_episodes` / `--episodes` | 评估集数 | `1` |
| `--max_steps` | 每集最大控制步数 | `500` |
| `--action_repeat` | 同一动作连续下发的控制步数 | `1` |
| `--camera_warmup_steps` | 每集推理前拉相机帧的次数 | `10` |
| `--no_images` | 向策略发送空图 | 未启用 |
| `--preview` | 弹出相机预览窗口 | 未启用 |
| `--no_realtime` | 关闭按 `real_time_step` 补齐的控制周期 | 未启用 |
| `--pin_joints` | `env.reset` 后调用 `constraints_hook` | 未启用 |
| `--kp` 等 OSC 参数 | 见 `controllers.add_osc_tuning_args` | `g1_omnipicker` 默认 `--kp 220` |
| `--grasp_integral` 等 | 近桌右臂外环积分 | 未启用 |

`--preview` 默认关闭。需要看画面时显式加上。`--sleep` / `--no_preview` 已不存在。
