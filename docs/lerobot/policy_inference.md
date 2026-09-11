# 策略推理接口

在线推理把 OpenPI WebSocket 策略接到 `DataCollectionManager` 的控制器上。训练写入与推理读取共用同一个 `PolicySchema`。

```
obs → PolicySchema.build_state
    → PolicyClient.infer_action_chunk
    → PolicySchema.parse_action
    → PolicyDevice.update → Controller
```

## PolicySchema

抽象定义在 `src/policy/schema.py`。新增机器人时实现：

1. `build_state(obs)`：与采集时写入 parquet 的向量一致
2. `parse_action(raw)`：与 `build_state` 互逆（夹爪归一化 / 反归一化必须成对）

不要在 eval 脚本里再写一份 `_denorm_grip`。

## PolicyClient

`src/policy/client.py` 封装 `openpi_client.WebsocketClientPolicy`。控制循环里**手动遍历** action chunk，不使用 `ActionChunkBroker`，以免改变原有时序。

```python
from policy.client import PolicyClient
from policy.dual_arm_schema import g1_omnipicker_schema

schema = g1_omnipicker_schema()
client = PolicyClient(
    host="localhost",
    port=8010,
    prompt="按红色按钮",  # 必须与数据集 meta/tasks.jsonl 一致
    camera_name_map={"camera_head_color": "cam_head"},
    cameras=cameras,
    schema=schema,
)
chunk = client.infer_action_chunk(state)
```

`CameraObservationBuilder` 把 WebSocket 相机帧整理成策略需要的 CHW 图像。

## PolicyDevice

`src/devices/policy_device.py` 在 `update()` 时把 `parse_action` 的字段分发给已 `bind` 的控制器：

```python
from devices.policy_device import PolicyDevice

device = PolicyDevice(schema)
device.bind("l_pos_b", l_arm.update_action_position)
device.bind("l_quat_b", l_arm.update_action_axisangle)
device.bind("r_pos_b", r_arm.update_action_position)
device.bind("r_quat_b", r_arm.update_action_axisangle)
device.bind("l_grip_ctrl", l_grip.update_ctrl)
device.bind("r_grip_ctrl", r_grip.update_ctrl)
device.set_raw_action(model_action)
```

不要改通用 `PolicyDevice` 来做机型约束。需要钉关节时，由业务 wrapper 把钩子传给通用入口的 `main(constraints_hook=...)`，再开 `--pin_joints`。近桌外环积分走 `--grasp_integral`，由 `controllers.setup_grasp_integral` 包一层位置回调。

## 通用入口

装配入口是 `src/examples/inference/infer_lerobot.py`，机型注册在 `src/examples/inference/agents.py`。当前可用 `openloong` / `tiangong2` / `g1_omnipicker` / `g1_pick`。

```bash
cd src/examples/inference
python infer_lerobot.py \
  --agent_name g1_omnipicker \
  --level default \
  --task_config ../southgrid/configs/example.yaml \
  --host 127.0.0.1 --port 8010 \
  --prompt "按红色按钮"
```

比赛任务的 prompt 与布局注入见 `src/examples/southgrid/inference/`。完整参数表见 [examples/inference/README.md](../../src/examples/inference/README.md)。

## Manager 模式

推理循环把 `manager.mode` 设为 `DataCollectionManager.DataCollectionMode.INFERENCE`，UI 文案用 `inference_ui_message`。控制周期默认按 `real_time_step` 补齐，可用 `manager.realtime_pacing = False` 或入口的 `--no_realtime` 关闭。

训练与部署服务端流程见 [openpi_deployment.md](openpi_deployment.md)。
