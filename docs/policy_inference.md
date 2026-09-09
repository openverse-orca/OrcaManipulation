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

比赛任务若要锁定某条臂或加积分门控，可在示例层包一层 Device（见 `examples/southgrid/inference/` 工具整理脚本），不要改通用 `PolicyDevice`。

## Manager 模式

推理循环把 `manager.mode` 设为 `DataCollectionManager.DataCollectionMode.INFERENCE`，UI 文案用 `inference_ui_message`。

训练与部署服务端流程见 [openpi_deployment.md](openpi_deployment.md)。
