# OrcaManipulation 二次开发指南

本文档说明如何基于 OrcaManipulation 框架进行二次开发。框架提供了核心的数据采集流程，您只需实现特定组件即可快速搭建自己的机器人数据采集系统。

---

## 📋 目录

- [快速开始](#快速开始)
- [核心概念](#核心概念)
- [必须实现的组件](#必须实现的组件)
  - [1. 数据存储器 (DataStorage)](#1-数据存储器-datastorage)
  - [1b. LeRobot 数据存储](#1b-lerobot-数据存储)
  - [1c. 策略 Schema](#1c-策略-schema)
  - [2. 控制器配置 (Controllers)](#2-控制器配置-controllers)
  - [3. 场景配置 (Scene Config)](#3-场景配置-scene-config)
  - [4. 任务定义 (Task)](#4-任务定义-task)
- [可选实现的组件](#可选实现的组件)
  - [5. 插值器 (Interpolator)](#5-插值器-interpolator)
  - [6. 自定义控制器 (Custom Controller)](#6-自定义控制器-custom-controller)
  - [7. 自定义设备 (Custom Device)](#7-自定义设备-custom-device)
- [完整示例](#完整示例)

---

## 快速开始

框架支持三种数据采集模式：

1. **TELECONTROL 模式**：通过 VR 手柄遥控机器人采集真实演示数据
2. **AUGMENTATION 模式**：读取已有数据并回放，应用插值和噪声增强数据集
3. **INFERENCE 模式**（可选）：用 `PolicyDevice` 在线执行策略，主循环仍是 `manager.run()`

参考示例：
- `examples/dataCollection/data_collection_tele.py` - 遥控采集（HDF5）
- `examples/dataCollection/data_collection_tele_lerobot.py` - 遥控采集（LeRobot）
- `examples/dataCollection/data_collection_scripted.py` - 脚本化采集（HDF5）
- `examples/dataCollection/data_collection_scripted_lerobot.py` - 脚本化采集（LeRobot）
- `examples/dataCollection/data_collection_aug.py` - 数据增强示例
- `examples/dataCollection/data_collection_infer.py` - 在线推理示例

---

## 核心概念

```
DataCollectionManager (核心调度器)
    ├── Device          # 输入设备 (VR手柄/数据回放)
    ├── Controllers     # 控制器列表 (臂/夹爪/任务状态)
    ├── SceneManager    # 场景管理 (物体/光照随机化)
    ├── Task            # 任务定义 (成功判定/任务信息)
    └── DataStorage     # 数据存储 (观测采集/HDF5保存)
```

---

## 必须实现的组件

### 1. 数据存储器 (DataStorage)

**作用**: 定义采集什么数据、如何保存数据

**继承**: `AbstractDataStorage`

**必须实现的方法**:

```python
from dataStorage.abstract_data_storage import AbstractDataStorage
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
import numpy as np
import h5py

class MyDataStorage(AbstractDataStorage):
    def __init__(self, dataset_path: str, hdf5_path: str):
        super().__init__(dataset_path=dataset_path, hdf5_path=hdf5_path)
        # 初始化数据容器
        self.data["time_step"] = []
    
    def obs_callback(self, env: OrcaGymLocalEnv) -> dict:
        """
        定义每一步采集的观测数据
        返回: dict, key 为数据路径 (如 "/action/joint/position")
        """
        obs = {}
        # 示例: 采集关节位置
        joint_names = [env.joint(name) for name in ["joint1", "joint2"]]
        qpos = env.query_joint_qpos(joint_names)
        obs["/action/joint/position"] = np.array([qpos[j] for j in joint_names])
        
        # 示例: 采集末端位姿
        ee_site = env.site("ee_site")
        ee_info = env.query_site_pos_and_quat([ee_site])[ee_site]
        obs["/action/end/position"] = ee_info["xpos"]
        obs["/action/end/orientation"] = ee_info["xquat"]
        
        return obs
    
    def collection_data(self, data: dict, env: OrcaGymLocalEnv, **kwargs):
        """
        将每一步的观测数据加入缓存
        data: obs_callback 返回的数据
        """
        for key, value in data.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
        self.data["time_step"].append(env.data.time)
    
    def _save_data(self, **kwargs):
        """
        将缓存的数据保存到 HDF5 文件
        """
        import os
        os.makedirs(self.get_current_unit_path(), exist_ok=True)
        
        hdf5_path = self.get_hdf5_absolute_path()
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
        
        with h5py.File(hdf5_path, 'w') as f:
            for key, value in self.data.items():
                self.create_dataset(f, key, data=np.array(value), 
                                  compression="gzip", compression_opts=4)
    
    def clear_data(self):
        """清空缓存"""
        super().clear_data()
        self.data["time_step"] = []
```

**关键点**:
- `obs_callback`: 定义**采集什么数据** (关节、末端、力传感器等)
- `collection_data`: 数据如何**暂存**
- `_save_data`: 数据如何**持久化**到 HDF5
- 数据路径 (如 `/action/joint/position`) 建议使用层级结构，方便后续处理

**参考**: `src/dataStorage/openloong_data_storage.py`

### 1b. LeRobot 数据存储

把 `obs_callback` 的观测写成 LeRobot v2.1 时，不要再在 Storage 里实现 `build_state`，也不要在入口脚本里创建 writer / 拉相机。组合方式：

1. 实现或选用一个 `PolicySchema`
2. HDF5 Storage 继续负责 `obs_callback`（默认路径不变）
3. LeRobot Storage 在构造时 `setup_lerobot(...)`，由 `manager.run()` 调用 `open_capture_session` / `close_capture_session` 管理相机与 writer

```python
from dataStorage.g1_lerobot_storage import G1OmniPickerLeRobotStorage

storage = G1OmniPickerLeRobotStorage(
    dataset_path=scratch_dir,
    repo_id="local/g1_omnipicker",
    root=lerobot_out,
    fps=20,
    camera_map=camera_map,
    task="robot manipulation",
)
manager = DataCollectionManager(..., data_storage=storage)
manager.save_policy = "on_success"  # 可选 "always"
manager.run()
```

通用入口：`data_collection_tele_lerobot.py --lerobot_out ...`。

约定与目录结构见 [docs/lerobot_dataset.md](docs/lerobot_dataset.md)；相机端口见 [docs/cameras_and_video.md](docs/cameras_and_video.md)。

### 1c. 策略 Schema

`src/policy/schema.py` 的 `PolicySchema` 同时定义写入数据集的状态向量和推理时的动作反变换。训练与推理必须共用同一个实现。

```python
from policy.dual_arm_schema import g1_omnipicker_schema
from policy.client import PolicyClient
from devices.policy_device import PolicyDevice

schema = g1_omnipicker_schema()
client = PolicyClient(host, port, prompt, camera_name_map, cameras, schema=schema)
device = PolicyDevice(schema)
device.bind("r_pos_b", r_arm.update_action_position)
device.set_raw_action(client.infer_action_chunk(state)[0])
```

接口说明见 [docs/policy_inference.md](docs/policy_inference.md)。

---

### 2. 控制器配置 (Controllers)

**作用**: 配置机器人使用的控制器（臂控制器、夹爪控制器等）

**两种方式**:

#### 方式 A: 使用框架提供的控制器

框架已提供常用控制器：
- `add_arm_osc_pico_controller` - OSC臂控制器 (VR手柄)
- `add_arm_osc_openloong_data_controller` - OSC臂控制器 (数据回放)
- `add_gripper_2f85_pico_controller` - 2F85夹爪控制器 (VR手柄)
- `add_gripper_2f85_openloong_data_controller` - 2F85夹爪控制器 (数据回放)
- `add_task_status_pico_controller` - 任务状态控制器 (VR手柄)
- `add_task_status_openloong_data_controller` - 任务状态控制器 (数据回放)
- `add_episode_control_pico_controller` - 可选：右 Grip 丢弃本集 / 双 Grip 终止（不装即无此行为）
- `add_joint_hold_controller` - 可选：把指定关节锁在参考位姿

**使用示例** (TELECONTROL 模式):

```python
from controllers import controllers

# 添加左臂控制器
controllers.add_arm_osc_pico_controller(
    data_collection_manager, env, 
    arm_config=left_arm_config,  # 臂配置字典
    base_body="base",
    device=pico_joystick_device,
    key=PicoJoystickKey.L_TRANSFORM  # 左手柄
)

# 添加左夹爪控制器
controllers.add_gripper_2f85_pico_controller(
    data_collection_manager, env,
    gripper_config=gripper_config,
    base_body="base",
    device=pico_joystick_device,
    keys=[PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER]
)
```

**使用示例** (AUGMENTATION 模式):

```python
# 添加左臂控制器 (从数据读取)
controllers.add_arm_osc_openloong_data_controller(
    data_collection_manager, env,
    arm_config=left_arm_config,
    base_body="base",
    device=data_device,  # DataDevice
    left_arm=True
)
```

**机器人配置示例**:

```python
# conf/robot_conf.py
left_arm_config = {
    "joint_names": ["l_arm_joint1", "l_arm_joint2", ...],
    "neutral_joint_values": [0.0, -1.57, ...],
    "motors_names": ["l_arm_motor1", "l_arm_motor2", ...],
    "motors_init_ctrl": [0.0, 0.0, ...],
    "motors_ranges": [[-2.0, 2.0], ...],
    "ee_site_name": "l_ee_site"
}

gripper_config = {
    "joint_names": ["l_gripper_joint1", "l_gripper_joint2"],
    "actuator_names": ["l_gripper_motor"],
    "init_ctrl": [0.0],
    "actuator_ranges": [[0.0, 1.0]]
}
```

#### 方式 B: 自定义控制器

如果需要特殊控制逻辑，可以实现自定义控制器 (见[可选组件](#6-自定义控制器-custom-controller))

**参考**: `src/controllers/controllers.py`, `src/conf/openloong_conf.py`

---

### 3. 场景配置 (Scene Config)

**作用**: 定义场景中的物体、光照及其随机化策略

**格式**: YAML 文件

**配置示例**:

```yaml
# example.yaml
level_name: "my_scene"
type: "pick_and_place"

# 物体配置
actor:
  names: ["cube1", "cube2", "sphere1"]
  spawnable: ["assets/prefabs/cube", "assets/prefabs/cube", "assets/prefabs/sphere"]
  joints: ["cube1_joint", "cube2_joint", "sphere1_joint"]
  joints_dof: [6, 6, 6]  # 6-DOF: 位置+旋转
  
  random:
    qpos: true  # 启用位置随机化
    nums: [1, 2]  # 每次随机生成 1-2 个物体
    
    # 6-DOF 随机范围
    six_dof:
      center: [0.5, 0.0, 1.0]  # 中心点
      bound_position: [[-0.2, 0.2], [-0.2, 0.2], [0, 0]]  # xyz偏移
      bound_rotation: [[0, 0], [0, 0], [0, 3.14159]]  # xyz旋转角度

# 光照配置
light:
  names: ["spot1", "spot2"]
  spawnable: ["prefabs/spotlight", "prefabs/spotlight"]
  random:
    position: true
    rotation: false
    center: [0.5, 0.0, 2.0]
    bound_position: [[-1, 1], [-1, 1], [0, 0.5]]
    nums: [1, 2]
    cycle: 20  # 每 20 个 episode 更新一次光照

# 任务配置
task:
  type: "pick_and_place"
  goal:
    name: "GoalBox"
    site: "goal_site"
```

**关键参数**:
- `joints_dof`: 物体自由度
  - `6`: 完整 6-DOF (位置+旋转)
  - `3`: 3-DOF 球形关节
  - `1`: 1-DOF 旋转关节
- `random.nums`: `[min, max]` 每次随机生成的物体数量
- `center + bound_position`: 定义物体的随机生成范围
- `cycle`: 光照更新周期 (避免每个 episode 都重载场景)

**参考**: `src/examples/dataCollection/example.yaml`

---

### 4. 任务定义 (Task)

**作用**: 定义任务目标、成功判定逻辑、任务元数据

**继承**: `AbstractTask`

**必须实现的方法**:

```python
from task.abstract_task import AbstractTask
from scene.scene_manager import SceneManager
import numpy as np

class MyPickPlaceTask(AbstractTask):
    def __init__(self, env):
        super().__init__(env)
        self.target_object = None
        self.goal_position = None
    
    def _get_task(self, scene_manager: SceneManager, task_info: dict = None) -> bool:
        """
        获取任务 (从场景或 task_info 恢复)
        返回: True 表示任务有效 (未完成), False 表示任务已完成
        """
        if task_info is not None:
            # 数据增强模式: 从 task_info 恢复任务
            self.target_object = task_info["target_object"]
            self.goal_position = task_info["goal_position"]
        else:
            # 遥控模式: 从场景随机选择目标物体
            scene_info = scene_manager.get_scene_info()
            self.target_object = np.random.choice(list(scene_info.keys()))
            task_config = scene_manager.get_task_config()
            self.goal_position = task_config["goal"]["site"]
        
        return not self.is_success()  # 任务未完成返回 True
    
    def is_success(self) -> bool:
        """
        判断任务是否成功
        """
        # 示例: 判断物体是否在目标区域内
        obj_pos = self.env.query_joint_qpos([self.target_object])[self.target_object][:3]
        goal_pos = self.env.query_site_pos([self.goal_position])[self.goal_position]
        distance = np.linalg.norm(obj_pos - goal_pos)
        return distance < 0.1  # 10cm 以内认为成功
    
    def get_task_description(self) -> str:
        """
        返回任务描述 (用于日志)
        """
        return f"Pick {self.target_object} and place it to {self.goal_position}"
    
    def get_task_info(self) -> dict:
        """
        返回任务元数据 (保存到 HDF5)
        """
        return {
            "target_object": self.target_object,
            "goal_position": self.goal_position
        }
```

**关键点**:
- `_get_task`: 支持两种模式
  - `task_info is None`: TELECONTROL 模式，从场景获取任务
  - `task_info is not None`: AUGMENTATION 模式，从数据恢复任务
- `is_success`: 判定任务成功的核心逻辑
- `get_task_info`: 返回的数据会保存到 HDF5，在数据增强时用于恢复任务

**参考**: `src/task/pick_place_task.py`

---

## 可选实现的组件

### 5. 插值器 (Interpolator)

**作用**: 对采集的数据进行插值增强，生成更多训练样本

**继承**: `AbstractInterpolator`

**使用场景**: 在 AUGMENTATION 模式下使用

**实现示例**:

```python
from devices.Interpolator.abstract_interpolator import AbstractInterpolator
import numpy as np

class MyInterpolator(AbstractInterpolator):
    def __init__(self, noise_value: float, interpolation_factor: int = 2):
        """
        @param noise_value: 噪声强度
        @param interpolation_factor: 插值倍数 (每两个点之间插入的点数)
        """
        super().__init__(noise_value)
        self.factor = interpolation_factor
    
    def get_interpolation_paths(self) -> list[str]:
        """
        返回需要插值的数据集路径 (必须与 DataStorage 中的路径一致)
        """
        return [
            "/action/joint/position",
            "/action/end/position",
            "/action/end/orientation"  # 四元数
        ]
    
    def interpolate(self, dataset: np.array, **kwargs) -> np.array:
        """
        对数据进行插值
        @param dataset: 原始数据 (N, D)
        @param kwargs: 其他参数 (如 dataset_path)
        @return: 插值后的数据 (N', D)
        """
        dataset_path = kwargs.get("dataset_path")
        
        if "orientation" in dataset_path:
            # 四元数使用 SLERP 插值
            return self._slerp_interpolate(dataset)
        else:
            # 其他数据使用线性插值
            return self._linear_interpolate(dataset)
    
    def _linear_interpolate(self, dataset: np.array) -> np.array:
        """线性插值"""
        n = len(dataset)
        result = []
        for i in range(n - 1):
            result.append(dataset[i])
            for j in range(1, self.factor):
                t = j / self.factor
                interpolated = (1 - t) * dataset[i] + t * dataset[i + 1]
                # 添加噪声
                noise = np.random.uniform(-self.noise_value, self.noise_value, 
                                         size=interpolated.shape)
                result.append(interpolated + noise)
        result.append(dataset[-1])
        return np.array(result)
    
    def _slerp_interpolate(self, dataset: np.array) -> np.array:
        """球面线性插值 (用于四元数)"""
        # 实现 SLERP 逻辑 (见 abstract_interpolator.py 中的示例)
        pass
```

**使用方式**:

```python
# 在 AUGMENTATION 模式下创建 DataDevice 时传入
from devices.data_device import DataDevice

data_device = DataDevice(
    dataset_path="dataset/",
    hdf5_path="record/data.hdf5",
    interpolator=MyInterpolator(noise_value=0.03, interpolation_factor=2)
)
```

**关键点**:
- 不同类型的数据需要不同的插值方法:
  - **位置/关节**: 线性插值或三次样条插值
  - **四元数**: SLERP (球面线性插值)
  - **离散值** (如夹爪开合): 最近邻或取整
- 添加适量噪声可以提升策略鲁棒性

**参考**: `src/devices/Interpolator/abstract_interpolator.py`

---

### 6. 自定义控制器 (Custom Controller)

**作用**: 实现特殊的控制逻辑 (如阻抗控制、混合控制等)

**继承**: `AbstractController`

**实现示例**:

```python
from controllers.abstract_controller import AbstractController
import numpy as np

class MyCustomController(AbstractController):
    def __init__(self, env, ctrl_name: list[str], init_ctrl: dict[str, float], base_body: str):
        super().__init__(env, ctrl_name, init_ctrl, base_body)
        # 初始化控制器状态
        self.target_position = np.zeros(3)
    
    def run_controller(self) -> dict[int, float]:
        """
        运行控制器，返回控制指令
        返回: {actuator_id: control_value}
        """
        # 实现控制逻辑
        current_pos = self._get_current_position()
        error = self.target_position - current_pos
        
        # 简单 PD 控制
        kp = 1.0
        control = kp * error
        
        # 映射到执行器
        result = {}
        for i, ctrl_idx in enumerate(self.ctrl_index):
            result[ctrl_idx] = control[i] if i < len(control) else 0.0
        
        return result
    
    def update_target(self, position: np.array):
        """更新目标位置 (由 Device 调用)"""
        self.target_position = position
```

**集成到框架**:

```python
# 创建控制器
my_controller = MyCustomController(env, ctrl_name, init_ctrl, base_body)

# 绑定设备输入
device.bind_transform_event(key, my_controller.update_target)

# 添加到管理器
data_collection_manager.add_controller(my_controller)
```

**参考**: `src/controllers/controller_arm.py`, `src/controllers/controller_2f85.py`

---

### 7. 自定义设备 (Custom Device)

**作用**: 支持新的输入设备 (如键盘、鼠标、3D鼠标等)

**继承**: `AbstractDevice`

**实现示例**:

```python
from devices.abstract_device import AbstractDevice
import numpy as np

class KeyboardDevice(AbstractDevice):
    def __init__(self):
        super().__init__()
        self.key_events = {}  # {key: callback}
        self.current_position = np.zeros(3)
    
    def update(self):
        """
        更新设备状态 (轮询或事件驱动)
        """
        # 读取键盘输入
        keys = self._read_keyboard()  # 实现具体的键盘读取
        
        # 触发事件
        for key, callback in self.key_events.items():
            if key in keys:
                callback(self.current_position, None)
    
    def bind_key_event(self, key: str, callback):
        """绑定按键事件"""
        self.key_events[key] = callback
```

**参考**: `src/devices/abstract_device.py`

---

## 完整示例

### TELECONTROL 模式 (VR 遥控采集)

```python
import os
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from scene.scene_manager import SceneManager
from task.pick_place_task import PickPlaceTask
from controllers import controllers
from dataStorage.my_data_storage import MyDataStorage
from yaml import load, Loader

# 配置
ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
orcagym_addr = "localhost:50051"
base_dir = os.path.dirname(__file__)

# 1. 创建设备
device = PicoJoystickDevice(PicoJoystick())

# 2. 创建场景管理器
with open(os.path.join(base_dir, "scene_config.yaml"), "r") as f:
    config = load(f, Loader=Loader)
scene_manager = SceneManager(orcagym_addr, config=config)

# 3. 创建数据存储
data_storage = MyDataStorage(
    dataset_path=os.path.join(base_dir, "dataset"),
    hdf5_path="record/data.hdf5"
)
data_storage.set_video_path("video")

# 4. 创建数据采集管理器
manager = DataCollectionManager(
    agent_name="my_robot",
    env_name="DataCollection",
    entry_point=ENTRY_POINT,
    default_joint_values={"joint1": 0.0, "joint2": 0.0},
    obs_callback=data_storage.obs_callback,
    device=device,
    scene_manager=scene_manager,
    data_storage=data_storage,
)

# 5. 添加控制器
controllers.add_arm_osc_pico_controller(
    manager, manager.env, arm_config, "base", device, PicoJoystickKey.R_TRANSFORM
)
controllers.add_gripper_2f85_pico_controller(
    manager, manager.env, gripper_config, "base", device, 
    [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER]
)

# 6. 设置任务
manager.set_task(PickPlaceTask(manager.env))
controllers.add_task_status_pico_controller(manager, manager.env, device, "base")

# 7. 运行
manager.mode = DataCollectionManager.DataCollectionMode.TELECONTROL
manager.save_video = True
manager.run()
```

### AUGMENTATION 模式 (数据增强)

```python
from devices.data_device import DataDevice
from devices.Interpolator.abstract_interpolator import MyInterpolator

# 1. 创建数据设备 (替代 VR 设备)
data_device = DataDevice(
    dataset_path=os.path.join(base_dir, "dataset"),  # 已有数据集
    hdf5_path="record/data.hdf5",
    interpolator=MyInterpolator(noise_value=0.03)  # 可选: 插值器
)

# 2. 其他组件与 TELECONTROL 模式类似
# ...

# 3. 使用数据控制器 (替代 Pico 控制器)
controllers.add_arm_osc_openloong_data_controller(
    manager, manager.env, arm_config, "base", data_device, left_arm=True
)

# 4. 运行
manager.mode = DataCollectionManager.DataCollectionMode.AUGMENTATION
manager.save_video = True
manager.run()
```

---

## 总结

### 必须实现

| 组件 | 作用 | 文件位置 |
|------|------|---------|
| DataStorage | 定义采集数据和保存方式 | `dataStorage/` |
| Controllers | 配置机器人控制器 | 配置字典 或 `controllers/` |
| Scene Config | 配置场景和随机化策略 | YAML 文件 |
| Task | 定义任务和成功判定 | `task/` |

### 可选实现

| 组件 | 作用 | 使用场景 |
|------|------|---------|
| Interpolator | 数据插值增强 | AUGMENTATION 模式 |
| Custom Controller | 自定义控制逻辑 | 特殊控制需求 |
| Custom Device | 新输入设备支持 | 非 VR 设备 |

---

## 常见问题

### Q1: 如何调试控制器？

在 `run_controller` 中添加日志：

```python
def run_controller(self):
    orca_logger.debug(f"Current position: {self._get_position()}")
    # ...
```

### Q2: 数据采集频率如何控制？

在 `DataCollectionManager` 初始化时设置：

```python
DataCollectionManager(
    frame_skip=20,      # 仿真步数
    time_step=0.001,    # 仿真时间步长
    # 实际控制频率 = 1 / (frame_skip * time_step) = 50 Hz
)
```

### Q3: 如何查看 HDF5 数据？

```python
import h5py

with h5py.File("dataset/xxx/record/data.hdf5", "r") as f:
    print(f.keys())  # 查看所有数据集
    position = f["/action/joint/position"][:]  # 读取数据
    print(position.shape)
```

### Q4: 插值后数据长度不一致怎么办？

确保插值器对所有数据使用相同的插值策略。参考 `OpenLoongInterpolator`，使用 `save_indices` 保持一致性。

---

## 注释与 docstring 约定

- **新增与迁入的代码**使用 Google 中文风格（`Args:` / `Returns:` / `Raises:`）。
- **不要**为统一风格去改未改动的老文件。老代码里的 `@description` / `@param` / `@return` 可以保留。
- 比赛任务专属说明（中文 prompt、布局名、按键）写在 `src/examples/southgrid/`，不要写进通用模块。

## 更多资源

- 示例代码: `src/examples/dataCollection/`
- 南网示例: `src/examples/southgrid/`
- 参考实现: `src/dataStorage/openloong_data_storage.py`
- 控制器示例: `src/controllers/`
- 任务示例: `src/task/pick_place_task.py`
- LeRobot / 推理 / 相机: `docs/lerobot_dataset.md`、`docs/policy_inference.md`、`docs/cameras_and_video.md`

如有问题，请参考源码或提交 Issue。

