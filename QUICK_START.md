# 快速开始指南

基于 OrcaManipulation 框架快速搭建数据采集系统的最小步骤。

---

## 🚀 五步搭建数据采集系统

### 第 1 步: 实现数据存储器

```python
# dataStorage/my_robot_storage.py
from dataStorage.abstract_data_storage import AbstractDataStorage
import numpy as np
import h5py

class MyRobotStorage(AbstractDataStorage):
    def __init__(self, dataset_path: str, hdf5_path: str):
        super().__init__(dataset_path=dataset_path, hdf5_path=hdf5_path)
        self.data["time_step"] = []
    
    def obs_callback(self, env):
        """定义采集什么数据"""
        obs = {}
        # TODO: 添加你需要采集的数据
        # 示例: obs["/action/joint/position"] = env.query_joint_qpos(...)
        return obs
    
    def collection_data(self, data: dict, env, **kwargs):
        """数据暂存"""
        for key, value in data.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
        self.data["time_step"].append(env.data.time)
    
    def _save_data(self, **kwargs):
        """保存到 HDF5"""
        import os
        os.makedirs(self.get_current_unit_path(), exist_ok=True)
        hdf5_path = self.get_hdf5_absolute_path()
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
        
        with h5py.File(hdf5_path, 'w') as f:
            for key, value in self.data.items():
                self.create_dataset(f, key, data=np.array(value), 
                                  compression="gzip", compression_opts=4)
    
    def clear_data(self):
        super().clear_data()
        self.data["time_step"] = []
```

### 第 2 步: 配置机器人控制器

```python
# conf/my_robot_conf.py
base_body = "base_link"

arm_config = {
    "joint_names": ["shoulder", "elbow", "wrist"],  # 关节名称
    "neutral_joint_values": [0.0, -1.57, 0.0],     # 初始位置
    "motors_names": ["shoulder_motor", "elbow_motor", "wrist_motor"],
    "motors_init_ctrl": [0.0, 0.0, 0.0],
    "motors_ranges": [[-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]],
    "ee_site_name": "ee_site"  # 末端执行器 site
}

gripper_config = {
    "joint_names": ["gripper_left", "gripper_right"],
    "actuator_names": ["gripper_motor"],
    "init_ctrl": [0.0],
    "actuator_ranges": [[0.0, 1.0]]
}
```

### 第 3 步: 配置场景

```yaml
# scene_config.yaml
level_name: "my_scene"
type: "pick_and_place"

actor:
  names: ["cube1", "cube2"]
  spawnable: ["assets/prefabs/cube", "assets/prefabs/cube"]
  joints: ["cube1_joint", "cube2_joint"]
  joints_dof: [6, 6]
  
  random:
    qpos: true
    nums: [1, 2]
    six_dof:
      center: [0.5, 0.0, 1.0]
      bound_position: [[-0.2, 0.2], [-0.2, 0.2], [0, 0]]
      bound_rotation: [[0, 0], [0, 0], [0, 3.14159]]

light:
  names: ["spot1"]
  spawnable: ["prefabs/spotlight"]
  random:
    position: false
    rotation: false
    center: [0, 0, 2]
    bound_position: [[-1, 1], [-1, 1], [0, 0]]
    nums: [1, 1]
    cycle: 20

task:
  type: "pick_and_place"
  goal:
    name: "GoalBox"
    site: "goal_site"
```

### 第 4 步: 实现任务

```python
# task/my_task.py
from task.abstract_task import AbstractTask
import numpy as np

class MyTask(AbstractTask):
    def __init__(self, env):
        super().__init__(env)
        self.target_object = None
    
    def _get_task(self, scene_manager, task_info=None):
        """获取任务"""
        if task_info is not None:
            # 数据增强模式
            self.target_object = task_info["target_object"]
        else:
            # 遥控模式
            scene_info = scene_manager.get_scene_info()
            self.target_object = list(scene_info.keys())[0]
        
        return not self.is_success()
    
    def is_success(self):
        """判断任务是否成功"""
        # TODO: 实现成功判定逻辑
        return False
    
    def get_task_description(self):
        return f"Pick {self.target_object}"
    
    def get_task_info(self):
        return {"target_object": self.target_object}
```

### 第 5 步: 编写采集脚本

```python
# data_collection.py
import os
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from scene.scene_manager import SceneManager
from controllers import controllers
from yaml import load, Loader

# 导入自定义组件
from dataStorage.my_robot_storage import MyRobotStorage
from task.my_task import MyTask
from conf import my_robot_conf

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
base_dir = os.path.dirname(__file__)

# 创建组件
device = PicoJoystickDevice(PicoJoystick())

with open("scene_config.yaml", "r") as f:
    config = load(f, Loader=Loader)
scene_manager = SceneManager("localhost:50051", config=config)

data_storage = MyRobotStorage(
    dataset_path=os.path.join(base_dir, "dataset"),
    hdf5_path="record/data.hdf5"
)
data_storage.set_video_path("video")

# 创建管理器
default_joint_values = dict(zip(
    my_robot_conf.arm_config["joint_names"],
    my_robot_conf.arm_config["neutral_joint_values"]
))

manager = DataCollectionManager(
    agent_name="my_robot",
    env_name="DataCollection",
    entry_point=ENTRY_POINT,
    default_joint_values=default_joint_values,
    obs_callback=data_storage.obs_callback,
    device=device,
    scene_manager=scene_manager,
    data_storage=data_storage,
)

# 添加控制器
controllers.add_arm_osc_pico_controller(
    manager, manager.env, 
    my_robot_conf.arm_config, 
    my_robot_conf.base_body,
    device, 
    PicoJoystickKey.R_TRANSFORM
)

controllers.add_gripper_2f85_pico_controller(
    manager, manager.env,
    my_robot_conf.gripper_config,
    my_robot_conf.base_body,
    device,
    [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER]
)

# 设置任务
manager.set_task(MyTask(manager.env))
controllers.add_task_status_pico_controller(
    manager, manager.env, device, my_robot_conf.base_body
)

# 运行
manager.mode = DataCollectionManager.DataCollectionMode.TELECONTROL
manager.save_video = True
manager.run()
```

---

## 🎮 控制说明

### VR 手柄操作

| 按键 | 功能 |
|------|------|
| 右手柄移动 | 控制机械臂末端位置 |
| A/B 按钮 | 控制夹爪开合 |
| 右扳机 | 夹爪闭合程度 |
| 左手柄握持按钮 | 开始/结束任务记录 |

### 数据保存

- 按下左手柄握持按钮开始记录
- 完成任务后再次按下结束记录
- 如果任务成功，数据自动保存到 `dataset/{uuid}/record/data.hdf5`
- 如果任务失败，数据自动丢弃

---

## 📊 数据增强 (可选)

如果要对采集的数据进行插值增强：

### 1. 实现插值器

```python
# devices/Interpolator/my_interpolator.py
from devices.Interpolator.abstract_interpolator import AbstractInterpolator
import numpy as np

class MyInterpolator(AbstractInterpolator):
    def __init__(self, noise_value: float):
        super().__init__(noise_value)
    
    def get_interpolation_paths(self):
        """返回需要插值的数据路径"""
        return ["/action/joint/position", "/action/end/position"]
    
    def interpolate(self, dataset: np.array, **kwargs):
        """实现插值逻辑"""
        # 简单线性插值示例
        n = len(dataset)
        result = []
        for i in range(n - 1):
            result.append(dataset[i])
            # 插入中间点
            mid = (dataset[i] + dataset[i + 1]) / 2
            noise = np.random.uniform(-self.noise_value, self.noise_value, mid.shape)
            result.append(mid + noise)
        result.append(dataset[-1])
        return np.array(result)
```

### 2. 修改采集脚本

```python
# data_collection_augmentation.py
from devices.data_device import DataDevice
from devices.Interpolator.my_interpolator import MyInterpolator

# 替换 PicoJoystickDevice 为 DataDevice
data_device = DataDevice(
    dataset_path=os.path.join(base_dir, "dataset"),  # 已有数据集
    hdf5_path="record/data.hdf5",
    interpolator=MyInterpolator(noise_value=0.03)
)

# 使用数据控制器
controllers.add_arm_osc_openloong_data_controller(
    manager, manager.env,
    my_robot_conf.arm_config,
    my_robot_conf.base_body,
    data_device,
    left_arm=True  # 根据你的配置调整
)

# 设置为增强模式
manager.mode = DataCollectionManager.DataCollectionMode.AUGMENTATION
```

---

## 📖 参考示例

完整示例请查看：
- `src/examples/dataCollection/data_collection_tele.py` - 遥控采集
- `src/examples/dataCollection/data_collection_aug.py` - 数据增强
- `src/dataStorage/openloong_data_storage.py` - 存储器参考
- `src/conf/openloong_conf.py` - 配置参考

详细文档: [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md)

---

## ❓ 常见问题

**Q: 如何查看采集的数据？**

```python
import h5py
with h5py.File("dataset/{uuid}/record/data.hdf5", "r") as f:
    print(list(f.keys()))  # 查看所有数据集
    data = f["/action/joint/position"][:]
    print(data.shape)
```

**Q: 控制频率如何调整？**

在 `DataCollectionManager` 初始化时：
```python
DataCollectionManager(
    frame_skip=20,     # 每 20 个仿真步执行一次控制
    time_step=0.001,   # 仿真步长 1ms
    # 控制频率 = 1/(20*0.001) = 50 Hz
)
```

**Q: 如何禁用某些执行器？**

```python
# 禁用位置控制器 (如果使用力矩控制)
manager.set_disable_actuator_group([position_actuator_group_id])
```

**Q: 采集时视频保存在哪？**

在数据单元目录下的 `video/` 子目录：
```
dataset/
  └── {uuid}/
      ├── record/
      │   └── data.hdf5
      └── video/
          └── {timestamp}.mp4
```

---

## 🔗 更多资源

- 详细开发文档: [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md)
- 框架架构说明: 查看主 README
- API 文档: 查看各模块的 docstring

祝您数据采集顺利！🎉

