# OrcaManipulation

机器人遥操作与数据采集框架 - 快速构建高质量的机器人演示数据集

---

## ✨ 特性

- 🎮 **双模式支持**: TELECONTROL (VR遥控) + AUGMENTATION (数据增强)
- 🎯 **模块化设计**: 清晰的分层架构，易于扩展和定制
- 🔧 **开箱即用**: 提供常用控制器（OSC臂控制、夹爪控制）
- 🎲 **场景随机化**: 支持物体位姿、光照的随机化配置
- 💾 **高效存储**: HDF5 格式，支持压缩和元数据管理
- 🎬 **视频录制**: 自动保存任务执行视频
- 📈 **数据增强**: 内置插值器，扩充数据集规模

---

## 🏗️ 架构概览

```
DataCollectionManager (核心调度器)
    ├── Device Layer          # 输入设备
    │   ├── PicoJoystickDevice   (VR手柄)
    │   └── DataDevice           (数据回放)
    ├── Controller Layer      # 控制器
    │   ├── ControllerArm        (臂控制)
    │   ├── Controller2F85       (夹爪控制)
    │   └── TaskStatusController (任务状态)
    ├── Scene Layer          # 场景管理
    │   └── SceneManager         (物体/光照随机化)
    ├── Task Layer           # 任务定义
    │   └── AbstractTask         (任务目标/成功判定)
    └── Storage Layer        # 数据存储
        └── AbstractDataStorage  (数据采集/HDF5保存)
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据采集相关脚本（同一目录）

在运行下列脚本前，请确保 **OrcaGym / 仿真服务** 已启动，且地址与代码中一致（默认 **`localhost:50051`**）。建议在示例目录下执行：

```bash
cd src/examples/dataCollection
```

三个脚本共用同一套必填参数（`task_config` 为相对于本目录的 YAML，例如 `example.yaml`、`warehouse.yaml`）：

| 参数 | 说明 |
|------|------|
| `--level` | 场景 / 关卡名，用于区分数据子目录（与 `dataset`、`aug_dataset` 下的文件夹名一致） |
| `--agent_name` | `openloong` 或 `tiangong2` |
| `--task_config` | 场景与随机化等配置（YAML 文件名） |

#### 遥控采集：`data_collection_tele.py`

VR 手柄（Pico）遥操作，**OSC 双臂** + 夹爪 + 任务状态；模式为 **TELECONTROL**。数据写入：

`dataset/<agent_name>/<level>/<episode_uuid>/record/proprio_stats.hdf5`（及 `video/` 等）。

```bash
python data_collection_tele.py --level tele --agent_name openloong --task_config example.yaml
```

#### 相机监控端口配置

在遥操作采集过程中，可以通过 `add_monitor_port()` 方法注册相机监控端口，用于实时显示相机画面：

```python
# 在 data_collection_tele.py 中添加
data_collection_manager.add_monitor_port(7080)  # 左手臂彩色相机
data_collection_manager.add_monitor_port(7081)  # 左手臂深度相机
data_collection_manager.add_monitor_port(7090)  # 右手臂彩色相机
data_collection_manager.add_monitor_port(7091)  # 右手臂深度相机
```

| 端口 | 相机类型 | 说明 |
|------|---------|------|
| 7080 | colorport | 左手臂彩色图像 |
| 7081 | depthport | 左手臂深度图像 |
| 7090 | colorport | 右手臂彩色图像 |
| 7091 | depthport | 右手臂深度图像 |

> **注意**：需要确保 OrcaLab 中已正确配置对应的相机端口映射，否则可能出现花屏或无法显示的情况。

#### 数据增广：`data_collection_aug.py`

从 **`dataset/<agent_name>/<level>/`** 下各回合目录读取 HDF5，经 **`OpenLoongInterpolator`（插值 + 噪声）** 后，在仿真中用 **IK 数据控制器** 跟踪轨迹并重采集，结果写入 **`aug_dataset/<agent_name>/<level>/`**（结构与原数据集类似）。用于在固定演示基础上扩充轨迹分布。

```bash
python data_collection_aug.py --level tele --agent_name openloong --task_config example.yaml
```

#### 回放验证：`data_collection_replay.py`

仅回放、**不写入新的 HDF5**（`data_storage=None`），用于检查原始数据或增广后数据。通过 **`--data_root`** 选择读 `dataset` 还是 `aug_dataset`；通过 **`--replay_mode`** 选择控制方式：`osc`（默认）、`ik`、`position`。

```bash
# 回放遥操作原始数据
python data_collection_replay.py --level tele --agent_name openloong --task_config example.yaml --data_root dataset --replay_mode osc

# 回放增广后的数据（默认 data_root 即为 aug_dataset，可省略）
python data_collection_replay.py --level tele --agent_name openloong --task_config example.yaml --replay_mode ik
```

**推荐顺序**：先 **`data_collection_tele`** 积累 `dataset` → 再 **`data_collection_aug`** 生成 `aug_dataset` → 需要时用 **`data_collection_replay`** 抽查两种来源的轨迹。

### 3. 查看数据

单条回合路径示例（将 `openloong`、`tele`、`<uuid>` 换成你的 `agent_name`、`level` 与目录名）：

```python
import h5py
path = "dataset/openloong/tele/<uuid>/record/proprio_stats.hdf5"
with h5py.File(path, "r") as f:
    print(list(f.keys()))  # 查看 HDF5 顶层组 / 数据集
```

---

## 📚 文档

- **[快速开始指南](./QUICK_START.md)** - 5步搭建数据采集系统
- **[开发者指南](./DEVELOPER_GUIDE.md)** - 详细的API文档和最佳实践

---

## 🎯 使用流程

### 1) 遥控采集（`data_collection_tele.py`）— TELECONTROL

```
VR手柄 → PicoJoystickDevice → OSC臂控 + 夹爪 → Robot → OpenLoong/Tiangong DataStorage
                                                              ↓
                                                    dataset/<agent>/<level>/<uuid>/
```

1. 启动 OrcaGym 与仿真，执行 `data_collection_tele.py` 并传入 `--level`、`--agent_name`、`--task_config`
2. 戴上 VR，按场景 UI 提示操作
3. 左手柄握持键：**开始 / 结束** 一条轨迹记录（与 `TaskStatusController` 行为一致）
4. 任务成功则本回合写入 `dataset/...`；失败则丢弃

### 2) 数据增广（`data_collection_aug.py`）— AUGMENTATION

```
dataset/.../uuid/record/proprio_stats.hdf5
    → DataDevice + OpenLoongInterpolator
    → IK 数据臂控 + 夹爪数据控 → 仿真重放并再采集 → aug_dataset/<agent>/<level>/...
```

1. 保证 `dataset/<agent_name>/<level>/` 下已有至少一个带 `record/proprio_stats.hdf5` 的回合目录
2. 使用与采集时一致的 `--level`、`--agent_name`、`--task_config`（场景需能对上元数据中的物体布局）
3. 运行 `data_collection_aug.py`；脚本内插值器与 IK 控制为固定实现，如需 OSC/position 增广需在代码层替换控制器（参见 `DEVELOPER_GUIDE.md`）

### 3) 回放检查（`data_collection_replay.py`）

```
aug_dataset 或 dataset → DataDevice（无插值器）→ osc / ik / position 数据臂控 → 仅可视化验证
```

1. 使用 `--data_root dataset` 检查原始遥操作数据，或 `--data_root aug_dataset`（默认）检查增广结果
2. `--replay_mode` 与增广脚本中的 IK 无绑定关系，仅影响「回放时」用哪种控制律跟踪 HDF5

---

## 🧩 核心组件

### 必须实现的组件

| 组件 | 说明 | 示例 |
|------|------|------|
| **DataStorage** | 定义采集什么数据、如何保存 | [openloong_data_storage.py](src/dataStorage/openloong_data_storage.py) |
| **Controllers** | 配置机器人控制器 | [controllers.py](src/controllers/controllers.py) |
| **Scene Config** | 场景和随机化配置 | [example.yaml](src/examples/dataCollection/example.yaml) |
| **Task** | 任务定义和成功判定 | [pick_place_task.py](src/task/pick_place_task.py) |

### 可选实现的组件

| 组件 | 说明 | 使用场景 |
|------|------|---------|
| **Interpolator** | 数据插值增强 | 数据增强模式 |
| **Custom Controller** | 自定义控制逻辑 | 特殊控制需求 |
| **Custom Device** | 自定义输入设备 | 非VR设备 |

---

## 📂 项目结构

```
OrcaManipulation/
├── src/
│   ├── dataCollectionManager/   # 核心调度器
│   ├── devices/                 # 设备层
│   │   └── Interpolator/        # 插值器
│   ├── controllers/             # 控制器
│   ├── scene/                   # 场景管理
│   ├── task/                    # 任务定义
│   ├── dataStorage/             # 数据存储
│   ├── envs/                    # 环境定义
│   ├── conf/                    # 机器人配置
│   └── examples/                # 示例代码
│       └── dataCollection/
│           ├── data_collection_tele.py    # VR 遥操作采集（→ dataset）
│           ├── data_collection_aug.py     # 数据增广（dataset → aug_dataset）
│           ├── data_collection_replay.py  # 轨迹回放验证（可选 data_root / replay_mode）
│           ├── example.yaml               # 场景配置示例
│           └── warehouse.yaml             # 另一场景配置示例
├── QUICK_START.md               # 快速开始
├── DEVELOPER_GUIDE.md           # 开发者指南
└── README.md                    # 本文件
```

---

## 🎮 VR 手柄操作

| 按键 | 功能 |
|------|------|
| 右手柄位置/姿态 | 控制机械臂末端位置和姿态 |
| 左手柄位置/姿态 | 控制另一侧机械臂 (双臂) |
| A / B 按钮 | 夹爪开合 |
| X / Y 按钮 | 夹爪开合 (另一侧) |
| 扳机 | 夹爪闭合程度 |
| 左手柄握持按钮 | 开始/结束任务记录 |

---

## 📊 数据格式

### HDF5 结构

遥操作与增广脚本约定的根目录为 **`dataset/<agent_name>/<level>/`** 与 **`aug_dataset/<agent_name>/<level>/`**，其下每个 **`{episode_uuid}`** 为一回合：

```
dataset/   （或 aug_dataset/）
  └── <agent_name>/
      └── <level>/
          └── {episode_uuid}/
              ├── record/
              │   └── proprio_stats.hdf5
              │       ├── /action/joint/position       # (N, D) 关节位置
              │       ├── /action/end/position         # 末端位置（维度随双臂 flatten 等可能变化）
              │       ├── /action/end/orientation      # 末端姿态（四元数）
              │       ├── /action/effector/motor       # 夹爪等
              │       ├── time_step                    # (N,) 时间戳
              │       ├── task_info                    # 任务元数据
              │       └── scene_info                   # 场景元数据
              └── video/
                  └── ...
```

---

## 🔧 配置说明

### 机器人配置

```python
# conf/robot_conf.py
arm_config = {
    "joint_names": [...],           # 关节名称列表
    "neutral_joint_values": [...],  # 初始关节角度
    "motors_names": [...],          # 电机名称列表
    "motors_init_ctrl": [...],      # 电机初始控制值
    "motors_ranges": [...],         # 电机控制范围
    "ee_site_name": "ee_site"       # 末端执行器site名称
}
```

### 场景配置

```yaml
# scene_config.yaml
actor:
  names: [...]              # 物体名称
  spawnable: [...]          # 物体资产路径
  joints_dof: [6, 6, ...]   # 自由度 (1/3/6)
  random:
    qpos: true              # 启用位置随机化
    nums: [1, 3]            # 随机生成数量范围
    six_dof:
      center: [x, y, z]     # 中心点
      bound_position: [...]  # 位置随机范围
      bound_rotation: [...]  # 旋转随机范围
```

---

## 💡 最佳实践

### 数据采集建议

1. **采集前准备**
   - 确保场景光照充足
   - 检查 VR 手柄电量
   - 测试控制器映射是否正确

2. **采集过程**
   - 保持动作流畅自然
   - 避免突然加速或减速
   - 确保任务成功后再结束记录

3. **质量控制**
   - 定期检查采集的数据
   - 及时删除失败或低质量的数据
   - 保持数据集的多样性

### 数据增强建议

1. **插值策略**
   - 关节/位置: 线性或三次样条插值
   - 四元数: 使用 SLERP 球面插值
   - 离散值: 最近邻或取整

2. **噪声添加**
   - 适量噪声 (0.01-0.05) 提升鲁棒性
   - 避免过大噪声破坏任务可行性

3. **数据检查**
   - 插值后检查数据长度一致性
   - 可视化轨迹判断合理性

---

## 🐛 故障排查

### 常见问题

**Q: VR 手柄连接不上？**
- 检查 VR 设备是否正常启动
- 确认 PicoJoystick 服务是否运行

**Q: 控制器不响应？**
- 检查控制器绑定是否正确
- 查看日志输出是否有错误信息
- 确认执行器没有被禁用

**Q: 数据保存失败？**
- 检查磁盘空间是否充足
- 确认 HDF5 路径有写入权限
- 查看是否有异常日志

**Q: 任务一直判定失败？**
- 检查 `is_success()` 逻辑是否正确
- 降低成功判定阈值进行测试
- 可视化目标区域和物体位置

**Q: `data_collection_replay.py` 报 `aug_dataset/...` 目录不存在？**
- 若回放的是遥操作原始数据，请加上 **`--data_root dataset`**，并保证 `dataset/<agent>/<level>/` 下已有回合子目录。
- 若回放增广数据，请先运行 **`data_collection_aug.py`** 生成 `aug_dataset/...`。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

[添加您的许可证信息]

---

## 📧 联系方式

如有问题或建议，请联系: [添加您的联系方式]

---

## 🙏 致谢

基于 [OrcaGym](https://github.com/...) 构建

---

**Happy Data Collecting! 🎉**
