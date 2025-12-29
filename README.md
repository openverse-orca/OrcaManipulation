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

### 2. 运行示例

#### 遥控采集模式

```bash
cd src/examples/dataCollection
python data_collection_tele.py
```

使用 VR 手柄遥控机器人，采集真实演示数据。

#### 数据增强模式

```bash
python data_collection_aug.py
```

读取已有数据并回放，应用插值和噪声生成增强数据。

### 3. 查看数据

```python
import h5py
with h5py.File("dataset/{uuid}/record/data.hdf5", "r") as f:
    print(f.keys())  # 查看所有数据集
```

---

## 📚 文档

- **[快速开始指南](./QUICK_START.md)** - 5步搭建数据采集系统
- **[开发者指南](./DEVELOPER_GUIDE.md)** - 详细的API文档和最佳实践

---

## 🎯 使用流程

### Mode 1: TELECONTROL - 遥控采集

```
VR手柄 → PicoJoystickDevice → Controllers → Robot → DataStorage
                                                        ↓
                                                   HDF5 + Video
```

1. 戴上 VR 头显，启动数据采集脚本
2. 操作 VR 手柄控制机器人
3. 按下左手柄握持按钮开始记录
4. 完成任务后再次按下结束记录
5. 任务成功则数据自动保存，失败则丢弃

### Mode 2: AUGMENTATION - 数据增强

```
HDF5 → DataDevice → Interpolator → Controllers → Robot → DataStorage
         ↓                                                    ↓
    Task Info                                          Enhanced HDF5
```

1. 指定已有数据集路径
2. 配置插值器（可选）
3. 运行数据增强脚本
4. 框架自动读取数据、插值、回放、保存

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
│           ├── data_collection_tele.py   # 遥控采集示例
│           ├── data_collection_aug.py    # 数据增强示例
│           └── example.yaml              # 场景配置示例
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

```
dataset/
  └── {episode_uuid}/
      ├── record/
      │   └── proprio_stats.hdf5
      │       ├── /action/joint/position       # (N, D) 关节位置
      │       ├── /action/end/position         # (N, 3) 末端位置
      │       ├── /action/end/orientation      # (N, 4) 末端姿态(四元数)
      │       ├── /action/effector/motor       # (N, M) 夹爪控制
      │       ├── time_step                    # (N,) 时间戳
      │       ├── task_info                    # JSON 任务元数据
      │       └── scene_info                   # JSON 场景元数据
      └── video/
          └── {timestamp}.mp4
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
