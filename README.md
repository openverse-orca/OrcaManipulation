# OrcaManipulation（Suzhou_industrial_park 分支）

机器人遥操作与数据采集框架 —— 面向苏州工业园吧台抓杯子场景的演示数据采集与回放。

本分支在框架主干基础上，面向 **D12（青龙）人形机器人 + 吧台抓杯子场景** 的遥操作数据采集做了专项适配，并已完成对 **OrcaLab 26.9.1 新相机接口** 的兼容改造。

---

## ✨ 分支特性

- 🦾 **D12 青龙机器人全链路支持**：双臂 OSC 遥操作、双 2F85 夹爪、腰部关节控制与记录
- 🎮 **Pico VR 遥操作**：手柄双臂控制、任务状态切换（开始/结束采集）
- 🥤 **吧台抓杯子场景**：D12 双臂在吧台完成杯子抓取/放置，物体/灯光随机化可配置
- 🎬 **视频录制双版本兼容**：
  - OrcaLab ≥ 26.9.1：客户端 WebSocket 推流 + 按仿真步区间 remux 落盘，**每相机一个 mp4**
  - OrcaLab ≤ 26.7.3：走引擎侧 `BeginSaveMp4File` 录制
- 💾 **HDF5 数据落盘**：关节角、末端位姿、夹爪、腰部、时间戳与任务/场景元数据
- 🔁 **数据回放**：`data_collection_aug.py` 将已采数据回放到仿真场景，可用于质检与数据增强
- 🧪 **多传送带联调脚本**（附带工具）：`run_conveyor_multi_env.py` 支持多 env / 单 env 驱动多条传送带，适用于传送带类场景

---

## 🖥️ 环境要求

| 组件 | 要求 |
|------|------|
| OrcaLab | 26.9.1（推荐，当前分支已适配）或 26.7.3 |
| Python 环境 | conda 环境 `orcalab_lerobot`（26.9.1，内含 `orca-gym`、`av` 等）或 `orcalab`（26.7.3） |
| VR 设备 | Pico 手柄（可选，遥操作模式需要） |
| 资产包 | 须从**测试库（测试资产库）** 订阅以下两个资产包（详见下方说明） |

> ⚠️ **版本匹配很重要**：客户端 Python 环境里的 `orca-gym` 版本必须与正在运行的 OrcaLab 版本一致，否则 gRPC proto 不匹配会报错（详见 [常见问题](#-常见问题)）。

### 资产订阅（必做）

数采所用的机器人与场景资产需先在 OrcaLab 中从**测试库（测试资产库）**完成订阅，否则场景与机器人模型无法加载：

| 资产包 | 用途 |
|--------|------|
| **d12_wais** | D12（青龙）人形机器人模型（含双臂、双 2F85 夹爪、腰部关节） |
| **Suzhou_Industrial_Park** | 苏州工业园吧台场景（`sip_hotelreception`，含吧台、杯子等物体） |

操作步骤（OrcaLab 内）：

1. 打开 OrcaLab，进入资产库，切换到**测试库（测试资产库）**；
2. 搜索并订阅上述两个资产包；
3. 订阅完成后，即可在场景加载列表中找到 `sip_hotelreception` 场景与 D12 机器人资产。

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 📂 目录结构

```
OrcaManipulation/
├── src/
│   ├── dataCollectionManager/   # 核心调度器 DataCollectionManager（数采主循环、录制区间管理）
│   ├── devices/                 # 设备层
│   │   ├── abstract_device.py   #   PicoJoystickDevice 等
│   │   ├── data_device.py       #   DataDevice（回放已采数据）
│   │   └── Interpolator/        #   数据插值增强
│   ├── controllers/             # 控制器（OSC 臂控、夹爪、腰部、任务状态）
│   ├── scene/                   # SceneManager 场景管理（物体/灯光随机化、UI 消息）
│   ├── task/                    # 任务定义与成功判定（PickPlaceTask / EmptyTask）
│   ├── dataStorage/             # 数据存储（HDF5 + 视频，双版本录制接口）
│   │   └── openloong_data_storage.py  # D12/OpenLoong 数据存储实现
│   ├── envs/                    # gymnasium 环境封装
│   ├── conf/                    # 机器人配置
│   │   ├── d12_conf.py          #   D12 青龙配置（本分支主用）
│   │   ├── openloong_conf.py / g1_conf.py / g1_pick_osc_conf.py
│   └── examples/
│       └── dataCollection/
│           ├── data_collection_tele.py      # 遥操作数采入口（主脚本）
│           ├── data_collection_aug.py       # 数据回放/增强入口
│           ├── run_conveyor_multi_env.py    # 多传送带联调脚本（附带工具）
│           ├── conveyor_collect.yaml       # 场景数采配置（文件名沿用历史命名）
│           ├── example.yaml                # 通用场景配置模板
│           ├── dataset/                    # 采集数据落盘目录（仓库默认为空，运行时自动创建）
│           └── logs/                        # 运行日志
│       └── 超市场景青龙机器人数采案例/     # 超市场景参考（含 OrcaLab 数采示例文档）
├── QUICK_START.md               # 快速开始（自定义机器人接入五步法）
├── DEVELOPER_GUIDE.md           # 开发者指南（详细 API）
├── 数据采集重构指南.md           # 数采流程重构说明
└── README.md                    # 本文件
```

---

## 🚀 快速开始

### 1. 启动仿真端（加载 JSON 布局）

1. 打开 OrcaLab，完成[资产订阅](#资产订阅必做)；
2. 加载仓库自带的布局文件 [pick bottle.json](src/examples/dataCollection/pick%20bottle.json)（OrcaLab 菜单 → 打开/加载布局，选择该文件）。布局会自动还原：吧台场景 `sip_hotelreception` + 已摆好位的 `d12_waist_motor_usda` 机器人 + 视角，无需手动摆放机器人；
3. 进入运行时；
4. 确认 gRPC 地址为 `localhost:50051`（脚本默认值）。

> ⚠️ 请通过加载 JSON 布局的方式启动场景，而不是只手动加载 `sip_hotelreception` 场景——布局中包含机器人在吧台前的正确摆位，手动加载场景会导致场景里没有机器人（或机器人位置不对），脚本启动时会报 `KeyError: 'd12_waist_motor_usda_l_fingers_actuator'` 之类的执行器找不到错误。

### 2. 遥操作数采

```bash
cd src/examples/dataCollection
conda activate orcalab_lerobot   # OrcaLab 26.9.1；若用 26.7.3 则 activate orcalab
python data_collection_tele.py --agent_name d12
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--agent_name` | 机器人型号：`d12`（默认）、`openloong`、`g1`、`g1_pick_osc` |
| `--no-save-output` | 只跑仿真不落盘（不保存 hdf5 与视频） |

### 3. 采集流程（Pico 操作）

1. 脚本启动后自动进入 NOT_STARTED 状态；
2. 按下 **左手柄 Grip（L_GRIPBUTTON）**：NOT_STARTED → **RUNNING**，开始记录 hdf5 并录制视频；
3. 操作左右手柄完成抓放任务；
4. 再次按下 Grip：RUNNING → **END**，停止录制并落盘；
5. 若任务成功（或配置了 `always_save: true`）则保存，否则丢弃该条数据；
6. 场景自动重置，进入下一条采集。

| 手柄输入 | 功能 |
|----------|------|
| 右手柄位置/姿态 | 右臂末端 OSC 控制 |
| 左手柄位置/姿态 | 左臂末端 OSC 控制 |
| X / Y / 左扳机 | 左夹爪控制 |
| A / B / 右扳机 | 右夹爪控制 |
| 左手 Grip | 任务状态切换（开始/结束采集） |
| X+Y / A+B | 传送带停止/继续（仅传送带场景，需 `use_pico_buttons: true`） |

### 4. 数据回放

回放脚本读取已采 hdf5，在仿真中重演动作序列（场景/物体姿态由 `scene_info` 还原）：

```bash
cd src/examples/dataCollection
conda activate orcalab_lerobot
python data_collection_aug.py
```

- 回放数据目录：`dataset/d12/`（须与采集时场景一致）；
- 默认 `loop_playback=True`，顺序循环回放全部 episode，Ctrl+C 退出；
- 当前默认 `save_video = False`，即回放过程不重新录制。

---

## ⚙️ 场景配置（conveyor_collect.yaml）

数采脚本启动时读取该 yaml（文件名沿用历史命名，当前吧台抓杯子场景同样使用此文件）。当前场景相关要点：

```yaml
level_name: "conveyor_collect"   # 场景标识（当前代码未使用，实际场景在 OrcaLab 中手动加载）
type: "collect_only"             # collect_only 等模式使用 EmptyTask（不做成功判定）

collection:
  always_save: true              # 无论任务成功与否都保存本条数据
```

- `type: "collect_only"`：采集模式不做抓取成功判定，按手柄 Grip 控制起止即可；
- `collection.always_save`：true 时每条 episode 无论成功与否都落盘；
- 物体随机摆放、灯光随机等通过 `actor` / `light` 段配置（可参考 [example.yaml](src/examples/dataCollection/example.yaml)）；
- 吧台场景（`sip_hotelreception`）须通过加载 OrcaLab 布局文件 [pick bottle.json](src/examples/dataCollection/pick%20bottle.json) 启动（见[快速开始](#-快速开始)第 1 步），布局内含机器人在吧台前的摆位；

### 传送带配置（可选，当前吧台场景未使用）

文件中的 `conveyor` 段为传送带类场景保留的能力，当前吧台抓杯子场景不涉及传送带，无需修改：

```yaml
conveyor:
  enable: true
  board_joint_name: "Geom_Track_Geom_Track_Joint"  # 按场景实际 joint 名填写
  start_mode: "task_status"  # RUNNING 才动；auto=脚本启动即动
  speed: 0.05               # m/s
```

- 传送带板件按时间函数直线移动，物品被动摩擦跟随（不直接写物品 qpos/qvel）；
- 运行时可通过 `DataCollectionManager.set_conveyor_speed()` 调速，物理速度与动画速度自动同步。

---

## 🎬 视频录制说明（重要）

OrcaLab 26.9.1 起，引擎侧 MP4 录制 RPC（`BeginSaveMp4File`/`StopSaveMp4File`）已被移除，本分支在
[dataStorage/abstract_data_storage.py](src/dataStorage/abstract_data_storage.py) 中做了**双版本自动兼容**：

| OrcaLab 版本 | 录制方式 | 产物 |
|--------------|----------|------|
| ≥ 26.9.1 | `start_streaming` 开推流 + `save_streaming` 按仿真步区间 `[start, end]` 客户端 remux | `video/<相机名>.mp4`（每相机一个文件） |
| ≤ 26.7.3 | 引擎侧 `begin_save_video` / `stop_save_video` | 引擎写出的 mp4 |

版本探测通过 `hasattr(env, "save_streaming")` 完成，**同一份代码可直接在两个版本上运行**。

26.9.1 模式下的工作流程：

1. `DataCollectionManager.run()` 启动时调用 `prepare_video_streams()`：枚举场景相机（`get_camera_names`），逐台 `start_streaming` 开启推流与录制缓冲，并开启同步渲染；
2. 任务切到 RUNNING：记录区间起始仿真步，并渲染一帧 IDR 关键帧（保证截断后视频可播放）；
3. 仿真循环中每步 `render(simulate_index=...)` 做帧对齐；
4. 任务切到 END：对每台相机 `save_streaming` 落盘 `video/<相机名>.mp4`，阻塞等待写盘完成。

> 📌 若启动日志提示「场景中没有已注册的相机」，需要在 OrcaLab 场景中添加相机后才能录制。

---

## 📊 数据格式

> 仓库中的 `dataset/` 目录刻意保持为空：克隆后自行运行数采即可生成数据，目录由脚本自动创建。

每条 episode 一个 uuid 目录：

```
dataset/
└── d12/                                 # 按 agent_name 分目录
    └── {episode_uuid}/
        ├── record/
        │   └── proprio_stats.hdf5
        │       ├── /action/joint/position      # (N, 15) 关节角（双臂 7+7 + 腰 1）
        │       ├── /action/end/position         # (N, 2, 3) 双臂末端位置
        │       ├── /action/end/orientation      # (N, 2, 4) 双臂末端姿态（四元数）
        │       ├── /action/effector/motor       # (N, 2)   夹爪电机控制
        │       ├── /action/effector/position    # (N, 2)   夹爪位置
        │       ├── /action/waist/angle          # (N, 1)   腰部关节角
        │       ├── /time_step                   # (N,)     时间戳
        │       ├── /task_info                   # JSON 任务元数据
        │       └── /scene_info                  # JSON 场景元数据（物体位姿，用于回放还原）
        └── video/
            ├── camera_head.mp4                  # 头部相机（26.9.1：每相机一个文件）
            ├── camera_wrist_l.mp4               # 左腕相机
            └── camera_wrist_r.mp4               # 右腕相机
```

读取示例：

```python
import h5py
with h5py.File("dataset/d12/{uuid}/record/proprio_stats.hdf5", "r") as f:
    joint = f["/action/joint/position"][:]   # (N, 15)
```

---

## 🧪 多传送带联调（附带工具，传送带类场景适用）

```bash
# 单进程多 env（每条传送带一个 env）
python run_conveyor_multi_env.py --track-names "Geom_Track1,Geom_Track2,Geom_Track3"

# 规范命名自动生成：Geom_Track1_Joint, Geom_Track2_Joint, ...
python run_conveyor_multi_env.py --track-prefix Geom_Track --track-suffix _Joint --num-tracks 3

# 单 env 模式（推荐：一个 env 同时驱动多条传送带，只渲染/步进一套仿真）
python run_conveyor_multi_env.py --track-prefix Geom_Track --track-suffix _Joint --num-tracks 3 --single-env
```

---

## 🧩 核心组件与扩展

本分支沿用主干的分层架构：

```
DataCollectionManager (核心调度器)
    ├── Device Layer          # 输入设备
    │   ├── PicoJoystickDevice   (VR 遥操作)
    │   └── DataDevice           (数据回放)
    ├── Controller Layer      # 控制器
    │   ├── Arm OSC / Gripper 2F85 / Waist / TaskStatus
    ├── Scene Layer           # SceneManager（物体/灯光随机化、传送带同步）
    ├── Task Layer            # AbstractTask（任务目标/成功判定）
    └── Storage Layer         # AbstractDataStorage（HDF5 + 双版本视频录制）
```

接入自定义机器人/场景请参考：

- [QUICK_START.md](./QUICK_START.md) —— 五步接入法（存储器 → 控制器 → 场景 → 任务 → 脚本）
- [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) —— 详细 API 与最佳实践
- [数据采集重构指南.md](./数据采集重构指南.md) —— 数采流程设计说明
- 超市场景参考：[src/examples/超市场景青龙机器人数采案例](src/examples/超市场景青龙机器人数采案例)

---

## 🐛 常见问题

**Q: 录制视频时报 `grpc.aio._call.AioRpcError: StatusCode.UNIMPLEMENTED`？**

这是**客户端 orca-gym 版本与运行中的 OrcaLab 版本不匹配**导致：
- 26.7.3 的客户端对 26.9.1 的 OrcaLab 调用已删除的 `BeginSaveMp4File` RPC → UNIMPLEMENTED；
- 解决：用与 OrcaLab 版本一致的 conda 环境运行脚本（26.9.1 → `orcalab_lerobot`）。

**Q: OrcaLab 26.9.1 下不报错但也没有视频？**

旧接口在 26.9.1 客户端中已是 no-op。请确认使用的是本分支代码（含 `save_streaming` 适配），并确认场景中已添加相机。

**Q: 视频只有零点几秒？**

多为测试片段（RUNNING 后立即 END）。数采正式数据建议单条保持足够时长，并定期清理测试数据。

**Q: 回放时提示找不到 hdf5 / 场景对不上？**

- 回放数据目录须为 `dataset/d12/`（含 uuid 子目录），脚本与采集时的场景必须一致；
- 必须用与 OrcaLab 版本一致的 Python 环境运行回放脚本。

**Q: VR 手柄连接不上 / 控制器不响应？**

- 确认 Pico 设备与 `PicoJoystick` 服务正常；
- 查看日志：`src/examples/dataCollection/logs/data_collection.log`；
- 确认执行器未被禁用（d12 默认禁用位置控制组，走力矩控制）。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。

## 📄 许可证

见 [LICENSE](./LICENSE)。
