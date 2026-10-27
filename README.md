# G1 + DEX1-1 试管抓取遥操作数据采集

本分支 `g1_pick_tube` 提供 Unitree G1（29DOF）+ DEX1-1 夹爪在试管场景下的 VR 遥操作数据采集与回放方案，包含采集代码与场景布局文件。

## 一、目录结构

```
├── README.md                    # 本文档
├── requirements.txt             # Python 依赖
├── g1_pick_tube.json            # 场景布局文件（关卡 + actor 布局）
└── src/                         # 采集/回放代码与依赖
    ├── conf/g1_dex1_osc_conf.py # 机器人/夹爪参数
    ├── controllers/             # OSC 手臂、DEX1 阻抗夹爪
    ├── dataCollectionManager/   # 采集主循环
    ├── dataStorage/             # LeRobot v2.1 数据集写入
    ├── devices/                 # PICO / 数据回放设备
    ├── envs/dataCollection/     # 仿真环境封装
    ├── examples/dataCollection/
    │   ├── g1_dex1_collection_tele.py    # 遥操作采集
    │   └── g1_dex1_collection_replay.py  # 数据集回放
    ├── scene/                   # 场景与 actor 管理
    ├── sensor/                  # 相机推流
    └── task/                    # 空任务
```

### 依赖与环境

- Python 3.12，`orca-gym >= 26.9.1`（相机推流与录制器接口从 26.9.1 起提供），推荐直接使用 `orcalab_lerobot` 环境
- 安装依赖：`pip install -r requirements.txt`
- 视频编码使用 PyAV + `h264_nvenc`，需要 NVIDIA GPU；无 GPU 时无法产出 mp4，只能落盘 parquet

## 二、部署步骤

### 方式一：OrcaLab 资产订阅 + 布局文件（推荐）

1. 打开 OrcaLab，进入「资产中心」。
2. 搜索并订阅以下 **两个资产**：
   - **试管操作**（`testtube_operation`）—— 场景包
   - **宇树夹试管(改摄像头)**（`g1_dex_tube_66666666`）—— 机器人
3. 订阅完成后，在 OrcaStudio 中加载仓库根目录的布局文件 `g1_pick_tube.json`，它会还原：
   - 关卡：`testtube_operation`
   - Actors：`Group`、`g1_pick_usda`
   - 相机机位与视角
   - **三路相机的录制开关**（`Enable` / `ColorCamera` / `IsRecording`）
4. 确认场景层级中出现 `g1_pick_usda` 机器人。
5. 运行遥操作采集代码（见第三节）。

> 布局文件已包含关卡引用与 actor 布局，加载后无需手动摆放机器人。
>
> **必须加载布局文件**：相机录制开关只保存在布局文件里，场景 prefab 中没有。不加载布局 → 相机不推流 → 只产出 HDF5、没有 mp4。

## 三、数采流程

### 1. 启动

1. 启动 OrcaStudio，打开 `Levels/shihua_xiugai_change_tube` 场景，确认机器人已加载。
2. 连接 PICO 手柄，确认手柄已被识别。
3. 使用 `orcalab_lerobot` 环境（orca-gym 26.9.1，支持相机推流录制），在仓库根目录执行：

```bash
# 1. 进入采集脚本目录（在仓库根目录下执行）
cd src/examples/dataCollection

# 2. 采集 10 条数据
#    --num_episodes 10 表示采集 10 集（10 条），采满自动退出
#    每集：按一次左 Grip 开始录制，再按一次左 Grip 结束并落盘
~/miniconda3/envs/orcalab_lerobot/bin/python -u g1_dex1_collection_tele.py \
    --orcagym_addr localhost:50051 \
    --agent_name g1_pick_usda \
    --pos_scale 0.8 \
    --num_episodes 10 \
    --lerobot_out lerobot_datasets/g1_dex1
```

数据保存在 `src/examples/dataCollection/lerobot_datasets/g1_dex1`。

### 2. 采集

1. 脚本启动后自动连接场景、加载机器人并启动三路相机推流（约数秒，日志出现相机就绪后再操作）。
2. 按一下**左 Grip** 开始本集录制。
3. 用手柄控制右臂与右夹爪完成任务（左臂/左手已禁用，锁在安全停靠姿态）。
4. 再按一下**左 Grip** 结束本集，数据自动落盘，场景重置，等待下一集。

### 3. 退出

`Ctrl+C` 结束脚本，退出时会关闭相机推流并保存日志。

### 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--agent_name` | `g1_dex1` | 必须与场景中机器人对象前缀一致（本场景为 `g1_pick_usda`） |
| `--num_episodes` | `0`（不限） | 采集条数，`10` 即采 10 集后自动退出 |
| `--resume` | - | 追加到已有数据集，不清空 |
| `--pos_scale` | `1.0` | 手柄位移→末端位移比例，精细操作建议 0.3~0.8 |
| `--osc_kp` | `300` | 任务空间刚度 |
| `--pos_deadzone` | `0.003` | 静止死区，手抖可调大 |
| `--ema_alpha` | `0.35` | 输入低通系数 |
| `--cameras` | `head,wrist_r` | 录制相机 |
| `--no_cameras` | - | 只存 11 维 state/action，不写图像 |
| `--lerobot_out` | `lerobot_datasets/g1_dex1` | LeRobot 数据集目录 |
| `--fps` | `20` | 写入数据集的帧率 |
| `--frame_skip` | `5` | 每个控制周期的物理步数 |
| `--time_step` | `0.001` | 物理步长 |

### 按键映射

| 操作 | 说明 |
|---|---|
| 左 Grip | 按一下开始采集，再按一下结束 |
| 右扳机 | 右夹爪开度（按下闭合） |
| A | 右夹爪全开 |
| B | 右夹爪闭合下限抬升（保持夹紧） |
| 右手柄位姿 | 控制右臂 |

> 左臂与左手在采集、回放中均不接控制器，锁在安全停靠姿态（肩 roll ≈ 7.3°）。
> 左手柄位姿、左扳机与 X / Y 键不再生效，左手始终保持在停靠开度。

## 四、数据存储结构

数据集为 LeRobot v2.1 格式，保存在 `--lerobot_out` 指定的目录（默认脚本同级的 `lerobot_datasets/g1_dex1`）：

```
src/examples/dataCollection/lerobot_datasets/g1_dex1/
├── meta/info.json                  # 特征、帧率、总集数
├── data/chunk-000/
│   └── episode_000000.parquet      # observation.state / action（11 维）与时间戳
└── videos/chunk-000/
    ├── observation.images.cam_head/episode_000000.mp4
    └── observation.images.cam_wrist_r/episode_000000.mp4
```

state/action 为 11 维：`r_pos(3), r_quat_xyzw(4), l_grip_inner_norm, l_grip_outer_norm, r_grip_inner_norm, r_grip_outer_norm`。左臂在采集中锁在停靠姿态，不写入数据。

- 每完成一集自动写一个 `episode_XXXXXX.parquet` 与对应视频
- `--lerobot_out` 可区分任务或批次，如 `--lerobot_out lerobot_datasets/tube_v1`
- 日志写入 `src/examples/dataCollection/logs/g1_dex1_collection.log`
- `dataset/`、`lerobot_datasets/` 与 `logs/` 下的实际数据已被 `.gitignore` 忽略，不会误提交

### 只有 HDF5、没有 mp4 时排查

视频走 OrcaStudio 相机推流，任一环节不满足都会**静默降级为只存 HDF5**（日志里出现 `[相机] 无可用相机, 本集仅保存 HDF5`）。按顺序检查：

| # | 检查项 | 说明 |
|---|---|---|
| 1 | 已加载 `g1_pick_tube.json` 布局 | **最常见原因**。相机录制开关（`Enable`/`ColorCamera`/`IsRecording`）只在布局文件里，场景 prefab 中没有；不加载布局则相机处于未录制状态 |
| 2 | Python 环境 | 必须是 `orcalab_lerobot`（orca-gym **26.9.1**）。`orcalab` 环境的 26.7.3 没有 `start_streaming`/`get_recorder_manager`，日志会报 `当前 orca-gym 不含推流接口` |
| 3 | 相机名匹配 | 后端注册名需含 `head_cam`/`camera_left`/`camera_right`（允许 uuid 后缀）。日志会打印实际注册名，可据此核对 |
| 4 | 机器人资产带相机 | 需订阅带摄像头的「宇树夹试管(改摄像头)」，不含摄像头的版本场景里没有相机实体 |
| 5 | 未关闭录制 | 启动命令不要带 `--no_cameras` |

启动后正常应看到三行 `[相机] ✓ head(...) 推流就绪`，约 1~2 秒/路。若某路超时 30 秒，只有该路缺视频，其余正常。

## 五、数据回放

`g1_dex1_collection_replay.py` 读取采集生成的 LeRobot 数据集，在仿真中 1:1 重放右臂末端轨迹与夹爪开合，用于质检与失败片段复查。

```bash
cd src/examples/dataCollection

# 回放数据集全部集（按 episode 序号，播完退出）
~/miniconda3/envs/orcalab_lerobot/bin/python g1_dex1_collection_replay.py \
    --agent_name g1_pick_usda

# 只回放指定集：序号（从 0 开始）或 parquet 文件名前缀
~/miniconda3/envs/orcalab_lerobot/bin/python g1_dex1_collection_replay.py \
    --agent_name g1_pick_usda --episode 0

# 指定数据集目录 / 循环回放
~/miniconda3/envs/orcalab_lerobot/bin/python g1_dex1_collection_replay.py \
    --lerobot_dataset lerobot_datasets/g1_dex1 --loop
```

回放说明：

- 读取每集 parquet 的 `observation.state`（11 维），右臂按 B 系位置/四元数走 OSC 控制
- 相邻数据帧之间插值出 `steps_per_frame` 个控制周期（位置线性、姿态 slerp），每个控制周期都下发新目标，严格按原速 1:1 回放，避免逐帧保持的阶梯式抖动
- 夹爪不再直接用记录开度作为目标：记录值是被试管挡住后的实际位置，直接回放会导致接触后夹持力为 0。改为按开度还原“张开/闭合”意图（迟滞判定），闭合时命令全闭以复现采集时的夹持力（约 8–22N）
- 左臂/左手不参与回放（数据不含左臂），由 `pin_all_joints` 锁在停靠姿态，且不再挂左手控制器，避免电机与钉住约束互扯
- 按数据集 `fps` 与控制周期（`time_step × frame_skip`）自动换算每帧的插值周期数
- LeRobot 数据集不含物体/场景状态与 `initial_joint_qpos`，每集使用场景初始状态
- `--episode` 支持序号或文件名前缀；不给则按 episode 顺序播放全部
- 日志写入 `src/examples/dataCollection/logs/g1_dex1_replay.log`

## 六、说明

- 采集程序会在每集开始时重新激活基座与腰腿关节的钉住约束，避免场景重载后机器人倒下。
- 左臂锁死在停靠姿态 `[0, 0.127, 0, 1.5708, 0, 0, 0]`（肩 roll ≈ 7.3°，参考 Binjiang_Competition），每个物理步写回 qpos/qvel 并清零电机控制，采集与回放一致。
- 夹爪为单电机 + 齿轮齿条耦合，采用官方位置阻抗控制（`tau = kp*(q_d - q) - kd*dq`），kp=1500、kd=15、最大夹持力 120N。
- 夹爪行程 `q_open=-0.020`、`q_close=0.0245`（官方 range 上限）。
