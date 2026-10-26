# G1 + DEX1-1 试管抓取遥操作数据采集

本分支 `g1_pick_tube` 提供 Unitree G1（29DOF）+ DEX1-1 夹爪在试管场景下的 VR 遥操作数据采集方案，包含机器人模型、OrcaStudio 场景资产与采集代码。

## 一、目录结构

```
├── models/                      # 机器人模型（MuJoCo）
│   ├── g1_pick.xml              # G1 + 双臂 DEX1-1 夹爪
│   └── assets/                  # 网格资源（含 assets/dex1/）
├── orcastudio_scene/            # OrcaStudio 场景资产
│   ├── Assets/                  # TestTube_Operation / Importer / Prefabs 等
│   ├── Levels/
│   │   └── shihua_xiugai_change_tube/   # 场景 Level
│   ├── ReflectionProbes/
│   └── DiffuseProbeGrids/
└── src/                         # 采集代码
    ├── conf/g1_dex1_osc_conf.py
    ├── controllers/controller_dex1_force.py
    ├── dataStorage/g1_dex1_data_storage.py
    └── examples/dataCollection/g1_dex1_collection_tele.py
```

## 二、部署步骤

### 1. 导入场景到 OrcaStudio

把 `orcastudio_scene/` 目录下的 **所有内容** 直接复制粘贴到 OrcaStudio 项目根目录，与已有的 `Assets/`、`Levels/` 等目录合并：

```bash
cp -r orcastudio_scene/Assets orcastudio_scene/Levels \
      orcastudio_scene/ReflectionProbes orcastudio_scene/DiffuseProbeGrids \
      <OrcaStudio项目根目录>/
```

复制完成后在 OrcaStudio 中打开 `Levels/shihua_xiugai_change_tube` 场景。

### 2. 导入机器人模型

把 `models/` 整个目录放到 OrcaStudio 的 MuJoCo 模型目录下。注意 `g1_pick.xml` 必须与其 `assets/` 保持同级（XML 内 `meshdir="assets"`），否则网格加载失败。

### 3. 场景检查

打开场景后确认：

- 机器人对象名带 `g1_pick_usda` 前缀（与场景中 `Assets/Prefabs/g1_pick_usda.prefab` 对应）
- 模型含 4 个 DEX1 关节（`left/right_dex1_joint1`、`joint2`）、2 个电机（`left/right_dex1_motor`）
- 相机 `head_cam`、`camera_left`、`camera_right` 正常注册

## 三、数采流程

### 1. 启动

1. 启动 OrcaStudio，打开 `Levels/shihua_xiugai_change_tube` 场景，确认机器人已加载。
2. 连接 PICO 手柄，确认手柄已被识别。
3. 使用 `orcalab_lerobot` 环境（orca-gym 26.9.1，支持相机推流录制）启动采集脚本：

```bash
cd src/examples/dataCollection
~/miniconda3/envs/orcalab_lerobot/bin/python g1_dex1_collection_tele.py \
    --agent_name g1_pick_usda --pos_scale 0.8
```

### 2. 采集

1. 脚本启动后自动连接场景、加载机器人并启动三路相机推流（约数秒，日志出现相机就绪后再操作）。
2. 按一下**左 Grip** 开始本集录制。
3. 用手柄控制双臂与夹爪完成任务。
4. 再按一下**左 Grip** 结束本集，数据自动落盘，场景重置，等待下一集。

### 3. 退出

`Ctrl+C` 结束脚本，退出时会关闭相机推流并保存日志。

### 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--agent_name` | `g1_dex1` | 必须与场景中机器人对象前缀一致（本场景为 `g1_pick_usda`） |
| `--pos_scale` | `1.0` | 手柄位移→末端位移比例，精细操作建议 0.3~0.8 |
| `--osc_kp` | `300` | 任务空间刚度 |
| `--pos_deadzone` | `0.003` | 静止死区，手抖可调大 |
| `--ema_alpha` | `0.35` | 输入低通系数 |
| `--cameras` | `head,wrist_l,wrist_r` | 录制相机 |
| `--no_cameras` | - | 只存 HDF5，不录视频 |
| `--level` | `default` | 数据集子目录 |
| `--frame_skip` | `5` | 每个控制周期的物理步数 |
| `--time_step` | `0.001` | 物理步长 |

### 按键映射

| 操作 | 说明 |
|---|---|
| 左 Grip | 按一下开始采集，再按一下结束 |
| 左扳机 | 左夹爪开度（按下闭合） |
| 右扳机 | 右夹爪开度（按下闭合） |
| X / A | 左 / 右夹爪全开 |
| Y / B | 左 / 右夹爪中位 |
| 手柄位姿 | 控制左右手臂 |

## 四、数据存储结构

数据保存在脚本同级的 `dataset` 目录（仓库已保留该空目录）：

```
src/examples/dataCollection/dataset/
└── <agent_name>/
    └── <level>/
        └── <uuid>/
            ├── record/proprio_stats.hdf5   # 关节位置、速度、动作等状态数据
            └── video/
                ├── head.mp4
                ├── wrist_l.mp4
                └── wrist_r.mp4
```

以默认参数为例：`dataset/g1_pick_usda/default/<uuid>/`

- 每集自动生成一个 `<uuid>` 目录
- `--level` 用于区分任务或批次，如 `--level tube_v1`
- 日志写入 `src/examples/dataCollection/logs/g1_dex1_collection.log`
- `dataset/` 与 `logs/` 下的实际数据已被 `.gitignore` 忽略，不会误提交，仓库仅保留 `.gitkeep` 占位目录

## 五、说明

- 采集程序会在每集开始时重新激活基座与腰腿关节的钉住约束，避免场景重载后机器人倒下。
- 夹爪为单电机 + 齿轮齿条耦合，采用官方位置阻抗控制（`tau = kp*(q_d - q) - kd*dq`），kp=1500、kd=15、最大夹持力 120N。
- 夹爪行程 `q_open=-0.020`、`q_close=0.0245`（官方 range 上限）。
