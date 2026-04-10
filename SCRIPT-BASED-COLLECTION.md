# 基于脚本的数据采集（`data_collection_scripted.py`）

本文说明如何在 **OrcaLab 仿真已启动** 的前提下，使用 `src/examples/dataCollection/data_collection_scripted.py` 通过 **预定义末端轨迹（基座系 B）+ 夹爪指令** 驱动 OpenLoong 双臂（OSC），完成与 `data_collection_tele` 相同 gRPC 地址（默认 `localhost:50051`）下的脚本运行。无需 Pico、不依赖 HDF5 回放作为输入。

---

## 1. 前置条件

- 已启动 **OrcaLab** 仿真，OrcaGym 可连。
- 机器人侧使用 **`conf/openloong_conf.py`** 中的 OpenLoong 配置（双臂 OSC + 2F85 夹爪）。
- 在 **`src/examples/dataCollection`** 目录下执行 Python（见下文命令），以便 `--task_config`、`--pose_file` 使用相对路径时能正确找到与脚本同目录的 YAML。

---

## 2. 推荐执行命令

在仓库内进入示例目录后执行：

```bash
cd src/examples/dataCollection
python data_collection_scripted.py --level shop_scaning --task_config scripted-example.yaml --pose_file pose.yaml
```

需要把本回合录成 **HDF5**（与 `data_collection_tele` 相同目录与文件布局，供 `data_collection_aug.py` 回放）时，增加 **`--record_hdf5`**：

```bash
python data_collection_scripted.py --level shop_scaning --task_config scripted-example.yaml --pose_file pose.yaml --record_hdf5
```

数据写入：`src/examples/dataCollection/dataset/openloong/<level>/<uuid>/record/proprio_stats.hdf5`。  
使用 **`data_collection_aug.py`** 时：`--agent_name openloong`，**`--level` 与本脚本一致**，且 `DataDevice` 读取的 `dataset/...` 路径与 tele 相同（`examples/dataCollection/dataset/openloong/<level>`），即可加载上述单元目录中的 HDF5 进行回放。

说明：

| 参数 | 含义 |
|------|------|
| `--level` | 场景/数据集逻辑名，用于日志与内部路径占位（如 `dataset/openloong/<level>/`）；与 `scripted-example.yaml` 内的 `level_name` 可不同，按你实际命名使用。 |
| `--task_config` | **场景与物体生成** 的 YAML，与遥操作 `data_collection_tele` 所用格式一致（如 `scripted-example.yaml`）。路径相对于 **`data_collection_scripted.py` 所在目录**。 |
| `--pose_file` | **末端轨迹与夹爪** 的 JSON/YAML（如 `pose.yaml`）。路径相对于脚本目录。 |
| `--record_hdf5` | 本回合成功结束时保存 HDF5；失败则丢弃缓冲。默认不录视频（与仅落 HDF5 供 aug 回放一致）。不加则不落盘 HDF5。 |

---

## 3. 脚本在做什么（概念）

1. 加载场景配置 → `SceneManager` 与仿真对齐 → `env.reset()`、`update_scene()`。
2. 若带 **`--dump_pose`**：打印基座系下左/右臂末端及名称含 `bottle`、`basket` 的 body 位置后退出，便于填写 `r_target_b` / `l_target_b`。
3. 否则根据 `pose_file` 生成整段离散轨迹（位置线性插值、姿态球面插值），由 **`ScriptedTrajectoryDevice`** 逐步写入 OSC 与夹爪控制器，执行 **`run_episode()`** 一整回合。

位姿均在 **机器人基座 body**（配置中 `base_link`，仿真中常解析为 `openloong_base_link`）定义的 **B 系**下，与 `--dump_pose` 输出一致。

---

## 4. `pose.yaml`（分段轨迹）要点

典型结构包含 **`gripper_open` / `gripper_close`** 与 **`segments`** 列表。每一段可包含：

| 字段 | 说明 |
|------|------|
| `steps` | 本段仿真步数；位置/姿态在该段内从段初插值到段末。 |
| `l_hold` / `r_hold` | 为 `true` 时该臂本段保持段初位置（与目标字段互斥一侧）。 |
| `l_target_b` / `r_target_b` | 段末 **绝对** B 系末端位置 `[x,y,z]`（米）。 |
| `l_delta_b` / `r_delta_b` | 相对本段起点的位移 `[dx,dy,dz]`。 |
| `l_quat_b` / `r_quat_b` | 可选，段末姿态四元数 `x,y,z,w`。 |
| `gripper_l` / `gripper_r` | `open` \| `close` \| `hold` \| 数值；整段内保持该夹爪指令。 |

多段首尾相接：上一段终点作为下一段起点。仓库内 **`pose.yaml`** 可作为抓取-放置分段示例参考。

---

## 5. 其他常用参数（可选）

- **`--dump_pose`**：不调轨迹，仅打印当前场景下末端与相关物体在 B 系下的位置，用于标定 YAML 中的目标点。
- **`--delta_b BX BY BZ`**：无分段文件时，可用单段 **双臂同加** 位移（与命令行/文件中的绝对目标二选一逻辑见脚本内 `_resolve_trajectory_args`）。
- **`--l_target_b` / `--r_target_b`**、**`--l_quat_b` / `--r_quat_b`**：无 `segments` 时的单段绝对目标；需成对提供左右目标位置。
- **`--steps`**：单段模式下的步数覆盖。
- **`--gripper_open` / `--gripper_close`**：覆盖文件中的夹爪开闭电机标定。

完整参数说明见 `data_collection_scripted.py` 内 `argparse` 与文件头注释。

---

## 6. 日志

脚本使用日志器名 `ScriptedGrasp`，默认写入：

`src/examples/dataCollection/logs/scripted_grasp.log`

---

## 7. 与「数据采集 / HDF5」的关系

- **默认（不加 `--record_hdf5`）**：仅运行脚本轨迹；`DataCollectionManager` 的 `data_storage=None`，不落 HDF5。
- **`--record_hdf5`**：将 `OpenLoongDataStorage` 接入 `DataCollectionManager`，在任务成功结束时 **`save_data`**，写入与 **`data_collection_tele`** 相同的单元目录与 `record/proprio_stats.hdf5` 字段；失败则 **`clear_data`**。脚本内 **`save_video` 默认为 False**（只落 HDF5）；需要与 tele 一样另存视频时可在代码中打开 `save_video` 并设置 `set_video_path`。
- **回放 / 增强采集**：对 `dataset/openloong/<level>` 下已保存的单元，使用 **`data_collection_aug.py`**（`DataDevice` + 相同 `level`），即可按 tele 录制数据的方式回放；本脚本写入的 HDF5 布局与之兼容。

---

## 8. 数据回放（`data_collection_aug.py`）

本节说明如何用 **增强采集 / 回放** 脚本，读取由 **`data_collection_scripted.py --record_hdf5`**（或 **`data_collection_tele`）写入的 HDF5，在仿真里 **复现末端指令** 并可选 **再采一条增强数据**。

### 8.1 前置条件

- 已启动 **OrcaLab**，gRPC 与采集时一致（默认 `localhost:50051`）。
- **`dataset/openloong/<level>/`** 下至少存在一个 **单元子目录**（一般为 UUID），其内包含 **`record/proprio_stats.hdf5`**，且含 HDF5 中的 **`task_info` / `scene_info`**（与 tele 落盘格式一致）。
- **`--level`** 必须与录制该批数据时使用的 **`--level` 相同**（路径依赖 `.../dataset/openloong/<level>/`）。

### 8.2 回放如何工作（概念）

- **`DataDevice`** 在 `dataset/openloong/<level>/` 下扫描 **子目录**，按目录名将每个单元视为一次可回放序列；`loop_playback=True` 时队列用完后会从头再播。
- 每个单元内读取 **`record/proprio_stats.hdf5`**，将 **`/action/end/position`**、**`/action/end/orientation`** 等按时间步送给 **`add_arm_osc_openloong_data_controller`**，由 OSC 跟踪录制的末端轨迹；夹爪同理绑定 HDF5 中的电机/关节相关字段。
- **`DataCollectionManager`** 处于 **`AUGMENTATION`** 模式：`update_scene()` 会根据 HDF5 中的 **`scene_info`** 等恢复场景与物体位姿，再执行本回合（与纯 tele 采集时的 **`TELECONTROL`** 分支不同）。
- **输出**：新数据默认写入 **`aug_dataset/openloong/<level>/`**（与读取的 **`dataset/...`** 分离），避免覆盖原始 tele/脚本采集。

### 8.3 推荐命令

在 **`src/examples/dataCollection`** 目录下执行（与采集脚本相同）：

```bash
cd src/examples/dataCollection
python data_collection_aug.py --level shop_scaning --agent_name openloong --task_config scripted-example.yaml
```

说明：

| 参数 | 回放时的注意点 |
|------|----------------|
| `--level` | 必须与 HDF5 所在路径中的 `<level>` 一致（见上文）。 |
| `--agent_name` | 脚本录制为 OpenLoong 时使用 **`openloong`**。 |
| `--task_config` | 场景 YAML，宜与采集时 **同一套或等价** 的任务/物体配置，以便 `scene_info` 与场景一致；若差异过大可能导致恢复或任务异常。 |

日志默认在 **`logs/data_collection.log`**。

### 8.4 与「仅脚本、不回放」的区别

| 方式 | 输入 | 控制来源 |
|------|------|----------|
| `data_collection_scripted.py` | `pose.yaml` 等 | 预计算轨迹 → `ScriptedTrajectoryDevice` |
| `data_collection_aug.py` | `dataset/...` 下 HDF5 | 录制时的观测/动作序列 → `DataDevice` → OSC |

二者都走 OSC 双臂与 2F85，但 **回放完全由磁盘上的 HDF5 驱动**，不再读取 `pose.yaml`。

---

## 9. 相关文件

| 文件 | 作用 |
|------|------|
| `src/examples/dataCollection/data_collection_scripted.py` | 脚本入口与轨迹设备 |
| `src/examples/dataCollection/data_collection_aug.py` | HDF5 回放与增强采集 |
| `src/examples/dataCollection/pose.yaml` | 分段位姿与夹爪示例 |
| `src/examples/dataCollection/scripted-example.yaml` | 场景/物体任务配置示例 |
| `src/conf/openloong_conf.py` | OpenLoong 臂、末端 site、夹爪、基座名 |
| `QUICK_START.md` | 项目快速上手（总览） |

更完整的开发与控制器说明见 **`DEVELOPER_GUIDE.md`**。
