# BW10 数采与回放说明

本目录为 **BW10（青龙双臂，商超扫码等）** 的独立说明；**可执行脚本**在同级目录的 [`../dataCollection/`](../dataCollection/) 下。运行前请确保仿真 / OrcaLab 已就绪，VR（Pico + OrcaGymCtrl）已按项目要求配置。脚本默认连接 `localhost:50051`（可用参数覆盖）。

## 相关文件

| 路径 | 作用 |
|------|------|
| [../dataCollection/bw10_collection_tele.py](../dataCollection/bw10_collection_tele.py) | Pico 遥操作数采：双臂 **OSC**、双手 **2F85**、任务 `ScanQRTask` |
| [../dataCollection/bw10_collection_replay.py](../dataCollection/bw10_collection_replay.py) | 从 HDF5 轨迹回放，**不落盘** |
| [../dataCollection/3c_scan.yaml](../dataCollection/3c_scan.yaml) | 常用场景配置示例 |
| [../../conf/bw10_conf.py](../../conf/bw10_conf.py) | 关节、夹爪、末端 site 等 |
| [../../dataStorage/bw01_data_storage.py](../../dataStorage/bw01_data_storage.py) | 写盘：每回合 `<uuid>/record/proprio_stats.hdf5` |

日志（相对 `dataCollection` 目录）：数采 `logs/bw10_collection.log`，回放 `logs/bw10_replay.log`。

---

## 数据根目录怎么定（数采）

脚本在 **`src/examples/dataCollection/`**（下文记 **`dataCollection`**）下拼接一段**相对路径**作为「数据根」；其下每个 **`uuid` 文件夹**为一回合，内含 `record/proprio_stats.hdf5`、`video/` 等。

优先级（从高到低）：

1. **`--dataset_root <路径>`**  
   直接指定相对 `dataCollection` 的根，例如 `dataset/bw10/bw10`。  
   **若指定，则不再使用下面拼接方式。**

2. **`--level`**（通常与 **`--agent_name`** 合用）  
   数据根为：**`<dataset_parent>/<agent_name>/<level>/`**  
   - 默认 `--dataset_parent dataset`、`--agent_name bw10`  
   - 例：`--level bw10` → **`dataset/bw10/bw10/`**

3. **都不加时**  
   数据根为单层 **`dataset/`**（下一层直接是多个 uuid）。

其它常用参数见 `python bw10_collection_tele.py -h`，例如 `--task_config`（默认 `3c_scan.yaml`）、`--orcagym_addr`、`--frame_skip`、`--time_step`。

---

## 遥操作采集：`bw10_collection_tele.py`

**操作建议**：双臂由 VR Transform 控制（OSC）；夹爪见项目 VR 映射；**左手柄握持键**开始 / 结束一条轨迹记录；任务为扫码相关 `ScanQRTask`。

### 示例一：按层级保存（与青龙目录习惯一致）

```bash
cd src/examples/dataCollection
python bw10_collection_tele.py \
  --task_config 3c_scan.yaml \
  --level bw10 \
  --agent_name bw10
```

数据落在：`dataCollection/dataset/bw10/bw10/<uuid>/record/proprio_stats.hdf5`。

### 示例二：手写数据根

```bash
python bw10_collection_tele.py \
  --task_config 3c_scan.yaml \
  --dataset_root dataset/bw10/bw10
```

与示例一在默认参数下路径等价。

---

## 回放：`bw10_collection_replay.py`

从「数据根」下每个**一级子目录（uuid）**读取 `record/proprio_stats.hdf5`，在仿真里跟踪轨迹；**不写新 HDF5**。任务同为 `ScanQRTask`。

**路径约定应与数采一致**：回放侧 `--level` / `--agent_name` / `--dataset_parent` / `--dataset_root` 的规则与数采相同。

若 `dataset` 下混有 **`openloong`** 等非 uuid 目录，`DataDevice` 可能误把目录名当回合，此时请用 **`--replay_dir`**，指向「下面**只有** uuid 子文件夹」的那一层父路径。

### 回放模式 `--replay_mode`

| 取值 | 含义 |
|------|------|
| `osc`（默认） | 末端 OSC 跟踪 HDF5 末端位姿（与遥操作臂控一致） |
| `ik` | IK 跟踪末端位姿 |
| `position` | 跟踪关节序列 `/action/joint/position` |

### 循环回放

- **`--replay_loops N`**：数据根下全部回合顺序播完算 **1 轮**，重复 **N** 轮；默认 `1`。  
- 默认每轮开始前将手臂关节重置为 neutral；加 **`--no_reset_pose_between_loops`** 可关闭该重置。

### 示例三：与示例一同路径回放

```bash
cd src/examples/dataCollection
python bw10_collection_replay.py \
  --task_config 3c_scan.yaml \
  --level bw10 \
  --agent_name bw10 \
  --replay_mode osc
```

### 示例四：显式指定 uuid 父目录

```bash
python bw10_collection_replay.py \
  --task_config 3c_scan.yaml \
  --replay_dir ./dataset/bw10/bw10 \
  --replay_mode osc
```

### 示例五：重复回放 5 轮（例如统计成功率）

```bash
python bw10_collection_replay.py \
  --task_config 3c_scan.yaml \
  --replay_dir ./dataset/bw10/bw10 \
  --replay_loops 5 \
  --replay_mode osc
```

---

## 磁盘目录结构（示意）

```
dataCollection/
  └── dataset/                      # 或 aug_dataset / 自定义 dataset_parent
      └── bw10/
          └── bw10/                  # 使用 --level / --agent_name 拼接时
              └── <uuid>/
                  ├── record/
                  │   └── proprio_stats.hdf5
                  └── video/
                      └── ...
```

未使用 `--level` 时，也可能是 **`dataCollection/dataset/<uuid>/...`** 的扁平结构。

---

更多通用采集流程见仓库根目录 [README.md](../../../README.md) 与 [QUICK_START.md](../../../QUICK_START.md)。
