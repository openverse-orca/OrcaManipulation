# OrcaManipulation v5

基于 OrcaManipulation / OrcaLab 仿真的 **LeRobot 直采 → 训练 → 推理** 最小可交付流程。

本仓库从上游 [`openverse-orca/OrcaManipulation`](https://github.com/openverse-orca/OrcaManipulation) 裁剪而来，只保留 d12 仓储任务（`competition_warehouse`）的数据采集、回放、推理主线及其必要依赖，不含示例数据。

```
安装环境  ──►  启动仿真  ──►  采集数据  ──►  回放校验  ──►  上传训练  ──►  推理
 第0步         第1步          第2步         第3步          服务器端       本地连接策略服务
```

## 👉 从这里开始

完整、分步、可照抄的运行手册见 **[docs/快速开始.md](docs/快速开始.md)**（强烈建议第一次使用先看这个）。
服务器训练/策略服务部署见 **[docs/训练和推理.md](docs/训练和推理.md)**。

下面是各阶段的命令摘要，细节与成功标志请看快速开始。

```bash
# 0. 安装环境（首次）
bash install.sh        # 详见快速开始第 0 步

# 1. 启动仿真（终端 1，保持运行）
orcalab                # 选 competition_warehouse 场景 + 布局，点「启动仿真」

# 2. 采集数据（终端 2）
python collect_lerobot.py --pose_file pose_mp_no_barcode.yaml \
    --rand_file rand_no_barcode.yaml --lerobot_out /path/to/out_dataset \
    --repo_id your_org/competition_warehouse_quat --episodes 50 --fps 30

# 3. 回放校验
python replay_lerobot.py --dataset_dir /path/to/out_dataset --episode 01 --sleep

# 4. 推理（可选，需先在服务器训练并启动 openpi policy server）
python eval.py --task_config competition_warehouse.yaml \
    --host localhost --port 8010 --max_steps 500
```

采集 / 回放 / 推理脚本均在 `src/examples/d12_competition_warehouse/` 下运行。

---

## Pipeline 总览

```
LeRobot 直采 (collect_lerobot.py)
        │  图像 + 16维末端位姿(state) + action，构造时按仿真时间对齐，fps=30
        ├──────────────► 回放校验 (replay_lerobot.py)
        ├──────────────► openpi JAX 训练（服务器端，见 docs/训练和推理.md）
        └──────────────► 推理 (eval.py)  ◄── 连接 openpi policy server
```

- **单一产数路径**：LeRobot 直采。采集时在仿真主循环内按仿真时间（`env.data.time`）每 1/30s 同步抓取 3 路相机 RGB + 末端状态，当场写入 LeRobot v2.1 数据集（“对齐即构造”）。
- **取图走内存相机流**：通过 `CameraWrapper` WebSocket H.264 流直取最新帧（~ms 级、不落盘），不再用 `env.get_frame_png` 逐帧磁盘往返，仿真维持目标频率。
- **视频后台编码**：每集只同步落 parquet + 元数据，av1 视频编码交给后台线程并行，主采集循环不被编码阻塞。
- 不含任何离线 HDF5→LeRobot 转换路径（依赖墙钟视频帧数对齐、有隐含减速比假设，已弃用）。

## 环境依赖

- conda 环境 `orcalab_lerobot`，Python 3.12
- `orca_gym`（OrcaStudio 渲染端独立进程，默认 gRPC `localhost:50051`）
- `lerobot` 0.3.x、`openpi_client`（已 vendor 于 `third_party/openpi/packages/openpi-client`）
- 其余见 [`requirements.txt`](requirements.txt)

安装步骤见 [快速开始第 0 步](docs/快速开始.md#第-0-步一次性环境安装首次使用才需要)。

## 附录

### State / Action 布局（16 维）

```
0-2    左臂末端位置 (x, y, z)         相对 base_link
3-6    左臂末端四元数 (x, y, z, w)
7-9    右臂末端位置 (x, y, z)
10-13  右臂末端四元数 (x, y, z, w)
14     左夹爪 [0, 1]
15     右夹爪 [0, 1]
action[i] = state[i+1]（next-state 约定）
```

相机：`camera_head_color → cam_head`、`camera_wrist_l_color → cam_wrist_l`、`camera_wrist_r_color → cam_wrist_r`。
WebSocket 端口：head=7070、wrist_l=7080、wrist_r=7090（运行时可用 `ss -tlnp` 核对）。

### 可选命令行开关

| 选项 | 脚本 | 作用 |
|------|------|------|
| `--record-scene` | collect_lerobot.py | 记录每集 C12C 随机位姿到 `c12c_poses.json` |
| `--verbose` | collect_lerobot.py | 打印逐帧采集诊断日志 |
| `--resume` | collect_lerobot.py | 追加到已有数据集（断点续采） |
| `--restore-scene` | replay_lerobot.py | 回放时复原 C12C 位姿，复现采集场景 |
| `--steps_per_frame N` | replay_lerobot.py | 每个 30fps 动作的仿真步数（默认 20） |

### 目录结构

```
src/
  conf/            d12_conf.py（机器人/场景配置）
  controllers/     OSC 臂控制 + 2F85 夹爪 + 任务状态控制
  dataCollectionManager/  episode 主循环
  dataStorage/     d12_data_storage（采集回调，含末端位姿/夹爪）
  devices/         设备/插值器抽象
  envs/            dataCollection_env（OrcaGym env 入口）
  scene/           scene_manager + 随机化
  task/            任务抽象
  examples/d12_competition_warehouse/
                   collect_lerobot.py  直采（内存相机流 + 后台编码）
                   replay_lerobot.py   回放
                   eval.py             推理
                   data_collection_mp.py  env/manager 装配（直采复用）
                   run_collection.py      抓取/MP 脚本框架（直采复用依赖）
                   *.yaml / *.json     场景/位姿/随机化配置
third_party/openpi/packages/openpi-client/   openpi 客户端（推理用）
docs/              快速开始 + 训练和推理说明
```
