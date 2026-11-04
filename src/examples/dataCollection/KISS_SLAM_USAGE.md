# KISS-SLAM 在线建图使用说明

基于 [KISS-SLAM](https://github.com/PRBonn/kiss-slam) 的 g1_omnipicker 在线 LiDAR SLAM 方案。
**全程不依赖 ROS2**，数据通过 gRPC 直接从仿真拉取，配准、回环、位姿图优化全部在 Python 进程内完成。

## 1. 系统架构

```
┌─────────────────────────┐    gRPC (localhost:50051)    ┌──────────────────────────┐
│ 仿真 + 遥操作            │ ───────────────────────────► │ g1_lidar_kiss_slam.py    │
│ g1_omnipicker_collection │   QueryLiDARPointCloud      │  ① 拉取世界系点云         │
│ _tele.py (无需 --odom)   │   QueryBodyPosMatQuat       │  ② 变换到雷达局部系        │
└─────────────────────────┘                             │  ③ KISS-SLAM 逐帧处理     │
                                                        │  ④ 结束时位姿图优化输出    │
                                                        └──────────────────────────┘
```

说明：

- gRPC 返回的点云是**世界坐标系**数据，脚本通过查询雷达 body 世界位姿做 `p_local = Rᵀ(p_world − t)` 变换，
  恢复成传感器局部系后再喂给 SLAM（这是 KISS-SLAM 的标准输入格式）
- 仿真点云为瞬时快照、无运动畸变，配置文件中 `deskew: false`（去畸变必须关闭）
- 里程计由 KISS-SLAM 从点云配准自估计，不使用仿真真值；真值位姿仅用于结束后评估精度（ATE）

## 2. 前置条件

| 依赖 | 说明 |
|---|---|
| conda 环境（如 `lidar`，Python 3.12） | 运行 SLAM 脚本 |
| `kiss-slam` | `pip install kiss-slam`（会自动带上 kiss-icp、numpy 等依赖） |
| `grpcio`、`orca_gym` | gRPC 数据拉取 |
| `viser` | 仅 `--viz` 可视化需要（`pip install viser`，浏览器打开） |

不需要：ROS2、rviz、tf、任何 ROS 话题。

## 3. 启动步骤

### 终端 1：仿真 + 遥操作（或任何让 gRPC server 跑起来的方式）

```bash
cd <仓库路径>/src/examples/dataCollection
python g1_omnipicker_collection_tele.py --level default
```

注意**不加** `--odom`（那是 rtabmap 方案的开关，本方案用不到）。

### 终端 2：在线 SLAM

```bash
cd <仓库路径>/src/examples/dataCollection
conda activate lidar
python g1_lidar_kiss_slam.py --viz --gt-odom
```

用遥杆遥控机器人在场景中移动即可实时建图。结束后在终端按 **Ctrl+C**，脚本会自动做位姿图优化并保存全部结果。

> **推荐在仿真场景中使用 `--gt-odom`**（原因见下节"里程计模式说明"）：实测同一圈轨迹，
> 纯扫描配准 ATE 5.9m / 回环 0；`--gt-odom` ATE 0.17m / 回环正常。

### 里程计模式说明

| 模式 | 开关 | 原理 | 适用条件 |
|---|---|---|---|
| 扫描配准里程计 | （默认） | 纯 ICP 扫描配准估计运动，无任何先验 | 场景结构丰富（墙、障碍物多方向可见）的真实雷达数据 |
| GT 里程计 | `--gt-odom` | 用 gRPC 真值位姿作为逐帧里程计，回环检测/位姿图优化/建图照常工作 | 本仿真场景：仓库尺度大（±80m）、地面平坦开阔，逐帧扫描配准的**平移量基本不可观**（旋转可由地面/墙面约束恢复，平移沿行进方向退化），架构等价于 rtabmap + `/odom` |

两种模式输出格式完全一致，均可配合 `--viz`、`--record-to` 使用。

## 4. 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--viz` | 关 | 开启 open3d 实时可视化窗口 |
| `--gt-odom` | 关 | GT 里程计模式（见上节"里程计模式说明"） |
| `--duration 秒数` | 无限 | 运行指定秒数后自动结束（调试用） |
| `--record-to 目录` | 无 | 把局部系原始帧同步录盘（frames/*.npy + meta.npz），供离线复跑 |
| `--out 目录` | `slam_output` | 结果输出目录 |
| `--config 路径` | `kiss_slam_g1_indoor.yaml` | KISS-SLAM 配置文件 |
| `--hz` | 10 | 点云拉取频率 |
| `--addr` | `localhost:50051` | gRPC 服务器地址 |
| `--entity` | `LiDAR` | LiDAR 实体名 |
| `--lidar-body` | `g1_omnipicker_LiDAR_body` | 雷达 body 名（查世界位姿用） |

## 5. 实时可视化窗口（--viz）

| 元素 | 含义 |
|---|---|
| 亮色点云 | 结构点（墙、货架等，各帧点云按估计位姿拼合、体素去重，实时增长） |
| 土黄色点云 | 地面点 |
| 绿色线 | 估计轨迹 |
| 红色线段 | 回环约束（检测到回环时出现） |

启动后自动在浏览器打开 `http://127.0.0.1:8080`（端口被占时自动顺延到 8081-8084）。

- **鼠标操作**：左键拖拽旋转 / 右键拖拽平移 / 滚轮缩放；左上角场景树可单独隐藏/显示各类元素
- **关闭网页 ≠ 结束**：关掉浏览器页签后程序继续建图（服务在后台），重新打开 URL 即可再看
- **保存结果用终端 Ctrl+C**：结束时自动做位姿图优化并输出全部文件

## 6. 输出说明（--out 目录）

| 文件 | 内容 |
|---|---|
| `poses_kitti.txt` / `poses_tum.txt` | 估计轨迹（fine-grained 位姿图优化后，KITTI/TUM 格式） |
| `gt_poses_kitti.txt` | GT 轨迹（雷达真值位姿，来自仿真） |
| `ate.txt` | ATE RMSE（估计 vs GT，Umeyama 对齐后）、扫描数、回环数 |
| `local_maps/plys/*.ply` | 各局部地图点云，在世界系下已对齐，可用 MeshLab / CloudCompare / Open3D 查看拼合效果 |
| `global_map.ply` | **全局累积地图**（体素去重 0.1m，实时可视化显示的就是它），推荐直接用这个看建图效果 |
| `trajectory.png` | 俯视轨迹图，红色线段为回环约束 |

终端结束时会打印关键指标：

```
[INFO] loop closures found: N
[INFO] ATE RMSE (vs GT, aligned): X.XXXX m over NNN scans
```

## 7. 建图质量检查

跑一段包含"绕一圈回到起点"的轨迹，检查：

1. `closures > 0` —— 回环检测生效（没触发说明轨迹太短或没形成回环）
2. ATE RMSE 应在厘米级 —— 仿真点云无噪声，精度理论上限很高
3. `trajectory.png` 轨迹闭合、红线连接位置正确
4. `local_maps/plys` 拼合后墙面等结构无重影

## 8. 参数调优（kiss_slam_g1_indoor.yaml）

默认值已按室内仓储场景调好（voxel 0.25m、局部地图 5m 切分）。常见调整：

| 现象 | 调整方向 |
|---|---|
| 大场景远端配准不稳 | 增大 `odometry.preprocessing.max_range`（当前 30m） |
| 回环漏检 | 降低 `loop_closer.overlap_threshold`（当前 0.4）或 `detector.hamming_distance_threshold` |
| 回环误检 | 提高上述两个阈值 |
| 节点太稀（回环粒度粗） | 减小 `local_mapper.splitting_distance`（当前 5m） |
| 车速很慢、地图更新太慢 | 减小 `local_mapper.splitting_distance` |

## 9. 与 rtabmap 方案的关系

两套方案共用同一个 tele 脚本，可按需选择：

| | rtabmap（见 SLAM_USAGE.md） | KISS-SLAM（本文档） |
|---|---|---|
| ROS2 依赖 | 必须 | 不需要 |
| 里程计来源 | 仿真真值（freejoint，`--odom`） | 默认点云配准自估计；`--gt-odom` 时用仿真真值 |
| 适用场景 | 需要标准 ROS 生态（导航栈等） | 轻量部署、无 ROS 环境 |
| 启动方式 | `--odom` + ROS 话题链路 | 直接运行本脚本 |

注：rtabmap 的建图质量依赖 `/odom` 真值里程计；`--gt-odom` 模式与之同构（真值里程计 + 扫描建图/回环），
二者的建图上限一致。默认的纯扫描配准模式在开阔仿真场景中平移退化（见"里程计模式说明"），
但在结构丰富的真实场景中可独立工作，两种模式保留了无先验 SLAM 的完整能力。
