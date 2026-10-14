# g1_omnipicker LiDAR SLAM 建图使用说明

基于 rtabmap 的 3D 建图流程：仿真中遥操作机器人移动，LiDAR 点云 + freejoint 里程计
输入 rtabmap 实时建图。

## 系统架构

三个进程协同工作，按顺序启动：

| 进程 | 脚本 | 作用 |
|---|---|---|
| 遥操作/仿真 | `g1_omnipicker_collection_tele.py` | 驱动仿真与遥操作；自动拉起里程计桥接子进程 |
| 里程计桥接 | `ros2_odom_bridge.py` | 由 tele 脚本自动启动，无需手动运行；发布 `/odom` 和 TF `odom → base_scan`（freejoint 真值位姿） |
| LiDAR 桥接 | `g1_lidar_ros2_bridge.py` | 查询 gRPC 点云（世界系），变换到机器人基座系后发布 `/point_cloud`、`/scan` |
| rtabmap | ROS2 launch | 订阅 `/odom` + `/point_cloud` 建图，发布 TF `map → odom` |

TF 树：`map → odom → base_scan`（`odom → base_scan` 由里程计桥接发布，
`map → odom` 由 rtabmap 发布）。

## 前置条件

- ROS2 Humble 已安装（`/opt/ros/humble`）
- conda 环境 `orcalab9`（遥操作脚本依赖）
- LiDAR 桥接使用 Python 3.10 环境并安装 grpcio / numpy / orca_gym
  （ROS2 Humble 的 rclpy 仅提供 python3.10 版本，3.12 无法 import），
  脚本会自动探测系统 ROS2 并处理 PYTHONPATH，无需手动 source

## 启动步骤

### 1. 启动遥操作（终端 1）

会同时启动仿真、里程计桥接（自动）：

```bash
conda activate orcalab9
cd <仓库路径>/src/examples/dataCollection
python g1_omnipicker_collection_tele.py --level default
```

### 2. 启动 LiDAR 桥接（终端 2）

```bash
conda activate <python310环境名>
cd <仓库路径>/src/examples/dataCollection
python g1_lidar_ros2_bridge.py --entity LiDAR --frame_id base_scan --hz 10
```

参数说明：
- `--entity`：LiDAR 实体名（默认 `LiDAR`）
- `--frame_id`：点云/雷达消息坐标系（默认 `base_scan`，与里程计子坐标系一致）
- `--base_joint`：机器人基座 freejoint 名（默认 `g1_omnipicker_free_joint`，
  用于世界系点云 → 基座系变换；换机器人时改此项）
- `--hz`：发布频率，默认 10 Hz

注意：**不要同时运行** `python -m orca_gym.tools.lidar_ros2_bridge`（venv 内未修改的
旧版本，发布的是世界系点云）。同一话题出现两个发布者会导致建图来回跳变。

### 3. 启动 rtabmap 建图（终端 3）

```bash
source /opt/ros/humble/setup.bash
ros2 launch rtabmap_launch rtabmap.launch.py \
    subscribe_scan_cloud:=true scan_cloud_topic:=/point_cloud \
    scan_topic:=/dummy_scan \
    depth:=false \
    icp_odometry:=false visual_odometry:=false \
    odom_topic:=/odom \
    approx_sync:=true approx_sync_max_interval:=0.15 \
    frame_id:=base_scan \
    rtabmap_viz:=true rviz:=false \
    rtabmap_args:="--delete_db_on_start"
```

关键参数说明：
- `icp_odometry` / `visual_odometry` 均为 `false`：使用 `/odom`（仿真真值）作为里程计，
  不让 rtabmap 自行推算
- `odom_topic:=/odom`：注意必须带 `/`（默认相对名会解析到 `/rtabmap/odom` 导致收不到）
- `approx_sync:=true`：`/odom` 与 `/point_cloud` 来自两个进程、时间戳不可能完全相等，
  exact sync 会导致收不到任何数据；`approx_sync_max_interval` 限制配对时间间隔
- `rtabmap_args:="--delete_db_on_start"`：清空旧地图重新建图，调试阶段建议保留

启动后用遥杆驱动机器人移动即可建图。

## 数据话题一览

| 话题 / TF | 类型 | 发布者 | 说明 |
|---|---|---|---|
| `/point_cloud` | sensor_msgs/PointCloud2 | LiDAR 桥接 | 基座系点云（含 intensity），10 Hz |
| `/scan` | sensor_msgs/LaserScan | LiDAR 桥接 | 2D 扫描（最低垂直层） |
| `/odom` | nav_msgs/Odometry | 里程计桥接 | freejoint 真值位姿 + 速度 |
| TF `odom → base_scan` | tf2 | 里程计桥接 | 随机器人运动实时更新 |
| TF `map → odom` | tf2 | rtabmap | 建图修正，慢变化 |

## 验证与检查

```bash
source /opt/ros/humble/setup.bash

# 确认两个关键话题都在发布（正常 /odom 几十 Hz，/point_cloud 10 Hz）
ros2 topic hz /odom
ros2 topic hz /point_cloud

# 确认 /point_cloud 只有一个发布者（出现 2 个会导致地图来回跳变）
ros2 topic info /point_cloud -v

# 查看完整 TF 树（应看到 map -> odom -> base_scan）
ros2 run tf2_tools view_frames && evince frames.pdf
```

RViz 快速目检：Fixed Frame 设为 `odom`，添加 PointCloud2 显示 `/point_cloud`，
驱动机器人移动——点云应呈现为**静止不动的世界**（墙壁不重影、不拖影）。
若出现拖影/重影，说明点云坐标系与里程计不一致，检查 LiDAR 桥接是否在运行
修改后的 `g1_lidar_ros2_bridge.py`。

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| rtabmap 报 "Did not receive data since 5 seconds" | 通常是 `approx_sync` 为 `false`（exact sync）导致；确认启动命令中 `approx_sync:=true`；其次检查两个话题是否都在发布 |
| 新扫到的点云来回跳变 | `/point_cloud` 上存在两个发布者（新旧 LiDAR 桥接同时运行）；`ros2 topic info /point_cloud -v` 检查并杀掉多余进程 |
| 建图结果异常残留旧数据 | 重启 rtabmap 时保留 `--delete_db_on_start` |

## 关闭顺序

1. `Ctrl+C` 关闭 rtabmap（终端 3）
2. `Ctrl+C` 关闭 LiDAR 桥接（终端 2）
3. `Ctrl+C` 关闭遥操作（终端 1，里程计桥接子进程会随之自动退出）
