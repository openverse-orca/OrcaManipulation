# LiDAR 建图 → 定位 → 导航 使用指南

本系列共 3 个入口脚本（外加 1 个前置脚本），全部在 `src/examples/dataCollection/` 下，构成 **建图 → 定位 → 导航** 的完整链路。

## 1. g1_lidar_kiss_slam.py —— 建图（前置）

在线 KISS-SLAM：订阅 gRPC LiDAR 点云流，实时跑 SLAM 建图，输出到 `slam_output/`；
加 `--record-to` 可同时把逐帧局部系点云 + GT 位姿录下来（`frames/*.npy` + `meta.npz`）
—— 这是后面 refmap 重建的输入。

```bash
# 录制 + 在线建图 (--gt-odom: 真值里程计锚定, 开阔场景推荐)
# 开车需要另开一个终端跑 tele 遥控, 录完 Ctrl-C 结束
python g1_lidar_kiss_slam.py --record-to dataset/slam_diag_new --gt-odom
```

## 2. g1_map_tools.py —— 地图工具（①）

两个子命令：

- **`refmap`**：用录制数据的 GT 位姿重建几何一致的参考地图（`ref_map.ply`，锚定录制首帧）
  —— 比 SLAM est 位姿建的 `global_map.ply` 质量高一个量级，是定位/导航的推荐地图
- **`grid`**：PLY 点云 → 2D 占据栅格（`ref_grid.npz/pgm/yaml/png`），
  z 高度带过滤地面点 + 机器人半径膨胀，供 A* 规划用

```bash
python g1_map_tools.py refmap --frames dataset/slam_diag --out slam_output/ref_map.ply
python g1_map_tools.py grid --map slam_output/ref_map.ply --out-prefix slam_output/ref_grid
```

## 3. g1_lidar_localizer.py —— 实时重定位（②）

在已有参考地图上做在线定位：GT/里程计锚定 + **有界相关性扫描匹配**
（粗 2D 似然场筛 top-40 → 细 3D KDTree 定夺，以预测位姿为窗口中心，
修正量有界、无累积漂移）。核心类 `CorrelativeMatcher` 也被 autopilot 复用。

```bash
# 离线回放验证 (用录制的 662 帧)
python g1_lidar_localizer.py replay --map slam_output/ref_map.ply --frames dataset/slam_diag
```

## 4. g1_autopilot.py —— 自主导航（③）

进程内自带仿真装配（与 tele 相同）：A* 在 ref_grid 上全局规划（禁切角 + 视线捷径化）
→ 纯跟踪 → Ackermann 几何 → DATA 模式下发轮速/转角。
定位默认 GT 里程计锚定，`--use-matcher` 启用相关性匹配修正；到点后 task_status END 自动退出。

```bash
python g1_autopilot.py --goal 5,5 --viz                     # 基础导航 + viser 可视化
python g1_autopilot.py --goal 5,5 --use-matcher --viz        # 带在线重定位修正
```

## 数据流

```
g1_lidar_kiss_slam.py (录制)
        │  dataset/slam_diag/  (frames/*.npy + meta.npz)
        ▼
g1_map_tools.py refmap  →  slam_output/ref_map.ply   (参考地图, 锚定录制首帧)
        │
        ▼
g1_map_tools.py grid   →  slam_output/ref_grid.npz   (占据栅格, 供 A* 规划)
        │
        ▼
g1_autopilot.py        (消费栅格做规划 + 地图做匹配修正)
```

另外 tele 脚本 `g1_omnipicker_collection_tele.py` 不变，仍是人工遥操作的入口。

## 常用参数说明（g1_autopilot.py）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--goal` | 必填 | 地图系目标点 `x,y`（米），范围看 `ref_grid.png`，需在 x[0,15] y[-2.7,10.5] 内 |
| `--grid` | `slam_output/ref_grid.npz` | 规划用占据栅格 |
| `--map` | `slam_output/ref_map.ply` | 参考地图（仅 `--use-matcher` 需要） |
| `--use-matcher` | 关 | 启用有界相关性匹配在线修正 (~5Hz)；默认仅 GT 里程计锚定 |
| `--viz` | 关 | viser 浏览器可视化（路径/机器人），打开 `http://127.0.0.1:8090` |
| `--max-speed` | 0.4 | 巡航速度 m/s |
| `--lookahead` | 0.6 | 纯跟踪前视距离 m |
| `--goal-tol` | 0.25 | 到点判定半径 m |
| `--replan-dist` | 0.8 | 偏离路径超过该距离触发重规划 m |
