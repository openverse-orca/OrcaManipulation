"""
g1 omnipicker 在线 LiDAR SLAM — 基于 KISS-SLAM, 无 ROS 依赖

数据链路 (纯 gRPC, 不依赖 ROS2):
  gRPC server (50051)
    ├── QueryLiDARPointCloud  -> 世界系点云 (bin x layer x 3)
    └── QueryBodyPosMatQuat   -> 雷达世界位姿 (GT, 用于把世界系点云变换回雷达局部系)
  每帧: p_local = R^T (p_world - t)  ->  KissSLAM.process_scan()
  结束 (Ctrl+C 或 --duration 到时): 位姿图优化 -> 输出轨迹 / 局部地图 PLY / ATE 评估

用法 (lidar conda 环境, 需安装 kiss-slam / grpcio / orca_gym):
  python g1_lidar_kiss_slam.py                          # 实时 SLAM, Ctrl+C 结束
  python g1_lidar_kiss_slam.py --duration 60            # 跑 60 秒自动结束
  python g1_lidar_kiss_slam.py --viz                    # 开启 open3d 实时可视化
  python g1_lidar_kiss_slam.py --record-to dataset/run1 # 同时把原始帧录盘 (供离线复跑)

输出 (out_dir, 默认 slam_output):
  poses_kitti.txt / poses_tum.txt   估计轨迹 (fine-grained 位姿图优化后)
  gt_poses_kitti.txt                GT 轨迹 (雷达真值位姿)
  ate.txt                           ATE RMSE (估计 vs GT, Umeyama 对齐后)
  local_maps/plys/*.ply             各局部地图点云 (世界系拼合即全局地图)
  global_map.ply                    体素去重后的全局累积地图 (0.1m, 逐帧估计位姿拼合)
  trajectory.png                    轨迹 + 回环约束可视化
  (若 --record-to) frames/*.npy + meta.npz   原始数据, 可离线复跑

注意:
  - 点云是世界系数据, 必须变换回雷达局部系再喂给 SLAM (本脚本已处理)
  - 仿真点云为瞬时快照, 无运动畸变, 配置中 deskew 必须为 false
"""

import argparse
import os
import time

import grpc
import numpy as np

from orca_gym.protos import mjc_message_pb2
from orca_gym.protos import mjc_message_pb2_grpc

from kiss_icp.voxelization import voxel_down_sample
from kiss_slam.config import load_config
from kiss_slam.slam import KissSLAM

DEFAULT_LIDAR_ENTITY = "LiDAR"
DEFAULT_LIDAR_BODY = "g1_omnipicker_LiDAR_body"
GRPC_OPTIONS = [
    ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
    ("grpc.max_send_message_length", 1024 * 1024 * 1024),
]


# ---------------------------------------------------------------------------
# gRPC 数据层
# ---------------------------------------------------------------------------

def make_stub(addr):
    channel = grpc.insecure_channel(addr, options=GRPC_OPTIONS)
    return mjc_message_pb2_grpc.GrpcServiceStub(channel)


def query_lidar(stub, entity_name):
    """拉取一帧 LiDAR 数据。返回 dict(points_world, ranges, valid, 元数据) 或 None。"""
    request = mjc_message_pb2.LiDARPointCloudRequest(entity_name=entity_name)
    try:
        response = stub.QueryLiDARPointCloud(request, timeout=2.0)
    except grpc.RpcError as e:
        print(f"[WARN] QueryLiDARPointCloud failed: {e.code()}")
        return None
    if response.status == mjc_message_pb2.LiDARPointCloudResponse.NO_DATA:
        return None
    if response.status == mjc_message_pb2.LiDARPointCloudResponse.ENTITY_NOT_FOUND:
        print(f"[ERROR] LiDAR entity not found: {entity_name}")
        return None

    if response.point_data:
        points = np.frombuffer(response.point_data, dtype=np.float32).copy()
        points = points.reshape(response.bin_count, response.vertical_layers, 3)
    else:
        points = np.zeros((response.bin_count, response.vertical_layers, 3), np.float32)
    if response.range_data:
        ranges = np.frombuffer(response.range_data, dtype=np.float32).copy()
        ranges = ranges.reshape(response.bin_count, response.vertical_layers)
    else:
        ranges = np.zeros_like(points[:, :, 0])

    return {
        "points_world": points,
        "ranges": ranges,
        "valid": ranges > 0,
        "max_range": response.max_range,
        "min_range": response.min_range,
    }


def query_lidar_pose(stub, body_name):
    """查询雷达 body 的世界位姿。返回 (t(3,), R(3x3)) 或 None。"""
    request = mjc_message_pb2.QueryBodyPosMatQuatRequest(body_name_list=[body_name])
    try:
        response = stub.QueryBodyPosMatQuat(request, timeout=2.0)
    except grpc.RpcError as e:
        print(f"[WARN] QueryBodyPosMatQuat failed: {e.code()}")
        return None
    for item in response.body_pos_mat_quat_list:
        if item.body_name == body_name:
            t = np.array(item.pos, dtype=np.float64)
            R = np.array(item.mat, dtype=np.float64).reshape(3, 3)
            return t, R
    print(f"[ERROR] body not found: {body_name}")
    return None


def transform_to_local(points_world, t, R):
    """世界系点云 -> 雷达局部系: p_local = R^T (p_world - t)。points: (..., 3)"""
    flat = points_world.reshape(-1, 3)
    return (flat - t) @ R


# ---------------------------------------------------------------------------
# 评估工具
# ---------------------------------------------------------------------------

def umeyama_alignment(X, Y):
    """返回 (R, t) 使 Y ≈ R @ X + t (尺度固定为 1)。X, Y: (N, 3)"""
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mx, Y - my
    cov = Yc.T @ Xc / len(X)
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return R, my - R @ mx


def ate_rmse(est, gt):
    """Umeyama 对齐后的 ATE RMSE (m)。est, gt: (N, 3)"""
    n = min(len(est), len(gt))
    if n < 2:
        return float("nan")
    R, t = umeyama_alignment(est[:n], gt[:n])
    aligned = (R @ est[:n].T).T + t
    return float(np.sqrt(np.mean(np.sum((aligned - gt[:n]) ** 2, axis=1))))


# ---------------------------------------------------------------------------
# 全局地图累积 (体素去重)
# ---------------------------------------------------------------------------

class GlobalMapAccumulator:
    """把各帧点云按估计位姿变换到地图系后做体素级去重累积。"""

    def __init__(self, voxel=0.1):
        self.voxel = voxel
        self._voxels = set()
        self._blocks = []
        self.n_points = 0

    def update(self, points_map):
        """points_map: (N, 3) 地图系点云。返回本帧新增体素的代表点。"""
        if len(points_map) == 0:
            return np.zeros((0, 3))
        keys = np.floor(points_map / self.voxel).astype(np.int64)
        uniq, idx = np.unique(keys, axis=0, return_index=True)
        mask = np.array([tuple(k) not in self._voxels for k in uniq])
        self._voxels.update(tuple(k) for k in uniq[mask])
        new_pts = points_map[idx[mask]]
        if len(new_pts):
            self._blocks.append(new_pts.astype(np.float32))
            self.n_points += len(new_pts)
        return new_pts

    def all_points(self):
        if not self._blocks:
            return np.zeros((0, 3))
        return np.vstack(self._blocks)


def save_ply_ascii(points, path):
    """无 open3d 依赖的 PLY 导出。points: (N, 3)"""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, points, fmt="%.4f")


# ---------------------------------------------------------------------------
# 非阻塞实时可视化 (viser 浏览器窗口)
# ---------------------------------------------------------------------------

class LiveVisualizer:
    """
    非阻塞实时建图可视化 (viser, 浏览器窗口): 亮色=结构点 (墙/货架等), 暗色=地面点,
    绿线=估计轨迹, 红线=回环约束。
    浏览器内操作: 左键拖拽旋转 / 右键拖拽平移 / 滚轮缩放, 支持多视角截图等;
    关闭网页不影响建图, 结束保存用终端 Ctrl+C。
    """

    def __init__(self):
        import webbrowser
        import viser
        self.server = None
        self.port = None
        for port in (8080, 8081, 8082, 8083, 8084):
            try:
                self.server = viser.ViserServer(host="127.0.0.1", port=port,
                                                verbose=False)
                self.port = port
                break
            except OSError:
                continue
        if self.server is None:
            raise RuntimeError("no free port in 8080-8084 for viser server")
        # 地图系 z 轴为高度方向
        self.server.scene.set_up_direction("+z")
        url = f"http://127.0.0.1:{self.port}"
        print(f"[INFO] viz ready -> {url} (浏览器操作: 左键旋转 / 右键平移 / 滚轮缩放)")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        self.closed = False
        self._traj_xyz = []
        self._n_chunks = 0

    def update(self, new_map_points, pose_map, key_poses, closures):
        if self.closed:
            return
        scene = self.server.scene
        # 1. 地图增量: 每次调用一个独立点云块 (增量发送, 无需全量重传)
        #    地面点 (地图系 z < -1.0m, 雷达离地约 1.42m) 用土黄, 墙/货架等结构点保持亮蓝灰
        if len(new_map_points):
            pts = np.asarray(new_map_points)
            colors = np.tile(np.array([[0.75, 0.80, 0.90]]), (len(pts), 1))
            colors[pts[:, 2] < -1.0] = [0.55, 0.45, 0.25]
            scene.add_point_cloud(f"map/chunk_{self._n_chunks:06d}", points=pts,
                                  colors=colors, point_size=0.02,
                                  point_shape="circle", precision="float32")
            self._n_chunks += 1
        # 2. 轨迹线 (同名替换, 全量重发, 数据量小)
        xyz = pose_map[:3, 3]
        self._traj_xyz.append(xyz)
        if len(self._traj_xyz) >= 2:
            t = np.array(self._traj_xyz)
            seg = np.stack([t[:-1], t[1:]], axis=1)
            scene.add_line_segments("trajectory", points=seg,
                                    colors=(0.1, 0.6, 0.1), thickness=2.0,
                                    thickness_units="screen")
        # 3. 回环线 (同名替换)
        if closures:
            key_xyz = np.array([k[:3, 3] for k in key_poses])
            seg = np.stack([np.array([key_xyz[i] for i, _ in closures]),
                            np.array([key_xyz[j] for _, j in closures])], axis=1)
            scene.add_line_segments("closures", points=seg,
                                    colors=(1.0, 0.1, 0.1), thickness=2.0,
                                    thickness_units="screen")


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def pose_to_kitti_line(T):
    return " ".join(f"{v:.9f}" for v in np.asarray(T)[:3].reshape(-1))


def save_poses_kitti(poses, path):
    with open(path, "w") as f:
        for T in poses:
            f.write(pose_to_kitti_line(T) + "\n")


def save_poses_tum(poses, timestamps, path):
    with open(path, "w") as f:
        for T, ts in zip(poses, timestamps):
            # TUM: x y z qx qy qz qw
            R = np.asarray(T)[:3, :3]
            w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
            qx = (R[2, 1] - R[1, 2]) / (4 * w) if w > 1e-9 else 0.0
            qy = (R[0, 2] - R[2, 0]) / (4 * w) if w > 1e-9 else 0.0
            qz = (R[1, 0] - R[0, 1]) / (4 * w) if w > 1e-9 else 0.0
            t = np.asarray(T)[:3, 3]
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {w:.6f}\n")


def write_local_maps(slam, plys_dir):
    os.makedirs(plys_dir, exist_ok=True)
    for local_map in slam.local_map_graph.local_maps():
        local_map.write(os.path.join(plys_dir, "{:06d}.ply".format(local_map.id)))


def write_trajectory_plot(poses, slam, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib unavailable, skip trajectory.png")
        return
    loc = np.array([p[:3, 3] for p in poses])
    plt.figure(figsize=(8, 8))
    plt.scatter(loc[:, 0], loc[:, 1], s=0.1, color="black")
    key_poses = slam.get_keyposes()
    for i, j in slam.get_closures():
        plt.plot([key_poses[i][0, -1], key_poses[j][0, -1]],
                 [key_poses[i][1, -1], key_poses[j][1, -1]],
                 color="red", linewidth=1)
    plt.axis("equal")
    plt.title("estimated trajectory (red: loop closures)")
    plt.savefig(path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# 在线 SLAM
# ---------------------------------------------------------------------------

class OnlineSlamRunner:
    def __init__(self, stub, config, out_dir, entity, lidar_body, hz,
                 record_to=None, viz=False, gt_odom=False):
        self.stub = stub
        self.slam = KissSLAM(config)
        self.out_dir = out_dir
        self.gt_odom = gt_odom
        self._anchor = None    # 当前子地图起点的 GT 世界位姿 (GT 里程计模式)
        self._O_prev = None    # 当前子地图内上一帧的真值相对位姿
        self.entity = entity
        self.lidar_body = lidar_body
        self.interval = 1.0 / hz
        self.record_to = record_to
        self.frames_dir = os.path.join(record_to, "frames") if record_to else None
        # 全局地图累积 (无论是否可视化都进行, 结束时导出 global_map.ply)
        self.accumulator = GlobalMapAccumulator(voxel=0.1)
        # 非阻塞 open3d 实时可视化: 灰=累积地图, 绿=轨迹, 红=回环
        # 无需按键; 关闭窗口后转为无界面模式继续跑, 用终端 Ctrl+C 保存结果
        self.visualizer = LiveVisualizer() if viz else None
        self.n_non_atomic = 0

        self.gt_poses = []       # 每个处理帧的雷达 GT 位姿 (4x4)
        self.timestamps = []     # 每个处理帧的时间戳 (s)
        self.n_scans = 0
        self._warned = False

    def _warn_once(self, msg):
        if not self._warned:
            self._warned = True
            print(f"[WARN] {msg} (only logged once)")

    def _fetch_frame(self):
        """
        拉取一帧并变换到雷达局部系 (pose-bracketing 保证点云与位姿原子一致):
        先查位姿 -> 拉点云 -> 再查位姿, 若两次位姿相同则点云与位姿严格同步;
        若不同说明拉取期间仿真步进了 (计数统计), 用后查的位姿, 误差不超过一步。
        返回 (points_local(Nx3), gt_pose(4x4)) 或 None。
        """
        pose_a = query_lidar_pose(self.stub, self.lidar_body)
        if pose_a is None:
            self._warn_once("lidar pose unavailable, frame skipped")
            return None
        data = query_lidar(self.stub, self.entity)
        if data is None:
            return None
        pose_b = query_lidar_pose(self.stub, self.lidar_body)
        if pose_b is None:
            return None
        if not (np.allclose(pose_a[0], pose_b[0]) and np.allclose(pose_a[1], pose_b[1])):
            self.n_non_atomic += 1
        t, R = pose_b
        points_local = transform_to_local(data["points_world"][data["valid"]], t, R)
        gt = np.eye(4)
        gt[:3, :3] = R
        gt[:3, 3] = t
        return points_local, gt

    def _process_scan_gt(self, points_local, gt_world):
        """
        GT 里程计模式: 复刻 KissSLAM.process_scan, 但逐帧位姿用 gRPC 真值替代 ICP 输出。
        轨迹 / voxel_grid 累积图 / 关键位姿全部以真值落地; 回环检测与位姿图照常工作。
        ICP 仍然运行 (维持 adaptive threshold 等内部状态), 其位姿输出被真值覆盖。
        里程计按 KISS-SLAM 约定在每个子地图起点重置为 identity (generate_new_node
        内部会重置), 因此真值位姿按 "当前子地图起点" 锚定, 切分后重新锚定。
        """
        slam = self.slam
        if self._anchor is None:
            self._anchor = gt_world.copy()
            self._O_prev = np.eye(4)
        O = np.linalg.inv(self._anchor) @ gt_world
        deskewed, _ = slam.odometry.register_frame(points_local,
                                                   np.zeros(len(points_local)))
        slam.odometry.last_pose = O
        slam.odometry.last_delta = np.linalg.inv(self._O_prev) @ O
        self._O_prev = O
        mapping_frame = voxel_down_sample(deskewed, slam.local_map_voxel_size)
        slam.voxel_grid.integrate_frame(mapping_frame, O)
        slam.local_map_graph.last_local_map.local_trajectory.append(O)
        if np.linalg.norm(O[:3, -1]) > slam.local_map_splitting_distance:
            slam.generate_new_node()
            # generate_new_node 已把内部里程计重置为 identity, 真值锚点随之切换
            self._anchor = gt_world.copy()
            self._O_prev = np.eye(4)

    def run(self, duration=None):
        if self.frames_dir is not None:
            os.makedirs(self.frames_dir, exist_ok=True)
        print(f"[INFO] online KISS-SLAM started: entity={self.entity}, "
              f"body={self.lidar_body}, out={self.out_dir}"
              + (f", record_to={self.record_to}" if self.record_to else "")
              + (", mode=GT-ODOM" if self.gt_odom else ", mode=scan-odom"))
        t_start = time.time()
        t_next = time.perf_counter()
        try:
            while True:
                if duration is not None and time.time() - t_start >= duration:
                    print(f"[INFO] duration {duration}s reached")
                    break
                t_next += self.interval

                result = self._fetch_frame()
                if result is not None:
                    points_local, gt = result
                    try:
                        # kiss_icp 的 preprocessor 无条件调用 timestamps.ravel(),
                        # 必须传数组; deskew=false 时内容被忽略, 全 0 即可
                        if self.gt_odom:
                            self._process_scan_gt(points_local, gt)
                        else:
                            self.slam.process_scan(points_local, np.zeros(len(points_local)))
                    except Exception as e:
                        self._warn_once(f"process_scan failed: {e}")
                        continue
                    self.n_scans += 1
                    self.gt_poses.append(gt)
                    self.timestamps.append(time.time() - t_start)
                    if self.frames_dir is not None:
                        np.save(os.path.join(
                            self.frames_dir, f"frame_{self.n_scans - 1:06d}.npy"),
                            points_local.astype(np.float32))

                    if self.n_scans % 50 == 0:
                        elapsed = time.time() - t_start
                        closures = len(self.slam.get_closures())
                        odom = self.slam.odometry.last_pose[:3, 3]
                        print(f"[INFO] scan {self.n_scans} ({elapsed:.0f}s, "
                              f"{self.n_scans / elapsed:.1f} fps), "
                              f"odom(local)=[{odom[0]:.2f},{odom[1]:.2f},{odom[2]:.2f}], "
                              f"closures={closures}, "
                              f"map={self.accumulator.n_points} pts")

                    # 全局地图累积 + 实时可视化 (每 2 帧渲染一次, ~5Hz)
                    current_node = self.slam.local_map_graph.last_local_map
                    pose_map = current_node.keypose @ self.slam.odometry.last_pose
                    pts_map = points_local @ pose_map[:3, :3].T + pose_map[:3, 3]
                    new_pts = self.accumulator.update(pts_map)
                    if self.visualizer is not None and self.n_scans % 2 == 0:
                        self.visualizer.update(new_pts, pose_map,
                                               self.slam.get_keyposes(),
                                               self.slam.get_closures())

                # 按目标频率节奏控制
                sleep = t_next - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    t_next = time.perf_counter()  # 落后太多则重置节奏
        except KeyboardInterrupt:
            print("\n[INFO] interrupted by user")
        return self.finalize()

    def finalize(self):
        if self.n_scans < 2:
            print("[WARN] not enough scans, nothing to save")
            return None
        print(f"[INFO] finalizing ({self.n_scans} scans)...")
        # 结束最后一段局部地图并做 fine-grained 位姿图优化 (与 SlamPipeline 一致)
        self.slam.generate_new_node()
        self.slam.local_map_graph.erase_last_local_map()
        poses, _ = self.slam.fine_grained_optimization()
        poses = np.array(poses)

        os.makedirs(self.out_dir, exist_ok=True)
        save_poses_kitti(poses, os.path.join(self.out_dir, "poses_kitti.txt"))
        save_poses_tum(poses, self.timestamps,
                       os.path.join(self.out_dir, "poses_tum.txt"))
        write_local_maps(self.slam, os.path.join(self.out_dir, "local_maps", "plys"))
        write_trajectory_plot(poses, self.slam,
                              os.path.join(self.out_dir, "trajectory.png"))
        save_ply_ascii(self.accumulator.all_points(),
                       os.path.join(self.out_dir, "global_map.ply"))

        closures = len(self.slam.get_closures())
        print(f"[INFO] loop closures found: {closures}")
        print(f"[INFO] global map: {self.accumulator.n_points} points "
              f"(voxel 0.1m) -> global_map.ply")
        if self.n_non_atomic:
            print(f"[INFO] non-atomic fetches: {self.n_non_atomic}/{self.n_scans} "
                  f"(点云与位姿拉取间发生了仿真步进, 位姿误差不超过一步)")

        # GT 评估: 估计轨迹 (传感器系) vs 雷达 GT 位姿 (世界系), Umeyama 对齐后 ATE
        gt_poses = np.array(self.gt_poses)
        save_poses_kitti(gt_poses, os.path.join(self.out_dir, "gt_poses_kitti.txt"))
        est_xyz = poses[:, :3, 3]
        gt_xyz = gt_poses[:, :3, 3]
        ate = ate_rmse(est_xyz, gt_xyz)
        with open(os.path.join(self.out_dir, "ate.txt"), "w") as f:
            f.write(f"ate_rmse {ate:.6f}\n")
            f.write(f"n_scans {self.n_scans}\n")
            f.write(f"closures {closures}\n")
        print(f"[INFO] ATE RMSE (vs GT, aligned): {ate:.4f} m over {self.n_scans} scans")

        if self.frames_dir is not None:
            np.savez(os.path.join(self.record_to, "meta.npz"),
                     timestamps=np.array(self.timestamps),
                     gt_poses=gt_poses,
                     n_scans=self.n_scans)
            print(f"[INFO] recorded frames -> {self.record_to}")

        print(f"[INFO] outputs -> {self.out_dir}")
        return poses


def main():
    parser = argparse.ArgumentParser(description="g1 omnipicker online LiDAR SLAM (KISS-SLAM, no ROS)")
    parser.add_argument("--addr", type=str, default="localhost:50051",
                        help="gRPC server address")
    parser.add_argument("--entity", type=str, default=DEFAULT_LIDAR_ENTITY,
                        help="LiDAR entity name")
    parser.add_argument("--lidar-body", type=str, default=DEFAULT_LIDAR_BODY,
                        help="LiDAR body name, 用于查询雷达世界位姿")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "kiss_slam_g1_indoor.yaml"),
                        help="KISS-SLAM 配置文件")
    parser.add_argument("--out", type=str, default="slam_output",
                        help="输出目录")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="点云拉取频率")
    parser.add_argument("--duration", type=float, default=None,
                        help="运行秒数 (默认一直跑到 Ctrl+C)")
    parser.add_argument("--record-to", type=str, default=None,
                        help="同时把局部系原始帧录到该目录 (供离线复跑)")
    parser.add_argument("--viz", action="store_true",
                        help="开启 open3d 实时可视化 (需 pip install open3d)")
    parser.add_argument("--gt-odom", action="store_true",
                        help="GT 里程计模式: 用 gRPC 真值位姿替代扫描配准里程计 "
                             "(仿真场景开阔地面平移退化, 逐帧扫描配准无法恢复平移; "
                             "回环检测/位姿图/建图照常工作, 架构等价于 rtabmap + /odom)")
    args = parser.parse_args()

    config = load_config(args.config)
    stub = make_stub(args.addr)

    runner = OnlineSlamRunner(
        stub=stub,
        config=config,
        out_dir=args.out,
        entity=args.entity,
        lidar_body=args.lidar_body,
        hz=args.hz,
        record_to=args.record_to,
        viz=args.viz,
        gt_odom=args.gt_odom,
    )
    runner.run(duration=args.duration)


if __name__ == "__main__":
    main()
