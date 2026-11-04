"""
SLAM 地图后处理工具 — 合并局部地图 / 生成 2D 栅格占据地图

子命令:
  merge  拼合 local_maps/plys/*.ply (各局部地图世界系已对齐, 合并即全局地图),
         可选体素去重, 输出单个 PLY。用于修复被中断的 global_map.ply。
  refmap 从录制数据 (frames/*.npy + meta.npz, --record-to 的产物) + GT 位姿
         重建几何一致的参考地图 (锚定录制首帧), 含自体回波清洗。
         SLAM est 位姿建的地图在货架区有 10-50cm 局部畸变 (平移退化残留),
         使扫描匹配定位不可靠; 本命令用真值位姿得到无畸变参考地图。
  grid   PLY -> 2D occupancy grid:
           - 过滤地面点 (z < z_min 视为地面, 不算障碍)
           - 按分辨率栅格化占据
           - 按机器人半径膨胀 (供 A* 规划安全使用)
         输出 pgm (ROS map_server 兼容) + yaml 元数据 + npz (规划器直接加载) + png 预览

用法 (lidar conda 环境):
  python g1_map_tools.py merge --ply-dir slam_output/local_maps/plys \
                              --out slam_output/global_map.ply [--voxel 0.05]
  python g1_map_tools.py refmap --frames dataset/slam_diag \
                              --out slam_output/ref_map.ply [--voxel 0.05]
  python g1_map_tools.py grid --map slam_output/ref_map.ply \
                              --out-prefix slam_output/grid [--res 0.1] [--robot-radius 0.35]
"""

import argparse
import os

import numpy as np
import open3d as o3d


def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise RuntimeError(f"empty point cloud: {path}")
    return pts


def cmd_merge(args):
    ply_files = sorted(
        os.path.join(args.ply_dir, f) for f in os.listdir(args.ply_dir)
        if f.endswith(".ply"))
    if not ply_files:
        raise RuntimeError(f"no .ply files in {args.ply_dir}")
    pts = np.concatenate([load_ply(f) for f in ply_files], axis=0)
    print(f"[INFO] merged {len(ply_files)} local maps, {len(pts)} points total")

    # 自体回波清洗: 遥操作时机械臂/桅杆会被 LiDAR 打到, 在世界系地图上抹成
    # 紧贴轨迹的噪声带 (局部系全部落在 +-0.6m 车体包络内)。
    # 机器人无碰撞行驶 -> 距离轨迹过近且高于地面的点必为自体回波, 剔除。
    if args.traj and args.clean_radius > 0:
        from scipy.spatial import cKDTree
        traj = np.loadtxt(args.traj)  # poses_tum.txt: ts x y z ...
        tree = cKDTree(traj[:, 1:3])
        dist, _ = tree.query(pts[:, :2], k=1)
        self_mask = (dist < args.clean_radius) & (pts[:, 2] > args.clean_z_min)
        print(f"[INFO] removed {self_mask.sum()} self-return points "
              f"(dist<{args.clean_radius}m of trajectory, z>{args.clean_z_min})")
        pts = pts[~self_mask]

    if args.voxel > 0:
        # 体素去重 (与建图全局地图一致的思路, 防止重叠区域点密度过高)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(args.voxel)
        pts = np.asarray(pcd.points)
        print(f"[INFO] after voxel {args.voxel}m dedup: {len(pts)} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if not o3d.io.write_point_cloud(args.out, pcd, write_ascii=False):
        raise RuntimeError(f"failed to write {args.out}")
    z = pts[:, 2]
    print(f"[INFO] wrote {args.out}: {len(pts)} points, "
          f"z range [{z.min():.2f}, {z.max():.2f}], "
          f"xy range x[{pts[:,0].min():.1f},{pts[:,0].max():.1f}] "
          f"y[{pts[:,1].min():.1f},{pts[:,1].max():.1f}]")


def cmd_refmap(args):
    """从录制数据 + GT 位姿重建参考地图 (锚定录制首帧, 即地图系原点)。"""
    meta_path = os.path.join(args.frames, "meta.npz")
    if not os.path.exists(meta_path):
        raise RuntimeError(f"meta.npz not found in {args.frames} "
                           f"(需要 --record-to 录制的 frames/*.npy + meta.npz)")
    meta = np.load(meta_path)
    gt = meta["gt_poses"]  # (N, 4, 4) 世界系 GT 位姿
    n = len(gt)
    G0_inv = np.linalg.inv(gt[0])  # 地图系锚定录制首帧

    # 地图系轨迹 (自体回波清洗用)
    from scipy.spatial import cKDTree
    R2, t2 = G0_inv[:2, :2], G0_inv[:2, 3]
    traj_xy = gt[:, :2, 3] @ R2.T + t2
    ttree = cKDTree(traj_xy)

    blocks, n_self = [], 0
    for i in range(0, n, args.every):
        p = np.load(os.path.join(args.frames, "frames", f"frame_{i:06d}.npy"))
        if args.max_range > 0:
            # 与 KISS-SLAM 配置一致的距离截止: 掠射地面远回波 (可达 90m) 对建图无用
            p = p[np.linalg.norm(p[:, :2], axis=1) <= args.max_range]
        T = G0_inv @ gt[i]
        pw = (p @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        if args.clean_radius > 0:
            # 与 merge 同规则: 距轨迹 <radius 且高于地面的点为机械臂/桅杆自体回波
            dist, _ = ttree.query(pw[:, :2], k=1)
            self_mask = (dist < args.clean_radius) & (pw[:, 2] > args.clean_z_min)
            n_self += int(self_mask.sum())
            pw = pw[~self_mask]
        blocks.append(pw)
    pts = np.concatenate(blocks)
    print(f"[INFO] loaded {len(range(0, n, args.every))} frames, {len(pts)} points, "
          f"removed {n_self} self-return points")

    if args.voxel > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(args.voxel)
        pts = np.asarray(pcd.points)
        print(f"[INFO] after voxel {args.voxel}m dedup: {len(pts)} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if not o3d.io.write_point_cloud(args.out, pcd, write_ascii=False):
        raise RuntimeError(f"failed to write {args.out}")
    z = pts[:, 2]
    print(f"[INFO] wrote {args.out}: {len(pts)} points, "
          f"z range [{z.min():.2f}, {z.max():.2f}], "
          f"xy range x[{pts[:,0].min():.1f},{pts[:,0].max():.1f}] "
          f"y[{pts[:,1].min():.1f},{pts[:,1].max():.1f}]")


def cmd_grid(args):
    pts = load_ply(args.map)
    print(f"[INFO] loaded {len(pts)} points from {args.map}")

    # 1. 高度带过滤: 只保留障碍高度带内的结构点 (地面点 z < z_min 不算障碍)
    band = pts[(pts[:, 2] >= args.z_min) & (pts[:, 2] <= args.z_max)]
    print(f"[INFO] structure band z in [{args.z_min}, {args.z_max}]: "
          f"{len(band)} points ({100.0 * len(band) / len(pts):.1f}%)")

    # 2. 栅格化: 计算原点/尺寸, 索引落格
    res = args.res
    x0, y0 = band[:, 0].min() - 2 * res, band[:, 1].min() - 2 * res
    nx = int(np.ceil((band[:, 0].max() - x0) / res)) + 3
    ny = int(np.ceil((band[:, 1].max() - y0) / res)) + 3
    ix = ((band[:, 0] - x0) / res).astype(np.int64)
    iy = ((band[:, 1] - y0) / res).astype(np.int64)
    occ = np.zeros((ny, nx), dtype=bool)  # 行=y, 列=x
    occ[iy, ix] = True
    n_occ = occ.sum()
    print(f"[INFO] grid {nx}x{ny} @ {res}m, occupied cells: {n_occ}")

    # 3. 机器人半径膨胀 (供规划用, 得到安全距离)
    from scipy.ndimage import distance_transform_edt
    if args.robot_radius > 0:
        dist = distance_transform_edt(~occ, sampling=res)
        occ_plan = dist <= args.robot_radius
    else:
        occ_plan = occ
    print(f"[INFO] after inflation r={args.robot_radius}m: {occ_plan.sum()} cells")

    # 4. 输出 npz (规划器加载: occ 原始, occ_plan 膨胀, origin, res)
    prefix = args.out_prefix
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    np.savez_compressed(
        prefix + ".npz",
        occ=occ, occ_plan=occ_plan,
        origin=np.array([x0, y0]), res=res)
    print(f"[INFO] wrote {prefix}.npz")

    # 5. 输出 pgm + yaml (ROS map_server 兼容: 0=占据, 254=free)
    img = np.full(occ.shape, 254, dtype=np.uint8)
    img[occ_plan] = 0
    with open(prefix + ".pgm", "wb") as f:
        f.write(f"P5\n{nx} {ny}\n255\n".encode())
        f.write(img.tobytes())
    with open(prefix + ".yaml", "w") as f:
        f.write(f"image: {os.path.basename(prefix)}.pgm\n"
                f"resolution: {res}\n"
                f"origin: [{x0}, {y0}, 0.0]\n"
                f"negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    print(f"[INFO] wrote {prefix}.pgm / {prefix}.yaml")

    # 6. PNG 预览 (黑=障碍, 深灰=膨胀带, 白=free)
    try:
        from PIL import Image
        vis = np.full(occ.shape, 255, dtype=np.uint8)
        vis[occ_plan] = 100
        vis[occ] = 0
        # pgm 行 0 对应 y 最小, 翻转让图像 y 轴朝上
        Image.fromarray(vis[::-1]).save(prefix + ".png")
        print(f"[INFO] wrote {prefix}.png")
    except ImportError:
        print("[WARN] PIL not available, skip png preview")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("merge", help="拼合局部地图 PLY 为单个全局地图")
    p.add_argument("--ply-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--voxel", type=float, default=0.05,
                   help="体素去重分辨率, 0 关闭 (default 0.05)")
    p.add_argument("--traj", default=None,
                   help="轨迹文件 poses_tum.txt, 提供后启用自体回波清洗")
    p.add_argument("--clean-radius", type=float, default=0.3,
                   help="自体回波清洗半径 m (default 0.3, 0 关闭)")
    p.add_argument("--clean-z-min", type=float, default=-1.0,
                   help="自体回波清洗只作用于高于此高度 (地图系 z) 的点 (default -1.0)")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("refmap", help="从录制数据 + GT 位姿重建参考地图 PLY")
    p.add_argument("--frames", required=True,
                   help="录制数据目录 (含 frames/*.npy + meta.npz)")
    p.add_argument("--out", required=True)
    p.add_argument("--voxel", type=float, default=0.05,
                   help="体素去重分辨率, 0 关闭 (default 0.05)")
    p.add_argument("--every", type=int, default=1,
                   help="帧采样间隔 (default 1 = 全部帧, 加速可增大)")
    p.add_argument("--max-range", type=float, default=30.0,
                   help="距离截止 m (default 30, 0 关闭; 与 KISS-SLAM 配置一致)")
    p.add_argument("--clean-radius", type=float, default=0.3,
                   help="自体回波清洗半径 m (default 0.3, 0 关闭)")
    p.add_argument("--clean-z-min", type=float, default=-1.0,
                   help="自体回波清洗只作用于高于此高度 (地图系 z) 的点 (default -1.0)")
    p.set_defaults(func=cmd_refmap)

    p = sub.add_parser("grid", help="PLY -> 2D occupancy grid")
    p.add_argument("--map", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--res", type=float, default=0.1, help="栅格分辨率 m (default 0.1)")
    p.add_argument("--robot-radius", type=float, default=0.2,
                   help="膨胀半径 m (default 0.2, 仓库过道较窄, 过大会封死可通行路径)")
    p.add_argument("--z-min", type=float, default=-1.0,
                   help="障碍高度带下界 (地图系 z, 默认 -1.0, 之下视为地面)")
    p.add_argument("--z-max", type=float, default=1.0,
                   help="障碍高度带上界 (地图系 z, default 1.0)")
    p.set_defaults(func=cmd_grid)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
