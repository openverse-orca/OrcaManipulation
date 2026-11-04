"""
g1 omnipicker 地图重定位 — 在已建 KISS-SLAM 地图上实时定位, 无 ROS 依赖

架构 (与 rtabmap + /odom 同构):
  里程计: freejoint 位姿 (仿真真值里程计, 可 --inject-drift 注入漂移模拟真实轮里程计)
  地图匹配: 有界相关性扫描匹配 — 以里程计预测为中心的窗口内
    (2D 似然场粗筛 -> 3D 最近邻精打分) 取分最大位姿, 修正 T_map_odom。
    不用 ICP: 开阔场景远距离结构对平移不敏感, point-to-plane 存在
    平移退化平底谷, 逐帧厘米级滑动且 rmse 无变化, 误差持续累积;
    相关性匹配的真值是严格极大, 窗口半径即修正量上限。
  输出: T_map_odom 修正量 (轨迹 + 修正日志), 结束时输出 ATE (vs GT)

地图: 推荐使用 g1_map_tools.py refmap 从录制数据重建的参考地图
  (SLAM est 位姿建的地图有 10-50cm 局部畸变, 扫描匹配不可靠, 详见文档)。

用法 (lidar conda 环境, 需 仿真 gRPC server 在跑 + 已建地图):
  python g1_lidar_localizer.py --map slam_output/ref_map.ply --viz
      初始化: 默认假设机器人摆在建图起点 (地图系锚定录制/建图首帧)
  python g1_lidar_localizer.py --map ... --init 3.5,-2.0,90   # 手动指定地图系初始位姿 x,y,yaw_deg
  python g1_lidar_localizer.py --map ... --global-init --grid slam_output/ref_grid.npz
      实验性粒子滤波全局定位 (准周期货架区不可靠, 详见文档)
  python g1_lidar_localizer.py --map ... --inject-drift 0.05,0.10   # 5cm/s 平移 + 0.1deg/s 偏航
  python g1_lidar_localizer.py --map ... --duration 60             # 60 秒自动结束

输出 (--out-dir, 默认 localization_output):
  poses_tum.txt     估计轨迹 (地图系)
  gt_poses_tum.txt  GT 轨迹 (地图系)
  odom_poses_tum.txt 里程计轨迹 (漂移注入后, 供对比)
  ate.txt           est vs GT / odom vs GT 的 ATE RMSE (Umeyama 对齐)
"""

import argparse
import os
import time

import numpy as np
import open3d as o3d

from g1_lidar_kiss_slam import (
    DEFAULT_LIDAR_BODY, DEFAULT_LIDAR_ENTITY,
    ate_rmse, make_stub, query_lidar, query_lidar_pose, transform_to_local,
)


def se2(x, y, yaw):
    T = np.eye(4)
    T[:3, :3] = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])
    T[:3, 3] = [x, y, 0]
    return T


def pose_to_mat(t, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class MapMatcher:
    """点云到地图的 ICP 配准器。map: 地图系点云 (预处理成体素+法线)。"""

    def __init__(self, map_ply, voxel=0.1):
        pcd = o3d.io.read_point_cloud(map_ply)
        n0 = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel)
        # ICP 在地面上几乎无约束, 仅用结构点 (z > -1.0) 建匹配目标, 减少地面主导
        pts = np.asarray(pcd.points)
        keep = pts[:, 2] > -1.0
        self.map = o3d.geometry.PointCloud()
        self.map.points = o3d.utility.Vector3dVector(pts[keep])
        self.map.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
        self.n_map = len(self.map.points)
        print(f"[INFO] map matcher: {n0} -> structure {self.n_map} points @ {voxel}m voxel")

    def match(self, scan_local, T_init, max_corr=0.5, iters=30):
        """scan_local: 雷达局部系扫描点 (Nx3); T_init: ICP 初值 (地图系位姿)。
        返回配准后的 T (4x4, 地图系雷达位姿) 与 fitness/inlier_rmse, 或 None。"""
        scan = o3d.geometry.PointCloud()
        scan.points = o3d.utility.Vector3dVector(scan_local)
        reg = o3d.pipelines.registration.registration_icp(
            scan, self.map, max_corr, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iters, relative_fitness=1e-6, relative_rmse=1e-6))
        if reg.fitness < 0.3:
            return None, reg.fitness, reg.inlier_rmse
        return reg.transformation, reg.fitness, reg.inlier_rmse


class CorrelativeMatcher:
    """有界相关性扫描匹配器 (在线跟踪用)。

    以里程计预测为中心的窗口内搜索地图系位姿: 2D 似然场全量粗筛
    -> top 候选用 3D KDTree 最近邻距离精打分 -> 细分辨率复扫。
    2D 场会被多层结构高度压扁 (伪高分), 必须以 3D 分数定夺。
    """

    def __init__(self, map_ply, voxel=0.05, match_range=15.0, max_pts=400,
                 dmax=0.3):
        from scipy.ndimage import distance_transform_edt
        from scipy.spatial import cKDTree
        pcd = o3d.io.read_point_cloud(map_ply).voxel_down_sample(voxel)
        pts = np.asarray(pcd.points)
        struct = pts[pts[:, 2] > -1.0]
        self.tree = cKDTree(struct)
        self.match_range = match_range
        self.max_pts = max_pts
        self.dmax = dmax
        # 2D 似然场 (粗筛): 结构点 xy 栅格化 -> 距离变换
        res = 0.1
        x0, y0 = struct[:, 0].min() - 2 * res, struct[:, 1].min() - 2 * res
        nx = int(np.ceil((struct[:, 0].max() - x0) / res)) + 3
        ny = int(np.ceil((struct[:, 1].max() - y0) / res)) + 3
        occ = np.zeros((ny, nx), dtype=bool)
        occ[((struct[:, 1] - y0) / res).astype(np.int64),
            ((struct[:, 0] - x0) / res).astype(np.int64)] = True
        dist = distance_transform_edt(~occ, sampling=res)
        self.origin = np.array([x0, y0])
        self.res = res
        self.field = np.maximum(0.0, 1.0 - dist / (2 * dmax))
        print(f"[INFO] correlative matcher: {len(struct)} structure pts "
              f"@ {voxel}m voxel, field {nx}x{ny} @ {res}m")

    def initial_match(self, scan_local, occ_plan, origin, res, hint=None,
                      min_score=0.55, min_margin=0.08, progress_cb=None):
        """停车初定位：全图粗搜、3D复核及多峰拒绝；hint仅限定局部搜索。
        progress_cb(f): 0-1 计算进度回调 (可选, 供终端进度条)。"""
        scan_local = np.asarray(scan_local)
        scan_local = scan_local[np.all(np.isfinite(scan_local), axis=1)]
        b2 = _structure_band_2d(scan_local, self.match_range, self.max_pts)
        b3 = _structure_band_3d(scan_local, self.match_range, self.max_pts)
        info = {"reason": "too_few_points", "points": len(b3)}
        if len(b3) < 60:
            return None, info

        def _prog(f):
            if progress_cb is not None:
                progress_cb(min(1.0, max(0.0, f)))

        def free(T):
            gx, gy = np.floor((T[:2, 3] - origin) / res).astype(int)
            return 0 <= gy < occ_plan.shape[0] and 0 <= gx < occ_plan.shape[1] and not occ_plan[gy, gx]

        if hint is not None:
            seeds = [hint]
        else:
            # 分块评分，避免全图位姿与扫描点的笛卡尔积占满内存。
            stride = max(1, int(round(0.3 / res)))
            ys, xs = np.where(~occ_plan[::stride, ::stride].astype(bool))
            xy = np.column_stack([xs * stride, ys * stride]) * res + origin
            if not len(xy):
                return None, {"reason": "no_free_candidates"}
            candidates, scores = [], []
            yaws = np.radians(np.arange(-180, 180, 5))
            for yi, yaw in enumerate(yaws):
                for start in range(0, len(xy), 256):
                    chunk = xy[start:start + 256]
                    poses = np.column_stack([chunk, np.full(len(chunk), np.cos(yaw)), np.full(len(chunk), np.sin(yaw))])
                    scores.append(_scan_pose_score(b2, poses, self.origin, self.res, self.field))
                    candidates.append(poses)
                _prog(0.3 * (yi + 1) / len(yaws))
            poses = np.concatenate(candidates)
            order = np.argsort(np.concatenate(scores))[::-1]
            # 相邻粗格只保留一个种子，给不同位置/朝向留下复核机会。
            selected = []
            for i in order:
                p = poses[i]
                yaw = np.arctan2(p[3], p[2])
                if any(np.linalg.norm(p[:2] - q[:2, 3]) < 0.45 and
                       abs(np.arctan2(np.sin(yaw - np.arctan2(q[1, 0], q[0, 0])), np.cos(np.arctan2(q[1, 0], q[0, 0])))) < np.radians(12)
                       for q in selected):
                    continue
                selected.append(se2(p[0], p[1], yaw))
                if len(selected) >= 80:
                    break
            # 先以3D粗分排序；全局搜索仍是有限候选近似，不保证全局唯一。
            selected.sort(key=lambda T: _score3d(self.tree, b3, T), reverse=True)
            seeds = selected[:24]
            _prog(0.35)

        ranked = []
        for si, seed in enumerate(seeds):
            T = self.match(seed, scan_local)
            if T is not None and np.all(np.isfinite(T)) and free(T):
                ranked.append((_score3d(self.tree, b3, T), T))
            _prog(0.35 + 0.65 * (si + 1) / len(seeds))
        _prog(1.0)
        if not ranked:
            return None, {"reason": "no_valid_match"}
        ranked.sort(key=lambda item: item[0], reverse=True)
        score, best = ranked[0]
        yaw = np.arctan2(best[1, 0], best[0, 0])
        rivals = [s for s, T in ranked[1:] if
                  np.linalg.norm(T[:2, 3] - best[:2, 3]) > 0.5 or
                  abs(np.arctan2(np.sin(np.arctan2(T[1, 0], T[0, 0]) - yaw), np.cos(np.arctan2(T[1, 0], T[0, 0]) - yaw))) > np.radians(15)]
        margin = score - max(rivals) if rivals else None
        info = {"reason": "ok", "score": float(score), "margin": margin,
                "candidates": len(ranked), "hinted": hint is not None,
                "pose": [float(best[0, 3]), float(best[1, 3]), float(np.degrees(yaw))]}
        if score < min_score:
            info["reason"] = "low_score"
        elif hint is None and margin is not None and margin < min_margin:
            info["reason"] = "ambiguous"
        return (best if info["reason"] == "ok" else None), info

    def match(self, T_init, scan_local):
        """T_init: 里程计预测的地图系雷达位姿 (4x4); scan_local: 局部系点云 (Nx3)。
        返回匹配后的位姿 T (4x4, 地图系) 或 None (结构点不足)。"""
        b2 = _structure_band_2d(scan_local, max_range=self.match_range,
                                max_pts=self.max_pts)
        b3 = _structure_band_3d(scan_local, max_range=self.match_range,
                                max_pts=self.max_pts)
        if len(b3) < 30:
            return None
        x, y = T_init[0, 3], T_init[1, 3]
        yaw = np.arctan2(T_init[1, 0], T_init[0, 0])
        # 两轮: 粗 (±0.5m/±6°, 2D 场筛 top-40 后 3D 定夺) -> 细 (±0.1m/±1.2°)
        for win_xy, step_xy, win_yaw, step_yaw, topk in (
                (0.5, 0.1, np.radians(6), np.radians(2), 40),
                (0.1, 0.025, np.radians(1.2), np.radians(0.3), 0)):
            xs = x + np.arange(-win_xy, win_xy + 1e-9, step_xy)
            ys = y + np.arange(-win_xy, win_xy + 1e-9, step_xy)
            yaws = yaw + np.arange(-win_yaw, win_yaw + 1e-9, step_yaw)
            P = np.array([[a, b, np.cos(g), np.sin(g)]
                          for a in xs for b in ys for g in yaws])
            sc2 = _scan_pose_score(b2, P, self.origin, self.res, self.field)
            cand = P if topk == 0 else P[sc2.argsort()[::-1][:topk]]
            sc3 = np.array([_score3d(self.tree, b3, se2(q[0], q[1],
                                                        np.arctan2(q[3], q[2])))
                            for q in cand])
            k = sc3.argmax()
            x, y, yaw = cand[k, 0], cand[k, 1], np.arctan2(cand[k, 3], cand[k, 2])
        return se2(x, y, yaw)


class LocalizerViz:
    """viser 可视化: 地图一次加载 + 扫描/轨迹/位姿实时更新"""

    def __init__(self, map_pts):
        import viser
        self.server = None
        for port in (8085, 8086, 8087):
            try:
                self.server = viser.ViserServer(host="127.0.0.1", port=port,
                                                verbose=False)
                break
            except OSError:
                continue
        self.server.scene.set_up_direction("+z")
        colors = np.tile(np.array([[0.75, 0.80, 0.90]]), (len(map_pts), 1))
        colors[map_pts[:, 2] < -1.0] = [0.55, 0.45, 0.25]
        self.server.scene.add_point_cloud("map", points=map_pts, colors=colors,
                                          point_size=0.03, point_shape="circle",
                                          precision="float32")
        print(f"[INFO] viz ready -> http://127.0.0.1:8085")
        self._est_xyz = []
        self._odom_xyz = []

    def update(self, scan_map, T_est, T_odom):
        scene = self.server.scene
        if len(scan_map):
            scene.add_point_cloud("scan", points=np.asarray(scan_map),
                                  colors=(0.0, 0.9, 0.3), point_size=0.05,
                                  precision="float32")
        # 机器人当前位置: 估计位姿处画朝向箭头 (红色, 区别于绿色扫描/轨迹)
        x, y, z = T_est[:3, 3]
        yaw = np.arctan2(T_est[1, 0], T_est[0, 0])
        arrow = np.array([[x, y, z],
                          [x + 0.6 * np.cos(yaw), y + 0.6 * np.sin(yaw), z]])
        scene.add_line_segments("robot", points=arrow[None], colors=(1.0, 0.2, 0.1),
                                thickness=4.0, thickness_units="screen")
        self._est_xyz.append(T_est[:3, 3])
        self._odom_xyz.append(T_odom[:3, 3])
        if len(self._est_xyz) >= 2:
            e = np.array(self._est_xyz)
            od = np.array(self._odom_xyz)
            seg_e = np.stack([e[:-1], e[1:]], axis=1)
            seg_o = np.stack([od[:-1], od[1:]], axis=1)
            scene.add_line_segments("est_traj", points=seg_e, colors=(0.1, 0.6, 0.1),
                                    thickness=2.0, thickness_units="screen")
            scene.add_line_segments("odom_traj", points=seg_o, colors=(0.9, 0.5, 0.1),
                                    thickness=2.0, thickness_units="screen")


def _load_field(grid_npz, dmax=0.3):
    d = np.load(grid_npz)
    occ, origin, res = d["occ"], d["origin"], float(d["res"])
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~occ, sampling=res)
    return occ, origin, res, np.maximum(0.0, 1.0 - dist / dmax)


def _scan_pose_score(pts2, poses, origin, res, field):
    """poses: (M,4) [x,y,cos,sin]; 返回每个位姿的平均似然 (M,)。
    出界点计 0 分 (分母为全部点), 防止靠丢点拿虚高分。"""
    ny, nx = field.shape
    c, s = poses[:, 2], poses[:, 3]
    M = len(poses)
    gx = (poses[:, 0][:, None] + c[:, None] * pts2[:, 0][None, :] - s[:, None] * pts2[:, 1][None, :] - origin[0]) / res
    gy = (poses[:, 1][:, None] + s[:, None] * pts2[:, 0][None, :] + c[:, None] * pts2[:, 1][None, :] - origin[1]) / res
    gi = gx.astype(np.int64); gj = gy.astype(np.int64)
    ok = (gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny)
    score = np.zeros(M)
    np.add.at(score, np.nonzero(ok)[0], field[gj[ok], gi[ok]])
    return score / len(pts2)


def _structure_band_2d(scan_local, max_range=10.0, max_pts=400):
    """取结构高度带 (z > -0.9) 且距离在地图覆盖范围内的点, 降采样返回 (N,2)"""
    m = scan_local[:, 2] > -0.9
    r = np.linalg.norm(scan_local[:, :2], axis=1)
    pts = scan_local[m & (r < max_range)][:, :2]
    if len(pts) > max_pts:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_pts, replace=False)]
    return pts


def _score3d(tree, pts3, T, dmax=0.3):
    """3D 似然: 扫描点经 T 投影后到地图结构点的最近邻距离打分。
    2D 打分在多层货架仓库会被高度压扁误导 (不同高度的结构投影重合),
    3D 距离不会。T: 2D 位姿 se2 (平面运动), pts3 保持 3D。"""
    p = pts3 @ T[:3, :3].T + T[:3, 3]
    d, _ = tree.query(p, k=1)
    return float(np.maximum(0.0, 1.0 - d / dmax).mean())


def _structure_band_3d(scan_local, max_range=10.0, max_pts=400):
    """结构高度带 + 距离截止的 3D 扫描点, 降采样"""
    m = scan_local[:, 2] > -0.9
    r = np.linalg.norm(scan_local[:, :2], axis=1)
    pts = scan_local[m & (r < max_range)]
    if len(pts) > max_pts:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_pts, replace=False)]
    return pts


def _candidates_2d(scan_local, grid_npz, map_tree, n_top=400, n_final=120):
    """首帧候选生成: 2D 似然场粗搜 top-K (广撒网, 高度压扁会引入大量伪候选),
    再用 3D 最近邻似然精筛到 n_final。返回 (M,4) [x,y,cos,sin] 及 3D 分数。"""
    occ, origin, res, field = _load_field(grid_npz)
    pts2 = _structure_band_2d(scan_local)
    if len(pts2) < 30:
        return None, None
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~occ, sampling=res)
    ys, xs = np.where(dist > 0.3)
    ys, xs = ys[::3], xs[::3]
    cand = np.column_stack([xs * res + origin[0], ys * res + origin[1]])
    out, scores = [], []
    for yaw in np.arange(0, 360, 10):
        poses = np.column_stack([cand, np.full(len(cand), np.cos(np.radians(yaw))),
                                 np.full(len(cand), np.sin(np.radians(yaw)))])
        sc = _scan_pose_score(pts2, poses, origin, res, field)
        out.append(poses)
        scores.append(sc)
    poses = np.concatenate(out)
    poses = poses[np.concatenate(scores).argsort()[::-1][:n_top]]
    # 3D 精筛
    band3 = _structure_band_3d(scan_local)
    scored = []
    for p in poses:
        T = se2(p[0], p[1], np.arctan2(p[3], p[2]))
        scored.append(_score3d(map_tree, band3, T))
    scored = np.array(scored)
    k = scored.argsort()[::-1][:n_final]
    return poses[k], scored[k]


def global_init_pf(get_frame_rel, grid_npz, map_tree, n_frames=16, n_keep=60,
                   verbose=True):
    """粒子滤波全局初始定位 (多帧运动消歧, 解决准周期货架区的单帧歧义)。

    get_frame_rel: 迭代器, 每次 yield (scan_local, T_rel):
        scan_local: 当前帧雷达局部系点云 (Nx3)
        T_rel: 上一帧->当前帧的相对运动 (4x4, 局部系: T_p' = T_p @ T_rel)
    map_tree: 地图结构点 cKDTree (3D 重打分用)
    返回收敛后的最优位姿 T (4x4 地图系雷达位姿) 或 None。"""
    first = next(get_frame_rel, None)
    if first is None:
        return None
    scan0, _ = first
    parts, sc = _candidates_2d(scan0, grid_npz, map_tree)
    if parts is None:
        print("[WARN] global init: too few structure points")
        return None
    weights = np.maximum(sc, 1e-3)
    if verbose:
        print(f"[INFO] global init: {len(parts)} candidates, "
              f"top 3D score {sc[0]:.3f}")

    best_T = None
    for i in range(n_frames):
        item = next(get_frame_rel, None)
        if item is None:
            break
        scan, T_rel = item
        # 1) 传播 (平面运动)
        c, s = parts[:, 2], parts[:, 3]
        c2, s2 = c * T_rel[0, 0] - s * T_rel[1, 0], s * T_rel[0, 0] + c * T_rel[1, 0]
        parts = np.column_stack([
            parts[:, 0] + c * T_rel[0, 3] - s * T_rel[1, 3],
            parts[:, 1] + s * T_rel[0, 3] + c * T_rel[1, 3],
            c2, s2])
        # 2) 3D 重打分 (高度不可压扁, 多层货架不会产生伪高分)
        band3 = _structure_band_3d(scan)
        sc_new = np.array([
            _score3d(map_tree, band3, se2(p[0], p[1], np.arctan2(p[3], p[2])))
            for p in parts])
        # 3) 累积权重 + 重采样
        weights = weights * np.maximum(sc_new, 1e-3)
        k = weights.argsort()[::-1][:n_keep]
        parts, weights = parts[k], weights[k] / weights[k].sum()
        if verbose:
            xy = parts[:, :2]
            spread = np.linalg.norm(xy - xy[0], axis=1)
            print(f"  frame {i+1}: top w {weights[0]:.2f}, "
                  f"cluster spread p90 {np.percentile(spread, 90):.2f}m")
        best_T = parts[0]

    if best_T is None:
        return None
    yaw = np.arctan2(best_T[3], best_T[2])
    return se2(best_T[0], best_T[1], yaw)


def run(args):
    stub = make_stub(args.addr)
    cmatcher = CorrelativeMatcher(args.map, voxel=args.voxel)
    matcher = MapMatcher(args.map, voxel=args.voxel) if args.global_init else None
    os.makedirs(args.out_dir, exist_ok=True)

    drift_v, drift_yaw = 0.0, 0.0
    if args.inject_drift:
        drift_v, drift_yaw = [float(v) for v in args.inject_drift.split(",")]
        print(f"[INFO] drift injected: {drift_v} m/s, {drift_yaw} deg/s")

    viz = LocalizerViz(np.asarray(o3d.io.read_point_cloud(args.map).points)) \
        if args.viz else None

    T_map_odom = np.eye(4)  # map <- odom 修正量
    D = np.eye(4)           # 注入的里程计漂移 (odom = gt @ D)
    est_list, gt_list, odom_list = [], [], []
    t0 = time.time()
    last_t = t0
    n_icp = n_fail = 0

    # ---- 初始位姿 ----
    # 地图系锚定在 SLAM 建图首帧 (非世界系)。约定: 建图时的起点是停车位,
    # 新会话机器人也摆在同一起点 -> T_map_odom = inv(当前真值位姿)。
    # (--global-init 提供实验性粒子滤波全局定位, 但准周期货架 + 地图细尺度畸变
    #  下单帧打分存在本质歧义, 不可靠, 详见文档)
    if args.init_at_start or (not args.global_init and not args.init):
        pose0 = query_lidar_pose(stub, args.lidar_body)
        if pose0 is not None:
            T_map_odom = np.linalg.inv(pose_to_mat(*pose0))
            print(f"[INFO] init at map start: world pose "
                  f"({pose0[0][0]:.2f}, {pose0[0][1]:.2f}), T_map_odom set")
    if args.init:
        x, y, yaw = [float(v) for v in args.init.split(",")]
        pose0 = query_lidar_pose(stub, args.lidar_body)
        if pose0 is not None:
            T_map_odom = se2(x, y, np.radians(yaw)) @ np.linalg.inv(
                pose_to_mat(*pose0))
            print(f"[INFO] init from manual pose ({x}, {y}, {yaw}deg)")

    if args.global_init:
        last_odom = [None]  # PF 最后一帧的 GT 位姿 (T_init2d 对应时刻)

        def _init_frames():
            prev_T, n = None, 0
            while n < 18:
                time.sleep(max(0.05, args.rate_interval))
                data = query_lidar(stub, args.entity)
                pose = query_lidar_pose(stub, args.lidar_body)
                if data is None or pose is None:
                    continue
                t_g, R_g = pose
                pts = transform_to_local(data["points_world"][data["valid"]], t_g, R_g)
                T_now = pose_to_mat(t_g, R_g)
                T_rel = np.eye(4) if prev_T is None else \
                    np.linalg.inv(prev_T) @ T_now
                prev_T, n = T_now, n + 1
                last_odom[0] = T_now
                yield pts, T_rel

        print("[INFO] running global init (drive around to disambiguate)...")
        from scipy.spatial import cKDTree
        map_tree = cKDTree(np.asarray(matcher.map.points))
        T_init2d = global_init_pf(_init_frames(), args.grid, map_tree)
        if T_init2d is not None:
            pose0 = query_lidar_pose(stub, args.lidar_body)
            if pose0 is not None:
                t_g0, R_g0 = pose0
                T_odom0 = pose_to_mat(t_g0, R_g0)
                # PF 位姿对应其最后一帧, 先建立 map<-odom 再外推到当前帧作初值
                T_map_odom = T_init2d @ np.linalg.inv(last_odom[0])
                T_guess = T_map_odom @ T_odom0
                data0 = query_lidar(stub, args.entity)
                if data0 is not None:
                    pts0 = transform_to_local(
                        data0["points_world"][data0["valid"]], t_g0, R_g0)
                    band0 = pts0[pts0[:, 2] > -0.9]
                    T_est0, fitness, rmse = matcher.match(band0, T_guess,
                                                         max_corr=1.0, iters=50)
                    if T_est0 is not None:
                        T_map_odom = T_est0 @ np.linalg.inv(T_odom0)
                        print(f"[INFO] initial localization refined: fitness "
                              f"{fitness:.2f} rmse {rmse:.3f}")
                    else:
                        print("[WARN] initial ICP refine failed, keep PF pose")
        else:
            print("[WARN] global init did not converge (robot outside mapped "
                  "area or no motion?), using identity map<-odom")

    print("[INFO] running, Ctrl+C to stop...")
    try:
        while True:
            if args.duration and time.time() - t0 > args.duration:
                break
            time.sleep(max(0.0, args.rate_interval - (time.time() - last_t)))
            last_t = time.time()

            data = query_lidar(stub, args.entity)
            pose = query_lidar_pose(stub, args.lidar_body)
            if data is None or pose is None:
                continue
            t_gt, R_gt = pose
            T_gt = pose_to_mat(t_gt, R_gt)

            # 1. 里程计: GT + 注入漂移
            dt = min(1.0, time.time() - (est_list[-1][0] if est_list else t0))
            if drift_v or drift_yaw:
                ang = np.radians(drift_yaw) * dt
                D = D @ se2(drift_v * dt, 0.0, ang)
            T_odom = T_gt @ D

            # 2. 扫描 -> 局部系, 以 T_map_odom @ T_odom 为窗口中心做有界相关性匹配
            pts_local = transform_to_local(
                data["points_world"][data["valid"]], t_gt, R_gt)
            T_init = T_map_odom @ T_odom
            T_est = cmatcher.match(T_init, pts_local)
            if T_est is None:
                n_fail += 1
                print(f"[WARN] too few structure points, keep odometry "
                      f"prediction ({n_fail} fails)")
                T_est = T_init
            else:
                n_icp += 1
                # 3. 更新修正量 T_map_odom (窗口本身有界, 跳变拒绝仅为保险)
                T_new = T_est @ np.linalg.inv(T_odom)
                if np.all(np.isfinite(T_new)):
                    jump = np.linalg.norm(T_new[:3, 3] - T_map_odom[:3, 3])
                    if jump < args.max_correction_jump:
                        T_map_odom = T_new
                    else:
                        print(f"[WARN] correction jump {jump:.2f}m rejected")

            now = time.time() - t0
            est_list.append((now, T_est))
            gt_list.append((now, T_gt))
            odom_list.append((now, T_odom))
            if viz:
                scan_map = pts_local @ T_est[:3, :3].T + T_est[:3, 3]
                viz.update(scan_map[::3], T_est, T_odom)
    except KeyboardInterrupt:
        print("\n[INFO] interrupted")

    # ---- 保存与评估 ----
    def dump(path, items):
        with open(path, "w") as f:
            for ts, T in items:
                q = np.empty(4)
                from scipy.spatial.transform import Rotation as _Rot
                q[:] = _Rot.from_matrix(T[:3, :3]).as_quat()  # x y z w
                t = T[:3, 3]
                f.write(f"{ts:.3f} {t[0]:.4f} {t[1]:.4f} {t[2]:.4f} "
                        f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    dump(f"{args.out_dir}/poses_tum.txt", est_list)
    dump(f"{args.out_dir}/gt_poses_tum.txt", gt_list)
    dump(f"{args.out_dir}/odom_poses_tum.txt", odom_list)
    est_xyz = np.array([T[:3, 3] for _, T in est_list])
    gt_xyz = np.array([T[:3, 3] for _, T in gt_list])
    odom_xyz = np.array([T[:3, 3] for _, T in odom_list])
    ate_est = ate_rmse(est_xyz, gt_xyz)
    ate_odom = ate_rmse(odom_xyz, gt_xyz)
    with open(f"{args.out_dir}/ate.txt", "w") as f:
        f.write(f"ate_est_vs_gt {ate_est:.4f}\nate_odom_vs_gt {ate_odom:.4f}\n"
                f"n_frames {len(est_list)}\nicp_ok {n_icp} icp_fail {n_fail}\n")
    print(f"[INFO] est  ATE RMSE (vs GT, aligned): {ate_est:.4f} m")
    print(f"[INFO] odom ATE RMSE (vs GT, aligned): {ate_odom:.4f} m "
          f"{'(漂移注入下)' if (drift_v or drift_yaw) else ''}")
    print(f"[INFO] frames {len(est_list)}, ICP ok {n_icp} / fail {n_fail}")
    print(f"[INFO] saved to {args.out_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", required=True, help="已建地图 PLY (g1_map_tools merge 输出)")
    parser.add_argument("--grid", default=None,
                        help="栅格地图 npz (g1_map_tools grid 输出), --global-init 全局定位用")
    parser.add_argument("--init-at-start", action="store_true",
                        help="显式声明按建图起点初始化 (无 --init/--global-init 时本就是默认行为)")
    parser.add_argument("--init", default=None,
                        help="手动指定初始位姿 'x,y,yaw_deg' (地图系, 雷达在地图中的位姿)")
    parser.add_argument("--global-init", action="store_true",
                        help="实验性: 粒子滤波全局初始定位 (需 --grid, 准周期货架区不可靠)")
    parser.add_argument("--addr", default="localhost:50051")
    parser.add_argument("--entity", default=DEFAULT_LIDAR_ENTITY)
    parser.add_argument("--lidar-body", default=DEFAULT_LIDAR_BODY)
    parser.add_argument("--viz", action="store_true", help="viser 浏览器可视化")
    parser.add_argument("--inject-drift", default=None,
                        help="注入里程计漂移 '平移m/s,偏航deg/s' (验证地图匹配修正)")
    parser.add_argument("--rate-interval", type=float, default=0.2,
                         help="匹配周期 s (default 0.2 = 5Hz)")
    parser.add_argument("--max-correction-jump", type=float, default=0.5,
                         help="T_map_odom 单次允许修正量上限 m (default 0.5)")
    parser.add_argument("--voxel", type=float, default=0.05,
                        help="地图匹配体素 m (default 0.05)")
    parser.add_argument("--duration", type=float, default=None, help="运行秒数")
    parser.add_argument("--out-dir", default="localization_output")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
