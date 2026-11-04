"""
g1 omnipicker 自主导航 — 栅格地图 A* 全局规划 + 纯跟踪 + Ackermann 驱动下发

进程内架构 (与 tele 相同): 本脚本自带 MuJoCo 仿真 + gRPC server,
导航逻辑在 AutopilotDevice.update() 中执行 (device -> controller -> data.ctrl),
驱动通过 ControllerSteeringDrive DATA 模式下发 [v_fl, v_fr, δ_l, δ_r]。

链路:
  定位: 启动时停车扫描匹配地图，第二帧在首帧结果邻域复扫，一致后建立 map<-odom；
        GT 位姿仅提供里程计，不将当前位置默认当作地图原点。
        --use-matcher 时行驶中继续启用有界相关性匹配修正 (~5Hz)。
  规划: --grid 的 occ_plan (膨胀栅格) 上 A* (8 邻域, 禁止切角) + 视线捷径化。
  跟踪: 纯跟踪 (lookahead) -> 曲率 -> Ackermann 转角 -> 轮速 (v / 轮半径)。

用法 (lidar conda 环境, 单终端):
  python g1_autopilot.py --goal 5.0,8.0                       # 地图系目标 (米)
  python g1_autopilot.py --goal 5.0,8.0 --viz                 # viser 显示路径
  python g1_autopilot.py --goal 5.0,8.0 --use-matcher \
      --map slam_output/ref_map.ply                          # 开启在线地图匹配修正

约定: --map 与 --grid 必须来自同一地图；启动匹配不可信时停车，不回退原点。
      重复货架导致歧义时可用 --init x,y,yaw_deg 提供近似位置缩小搜索范围。
      目标点坐标看 slam_output/ref_grid.png (图片 y 轴向上, 与地图系一致)。
"""

import argparse
import heapq
import os
import sys
import time
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from yaml import load, Loader

from orca_gym.log.orca_log import get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask
from devices.abstract_device import AbstractDevice
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers.controllers import (
    create_arm_osc_controller,
    create_gripper_2f85_controller,
    create_steering_drive_controller,
)
from controllers.controller_2f85 import Controller2F85
from controllers.controller_wheel_drive import ControllerSteeringDrive
from controllers.controller_task import TaskStatusController
from conf import g1_omnipicker_conf

from g1_lidar_kiss_slam import (
    DEFAULT_LIDAR_BODY, DEFAULT_LIDAR_ENTITY,
    make_stub, query_lidar, query_lidar_pose, transform_to_local,
)
from g1_lidar_localizer import CorrelativeMatcher, pose_to_mat

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "g1_autopilot.log"

orca_logger = get_orca_logger(
    name="Autopilot",
    log_file=log_file,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

WHEEL_RADIUS = 0.03  # m, 模型轮 geom 半径 (cylinder size 0.03 0.025)
# 雷达安装点到后轴中点的纵向杠杆 (雷达 x=+0.118, 后轴 x=-0.21, 见机器人 XML);
# 纯跟踪/弧线检查/偏离判定必须用后轴参考点, 用雷达位姿会持续欠转向、拐弯外甩。
LIDAR_TO_REAR_AXLE = 0.328


# ---------------------------------------------------------------------------
# A* 全局规划 (膨胀占据栅格上, 8 邻域, 禁止贴角穿越)
# ---------------------------------------------------------------------------

def load_grid(npz_path, robot_radius=0.0):
    d = np.load(npz_path)
    occ_plan, origin, res = d["occ_plan"], d["origin"], float(d["res"])
    # 运行时用更小半径从原始 occ 重新膨胀 (免重新生成地图); npz 无原始 occ 时沿用烘焙值
    if robot_radius > 0 and "occ" in d:
        from scipy.ndimage import distance_transform_edt
        occ_plan = distance_transform_edt(~d["occ"], sampling=res) <= robot_radius
        print(f"[INFO] re-inflated occ_plan with r={robot_radius}m (from raw occ)")
    occ_raw = d["occ"] if "occ" in d else occ_plan
    return occ_plan, occ_raw, origin, res


def nearest_free_cell(occ_plan, cell):
    """cell 被占据时螺旋外扩找最近自由格; 全图无自由格返回 None。"""
    if not occ_plan[cell[0], cell[1]]:
        return cell
    ny, nx = occ_plan.shape
    for r in range(1, max(ny, nx)):
        for iy in range(cell[0] - r, cell[0] + r + 1):
            for ix in (cell[1] - r, cell[1] + r):
                if 0 <= iy < ny and 0 <= ix < nx and not occ_plan[iy, ix]:
                    return (iy, ix)
        for ix in range(cell[1] - r, cell[1] + r + 1):
            for iy in (cell[0] - r, cell[0] + r):
                if 0 <= iy < ny and 0 <= ix < nx and not occ_plan[iy, ix]:
                    return (iy, ix)
    return None


def astar(occ_plan, start, goal):
    """start/goal: (iy, ix)。返回路径格列表 (含首尾) 或 None。"""
    ny, nx = occ_plan.shape
    if occ_plan[start] or occ_plan[goal]:
        return None
    h = lambda c: (abs(c[0] - goal[0]) + abs(c[1] - goal[1])) * 0.1
    open_heap = [(h(start), 0.0, start)]
    g_cost = {start: 0.0}
    came = {}
    diag = np.sqrt(2)
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if g > g_cost.get(cur, np.inf):
            continue
        for dy, dx, w in ((-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
                          (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)):
            nxt = (cur[0] + dy, cur[1] + dx)
            if not (0 <= nxt[0] < ny and 0 <= nxt[1] < nx) or occ_plan[nxt]:
                continue
            if dx and dy and (occ_plan[cur[0] + dy, cur[1]] or occ_plan[cur[0], cur[1] + dx]):
                continue  # 禁止斜穿两格障碍的夹角
            ng = g + w
            if ng < g_cost.get(nxt, np.inf):
                g_cost[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
    return None


def _line_free(occ_plan, a, b):
    """a->b 直线途经格全自由 (整数超级覆盖近似)。"""
    n = max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2 + 1
    previous = a
    for t in np.linspace(0, 1, n):
        iy = int(round(a[0] + (b[0] - a[0]) * t))
        ix = int(round(a[1] + (b[1] - a[1]) * t))
        if occ_plan[iy, ix]:
            return False
        if iy != previous[0] and ix != previous[1]:
            if occ_plan[previous[0], ix] or occ_plan[iy, previous[1]]:
                return False
        previous = (iy, ix)
    return True


def shortcut(path_cells, occ_plan):
    """贪心视线捷径化: 能直达就跳过中间点。"""
    out = [path_cells[0]]
    i = 0
    while i < len(path_cells) - 1:
        j = len(path_cells) - 1
        while j > i + 1 and not _line_free(occ_plan, path_cells[i], path_cells[j]):
            j -= 1
        out.append(path_cells[j])
        i = j
    return out


def plan_path(occ_plan, origin, res, x, y, gx, gy):
    """地图系坐标 -> A* -> 捷径 -> 地图系路径点 (N,2)。"""
    ny, nx = occ_plan.shape
    # 注意 floor 而非 int(): int() 对负数向零截断会产生负索引回绕; 越界直接拒绝
    to_cell = lambda px, py: (
        int(np.floor((py - origin[1]) / res)),
        int(np.floor((px - origin[0]) / res)),
    )
    start_raw, goal_raw = to_cell(x, y), to_cell(gx, gy)
    for name, (iy, ix) in (("start", start_raw), ("goal", goal_raw)):
        if not (0 <= iy < ny and 0 <= ix < nx):
            print(f"[WARN] plan_path: {name} ({ix}, {iy}) outside grid, reject")
            return None
    if occ_plan[start_raw] or occ_plan[goal_raw]:
        print("[WARN] start/goal inside inflated obstacle, stop instead of moving endpoint")
        return None
    start, goal = start_raw, goal_raw
    cells = astar(occ_plan, start, goal)
    if cells is None:
        return None
    cells = shortcut(cells, occ_plan)
    pts = np.array([[origin[0] + c[1] * res, origin[1] + c[0] * res] for c in cells])
    # 首尾替换为精确坐标, 避免栅格中心引入的固定偏差;
    # 但若首尾格被占据而被挪到最近自由格, 则保留挪动后的位置
    if start_raw == start:
        pts[0] = [x, y]
    if goal_raw == goal:
        pts[-1] = [gx, gy]
    return pts


# ---------------------------------------------------------------------------
# 可视化 (可选)
# ---------------------------------------------------------------------------

class NavViz:
    def __init__(self, occ, origin, res):
        import webbrowser
        import viser
        self.server = None
        for port in (8090, 8091, 8092, 8093, 8094):
            try:
                self.server = viser.ViserServer(host="127.0.0.1", port=port,
                                                 verbose=False)
                break
            except OSError:
                continue
        if self.server is None:
            raise RuntimeError("no free port in 8090-8094 for viser server")
        self.server.scene.set_up_direction("+z")
        iy, ix = np.where(occ)
        pts = np.column_stack([origin[0] + ix * res, origin[1] + iy * res,
                               np.zeros(len(ix))])
        self.server.scene.add_point_cloud("obstacles", points=pts,
                                          colors=(0.2, 0.2, 0.2), point_size=0.04,
                                          precision="float32")
        print(f"[INFO] viz ready -> http://127.0.0.1:{port}")
        try:
            webbrowser.open(f"http://127.0.0.1:{port}")
        except Exception:
            pass

    def update(self, path, robot_xy, yaw, goal_xy):
        scene = self.server.scene
        if path is not None and len(path) >= 2:
            seg = np.stack([path[:-1], path[1:]], axis=1)          # (N, 2, 2)
            seg = np.concatenate([seg, np.zeros((*seg.shape[:2], 1))], axis=-1)
            scene.add_line_segments("path", points=seg,
                                    colors=(0.1, 0.6, 1.0), thickness=3.0,
                                    thickness_units="screen")
        head = np.array([[robot_xy[0], robot_xy[1], 0],
                         [robot_xy[0] + 0.5 * np.cos(yaw), robot_xy[1] + 0.5 * np.sin(yaw), 0]])
        scene.add_line_segments("robot", points=head[None], colors=(0.0, 0.9, 0.3),
                                thickness=4.0, thickness_units="screen")
        scene.add_point_cloud("goal", points=np.array([[goal_xy[0], goal_xy[1], 0]]),
                              colors=(1.0, 0.2, 0.1), point_size=0.1, precision="float32")


class InitProgressBar:
    """停车初定位期间的终端进度条: 后台线程刷新, 扫描匹配阻塞主线程时也能持续刷新。

    进度条反映当前这次扫描匹配的计算进度 (initial_match 内部按阶段回调 set_frac),
    计算结束恰好填满; 另显示本次尝试的墙钟耗时。"""

    def __init__(self, stage=""):
        import itertools
        import threading
        self._stage = stage
        self._frac = 0.0
        self._t0 = time.monotonic()
        self._spin = itertools.cycle("|/-\\")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_frac(self, f):
        self._frac = max(0.0, min(1.0, float(f)))

    def _run(self):
        while not self._stop.wait(0.1):
            el = time.monotonic() - self._t0
            n = int(self._frac * 24)
            bar = "█" * n + "░" * (24 - n)
            print(f"\r  [LOCALIZE] {bar} {self._frac * 100:3.0f}% "
                  f"{el:5.1f}s {next(self._spin)} {self._stage}  ", end="", flush=True)
        print("\r" + " " * 96 + "\r", end="", flush=True)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# 导航 device: 每控制步被 DataCollectionManager 调用
# ---------------------------------------------------------------------------

class AutopilotDevice(AbstractDevice):
    def __init__(self, args, drive, l_grip, r_grip, task_status, arms=None):
        super().__init__()
        self.drive = drive
        self.l_grip = l_grip
        self.r_grip = r_grip
        self.arms = arms or []
        self.last_arm_t = 0.0
        self.task_status = task_status
        self.args = args
        self.stub = make_stub(args.addr)
        self.occ_plan, self.occ_raw, self.origin, self.res = load_grid(
            args.grid, getattr(args, "robot_radius", 0.0))
        # 原始占据 + 5cm 余量: 执行层足迹检查用 (区别于规划层的圆形膨胀)
        from scipy.ndimage import distance_transform_edt
        self.occ_margin = distance_transform_edt(~self.occ_raw, sampling=self.res) <= 0.05
        self.goal_xy = np.array([float(v) for v in args.goal.split(",")[:2]]) \
            if args.goal else np.zeros(2)
        self.matcher = CorrelativeMatcher(args.map) if not args.drive_test else None
        self.init_candidate = None
        self.init_confirmations = 0
        self.init_started_t = None
        self.last_init_t = -np.inf
        self.viz = NavViz(np.load(args.grid)["occ"], self.origin, self.res) \
            if args.viz else None

        self.T_map_odom = None      # map <- odom 修正量
        self.T_map_odom_init = None  # --init 手动初值 (SE2 地图系位姿)
        if args.init:
            x, y, yaw = [float(v) for v in args.init.split(",")]
            self.T_map_odom_init = (x, y, np.radians(yaw))
        self.path = None
        self.cursor = 0
        self.last_plan_t = 0.0
        self.blocked_count = 0    # 前视弧线连续受阻计数 (用于触发倒车脱困)
        self.recovering = False   # 倒车脱困进行中
        self.recover_xy = (0.0, 0.0)
        self.recover_t0 = 0.0
        self.recovery_attempts = 0
        self.recover_yaw0 = 0.0      # 脱困起始航向 (固定方向计量倒车距离)
        self.recover_kappa = 0.0     # 脱困倒车曲率 (反号打舵实现三点转向)
        self.last_track_kappa = 0.0  # 最近一次纯跟踪指令曲率 (决定脱困打舵方向)
        self.halted = False
        self.progress_xy = None
        self.progress_t = 0.0
        self.last_v_cmd = 0.0
        self.started = False
        self.arrived = False
        self.n = 0
        self.last_pose_t = 0.0
        self.last_match_t = 0.0
        self.T_odom = None
        self.pose_xy_yaw = (0.0, 0.0, 0.0)
        self.zero_ctrl = np.zeros(4, dtype=np.float32)
        self.grip_init = np.array(g1_omnipicker_conf.gripper_l["init_ctrl"],
                                  dtype=np.float32)
        # --drive-test "v,steer": 调试模式, 恒定轮速/转向, 不做导航
        self.drive_test = None
        if getattr(args, "drive_test", None):
            v, steer = [float(v) for v in args.drive_test.split(",")]
            self.drive_test = np.array(
                [v / WHEEL_RADIUS, v / WHEEL_RADIUS, steer, steer],
                dtype=np.float32)
        self.test_steps = 0

    def _localize(self):
        """~50Hz: GT 里程计 + (可选) ~5Hz 相关性匹配修正。返回 (x, y, yaw) 地图系。"""
        now = float(self.drive.env.data.time)
        if self.T_odom is not None and now - self.last_pose_t < 0.02:
            return self.pose_xy_yaw
        self.last_pose_t = now
        # #region debug-point D:localization-timing
        debug_pose_t0 = time.perf_counter()
        # #endregion
        pose = query_lidar_pose(self.stub, self.args.lidar_body)
        if pose is None or not all(np.all(np.isfinite(p)) for p in pose):
            self._halt("localization unavailable; refusing stale pose")
            return self.pose_xy_yaw
        self.T_odom = pose_to_mat(*pose)
        if self.T_map_odom is None:
            self.drive.update_ctrl(self.zero_ctrl)
            if self.drive_test is not None:
                # 恒定驱动测试只输出相对起点位姿，不作为导航地图定位。
                self.T_map_odom = np.linalg.inv(self.T_odom)
            else:
                if self.init_started_t is None:
                    self.init_started_t = time.monotonic()
                    orca_logger.info("[LOCALIZE] 停车定位中，尚未允许导航")
                if time.monotonic() - self.init_started_t > self.args.init_timeout:
                    self._halt("initial localization timeout; provide --init hint or inspect map/scan")
                    return self.pose_xy_yaw
                if now - self.last_init_t < 0.2:
                    return self.pose_xy_yaw
                self.last_init_t = now
                data = query_lidar(self.stub, self.args.entity)
                if data is None:
                    return self.pose_xy_yaw
                pts_local = transform_to_local(data["points_world"][data["valid"]], pose[0], pose[1])
                hint = None
                stage = "全图搜索"
                if self.T_map_odom_init is not None:
                    hx, hy, ha = self.T_map_odom_init
                    hint = np.eye(4)
                    hint[:2, :2] = [[np.cos(ha), -np.sin(ha)], [np.sin(ha), np.cos(ha)]]
                    hint[:2, 3] = [hx, hy]
                    stage = "初值验证"
                elif self.init_candidate is not None:
                    # 第二帧复用首帧结果, 只在其邻域局部复扫确认
                    # (首帧已做全图穷举+多峰拒绝, 免去第二次全图搜索)
                    hint = self.init_candidate @ self.T_odom
                    stage = f"邻域复扫 (已一致 {self.init_confirmations}/2)"
                bar = InitProgressBar(stage)
                T, info = self.matcher.initial_match(pts_local, self.occ_plan, self.origin, self.res,
                                                     hint=hint, min_score=self.args.init_min_score,
                                                     min_margin=self.args.init_min_margin,
                                                     progress_cb=bar.set_frac)
                bar.stop()
                orca_logger.info(f"[LOCALIZE] {info}")
                if time.monotonic() - self.init_started_t > self.args.init_timeout:
                    self._halt("initial localization timeout")
                    return self.pose_xy_yaw
                if T is None:
                    self.init_confirmations = 0
                    self.init_candidate = None
                    if info["reason"] != "too_few_points":
                        self._halt("initial localization rejected; inspect [LOCALIZE] or supply --init hint")
                    return self.pose_xy_yaw
                candidate = T @ np.linalg.inv(self.T_odom)
                consistent = False
                if self.init_candidate is not None:
                    relative = np.linalg.inv(self.init_candidate) @ candidate
                    consistent = np.linalg.norm(relative[:2, 3]) < 0.15 and abs(np.arctan2(relative[1, 0], relative[0, 0])) < np.radians(5)
                self.init_confirmations = self.init_confirmations + 1 if consistent else 1
                self.init_candidate = candidate
                if self.init_confirmations < 2:
                    return self.pose_xy_yaw
                self.T_map_odom = candidate
                self.last_match_t = now
                orca_logger.info(f"[LOCALIZE] 定位通过 (耗时 {time.monotonic() - self.init_started_t:.1f}s)，"
                                 f"两次扫描结果一致，地图位姿={info['pose']}")
        T_est = self.T_map_odom @ self.T_odom
        if self.args.use_matcher and self.matcher is not None and now - self.last_match_t > 0.2:
            self.last_match_t = now
            data = query_lidar(self.stub, self.args.entity)
            if data is not None:
                pts_local = transform_to_local(
                    data["points_world"][data["valid"]], pose[0], pose[1])
                T_m = self.matcher.match(T_est, pts_local)
                if T_m is not None:
                    T_new = T_m @ np.linalg.inv(self.T_odom)
                    if np.all(np.isfinite(T_new)) and \
                            np.linalg.norm(T_new[:3, 3] - self.T_map_odom[:3, 3]) < 0.5:
                        self.T_map_odom = T_new
                    T_est = self.T_map_odom @ self.T_odom
        x, y = T_est[0, 3], T_est[1, 3]
        yaw = np.arctan2(T_est[1, 0], T_est[0, 0])
        self.pose_xy_yaw = (x, y, yaw)
        # #region debug-point D:localization-timing
        self.debug_localize_max_ms = max(getattr(self, "debug_localize_max_ms", 0.0), (time.perf_counter() - debug_pose_t0) * 1000)
        # #endregion
        return self.pose_xy_yaw

    def _pure_pursuit(self, x, y, yaw):
        """沿路径投影后取前视点，返回 (ctrl, arrived, v, 中心转角)。"""
        pos = np.array([x, y])
        dist_goal = np.linalg.norm(self.goal_xy - pos)
        if dist_goal < self.args.goal_tol:
            return self.zero_ctrl, True, 0.0, 0.0
        # 先找最近线段投影，再沿折线走 lookahead，而不是直接追远端拐点。
        target = self.path[-1].copy()
        best = np.inf
        segment, projection = 0, self.path[0].copy()
        for i in range(len(self.path) - 1):
            a, b = self.path[i:i + 2]
            ab = b - a
            t = np.clip(np.dot(pos - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
            p = a + t * ab
            d = np.linalg.norm(p - pos)
            if d < best:
                best, segment, projection = d, i, p
        remaining = self.args.lookahead
        sharp_turn = False
        for i in range(segment, len(self.path) - 1):
            end = self.path[i + 1]
            length = np.linalg.norm(end - projection)
            if length >= remaining:
                target = projection + (end - projection) * remaining / length
                break
            remaining -= length
            projection = end
            # 前视跨过拐点: 记录拐角尖锐程度, 用于进弯前提前减速
            if i + 2 < len(self.path):
                a = self.path[i + 1] - self.path[i]
                b = self.path[i + 2] - self.path[i + 1]
                turn = np.arctan2(a[0] * b[1] - a[1] * b[0],
                                  a[0] * b[0] + a[1] * b[1])
                if abs(turn) > np.radians(25):
                    sharp_turn = True
        delta = target - pos
        if np.cos(yaw) * delta[0] + np.sin(yaw) * delta[1] <= 0:
            self._halt("tracking target behind vehicle; forward-only path is not executable")
            return self.zero_ctrl, False, 0.0, 0.0
        distance = max(np.linalg.norm(delta), 1e-6)
        lateral = -np.sin(yaw) * delta[0] + np.cos(yaw) * delta[1]
        kappa = 2 * lateral / distance ** 2
        # 限制内侧轮不超过执行器角度上限，预测使用同一受限曲率。
        min_radius = self.args.wheelbase / np.tan(self.args.max_steer) + self.args.track_width / 2
        kappa = np.clip(kappa, -1 / min_radius, 1 / min_radius)
        self.last_track_kappa = float(kappa)
        # #region debug-point C:tracking-geometry
        self.debug_tracking = {"target": target.tolist(), "requested_kappa": float(2 * lateral / distance ** 2), "applied_kappa": float(kappa), "min_radius": float(min_radius)}
        # #endregion
        center_angle = np.arctan(kappa * self.args.wheelbase)
        dl = np.arctan2(self.args.wheelbase * kappa, 1 - self.args.track_width * kappa / 2)
        dr = np.arctan2(self.args.wheelbase * kappa, 1 + self.args.track_width * kappa / 2)
        v = self.args.max_speed * max(0.25, 1.0 - abs(center_angle) / self.args.max_steer)
        if sharp_turn:
            # 拐角在前视范围内: 进弯前主动限速, 减小欠转向外甩和弧线受阻刹停
            v = min(v, 0.5 * self.args.max_speed)
        v = min(v, 0.8 * dist_goal)
        # 前驱左右轮按各自转弯半径给轮速，减少等速转弯造成的滑动。
        wl = v * np.hypot(1 - kappa * self.args.track_width / 2, kappa * self.args.wheelbase) / WHEEL_RADIUS
        wr = v * np.hypot(1 + kappa * self.args.track_width / 2, kappa * self.args.wheelbase) / WHEEL_RADIUS
        scale = max(1.0, max(abs(wl), abs(wr)) / 30.0)
        return np.array([wl / scale, wr / scale, dl, dr], dtype=np.float32), False, v / scale, center_angle

    def _arc_blocked(self, x, y, yaw, v, center_angle):
        """沿弧线扫掠车体矩形足迹, 对原始占据+5cm 余量检查。
        矩形以后轴为原点: 前 +0.35 / 后 -0.15 x 半宽 body_halfwidth (含手臂)。
        规划层用圆形膨胀栅格 (A*), 执行层用真实矩形 —— 圆形近似在拐弯时
        会漏掉内后角与手臂的扫掠区, 这是拐弯剐蹭的根因。"""
        # #region debug-point A:arc-sample
        self.debug_arc_hit = None
        # #endregion
        horizon = np.sign(v) * max(0.15, 1.5 * abs(v))
        kappa = np.tan(center_angle) / self.args.wheelbase
        hw = self.args.body_halfwidth
        # 车体矩形采样点 (后轴系): 4 角 + 前中 + 两侧中
        offs = np.array([[0.35, hw], [0.35, -hw], [-0.15, hw], [-0.15, -hw],
                         [0.35, 0.0], [0.1, hw], [0.1, -hw]], dtype=float)
        n = max(8, int(np.ceil(abs(horizon) / (0.25 * self.res))))
        for s in np.linspace(0, horizon, n + 1):
            if abs(kappa) < 1e-6:
                px, py, pyaw = x + s * np.cos(yaw), y + s * np.sin(yaw), yaw
            else:
                px = x + (np.sin(yaw + kappa * s) - np.sin(yaw)) / kappa
                py = y - (np.cos(yaw + kappa * s) - np.cos(yaw)) / kappa
                pyaw = yaw + kappa * s
            c, sn = np.cos(pyaw), np.sin(pyaw)
            wx = px + offs[:, 0] * c - offs[:, 1] * sn
            wy = py + offs[:, 0] * sn + offs[:, 1] * c
            gx = np.floor((wx - self.origin[0]) / self.res).astype(int)
            gy = np.floor((wy - self.origin[1]) / self.res).astype(int)
            if not ((gx >= 0) & (gx < self.occ_margin.shape[1]) &
                    (gy >= 0) & (gy < self.occ_margin.shape[0])).all():
                # #region debug-point A:arc-boundary
                self.debug_arc_hit = {"reason": "outside_grid", "xy": [float(px), float(py)], "s": float(s)}
                # #endregion
                return True
            if self.occ_margin[gy, gx].any():
                # #region debug-point A:arc-obstacle
                self.debug_arc_hit = {"reason": "footprint_obstacle", "xy": [float(px), float(py)], "cell": [int(gy[self.occ_margin[gy, gx]][0]), int(gx[self.occ_margin[gy, gx]][0])], "s": float(s)}
                # #endregion
                return True
        return False

    def _dist_to_path(self, x, y):
        """当前位姿到路径折线的垂直距离 (点到各线段最近距离取 min)。"""
        p = self.path
        best = np.inf
        for i in range(len(p) - 1):
            ax, ay = p[i]
            bx, by = p[i + 1]
            abx, aby = bx - ax, by - ay
            ab2 = abx * abx + aby * aby
            if ab2 < 1e-9:
                d = np.hypot(x - ax, y - ay)
            else:
                t = np.clip(((x - ax) * abx + (y - ay) * aby) / ab2, 0.0, 1.0)
                d = np.hypot(x - (ax + t * abx), y - (ay + t * aby))
            best = min(best, d)
        return best

    def _halt(self, reason):
        self.halted = True
        self.recovering = False
        self.last_v_cmd = 0.0
        self.drive.update_ctrl(self.zero_ctrl)
        print(f"[ERROR] navigation halted: {reason}; inspect before restarting")

    def _begin_recovery(self, x, y, now):
        self.drive.update_ctrl(self.zero_ctrl)
        self.last_v_cmd = 0.0
        self.path = None
        if self.recovery_attempts >= self.args.max_recoveries:
            self._halt("recovery budget exhausted")
            return
        # 三点转向: 与被卡的前进曲率反号打满舵倒车, 倒车中车头朝拐角方向旋转,
        # 使下一次前进弧线能绕过小于最小转弯半径的拐角; 直线倒车航向不变,
        # 退回同类姿态后必然再次被同一拐角卡住 (前进后退死循环的根因)。
        kappa_rev = 0.0
        if self.last_track_kappa:
            # 与 _pure_pursuit 同一限幅基准 (内侧轮不超过 max_steer)
            min_radius = self.args.wheelbase / np.tan(self.args.max_steer) \
                + self.args.track_width / 2
            kappa_rev = -np.sign(self.last_track_kappa) / min_radius
        if kappa_rev and not self._arc_blocked(
                x, y, self.pose_xy_yaw[2], -self.args.recover_speed,
                np.arctan(kappa_rev * self.args.wheelbase)):
            self.recover_kappa = kappa_rev
        elif not self._arc_blocked(
                x, y, self.pose_xy_yaw[2], -self.args.recover_speed, 0.0):
            self.recover_kappa = 0.0
        else:
            self._halt("reverse corridor blocked or current pose inside obstacle")
            return
        self.recovery_attempts += 1
        self.recovering = True
        self.recover_xy = (x, y)
        self.recover_yaw0 = self.pose_xy_yaw[2]
        self.recover_t0 = now
        print(f"[WARN] recovery {self.recovery_attempts}/{self.args.max_recoveries}: "
              f"reverse kappa {self.recover_kappa:.2f} (three-point turn)")

    def _stalled(self, x, y, now):
        """持续有前进指令但实际位移不足，独立于地图预测检测卡死。"""
        if self.progress_xy is None or self.last_v_cmd <= 0:
            self.progress_xy, self.progress_t = (x, y), now
            return False
        if now - self.progress_t < self.args.stuck_timeout:
            return False
        distance = np.hypot(x - self.progress_xy[0], y - self.progress_xy[1])
        self.progress_xy, self.progress_t = (x, y), now
        if distance < self.args.stuck_distance:
            print(f"[WARN] no progress: {distance:.3f}m in {self.args.stuck_timeout}s")
            return True
        # 恢复次数按本次导航累计；短距离前进不代表已越过受阻拐角。
        return False

    def update(self):
        # 所有提前返回分支也保持夹爪输入有效。
        self.l_grip.update_ctrl(self.grip_init)
        self.r_grip.update_ctrl(self.grip_init)
        # 双臂 OSC 目标重锚定 (~50Hz): OSC 的 action 是世界系绝对位姿,
        # 若不刷新, 车开走后目标仍钉在出生点, 手臂会向后拽住车身。
        # 每步喂零增量 (等价 tele 中手柄不动), 目标跟随当前基座。
        if self.arms:
            now = time.time()
            if now - self.last_arm_t >= 0.02:
                self.last_arm_t = now
                for arm in self.arms:
                    arm.update_goal(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        if self.halted:
            self.drive.update_ctrl(self.zero_ctrl)
            return
        if self.drive_test is not None:
            x, y, yaw = self._localize()
            if self.halted:
                return
            self.test_steps += 1
            if self.test_steps == 1:
                print(f"[TEST] start pose ({x:.3f}, {y:.3f}, {np.degrees(yaw):.1f}deg), "
                      f"ctrl = {self.drive_test.tolist()}")
            if self.test_steps % 100 == 0 or self.test_steps >= 600:
                # 调试: 打印同名 body 的全部匹配 (排查重复 body / 查询异常)
                n_match = -1
                try:
                    from orca_gym.protos import mjc_message_pb2
                    rsp = self.stub.QueryBodyPosMatQuat(
                        mjc_message_pb2.QueryBodyPosMatQuatRequest(
                            body_name_list=[self.args.lidar_body]), timeout=2.0)
                    poses = [f"({it.pos[0]:.3f},{it.pos[1]:.3f})"
                             for it in rsp.body_pos_mat_quat_list
                             if it.body_name == self.args.lidar_body]
                    n_match = len(poses)
                except Exception as e:
                    poses = [f"query failed: {e}"]
                print(f"[TEST] step {self.test_steps}: pose ({x:.3f}, {y:.3f}, "
                      f"{np.degrees(yaw):.1f}deg), matches={n_match} {poses}")
            if self.test_steps >= 600:  # 600 步 x 5ms = 3s 仿真时间
                self.drive.update_ctrl(self.zero_ctrl)
                self.task_status.update_task_status(True)
                self.task_status.update_task_status(True)
                return
            self.drive.update_ctrl(self.drive_test)
            self.l_grip.update_ctrl(self.grip_init)
            self.r_grip.update_ctrl(self.grip_init)
            return
        if self.arrived:
            self.drive.update_ctrl(self.zero_ctrl)
            self.l_grip.update_ctrl(self.grip_init)
            self.r_grip.update_ctrl(self.grip_init)
            return
        x, y, yaw = self._localize()
        if self.halted or self.T_map_odom is None:
            self.drive.update_ctrl(self.zero_ctrl)
            return
        # 控制参考点转换: 雷达位姿 -> 后轴中点 (纯跟踪/弧线/偏离/规划锚定统一用后轴)
        x -= LIDAR_TO_REAR_AXLE * np.cos(yaw)
        y -= LIDAR_TO_REAR_AXLE * np.sin(yaw)
        if not self.started:
            self.started = True
            self.task_status.update_task_status(True)
            print(f"[INFO] navigating to {self.goal_xy.tolist()} ...")
        self.n += 1
        now = float(self.drive.env.data.time)

        if not self.recovering and self._stalled(x, y, now):
            self._begin_recovery(x, y, now)
            return
        # 倒车每步检查后方；失败锁定停车，不能超时后继续向同一障碍冲。
        if self.recovering:
            dist = -((x - self.recover_xy[0]) * np.cos(self.recover_yaw0) +
                     (y - self.recover_xy[1]) * np.sin(self.recover_yaw0))
            if dist >= self.args.recover_dist:
                self.recovering = False
                self.blocked_count = 0
                self.path = None
                self.progress_xy = None
                self.drive.update_ctrl(self.zero_ctrl)
                print(f"[INFO] recovery done (backed {dist:.2f}m), replanning")
                return
            kappa = self.recover_kappa
            if now - self.recover_t0 > 8.0 or self._arc_blocked(
                    x, y, yaw, -self.args.recover_speed,
                    np.arctan(kappa * self.args.wheelbase)):
                self._halt("reverse blocked or recovery timeout")
                return
            # 反号打舵倒车: 左右轮按各自转弯半径给轮速 (同 _pure_pursuit), v 取负
            v = -self.args.recover_speed
            dl = np.arctan2(self.args.wheelbase * kappa, 1 - self.args.track_width * kappa / 2)
            dr = np.arctan2(self.args.wheelbase * kappa, 1 + self.args.track_width * kappa / 2)
            wl = v * np.hypot(1 - kappa * self.args.track_width / 2, kappa * self.args.wheelbase) / WHEEL_RADIUS
            wr = v * np.hypot(1 + kappa * self.args.track_width / 2, kappa * self.args.wheelbase) / WHEEL_RADIUS
            self.drive.update_ctrl(np.array([wl, wr, dl, dr], dtype=np.float32))
            return

        # 规划 / 重规划:
        # a) 首次或路径失效时规划
        # b) 周期性重规划 (每 replan_period 秒, 从当前位置锚定新路径)
        # c) 偏离路径超过 replan_dist 时重规划
        need_plan = self.path is None
        if not need_plan and self.path is not None:
            if now - self.last_plan_t > self.args.replan_period:
                need_plan = True
                print(f"[INFO] periodic replan ({self.args.replan_period}s)")
            elif self._dist_to_path(x, y) > self.args.replan_dist:
                need_plan = True
                print(f"[WARN] deviated {self._dist_to_path(x, y):.2f}m from path, replanning")
        if need_plan:
            self.path = plan_path(self.occ_plan, self.origin, self.res,
                                  x, y, self.goal_xy[0], self.goal_xy[1])
            if self.path is None:
                self._halt("A* found no safe path")
                return
            self.cursor = 0
            self.last_plan_t = now
            print(f"[INFO] path planned: {len(self.path)} waypoints")

        # 前视防撞: 纯跟踪将走的弧线若撞占据格, 刹停重规划;
        # 连续受阻 (纯跟踪只会向前追点, 贴墙后左右拐都出不来) 则倒车脱困
        ctrl, arrived, v_cmd, steer_cmd = self._pure_pursuit(x, y, yaw)
        if self.halted:
            return
        if not arrived and self._arc_blocked(x, y, yaw, v_cmd, steer_cmd):
            self.blocked_count += 1
            self.drive.update_ctrl(self.zero_ctrl)
            self.last_v_cmd = 0.0
            self.progress_xy, self.progress_t = (x, y), now
            # #region debug-point A:blocked-geometry
            import json, urllib.request
            event = {"sessionId": "corner-recovery-loop", "runId": "post-fix", "hypothesisId": "A,B,C,D", "location": "update:arc_blocked", "msg": "[DEBUG] nav_blocked", "data": {"sim_t": now, "pose": [float(x), float(y), float(yaw)], "path": self.path.tolist(), "tracking": self.debug_tracking, "hit": self.debug_arc_hit, "requested_v": float(v_cmd), "steer": float(steer_cmd), "blocked_count": self.blocked_count, "attempts": self.recovery_attempts, "localize_max_ms": getattr(self, "debug_localize_max_ms", None)}}
            orca_logger.info(json.dumps(event))
            try:
                urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:7777/event", data=json.dumps(event).encode(), headers={"Content-Type": "application/json"}), timeout=0.02).close()
            except OSError:
                pass  # 已写入项目日志，采集服务不可用时不影响停车。
            self.debug_localize_max_ms = 0.0
            # #endregion
            if self.blocked_count >= 3:
                self._begin_recovery(x, y, now)
            else:
                print(f"[WARN] arc blocked ahead, braking + replanning")
                self.path = None
            self.drive.update_ctrl(self.zero_ctrl)
            self.l_grip.update_ctrl(self.grip_init)
            self.r_grip.update_ctrl(self.grip_init)
            return
        self.blocked_count = 0  # 弧线畅通, 清零连续受阻计数
        self.last_v_cmd = v_cmd
        self.drive.update_ctrl(ctrl)
        self.l_grip.update_ctrl(self.grip_init)
        self.r_grip.update_ctrl(self.grip_init)
        if arrived:
            self.arrived = True
            self.drive.update_ctrl(self.zero_ctrl)
            print(f"[INFO] goal reached at ({x:.2f}, {y:.2f}), stopping.")
            self.task_status.update_task_status(True)  # RUNNING -> END
        if self.viz is not None and self.n % 10 == 0:
            self.viz.update(self.path, (x, y), yaw, self.goal_xy)


# ---------------------------------------------------------------------------
# 主流程: 与 tele 相同的进程内装配, 驱动控制器换成 DATA 模式
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--goal", default=None, help="地图系目标点 'x,y' (米), 坐标看 ref_grid.png")
    parser.add_argument("--drive-test", default=None,
                        help="调试: 恒定驱动 'v,steer' (m/s, rad), 不做导航, 3s 后停止")
    parser.add_argument("--grid", default="slam_output/ref_grid.npz",
                        help="栅格 npz (g1_map_tools grid 输出)")
    parser.add_argument("--map", default="slam_output/ref_map.ply",
                        help="参考地图 PLY，启动定位必需，--use-matcher 时也用于行驶中修正")
    parser.add_argument("--use-matcher", action="store_true",
                        help="启动定位通过后，行驶中继续启用有界相关性匹配修正 (~5Hz)")
    parser.add_argument("--init", default=None,
                        help="可选的地图系初值提示 'x,y,yaw_deg'，仍需扫描匹配验证")
    parser.add_argument("--init-min-score", type=float, default=0.55,
                        help="启动定位最低3D匹配分数 [0,1]，默认0.55")
    parser.add_argument("--init-min-margin", type=float, default=0.08,
                        help="全局定位最优与异位姿候选最低分差，默认0.08")
    parser.add_argument("--init-timeout", type=float, default=120.0,
                        help="停车初定位墙钟超时秒数，超时拒绝导航")
    parser.add_argument("--viz", action="store_true", help="viser 浏览器可视化 (路径/机器人)")
    parser.add_argument("--addr", default="localhost:50051")
    parser.add_argument("--entity", default=DEFAULT_LIDAR_ENTITY)
    parser.add_argument("--lidar-body", default=DEFAULT_LIDAR_BODY)
    parser.add_argument("--level", type=str, default="default")
    parser.add_argument("--max-speed", type=float, default=0.4, help="巡航速度 m/s (default 0.4)")
    parser.add_argument("--lookahead", type=float, default=0.6, help="纯跟踪前视距离 m (default 0.6)")
    parser.add_argument("--goal-tol", type=float, default=0.25, help="到点判定半径 m (default 0.25)")
    parser.add_argument("--replan-period", type=float, default=1.0,
                        help="周期性重规划间隔 (秒), 从当前位置重新锚定路径")
    parser.add_argument("--replan-dist", type=float, default=0.8,
                        help="偏离路径超过该距离触发重规划 m (default 0.8)")
    parser.add_argument("--stuck-timeout", type=float, default=1.0,
                        help="有前进指令但位移不足的检测窗口，仿真秒")
    parser.add_argument("--stuck-distance", type=float, default=0.03,
                        help="卡死检测窗口内的最小位移 m")
    parser.add_argument("--recover-dist", type=float, default=0.6,
                        help="倒车脱困距离 m (default 0.6)")
    parser.add_argument("--recover-speed", type=float, default=0.25,
                        help="倒车速度 m/s (default 0.25)")
    parser.add_argument("--max-recoveries", type=int, default=2,
                        help="单次导航累计倒车脱困上限，达到后再次受阻则停车；0 禁止倒车")
    parser.add_argument("--wheelbase", type=float, default=0.42)
    parser.add_argument("--track-width", type=float, default=0.26)
    parser.add_argument("--max-steer", type=float, default=0.6)
    parser.add_argument("--robot-radius", type=float, default=0.35,
                        help="A* 规划层圆形膨胀半径 m (从 npz 原始 occ 重新膨胀; "
                             "0 = 使用烘焙的 occ_plan; 执行层防撞用车体矩形, 不受此参数影响)")
    parser.add_argument("--body-halfwidth", type=float, default=0.28,
                        help="车体矩形足迹半宽 m (含手臂, 挂点 y=0.21 + 臂半径; "
                             "执行层弧线防撞扫掠用)")
    args = parser.parse_args()
    if not (0 < args.init_min_score <= 1 and 0 < args.init_min_margin <= 1
            and np.isfinite(args.init_timeout) and args.init_timeout > 0):
        parser.error("启动定位分数/分差必须在 (0,1]，超时必须为有限正数")
    if args.init:
        try:
            initial_pose = np.array([float(v) for v in args.init.split(",")])
        except ValueError:
            parser.error("--init 格式必须为 x,y,yaw_deg")
        if initial_pose.shape != (3,) or not np.all(np.isfinite(initial_pose)):
            parser.error("--init 必须提供三个有限数值 x,y,yaw_deg")

    orcagym_addr = "localhost:50051"
    default_joint_values = {}
    for joint_name, value in zip(g1_omnipicker_conf.l_arm["joint_names"],
                                 g1_omnipicker_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(g1_omnipicker_conf.r_arm["joint_names"],
                                 g1_omnipicker_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value

    with open(os.path.join(base_dir, "example.yaml"), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(orcagym_addr, config=config)
    scene_manager.show_ui_message(1, "自动驾驶: A* 导航中", "0xffff00", showtime=5)
    scene_manager.get_scene_data("g1_autopilot.py", "beginscene")

    data_collection_manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=lambda env: np.zeros(1, dtype=np.float32),
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=5,
    )
    env = data_collection_manager.env
    data_collection_manager.set_disable_actuator_group(
        [g1_omnipicker_conf.positions_group])

    # 双臂 OSC: device 每步喂零增量目标, 使末端跟随基座 (保持相对车体不动)
    arms = []
    for arm_conf in (g1_omnipicker_conf.l_arm, g1_omnipicker_conf.r_arm):
        ctrl_name = [env.actuator(m) for m in arm_conf["motors_names"]]
        init_ctrl = {n: v for n, v in zip(ctrl_name, arm_conf["motors_init_ctrl"])}
        arm_ctrl = create_arm_osc_controller(env, arm_conf, g1_omnipicker_conf.base_body,
                                             ctrl_name, init_ctrl)
        data_collection_manager.add_controller(arm_ctrl)
        arms.append(arm_ctrl)

    # 双夹爪 DATA 模式 (device 每步喂初始值保持张开)
    grips = []
    for grip_conf in (g1_omnipicker_conf.gripper_l, g1_omnipicker_conf.gripper_r):
        gname = [env.actuator(n) for n in grip_conf["actuator_names"]]
        init_g = {n: v for n, v in zip(gname, grip_conf["init_ctrl"])}
        grip = create_gripper_2f85_controller(
            env, grip_conf, g1_omnipicker_conf.base_body, gname, init_g,
            Controller2F85.ControllerType.DATA)
        data_collection_manager.add_controller(grip)
        grips.append(grip)

    # 前轮转向驱动 DATA 模式: [v_fl, v_fr, δ_l, δ_r]
    drive_conf = g1_omnipicker_conf.front_drive
    ctrl_name = [env.actuator(m) for m in drive_conf["actuator_names"]]
    init_ctrl = {n: v for n, v in zip(ctrl_name, drive_conf["init_ctrl"])}
    drive = create_steering_drive_controller(
        env, drive_conf, ctrl_name, init_ctrl,
        ControllerSteeringDrive.ControllerType.DATA)
    data_collection_manager.add_controller(drive)

    task_status = TaskStatusController(env, g1_omnipicker_conf.base_body,
                                       is_controller=False)
    data_collection_manager.set_task_status_controller(task_status)
    data_collection_manager.set_task(EmptyTask(env))
    data_collection_manager.save_video = False

    device = AutopilotDevice(args, drive, grips[0], grips[1], task_status, arms=arms)
    data_collection_manager.set_device(device)

    env.disable_actuator(data_collection_manager.disable_actuator_group)
    try:
        env.reset()
        if not data_collection_manager.update_scene():
            orca_logger.info("update_scene failed, exit")
            return
        task_is_success, *_ = data_collection_manager.run_episode()
        orca_logger.info(f"Episode finished, success={task_is_success}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as e:
        orca_logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        orca_logger.info("Exiting program")
        os._exit(0)
