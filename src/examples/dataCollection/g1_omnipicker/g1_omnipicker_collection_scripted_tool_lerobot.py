"""G1 OmniPicker 工具整理脚本化数据采集。"""
import argparse
import os
import sys
import time
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
_common_dir = os.path.abspath(os.path.join(base_dir, "..", "common"))
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)

import numpy as np
from scipy.spatial.transform import Rotation as R
from yaml import Loader, load, safe_load

import data_collection_scripted as scripted

from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controller_task import TaskStatus, TaskStatusController
from controllers.controllers import (
    create_arm_osc_controller,
    create_gripper_2f85_reverse_controller,
)
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_CAMERA_MAP,
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage, LeRobotDatasetWriter
from devices.abstract_device import AbstractDevice
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_scripted_tool_lerobot_stream"

log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="G1ToolScripted",
    log_file="g1_omnipicker_collection_scripted_tool_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

_L_INIT_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 顺序必须与 _TOOL_BODY_JOINT_NAMES 一致。
_TOOL_NAMES = ["扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)"]

# 基座坐标系；随机化仅修改槽位 Y。
_TOOL_BODY_JOINT_NAMES = [
    "Group_Interactive_Spanner_task_spanner_joint",
    "Group_Interactive_Screwdriver_task_screwdriver_joint",
    "Group_Interactive_ElectriciansKnife01_task_electriciansknife01_joint",
    "Group_Interactive_Flashlight_task_flashlight_joint",
    "Group_Interactive_ElectriciansKnife02_task_electriciansknife02_joint",
]
# _place_tools 使用的货架参考位置。
_TOOL_REFERENCE_POS_B = np.asarray(
    [
        [0.5654831, -0.0693993, 0.1514528],
        [0.5599098, -0.1873288, 0.1568153],
        [0.5764828, -0.3049245, 0.1447767],
        [0.5079708, -0.4143662, 0.1536523],
        [0.5802917, -0.5180783, 0.1439366],
    ],
    dtype=np.float64,
)
_TOOL_REFERENCE_QUAT_XYZW_B = np.asarray(
    [
        [-0.0, 0.7071068, 0.7071067, 0.0],
        [0.4999249, -0.5000751, -0.4999244, 0.5000756],
        [-0.0, 0.7071065, 0.7071071, 0.0],
        [-0.5, 0.5, -0.4999995, 0.5000005],
        [1.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# 工具箱底座 body（内侧底面碰撞体挂在此 body 上）
_TOOLBOX_BASE_BODY = "Group_Interactive_ToolBox_Base_bodyjoint"
# 几何回退（仅用于箱底标定）：工具 COM 相对底面高度上限（米）
_BOX_BOTTOM_Z_TOL = 0.10
_BOX_BOTTOM_XY_MARGIN = 0.02
# 末端离开箱体 XY 区域后的接触检测超时，单位秒。
_PLACE_CHECK_TIMEOUT_S = 2.5
# 末工具后撤离（保证 EE xy 出整箱矩形）的步数
_PLACE_RETREAT_STEPS = 50
# EE 是否仍在「箱子 xy 矩形」内时的外扩（米）；矩形用整箱 AABB，不是箱底薄板
_PLACE_BOX_XY_MARGIN = 0.02
# 机器人 body 名关键字（工具仍与其接触 = 未放下）
_ROBOT_BODY_KEYWORDS = (
    "g1_omnipicker",
    "gripper",
    "2f85",
    "palm",
    "finger",
    "hand",
)


# ---------------------------------------------------------------------------
# 箱底内侧表面标定 + 入箱接触检测
# ---------------------------------------------------------------------------

def _body_rot_in_B(env, body: str, base_body: str) -> R:
    """body 相对 base_body 的旋转（scipy Rotation）。"""
    _, _, q_w = env.get_body_xpos_xmat_xquat([body])
    _, _, qb_w = env.get_body_xpos_xmat_xquat([base_body])
    q_w = np.asarray(q_w, dtype=np.float64).reshape(4)
    qb_w = np.asarray(qb_w, dtype=np.float64).reshape(4)
    R_base = R.from_quat(qb_w[[1, 2, 3, 0]])
    R_body = R.from_quat(q_w[[1, 2, 3, 0]])
    return R_base.inv() * R_body


def _geom_corners_in_B(env, g: dict, base_body: str) -> np.ndarray:
    """单个 geom 的 8 个角点，变换到 base 系，shape (8, 3)。"""
    body = str(g.get("BodyName") or "")
    body_pos_b = np.asarray(
        env.query_position_body_B(body, base_body), dtype=np.float64
    )
    R_body_b = _body_rot_in_B(env, body, base_body)
    g_pos = np.asarray(g["Pos"], dtype=np.float64).reshape(3)
    g_quat = np.asarray(g["Quat"], dtype=np.float64).reshape(4)  # wxyz
    R_geom = R.from_quat(g_quat[[1, 2, 3, 0]])
    size = np.asarray(g["Size"], dtype=np.float64).reshape(3)
    sx, sy, sz = [float(abs(v)) for v in size]
    corners_local = np.array(
        [[a, b, c] for a in (-sx, sx) for b in (-sy, sy) for c in (-sz, sz)],
        dtype=np.float64,
    )
    corners_body = g_pos + R_geom.apply(corners_local)
    return body_pos_b + R_body_b.apply(corners_body)


def _calibrate_toolbox_xy_aabb(env, base_body: str) -> tuple[np.ndarray, np.ndarray, int]:
    """整箱（所有 ToolBox* geom）在 base 系的 xy AABB，供 EE 投影是否仍在箱上判定。"""
    geom_dict = env.model.get_geom_dict()
    xy_min = None
    xy_max = None
    n_geoms = 0
    for _gname, g in geom_dict.items():
        body = str(g.get("BodyName") or "")
        if "ToolBox" not in body:
            continue
        try:
            corners_b = _geom_corners_in_B(env, g, base_body)
        except Exception:
            continue
        cmin = corners_b[:, :2].min(axis=0)
        cmax = corners_b[:, :2].max(axis=0)
        xy_min = cmin if xy_min is None else np.minimum(xy_min, cmin)
        xy_max = cmax if xy_max is None else np.maximum(xy_max, cmax)
        n_geoms += 1
    if xy_min is None or xy_max is None:
        raise RuntimeError("未找到 ToolBox geom，无法标定整箱 xy")
    return (
        xy_min.astype(np.float64),
        xy_max.astype(np.float64),
        int(n_geoms),
    )


def _calibrate_box_bottom_inner(env, base_body: str) -> dict:
    """标定箱底内侧 + 整箱 xy 矩形。

    返回 dict:
      xy_min, xy_max       : 箱底薄板 xy（入箱几何回退）
      box_xy_min/max       : 整箱 AABB xy（EE 投影是否仍在箱上）
      z_surface            : 底面上表面 z（base 系）
      sample_points        : 底面采样点
      geom_name            : 选用的底面 geom
    """
    geom_dict = env.model.get_geom_dict()
    box_pos_b = np.asarray(
        env.query_position_body_B(_TOOLBOX_BASE_BODY, base_body), dtype=np.float64
    )
    R_box_b = _body_rot_in_B(env, _TOOLBOX_BASE_BODY, base_body)

    best = None
    best_score = -1.0
    for gname, g in geom_dict.items():
        if g.get("BodyName") != _TOOLBOX_BASE_BODY:
            continue
        size = np.asarray(g["Size"], dtype=np.float64).reshape(3)
        dims = np.sort(np.abs(size))
        thickness, mid, length = float(dims[0]), float(dims[1]), float(dims[2])
        area = mid * length
        score = area / max(thickness, 1e-6)
        if score > best_score:
            best_score = score
            best = (gname, g, size)

    if best is None:
        raise RuntimeError("未找到 ToolBox Base 的碰撞 geom，无法标定箱底")

    gname, g, size = best
    g_pos = np.asarray(g["Pos"], dtype=np.float64).reshape(3)
    g_quat = np.asarray(g["Quat"], dtype=np.float64).reshape(4)  # wxyz
    R_geom = R.from_quat(g_quat[[1, 2, 3, 0]])
    sx, sy, sz = [float(abs(v)) for v in size]
    corners_local = np.array(
        [[a, b, c] for a in (-sx, sx) for b in (-sy, sy) for c in (-sz, sz)],
        dtype=np.float64,
    )
    corners_box = g_pos + R_geom.apply(corners_local)
    corners_b = box_pos_b + R_box_b.apply(corners_box)

    xy_min = corners_b[:, :2].min(axis=0)
    xy_max = corners_b[:, :2].max(axis=0)
    z_surface = float(corners_b[:, 2].max())  # 底板顶面

    # 若已有工具与箱底接触 / 落在底面范围内，用公共接触带收紧底面（不影响整箱 xy）
    contact_tools_xy = []
    contact_tools_z = []
    for tool_body in _TOOL_BODY_JOINT_NAMES:
        ok, _, pos_b = _tool_resting_on_box_bottom(
            env, tool_body, base_body,
            xy_min=xy_min, xy_max=xy_max, z_surface=z_surface,
            use_contact=True, use_geom=True,
        )
        if ok and pos_b is not None:
            contact_tools_xy.append(pos_b[:2])
            contact_tools_z.append(float(pos_b[2]))

    if len(contact_tools_z) >= 2:
        zs = np.asarray(contact_tools_z, dtype=np.float64)
        xys = np.stack(contact_tools_xy, axis=0)
        z_surface = float(np.median(zs) - 0.02)
        pad = 0.03
        xy_min = np.minimum(xy_min, xys.min(axis=0) - pad)
        xy_max = np.maximum(xy_max, xys.max(axis=0) + pad)
        orca_logger.info(
            f"[箱底标定] 用 {len(contact_tools_z)} 件已入箱工具收紧底面 "
            f"z≈{z_surface:.4f}"
        )

    box_xy_min, box_xy_max, n_box_geoms = _calibrate_toolbox_xy_aabb(env, base_body)

    xs = np.linspace(xy_min[0], xy_max[0], 3)
    ys = np.linspace(xy_min[1], xy_max[1], 3)
    sample_points = np.array(
        [[x, y, z_surface] for x in xs for y in ys], dtype=np.float64
    )

    info = {
        "xy_min": xy_min.astype(np.float64),
        "xy_max": xy_max.astype(np.float64),
        "box_xy_min": box_xy_min,
        "box_xy_max": box_xy_max,
        "z_surface": z_surface,
        "sample_points": sample_points,
        "geom_name": gname,
    }
    orca_logger.info(
        f"[箱底标定] geom={gname} | "
        f"底面xy=[{xy_min[0]:.3f},{xy_max[0]:.3f}]×[{xy_min[1]:.3f},{xy_max[1]:.3f}] "
        f"z_surface={z_surface:.4f} | "
        f"整箱xy=[{box_xy_min[0]:.3f},{box_xy_max[0]:.3f}]×"
        f"[{box_xy_min[1]:.3f},{box_xy_max[1]:.3f}] ({n_box_geoms} geoms)"
    )
    print(
        f"  箱底内侧: z={z_surface:.3f}  "
        f"xy=[{xy_min[0]:.2f},{xy_max[0]:.2f}]×[{xy_min[1]:.2f},{xy_max[1]:.2f}]",
        flush=True,
    )
    print(
        f"  整箱xy(EE判定): "
        f"[{box_xy_min[0]:.2f},{box_xy_max[0]:.2f}]×"
        f"[{box_xy_min[1]:.2f},{box_xy_max[1]:.2f}]",
        flush=True,
    )
    return info


def _body_name_matches(contact_body: str, target: str) -> bool:
    """接触体名可能带场景前缀，做包含匹配。"""
    if not contact_body or not target:
        return False
    return (
        contact_body == target
        or contact_body.endswith(target)
        or target in contact_body
    )


def _is_toolbox_body(name: str) -> bool:
    return bool(name) and (
        "ToolBox" in name or _body_name_matches(name, _TOOLBOX_BASE_BODY)
    )


def _is_robot_body(name: str) -> bool:
    if not name:
        return False
    low = name.lower()
    return any(k in low for k in _ROBOT_BODY_KEYWORDS)


def _match_tool_body(name: str) -> str | None:
    """若 name 对应某件工具，返回其 canonical body/joint 名。"""
    for t in _TOOL_BODY_JOINT_NAMES:
        if _body_name_matches(name, t):
            return t
    return None


def _tool_resting_on_box_bottom(
    env,
    tool_body: str,
    base_body: str,
    xy_min=None,
    xy_max=None,
    z_surface: float | None = None,
    use_contact: bool = True,
    use_geom: bool = True,
    z_tol: float = _BOX_BOTTOM_Z_TOL,
    xy_margin: float = _BOX_BOTTOM_XY_MARGIN,
) -> tuple[bool, str, np.ndarray | None]:
    """箱底标定用的宽松几何/接触回退（非采集主判定）。"""
    try:
        pos_b = np.asarray(
            env.query_position_body_B(tool_body, base_body), dtype=np.float64
        )
    except Exception as e:
        return False, f"query_pos_fail:{e}", None

    if use_contact:
        try:
            contacts = env.query_contact_simple()
            model = env.model
            for c in contacts:
                b1 = model.get_geom_body_name(c["Geom1"])
                b2 = model.get_geom_body_name(c["Geom2"])
                bodies = (b1, b2)
                on_tool = any(_body_name_matches(b, tool_body) for b in bodies)
                if on_tool and any(_is_toolbox_body(b) for b in bodies):
                    return True, "contact_toolbox", pos_b
        except Exception as e:
            orca_logger.warning(f"[入箱检测] query_contact_simple 失败: {e}")

    if use_geom and xy_min is not None and xy_max is not None and z_surface is not None:
        xy_min = np.asarray(xy_min, dtype=np.float64)
        xy_max = np.asarray(xy_max, dtype=np.float64)
        in_xy = (
            (pos_b[0] >= xy_min[0] - xy_margin)
            and (pos_b[0] <= xy_max[0] + xy_margin)
            and (pos_b[1] >= xy_min[1] - xy_margin)
            and (pos_b[1] <= xy_max[1] + xy_margin)
        )
        dz = float(pos_b[2] - z_surface)
        if in_xy and (-0.03 <= dz <= z_tol):
            return True, f"geom_xy_z(dz={dz*1000:.0f}mm)", pos_b
        return False, f"xy={'ok' if in_xy else 'out'}|dz={dz*1000:.0f}mm", pos_b

    return False, "no_contact", pos_b


def _is_soft_world_contact(name: str, cat: str) -> bool:
    """判断可忽略的世界平面接触。"""
    if cat != "other" or not name:
        return False
    return "world" in name.lower()


def _tool_geom_in_box(
    pos_b: np.ndarray,
    box_bottom: dict | None,
    *,
    xy_margin: float = 0.04,
    z_tol: float = _BOX_BOTTOM_Z_TOL,
) -> bool:
    """COM 是否落在箱底 xy 内且靠近底面（接触白名单的几何回退）。"""
    if box_bottom is None or pos_b is None:
        return False
    xy_min = np.asarray(box_bottom["xy_min"], dtype=np.float64)
    xy_max = np.asarray(box_bottom["xy_max"], dtype=np.float64)
    z_surface = float(box_bottom["z_surface"])
    in_xy = (
        (pos_b[0] >= xy_min[0] - xy_margin)
        and (pos_b[0] <= xy_max[0] + xy_margin)
        and (pos_b[1] >= xy_min[1] - xy_margin)
        and (pos_b[1] <= xy_max[1] + xy_margin)
    )
    dz = float(pos_b[2] - z_surface)
    return bool(in_xy and (-0.03 <= dz <= z_tol))


def _classify_tool_place_contacts(
    env,
    tool_body: str,
    base_body: str,
    allowed_inbox_tools: set[str],
    box_bottom: dict | None = None,
) -> tuple[str, str, np.ndarray | None]:
    """采集主判定：检查工具当前接触是否全部合法。

    返回 (status, detail, pos_b)：
      - waiting: 尚无接触，继续等
      - ok: 已有接触且全部为箱子内表面 / 已入箱工具
      - fail: 存在非法接触（机器人=未放下，货架工具/其它物体等）

    说明：仅 other:world 且 COM 已在箱内几何范围内时，视为合法入箱
    （箱底碰撞体未及时报到、或地面平面穿透误报时常见）。
    """
    try:
        pos_b = np.asarray(
            env.query_position_body_B(tool_body, base_body), dtype=np.float64
        )
    except Exception as e:
        return "fail", f"query_pos_fail:{e}", None

    partners: list[tuple[str, str]] = []  # (body_name, category)
    try:
        contacts = env.query_contact_simple()
        model = env.model
        for c in contacts:
            b1 = model.get_geom_body_name(c["Geom1"])
            b2 = model.get_geom_body_name(c["Geom2"])
            if _body_name_matches(b1, tool_body):
                other = b2
            elif _body_name_matches(b2, tool_body):
                other = b1
            else:
                continue
            if _body_name_matches(other, tool_body):
                continue
            if _is_toolbox_body(other):
                partners.append((other, "box"))
                continue
            matched_tool = _match_tool_body(other)
            if matched_tool is not None:
                if matched_tool in allowed_inbox_tools:
                    partners.append((other, "inbox_tool"))
                else:
                    partners.append((other, "rack_tool"))
                continue
            if _is_robot_body(other):
                partners.append((other, "robot"))
                continue
            partners.append((other, "other"))
    except Exception as e:
        return "fail", f"query_contact_fail:{e}", pos_b

    if not partners:
        # 尚无接触：若已稳定落在箱内几何带，也算成功（避免只靠接触漏检）
        if _tool_geom_in_box(pos_b, box_bottom):
            return "ok", "legal_contact|geom_inbox", pos_b
        return "waiting", "no_contact", pos_b

    legal_partners = [
        (n, cat) for n, cat in partners if cat in ("box", "inbox_tool")
    ]
    illegal = [
        (n, cat) for n, cat in partners if cat not in ("box", "inbox_tool")
    ]
    # 已有合法箱/叠放接触时，忽略 world 噪声
    if legal_partners:
        illegal = [
            (n, cat) for n, cat in illegal
            if not _is_soft_world_contact(n, cat)
        ]
        if not illegal:
            legal = sorted({cat for _, cat in legal_partners})
            return "ok", "legal_contact|" + "+".join(legal), pos_b

    if illegal:
        only_soft_world = all(
            _is_soft_world_contact(n, cat) for n, cat in illegal
        )
        if only_soft_world and _tool_geom_in_box(pos_b, box_bottom):
            return "ok", "legal_contact|geom_inbox+world", pos_b
        bits = [f"{cat}:{n}" for n, cat in illegal]
        return "fail", "illegal_contact|" + ",".join(bits), pos_b

    legal = sorted({cat for _, cat in partners})
    return "ok", "legal_contact|" + "+".join(legal), pos_b


# ---------------------------------------------------------------------------
# G1ScriptedTrajectoryDevice（离线预计算轨迹回放）
# ---------------------------------------------------------------------------

class G1ScriptedTrajectoryDevice(AbstractDevice):
    """每步把预计算好的 B 系末端位姿写入 OSC 臂控制器，夹爪广播到双执行器。"""

    def __init__(
        self,
        l_arm,
        r_arm,
        l_grip: Controller2F85Reverse,
        r_grip: Controller2F85Reverse,
        task_status: TaskStatusController,
        l_pos: np.ndarray,
        l_quat_xyzw: np.ndarray,
        r_pos: np.ndarray,
        r_quat_xyzw: np.ndarray,
        l_grip_motor: np.ndarray,
        r_grip_motor: np.ndarray,
        seg_bounds=None,
        place_arm_at=None,
        box_bottom=None,
        base_body_query: str | None = None,
        place_timeout_steps: int = 80,
    ):
        super().__init__()
        n = len(l_pos)
        assert (
            len(r_pos) == n
            and len(l_quat_xyzw) == n
            and len(r_quat_xyzw) == n
            and len(l_grip_motor) == n
            and len(r_grip_motor) == n
        )
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.l_grip = l_grip
        self.r_grip = r_grip
        self.task_status = task_status
        self.l_pos = l_pos
        self.l_quat_xyzw = l_quat_xyzw
        self.r_pos = r_pos
        self.r_quat_xyzw = r_quat_xyzw
        self.l_grip_motor = l_grip_motor
        self.r_grip_motor = r_grip_motor
        self.t = 0
        # 段边界诊断：{step_index: (label, commanded_r_target_b)}
        self.seg_bounds = seg_bounds or {}
        # 入箱监视：该工具全部路点结束步 → tool_idx
        self.place_arm_at = place_arm_at or {}  # {step: tool_idx}
        self.box_bottom = box_bottom
        self.base_body_query = base_body_query
        self.place_timeout_steps = max(1, int(place_timeout_steps))
        self.place_failed = False
        self.place_fail_reason = ""
        # 本集已判定入箱成功的工具 body（可供后续叠放）
        self._placed_inbox_tools: set[str] = set()
        # 活跃监视器：wait_ee_outside → wait_contact
        # （该工具全部路点结束后启动；不检测“离开动作”，只看 EE xy 是否已出整箱矩形）
        self._place_watchers: list[dict] = []

    def _force_task_end(self):
        """尽快结束本集录制（RUNNING→END）。已是 END 时不再推进，避免回到 NOT_STARTED。"""
        if self.task_status.current_status == TaskStatus.END:
            return
        if self.task_status.current_status == TaskStatus.NOT_STARTED:
            self.task_status.update_task_status(True)
        if self.task_status.current_status == TaskStatus.RUNNING:
            self.task_status.update_task_status(True)

    def _query_ee_xy_b(self):
        """右臂末端在 base 系的 xy；失败返回 None。"""
        try:
            ee_b = self.r_arm.env.query_site_pos_and_quat_B(
                [self.r_arm.ee_name], [self.r_arm.base_link]
            )
            pos = np.asarray(ee_b[self.r_arm.ee_name]["xpos"], dtype=np.float64)
            return pos[:2]
        except Exception as e:
            orca_logger.warning(f"[入箱检测] 查询 EE 失败: {e}")
            return None

    def _ee_in_box_xy(self, ee_xy) -> bool:
        """EE 的 xy 投影是否仍在整箱矩形内（含外扩）。"""
        if ee_xy is None or self.box_bottom is None:
            return False
        xy_min = np.asarray(
            self.box_bottom.get("box_xy_min", self.box_bottom["xy_min"]),
            dtype=np.float64,
        )
        xy_max = np.asarray(
            self.box_bottom.get("box_xy_max", self.box_bottom["xy_max"]),
            dtype=np.float64,
        )
        m = _PLACE_BOX_XY_MARGIN
        return bool(
            (ee_xy[0] >= xy_min[0] - m)
            and (ee_xy[0] <= xy_max[0] + m)
            and (ee_xy[1] >= xy_min[1] - m)
            and (ee_xy[1] <= xy_max[1] + m)
        )

    def _fail_one_tool(self, tool_idx: int, when_label: str, detail: str, pos_b) -> None:
        tool_name = _TOOL_NAMES[tool_idx]
        pos_str = (
            f"pos={np.asarray(pos_b).round(4).tolist()}"
            if pos_b is not None
            else "pos=?"
        )
        self.place_failed = True
        self.place_fail_reason = (
            f"{when_label}: 工具 {tool_name} 入箱失败 ({detail}) {pos_str}"
        )
        orca_logger.warning(f"[入箱检测] ✗ {self.place_fail_reason} → 放弃本集")
        print(f"  ✗ 入箱失败: {self.place_fail_reason}", flush=True)
        self._force_task_end()

    def _tick_place_monitors(self):
        """该工具路点全部结束后：等 EE xy 已出整箱矩形 → 再查接触白名单。

        合法接触：箱子内表面、已入箱工具。
        非法接触：机器人（未放下）、货架工具、其它物体。
        超时仍无接触：失败。
        """
        if self.place_failed or self.box_bottom is None:
            return

        if self.t in self.place_arm_at:
            tool_idx = int(self.place_arm_at[self.t])
            self._place_watchers.append(
                {
                    "tool_idx": tool_idx,
                    "phase": "wait_ee_outside",
                    "since_outside": 0,
                }
            )
            ee_xy0 = self._query_ee_xy_b()
            ee_str = (
                f"ee_xy={np.asarray(ee_xy0).round(4).tolist()}"
                if ee_xy0 is not None
                else "ee_xy=?"
            )
            orca_logger.info(
                f"[入箱检测] 启动监视 {_TOOL_NAMES[tool_idx]} "
                f"（该工具路点结束@step{self.t}；{ee_str}；"
                f"待 EE xy 出整箱矩形后再等接触，"
                f"超时 {self.place_timeout_steps} 步）"
            )

        if not self._place_watchers:
            return

        ee_xy = self._query_ee_xy_b()
        ee_in_box = self._ee_in_box_xy(ee_xy)
        still_active: list[dict] = []
        for w in self._place_watchers:
            tool_idx = int(w["tool_idx"])
            name = _TOOL_NAMES[tool_idx]
            tool_body = _TOOL_BODY_JOINT_NAMES[tool_idx]
            phase = w["phase"]
            if phase == "wait_ee_outside":
                if not ee_in_box:
                    w["phase"] = "wait_contact"
                    w["since_outside"] = 0
                    ee_str = (
                        f"ee_xy={np.asarray(ee_xy).round(4).tolist()}"
                        if ee_xy is not None
                        else "ee_xy=?"
                    )
                    orca_logger.info(
                        f"[入箱检测] {name}: EE xy 已出整箱矩形（{ee_str}），"
                        f"等待工具产生接触"
                    )
                still_active.append(w)
            elif phase == "wait_contact":
                w["since_outside"] = int(w.get("since_outside", 0)) + 1
                status, detail, pos_b = _classify_tool_place_contacts(
                    self.r_arm.env,
                    tool_body,
                    self.base_body_query,
                    self._placed_inbox_tools,
                    self.box_bottom,
                )
                pos_str = (
                    f"pos={np.asarray(pos_b).round(4).tolist()}"
                    if pos_b is not None
                    else "pos=?"
                )
                if status == "waiting":
                    if w["since_outside"] >= self.place_timeout_steps:
                        self._fail_one_tool(
                            tool_idx,
                            f"EE出箱后{w['since_outside']}步仍无接触",
                            detail,
                            pos_b,
                        )
                        return
                    still_active.append(w)
                    continue
                if status == "fail":
                    self._fail_one_tool(
                        tool_idx,
                        f"EE出箱后接触非法",
                        detail,
                        pos_b,
                    )
                    return
                self._placed_inbox_tools.add(tool_body)
                orca_logger.info(
                    f"[入箱检测] ✓ {name} | {detail} | {pos_str} "
                    f"(EE出箱后 {w['since_outside']} 步) | "
                    f"已入箱={len(self._placed_inbox_tools)}"
                )
            else:
                still_active.append(w)
        self._place_watchers = still_active

    def _log_tracking(self, step_idx):
        """查询右臂末端实际 base 系位置，与命令目标对比，打印跟踪误差。"""
        label, cmd = self.seg_bounds[step_idx]
        try:
            ee_b = self.r_arm.env.query_site_pos_and_quat_B(
                [self.r_arm.ee_name], [self.r_arm.base_link]
            )
            actual = np.asarray(ee_b[self.r_arm.ee_name]["xpos"], dtype=np.float64)
            cmd = np.asarray(cmd, dtype=np.float64)
            err = actual - cmd
            err_norm = float(np.linalg.norm(err))
            orca_logger.info(
                f"[跟踪诊断] {label} @step{step_idx} | "
                f"命令={cmd.round(4).tolist()} 实际={actual.round(4).tolist()} "
                f"误差={err.round(4).tolist()} |误差|={err_norm * 1000:.1f}mm"
            )
        except Exception as e:
            orca_logger.warning(f"[跟踪诊断] {label} 查询失败: {e}")

    def update(self):
        if self.t >= len(self.l_pos):
            return
        if self.place_failed:
            # 已判定失败：保持最后指令并确保任务结束
            self._force_task_end()
            self.t = len(self.l_pos)
            return
        if self.t == 0:
            self.task_status.update_task_status(True)
        self._tick_place_monitors()
        if self.place_failed:
            self.t = len(self.l_pos)
            return
        if self.t in self.seg_bounds:
            self._log_tracking(self.t)
        self.l_arm.update_action_position(self.l_pos[self.t])
        self.l_arm.update_action_axisangle(self.l_quat_xyzw[self.t])
        self.r_arm.update_action_position(self.r_pos[self.t])
        self.r_arm.update_action_axisangle(self.r_quat_xyzw[self.t])
        l_n = len(self.l_grip.ctrl_index)
        r_n = len(self.r_grip.ctrl_index)
        self.l_grip.update_ctrl(np.full(l_n, self.l_grip_motor[self.t], dtype=np.float32))
        self.r_grip.update_ctrl(np.full(r_n, self.r_grip_motor[self.t], dtype=np.float32))
        if self.t == len(self.l_pos) - 1:
            self.task_status.update_task_status(True)
        self.t += 1


# ---------------------------------------------------------------------------
# 路点 YAML 加载
# ---------------------------------------------------------------------------

def _load_waypoint_yaml(path: str) -> dict:
    """加载路点 YAML，返回包含 gripper_open/close 和 waypoints 列表的 dict。

    支持 4 点或 6 点（6 = 在原放箱 3/4 前插入两条经由）。
    """
    with open(path, "r", encoding="utf-8") as f:
        spec = safe_load(f)

    g_open = float(spec.get("gripper_open", -0.8561))
    g_close = float(spec.get("gripper_close", 2.0))
    segs = spec.get("segments", [])
    if len(segs) not in (4, 6):
        raise ValueError(f"{path}: 期望 4 或 6 个路点，实际 {len(segs)} 个")

    waypoints = []
    for i, seg in enumerate(segs):
        pos = list(seg["r_target_b"])
        quat = list(seg["r_quat_b"])
        grip = str(seg.get("gripper_r", "open")).strip().lower()
        waypoints.append({"pos": pos, "quat": quat, "grip": grip})

    return {"g_open": g_open, "g_close": g_close, "waypoints": waypoints}


def _load_extra_slot_waypoints(path: str) -> dict:
    """加载槽位路点 YAML。

    返回 dict[key] = {"wp0": {"pos", "quat", "grip"}, "wp1": {...}}
    key 形如 "slot0_tool1"。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = safe_load(f) or {}

    extras: dict = {}
    for key, entry in raw.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue
        if "wp0" not in entry or "wp1" not in entry:
            continue
        parsed = {}
        for wp_name in ("wp0", "wp1"):
            wp = entry[wp_name]
            parsed[wp_name] = {
                "pos": list(wp["pos"]),
                "quat": list(wp["quat"]),
                "grip": str(wp.get("grip", "open")).strip().lower(),
            }
        extras[key] = parsed
    return extras


# 槽位路点 wp 的 Y 与目标槽参考 Y 最大允许偏差（米）。超出则回退平移。
_EXTRA_SLOT_Y_TOL = 0.05


def _apply_extra_slot_waypoints(
    warped: dict,
    slot_idx: int,
    tool_idx: int,
    extras: dict,
) -> tuple[dict, bool]:
    """若存在合法的 slotX_toolY 路点，用其绝对 wp0/wp1 覆盖（放箱路点不变）。

    合法性：wp0/wp1 的 y 须接近该槽位参考 y（|_TOOL_REFERENCE_POS_B[slot].y| 差 < 5cm），
    否则判定录错槽，不覆盖，继续使用平移结果。
    """
    if not extras:
        return warped, False
    key = f"slot{slot_idx}_tool{tool_idx}"
    override = extras.get(key)
    if override is None:
        return warped, False

    slot_y = float(_TOOL_REFERENCE_POS_B[slot_idx, 1])
    for wp_name in ("wp0", "wp1"):
        y = float(override[wp_name]["pos"][1])
        if abs(y - slot_y) > _EXTRA_SLOT_Y_TOL:
            orca_logger.warning(
                f"[槽位路点] 拒绝 {key}：{wp_name}.y={y:.4f} 与 slot{slot_idx} "
                f"参考y={slot_y:.4f} 偏差 {abs(y - slot_y)*1000:.0f}mm > "
                f"{_EXTRA_SLOT_Y_TOL*1000:.0f}mm，回退平移"
            )
            return warped, False

    for wp_idx, wp_name in enumerate(("wp0", "wp1")):
        wp = override[wp_name]
        warped["waypoints"][wp_idx]["pos"] = list(wp["pos"])
        warped["waypoints"][wp_idx]["quat"] = list(wp["quat"])
        warped["waypoints"][wp_idx]["grip"] = wp["grip"]
    return warped, True


def _sample_slot_assignment(rng: np.random.Generator) -> np.ndarray:
    """均匀随机排列 assignment[slot]=tool；手电筒(tool3) 禁止落在 slot4。

    在同一 rng 上拒绝采样，保证合法排列近似均匀（非枚举表）。
    """
    flashlight_idx = 3
    forbidden_slot = 4
    for _ in range(1000):
        assignment = rng.permutation(5)
        flashlight_slot = int(np.where(assignment == flashlight_idx)[0][0])
        if flashlight_slot != forbidden_slot:
            return assignment.astype(np.int64)
    raise RuntimeError("无法采到合法槽位排列（手电筒禁 slot4）")


def _place_tools(env, base_body: str, assignment: np.ndarray) -> None:
    """按槽位排列设置 5 件工具的 free joint。

    assignment[slot_idx] = tool_idx。目标 y 取槽位参考值，x/z/姿态
    保留该工具自身参考值。
    """
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.shape != (5,) or sorted(assignment.tolist()) != list(range(5)):
        raise ValueError(f"assignment 必须是 0..4 的排列，收到: {assignment.tolist()}")

    base_pos, _, base_quat_wxyz = env.get_body_xpos_xmat_xquat([base_body])
    base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)
    base_quat_wxyz = np.asarray(base_quat_wxyz, dtype=np.float64).reshape(4)
    base_rot = R.from_quat(base_quat_wxyz[[1, 2, 3, 0]])

    target_qpos = {}
    for slot_idx, tool_idx in enumerate(assignment):
        target_pos_b = _TOOL_REFERENCE_POS_B[tool_idx].copy()
        target_pos_b[1] = _TOOL_REFERENCE_POS_B[slot_idx, 1]
        target_rot_b = R.from_quat(_TOOL_REFERENCE_QUAT_XYZW_B[tool_idx])

        world_pos = base_pos + base_rot.apply(target_pos_b)
        world_quat_xyzw = (base_rot * target_rot_b).as_quat()
        world_quat_wxyz = world_quat_xyzw[[3, 0, 1, 2]]
        target_qpos[_TOOL_BODY_JOINT_NAMES[tool_idx]] = np.concatenate(
            [world_pos, world_quat_wxyz]
        )

    env.set_joint_qpos(target_qpos)
    env.mj_forward()


def _query_tool_positions_b(env, base_body: str) -> np.ndarray:
    """查询 5 件工具 body 中心在 base 系下的实际位置。"""
    return np.stack(
        [
            np.asarray(env.query_position_body_B(name, base_body), dtype=np.float64)
            for name in _TOOL_BODY_JOINT_NAMES
        ],
        axis=0,
    )


def _copy_tool_spec(tool_spec: dict) -> dict:
    """深拷贝工具路点 dict（避免改到加载缓存）。"""
    return {
        "g_open": tool_spec["g_open"],
        "g_close": tool_spec["g_close"],
        "waypoints": [
            {"pos": list(wp["pos"]), "quat": list(wp["quat"]), "grip": wp["grip"]}
            for wp in tool_spec["waypoints"]
        ],
    }


def _warp_waypoints(tool_spec: dict, delta_b: np.ndarray) -> dict:
    """返回路点副本：仅平移 wp0/wp1；放箱路点（4点 wp2+；6点 wp2..wp5）不变。"""
    delta_b = np.asarray(delta_b, dtype=np.float64).reshape(3)
    warped = _copy_tool_spec(tool_spec)
    for wp_idx in (0, 1):
        pos = np.asarray(warped["waypoints"][wp_idx]["pos"], dtype=np.float64)
        warped["waypoints"][wp_idx]["pos"] = (pos + delta_b).tolist()
    return warped


# 槽位 5 的抓取位置使用夹爪最大开度比例计算 Y 方向补偿。
_GRIPPER_OPEN_WIDTH_M = 0.085
_SLOT5_GRASP_RIGHT_FRAC = 0.5
_SLOT5_IDX = 4  # 最右侧槽位
_SLOT5_GRASP_DY = -_SLOT5_GRASP_RIGHT_FRAC * _GRIPPER_OPEN_WIDTH_M  # 偏右 → y 减小

# 槽位 4 的扳手抓取位置补偿。
_WRENCH_IDX = 0
_SLOT4_IDX = 3
_SLOT4_WRENCH_GRASP_DY = 0.008  # Y +8 mm
_SLOT4_WRENCH_GRASP_DZ = -0.035  # Z -35 mm


def _bias_grasp_waypoints(
    tool_spec: dict, dy: float = 0.0, dz: float = 0.0
) -> dict:
    """仅平移 wp0/wp1 的 y/z（放箱路点不变）。"""
    out = _copy_tool_spec(tool_spec)
    for wp_idx in (0, 1):
        pos = list(out["waypoints"][wp_idx]["pos"])
        pos[1] = float(pos[1]) + float(dy)
        pos[2] = float(pos[2]) + float(dz)
        out["waypoints"][wp_idx]["pos"] = pos
    return out


def _bias_grasp_waypoints_y(tool_spec: dict, dy: float) -> dict:
    """仅平移 wp0/wp1 的 y（放箱路点不变）。"""
    return _bias_grasp_waypoints(tool_spec, dy=dy, dz=0.0)


# ---------------------------------------------------------------------------
# 单工具轨迹构建（4 点 → 9 段；6 点 → 11 段）
# ---------------------------------------------------------------------------

def _build_tool_segments(
    wps,
    safe_z: float,
    steps_transit: int = 90,
    steps_descend: int = 110,
    steps_grasp: int = 55,
    steps_settle: int = 75,
    steps_lift: int = 70,
    steps_place_via: int = 35,
    steps_to_box: int = 70,
    steps_release: int = 40,
    steps_release_settle: int = 55,
    steps_lift_after: int = 55,
):
    """为单个工具生成轨迹 segments。

    wps 支持 4 或 6 个路点 dict（pos/quat/grip）：
      4 点: wp0接近 / wp1抓取 / wp2箱上方 / wp3松开
      6 点: wp0接近 / wp1抓取 / wp2经由1 / wp3经由2 / wp4箱上方 / wp5松开
            （经由插在原放箱 3/4 之前）
    """
    wps = list(wps)
    n = len(wps)
    if n not in (4, 6):
        raise ValueError(f"_build_tool_segments: 期望 4 或 6 个路点，实际 {n}")

    wp0, wp1 = wps[0], wps[1]
    if n == 4:
        via_wps: list = []
        wp_box, wp_rel = wps[2], wps[3]
        box_idx, rel_idx = 2, 3
    else:
        via_wps = [wps[2], wps[3]]
        wp_box, wp_rel = wps[4], wps[5]
        box_idx, rel_idx = 4, 5

    def above(pos, z):
        return [pos[0], pos[1], z]

    segs = [
        {
            "steps": steps_transit,
            "l_hold": True,
            "r_target_b": above(wp0["pos"], safe_z),
            "r_quat_b": wp0["quat"],
            "gripper_r": "open",
            "label": "S1-高位过渡",
        },
        {
            "steps": steps_descend,
            "l_hold": True,
            "r_target_b": wp0["pos"],
            "r_quat_b": wp0["quat"],
            "gripper_r": "open",
            "label": "S2-垂直下降(wp0)",
        },
        {
            "steps": steps_grasp,
            "l_hold": True,
            "r_target_b": wp1["pos"],
            "r_quat_b": wp1["quat"],
            "gripper_r": "open",
            "label": "S3-对准抓取点(wp1)",
        },
        {
            "steps": steps_settle,
            "l_hold": True,
            "r_target_b": wp1["pos"],
            "r_quat_b": wp1["quat"],
            "gripper_r": "close",
            "label": "S4-沉降闭爪(wp1)",
        },
        {
            "steps": steps_lift,
            "l_hold": True,
            "r_target_b": above(wp1["pos"], safe_z),
            "r_quat_b": wp1["quat"],
            "gripper_r": "close",
            "label": "S5-抬升",
        },
    ]

    seg_no = 6
    for vi, via in enumerate(via_wps):
        wp_i = 2 + vi
        segs.append(
            {
                "steps": steps_place_via,
                "l_hold": True,
                "r_target_b": via["pos"],
                "r_quat_b": via["quat"],
                "gripper_r": "close",
                "label": f"S{seg_no}-放箱经由{vi + 1}(wp{wp_i})",
            }
        )
        seg_no += 1

    segs.extend(
        [
            {
                "steps": steps_to_box,
                "l_hold": True,
                "r_target_b": wp_box["pos"],
                "r_quat_b": wp_box["quat"],
                "gripper_r": "close",
                "label": f"S{seg_no}-移到箱上(wp{box_idx})",
            },
            {
                "steps": steps_release,
                "l_hold": True,
                "r_target_b": wp_rel["pos"],
                "r_quat_b": wp_rel["quat"],
                "gripper_r": "close",
                "label": f"S{seg_no + 1}-逼近松开位(wp{rel_idx})",
            },
            {
                "steps": steps_release_settle,
                "l_hold": True,
                "r_target_b": wp_rel["pos"],
                "r_quat_b": wp_rel["quat"],
                "gripper_r": "open",
                "label": f"S{seg_no + 2}-沉降松开(wp{rel_idx})",
                "is_place_release_settle": True,
            },
            {
                "steps": steps_lift_after,
                "l_hold": True,
                "r_target_b": above(wp_rel["pos"], safe_z),
                "r_quat_b": wp_rel["quat"],
                "gripper_r": "open",
                "label": f"S{seg_no + 3}-松开后抬升",
            },
        ]
    )
    return segs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="G1 OmniPicker 工具整理脚本化自动采集 → LeRobot v2.1 格式"
    )
    parser.add_argument("--level", type=str, default="default")
    parser.add_argument("--task_config", type=str, default="../common/example.yaml")
    parser.add_argument("--lerobot_out", type=str, required=True)
    parser.add_argument("--repo_id", default="local/g1_omnipicker_tool")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--waypoint_files",
        type=str,
        default=",".join(
            os.path.join(base_dir, f"my_waypoint_tool{i}.yaml") for i in range(1, 6)
        ),
        help="5 个路点 YAML 路径，逗号分隔，顺序即抓取顺序（默认 my_waypoint_tool1..5.yaml）",
    )
    parser.add_argument("--num_episodes", type=int, default=1, help="采集轮数（默认1）")
    parser.add_argument("--task", type=str, default="整理工具", help="语言指令（默认：整理工具）")
    parser.add_argument(
        "--randomize", action="store_true",
        help="每个 episode 用 RNG 随机排列槽位（手电筒禁最右侧）；抓取按槽位左→右",
    )
    parser.add_argument(
        "--extra_slot_waypoints",
        type=str,
        default=os.path.join(base_dir, "my_slot_waypoints.yaml"),
        help="槽位路点 YAML。命中且 Y 与槽位一致时用绝对 wp0/wp1；传空字符串禁用",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机化基础种子；episode N 用 seed+N。默认不设=每次运行用时间种子（真随机）",
    )
    parser.add_argument(
        "--safe_z", type=float, default=0.50,
        help="安全过渡高度 base 系 z（米，默认 0.50）"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="轨迹整体提速倍率（默认1.0；1.5=快1.5倍，任务时长与数据时长同步缩短）",
    )
    parser.add_argument(
        "--num_tools", type=int, default=5,
        help="本集实际抓取的工具数量（默认5；调试时可设 1 只验证第一把）",
    )
    # 单位均为「帧」（1/fps 秒）；内部按 env.dt 自动放大为控制步。
    # 默认偏快：抓取仍留足沉降；放箱侧（尤其经由/松开，wp 几乎重合）尽量压缩。
    parser.add_argument(
        "--steps_transit", type=int, default=90,
        help="高位过渡帧数（默认90；内部按 env.dt 放大为控制步）",
    )
    parser.add_argument(
        "--steps_descend", type=int, default=110,
        help="垂直下降帧数（默认110；内部按 env.dt 放大为控制步）",
    )
    parser.add_argument(
        "--steps_grasp", type=int, default=55,
        help="移到抓取点帧数（默认55；内部按 env.dt 放大为控制步）",
    )
    parser.add_argument(
        "--steps_settle", type=int, default=75,
        help="抓取点沉降+闭爪驻留帧数（默认75；内部按 env.dt 放大为控制步）",
    )
    parser.add_argument(
        "--steps_lift", type=int, default=70,
        help="抬升帧数（默认70；内部按 env.dt 放大为控制步）",
    )
    parser.add_argument(
        "--steps_place_via",
        type=int,
        default=35,
        help="6 点模式：两条放箱经由段各自帧数（默认35≈1.8s@20fps；内部按 env.dt 放大；4 点 YAML 忽略）",
    )
    parser.add_argument(
        "--steps_to_box", type=int, default=70,
        help="移到工具箱上方帧数（默认70；内部按 env.dt 放大为控制步）",
    )
    parser.add_argument(
        "--steps_release",
        type=int,
        default=40,
        help="闭爪逼近松开位帧数（默认40；内部按 env.dt 放大；当前 wp 箱上/松开几乎重合）",
    )
    parser.add_argument(
        "--steps_release_settle", type=int, default=55,
        help="松开位沉降+张开驻留帧数（默认55；内部按 env.dt 放大为控制步）",
    )
    parser.add_argument(
        "--steps_lift_after", type=int, default=55,
        help="放置后抬升帧数（默认55；内部按 env.dt 放大为控制步）",
    )
    parser.add_argument(
        "--kp", type=float, default=220.0,
        help="OSC 阻抗刚度 kp（默认220）",
    )
    parser.add_argument(
        "--check_box_bottom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="该工具路点结束后，待 EE xy 出整箱矩形且产生接触，再按白名单检查",
    )
    parser.add_argument(
        "--place_check_timeout_s",
        type=float,
        default=_PLACE_CHECK_TIMEOUT_S,
        help=f"EE xy 出整箱后等待接触的最长时间（秒，默认{_PLACE_CHECK_TIMEOUT_S}）",
    )
    parser.add_argument(
        "--max_place_retries",
        type=int,
        default=8,
        help="单集入箱失败最大重试次数（默认8，每次可换 seed）",
    )
    parser.add_argument(
        "--clock",
        choices=("sim", "wall"),
        default="sim",
        help="采帧时钟源（sim=仿真时间，脚本化采集推荐；wall=墙钟）",
    )
    args = parser.parse_args()

    # ── 加载 5 个路点 YAML ──────────────────────────────────────────────────
    yaml_paths = [p.strip() for p in args.waypoint_files.split(",")]
    if len(yaml_paths) != 5:
        orca_logger.error(f"--waypoint_files 需要 5 个路径，收到 {len(yaml_paths)} 个")
        return

    tool_data: list[dict] = []
    g_open_global = -0.8561
    g_close_global = 2.0
    for i, path in enumerate(yaml_paths):
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            orca_logger.error(f"路点文件不存在: {path}")
            return
        td = _load_waypoint_yaml(path)
        tool_data.append(td)
        if i == 0:
            g_open_global = td["g_open"]
            g_close_global = td["g_close"]
        n_wp = len(td["waypoints"])
        orca_logger.info(
            f"工具 {i+1} ({_TOOL_NAMES[i]}): {path} | {n_wp} 点\n"
            f"  wp0={td['waypoints'][0]['pos']}  wp1={td['waypoints'][1]['pos']}  "
            f"release={td['waypoints'][-1]['pos']}"
        )

    # 槽位路点：加载后预检 Y 是否对齐槽位；不合格的 key 直接剔除
    extra_slot_waypoints: dict = {}
    if args.extra_slot_waypoints:
        extra_path = os.path.abspath(os.path.expanduser(args.extra_slot_waypoints))
        if os.path.isfile(extra_path):
            raw_extras = _load_extra_slot_waypoints(extra_path)
            for key, entry in raw_extras.items():
                try:
                    slot_idx = int(key.split("_")[0][4:])
                except (IndexError, ValueError):
                    orca_logger.warning(f"[槽位路点] 无法解析 key，忽略: {key}")
                    continue
                slot_y = float(_TOOL_REFERENCE_POS_B[slot_idx, 1])
                bad = False
                for wp_name in ("wp0", "wp1"):
                    y = float(entry[wp_name]["pos"][1])
                    if abs(y - slot_y) > _EXTRA_SLOT_Y_TOL:
                        orca_logger.warning(
                            f"[槽位路点] 剔除 {key}：{wp_name}.y={y:.4f} 偏离 "
                            f"slot{slot_idx} 参考y={slot_y:.4f} "
                            f"({abs(y - slot_y)*1000:.0f}mm)，请重新录制"
                        )
                        bad = True
                        break
                if not bad:
                    extra_slot_waypoints[key] = entry
            orca_logger.info(
                f"槽位路点: {extra_path} → 有效 {len(extra_slot_waypoints)}/"
                f"{len(raw_extras)} 条 ({', '.join(sorted(extra_slot_waypoints.keys())) or '无'})"
            )
            print(
                f"  槽位路点: 有效 {len(extra_slot_waypoints)}/{len(raw_extras)} 条",
                flush=True,
            )
        else:
            orca_logger.warning(f"槽位路点文件不存在，跳过: {extra_path}")
            print(f"  ⚠ 槽位路点文件不存在，跳过: {extra_path}", flush=True)

    lerobot_out = os.path.abspath(os.path.expanduser(args.lerobot_out))
    num_episodes = args.num_episodes

    # ── 配置与环境初始化 ────────────────────────────────────────────────────
    from conf import g1_omnipicker_conf as agent_conf

    default_joint_values: dict = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], _L_INIT_JOINT_VALUES):
        default_joint_values[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = v

    print("=" * 62, flush=True)
    print("  G1 OmniPicker 工具整理自动化采集", flush=True)
    print(f"  任务: {args.task}", flush=True)
    print(f"  工具顺序: {' → '.join(_TOOL_NAMES)}", flush=True)
    print(f"  随机化: {'开' if args.randomize else '关'}", flush=True)
    print(f"  安全高度: {args.safe_z} m (base系z)", flush=True)
    print(f"  轮数: {num_episodes}", flush=True)
    print(f"  输出目录: {lerobot_out}", flush=True)
    print("=" * 62, flush=True)

    orca_logger.info("Creating scene manager")
    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.show_ui_message(1, "脚本控制：G1 工具整理自动化采集", "0xffff00", showtime=5)
    scene_manager.get_scene_data(script_name, "beginscene")

    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", "g1_omnipicker_tool", args.level)
    storage = G1OmniPickerLeRobotStorage(dataset_path=scratch_dir)

    _n_motor = (
        len(agent_conf.gripper_l["actuator_names"])
        + len(agent_conf.gripper_r["actuator_names"])
    )

    def _obs_callback_safe(env):
        if env.model.nu == 0:
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(_n_motor, dtype=np.float32),
            }
        return storage.obs_callback(env)

    orca_logger.info("Creating DataCollectionManager")
    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback_safe,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False

    # 轨迹 --steps_* 以「帧」(1/fps) 设计；控制步长 env.dt = time_step×frame_skip。
    # steps_scale 把帧数放大为控制步，保证末端速度与数据时长 = 任务时长。
    env_dt = float(manager.time_step) * int(manager.frame_skip)
    frame_dt = 1.0 / float(args.fps)
    steps_scale = max(1, int(round(frame_dt / env_dt)))
    scaled_dt = steps_scale * env_dt
    orca_logger.info(
        f"[STEPS-SCALE] env.dt={env_dt*1000:.1f}ms "
        f"(time_step={manager.time_step} frame_skip={manager.frame_skip}) "
        f"frame_dt={frame_dt*1000:.1f}ms (@{args.fps}fps) "
        f"→ steps_scale={steps_scale} "
        f"(scaled_dt={scaled_dt*1000:.1f}ms)"
    )
    if abs(scaled_dt - frame_dt) > 1e-6:
        orca_logger.warning(
            f"[STEPS-SCALE] scaled_dt={scaled_dt*1000:.2f}ms 与 "
            f"1/fps={frame_dt*1000:.2f}ms 偏差 "
            f"{abs(scaled_dt - frame_dt)*1000:.2f}ms，数据时长可能略偏"
        )
    print(
        f"  [STEPS-SCALE] env.dt={env_dt*1000:.1f}ms → "
        f"steps_scale={steps_scale}（帧→控制步）",
        flush=True,
    )

    traj_speed = max(0.1, float(args.speed))
    num_tools = int(np.clip(args.num_tools, 1, 5))

    def _ctrl_steps(frames: int) -> int:
        """轨迹帧数（1/fps）→ 控制步数：按 env.dt 放大，再按 speed 提速。"""
        return max(1, int(round(float(frames) * steps_scale / traj_speed)))

    if traj_speed != 1.0 or num_tools != 5:
        orca_logger.info(
            f"[轨迹] speed={traj_speed:.2f}x，抓取工具数={num_tools}/5"
        )
        print(
            f"  [轨迹] speed={traj_speed:.2f}x | 抓取 {num_tools}/5 把工具",
            flush=True,
        )

    # ── 首次初始化 ──────────────────────────────────────────────────────────
    env.reset()
    time.sleep(0.1)

    if not manager.update_scene():
        orca_logger.error("首次 update_scene 失败，退出")
        env.close()
        return

    env.set_default_joint_values(default_joint_values)
    manager.set_disable_actuator_group([agent_conf.positions_group])

    # 箱底内侧标定（Base 最大薄板 geom + 可选已入箱工具公共面）
    base_body_query = env.body(agent_conf.base_body)
    box_bottom = None
    if args.check_box_bottom:
        try:
            box_bottom = _calibrate_box_bottom_inner(env, base_body_query)
        except Exception as e:
            orca_logger.error(f"箱底标定失败，关闭入箱检测: {e}")
            print(f"  ⚠ 箱底标定失败，关闭入箱检测: {e}", flush=True)

    ctrl_l_name = [env.actuator(m) for m in agent_conf.l_arm["motors_names"]]
    ctrl_r_name = [env.actuator(m) for m in agent_conf.r_arm["motors_names"]]
    init_l = {n: v for n, v in zip(ctrl_l_name, agent_conf.l_arm["motors_init_ctrl"])}
    init_r = {n: v for n, v in zip(ctrl_r_name, agent_conf.r_arm["motors_init_ctrl"])}
    l_arm = create_arm_osc_controller(env, agent_conf.l_arm, agent_conf.base_body, ctrl_l_name, init_l)
    r_arm = create_arm_osc_controller(env, agent_conf.r_arm, agent_conf.base_body, ctrl_r_name, init_r)

    kp_val = float(np.clip(args.kp, 1.0, 300.0))
    for _arm in (l_arm, r_arm):
        _arm.controller.kp = np.ones(6, dtype=np.float64) * kp_val
        _arm.controller.kd = 2.0 * np.sqrt(_arm.controller.kp)
    orca_logger.info(f"OSC 阻抗刚度 kp 设为 {kp_val}（kd=2√kp 临界阻尼）")

    l_gname = [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]]
    r_gname = [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]]
    init_lg = {n: v for n, v in zip(l_gname, agent_conf.gripper_l["init_ctrl"])}
    init_rg = {n: v for n, v in zip(r_gname, agent_conf.gripper_r["init_ctrl"])}
    l_grip = create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_l, agent_conf.base_body, l_gname, init_lg,
        Controller2F85Reverse.ControllerType.DATA,
    )
    r_grip = create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_r, agent_conf.base_body, r_gname, init_rg,
        Controller2F85Reverse.ControllerType.DATA,
    )

    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    task_status = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    manager.set_task_status_controller(task_status)
    manager.set_task(EmptyTask(env))

    # ── 相机初始化 ──────────────────────────────────────────────────────────
    cameras: dict = {}
    cam_hw = DEFAULT_HW
    camera_map = DEFAULT_CAMERA_MAP

    try:
        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
        env.begin_save_video(STREAM_TRIGGER_PATH)
        orca_logger.info("begin_save_video 已调用，触发相机推流")
        cameras = bring_up_cameras(camera_map)
        camera_map = {n: v for n, v in camera_map.items() if n in cameras}
        if cameras:
            cam_hw = probe_camera_hw(cameras, camera_map)
    except Exception as e:
        orca_logger.error(f"相机初始化失败: {e}")

    if not cameras:
        orca_logger.error("没有可用相机，退出")
        env.close()
        return

    cam_shape = (3, cam_hw[0], cam_hw[1])
    orca_logger.info(f"相机分辨率 {cam_hw[0]}x{cam_hw[1]}，fps={args.fps}")

    writer = LeRobotDatasetWriter.create(
        repo_id=args.repo_id,
        root=lerobot_out,
        fps=args.fps,
        camera_map=camera_map,
        state_dim=storage.state_dim,
        state_names=storage.state_names,
        cam_shape=cam_shape,
        resume=args.resume,
        robot_type="g1_omnipicker",
    )

    storage.configure_lerobot(
        fps=args.fps,
        cameras=cameras,
        camera_map=camera_map,
        target_hw=cam_hw,
        writer=writer,
        task=args.task,
        clock=args.clock,
    )

    orca_logger.info(f"开始采集，共 {num_episodes} 轮，任务: {args.task}，输出: {lerobot_out}")

    # 随机化基础种子：未指定 --seed 时用时间，保证每次运行不同；指定则整次运行可复现
    randomize_base_seed: int | None = None
    if args.randomize:
        randomize_base_seed = (
            int(args.seed)
            if args.seed is not None
            else int(time.time() * 1000) % (2**31 - 1)
        )
        orca_logger.info(
            f"[槽位随机化] base_seed={randomize_base_seed} "
            f"（episode N 使用 base_seed+N；手电筒禁 slot4）"
        )
        print(f"  随机种子 base_seed={randomize_base_seed}", flush=True)

    n_success = 0
    place_retry = 0  # 当前目标集的入箱失败重试计数

    try:
        with writer:
            while n_success < num_episodes:
                ep_display = n_success + 1
                orca_logger.info(
                    f"\n=== Episode {ep_display}/{num_episodes} "
                    f"(retry={place_retry}) | {args.task} ==="
                )
                print(
                    f"\n>>> 正在采集第 {ep_display}/{num_episodes} 轮"
                    f"{f'（入箱重试 {place_retry}）' if place_retry else ''} "
                    f"| 任务: {args.task}",
                    flush=True,
                )

                try:
                    scene_manager.show_ui_message(
                        1, f"采集中: {args.task}  ({ep_display}/{num_episodes})",
                        "0x00ff88", showtime=0
                    )
                except Exception:
                    pass

                env.reset()
                time.sleep(0.05)

                if not manager.update_scene():
                    orca_logger.info("update_scene 失败，停止")
                    break

                env.set_default_joint_values(default_joint_values)
                base_body = env.body(agent_conf.base_body)

                # ── 本集工具路点与抓取顺序 ─────────────────────────────────
                episode_tool_data = [_copy_tool_spec(td) for td in tool_data]
                grasp_order = list(range(5))  # 非随机：按工具编号 0..4
                episode_seed = None
                slot_for_tool = {i: i for i in range(5)}

                if args.randomize:
                    assert randomize_base_seed is not None
                    # 成功集用 base+n_success；失败重试用大步长错开，可换排列
                    episode_seed = (
                        int(randomize_base_seed)
                        + n_success
                        + place_retry * 10007
                    ) % (2**31 - 1)
                    rng_ep = np.random.default_rng(episode_seed)
                    assignment = _sample_slot_assignment(rng_ep)

                    _place_tools(env, base_body, assignment)
                    actual_pos_b = _query_tool_positions_b(env, base_body)
                    delta_b = actual_pos_b - _TOOL_REFERENCE_POS_B

                    slot_for_tool = {
                        int(tool_idx): int(slot_idx)
                        for slot_idx, tool_idx in enumerate(assignment)
                    }

                    episode_tool_data = []
                    override_hits: list[str] = []
                    for tool_idx in range(5):
                        slot_idx = slot_for_tool[tool_idx]
                        warped = _warp_waypoints(tool_data[tool_idx], delta_b[tool_idx])
                        warped, used_extra = _apply_extra_slot_waypoints(
                            warped, slot_idx, tool_idx, extra_slot_waypoints
                        )
                        if used_extra:
                            override_hits.append(
                                f"{_TOOL_NAMES[tool_idx]}@slot{slot_idx}"
                            )
                        episode_tool_data.append(warped)

                    grasp_order = [int(t) for t in assignment.tolist()]

                    layout = ", ".join(
                        f"slot{s}={_TOOL_NAMES[t]}" for s, t in enumerate(grasp_order)
                    )
                    pick_seq = " → ".join(_TOOL_NAMES[t] for t in grasp_order)
                    orca_logger.info(
                        f"[槽位随机化] ep={ep_display} seed={episode_seed} | {layout}"
                    )
                    orca_logger.info(f"[抓取顺序] {pick_seq}")
                    print(f"  布局: {layout}", flush=True)
                    print(f"  抓取: {pick_seq} | seed={episode_seed}", flush=True)
                    if override_hits:
                        msg = ", ".join(override_hits)
                        orca_logger.info(f"[槽位路点] 采用绝对 wp0/wp1: {msg}")
                        print(f"  [槽位路点] {msg}", flush=True)
                    for tool_idx in range(5):
                        slot_idx = slot_for_tool[tool_idx]
                        src = (
                            "路点"
                            if f"{_TOOL_NAMES[tool_idx]}@slot{slot_idx}" in override_hits
                            else "平移"
                        )
                        orca_logger.info(
                            f"  {_TOOL_NAMES[tool_idx]}->slot{slot_idx} [{src}] "
                            f"deltaY={delta_b[tool_idx, 1]*1000:.1f}mm "
                            f"wp1={episode_tool_data[tool_idx]['waypoints'][1]['pos']}"
                        )

                # 槽位5（最右）：抓取 wp0/wp1 再偏右
                tool_at_slot5 = int(grasp_order[_SLOT5_IDX])
                before_y = float(
                    episode_tool_data[tool_at_slot5]["waypoints"][1]["pos"][1]
                )
                episode_tool_data[tool_at_slot5] = _bias_grasp_waypoints_y(
                    episode_tool_data[tool_at_slot5], _SLOT5_GRASP_DY
                )
                after_y = float(
                    episode_tool_data[tool_at_slot5]["waypoints"][1]["pos"][1]
                )
                orca_logger.info(
                    f"[槽位5偏置] {_TOOL_NAMES[tool_at_slot5]} wp0/wp1 "
                    f"y += {_SLOT5_GRASP_DY*1000:.1f}mm "
                    f"(0.5×{_GRIPPER_OPEN_WIDTH_M*1000:.0f}mm 张开，偏右) "
                    f"wp1.y {before_y:.4f}→{after_y:.4f}"
                )
                print(
                    f"  [槽位5偏置] {_TOOL_NAMES[tool_at_slot5]} "
                    f"抓取 y {_SLOT5_GRASP_DY*1000:+.1f}mm（偏右）",
                    flush=True,
                )

                # 槽位 4 扳手抓取路点补偿。
                if int(slot_for_tool.get(_WRENCH_IDX, -1)) == _SLOT4_IDX:
                    w_before = episode_tool_data[_WRENCH_IDX]["waypoints"][1]["pos"]
                    episode_tool_data[_WRENCH_IDX] = _bias_grasp_waypoints(
                        episode_tool_data[_WRENCH_IDX],
                        dy=_SLOT4_WRENCH_GRASP_DY,
                        dz=_SLOT4_WRENCH_GRASP_DZ,
                    )
                    w_after = episode_tool_data[_WRENCH_IDX]["waypoints"][1]["pos"]
                    orca_logger.info(
                        f"[槽位4扳手偏置] wp0/wp1 "
                        f"y += {_SLOT4_WRENCH_GRASP_DY*1000:.1f}mm（左） "
                        f"z += {_SLOT4_WRENCH_GRASP_DZ*1000:.1f}mm（下） "
                        f"wp1 {list(np.asarray(w_before).round(4))}→"
                        f"{list(np.asarray(w_after).round(4))}"
                    )
                    print(
                        f"  [槽位4扳手偏置] 抓取 "
                        f"y {_SLOT4_WRENCH_GRASP_DY*1000:+.1f}mm（左） "
                        f"z {_SLOT4_WRENCH_GRASP_DZ*1000:+.1f}mm（下）",
                        flush=True,
                    )

                all_segments: list[dict] = []
                place_arm_at: dict[int, int] = {}  # 该工具全部路点结束步 → tool_idx
                # place_check_timeout_s × fps = 帧数；再 × steps_scale = 控制步
                place_timeout_steps = max(
                    1,
                    int(round(float(args.place_check_timeout_s) * args.fps))
                    * steps_scale,
                )
                retreat_steps = _ctrl_steps(_PLACE_RETREAT_STEPS)
                cum_steps = 0
                last_release_quat = None
                last_release_pos = None
                active_order = grasp_order[:num_tools]
                for slot_idx, tool_idx in enumerate(active_order):
                    td = episode_tool_data[tool_idx]
                    tool_segs = _build_tool_segments(
                        wps=td["waypoints"],
                        safe_z=args.safe_z,
                        steps_transit=_ctrl_steps(args.steps_transit),
                        steps_descend=_ctrl_steps(args.steps_descend),
                        steps_grasp=_ctrl_steps(args.steps_grasp),
                        steps_settle=_ctrl_steps(args.steps_settle),
                        steps_lift=_ctrl_steps(args.steps_lift),
                        steps_place_via=_ctrl_steps(args.steps_place_via),
                        steps_to_box=_ctrl_steps(args.steps_to_box),
                        steps_release=_ctrl_steps(args.steps_release),
                        steps_release_settle=_ctrl_steps(args.steps_release_settle),
                        steps_lift_after=_ctrl_steps(args.steps_lift_after),
                    )
                    for s in tool_segs:
                        if args.randomize:
                            prefix = (
                                f"槽位{slot_idx + 1}-"
                                f"工具{tool_idx + 1}({_TOOL_NAMES[tool_idx]})"
                            )
                        else:
                            prefix = f"工具{tool_idx + 1}({_TOOL_NAMES[tool_idx]})"
                        s["label"] = f"{prefix}-{s['label']}"
                    tool_start = cum_steps
                    last_release_pos = list(td["waypoints"][-1]["pos"])
                    last_release_quat = list(td["waypoints"][-1]["quat"])
                    tool_steps = sum(int(s["steps"]) for s in tool_segs)
                    # 该工具全部路点结束后启动监视：等 EE xy 出整箱矩形再查接触
                    if args.check_box_bottom and box_bottom is not None:
                        tool_end = tool_start + tool_steps - 1
                        place_arm_at[tool_end] = int(tool_idx)
                    all_segments.extend(tool_segs)
                    cum_steps += tool_steps
                    orca_logger.info(
                        f"  拼段 {len(tool_segs)} | {_TOOL_NAMES[tool_idx]} | "
                        f"{len(td['waypoints'])}点 | 接近位={td['waypoints'][0]['pos']}"
                    )

                # 末工具后追加撤离+驻留，保证 EE xy 能出整箱矩形并留接触检查窗口
                if (
                    args.check_box_bottom
                    and box_bottom is not None
                    and last_release_pos is not None
                    and last_release_quat is not None
                ):
                    xy_min = np.asarray(
                        box_bottom.get("box_xy_min", box_bottom["xy_min"]),
                        dtype=np.float64,
                    )
                    xy_max = np.asarray(
                        box_bottom.get("box_xy_max", box_bottom["xy_max"]),
                        dtype=np.float64,
                    )
                    retreat_x = float(0.5 * (xy_min[0] + xy_max[0]))
                    retreat_y = float(xy_min[1] - 0.12)
                    retreat_pos = [retreat_x, retreat_y, float(args.safe_z)]
                    all_segments.append(
                        {
                            "steps": retreat_steps,
                            "l_hold": True,
                            "r_target_b": retreat_pos,
                            "r_quat_b": last_release_quat,
                            "gripper_r": "open",
                            "label": "S-撤离箱上方",
                        }
                    )
                    # 末尾只需覆盖「离开后等接触」窗口；检查多在 1 步内完成
                    wait_steps = place_timeout_steps + 2 * steps_scale
                    all_segments.append(
                        {
                            "steps": wait_steps,
                            "l_hold": True,
                            "r_target_b": retreat_pos,
                            "r_quat_b": last_release_quat,
                            "gripper_r": "open",
                            "label": "S-入箱检查等待",
                        }
                    )
                    cum_steps += retreat_steps + wait_steps
                    orca_logger.info(
                        f"  追加撤离+等待: retreat={retreat_pos} "
                        f"retreat_steps={retreat_steps} "
                        f"timeout_steps={place_timeout_steps}"
                    )

                total_steps = cum_steps
                duration_s = total_steps * env_dt
                orca_logger.info(
                    f"总段数: {len(all_segments)}，总步数: {total_steps}，"
                    f"约 {duration_s:.1f}s "
                    f"(env.dt={env_dt*1000:.1f}ms × {total_steps}，"
                    f"steps_scale={steps_scale}，speed={traj_speed:.2f}x，"
                    f"工具={num_tools}/5，@{args.fps}fps)"
                    + (
                        f"；入箱监视 {len(place_arm_at)} 件"
                        f"（EE出箱后等接触，超时 {args.place_check_timeout_s:.1f}s）"
                        if place_arm_at
                        else ""
                    )
                )
                print(
                    f"  轨迹: {len(all_segments)} 段 / {total_steps} 步"
                    f"（约 {duration_s:.1f}s）",
                    flush=True,
                )

                # ── 段边界诊断表 ────────────────────────────────────────────
                seg_bounds = {}
                cum = 0
                for s in all_segments:
                    cum += int(s["steps"])
                    last_idx = cum - 1
                    seg_bounds[last_idx] = (s["label"], list(s["r_target_b"]))

                # ── 离线预构建完整轨迹 ──────────────────────────────────────
                l_pos, l_quat, r_pos, r_quat_traj, l_gm, r_gm = (
                    scripted.build_segmented_trajectory(
                        env, agent_conf, all_segments, g_open_global, g_close_global
                    )
                )

                device = G1ScriptedTrajectoryDevice(
                    l_arm, r_arm, l_grip, r_grip, task_status,
                    l_pos, l_quat, r_pos, r_quat_traj, l_gm, r_gm,
                    seg_bounds=seg_bounds,
                    place_arm_at=place_arm_at if box_bottom is not None else None,
                    box_bottom=box_bottom,
                    base_body_query=base_body,
                    place_timeout_steps=place_timeout_steps,
                )
                manager.set_device(device)
                manager.run_episode()

                if device.place_failed:
                    storage.clear_data()
                    place_retry += 1
                    orca_logger.warning(
                        f"[入箱检测] 丢弃本集并重采 "
                        f"(retry {place_retry}/{args.max_place_retries}) | "
                        f"{device.place_fail_reason}"
                    )
                    print(
                        f">>> [✗] 入箱失败，已删除本集缓存，"
                        f"准备重采 ({place_retry}/{args.max_place_retries})",
                        flush=True,
                    )
                    if place_retry >= args.max_place_retries:
                        orca_logger.error(
                            f"入箱重试超过 {args.max_place_retries} 次，停止采集"
                        )
                        print(">>> 入箱重试次数耗尽，停止", flush=True)
                        break
                    # 随机化时换 seed；非随机仅重跑同一轨迹
                    if args.randomize:
                        orca_logger.info(
                            f"[入箱检测] 更换 seed 重试 "
                            f"(next offset retry={place_retry})"
                        )
                    continue

                storage.save_data(
                    task_info=manager.task.get_task_info(),
                    scene_info=manager.scene_manager.get_scene_info(),
                    task_description=manager.task.get_task_description(),
                )
                n_success += 1
                place_retry = 0
                orca_logger.info(
                    f"[✓] Episode {n_success}/{num_episodes} 保存完毕"
                    f"（共 {writer.num_frames} 帧）"
                )
                print(f">>> [✓] Episode {n_success}/{num_episodes} 已保存", flush=True)

    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止采集")
        print("\n[停止] 采集已中断", flush=True)
    except Exception as e:
        orca_logger.error(f"采集异常: {e}\n{traceback.format_exc()}")
    finally:
        try:
            env.stop_save_video()
        except Exception:
            pass
        close_cameras(cameras)
        summary = f"采集结束，共 {writer.num_episodes} 集 / {writer.num_frames} 帧"
        orca_logger.info(summary)
        print(f"\n{'=' * 62}", flush=True)
        print(f"  {summary}", flush=True)
        print(f"  数据位于: {lerobot_out}", flush=True)
        print(f"{'=' * 62}", flush=True)
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as e:
        OrcaLog.get_instance().error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        orca_logger.info("Exiting program")
        os._exit(0)
