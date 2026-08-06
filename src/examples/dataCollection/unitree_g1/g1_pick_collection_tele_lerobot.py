"""宇树 G1 VR 遥操作数据采集。"""
import argparse
import dataclasses
import os
import sys
import threading
import time
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_ORCA_MANIP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DEFAULT_LEROBOT_ROOT = os.path.join(_ORCA_MANIP_ROOT, "L_dataset", "unitree")
DEFAULT_LEROBOT_OUT = os.path.join(DEFAULT_LEROBOT_ROOT, "g1_unitree_button")


def resolve_lerobot_out(path: str) -> str:
    """将相对路径解析到默认数据集目录，并展开绝对路径。"""
    p = os.path.expanduser(str(path).strip())
    if not os.path.isabs(p):
        p = os.path.join(DEFAULT_LEROBOT_ROOT, p)
    return os.path.abspath(p)

import numpy as np
from yaml import Loader, load

from conf import g1_pick_conf
from controllers import controllers
from controllers.controller_task import TaskStatus
from controllers.controller_task import TaskStatusController
from controllers.g1_pick_dual_arm_controller import G1PickDualArmIKController
from controllers.g1_pick_unitree_arm_ik import G1_29_ArmIK
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.g1_pick_data_storage import G1PickLeRobotStorage
from dataStorage.lerobot_data_storage import LeRobotDatasetWriter
from devices.g1_pick_device import G1PickPicoJoystickDevice
from devices.g1_pick_tv_pose_mapper import TvToOrcaPoseMapper, make_trans_x
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_pick_lerobot_stream"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="G1PickLeRobot",
    log_file="g1_pick_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


# =============================================================================
# 灵巧手控制器
# =============================================================================
class HandController:
    """灵巧手控制器 — 将 VR 扳机值映射为所有手指关节的开/合位置。

    全部使用 position 执行器，直接写入 qpos 值。
    扳机=0 → 手张开（open_positions），扳机=1 → 手握紧（close_positions）。
    若 hand_config.pin_thumb_fold=True：拇指 3 关节全程锁在闭合限位（不随扳机）。
    """

    def __init__(self, env, hand_config: dict, ctrl_name: list[str],
                 init_ctrl: dict[str, float], hand_side: str):
        self.env = env
        self.ctrl_name = ctrl_name
        self.init_ctrl = init_ctrl
        self.ctrl_index = self.init_ctrl_index()

        self.open_positions = np.array(hand_config["positions_init_ctrl"], dtype=np.float32)
        ranges = np.array(hand_config["positions_ranges"], dtype=np.float32)
        self.close_positions = np.where(
            np.abs(ranges[:, 1]) > np.abs(ranges[:, 0]),
            ranges[:, 1], ranges[:, 0]
        ).astype(np.float32)

        # 拇指折叠锁定：open/close 的拇指目标一致 → 扳机只驱动食指/中指
        self._pin_thumb_fold = bool(hand_config.get("pin_thumb_fold", False))
        if self._pin_thumb_fold and len(self.open_positions) >= 3:
            self.open_positions[:3] = self.close_positions[:3]
            # 同步 init_ctrl，避免 episode 起步瞬间拇指张开
            for i, name in enumerate(self.ctrl_name[:3]):
                self.init_ctrl[name] = float(self.close_positions[i])

        self._trigger_value = 0.0
        self.side = hand_side

    def init_ctrl_index(self) -> list[int]:
        return [self.env.model.actuator_name2id(n) for n in self.ctrl_name]

    def get_init_ctrl(self) -> dict[int, float]:
        return {self.env.model.actuator_name2id(n): self.init_ctrl[n]
                for n in self.ctrl_name if n in self.init_ctrl}

    def reset(self):
        pass

    def update_trigger_value(self, value: float):
        self._trigger_value = float(value)

    def run_controller(self) -> dict[int, float]:
        t = np.clip(self._trigger_value, 0.0, 1.0)
        positions = self.open_positions + t * (self.close_positions - self.open_positions)
        if self._pin_thumb_fold and len(positions) >= 3:
            positions[:3] = self.close_positions[:3]
        return {self.ctrl_index[i]: float(positions[i]) for i in range(len(self.ctrl_index))}


def create_hand_pico_controller(
    manager: DataCollectionManager,
    env,
    hand_config: dict,
    device: G1PickPicoJoystickDevice,
    trigger_key: PicoJoystickKey,
    hand_side: str,
):
    """创建灵巧手 VR 控制器，用扳机值映射开合。"""
    ctrl_name = [env.actuator(n) for n in hand_config["positions_names"]]
    init_ctrl = {n: v for n, v in zip(ctrl_name, hand_config["positions_init_ctrl"])}
    hc = HandController(env, hand_config, ctrl_name, init_ctrl, hand_side)
    device.bind_trigger_event(trigger_key, hc.update_trigger_value)
    manager.add_controller(hc)


# =============================================================================
# 腿部 + 腰部锁定控制器（hold 在初始 qpos）
# =============================================================================
class JointHoldController:
    """将指定关节锁定在给定位置，每 episode 重置时重新读取 qpos。"""

    def __init__(self, env, ctrl_name: list[str], init_positions: np.ndarray,
                 joint_names: list[str]):
        self.env = env
        self.ctrl_name = ctrl_name
        self._joint_names = joint_names
        self._joint_ids = [env.joint(n) for n in joint_names]
        self.ctrl_index = self.init_ctrl_index()
        self.hold_positions = np.asarray(
            [self._as_scalar(v) for v in np.asarray(init_positions).reshape(-1)],
            dtype=np.float32,
        )
        self.init_ctrl = self._build_init_ctrl()

    @staticmethod
    def _as_scalar(v) -> float:
        return float(np.asarray(v, dtype=np.float64).reshape(-1)[0])

    def _build_init_ctrl(self) -> dict[int, float]:
        return {
            self.env.model.actuator_name2id(n): self._as_scalar(self.hold_positions[i])
            for i, n in enumerate(self.ctrl_name)
        }

    def init_ctrl_index(self) -> list[int]:
        return [self.env.model.actuator_name2id(n) for n in self.ctrl_name]

    def get_init_ctrl(self) -> dict[int, float]:
        return self._build_init_ctrl()

    def reset(self):
        """每 episode 重新读取当前 qpos 并更新 hold 值。"""
        qpos = self.env.query_joint_qpos(self._joint_ids)
        self.hold_positions = np.array(
            [self._as_scalar(qpos[j]) for j in self._joint_ids], dtype=np.float32
        )
        self.init_ctrl = self._build_init_ctrl()

    def run_controller(self) -> dict[int, float]:
        return {
            self.ctrl_index[i]: self._as_scalar(self.hold_positions[i])
            for i in range(len(self.ctrl_index))
        }


def apply_arm_position_gains(
    env,
    kp: float = 150.0,
    kv: float | None = None,
    wrist_kp: float | None = None,
    kv_ratio: float = 0.11,
) -> None:
    """Override MuJoCo <position> arm gains on the live model.

    Scene prefab XML may ignore local g1_pick_robot.xml, so this writes
    env.gym._mjModel directly. Gravity sag is handled separately by injecting
    gravcomp="1" into the scene XML before model compilation (see
    _patched_load_model_xml); raising kp alone is no longer the primary remedy.

    kv defaults to kv_ratio*kp (≈0.085 @ kp=250 keeps prior ζ from 0.11@kp=150),
    not the legacy underdamped 0.0637*kp.
    """
    import mujoco

    kp = float(kp)
    if kv is None:
        kv = float(kv_ratio) * kp
    else:
        kv = float(kv)
    wrist_kp = float(wrist_kp if wrist_kp is not None else kp * 1.125)
    # Keep wrist damping ratio consistent with proximal joints.
    wrist_kv = kv * (wrist_kp / max(kp, 1e-6))

    gym = getattr(env, "gym", None) or getattr(getattr(env, "unwrapped", env), "gym", None)
    mj = getattr(gym, "_mjModel", None) if gym is not None else None
    if mj is None:
        orca_logger.warning("[ARM-GAIN] env.gym._mjModel unavailable; kp not applied")
        return

    prox = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    ]
    wrist = [
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    applied = []
    for short, use_kp, use_kv in (
        *[(n, kp, kv) for n in prox],
        *[(n, wrist_kp, wrist_kv) for n in wrist],
    ):
        full = env.actuator(short)
        aid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
        if aid < 0:
            orca_logger.warning(f"[ARM-GAIN] actuator not found: {full}")
            continue
        old_kp = float(mj.actuator_gainprm[aid, 0])
        # position: force = kp*(ctrl-q) - kv*qvel
        mj.actuator_gainprm[aid, 0] = use_kp
        mj.actuator_biasprm[aid, 1] = -use_kp
        mj.actuator_biasprm[aid, 2] = -use_kv
        applied.append(f"{short}:{old_kp:.1f}→{use_kp:.1f}")
    # 正式版不打印装配成功日志；失败仍走上方 warning


def lock_lower_body(manager: DataCollectionManager, env):
    """锁定腿部 + 腰部，保持当前 qpos（每 episode 重置时自动更新）。"""
    joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint",
        "left_hip_yaw_joint", "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint",
        "right_hip_yaw_joint", "right_knee_joint",
        "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint",
        "waist_pitch_joint",
    ]
    ctrl_name = [env.actuator(n) for n in joint_names]
    joint_ids = [env.joint(n) for n in joint_names]
    qpos = env.query_joint_qpos(joint_ids)
    init_positions = np.array(
        [float(np.asarray(qpos[j], dtype=np.float64).reshape(-1)[0]) for j in joint_ids],
        dtype=np.float32,
    )

    holder = JointHoldController(env, ctrl_name, init_positions, joint_names)
    manager.add_controller(holder)
    return holder


def pin_floating_base(env, agent_name: str) -> bool:
    """钉住浮动基座，效果接近静态刚体，但不改 nq（OrcaStudio 同步需要 nq 一致）。

    不能删 XML 里的 freejoint：本地 nq 会少 7，UpdateLocalEnv 把短 qpos 推给
    仍含 freejoint 的 OrcaStudio，关节映射错位，表现为手臂“锁死”。

    做法：包装 gym.mj_step，每子步前后把 freejoint 的 qpos/qvel 写回初值。
    """
    import mujoco

    gym = getattr(env, "gym", None) or getattr(getattr(env, "unwrapped", env), "gym", None)
    if gym is None or not hasattr(gym, "_mjModel") or not hasattr(gym, "_mjData"):
        orca_logger.warning("[BASE-PIN] env.gym._mjModel/_mjData unavailable")
        return False

    mj, md = gym._mjModel, gym._mjData
    jname = f"{agent_name}_floating_base_joint"
    jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid < 0:
        orca_logger.warning(f"[BASE-PIN] freejoint not found: {jname}")
        return False

    qadr = int(mj.jnt_qposadr[jid])
    dadr = int(mj.jnt_dofadr[jid])
    q0 = np.array(md.qpos[qadr:qadr + 7], dtype=np.float64, copy=True)
    _orig_mj_step = gym.mj_step

    def _mj_step_pinned(nstep=1):
        n = int(nstep) if nstep is not None else 1
        for _ in range(max(n, 1)):
            md.qpos[qadr:qadr + 7] = q0
            md.qvel[dadr:dadr + 6] = 0.0
            _orig_mj_step(1)
            md.qpos[qadr:qadr + 7] = q0
            md.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_forward(mj, md)

    gym.mj_step = _mj_step_pinned
    return True


_CABINET_BUTTON_JOINTS = (
    "Group_Static_ElectricalCabinet_button01_joint",
    "Group_Static_ElectricalCabinet_button02_joint",
    "Group_Static_ElectricalCabinet_Button03_joint",
    "Group_Static_ElectricalCabinet_Button04_joint",
)


def limit_cabinet_button_slides(
    env,
    agent_name: str = "unitree_humanoid_robot_1",
    *,
    toward_robot_m: float = 0.0,
    into_panel_m: float = 0.05,
    base_body_short: str | None = None,
    hard_clamp: bool = True,
    # 刚度过大时按压反力会把臂顶开（表现为 push 准、hold 误差反升）
    joint_stiffness: float = 80.0,
    joint_damping: float = 8.0,
) -> int:
    """限制电柜按钮 slide：禁止朝机器人方向弹出，只允许压入面板。

    默认 toward_robot_m=0：以当前 q0 为朝机器人侧硬墙，不允许任何弹出行程。
    压入侧保留 into_panel_m（默认 5cm）以便按下。

    防弹出策略（避免「先弹再缩回」）：
      1) jnt_range 朝机器人侧贴在 q0
      2) 提高 stiffness/damping，弹簧本身顶住接触反力
      3) mj_step 子步前/后硬夹 qpos，并清零朝外速度

    Returns:
        成功限位的关节数。
    """
    import mujoco

    from conf import g1_pick_conf

    gym = getattr(env, "gym", None) or getattr(getattr(env, "unwrapped", env), "gym", None)
    if gym is None or not hasattr(gym, "_mjModel") or not hasattr(gym, "_mjData"):
        orca_logger.warning("[BTN-LIMIT] env.gym._mjModel/_mjData unavailable")
        return 0
    mj, md = gym._mjModel, gym._mjData

    short = base_body_short or g1_pick_conf.base_body
    try:
        base_name = env.body(short)
    except Exception:
        base_name = f"{agent_name}_{short}"
    bid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, base_name)
    if bid < 0:
        orca_logger.warning(f"[BTN-LIMIT] base body not found: {base_name}")
        return 0
    robot = np.asarray(md.xpos[bid], dtype=np.float64).copy()

    toward = max(0.0, float(toward_robot_m))
    into = max(1e-4, float(into_panel_m))
    # (qadr, dadr, lo, hi, sign_toward_robot)  sign: +1 表示 +q 朝机器人
    clamp_specs: list[tuple[int, int, float, float, float]] = []
    n_ok = 0
    for jn in _CABINET_BUTTON_JOINTS:
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            continue
        if int(mj.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        qadr = int(mj.jnt_qposadr[jid])
        dadr = int(mj.jnt_dofadr[jid])
        bodyid = int(mj.jnt_bodyid[jid])
        q0 = float(md.qpos[qadr])
        xmat = np.asarray(md.xmat[bodyid], dtype=np.float64).reshape(3, 3)
        axis_w = xmat @ np.asarray(mj.jnt_axis[jid], dtype=np.float64)
        btn = np.asarray(md.xpos[bodyid], dtype=np.float64)
        to_robot = robot - btn
        nrm = float(np.linalg.norm(to_robot))
        if nrm < 1e-9:
            continue
        to_robot /= nrm
        # +axis 指向机器人 → 朝机器人一侧为 +q
        if float(np.dot(axis_w, to_robot)) >= 0.0:
            lo, hi = q0 - into, q0 + toward
            sign_toward = 1.0
        else:
            lo, hi = q0 - toward, q0 + into
            sign_toward = -1.0
        mj.jnt_limited[jid] = 1
        mj.jnt_range[jid, 0] = lo
        mj.jnt_range[jid, 1] = hi
        # 加硬弹簧：静止在 q0，接触反力不易顶飞
        # MuJoCo: stiffness 在 jnt_*，damping 在 dof_*
        mj.jnt_stiffness[jid] = float(joint_stiffness)
        mj.dof_damping[dadr] = float(joint_damping)
        md.qpos[qadr] = float(np.clip(q0, lo, hi))
        md.qvel[dadr] = 0.0
        clamp_specs.append((qadr, dadr, lo, hi, sign_toward))
        n_ok += 1
        orca_logger.info(
            f"[BTN-LIMIT] {jn}: q0={q0:+.4f} → range=[{lo:+.4f}, {hi:+.4f}] "
            f"toward_robot≤{toward:.4f}m  k={joint_stiffness:g} d={joint_damping:g}"
        )

    if n_ok and hard_clamp and clamp_specs:
        # 剥掉旧的按钮限位包装，避免每集叠一层
        _inner = gym.mj_step
        while getattr(_inner, "_is_btn_limit_wrap", False):
            _inner = _inner._btn_limit_inner

        def _clamp_btn(_specs=tuple(clamp_specs)):
            for qadr, dadr, lo, hi, sign_toward in _specs:
                q = float(md.qpos[qadr])
                if q < lo:
                    md.qpos[qadr] = lo
                    md.qvel[dadr] = 0.0
                elif q > hi:
                    md.qpos[qadr] = hi
                    md.qvel[dadr] = 0.0
                # 清零朝机器人方向的速度，防止下一步冲出
                v = float(md.qvel[dadr])
                if v * sign_toward > 0.0:
                    md.qvel[dadr] = 0.0

        def _mj_step_btn_clamped(nstep=1, _step=_inner):
            n = int(nstep) if nstep is not None else 1
            for _ in range(max(n, 1)):
                _clamp_btn()  # 步进前：禁止已越界/朝外速度
                _step(1)
                _clamp_btn()  # 步进后：立刻钉回，不等下一帧
            mujoco.mj_forward(mj, md)

        _mj_step_btn_clamped._is_btn_limit_wrap = True  # type: ignore[attr-defined]
        _mj_step_btn_clamped._btn_limit_inner = _inner  # type: ignore[attr-defined]
        gym.mj_step = _mj_step_btn_clamped

    if n_ok:
        try:
            mujoco.mj_forward(mj, md)
        except Exception:
            pass
    else:
        orca_logger.warning("[BTN-LIMIT] 未找到可限位的电柜按钮 slide 关节")
    return n_ok


# =============================================================================
# 双臂 Unitree CasADi IK + Pico 绑定
# =============================================================================
def add_dual_arm_unitree_ik_pico(
    dcm: DataCollectionManager,
    env,
    device: G1PickPicoJoystickDevice,
    is_running_fn=None,
    diag_every: int = 50,
    dbg_log=None,
    max_reach: float | None = None,
    project_reachable: bool = False,
    lock_left: bool = True,
    deadzone_pos_m: float = 0.006,
    deadzone_ori_deg: float = 2.0,
    goal_ema_alpha: float = 0.95,
) -> G1PickDualArmIKController:
    """绑定 Pico 右臂 TRANSFORM → Unitree G1_29 CasADi IK；左臂默认锁定。"""
    # Pico 轴约定：device 已做 Unity→MuJoCo；此处仅做轴重排/翻转（不再叠 OmniPicker 硬编码 90°）
    L_REMAP = [0, 2, 1]
    R_REMAP = [0, 2, 1]
    L_FLIP = np.array([1.0, 1.0, -1.0])
    R_FLIP = np.array([1.0, 1.0, -1.0])

    if dbg_log is not None:
        dbg_log(
            f"[DUAL-IK] building G1_29_ArmIK (Pinocchio+CasADi) "
            f"lock_left={lock_left} "
            f"dz={deadzone_pos_m}m/{deadzone_ori_deg}deg "
            f"ema={goal_ema_alpha} "
            f"max_reach={max_reach}"
        )
    ik_kwargs = {}
    if max_reach is not None:
        ik_kwargs["max_reach"] = float(max_reach)
    arm_ik = G1_29_ArmIK(**ik_kwargs)
    dual = G1PickDualArmIKController(
        env,
        base_body=g1_pick_conf.base_body,
        is_running_fn=is_running_fn,
        arm_ik=arm_ik,
        diag_every=diag_every,
        dbg_log=dbg_log,
        pos_remap_l=L_REMAP,
        pos_remap_r=R_REMAP,
        pos_flip_l=L_FLIP,
        pos_flip_r=R_FLIP,
        deadzone_pos_m=deadzone_pos_m,
        deadzone_ori_deg=deadzone_ori_deg,
        goal_ema_alpha=goal_ema_alpha,
        max_reach=max_reach,
        project_reachable=project_reachable,
        lock_left=lock_left,
    )

    _cb_count = {"L": 0, "R": 0}
    _drop_count = {"L": 0, "R": 0}

    def _make_cb(side: str):
        def callback(relative_position, relative_quat):
            if is_running_fn is not None and not is_running_fn():
                _drop_count[side] += 1
                if _drop_count[side] <= 3 or _drop_count[side] % 100 == 0:
                    msg = (
                        f"[GATE] drop transform {side} (not RUNNING) "
                        f"n={_drop_count[side]}"
                    )
                    if dbg_log is not None:
                        dbg_log(msg)
                    else:
                        orca_logger.info(msg)
                return

            _cb_count[side] += 1
            quat_wxyz = np.asarray(relative_quat, dtype=np.float64).reshape(4)
            if side == "L":
                dual.update_goal_L(relative_position, quat_wxyz)
            else:
                dual.update_goal_R(relative_position, quat_wxyz)

            if _cb_count[side] <= 5 or _cb_count[side] % max(1, diag_every) == 0:
                msg = (
                    f"[VR-{side}] #{_cb_count[side]} "
                    f"rel_pos=({relative_position[0]:.3f},{relative_position[1]:.3f},"
                    f"{relative_position[2]:.3f}) "
                    f"rel_quat_wxyz=({quat_wxyz[0]:.3f},{quat_wxyz[1]:.3f},"
                    f"{quat_wxyz[2]:.3f},{quat_wxyz[3]:.3f})"
                )
                if dbg_log is not None:
                    dbg_log(msg)
                else:
                    orca_logger.info(msg)

        return callback

    # 与智元一致：左臂锁定时不绑定左手柄位姿
    if not lock_left:
        device.bind_transform_event(PicoJoystickKey.L_TRANSFORM, _make_cb("L"))
    device.bind_transform_event(PicoJoystickKey.R_TRANSFORM, _make_cb("R"))
    dcm.add_controller(dual)
    if dbg_log is not None:
        dbg_log(
            f"[VR] dual-arm Unitree IK lock_left={lock_left} "
            f"remap L={L_REMAP} flip={L_FLIP.tolist()} | "
            f"R={R_REMAP} flip={R_FLIP.tolist()} (no OmniPicker rot overlay)"
        )
    return dual


def add_dual_arm_unitree_ik_televuer(
    dcm: DataCollectionManager,
    env,
    device,
    is_running_fn=None,
    diag_every: int = 50,
    dbg_log=None,
    goal_mode: str = "rebased_tv",
    dry_run: bool = False,
    max_pos_jump_m: float = 0.50,
    max_ori_jump_deg: float = 90.0,
    max_dq_step: float = 0.25,
    deadzone_pos_m: float = 0.006,
    deadzone_ori_deg: float = 2.0,
    goal_ema_alpha: float = 0.95,
    max_reach: float | None = None,
    project_reachable: bool = False,
    lock_left: bool = True,
) -> G1PickDualArmIKController:
    """Bind TeleVuer SE3 goals → Unitree G1_29 CasADi IK；左臂默认锁定。"""
    if dbg_log is not None:
        dbg_log(
            f"[DUAL-IK] building G1_29_ArmIK for televuer mode={goal_mode} "
            f"dry_run={dry_run} lock_left={lock_left} max_dq={max_dq_step} "
            f"dz={deadzone_pos_m}m/{deadzone_ori_deg}deg "
            f"ema={goal_ema_alpha} "
            f"max_reach={max_reach} project_reachable={project_reachable}"
        )
    ik_kwargs = {}
    if max_reach is not None:
        ik_kwargs["max_reach"] = float(max_reach)
    arm_ik = G1_29_ArmIK(**ik_kwargs)
    dual = G1PickDualArmIKController(
        env,
        base_body=g1_pick_conf.base_body,
        is_running_fn=is_running_fn,
        arm_ik=arm_ik,
        diag_every=diag_every,
        dbg_log=dbg_log,
        goal_mode=goal_mode,
        dry_run=dry_run,
        max_pos_jump_m=max_pos_jump_m,
        max_ori_jump_deg=max_ori_jump_deg,
        max_dq_step=max_dq_step,
        deadzone_pos_m=deadzone_pos_m,
        deadzone_ori_deg=deadzone_ori_deg,
        goal_ema_alpha=goal_ema_alpha,
        max_reach=max_reach,
        project_reachable=project_reachable,
        lock_left=lock_left,
    )

    def _on_dual_pose(T_l, T_r, ts):
        dual.set_goals(T_l, T_r, timestamp=ts)

    device.bind_dual_pose_event(_on_dual_pose)
    device.bind_reconnect_event(lambda: dual.request_rebase("xr_reconnect"))
    dcm.add_controller(dual)
    return dual


def create_hand_televuer_controller(
    manager: DataCollectionManager,
    env,
    hand_config: dict,
    device,
    side: str,
    is_running_fn=None,
):
    """灵巧手：TeleVuer 扳机 → HandController（已归一化到 [0,1]）。"""
    ctrl_name = [env.actuator(n) for n in hand_config["positions_names"]]
    init_ctrl = {n: v for n, v in zip(ctrl_name, hand_config["positions_init_ctrl"])}
    hc = HandController(env, hand_config, ctrl_name, init_ctrl, side)

    def _on_trigger(value: float):
        if is_running_fn is not None and not is_running_fn():
            return
        hc.update_trigger_value(value)

    if side.upper().startswith("L"):
        device.bind_left_trigger_event(_on_trigger)
    else:
        device.bind_right_trigger_event(_on_trigger)
    manager.add_controller(hc)


# =============================================================================
# Config / Diag / Wiring / Monitor / Runner  (单文件内聚类，不分文件)
# =============================================================================

@dataclasses.dataclass
class TeleopConfig:
    """从 argparse.Namespace 一次性构建，下游只依赖此对象，不再直接读 args。"""

    level: str
    task_config: str
    lerobot_out: str          # 已 abspath/expanduser
    repo_id: str
    task: str
    fps: int
    clock: str
    resume: bool
    orcagym_addr: str
    cameras: str
    cam_resolution: str
    camera_source: str
    local_xml: object         # str | None
    scene_json: str
    agent_name: str
    log_file: object          # str | None
    diag_tele: bool
    diag_every: int
    xr_backend: str
    tv_goal_mode: str
    tv_ee_dx: float
    tv_position_scale: float
    dry_run_tele: bool
    tv_max_pos_jump: float
    tv_max_ori_jump: float
    tv_max_dq_step: float
    tv_deadzone_pos: float
    tv_deadzone_ori: float
    tv_goal_ema: float
    ik_max_reach: float
    ik_project_reachable: bool
    diag_health: bool
    diag_joints: bool
    diag_joints_hz: float
    diag_log_dir: str
    tv_no_tls: bool
    tv_cert_file: object       # str | None
    tv_key_file: object        # str | None
    tv_host: str
    arm_kp: float
    arm_kv: object            # float | None  (explicit override)
    arm_kv_ratio: float
    arm_kv_eff: float         # derived: arm_kv if set, else arm_kv_ratio * arm_kp
    arm_gravcomp: float       # MuJoCo gravcomp scale for arm+hand bodies (0=off, 1=full)

    @classmethod
    def from_args(cls, args) -> "TeleopConfig":
        kv_eff = (
            float(args.arm_kv)
            if args.arm_kv is not None
            else args.arm_kv_ratio * args.arm_kp
        )
        return cls(
            level=args.level,
            task_config=args.task_config,
            lerobot_out=resolve_lerobot_out(args.lerobot_out),
            repo_id=args.repo_id,
            task=args.task,
            fps=args.fps,
            clock=args.clock,
            resume=args.resume,
            orcagym_addr=args.orcagym_addr,
            cameras=args.cameras,
            cam_resolution=args.cam_resolution,
            camera_source=args.camera_source,
            local_xml=args.local_xml,
            scene_json=args.scene_json,
            agent_name=args.agent_name,
            log_file=args.log_file,
            diag_tele=args.diag_tele,
            diag_every=args.diag_every,
            xr_backend=args.xr_backend,
            tv_goal_mode=args.tv_goal_mode,
            tv_ee_dx=args.tv_ee_dx,
            tv_position_scale=args.tv_position_scale,
            dry_run_tele=args.dry_run_tele,
            tv_max_pos_jump=args.tv_max_pos_jump,
            tv_max_ori_jump=args.tv_max_ori_jump,
            tv_max_dq_step=args.tv_max_dq_step,
            tv_deadzone_pos=args.tv_deadzone_pos,
            tv_deadzone_ori=args.tv_deadzone_ori,
            tv_goal_ema=args.tv_goal_ema,
            ik_max_reach=args.ik_max_reach,
            ik_project_reachable=args.ik_project_reachable,
            diag_health=args.diag_health,
            diag_joints=args.diag_joints,
            diag_joints_hz=args.diag_joints_hz,
            diag_log_dir=args.diag_log_dir,
            tv_no_tls=args.tv_no_tls,
            tv_cert_file=args.tv_cert_file,
            tv_key_file=args.tv_key_file,
            tv_host=args.tv_host,
            arm_kp=args.arm_kp,
            arm_kv=args.arm_kv,
            arm_kv_ratio=args.arm_kv_ratio,
            arm_kv_eff=kv_eff,
            arm_gravcomp=args.arm_gravcomp,
        )


class DiagLog:
    """封装诊断日志文件句柄与可选的 stdout Tee，替代 _log_fh/_Tee/_dbg_log 全局。"""

    def __init__(self, cfg: TeleopConfig) -> None:
        # 可选：终端输出 tee 到 log_file
        self._log_fh = None
        if cfg.log_file:
            os.makedirs(
                os.path.dirname(os.path.abspath(cfg.log_file)) or ".", exist_ok=True
            )
            self._log_fh = open(cfg.log_file, "w", buffering=1)

            class _Tee:
                def __init__(self, *files):
                    self.files = files

                def write(self, s):
                    for f in self.files:
                        f.write(s)
                        f.flush()

                def flush(self):
                    for f in self.files:
                        f.flush()

            sys.stdout = _Tee(sys.stdout, self._log_fh)
            sys.stderr = _Tee(sys.stderr, self._log_fh)
            print(f"[日志] 终端输出同时写入: {cfg.log_file}", flush=True)

        # 诊断日志文件（时间戳）
        _diag_enabled = bool(cfg.diag_tele or cfg.diag_joints or cfg.diag_health)
        _latest = Path("/tmp/debug_tele_g1pick.txt")
        if _diag_enabled:
            _diag_dir = Path(os.path.expanduser(cfg.diag_log_dir))
            _diag_dir.mkdir(parents=True, exist_ok=True)
            _stamp = time.strftime("%Y%m%d_%H%M%S")
            log_path = _diag_dir / f"debug_tele_g1pick_{_stamp}.txt"
            self._dbg_fh = open(str(log_path), "w", buffering=1)
            try:
                if _latest.is_symlink() or _latest.exists():
                    _latest.unlink()
                _latest.symlink_to(log_path)
            except OSError:
                pass
        else:
            log_path = _latest
            self._dbg_fh = open(str(log_path), "w", buffering=1)

        self.log_path = log_path
        self.latest_link = _latest
        self.enabled = _diag_enabled

    def log(self, msg: str) -> None:
        # 诊断未开启时不向终端输出诊断日志。
        if not self.enabled:
            return
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._dbg_fh.write(line + "\n")
        self._dbg_fh.flush()

    def close(self) -> None:
        try:
            self._dbg_fh.close()
        except Exception:
            pass
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass


class ControllerWiring:
    """装配器：绑定双臂 IK、灵巧手、腿锁、任务状态与输入事件回调。

    替代 main() 内散落的 _dual_arm_holder / _is_tele_running / _on_discard 等闭包。
    持有 discard_event，供 HealthMonitor 和 EpisodeRunner 共享。
    """

    def __init__(self) -> None:
        self.dual_arm_ctrl = None
        self.discard_event = threading.Event()
        self._prev_gate_running: list = [None]

    def wire(
        self,
        cfg: TeleopConfig,
        env,
        manager,
        xr_device,
        pico_device,
        diag: DiagLog,
    ):
        apply_arm_position_gains(
            env, kp=cfg.arm_kp, kv=cfg.arm_kv, kv_ratio=cfg.arm_kv_ratio
        )

        def _is_tele_running() -> bool:
            tsc = manager.task_status_controller
            return tsc is not None and tsc.current_status == TaskStatus.RUNNING

        self._is_tele_running = _is_tele_running

        if cfg.xr_backend == "pico":
            dual = add_dual_arm_unitree_ik_pico(
                manager, env, pico_device,
                is_running_fn=_is_tele_running,
                diag_every=cfg.diag_every,
                dbg_log=diag.log if cfg.diag_tele else None,
                max_reach=cfg.ik_max_reach,
                project_reachable=cfg.ik_project_reachable,
                deadzone_pos_m=cfg.tv_deadzone_pos,
                deadzone_ori_deg=cfg.tv_deadzone_ori,
                goal_ema_alpha=cfg.tv_goal_ema,
            )
        else:
            dual = add_dual_arm_unitree_ik_televuer(
                manager, env, xr_device,
                is_running_fn=_is_tele_running,
                diag_every=cfg.diag_every,
                dbg_log=diag.log if cfg.diag_tele else None,
                goal_mode=cfg.tv_goal_mode,
                dry_run=cfg.dry_run_tele,
                max_pos_jump_m=cfg.tv_max_pos_jump,
                max_ori_jump_deg=cfg.tv_max_ori_jump,
                max_dq_step=cfg.tv_max_dq_step,
                deadzone_pos_m=cfg.tv_deadzone_pos,
                deadzone_ori_deg=cfg.tv_deadzone_ori,
                goal_ema_alpha=cfg.tv_goal_ema,
                max_reach=cfg.ik_max_reach,
                project_reachable=cfg.ik_project_reachable,
            )
        dual.reset()
        self.dual_arm_ctrl = dual

        # 灵巧手
        if cfg.xr_backend == "pico":
            create_hand_pico_controller(
                manager, env, g1_pick_conf.l_hand,
                pico_device, PicoJoystickKey.L_TRIGGER, "L",
            )
            create_hand_pico_controller(
                manager, env, g1_pick_conf.r_hand,
                pico_device, PicoJoystickKey.R_TRIGGER, "R",
            )
        else:
            create_hand_televuer_controller(
                manager, env, g1_pick_conf.l_hand, xr_device, "L",
                is_running_fn=_is_tele_running,
            )
            create_hand_televuer_controller(
                manager, env, g1_pick_conf.r_hand, xr_device, "R",
                is_running_fn=_is_tele_running,
            )

        # 腿锁 + 钉住浮动基座（不删 freejoint，避免 nq 与 OrcaStudio 不一致）
        lock_lower_body(manager, env)
        pin_floating_base(env, cfg.agent_name)

        # 任务状态控制器
        manager.set_task(EmptyTask(env))
        if cfg.xr_backend == "pico":
            controllers.add_task_status_pico_controller(
                manager, env, pico_device, g1_pick_conf.base_body,
            )
        else:
            tsc = TaskStatusController(env, g1_pick_conf.base_body)

            def _tv_task_toggle(_pressed: bool = True):
                tsc.update_task_status(True, reason="left_squeeze")

            xr_device.bind_task_toggle_event(_tv_task_toggle)
            manager.set_task_status_controller(tsc)

        # 输入事件路由
        if cfg.xr_backend == "pico":
            self._setup_pico_gate(cfg, manager, pico_device, diag)
        else:
            self._setup_tv_events(cfg, manager, xr_device, diag)

        return dual

    def _setup_pico_gate(self, cfg, manager, pico_device, diag):
        _LOCKED_KEYS = {PicoJoystickKey.L_TRANSFORM}
        _all_keys = [k for k in pico_device.keys if k not in _LOCKED_KEYS]
        _pre_start_keys = [k for k in _all_keys if k == PicoJoystickKey.L_GRIPBUTTON]
        _counter = [0]
        _prev = self._prev_gate_running
        discard_event = self.discard_event
        diag.log(
            f"[GATE] keys_all={len(_all_keys)} locked={list(_LOCKED_KEYS)} "
            f"pre_start={_pre_start_keys} (IDLE only L_GRIPBUTTON; L arm locked)"
        )

        def _gated_pico_update():
            _counter[0] += 1
            tsc = manager.task_status_controller
            running = tsc is not None and tsc.current_status == TaskStatus.RUNNING

            if _prev[0] is None or running != _prev[0]:
                prev_val = _prev[0]
                if prev_val is None:
                    label = f"INIT→{'RUNNING' if running else 'IDLE'}"
                else:
                    label = (
                        f"{'RUNNING' if prev_val else 'IDLE'}→"
                        f"{'RUNNING' if running else 'IDLE'}"
                    )
                allow = _all_keys if running else _pre_start_keys
                diag.log(f"[GATE] {label} allow={allow}")
                _prev[0] = running

            if _counter[0] % 300 == 1 and cfg.diag_tele:
                pj = pico_device.pico_joystick
                raw = pj.current_key_state
                if raw:
                    lh = raw.get("leftHand", {}) or {}
                    rh = raw.get("rightHand", {}) or {}
                    lp = lh.get("position", {})
                    rp = rh.get("position", {})
                    lv = list(lp.values()) if isinstance(lp, dict) else lp
                    rv = list(rp.values()) if isinstance(rp, dict) else rp
                    diag.log(
                        f"[PICO] raw L={lv} R={rv} clients={len(pj.clients)} "
                        f"running={running}"
                    )
                else:
                    diag.log(
                        f"[PICO] raw empty clients={len(pj.clients)} running={running}"
                    )

            if running:
                pico_device.pico_joystick.update(_all_keys)
            else:
                pico_device.pico_joystick.update(_pre_start_keys)

        pico_device.update = _gated_pico_update

    def _setup_tv_events(self, cfg, manager, xr_device, diag):
        discard_event = self.discard_event

        def _on_discard():
            orca_logger.info("[Squeeze] 右 squeeze → 丢弃本集")
            print("[EP] END reason=right_squeeze_discard", flush=True)
            discard_event.set()
            manager._shutdown_requested = True  # noqa: SLF001

        def _on_shutdown():
            orca_logger.info("[Squeeze] 左右同按 → 终止采集")
            print("[EP] END reason=both_squeeze_shutdown", flush=True)
            manager._shutdown_requested = True  # noqa: SLF001

        def _on_disconnect(sustained: bool):
            diag.log(f"[TV] disconnect sustained={sustained}")
            if sustained:
                tsc = manager.task_status_controller
                if tsc is not None and tsc.current_status == TaskStatus.RUNNING:
                    prev = tsc.current_status
                    tsc.current_status = TaskStatus.END
                    print(
                        f"[EP] transition {prev.name}→END reason=disconnect_sustained",
                        flush=True,
                    )
                    orca_logger.info("[TV] sustained disconnect → TaskStatus END, hold arms")

        xr_device.bind_discard_event(_on_discard)
        xr_device.bind_shutdown_event(_on_shutdown)
        xr_device.bind_disconnect_event(_on_disconnect)
        diag.log(
            f"[GATE] televuer mode={cfg.tv_goal_mode} dry_run={cfg.dry_run_tele} "
            f"ee_dx={cfg.tv_ee_dx} diag_health={cfg.diag_health}"
        )


class HealthMonitor:
    """后台监控线程封装，替代 _monitor_stop / _tv_diag_n 与散落的 threading.Thread 创建。

    管理三类线程：
    - pico 后端：_pico_monitor（连接状态 + Grip 快捷键）
    - televuer 后端：_tv_monitor（连接 + health 心跳）
    - 两种后端均可开启：_joint_monitor（关节细粒度诊断）
    """

    _POLL_DT = 0.02

    def __init__(
        self,
        cfg: TeleopConfig,
        env,
        xr_device,
        pico_device,
        manager,
        wiring: ControllerWiring,
        diag: DiagLog,
    ) -> None:
        self._cfg = cfg
        self._env = env
        self._xr_device = xr_device
        self._pico_device = pico_device
        self._manager = manager
        self._wiring = wiring
        self._diag = diag
        self._stop = threading.Event()
        self._threads: list = []

    def start(self) -> None:
        cfg = self._cfg
        if cfg.xr_backend == "pico":
            self._threads.append(
                threading.Thread(target=self._pico_monitor, daemon=True)
            )
        else:
            self._threads.append(
                threading.Thread(target=self._tv_monitor, daemon=True)
            )
        if cfg.diag_joints:
            self._threads.append(
                threading.Thread(target=self._joint_monitor, daemon=True)
            )
        for t in self._threads:
            t.start()
        if cfg.diag_joints:
            _j_hz = max(0.5, float(cfg.diag_joints_hz))
            self._diag.log(
                f"[JOINT] monitor ON hz={_j_hz:.1f} "
                f"log={self._diag.log_path} "
                f"(L/R: sp sr sy el wr wp wy)"
            )

    def stop(self) -> None:
        self._stop.set()

    # ── Pico backend monitor ───────────────────────────────────────────────
    def _pico_monitor(self):
        _last_warn = [0.0]
        _first_ok = [False]
        pico_device = self._pico_device
        manager = self._manager
        diag = self._diag
        discard_event = self._wiring.discard_event

        while not self._stop.wait(self._POLL_DT):
            try:
                pj = pico_device.pico_joystick
                n_clients = len(pj.clients)
                raw_key = pj.current_key_state
                now = time.perf_counter()

                if n_clients == 0 or raw_key is None:
                    if now - _last_warn[0] >= 2.0:
                        _last_warn[0] = now
                        diag.log(
                            f"[PICO] no data clients={n_clients} "
                            f"key_state={'none' if raw_key is None else 'ok'}"
                        )
                elif not _first_ok[0]:
                    _first_ok[0] = True
                    diag.log(f"[PICO] connected clients={n_clients}")

                if n_clients > 0 and raw_key is not None:
                    lh = raw_key.get("leftHand", {}) or {}
                    rh = raw_key.get("rightHand", {}) or {}
                    l_grip = bool(lh.get("gripButtonPressed"))
                    r_grip = bool(rh.get("gripButtonPressed"))
                    both = l_grip and r_grip
                    r_only = r_grip and not l_grip

                    if both:
                        orca_logger.info("[Grip] 左右同按 → 终止采集")
                        manager._shutdown_requested = True  # noqa: SLF001
                    elif r_only:
                        orca_logger.info("[Grip] 右Grip单按 → 丢弃本集")
                        discard_event.set()
                        manager._shutdown_requested = True  # noqa: SLF001
            except Exception:
                pass

    # ── TeleVuer backend monitor ───────────────────────────────────────────
    def _tv_monitor(self):
        cfg = self._cfg
        xr_device = self._xr_device
        manager = self._manager
        diag = self._diag
        _tv_diag_n = [0]
        _health_interval = 1.0 if cfg.diag_health else 2.0

        while not self._stop.wait(_health_interval):
            if not (cfg.diag_tele or cfg.diag_health):
                continue
            try:
                tsc = manager.task_status_controller
                st = tsc.current_status.name if tsc is not None else "None"
                _tv_diag_n[0] += 1

                if cfg.diag_health and hasattr(xr_device, "health_snapshot"):
                    h = xr_device.health_snapshot()
                    visit = h.get("visit", "")
                    if visit and not getattr(self._tv_monitor, "_visit_logged", False):
                        self._tv_monitor._visit_logged = True  # type: ignore[attr-defined]
                        diag.log(f"[HEALTH][VISIT] open on Pico: {visit}")
                    diag.log(
                        f"[HEALTH] child={h['child']} "
                        f"exit={h['child_exitcode']} "
                        f"port8012={h['port8012']} "
                        f"tls={'on' if h.get('tls', True) else 'off'} "
                        f"uplink_age={h['uplink_age']:.2f}s "
                        f"cam_age={h['cam_age']:.2f}s "
                        f"ctrl_age={h['ctrl_age']:.2f}s "
                        f"cam_only={h['cam_only']} "
                        f"connected={h['connected']} motion={h['motion']} "
                        f"seq={h['seq']} cam={h['cam_seq']} ctrl={h['ctrl_seq']} "
                        f"cam_Hz={h['cam_rate']:.1f} ctrl_Hz={h['ctrl_rate']:.1f} "
                        f"status={st} dry_run={cfg.dry_run_tele}"
                    )
                    if h["seq_rising_no_motion"]:
                        diag.log(
                            "[HEALTH][HINT] seq rising but motion=False — "
                            "likely CONTROLLER_MOVE handler silent fail "
                            "(televuer except:pass) or XR session without "
                            "MotionControllers; 晃动手柄 / 重进 Pass Through"
                        )
                    if h["backpressure"]:
                        diag.log(
                            "[HEALTH][HINT] ctrl_rate≪cam_rate — "
                            "possible Vuer queue_len=3 backpressure dropping "
                            "CONTROLLER_MOVE (短促 squeeze 可能丢失)"
                        )
                    if h["child"] == "DEAD":
                        diag.log(
                            "[HEALTH][FATAL] Vuer 子进程已死 — "
                            "检查 8012 残留/证书；需重启数采脚本"
                        )
                    continue

                # Legacy 轻量 TV 行（仅 diag_tele 时）
                motion_ready = False
                try:
                    motion_ready = bool(
                        xr_device._tv_wrapper.get_tele_data().motion_data_ready  # noqa: SLF001
                    )
                except Exception:
                    pass
                cam_s = getattr(xr_device, "cam_seq", "?")
                ctrl_s = getattr(xr_device, "ctrl_seq", "?")
                diag.log(
                    f"[TV] #{_tv_diag_n[0]} connected={xr_device.connected} "
                    f"motion_ready={motion_ready} "
                    f"event_seq={xr_device.event_seq} "
                    f"cam_seq={cam_s} ctrl_seq={ctrl_s} "
                    f"status={st} dry_run={cfg.dry_run_tele}"
                )
                if not xr_device.connected and _tv_diag_n[0] in (1, 2, 5, 15):
                    diag.log(
                        "[TV] 未收到 XR 位姿：USB-local 请 "
                        "adb reverse tcp:8012 tcp:8012 后打开 "
                        "https://127.0.0.1:8012/ （先过证书），"
                        "等终端出现 websocket is connected 且右上角不再狂闪 reconnect，"
                        "再点 Pass Through；晃动手柄至 ctrl_seq 增长后左 squeeze 开始"
                    )
            except Exception:
                pass

    # ── Joint diagnostic monitor ───────────────────────────────────────────
    def _joint_monitor(self):
        cfg = self._cfg
        env = self._env
        manager = self._manager
        pico_device = self._pico_device
        wiring = self._wiring
        diag = self._diag

        _arm_joint_names = list(g1_pick_conf.l_arm["joint_names"]) + list(
            g1_pick_conf.r_arm["joint_names"]
        )
        _arm_joint_ids = [env.joint(n) for n in _arm_joint_names]
        _arm_act_ids = [
            env.model.actuator_name2id(env.actuator(n))
            for n in list(g1_pick_conf.l_arm["positions_names"])
            + list(g1_pick_conf.r_arm["positions_names"])
        ]
        _qn = ("sp", "sr", "sy", "el", "wr", "wp", "wy")
        _j_hz = max(0.5, float(cfg.diag_joints_hz))
        _j_dt = 1.0 / _j_hz
        _j_n = [0]
        _j_prev_q = [None]
        _j_prev_t = [None]
        _ee_l = env.site(g1_pick_conf.l_arm["ee_site_name"])
        _ee_r = env.site(g1_pick_conf.r_arm["ee_site_name"])
        _base = env.body(g1_pick_conf.base_body)

        def _fmt7_deg(arr7) -> str:
            d = np.degrees(np.asarray(arr7, dtype=np.float64).reshape(7))
            return " ".join(f"{n}={v:+.1f}" for n, v in zip(_qn, d))

        def _pico_ks_to_mj(raw_pos_list, raw_quat_list):
            """从 current_key_state 里的 Unity 坐标还原 MuJoCo 空间位姿。
            与 abstract_device.transform_event 保持完全一致。
            raw_pos_list : [x, y, z]  Unity position
            raw_quat_list: [x, y, z, w] Unity rotation (extract_key_state 顺序)
            """
            p = np.array(raw_pos_list, dtype=np.float64)[[2, 0, 1]]
            p[1] = -p[1]
            q = np.array(raw_quat_list, dtype=np.float64)
            return p, q

        while not self._stop.wait(_j_dt):
            try:
                now = time.perf_counter()
                qpos = env.query_joint_qpos(_arm_joint_ids)
                q = np.array(
                    [
                        float(np.asarray(qpos[j]).reshape(-1)[0])
                        for j in _arm_joint_ids
                    ],
                    dtype=np.float64,
                )
                ctrl = np.array(
                    [float(env.ctrl[i]) for i in _arm_act_ids],
                    dtype=np.float64,
                )
                dac = wiring.dual_arm_ctrl
                if dac is not None and hasattr(dac, "_cmd_q"):
                    cmd = np.asarray(dac._cmd_q, dtype=np.float64).reshape(14)
                else:
                    cmd = ctrl.copy()

                dq = np.zeros(14, dtype=np.float64)
                if _j_prev_q[0] is not None and _j_prev_t[0] is not None:
                    dt = max(1e-4, now - _j_prev_t[0])
                    dq = (q - _j_prev_q[0]) / dt
                _j_prev_q[0] = q.copy()
                _j_prev_t[0] = now

                err = cmd - q
                tsc = manager.task_status_controller
                status = (
                    tsc.current_status.name
                    if tsc is not None and hasattr(tsc.current_status, "name")
                    else str(getattr(tsc, "current_status", "?"))
                )
                max_abs_dq = float(np.max(np.abs(dq)))
                max_abs_err = float(np.max(np.abs(err)))
                max_abs_q = float(np.max(np.abs(q)))

                try:
                    ee = env.query_site_pos_and_quat_B([_ee_l, _ee_r], [_base])
                    pl = np.asarray(ee[_ee_l]["xpos"], dtype=np.float64).reshape(3)
                    pr = np.asarray(ee[_ee_r]["xpos"], dtype=np.float64).reshape(3)
                    ee_str = (
                        f"L=({pl[0]:+.3f},{pl[1]:+.3f},{pl[2]:+.3f}) "
                        f"R=({pr[0]:+.3f},{pr[1]:+.3f},{pr[2]:+.3f})"
                    )
                except Exception:
                    ee_str = "L=(nan) R=(nan)"

                _ks = (
                    pico_device.pico_joystick.current_key_state
                    if pico_device is not None
                    else None
                )
                if _ks is not None:
                    _lh = _ks.get("leftHand", {}) or {}
                    _rh = _ks.get("rightHand", {}) or {}
                    _lp_raw = _lh.get("position", [0.0, 0.0, 0.0])
                    _rp_raw = _rh.get("position", [0.0, 0.0, 0.0])
                    _lq_raw = _lh.get("rotation", [0.0, 0.0, 0.0, 1.0])
                    _rq_raw = _rh.get("rotation", [0.0, 0.0, 0.0, 1.0])
                    _lt = float(_lh.get("triggerValue", 0.0))
                    _rt = float(_rh.get("triggerValue", 0.0))
                    _lgr = bool(_lh.get("gripButtonPressed", False))
                    _rgr = bool(_rh.get("gripButtonPressed", False))
                    lp_mj, lq_mj = _pico_ks_to_mj(_lp_raw, _lq_raw)
                    rp_mj, rq_mj = _pico_ks_to_mj(_rp_raw, _rq_raw)
                    pico_ok = True
                else:
                    _lp_raw = _rp_raw = [0.0, 0.0, 0.0]
                    _lq_raw = _rq_raw = [0.0, 0.0, 0.0, 1.0]
                    _lt = _rt = 0.0
                    _lgr = _rgr = False
                    lp_mj = rp_mj = np.zeros(3)
                    lq_mj = rq_mj = np.array([0.0, 0.0, 0.0, 1.0])
                    pico_ok = False

                _j_n[0] += 1
                diag.log(
                    f"[JOINT] #{_j_n[0]} status={status} "
                    f"max|q|={max_abs_q:.3f}rad max|dq|={max_abs_dq:.2f}rad/s "
                    f"max|cmd-q|={max_abs_err:.3f}rad ee={ee_str}"
                )
                diag.log(
                    f"[PICO_IN] ok={pico_ok} "
                    f"L_unity=({_lp_raw[0]:.3f},{_lp_raw[1]:.3f},{_lp_raw[2]:.3f}) "
                    f"R_unity=({_rp_raw[0]:.3f},{_rp_raw[1]:.3f},{_rp_raw[2]:.3f}) "
                    f"L_trig={_lt:.2f} R_trig={_rt:.2f} "
                    f"L_grip={_lgr} R_grip={_rgr}"
                )
                diag.log(
                    f"[PICO_MJ] "
                    f"L_pos=({lp_mj[0]:+.3f},{lp_mj[1]:+.3f},{lp_mj[2]:+.3f}) "
                    f"L_quat=({lq_mj[0]:+.3f},{lq_mj[1]:+.3f},{lq_mj[2]:+.3f},{lq_mj[3]:+.3f}) "
                    f"R_pos=({rp_mj[0]:+.3f},{rp_mj[1]:+.3f},{rp_mj[2]:+.3f}) "
                    f"R_quat=({rq_mj[0]:+.3f},{rq_mj[1]:+.3f},{rq_mj[2]:+.3f},{rq_mj[3]:+.3f})"
                )
                diag.log(f"[JOINT] L_q[{_fmt7_deg(q[:7])}]")
                diag.log(f"[JOINT] R_q[{_fmt7_deg(q[7:])}]")
                diag.log(f"[JOINT] L_cmd[{_fmt7_deg(cmd[:7])}]")
                diag.log(f"[JOINT] R_cmd[{_fmt7_deg(cmd[7:])}]")
                diag.log(f"[JOINT] L_err[{_fmt7_deg(err[:7])}]")
                diag.log(f"[JOINT] R_err[{_fmt7_deg(err[7:])}]")
                diag.log(
                    f"[JOINT] L_dq/s[{_fmt7_deg(dq[:7])}] "
                    f"(deg/s; 乱扭时看 sy/el/wr 是否大幅跳动)"
                )
                diag.log(f"[JOINT] R_dq/s[{_fmt7_deg(dq[7:])}]")
            except Exception as e:
                if _j_n[0] < 3 or _j_n[0] % 50 == 0:
                    diag.log(f"[JOINT] poll failed: {e}")


class EpisodeRunner:
    """采集主循环封装，替代 main() 内的 while 块与 _discard_episode_event 全局。"""

    def __init__(
        self,
        cfg: TeleopConfig,
        env,
        manager,
        writer,
        storage,
        scene_manager,
        diag: DiagLog,
        discard_event: threading.Event,
        scratch_dir: str,
        init_video_started: bool = False,
    ) -> None:
        self._cfg = cfg
        self._env = env
        self._manager = manager
        self._writer = writer
        self._storage = storage
        self._scene_manager = scene_manager
        self._diag = diag
        self._discard_event = discard_event
        self._scratch_dir = scratch_dir
        self.video_started = init_video_started

    def run(self) -> None:
        cfg = self._cfg
        env = self._env
        manager = self._manager
        writer = self._writer
        storage = self._storage
        discard_event = self._discard_event

        _ep_idx = 0
        while not manager._shutdown_requested:  # noqa: SLF001
            _ep_idx += 1
            env.reset()
            time.sleep(0.1)

            if not cfg.local_xml:
                if not manager.update_scene():
                    orca_logger.info("update_scene 失败，停止采集")
                    break

            ep_dir = None
            ep_start = None
            if cfg.camera_source == "mp4":
                ep_dir = os.path.join(self._scratch_dir, "mp4", f"ep_{_ep_idx:06d}")
                os.makedirs(os.path.join(ep_dir, "video"), exist_ok=True)
                ep_start = time.perf_counter()
                env.begin_save_video(ep_dir)
                self.video_started = True

            _collecting_no = writer.num_episodes + 1
            orca_logger.info(f"========== 第 {_collecting_no} 集 ==========")
            if cfg.xr_backend == "pico":
                print(f"\n>>> 第 {_collecting_no} 集（左Grip开始，再按保存）", flush=True)
            else:
                print(
                    f"\n>>> 第 {_collecting_no} 集（左squeeze开始，再按保存）",
                    flush=True,
                )

            _t0 = time.perf_counter()
            _ok, _start, _end, _qpos = manager.run_episode()
            _dur = time.perf_counter() - _t0
            _nframes = storage.buffered_frame_count

            if cfg.camera_source == "mp4" and self.video_started:
                try:
                    env.stop_save_video()
                except Exception:
                    pass
                self.video_started = False

            if discard_event.is_set():
                discard_event.clear()
                manager._shutdown_requested = False  # noqa: SLF001
                storage.clear_data()
                orca_logger.info(f"[EP {_ep_idx}] 已丢弃本集")
                print(f"[EP] discard cleared ep={_ep_idx}", flush=True)
                continue

            if manager._shutdown_requested:  # noqa: SLF001
                storage.clear_data()
                print(f"[EP] shutdown abort ep={_ep_idx}", flush=True)
                break

            _cap_fps = (_nframes / _dur) if _dur > 0 else 0.0
            orca_logger.info(f"[EP {_ep_idx}] {_dur:.1f}s / {_nframes}帧 / fps={_cap_fps:.1f}")
            print(
                f"[EP] save ep={_ep_idx} frames={_nframes} dur={_dur:.1f}s "
                f"reason=left_squeeze_end",
                flush=True,
            )
            storage.save_data(
                task_info=manager.task.get_task_info(),
                scene_info=self._scene_manager.get_scene_info(),
                task_description=manager.task.get_task_description(),
                episode_video_dir=ep_dir,
                ep_start_wall=ep_start,
            )
            orca_logger.info(f"已保存，共 {writer.num_episodes} 集 / {writer.num_frames} 帧")
            print(f">>> 已保存，共 {writer.num_episodes} 集", flush=True)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="g1_pick VR 遥操作采集 → LeRobot v2.1")
    parser.add_argument("--level", type=str, default="default", help="场景名称")
    parser.add_argument("--task_config", default="../common/example.yaml", help="任务配置 YAML")
    parser.add_argument(
        "--lerobot_out",
        default=DEFAULT_LEROBOT_OUT,
        help=(
            f"数据集输出目录（默认 {DEFAULT_LEROBOT_OUT}；"
            f"相对路径会落到 {DEFAULT_LEROBOT_ROOT}/ 下）"
        ),
    )
    parser.add_argument("--repo_id", default="local/g1_pick", help="LeRobot repo_id")
    parser.add_argument("--task", default="g1 pick teleoperation", help="任务描述")
    parser.add_argument("--fps", type=int, default=20, help="采集帧率")
    parser.add_argument("--clock", choices=("sim", "wall"), default="wall")
    parser.add_argument("--resume", action="store_true", help="追加到已有数据集")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument("--cameras", default="head,wrist_r",
                        help="启用的相机列表（g1_pick 仅 head + wrist_r）")
    parser.add_argument("--cam_resolution", default="480x640", help="分辨率 HxW")
    parser.add_argument("--camera_source", choices=("websocket", "mp4"), default="websocket")
    parser.add_argument("--local_xml", default=None,
                        help="使用本地 patched XML（跳过 gRPC 加载）")
    parser.add_argument("--scene_json", default="unitree_button.json",
                        help="场景 JSON 文件名（仅文档/提示用；实际场景以 OrcaStudio 已加载为准）")
    parser.add_argument(
        "--agent_name",
        default="g1_pick",
        help="仿真中机器人 actor 名（关节前缀）。unitree_button 请用 unitree_humanoid_robot_1",
    )
    parser.add_argument("--log_file", default=None,
                        help="终端输出日志路径（同时输出到文件和终端）")
    parser.add_argument("--diag_tele", action=argparse.BooleanOptionalAction, default=False,
                        help="启用分层遥操诊断日志")
    parser.add_argument("--diag_every", type=int, default=50,
                        help="诊断日志节流间隔（步），默认 50")
    parser.add_argument(
        "--xr_backend",
        choices=("pico", "televuer"),
        default="pico",
        help="XR 输入后端：pico（默认，现有逻辑）或 televuer",
    )
    parser.add_argument(
        "--tv_goal_mode",
        choices=("rebased_tv", "absolute_tv"),
        default="rebased_tv",
        help="TeleVuer 目标模式（默认 rebased_tv clutch）",
    )
    parser.add_argument(
        "--tv_ee_dx",
        type=float,
        default=0.03,
        help="TeleVuer→Orca EE 校正 TransX(m)，默认 0.03（0.05→0.08）",
    )
    parser.add_argument(
        "--tv_position_scale",
        type=float,
        default=1.0,
        help="TeleVuer 位置尺度，默认 1.0",
    )
    parser.add_argument(
        "--dry_run_tele",
        action="store_true",
        help="只计算/打印 IK 目标与误差，不驱动臂部 actuator",
    )
    parser.add_argument(
        "--tv_max_pos_jump",
        type=float,
        default=0.50,
        help="TeleVuer 单帧最大位置步进 (m)；超限会限速追赶而非永久拒收",
    )
    parser.add_argument(
        "--tv_max_ori_jump",
        type=float,
        default=90.0,
        help="TeleVuer 单帧最大姿态步进 (deg)；超限会限速追赶而非永久拒收",
    )
    parser.add_argument(
        "--tv_max_dq_step",
        type=float,
        default=0.8,
        help="每控制周期最大关节步长 (rad)；默认 0.8≈24rad/s@30Hz；过大时偶发猛甩可调小",
    )
    parser.add_argument(
        "--tv_deadzone_pos",
        type=float,
        default=0.006,
        help="末端位置死区半径 (m)；在死区内不更新目标，默认 0.006",
    )
    parser.add_argument(
        "--tv_deadzone_ori",
        type=float,
        default=2.0,
        help="末端姿态死区半径 (deg)；在死区内不更新目标，默认 2.0",
    )
    parser.add_argument(
        "--tv_goal_ema",
        type=float,
        default=0.95,
        help="目标 EMA 平滑系数，范围为 [0,1]；1 表示不平滑，默认值为 0.95",
    )
    parser.add_argument(
        "--ik_max_reach",
        type=float,
        default=0.44,
        help="肩→末端参考半径 (m)，默认 0.44；仅用于 [GOAL] 的 excess 诊断，"
        "以及 --ik_project_reachable 开启时的钳制半径",
    )
    parser.add_argument(
        "--ik_project_reachable",
        action="store_true",
        help="将超出范围的末端目标投影到 max_reach 球面",
    )
    parser.add_argument(
        "--diag_health",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用 TeleVuer 分层健康状态日志；也可通过环境变量 G1_DIAG_HEALTH=1 启用",
    )
    parser.add_argument(
        "--diag_joints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用双臂关节状态的后台诊断",
    )
    parser.add_argument(
        "--diag_joints_hz",
        type=float,
        default=5.0,
        help="关节诊断采样频率，单位 Hz，默认值为 5",
    )
    parser.add_argument(
        "--diag_log_dir",
        default="/tmp/g1pick_tele_diag",
        help="诊断日志保存目录，默认 /tmp/g1pick_tele_diag",
    )
    parser.add_argument(
        "--tv_no_tls",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "使用 USB/adb loopback 明文 HTTP/WS；未提供证书时默认开启。"
            "兼容旧启动命令中的 --tv_no_tls。"
        ),
    )
    parser.add_argument(
        "--tv_cert_file",
        default=None,
        help="显式 TLS 证书路径；必须与 --tv_key_file 同时提供",
    )
    parser.add_argument(
        "--tv_key_file",
        default=None,
        help="显式 TLS 私钥路径；必须与 --tv_cert_file 同时提供",
    )
    parser.add_argument(
        "--tv_host",
        default="127.0.0.1",
        help=(
            "TeleVuer 监听地址，默认 127.0.0.1（配合 adb reverse 的 USB 通道）。"
            "改为其他地址时必须同时提供 --tv_cert_file/--tv_key_file，"
            "因为 WebXR 只在安全上下文中进入沉浸式会话。"
        ),
    )
    parser.add_argument(
        "--arm_kp",
        type=float,
        default=150.0,
        help="臂部 position 执行器 kp（默认 150）",
    )
    parser.add_argument(
        "--arm_kv",
        type=float,
        default=None,
        help="臂部 position 执行器 kv；默认 arm_kv_ratio*kp",
    )
    parser.add_argument(
        "--arm_kv_ratio",
        type=float,
        default=0.11,
        help="kv=ratio*kp（默认 0.11）",
    )
    parser.add_argument(
        "--arm_gravcomp",
        type=float,
        default=1.0,
        help="臂+手 body 重力补偿比例 0~1（默认 1.0=全补偿；0 关闭）",
    )
    args = parser.parse_args()
    cert_supplied = bool(args.tv_cert_file or args.tv_key_file)
    if bool(args.tv_cert_file) != bool(args.tv_key_file):
        parser.error("--tv_cert_file and --tv_key_file must be provided together")
    if args.tv_no_tls is None:
        args.tv_no_tls = not cert_supplied
    elif args.tv_no_tls and cert_supplied:
        parser.error("--tv_no_tls cannot be combined with TLS certificate paths")
    elif not args.tv_no_tls and not cert_supplied:
        parser.error("--no-tv_no_tls requires --tv_cert_file and --tv_key_file")
    if cert_supplied:
        args.tv_cert_file = os.path.abspath(os.path.expanduser(args.tv_cert_file))
        args.tv_key_file = os.path.abspath(os.path.expanduser(args.tv_key_file))
        for _label, _path in (("certificate", args.tv_cert_file), ("private key", args.tv_key_file)):
            if not os.path.isfile(_path):
                parser.error(f"TeleVuer TLS {_label} does not exist: {_path}")
    if args.tv_no_tls and args.tv_host not in ("127.0.0.1", "localhost", "::1"):
        parser.error(
            f"--tv_host {args.tv_host} requires --tv_cert_file/--tv_key_file; "
            "plain HTTP is restricted to loopback"
        )

    if os.environ.get("G1_DIAG_HEALTH", "").strip() in ("1", "true", "TRUE", "yes", "YES"):
        args.diag_health = True

    cfg = TeleopConfig.from_args(args)

    # ── 诊断日志 ────────────────────────────────────────────────────────────
    diag = DiagLog(cfg)
    diag.log(
        f"[DIAG] tele diag={'ON' if cfg.diag_tele else 'OFF'} every={cfg.diag_every} "
        f"file={diag.log_path} latest_link={diag.latest_link} "
        f"solver=UnitreeG1_29_CasADi"
    )
    print(
        f"[RUN] teleop  diag_tele={cfg.diag_tele} "
        f"diag_joints={cfg.diag_joints} diag_health={cfg.diag_health}",
        flush=True,
    )
    if diag.enabled:
        print(
            f"[DIAG] 本次完整日志 → {diag.log_path}\n"
            f"[DIAG] 最新软链     → {diag.latest_link}  (tail -f 可用)",
            flush=True,
        )
        DataCollectionManager.set_debug_log_fh(diag._dbg_fh)

    # ── 相机配置 ─────────────────────────────────────────────────────────────
    _CAM_KEY_MAP = {
        "head": "camera_head_color",
        "wrist_l": "camera_wrist_l_color",
        "wrist_r": "camera_wrist_r_color",
    }
    G1_PICK_CAMERA_MAP = {
        "camera_head_color": ("cam_head", 7070),
        "camera_wrist_l_color": ("cam_wrist_l", 7060),
        "camera_wrist_r_color": ("cam_wrist_r", 7080),
    }
    _enabled = {k.strip() for k in cfg.cameras.split(",")}
    camera_map = {
        env_name: (key, port)
        for env_name, (key, port) in G1_PICK_CAMERA_MAP.items()
        if any(env_name == _CAM_KEY_MAP.get(k) for k in _enabled)
    }
    if not camera_map:
        orca_logger.warning("--cameras 未匹配到已知相机，回退全路")
        camera_map = G1_PICK_CAMERA_MAP

    try:
        _h, _w = (int(x) for x in cfg.cam_resolution.lower().split("x"))
        cam_hw_override = (_h, _w)
    except Exception:
        cam_hw_override = DEFAULT_HW

    # ── 初始关节值 ───────────────────────────────────────────────────────────
    default_joint_values: dict = {}
    for conf in [g1_pick_conf.l_arm, g1_pick_conf.r_arm,
                 g1_pick_conf.l_hand, g1_pick_conf.r_hand]:
        for jn, v in zip(conf["joint_names"], conf["neutral_joint_values"]):
            default_joint_values[jn] = v

    # ── VR 设备 ──────────────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("  g1_pick LeRobot 数采启动中...", flush=True)
    print(f"  场景: {cfg.level}  fps: {cfg.fps}  clock: {cfg.clock}", flush=True)
    print(f"  输出目录: {cfg.lerobot_out}", flush=True)
    print(f"  XR backend: {cfg.xr_backend}", flush=True)
    print("=" * 60, flush=True)

    xr_device = None
    pico_device = None
    if cfg.xr_backend == "pico":
        print("  等待 Pico 连接...", flush=True)
        pico_device = G1PickPicoJoystickDevice(PicoJoystick())
        xr_device = pico_device
    else:
        print("  初始化 TeleVuer（pass-through，无沉浸画面）...", flush=True)
        from devices.g1_pick_televuer_device import TeleVuerDevice

        pose_mapper = TvToOrcaPoseMapper(
            T_ee_correction=make_trans_x(cfg.tv_ee_dx),
            position_scale=cfg.tv_position_scale,
        )
        xr_device = TeleVuerDevice(
            pose_mapper=pose_mapper,
            display_mode="pass-through",
            img_shape=cam_hw_override,
            binocular=False,
            cert_file="" if cfg.tv_no_tls else cfg.tv_cert_file,
            key_file="" if cfg.tv_no_tls else cfg.tv_key_file,
            host=cfg.tv_host,
            log_buttons=False,
            evt_log=False,
        )
        print(
            f"  TeleVuer visit URL: {xr_device.client_visit_url()} "
            f"(tls={'off' if cfg.tv_no_tls else 'on'})",
            flush=True,
        )

    # ── 场景管理 ─────────────────────────────────────────────────────────────
    scene_config_path = os.path.abspath(os.path.join(base_dir, cfg.task_config))
    with open(scene_config_path, "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)

    scene_manager = SceneManager(cfg.orcagym_addr, config=scene_config)
    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.show_ui_message(1, "g1_pick 遥操作启动中", "0xffff00", showtime=10)
    scene_manager.get_scene_data(script_name, "beginscene")

    # ── Storage ──────────────────────────────────────────────────────────────
    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", "g1_pick", cfg.level)
    storage = G1PickLeRobotStorage(dataset_path=scratch_dir)

    def _obs_callback_safe(env):
        if env.model.nu == 0:
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(14, dtype=np.float32),
                "/action/drive/ctrl": np.zeros(0, dtype=np.float32),
            }
        return storage.obs_callback(env)

    # ── DataCollectionManager ────────────────────────────────────────────────
    agent_name = cfg.agent_name
    manager = DataCollectionManager(
        agent_name=agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback_safe,
        env_index=0,
        device=xr_device,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=cfg.orcagym_addr,
        env_kwargs=(
            {
                "skip_grpc_load": True,
                "local_xml_path": cfg.local_xml,
                "xml_assets_dir": os.path.expanduser("~/.orcagym/tmp"),
            }
            if cfg.local_xml else None
        ),
    )
    env = manager.env
    manager.save_video = False

    # gRPC 模式：拦截 XML 加载，写入修改后的副本并保留原始缓存。
    # ───────────────────────────────────────────────────────────────────────
    # 策略：_orig_load() 返回 ~/.orcagym/tmp/<uuid>.xml（原始，只读）
    #       本钩子在同目录生成 <uuid>_teleop_patched.xml 并返回其路径。
    #       MjModel.from_xml_path 只依赖路径，mesh/STL 与原始 XML 同目录，解析正常。
    #       原始 XML 永不修改，其他模块（外部分析/回放/开发）读到的始终是干净的原文件。
    if not cfg.local_xml:
        _orig_load = env.gym.load_model_xml
        _gc = float(cfg.arm_gravcomp)

        async def _patched_load_model_xml():
            orig_path = await _orig_load()
            with open(orig_path, "r") as f:
                xml = f.read()

            patches_applied: list[str] = []

            # ── 1. 骨盆 / 躯干 / 双足 weld（软约束兜底；硬钉靠 pin_floating_base）──
            # 注意：禁止删除 freejoint。删掉会使本地 nq 少 7，而 OrcaStudio 仍按
            # 原模型收 qpos，UpdateLocalEnv 映射错位，表现为右臂“锁死”。
            welds_xml = ""
            for name, body in [
                (f"{agent_name}_pelvis_weld", f"{agent_name}_pelvis"),
                (f"{agent_name}_torso_weld", f"{agent_name}_torso_link_rev_1_0"),
                (f"{agent_name}_left_foot_weld", f"{agent_name}_left_ankle_roll_link"),
                (f"{agent_name}_right_foot_weld", f"{agent_name}_right_ankle_roll_link"),
            ]:
                if name not in xml:
                    welds_xml += f'        <weld active="true" name="{name}" body1="{body}" body2="world" solref="0.02 1" solimp="0.9 0.95 0.001"/>\n'
            if welds_xml:
                if "</equality>" in xml:
                    xml = xml.replace("</equality>", welds_xml + "</equality>", 1)
                else:
                    xml = xml.replace("</actuator>",
                                      "</actuator>\n    <equality>\n" + welds_xml + "    </equality>", 1)
                patches_applied.append("weld_base")

            # ── 2. 臂+手 body gravcomp（MuJoCo 原生重力补偿，不改 nq）─────────
            _arm_hand_links = [
                f"{agent_name}_{side}_{link}"
                for side in ("left", "right")
                for link in (
                    "shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link",
                    "elbow_link",
                    "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link",
                    "hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link",
                    "hand_index_0_link", "hand_index_1_link",
                    "hand_middle_0_link", "hand_middle_1_link",
                )
            ]
            gc_injected = 0
            if _gc > 0.0:
                for link_name in _arm_hand_links:
                    marker = f'name="{link_name}"'
                    if marker not in xml:
                        continue
                    idx = xml.index(marker)
                    body_start = xml.rfind("<body", 0, idx)
                    if "gravcomp=" in xml[body_start:idx + len(marker)]:
                        continue
                    xml = xml.replace(marker, f'gravcomp="{_gc}" {marker}', 1)
                    gc_injected += 1
                if gc_injected > 0:
                    patches_applied.append(f"gravcomp={_gc}(n={gc_injected})")

            # 写入修改后的副本，保留原始文件。
            import pathlib
            orig = pathlib.Path(orig_path)
            patched_path = str(orig.with_stem(orig.stem + "_teleop_patched"))
            with open(patched_path, "w") as f:
                f.write(xml)
            return patched_path

        env.gym.load_model_xml = _patched_load_model_xml

    # 本地 XML 模式：禁用 OrcaStudio 渲染
    if cfg.local_xml:
        orca_logger.info("local_xml 模式：禁用 gRPC 渲染")
        _orig_render = env.unwrapped.render

        def _noop_render(**kwargs):
            pass

        env.unwrapped.render = _noop_render
        env.unwrapped._skip_grpc_load = True

    # ── 初始化：env.reset + 控制器装配 + 相机 ────────────────────────────────
    cameras: dict = {}
    cam_hw = cam_hw_override
    video_started = False
    wiring = ControllerWiring()

    try:
        env.reset()
        time.sleep(0.1)

        scene_ok = True
        if not cfg.local_xml:
            scene_ok = manager.update_scene()
        else:
            orca_logger.info("local_xml 模式：跳过 gRPC update_scene，使用预缓存场景")

        if scene_ok:
            env.set_default_joint_values(default_joint_values)
            wiring.wire(cfg, env, manager, xr_device, pico_device, diag)

            orca_logger.info(f"尝试启用相机: {list(camera_map.keys())}")
            try:
                if cfg.camera_source == "websocket" and not cfg.local_xml:
                    os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
                    env.begin_save_video(STREAM_TRIGGER_PATH)
                    video_started = True
                    cameras = bring_up_cameras(camera_map)
                    camera_map = {n: v for n, v in camera_map.items() if n in cameras}
                    if cameras:
                        cam_hw = probe_camera_hw(cameras, camera_map, default_hw=cam_hw_override)
                    else:
                        orca_logger.warning("没有可用相机，将以无视频模式采集")
                elif cfg.local_xml:
                    orca_logger.info("local_xml 模式：跳过相机初始化（无 OrcaStudio 渲染）")
                else:
                    orca_logger.info("mp4 模式：跳过 WebSocket 相机连接")
            except Exception as cam_err:
                orca_logger.warning(f"相机初始化失败（将继续无视频模式）: {cam_err}")

    except KeyboardInterrupt:
        orca_logger.info("初始化阶段收到 Ctrl+C")
    except Exception as e:
        orca_logger.error(f"初始化失败: {e}\n{traceback.format_exc()}")

    cam_shape = (3, cam_hw[0], cam_hw[1])
    if cameras:
        orca_logger.info(f"相机分辨率 {cam_hw[0]}x{cam_hw[1]}, fps={cfg.fps}, 路数={len(cameras)}")

    # ── 健康监控线程 ──────────────────────────────────────────────────────────
    monitor = HealthMonitor(cfg, env, xr_device, pico_device, manager, wiring, diag)
    monitor.start()

    # ── 用户提示 ─────────────────────────────────────────────────────────────
    print("", flush=True)
    print("=" * 60, flush=True)
    print("  g1_pick VR 遥操作采集 (Unitree G1_29 CasADi IK)", flush=True)
    print(f"  任务: {cfg.task}", flush=True)
    print(f"  XR backend: {cfg.xr_backend}", flush=True)
    print(f"  数据输出: {cfg.lerobot_out}", flush=True)
    print("-" * 60, flush=True)
    if cfg.xr_backend == "pico":
        print("  【操作按键 · Pico】", flush=True)
        print("  左臂移动    已锁定（左臂全程静止）", flush=True)
        print("  右臂移动    右手柄位姿", flush=True)
        print("  左手抓握    左扳机", flush=True)
        print("  右手抓握    右扳机", flush=True)
        print("-" * 60, flush=True)
        print("  【采集流程】", flush=True)
        print("  左Grip×1=开始  左Grip×2=保存", flush=True)
        print("  右Grip=丢弃重置  左右Grip同按=退出", flush=True)
        ui_msg = "左Grip×1=开始 左Grip×2=保存 右Grip=丢弃 左右Grip同按=退出"
    else:
        print("  【操作按键 · TeleVuer】", flush=True)
        print("  左臂移动    已锁定（左臂全程静止）", flush=True)
        print("  右臂跟随    右手柄位姿（rebased_tv）", flush=True)
        print("  左手抓握    左扳机", flush=True)
        print("  右手抓握    右扳机", flush=True)
        print("-" * 60, flush=True)
        print("  【采集流程】", flush=True)
        print("  左 squeeze=开始/保存  右 squeeze=丢弃", flush=True)
        print("  左右 squeeze 同按=退出", flush=True)
        if cfg.dry_run_tele:
            print("  DRY-RUN：不驱动臂部 actuator，仅打印目标/误差", flush=True)
        ui_msg = "左squeeze=开始/保存 右squeeze=丢弃 左右同按=退出"
    print("  IDLE 时手臂 hold（不跑 IK）", flush=True)
    if cfg.diag_joints:
        print(
            f"  关节诊断 [JOINT] @ {cfg.diag_joints_hz:.0f}Hz → {diag.log_path}",
            flush=True,
        )
    print("=" * 60, flush=True)

    try:
        scene_manager.show_ui_message(1, ui_msg, "0x00ff00", showtime=0)
    except Exception as e:
        orca_logger.warning(f"show_ui_message 失败（orcalab 可能看不到操作提示）: {e}")

    # ── 采集主循环 ───────────────────────────────────────────────────────────
    writer = None
    runner = None
    try:
        writer = LeRobotDatasetWriter.create(
            repo_id=cfg.repo_id,
            root=cfg.lerobot_out,
            fps=cfg.fps,
            camera_map=camera_map,
            state_dim=storage.state_dim,
            state_names=storage.state_names,
            action_dim=storage.action_dim,
            action_names=storage.action_names,
            cam_shape=cam_shape,
            resume=cfg.resume,
            robot_type="g1_pick_q_delta",
        )
        storage.configure_lerobot(
            fps=cfg.fps, cameras=cameras, camera_map=camera_map,
            target_hw=cam_hw, writer=writer, task=cfg.task,
            clock=cfg.clock, camera_source=cfg.camera_source,
        )
        with writer:
            runner = EpisodeRunner(
                cfg, env, manager, writer, storage, scene_manager,
                diag, wiring.discard_event, scratch_dir,
                init_video_started=video_started,
            )
            runner.run()

    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt")
        print("\n[停止] 采集已中断", flush=True)
    except Exception as e:
        orca_logger.error(f"采集异常: {e}\n{traceback.format_exc()}")
    finally:
        monitor.stop()
        if writer is not None:
            try:
                writer.close()
                orca_logger.info("视频编码完成")
            except Exception:
                pass
        _video_active = runner.video_started if runner is not None else video_started
        if _video_active:
            try:
                env.stop_save_video()
            except Exception:
                pass
        close_cameras(cameras)
        if cfg.xr_backend == "televuer" and xr_device is not None:
            try:
                xr_device.close()
                orca_logger.info("TeleVuer closed")
            except Exception:
                pass
        try:
            env.close()
        except Exception:
            pass
        diag.close()
        s = f"结束，共 {writer.num_episodes if writer else 0} 集"
        orca_logger.info(s)
        print(
            f"\n{'='*60}\n"
            f"  {s}\n"
            f"  数据: {cfg.lerobot_out}\n"
            f"{'='*60}",
            flush=True,
        )


# =============================================================================
# 正式版日志：装配/注入类默认静默；排障用 *_test.py + --diag_*
# 保留：TeleVuer 初始化与 URL、采集起停/保存/丢弃、关键 IK 状态、结束摘要
# =============================================================================


if __name__ == "__main__":
    _xr_dev_for_cleanup = None
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as e:
        OrcaLog.get_instance().error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        os._exit(0)
