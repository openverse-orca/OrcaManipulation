import argparse
import os
import signal
import sys
import threading
import time
import traceback


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from yaml import load, Loader

from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import get_orca_logger
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers.controllers import install_osc_patches
from controllers.controller_task import TaskStatus
from controllers.controller_dex1_force import add_dex1_force_pico_controller
from conf import g1_dex1_osc_conf
from dataStorage.g1_dex1_lerobot_storage import (
    DEFAULT_HW,
    G1Dex1LeRobotStorage,
    LeRobotDatasetWriter,
    bring_up_cameras,
    camera_map_for,
    close_cameras,
    probe_camera_hw,
)

from orca_gym.adapters.robosuite.controllers import (
    controller_config,
    controller_factory,
)

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "g1_dex1_collection.log"

orca_logger = get_orca_logger(
    name="G1Dex1Collection",
    log_file=log_file,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


# ── 基座/腰/腿姿态钉住（自 Binjiang 竞赛栈移植）───────────────────────────

def _pin_all_joints_writeback(env, agent_name: str) -> bool:
    """旧机制（回退用）：每步 qpos 写回钉住基座、腰部和腿部。

    写回式钉住每步把关节硬拉回参考值，会向夹爪注入高频微振动，
    破坏摩擦接触的动量守恒，导致夹持物匀速爬滑脱落。
    仅在 equality 钉住不可用时使用。
    """
    import mujoco

    gym = getattr(env, "gym", None) or getattr(
        getattr(env, "unwrapped", env), "gym", None
    )
    if gym is None or not hasattr(gym, "_mjModel") or not hasattr(gym, "_mjData"):
        orca_logger.warning("[CONSTRAINT] 姿态约束初始化失败：模型不可用")
        return False

    mj, md = gym._mjModel, gym._mjData

    def _joint_qadr_dadr(short_name: str) -> tuple[int, int]:
        full = f"{agent_name}_{short_name}"
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, full)
        if jid < 0:
            raise ValueError("姿态约束初始化失败：模型关节不兼容")
        return int(mj.jnt_qposadr[jid]), int(mj.jnt_dofadr[jid])

    # 基座参考位姿
    base_qadr, base_dadr = _joint_qadr_dadr("floating_base_joint")
    base_q0 = np.array(md.qpos[base_qadr : base_qadr + 7], dtype=np.float64, copy=True)

    # 腰部参考姿态
    waist_names = g1_dex1_osc_conf.locked_waist_joints
    waist_qadrs, waist_dadrs, waist_q0s = [], [], []
    for short in waist_names:
        qa, da = _joint_qadr_dadr(short)
        waist_qadrs.append(qa)
        waist_dadrs.append(da)
        waist_q0s.append(float(md.qpos[qa]))

    # 腿部参考姿态（全 motor 化后腿部无位置伺服, 必须同样钉住防止瘫软）
    leg_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ]
    leg_qadrs, leg_dadrs, leg_q0s = [], [], []
    for short in leg_names:
        qa, da = _joint_qadr_dadr(short)
        leg_qadrs.append(qa)
        leg_dadrs.append(da)
        leg_q0s.append(float(md.qpos[qa]))

    md.qpos[base_qadr : base_qadr + 7] = base_q0
    md.qvel[base_dadr : base_dadr + 6] = 0.0
    for qa, q0 in zip(waist_qadrs, waist_q0s):
        md.qpos[qa] = q0
    for da in waist_dadrs:
        md.qvel[da] = 0.0
    mujoco.mj_forward(mj, md)

    _orig_mj_step = gym.mj_step

    def _mj_step_all_pinned(nstep=1):
        n = int(nstep) if nstep is not None else 1
        for _ in range(max(n, 1)):
            md.qpos[base_qadr : base_qadr + 7] = base_q0
            md.qvel[base_dadr : base_dadr + 6] = 0.0
            for qa, q0 in zip(waist_qadrs, waist_q0s):
                md.qpos[qa] = q0
            for da in waist_dadrs:
                md.qvel[da] = 0.0
            for qa, q0 in zip(leg_qadrs, leg_q0s):
                md.qpos[qa] = q0
            for da in leg_dadrs:
                md.qvel[da] = 0.0

            _orig_mj_step(1)

            md.qpos[base_qadr : base_qadr + 7] = base_q0
            md.qvel[base_dadr : base_dadr + 6] = 0.0
            for qa, q0 in zip(waist_qadrs, waist_q0s):
                md.qpos[qa] = q0
            for da in waist_dadrs:
                md.qvel[da] = 0.0
            for qa, q0 in zip(leg_qadrs, leg_q0s):
                md.qpos[qa] = q0
            for da in leg_dadrs:
                md.qvel[da] = 0.0

        mujoco.mj_forward(mj, md)

    gym.mj_step = _mj_step_all_pinned
    orca_logger.info("[CONSTRAINT] 写回式姿态约束已启用（回退机制）")
    return True


def _pin_all_joints_equality(env, agent_name: str) -> bool:
    """以求解器内 equality 约束钉住基座、腰部和腿部。

    模型 XML 中预定义了 eq_active="false" 的 pin_* 约束
    （1 个基座 weld + 15 个腰/腿 joint equality），此处按启动姿态激活：
      - joint equality：polycoef[0] 写入当前关节角，钉在当前姿态；
      - 基座 weld：使用编译时捕获的参考位姿。
    equality 约束由求解器内部求解，动量守恒，不产生写回式钉住的
    高频微振动，夹持物不会爬滑。
    """
    import mujoco

    gym = getattr(env, "gym", None) or getattr(
        getattr(env, "unwrapped", env), "gym", None
    )
    if gym is None or not hasattr(gym, "_mjModel") or not hasattr(gym, "_mjData"):
        orca_logger.warning("[CONSTRAINT] 姿态约束初始化失败：模型不可用")
        return False

    mj, md = gym._mjModel, gym._mjData

    def _joint_id(short_name: str) -> int:
        jid = mujoco.mj_name2id(
            mj, mujoco.mjtObj.mjOBJ_JOINT, f"{agent_name}_{short_name}"
        )
        if jid < 0:
            raise ValueError("姿态约束初始化失败：模型关节不兼容")
        return int(jid)

    pin_short_names = list(g1_dex1_osc_conf.locked_waist_joints) + [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ]

    # 定位预定义 pin 约束槽位（按类型+对象匹配，不依赖 equality 命名）
    joint_slots: list = []
    missing: list = []
    for short in pin_short_names:
        jid = _joint_id(short)
        slot = next(
            (
                e for e in range(mj.neq)
                if mj.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT
                and int(mj.eq_obj1id[e]) == jid
            ),
            -1,
        )
        if slot < 0:
            missing.append(f"pin_{short}")
        else:
            joint_slots.append((slot, jid))

    # 基座 weld（body1=pelvis, body2=world，允许两侧互换）
    base_body = int(mj.jnt_bodyid[_joint_id("floating_base_joint")])
    weld_slot = next(
        (
            e for e in range(mj.neq)
            if mj.eq_type[e] == mujoco.mjtEq.mjEQ_WELD
            and (
                (int(mj.eq_obj1id[e]) == base_body and int(mj.eq_obj2id[e]) == 0)
                or (int(mj.eq_obj2id[e]) == base_body and int(mj.eq_obj1id[e]) == 0)
            )
        ),
        -1,
    )
    if weld_slot < 0:
        missing.append("pin_floating_base(weld)")

    if missing:
        orca_logger.warning(
            "[CONSTRAINT] 模型缺少钉住约束: "
            + ", ".join(missing)
            + "（请导入含 pin_* 约束的 g1_dex1 模型）"
        )
        return False

    for slot, jid in joint_slots:
        mj.eq_data[slot, 0] = float(md.qpos[int(mj.jnt_qposadr[jid])])
        mj.eq_active0[slot] = 1  # env.reset 后保持激活
        md.eq_active[slot] = 1
    mj.eq_active0[weld_slot] = 1
    md.eq_active[weld_slot] = 1
    mujoco.mj_forward(mj, md)

    orca_logger.info(
        "[CONSTRAINT] 姿态钉住已启用（equality 求解器约束）: "
        f"基座 weld + {len(joint_slots)} 个腰/腿关节约束，动量守恒，抓取无爬滑"
    )
    return True


# 参考 Binjiang_Competition g1_dex1_osc_conf.l_arm["neutral_joint_values"]:
# 肩部 roll 稍外展 0.127rad(~7.3°)，避免左臂贴住躯干/桌面。采集与回放共用。
_L_ARM_DOCK = [0.0, 0.127, 0.0, 1.5708, 0.0, 0.0, 0.0]


def _pin_left_arm(env, agent_name: str) -> bool:
    """SouthGrid pin_all_joints 的左臂部分，原样搬过来。"""
    import mujoco

    gym = getattr(env, "gym", None) or getattr(
        getattr(env, "unwrapped", env), "gym", None
    )
    if gym is None or not hasattr(gym, "_mjModel") or not hasattr(gym, "_mjData"):
        orca_logger.warning("[CONSTRAINT] 姿态约束初始化失败：模型状态不可用")
        return False

    mj, md = gym._mjModel, gym._mjData

    def _joint_qadr_dadr(short_name: str) -> tuple[int, int]:
        full = f"{agent_name}_{short_name}"
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, full)
        if jid < 0:
            raise ValueError("姿态约束初始化失败：模型关节不兼容")
        return int(mj.jnt_qposadr[jid]), int(mj.jnt_dofadr[jid])

    def _actuator_id(short_name: str) -> int:
        full = f"{agent_name}_{short_name}"
        aid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
        if aid < 0:
            raise ValueError("姿态约束初始化失败：模型执行器不兼容")
        return aid

    l_arm_joint_names = g1_dex1_osc_conf.l_arm["joint_names"]
    l_arm_motor_names = g1_dex1_osc_conf.l_arm["motors_names"]
    l_arm_target = list(_L_ARM_DOCK)
    larm_qadrs: list[int] = []
    larm_dadrs: list[int] = []
    for short in l_arm_joint_names:
        qa, da = _joint_qadr_dadr(short)
        larm_qadrs.append(qa)
        larm_dadrs.append(da)
    larm_act_ids: list[int] = [_actuator_id(short) for short in l_arm_motor_names]

    for qa, q0 in zip(larm_qadrs, l_arm_target):
        md.qpos[qa] = float(q0)
    for da in larm_dadrs:
        md.qvel[da] = 0.0
    for aid in larm_act_ids:
        md.ctrl[aid] = 0.0
    mujoco.mj_forward(mj, md)

    if getattr(gym, "_g1_left_arm_pinned", False):
        return True

    _orig_mj_step = gym.mj_step

    def _mj_step_larm_pinned(nstep=1):
        n = int(nstep) if nstep is not None else 1
        for _ in range(max(n, 1)):
            for qa, q0 in zip(larm_qadrs, l_arm_target):
                md.qpos[qa] = float(q0)
            for da in larm_dadrs:
                md.qvel[da] = 0.0
            for aid in larm_act_ids:
                md.ctrl[aid] = 0.0
            _orig_mj_step(1)
            for qa, q0 in zip(larm_qadrs, l_arm_target):
                md.qpos[qa] = float(q0)
            for da in larm_dadrs:
                md.qvel[da] = 0.0
            for aid in larm_act_ids:
                md.ctrl[aid] = 0.0
        mujoco.mj_forward(mj, md)

    gym.mj_step = _mj_step_larm_pinned
    gym._g1_left_arm_pinned = True
    orca_logger.info("[CONSTRAINT] 左臂停靠姿态约束已启用")
    return True


def pin_all_joints(env, agent_name: str) -> bool:
    """基座、腰、腿用 equality 钉住；左臂按 SouthGrid 写回停靠姿态。"""
    if not _pin_all_joints_equality(env, agent_name):
        orca_logger.warning(
            "[CONSTRAINT] equality 钉住不可用，回退为 qpos 写回钉住"
            "（旧机制：机器人可站立，但夹持物可能缓慢滑落）"
        )
        if not _pin_all_joints_writeback(env, agent_name):
            return False
    return _pin_left_arm(env, agent_name)


def apply_grasp_solver_tuning(env) -> bool:
    """强制抓取求解器参数，防场景合并覆盖 XML option。

    MuJoCo 软摩擦约束存在"爬滑"：夹持法向力足够时物体仍以恒定速度滑出。
    OrcaLab 场景合并可能用场景 option 覆盖机器人 XML 的 option，故运行时
    直接写本地 mjModel.opt 强制生效。
    """
    gym = getattr(env, "gym", None) or getattr(
        getattr(env, "unwrapped", env), "gym", None
    )
    if gym is None or not hasattr(gym, "_mjModel"):
        orca_logger.warning("[GRIP] 抓取求解器参数设置失败：模型不可用")
        return False
    opt = gym._mjModel.opt
    opt.impratio = 50.0            # 摩擦/法向阻抗比(官方抓取推荐 10-100)
    opt.noslip_iterations = 3      # 防滑后处理: 每步清除摩擦接触残余滑移
    opt.noslip_tolerance = 1e-6
    orca_logger.info(
        f"[GRIP] 抓取求解器参数已强制: impratio={opt.impratio:.0f} "
        f"noslip_iterations={opt.noslip_iterations}"
    )
    return True


# ── OSC 手臂：kp 覆盖 + 静止稳定滤波 + pos_scale ─────────────────────────

class StableGoalFilter:
    """手柄输入静止稳定滤波（自 Binjiang 竞赛栈移植）。

    VR 手柄静止时位姿仍有毫米级噪声，经 OSC 高增益放大后表现为机械臂
    持续微晃，影响夹试管等精细操作：
      死区锁定: 输入相对上次接受值的位移/旋转变化小于死区 → 目标不变
      尖峰确认: 超死区后需连续 2 个新 TCP 采样确认才接受，滤单帧噪声
      EMA 低通: 接受的输入按 ema_alpha 平滑
      pos_scale: 手柄位移→末端目标位移的比例（精细模式 <1）
    """

    def __init__(self, update_goal, pos_scale=1.0,
                 pos_deadzone=0.003, rot_deadzone=0.0175, ema_alpha=0.35):
        self._update_goal = update_goal
        self.pos_scale = float(pos_scale)
        self.pos_deadzone = float(pos_deadzone)
        self.rot_deadzone = float(rot_deadzone)
        self.ema_alpha = float(ema_alpha)
        self._filt_pos = None
        self._filt_quat = None
        self._last_raw_pos = None
        self._last_raw_quat = None
        self._exceed_count = 0

    def reset(self):
        self._filt_pos = None
        self._filt_quat = None
        self._last_raw_pos = None
        self._last_raw_quat = None
        self._exceed_count = 0

    def __call__(self, relative_position, relative_quat):
        rel_pos = np.asarray(relative_position, dtype=np.float64).reshape(3)
        rel_quat = np.asarray(relative_quat, dtype=np.float64).reshape(4)
        rel_quat = rel_quat / max(float(np.linalg.norm(rel_quat)), 1e-12)

        if self._filt_pos is None:
            self._filt_pos = rel_pos
            self._filt_quat = rel_quat
            self._last_raw_pos = rel_pos
            self._last_raw_quat = rel_quat
            self._exceed_count = 0
        else:
            same_sample = (
                self._last_raw_pos is not None
                and np.array_equal(rel_pos, self._last_raw_pos)
                and np.array_equal(rel_quat, self._last_raw_quat)
            )
            self._last_raw_pos = rel_pos
            self._last_raw_quat = rel_quat
            if same_sample:
                return  # 同一 TCP 采样重复到达: 不改变任何状态

            pos_delta = float(np.linalg.norm(rel_pos - self._filt_pos))
            rot_delta = 2.0 * np.arccos(
                min(1.0, abs(float(np.dot(rel_quat, self._filt_quat))))
            )
            if pos_delta < self.pos_deadzone and rot_delta < self.rot_deadzone:
                self._exceed_count = 0
                return  # 手柄静止: 保持目标不变
            self._exceed_count += 1
            if self._exceed_count < 2:
                return  # 等待下一个采样确认(过滤单帧尖峰)

            a = float(np.clip(self.ema_alpha, 0.05, 1.0))
            filt_pos = a * rel_pos + (1.0 - a) * self._filt_pos
            q_blend = a * rel_quat + (1.0 - a) * self._filt_quat
            if float(np.dot(q_blend, self._filt_quat)) < 0.0:
                q_blend = -q_blend  # 四元数双覆盖: 统一符号防止插值绕远
            self._filt_pos = filt_pos
            self._filt_quat = q_blend / max(float(np.linalg.norm(q_blend)), 1e-12)
            self._exceed_count = 2  # 运动中保持, 后续超死区采样直接接受

        self._update_goal(self.pos_scale * self._filt_pos, self._filt_quat)


def create_arm_osc_controller_kp(env, arm_config, base_body, ctrl_name, init_ctrl):
    """复刻 controllers.create_arm_osc_controller，支持 conf 中 osc_kp 覆盖。

    robosuite 默认 kp=150 偏软（末端偏 10cm 才 15N 恢复力），大惯量手臂
    快速遥操滞后明显；conf osc_kp=300 提升跟手度（kv 由工厂按临界阻尼
    自动重算）。
    """
    from controllers.controller_arm import ControllerArm

    arm_joint_names = [
        env.joint(joint_name) for joint_name in arm_config["joint_names"]
    ]
    qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(arm_joint_names)
    joint_indexes = {
        "joints": arm_joint_names,
        "qpos": qpos_offsets,
        "qvel": qvel_offsets,
    }
    motors_ranges = [
        [r[0] for r in arm_config["motors_ranges"]],
        [r[1] for r in arm_config["motors_ranges"]],
    ]

    osc_config = controller_config.load_config("osc_pose")
    osc_config["sim"] = env.gym
    osc_config["eef_name"] = env.site(arm_config["ee_site_name"])
    osc_config["joint_indexes"] = joint_indexes
    osc_config["actuator_range"] = motors_ranges
    osc_config["policy_freq"] = 1.0 / env.dt
    osc_config["ndim"] = len(arm_joint_names)
    osc_config["control_delta"] = False
    if arm_config.get("osc_kp"):
        osc_config["kp"] = arm_config["osc_kp"]

    controller = controller_factory(osc_config["type"], osc_config)
    controller.update_initial_joints(arm_config["neutral_joint_values"])
    return ControllerArm(env, ctrl_name, init_ctrl, base_body, controller)


def add_arm_osc_stable_pico_controller(
    data_collection_manager,
    env,
    arm_config,
    base_body,
    device,
    key,
):
    """OSC 手臂控制器 + 静止稳定滤波（本仓库 controller_arm 不含滤波，
    故在绑定层包一层 StableGoalFilter，不改动仓库核心文件）。"""
    ctrl_name = [env.actuator(m) for m in arm_config["motors_names"]]
    init_ctrl = {n: v for n, v in zip(ctrl_name, arm_config["motors_init_ctrl"])}
    arm_ctrl = create_arm_osc_controller_kp(
        env, arm_config, base_body, ctrl_name, init_ctrl
    )
    stable_filter = StableGoalFilter(
        arm_ctrl.update_goal,
        pos_scale=arm_config.get("pos_scale", 1.0),
        pos_deadzone=arm_config.get("pos_deadzone", 0.003),
        rot_deadzone=arm_config.get("rot_deadzone", 0.0175),
        ema_alpha=arm_config.get("ema_alpha", 0.35),
    )
    # episode 重置时同步清空滤波状态（控制器 reset 后重新接受当前手柄位姿）
    _orig_reset = arm_ctrl.reset

    def _reset_with_filter():
        _orig_reset()
        stable_filter.reset()

    arm_ctrl.reset = _reset_with_filter
    device.bind_transform_event(key, stable_filter)
    data_collection_manager.add_controller(arm_ctrl)
    return arm_ctrl


def add_task_status_pico_controller_edge(
    data_collection_manager,
    env,
    device,
    base_body,
):
    """任务状态控制器（边沿触发版）。

    仓库版 add_task_status_pico_controller 直接绑定 grip 按键状态:
    手柄事件每控制帧回调, 按住期间持续为 True, TaskStatusController
    每 0.2s 翻一页状态(NOT_STARTED→RUNNING→END), 长按 >0.4s 会连翻
    两页, 表现为"一按采集就立即结束"。
    此处在绑定层加按下沿检测: 仅 False→True 跳变时翻转一次,
    按多久都只算一次, 不改动仓库核心文件。
    """
    from controllers.controller_task import TaskStatusController

    task_status_controller = TaskStatusController(env, base_body)
    state = {"prev": False}

    def _on_grip(pressed: bool):
        if pressed and not state["prev"]:
            task_status_controller.update_task_status(True)
        state["prev"] = pressed

    device.bind_grip_button_event(PicoJoystickKey.L_GRIPBUTTON, _on_grip)
    data_collection_manager.set_task_status_controller(task_status_controller)


def use_southgrid_gripper_mapping(gripper, latch_rate: float = 0.5):
    """DEX1 夹爪改用 SouthGrid 2F85 的手柄语义。

    扳机深度 → 闭合程度，松开扳机即张开；
    按住 Y/B → 闭合下限以 latch_rate/s 上升（松开扳机也保持夹紧）；
    按住 X/A → 闭合下限以同速率下降。
    实际闭合程度 = max(扳机深度, 闭合下限)，经阻抗律换算电机力。
    """
    state = {"latch": 0.0}
    span = gripper.q_close - gripper.q_open
    orig_reset = gripper.reset

    def _run_controller():
        dt = float(gripper.env.dt)
        if gripper.secondary_button:
            state["latch"] = min(1.0, state["latch"] + latch_rate * dt)
        elif gripper.primary_button:
            state["latch"] = max(0.0, state["latch"] - latch_rate * dt)
        k = np.e
        depth = (np.exp(k * float(gripper.trigger_value)) - 1.0) / (np.exp(k) - 1.0)
        norm = float(np.clip(max(depth, state["latch"]), 0.0, 1.0))
        gripper.q_target = gripper.q_open + span * norm
        tau = gripper._impedance_force(gripper.q_target)
        return {i: tau for i in gripper.ctrl_index}

    def _reset():
        orig_reset()
        state["latch"] = 0.0

    gripper.run_controller = _run_controller
    gripper.reset = _reset
    return gripper


# ── 相机推流（orcagym 26.9.1 VideoRecorderManager: 连接+缓冲+remux 一体）─────
# 旧引擎接口 BeginSaveMp4File 已被 OrcaLab 移除(UNIMPLEMENTED),
# 改用 start_streaming 推流 + save_streaming 按帧区间 remux 出 MP4。

# 逻辑相机角色关键词 → 后端注册名需包含的 token（后端名带 uuid 后缀）
_CAMERA_ROLE_TOKENS = {
    "head": ("head_cam", "head"),
    "wrist_r": ("camera_right", "right"),
    "wrist_l": ("camera_left", "left"),
}
# 推流端口兜底值（后端未分配端口时按角色重试, 与 Binjiang 流程一致）
_CAMERA_FALLBACK_PORTS = {"head": 7070, "wrist_l": 7072, "wrist_r": 7074}
_PORT_TO_ROLE = {port: role for role, port in _CAMERA_FALLBACK_PORTS.items()}
_sigint_armed = False


def _sigint_immediate(signum, frame):
    """Ctrl+C 立即打断当前等待。再按一次直接退出，不等清理。"""
    global _sigint_armed
    if _sigint_armed:
        os._exit(130)
    _sigint_armed = True
    raise KeyboardInterrupt


def _resolve_cameras(env, logical_keys: list[str]) -> dict[str, str]:
    """逻辑相机名 → 后端注册相机名（含 uuid 后缀），按角色关键词匹配。"""
    try:
        registered = list(env.get_camera_names())
    except Exception:
        registered = []
    resolved = {}
    for logical in logical_keys:
        if not registered or logical in registered:
            resolved[logical] = logical
            continue
        role = next((r for r in _CAMERA_ROLE_TOKENS if r in logical), None)
        cands: list[str] = []
        if role:
            for tok in _CAMERA_ROLE_TOKENS[role]:
                cands = [r for r in registered if tok in r]
                if cands:
                    break
        else:
            cands = [r for r in registered if logical in r]
        # 多候选取最短（主实体名短于派生名）
        resolved[logical] = min(cands, key=len) if cands else logical
    return resolved


def _render_ticks(env, seconds: float):
    """渲染若干秒（引擎仅在 render 调用时编码推流相机帧）。"""
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            env.render()
        except Exception:
            pass
        time.sleep(0.1)


def _camera_frames_flowing(env, manager, cam: str, probe_seconds: float = 0.8) -> bool:
    """探测相机帧是否在流动（probe 期间持续 render 触发引擎推流）。

    引擎只在 render() 调用时编码推流相机帧——等首帧期间必须主动渲染，
    否则永远收不到帧（Binjiang bring_up_cameras 的 render_fn 同理）。
    故无论是否已收到首帧, 都先渲染一个窗口再比较帧计数。
    健康信号用 buffered_frame_count（帧计数增长）; 不用 simulate_index
    ——引擎未启用索引计数器时帧的 sim_idx=-1，会误判。
    """
    recorder = manager.get_recorder(cam)
    if recorder is None:
        return False
    c0 = recorder.buffered_frame_count
    _render_ticks(env, probe_seconds)
    c1 = recorder.buffered_frame_count
    return c1 > c0


def _enable_simulate_index(env):
    """启用 env 内部 simulate_index 计数器（render 显式传一次 ≥0 索引）。

    未启用时引擎帧头 sim_idx=-1，录制器区间圈定失效（无法定位集的
    起止帧）。启用后 run_episode 里无参 render() 自动自增传递。
    幂等：计数器已启用（≥0）时不重置。
    """
    if getattr(env, "_current_simulate_index", -1) < 0:
        try:
            env.render(simulate_index=0)
            orca_logger.info("[相机] simulate_index 计数器已启用")
        except Exception as e:
            orca_logger.warning(f"[相机] 启用 simulate_index 失败: {e}")


def ensure_camera_streams(
    env, logical_keys: list[str], first_frame_timeout: float = 30.0
) -> dict[str, str]:
    """确保所有相机推流+录制器在线且帧在流动，返回 {逻辑名: 后端注册名}。

    每集调用一次：场景重载(env.reset/update_scene)可能打断推流或更换
    后端相机名，此处先重新解析名字，再对每路做健康检查——
    帧不流动则重启推流并等待首帧。全部失败时返回空 dict（HDF5 照常）。
    """
    if not logical_keys:
        return {}
    # 推流 API 仅 orca-gym 26.9.1+ 提供; 旧版(orcalab 环境的 26.7.3)会静默无视频
    if not hasattr(env, "start_streaming") or not hasattr(env, "get_recorder_manager"):
        orca_logger.error(
            "[相机] 当前 orca-gym 不含推流接口(start_streaming/get_recorder_manager), "
            "本集仅保存 HDF5。请改用 orcalab_lerobot 环境(orca-gym 26.9.1)"
        )
        return {}
    resolved = _resolve_cameras(env, logical_keys)
    try:
        manager = env.get_recorder_manager()
    except Exception as e:
        orca_logger.warning(f"[相机] 无法获取录制管理器: {e}")
        return {}
    online: dict[str, str] = {}
    for logical, cam in resolved.items():
        # 健康检查: 已有录制器且帧在流动 → 直接沿用
        if manager.get_recorder(cam) is not None and _camera_frames_flowing(
            env, manager, cam
        ):
            online[logical] = cam
            continue
        # 重启: 移除旧录制器 → 启动推流（端口缺省时按角色兜底）→ 等首帧
        try:
            manager.remove_recorder(cam)
        except Exception:
            pass
        try:
            env.start_streaming(cam, capture_rgb=True)
        except ValueError:
            try:
                env.start_streaming(
                    cam,
                    capture_rgb=True,
                    color_port=_CAMERA_FALLBACK_PORTS.get(logical, 7070),
                )
            except Exception as e:
                orca_logger.warning(f"[相机] {logical}({cam}) 推流启动失败: {e}")
                continue
        except Exception as e:
            orca_logger.warning(f"[相机] {logical}({cam}) 推流启动失败: {e}")
            continue
        # 启用 sim_idx 计数器: 录制器一启动就传, 保证后续到达帧索引有效
        _enable_simulate_index(env)
        # 等首帧（render 触发引擎编码推流）
        deadline = time.time() + first_frame_timeout
        while time.time() < deadline:
            if _camera_frames_flowing(env, manager, cam, probe_seconds=0.4):
                online[logical] = cam
                orca_logger.info(f"[相机] ✓ {logical}({cam}) 推流就绪")
                break
            time.sleep(0.2)
        if logical not in online:
            orca_logger.warning(f"[相机] {logical}({cam}) 等待首帧超时, 跳过")
    if not online:
        try:
            registered = list(env.get_camera_names())
        except Exception:
            registered = []
        orca_logger.error(
            f"[相机] 无可用相机, 本集仅保存 HDF5; 后端注册相机={registered}。"
            "排查: 1) 环境需 orcalab_lerobot(26.9.1) "
            "2) 场景中相机组件勾选 Enable/ColorCamera/IsRecording "
            "3) 相机名需含 head_cam|camera_left|camera_right "
            "4) 在 OrcaLab 中加载 src/scene/g1_pick_tube.json 还原相机"
        )
    return online


# ── 主流程 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="G1 + DEX1-1 夹爪 VR 遥操作数采（LeRobot v2.1）"
    )
    parser.add_argument("--level", type=str, default="default", help="场景的名称")
    parser.add_argument(
        "--agent_name", default="g1_pick_usda",
        help="OrcaLab 场景中的 agent 名称（须与资产前缀一致）",
    )
    parser.add_argument(
        "--task_config", default="example.yaml", help="场景配置 YAML 文件名"
    )
    parser.add_argument(
        "--osc_kp", type=float, default=None,
        help="OSC 任务空间刚度 kp（默认取 conf 中 300）",
    )
    parser.add_argument(
        "--pos_scale", type=float, default=None,
        help="手柄位移→末端位移比例（默认取 conf 中 1.0；精细操作建议 0.3~0.5）",
    )
    parser.add_argument(
        "--pos_deadzone", type=float, default=None,
        help="手柄输入位移死区/米（默认 0.003；静止仍微抖调大，移动发涩调小）",
    )
    parser.add_argument(
        "--ema_alpha", type=float, default=None,
        help="手柄输入 EMA 平滑系数（默认 0.35；越小越稳越滞后）",
    )
    parser.add_argument(
        "--dls_lambda", type=float, default=0.23,
        help="OSC 阻尼最小二乘最大系数 λ_max（与 SouthGrid 相同；0 表示用伪逆，手臂伸直会抖）",
    )
    parser.add_argument(
        "--dls_sigma_th", type=float, default=0.12,
        help="自适应阻尼触发阈值 σ_th（与 SouthGrid 相同；0 表示固定阻尼）",
    )
    parser.add_argument(
        "--null_kp", type=float, default=10.0,
        help="零空间关节复原增益（默认 10，临界阻尼自动计算）",
    )
    parser.add_argument(
        "--gripper_mode", choices=("southgrid", "hold"), default="southgrid",
        help="夹爪按键语义。southgrid：扳机按多深夹多紧、松开即张开，Y/B 按住锁紧、X/A 按住松开；"
        "hold：旧 DEX1 语义，松开扳机保持，X/A 全开，Y/B 半开",
    )
    parser.add_argument(
        "--time_step", type=float, default=0.001, help="MuJoCo 物理步长（秒）"
    )
    parser.add_argument(
        "--frame_skip", type=int, default=5, help="每控制周期的物理子步数"
    )
    parser.add_argument(
        "--cameras", type=str, default="head,wrist_r",
        help="写入 LeRobot 的相机，逗号分隔（head/wrist_r/wrist_l）。默认与 SouthGrid 相同：头和右腕",
    )
    parser.add_argument(
        "--no_cameras", action="store_true",
        help="不写图像，只写 11 维 state/action（右臂位姿 + 双手夹爪）",
    )
    parser.add_argument(
        "--cam_resolution",
        default="480x640",
        help="图像高x宽，默认 480x640，与 SouthGrid 相同",
    )
    parser.add_argument(
        "--lerobot_out",
        default=os.path.join(base_dir, "lerobot_datasets", "g1_dex1"),
        help="LeRobot 数据集目录",
    )
    parser.add_argument("--repo_id", default="local/g1_dex1", help="LeRobot repo_id")
    parser.add_argument(
        "--task",
        default="g1 dex1 tube teleoperation",
        help="写入每帧的任务描述",
    )
    parser.add_argument("--fps", type=int, default=20, help="采集帧率，默认 20")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=0,
        help="成功保存多少集后退出。0 表示不限，一直采到 Ctrl+C",
    )
    parser.add_argument(
        "--clock",
        choices=("sim", "wall"),
        default="wall",
        help="采帧时钟。wall 用系统时间，sim 用仿真时间",
    )
    parser.add_argument("--resume", action="store_true", help="追加到已有数据集")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    args = parser.parse_args()
    if args.num_episodes < 0:
        parser.error("--num_episodes 不能为负。0 表示一直采集，直到 Ctrl+C")

    level = args.level
    orcagym_addr = args.orcagym_addr

    # CLI 覆盖 conf 参数（双臂同步）
    if args.osc_kp is not None:
        g1_dex1_osc_conf.l_arm["osc_kp"] = args.osc_kp
        g1_dex1_osc_conf.r_arm["osc_kp"] = args.osc_kp
    if args.pos_scale is not None:
        g1_dex1_osc_conf.l_arm["pos_scale"] = args.pos_scale
        g1_dex1_osc_conf.r_arm["pos_scale"] = args.pos_scale
    if args.pos_deadzone is not None:
        g1_dex1_osc_conf.l_arm["pos_deadzone"] = args.pos_deadzone
        g1_dex1_osc_conf.r_arm["pos_deadzone"] = args.pos_deadzone
    if args.ema_alpha is not None:
        g1_dex1_osc_conf.l_arm["ema_alpha"] = args.ema_alpha
        g1_dex1_osc_conf.r_arm["ema_alpha"] = args.ema_alpha

    install_osc_patches(
        dls_lambda=args.dls_lambda,
        dls_sigma_th=args.dls_sigma_th,
        null_kp=args.null_kp,
    )
    if args.dls_lambda > 0.0:
        mode = "自适应" if args.dls_sigma_th > 0.0 else "固定"
        orca_logger.info(
            f"[CONTROL] OSC {mode}阻尼已启用 λ_max={args.dls_lambda} "
            f"σ_th={args.dls_sigma_th} null_kp={args.null_kp}"
        )
    else:
        orca_logger.warning("[CONTROL] OSC 未启用阻尼，手臂接近伸直时会抖")

    orca_logger.info(
        f"[CONTROL] OSC kp={g1_dex1_osc_conf.l_arm.get('osc_kp', 150)}, "
        f"pos_scale={g1_dex1_osc_conf.l_arm.get('pos_scale', 1.0)}, "
        f"静止滤波: deadzone={g1_dex1_osc_conf.l_arm.get('pos_deadzone', 0.003)}m/"
        f"{np.rad2deg(g1_dex1_osc_conf.l_arm.get('rot_deadzone', 0.0175)):.1f}°, "
        f"ema_alpha={g1_dex1_osc_conf.l_arm.get('ema_alpha', 0.35)}"
    )

    # 与 SouthGrid 相同：左臂停靠，右臂零位。
    default_joint_values = {}
    for jn, v in zip(g1_dex1_osc_conf.l_arm["joint_names"], _L_ARM_DOCK):
        default_joint_values[jn] = v
    for jn, v in zip(
        g1_dex1_osc_conf.r_arm["joint_names"],
        [0.0] * len(g1_dex1_osc_conf.r_arm["joint_names"]),
    ):
        default_joint_values[jn] = v

    print("=" * 60, flush=True)
    print("  G1 + DEX1 数采启动中...", flush=True)
    print(f"  场景: {level}  agent: {args.agent_name}", flush=True)
    print("  等待 Pico 连接...", flush=True)
    print("=" * 60, flush=True)

    orca_logger.info("Creating VR device")
    pico_joystick_device = PicoJoystickDevice(PicoJoystick())

    orca_logger.info("Creating scene manager")
    with open(
        os.path.join(base_dir, args.task_config), "r", encoding="utf-8"
    ) as f:
        config = load(f, Loader=Loader)
    # 覆盖 agent_joint_prefix 以匹配实际 agent 名称
    # (example.yaml 默认 g1_omnipicker_, 须改成命令行指定的 agent 前缀,
    #  否则录制/回放的关节过滤会错位)
    if "data_collection" in config:
        config["data_collection"]["agent_joint_prefix"] = f"{args.agent_name}_"
    else:
        config["data_collection"] = {"agent_joint_prefix": f"{args.agent_name}_"}
    scene_manager = SceneManager(orcagym_addr, config=config)

    script_name = (
        os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    )
    scene_manager.show_ui_message(
        1, "开始仿真程序，请按左右遥杆进行操作", "0xffff00", showtime=10
    )
    scene_manager.get_scene_data(script_name, "beginscene")

    lerobot_out = os.path.abspath(os.path.expanduser(args.lerobot_out))
    try:
        cam_h, cam_w = (int(x) for x in args.cam_resolution.lower().split("x"))
        cam_hw_override = (cam_h, cam_w)
    except Exception:
        orca_logger.warning(
            f"--cam_resolution 无法解析 '{args.cam_resolution}'，改用 {DEFAULT_HW}"
        )
        cam_hw_override = DEFAULT_HW
    camera_roles = (
        []
        if args.no_cameras
        else [c.strip() for c in args.cameras.split(",") if c.strip()]
    )
    camera_map = camera_map_for(camera_roles)

    orca_logger.info("Creating LeRobot data storage")
    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", "g1_dex1", level)
    data_storage = G1Dex1LeRobotStorage(dataset_path=scratch_dir)

    orca_logger.info("Creating data collection manager")
    data_collection_manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=data_storage.obs_callback,
        env_index=0,
        device=pico_joystick_device,
        scene_manager=scene_manager,
        data_storage=data_storage,
        frame_skip=args.frame_skip,
        time_step=args.time_step,
        orcagym_addr=orcagym_addr,
    )
    signal.signal(signal.SIGINT, _sigint_immediate)
    env = data_collection_manager.env
    env.reset()
    time.sleep(0.1)

    # ── DEX1 夹爪控制器（扳机深度→目标开度，阻抗律换算电机力）──────────
    # 左臂/左手全程停靠锁定, 不接控制器, 避免左手电机与钉住约束互扯导致抖动。
    orca_logger.info("Left arm/gripper disabled (docked)")
    orca_logger.info("Adding right DEX1 gripper controller")
    gripper_r = add_dex1_force_pico_controller(
        data_collection_manager,
        env,
        g1_dex1_osc_conf.gripper_r,
        g1_dex1_osc_conf.base_body,
        pico_joystick_device,
        [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
    )
    if args.gripper_mode == "southgrid":
        use_southgrid_gripper_mapping(gripper_r)
    orca_logger.info(f"[GRIP] 夹爪按键语义: {args.gripper_mode}")

    # 南网不给左臂接 OSC，左手柄位姿全程无效。
    orca_logger.info("Adding right arm OSC controller")
    add_arm_osc_stable_pico_controller(
        data_collection_manager,
        env,
        g1_dex1_osc_conf.r_arm,
        g1_dex1_osc_conf.base_body,
        pico_joystick_device,
        PicoJoystickKey.R_TRANSFORM,
    )

    # 抓取求解器参数 + 基座/腰/腿姿态钉住: 不在此处一次性激活,
    # 而是每集 reset/update_scene 后重新应用(防 reset/spawn 清除激活状态)
    apply_grasp_solver_tuning(env)
    pin_all_joints(env, args.agent_name)

    # Task + 任务状态（左Grip: 边沿触发, 开始采集 → 再按结束并保存）
    orca_logger.info("Setting task and task status controller")
    data_collection_manager.set_task(EmptyTask(env))
    add_task_status_pico_controller_edge(
        data_collection_manager, env, pico_joystick_device,
        g1_dex1_osc_conf.base_body,
    )

    # 图像由 LeRobot 写盘模块从相机流取帧。save_video 保持关闭，
    # 避免管理器再走 HDF5 的 begin_save_video。
    data_collection_manager.save_video = False
    cameras: dict = {}
    cam_hw = cam_hw_override
    if camera_map:
        orca_logger.info(f"启用相机: {list(camera_map.values())}")
        print("[场景] 机器人已就绪，正在启动相机推流...", flush=True)
        cameras = bring_up_cameras(env, camera_map)
        camera_map = {name: spec for name, spec in camera_map.items() if name in cameras}
        if cameras:
            cam_hw = probe_camera_hw(cameras, camera_map, default_hw=cam_hw_override)
        else:
            orca_logger.error("没有可用相机，退出。只要关节数据时请加 --no_cameras")
            close_cameras(cameras)
            try:
                env.close()
            except Exception:
                pass
            return
    else:
        print("[场景] 机器人已就绪，相机已关闭", flush=True)

    # 采集前输入门控（同 SouthGrid）：开始采集前只放行左 Grip，
    # 手臂和夹爪不响应手柄；RUNNING 后放行全部按键。
    # 左臂位姿全程不响应手柄，与 SouthGrid _LOCKED_KEYS 相同。
    locked_keys = {PicoJoystickKey.L_TRANSFORM}
    all_pico_keys = [k for k in pico_joystick_device.keys if k not in locked_keys]
    pre_start_keys = [k for k in all_pico_keys if k == PicoJoystickKey.L_GRIPBUTTON]

    def _gated_pico_update():
        tsc = data_collection_manager.task_status_controller
        if tsc is not None and tsc.current_status == TaskStatus.RUNNING:
            pico_joystick_device.pico_joystick.update(all_pico_keys)
        else:
            pico_joystick_device.pico_joystick.update(pre_start_keys)

    pico_joystick_device.update = _gated_pico_update

    # 右 Grip 单按：丢弃本集并重置场景；左右 Grip 同按：结束全部采集（同 SouthGrid）
    monitor_stop = threading.Event()
    discard_episode = threading.Event()
    grip_debounce_s = 0.3

    def _grip_monitor():
        r_only_prev = False
        both_prev = False
        last_fire = 0.0
        while not monitor_stop.wait(0.02):
            try:
                pj = pico_joystick_device.pico_joystick
                raw = pj.current_key_state
                if not pj.clients or raw is None:
                    continue
                l_grip = bool((raw.get("leftHand") or {}).get("gripButtonPressed"))
                r_grip = bool((raw.get("rightHand") or {}).get("gripButtonPressed"))
                both = l_grip and r_grip
                r_only = r_grip and not l_grip
                now = time.perf_counter()
                if now - last_fire >= grip_debounce_s:
                    if both and not both_prev:
                        orca_logger.info("[Grip] 左右Grip同按 → 结束全部采集")
                        try:
                            scene_manager.show_ui_message(
                                1, "采集终止，等待保存...", "0xff4400", showtime=0
                            )
                        except Exception:
                            pass
                        data_collection_manager._shutdown_requested = True
                        last_fire = now
                    elif r_only and not r_only_prev:
                        orca_logger.info("[Grip] 右Grip单按 → 丢弃本集，重置场景")
                        try:
                            scene_manager.show_ui_message(
                                1, "已丢弃，重置场景...", "0xff0000", showtime=2
                            )
                        except Exception:
                            pass
                        discard_episode.set()
                        data_collection_manager._shutdown_requested = True
                        last_fire = now
                r_only_prev = r_only
                both_prev = both
            except Exception:
                pass

    threading.Thread(target=_grip_monitor, daemon=True).start()

    print("", flush=True)
    print("=" * 60, flush=True)
    print("  【操作按键】（与 SouthGrid 一致）", flush=True)
    print("  注意        开始采集前手臂和夹爪不响应手柄", flush=True)
    print("  左臂/左夹爪  已禁用，锁在停靠姿态（左扳机 / X / Y 无效）", flush=True)
    print("  右臂移动    右手柄位姿", flush=True)
    if args.gripper_mode == "southgrid":
        print("  右夹爪      右扳机按多深夹多紧，松开张开 / 按住B锁紧 / 按住A解锁", flush=True)
    else:
        print("  右夹爪      右扳机闭合，松开保持 / A全开 / B半开", flush=True)
    print("  开始采集    轻按一下【左手柄 Grip】", flush=True)
    print("  结束保存    再轻按一下【左手柄 Grip】（无论成功与否均保存）", flush=True)
    print("  放弃本集    轻按【右手柄 Grip】（丢弃数据，重置场景）", flush=True)
    print("  结束全部    【左右 Grip 同时按下】（丢弃当前未保存集，等编码完成退出）", flush=True)
    print("  强制退出    终端 Ctrl+C（再按一次立即退出）", flush=True)
    if camera_map:
        print(
            f"  相机        {', '.join(key for key, _port in camera_map.values())}"
            f"  {cam_hw[0]}x{cam_hw[1]}  {args.fps} fps",
            flush=True,
        )
    else:
        print("  相机        已关闭（只写 11 维 state/action）", flush=True)
    if args.num_episodes > 0:
        print(f"  采集条数    {args.num_episodes}", flush=True)
    else:
        print("  采集条数    不限，Ctrl+C 结束", flush=True)
    print(f"  数据输出    {lerobot_out}", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)
    try:
        scene_manager.show_ui_message(
            1,
            "左Grip×1=开始 左Grip×2=保存 右Grip=丢弃重置 左右Grip同按=退出",
            "0x00ff00",
            showtime=0,
        )
    except Exception:
        pass

    # 每集 reset → update_scene → 重新钉住 → run_episode。钉住逻辑不改。
    writer = None
    try:
        writer = LeRobotDatasetWriter.create(
            repo_id=args.repo_id,
            root=lerobot_out,
            fps=args.fps,
            camera_map=camera_map,
            state_dim=data_storage.state_dim,
            state_names=data_storage.state_names,
            cam_shape=(3, cam_hw[0], cam_hw[1]),
            resume=args.resume,
            robot_type="g1_dex1",
        )
        data_storage.configure_lerobot(
            fps=args.fps,
            cameras=cameras,
            camera_map=camera_map,
            target_hw=cam_hw,
            writer=writer,
            task=args.task,
            clock=args.clock,
            camera_source="websocket",
        )
        orca_logger.info(
            f"进入采集主循环，LeRobot 输出: {lerobot_out}，"
            f"条数: {args.num_episodes if args.num_episodes > 0 else '不限'}"
        )
        saved_this_run = 0
        with writer:
            while not data_collection_manager._shutdown_requested:
                env.reset()
                time.sleep(0.1)
                if not data_collection_manager.update_scene():
                    orca_logger.info("update_scene 失败，停止采集")
                    break
                env.set_default_joint_values(default_joint_values)

                apply_grasp_solver_tuning(env)
                pin_all_joints(env, args.agent_name)

                episode_no = saved_this_run + 1
                quota = (
                    f"/{args.num_episodes}" if args.num_episodes > 0 else ""
                )
                orca_logger.info(
                    f"========== 第 {episode_no}{quota} 集"
                    "（左Grip×1 开始采集, 左Grip×2 结束保存）=========="
                )
                print(
                    f"\n>>> 第 {episode_no}{quota} 集: 按左Grip开始采集, 完成后再按左Grip保存",
                    flush=True,
                )

                task_is_success, _rec_start, _rec_end, _init_qpos = (
                    data_collection_manager.run_episode()
                )

                if discard_episode.is_set():
                    discard_episode.clear()
                    data_collection_manager._shutdown_requested = False
                    data_storage.clear_data()
                    orca_logger.info("已丢弃本集（右Grip），重置场景")
                    print(">>> 已丢弃本集，重置场景", flush=True)
                    continue

                if data_collection_manager._shutdown_requested:
                    data_storage.clear_data()
                    orca_logger.info("已请求停止，丢弃当前集")
                    break

                if task_is_success:
                    episodes_before = writer.num_episodes
                    data_storage.save_data(task=args.task)
                    saved_this_run += writer.num_episodes - episodes_before
                    orca_logger.info(
                        f"✓ 本集已保存，本次 {saved_this_run} 集，"
                        f"数据集共 {writer.num_episodes} 集 / {writer.num_frames} 帧"
                    )
                    print(
                        f">>> ✓ 本集已保存，本次 {saved_this_run} 集，"
                        f"数据集共 {writer.num_episodes} 集",
                        flush=True,
                    )
                    if args.num_episodes > 0 and saved_this_run >= args.num_episodes:
                        orca_logger.info(
                            f"已保存 {saved_this_run} 集，达到 --num_episodes，停止"
                        )
                        print(
                            f">>> 已保存 {saved_this_run} 集，达到 --num_episodes，停止",
                            flush=True,
                        )
                        break
                else:
                    data_storage.clear_data()
                    orca_logger.info("任务失败，本集已丢弃")
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止采集")
        try:
            data_storage.clear_data()
        except Exception:
            pass
    finally:
        monitor_stop.set()
        if writer is not None:
            try:
                print("\n[退出] 正在等待视频编码完成...", flush=True)
                writer.close()
                orca_logger.info("视频编码已结束")
            except Exception:
                pass
        close_cameras(cameras)
        try:
            env.reset()
            time.sleep(0.1)
            env.close()
        except Exception:
            pass


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
