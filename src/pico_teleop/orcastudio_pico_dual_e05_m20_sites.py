#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pico 双手遥操作脚本，对应新模型：
  dual_arm_mujoco_mobile_m20_pico_sites.xml

模型特点（dual_arm_mujoco_mobile_m20_pico_sites.xml）：
  - 可移动轮式底盘：浮动平面底座（世界 x/y 平移 + 车体偏航，高度锁定 0.58m），
    物理稳定不侧翻；4 个轮子（2 前轮带转向铰链、4 轮滚动铰链）为视觉模型，
    后轮滚动角用 equality 与同侧前轮耦合。
  - 身体 XT_Link / DT_Link / XB_Link / TB_Link 已 bake 为固定姿态
    （截图角度 XT-0.39/DT0.02/XB0.08/TB-0.28），无关节、无执行器。
  - 双臂为左/右 E05，各 6 个关节 + position 执行器。
  - 末端新增两个 site：left_eef_site / right_eef_site（M20 法兰前方 0.155m），
    IK 直接以这两个 site 为末端目标。
  - 左/右 M20 夹爪为单电机双指机构：jaw_a / jaw_b 两个 slide 关节
    （行程 [-0.0135, 0]，0 为全开，-0.0135 为闭合）由 equality 等值耦合
    对称闭合；每爪 1 个 velocity 电机执行器驱动 jaw_a，不用 position
    位置伺服，开度位置由脚本 M20TriggerController 做速度闭环决定。

控制映射（与智元 G1 遥操作一致）：
  Pico 左摇杆 (L_JOYSTICK_POSITION) x -> 底盘转向（阿克曼，前轮偏转+车体偏航）
  Pico 右摇杆 (R_JOYSTICK_POSITION) y -> 底盘前进/后退速度
  Pico 左手位姿 (L_TRANSFORM)         -> 操作者视角左侧臂 E05 末端位姿（custom IK）
  Pico 右手位姿 (R_TRANSFORM)         -> 操作者视角右侧臂 E05 末端位姿（custom IK）
  左/右 Grip 键                       -> clutch 模式下解锁/锁定对应手臂
  左扳机 (L_TRIGGER)                  -> 视角左侧臂的 M20 夹爪开合（连续量）
  右扳机 (R_TRIGGER)                  -> 视角右侧臂的 M20 夹爪开合（连续量）

左右配对说明（重要）：
  本模型 left_elfin / right_elfin 的命名与其物理安装侧相反（FK 实测：
  right_elfin 装在车体系 +x=车体左侧、left_elfin 在 -x=车体右侧）。
  站在车尾遥操时，操作者视角左边的臂是 right_elfin。因此默认按物理侧
  配对：左手柄->right_elfin、右手柄->left_elfin（即 --swap-hands 默认开），
  "动哪只手、视角哪边的臂就动"。若你站车头前面对机器人操控（此时视角
  左右与车体左右一致），用 --no-swap-hands 切回按模型命名配对。

启动安全：
  不强制写入猜测的中性姿态，而是读取 OrcaStudio 中机器人当前关节角作为
  IK 的初始关节值与初始位置指令，避免上电瞬间手臂跳动。

运行前提：
  1. OrcaStudio 已加载该 XML 并启动 gRPC 仿真（默认 localhost:50051）；
  2. Pico 端 OrcaGymCtrl 已连接（adb reverse tcp:8001 tcp:8001）。
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation as R

# ---- 让脚本在 urdf 目录下也能找到 OrcaManipulation 的 src ----
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for candidate in (
    THIS_DIR,
    os.path.abspath(os.path.join(THIS_DIR, "src")),
    os.path.expanduser("~/OrcaManipulation/src"),
    os.path.expanduser("~/桌面/OrcaManipulation/src"),
    os.path.abspath(os.path.join(THIS_DIR, "../../../OrcaManipulation/src")),
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

from controllers.abstract_controller import AbstractController
from controllers.controllers import create_arm_ik_controller
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import get_orca_logger

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

LOGGER = get_orca_logger(
    name="PicoDualE05Sites",
    log_file="pico_dual_e05_sites.log",
    max_bytes=5 * 1024 * 1024,
    backup_count=3,
    console_level="INFO",
    file_level="INFO",
    log_dir=THIS_DIR,
    use_colors=True,
    force_reinit=True,
)

# ---------------------------- 机器人配置 ----------------------------

# OrcaStudio 导入该 XML 时给模型名字加的命名空间前缀（agent 前缀 dual_arm_
# 会由 env.joint()/env.actuator()/env.site() 自动拼接，这里不含 agent 名）。
DEFAULT_NAMESPACE = "mujoco_mobile_m20_pico_sites_usda_"

# E05 六个关节的位置范围（弧度），与 XML ctrlrange 一致。
ARM_RANGES = [
    (-3.14, 3.14),  # joint1
    (-2.35, 2.35),  # joint2
    (-2.61, 2.61),  # joint3
    (-3.14, 3.14),  # joint4
    (-2.56, 2.56),  # joint5
    (-3.14, 3.14),  # joint6
]

# M20 slide 关节：XML range [-0.0135, 0]。
# 扳机 0 -> 全开 (0)，扳机 1 -> 闭合 (-0.0135)。模型读取失败时的回退常量。
M20_OPEN = 0.0
M20_CLOSED = -0.0135

# M20 夹爪电机速度环参数（位置目标 -> 电机速度指令）：
#   v = GRIPPER_KP_POS * (target - qpos)，限幅 ±GRIPPER_V_MAX
#   （与 XML velocity 执行器 ctrlrange ±0.05 一致）；无位置积分，
#   堵转/夹持力由执行器 forcerange ±35N 限住，与旧 position 伺服一致。
GRIPPER_KP_POS = 6.0   # 1/s：误差 >8.3mm 即满速，全程 ~0.3s 收敛
GRIPPER_V_MAX = 0.05   # m/s

# ---------------------------- 遥操作预备姿势 ----------------------------
#
# 重要：预备姿势必须是"肘弯曲、远离全伸展"的构型，不能把手摆在可达边界。
# 旧姿势夹爪距肩 1.086m（reach_limit=1.09，前伸余量仅 4mm），位置雅可比
# 最小奇异值仅 0.064（近奇异）。此时手柄让夹爪"往外/前伸"，物理上已无空间，
# DLS 只能输出大量关节转动而夹爪几乎不动——表现为"关节乱转、夹爪不跟随"。
#
# 下列新姿势由运动学数值逆解得到：夹爪位于肩前约 0.60m、外侧约 0.10m、
# 下方约 0.30m，双夹爪左右严格对称（±0.388, ~0.00, ~0.95），肘自然弯曲；
# 且夹爪(site 局部+z)水平指向车头正前方（世界 -y，与正前方夹角 <0.1°），
# 两夹爪互相平行——手柄在中立位朝前时夹爪也朝前，要它朝内/朝外只需小幅
# 转手腕（旧角度只约束了位置，夹爪保留了"朝外撇"的姿态，导致朝内要把手
# 腕拧很大）。姿态逆解保留原绕夹爪轴的 roll（只纠正偏航/俯仰），手腕最自然：
#   - 夹爪距肩 ~0.70m，各方向可达余量 ~0.39m（旧姿势仅 0.004m）；
#   - 位置雅可比最小奇异值 0.28（旧 0.064，提升 4 倍多），近奇异消除；
#   - λ=0.03 时径向/向上单步跟随率 99%、横向 84%，且每 0.1m 位移仅需
#     关节转 ~0.35rad（旧奇异点需 3.47rad，夹爪却不动）；
#   - 双臂在此姿势无自碰撞/互碰（ncon=0）。
#
# 若与你的站姿/手柄高度不符，直接改这六个角度即可（弧度），
# 或运行时加 --no-ready 保持 OrcaStudio 当前姿态。
READY_LEFT_Q = [0.19714, -0.38095, 1.99844, -0.90174, -1.0374, 0.80766]
READY_RIGHT_Q = [2.94165, 0.3835, -1.99618, 0.89903, 1.03642, -0.89842]

# ---------------------------- 底盘（虚拟阿克曼）参数 ----------------------------
#
# 原模型轮子是底座 mesh 的固定造型、不能改外观，因此底盘用世界系
# slide x/y + 车体 yaw 三个浮动速度关节（z 高度锁定），整车平移+转向。
# Python 端按自行车模型把摇杆量解算为车体速度：
#   v = throttle_y * MAX_SPEED                    (m/s，车头朝 NOSE_Y_SIGN·y 一侧)
#   δ = steering_x * MAX_STEER                    (虚拟前轮转角，rad)
#   yaw_rate = v * tan(δ) / WHEELBASE             (rad/s)
#   世界系 vx = -NOSE_Y_SIGN·v·sin(yaw),  vy = NOSE_Y_SIGN·v·cos(yaw)
MAX_SPEED = 1.2          # m/s，右摇杆满量程前进速度
MAX_STEER = 0.6          # rad，左摇杆满量程对应虚拟前轮转角（转向手感用）
WHEELBASE = 0.27         # m，前后轴距离，决定转向半径（视觉轮 y 向间距 0.135*2）
STEER_DEAD_ZONE = 0.1    # 左摇杆死区（与 G1 update_steering 一致）
THROTTLE_DEAD_ZONE = 0.1

# 车头朝向符号：本机型车头 = 手臂工作面一侧 = 世界 -y（NOSE_Y_SIGN=-1）。
# 世界系速度 = v·R_z(yaw)·(0, NOSE_Y_SIGN, 0)。
# 若实测前进方向仍相反，把 NOSE_Y_SIGN 改成 +1 即可整体掉头。
NOSE_Y_SIGN = -1.0


def build_robot_names(namespace: str) -> dict:
    """生成双臂、夹爪、site、基座的全部逻辑名称（不含 agent 前缀）。"""
    p = namespace
    return {
        "left_joints": [f"{p}left_elfin_joint{i}" for i in range(1, 7)],
        "right_joints": [f"{p}right_elfin_joint{i}" for i in range(1, 7)],
        "left_positions": [f"{p}drive_left_elfin_joint{i}" for i in range(1, 7)],
        "right_positions": [f"{p}drive_right_elfin_joint{i}" for i in range(1, 7)],
        "left_site": f"{p}left_eef_site",
        "right_site": f"{p}right_eef_site",
        # M20 单电机双指：每爪 1 个 velocity 电机（驱动 jaw_a）；
        # jaw_b 由 XML equality 机械耦合跟随，无需执行器。
        "left_gripper": [f"{p}drive_left_m20_motor"],
        "right_gripper": [f"{p}drive_right_m20_motor"],
        # 主动 jaw 关节（TriggerController 读行程 + 每步读 qpos 做速度闭环）
        "left_gripper_joints": [f"{p}left_m20_jaw_a_slide"],
        "right_gripper_joints": [f"{p}right_m20_jaw_a_slide"],
        # 双臂都挂在 XB_Link 下；XB_Link 随底盘一起运动，作为 IK 的基座
        # 参考体系（B 系），使手柄增量与世界位姿/底盘航向解耦。
        "base_body": f"{p}XB_Link",
        # 底盘浮动关节/车体（底盘控制器读取航向用）
        "base_yaw_joint": f"{p}base_yaw",
        "chassis_body": f"{p}mobile_base_visual",
        # 肩部 body（elfin 第一关节所在 link）：手柄目标"肩可达球"限幅中心。
        # 预备姿势接近全伸展（肩距 1.088m），目标出可达域时 DLS 会卡死乱扭。
        "left_shoulder": f"{p}left_elfin_link1",
        "right_shoulder": f"{p}right_elfin_link1",
        # 底盘 3 个平面速度执行器：[世界 vx, 世界 vy, 车体偏航角速度]
        # 原模型轮子是底座 mesh 的固定造型、外观不变，整车随浮动底座平移/转向。
        "base_drives": [
            f"{p}drive_base_x",
            f"{p}drive_base_y",
            f"{p}drive_base_yaw",
        ],
    }


def obs_callback(_env):
    # Gymnasium 不允许空的 Dict 观测空间，纯遥操给一个占位观测即可。
    return {"teleop": np.zeros(1, dtype=np.float32)}


# ---------------------------- 控制器 ----------------------------


def read_current_joint_values(env, logical_names):
    """读取 OrcaStudio/MuJoCo 中每个关节当前的标量 qpos。"""
    resolved = [env.joint(name) for name in logical_names]
    qmap = env.query_joint_qpos(resolved)
    values = []
    for name in resolved:
        arr = np.asarray(qmap[name], dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise RuntimeError(f"关节 {name} 的 qpos 应为标量，实际为 {arr}")
        values.append(float(arr[0]))
    return values


def verify_gripper_motors(env, names):
    """预检：确认 OrcaStudio 加载的是新版模型（M20 单电机 velocity 执行器）。

    若还在跑旧模型（4 个 drive_*_m20_jaw_a/b position 执行器，没有
    drive_*_m20_motor），M20TriggerController 会在 actuator_name2id 处
    KeyError；这里提前拦截并给出重新导入的处理指引。
    """
    missing = []
    for key in ("left_gripper", "right_gripper"):
        for logical in names[key]:
            try:
                env.model.actuator_name2id(env.actuator(logical))
            except KeyError:
                missing.append(logical)
    if missing:
        raise RuntimeError(
            f"模型中缺少夹爪电机执行器: {missing}。OrcaStudio 当前加载的还是"
            "旧模型（旧版是 drive_*_m20_jaw_a/b position 执行器）。请在 "
            "OrcaStudio 删除旧 agent 后重新导入 XML 原文件 "
            "dual_arm_mujoco_mobile_m20_pico_sites.xml（新版含 "
            "drive_left/right_m20_motor 电机与 equality 双指耦合）并重启仿真。"
        )


def make_arm_config(joint_names, position_names, ee_site, current_qpos):
    """position 执行器：初始关节值与初始 ctrl 均取当前仿真状态。"""
    if len(current_qpos) != 6:
        raise RuntimeError(f"E05 应有 6 个关节值，实际得到 {len(current_qpos)} 个")
    return {
        "joint_names": list(joint_names),
        "neutral_joint_values": list(current_qpos),
        "positions_names": list(position_names),
        "positions_init_ctrl": list(current_qpos),
        "positions_ranges": [list(r) for r in ARM_RANGES],
        "ee_site_name": ee_site,
    }


def boost_ik_tracking(arm, alpha=0.5, joint_delta=0.18, damping_lambda=0.03):
    """
    提高 custom IK 的跟随速度并在近奇异点平滑关节运动。

    - alpha=0.5 / max_joint_delta=0.18：默认 alpha=0.2 每步只追 20% 误差、
      delta 上限 0.1，遥操作时末端滞后；预备姿势改为肘弯曲(远离全伸展)后，
      正常 0.1m 手柄位移只需 ~0.35rad 关节增量，把 delta 上限放到 0.18 才不会
      把正常移动也限掉（旧值 0.12 在新姿势下会轻微限制横向跟随）。
    - damping_lambda（DLS 阻尼项 λ）：近全伸展等奇异构型用大阻尼防关节甩动，
      但阻尼过大会把笛卡尔移动"软化"（关节转、夹爪不走）。预备姿势已远离
      奇异（雅可比最小奇异值 0.28），取 0.03：实测径向/向上跟随率 99%、
      横向 84%，同时保留一定抗噪；IK 内部另有自适应奇异阻尼兜底。
    用 setter/实例属性调整，不改 site-packages。
    """
    ctrl = getattr(arm, "controller", None)
    if ctrl is None:
        return
    if hasattr(ctrl, "set_alpha"):
        ctrl.set_alpha(float(alpha))
    if hasattr(ctrl, "max_joint_delta"):
        ctrl.max_joint_delta = float(joint_delta)
    if hasattr(ctrl, "set_lambda"):
        ctrl.set_lambda(float(damping_lambda))


class HandEEBinding:
    """
    手柄位姿 -> 末端世界系目标的绝对绑定（VR 主从遥操作的标准做法）。

    绑定在重绑瞬间（按住 Grip / episode 重置后第一帧）捕获两组对应关系：
      1. "手柄当前位姿 <-> 末端当前世界位姿"
      2. 操作者的水平朝向（前 f / 左 l / 上 z），供映射 M = R_C·A·R_Hᵀ 使用。
    之后每帧把手柄在追踪空间中的位移分解到校准朝向，再映射到机器人
    当前车体系（随底盘 yaw 实时旋转）：

      d = p_now - p_ref
      df = d·f（前/后）, dl = d·l（左/右）, du = d·z（上/下）
      d_world = df·f_c - dl·l_c + du·z    （侧向恒翻转，见下）
      goal_pos = P_ee + scale · d_world
      goal_rot = (M_rot Δ M_rotᵀ) ⊗ Q_ee，Δ = q_now ⊗ q_ref⁻¹，
                 M_rot = R_C·R_Hᵀ（正旋转，无镜像）

    v6（针对"右手往外动、机械臂往内"）：位置侧向恒翻转，删除 face/same
    模式与 auto 站位判定。依据（FK 实测本模型）：右臂(right_elfin)末端在
    世界 +x = +l_c 一侧、左臂在 -l_c 一侧（车头 f_c = 世界 -y）。
    操作者右手"往外"在解剖学上恒为 -l 方向（dl<0）：
      - 旧 same 模式(A=I)：dl -> dl·l_c = -|dl|·l_c = 横穿中线往内 <- 症状
      - v6 侧向翻转     ：dl -> -dl·l_c = +|dl|·l_c = 右臂外侧   <- 恒成立
    即"手往外->臂往外"与站位（车头前/车尾后/侧面）无关；前伸 df->f_c
    沿车头、上抬 du->z 与 v5 完全相同，无回归。旋转用正映射 M_rot：
    腕转在机器人自身车体系里同向复现（站车尾时与 v5 完全一致，1:1 跟手），
    不引入镜像反号。

    保留 v5 的两处关键改进：
      1. 双手连线校准：l = (左手位置-右手位置) 的水平投影 = 操作者左向，
         f = l×z = 操作者前向。与手柄握姿/按 Grip 瞬间手腕角度无关。
         双手数据不可用或两腕水平间距 < 0.10m 时回退手柄姿态法并告警。
      2. 双重目标限幅：goal 先夹进重绑锚点 _ee_p 周围 goal_radius 球
         （默认 0.25m），再夹进肩部 body 当前位置周围 reach_limit 球
         （默认 1.09m < 全展 ~1.17m）。目标出可达域时 DLS 卡死/乱扭，
         限幅后手柄拉满时末端贴住虚拟球面，方向始终与手柄一致。

    其中 f_c/l_c 为机器人当前车头前向/左向（NOSE_Y_SIGN 决定）。
    L 手柄->左臂、R 手柄->右臂的配对不变。
    - scale：手柄位移->末端位移比例，臂展不够时调小（如 0.5）。
    - enabled=False（离合松开）期间不更新目标，避免锁定时目标漂移。
    """

    def __init__(self, env, arm, device, chassis_body, shoulder_body=None,
                 scale=1.0, goal_radius=0.25, reach_limit=1.09, tag="",
                 pos_deadzone=0.005, pos_smooth=0.5, rot_deadzone=0.02,
                 side_flip=True):
        self.env = env
        self.arm = arm
        self.device = device              # PicoJoystickDevice：读双手位置
        self.chassis_body = chassis_body
        self.shoulder_body = shoulder_body
        self.scale = float(scale)
        self.goal_radius = float(goal_radius)
        self.reach_limit = float(reach_limit) if reach_limit else None
        self.tag = str(tag)
        # side_flip：位置映射的侧向是否翻转。
        #   True（旧配对 L->left_elfin(-l_c 侧)）：dl -> -dl·l_c，"手往外->臂往外"；
        #   False（交换配对 L->right_elfin(+l_c 侧)）：dl -> +dl·l_c，同样
        #   "手往外->臂往外"（left_elfin/right_elfin 命名与物理安装侧相反，
        #   交换配对后翻转符号随之换号），且位置映射退化为正映射 M=R_C·R_Hᵀ。
        # 抗抖动：手柄静止时毫米级追踪噪声会被 IK（尤其近奇异点）放大成
        # 关节抖动。pos_deadzone 为手柄位移死区(米)；pos_smooth 为末端目标
        # 一阶低通系数(1.0=不滤波)；rot_deadzone 为旋转死区(弧度)。
        self.side_flip = bool(side_flip)
        self.pos_deadzone = float(pos_deadzone)
        self.pos_smooth = float(pos_smooth)
        self.rot_deadzone = float(rot_deadzone)
        self._goal_pos_filt = None        # 上一帧滤波后末端目标（世界系）
        self.enabled = True   # clutch 松开时置 False，暂停跟随
        self.bound = False
        self._p_ref = None
        self._q_ref = None    # wxyz
        self._ee_p = None     # 末端基准位置（世界系）
        self._ee_q = None     # 末端基准姿态（世界系, wxyz）
        self._f = None        # 校准时操作者水平前向（追踪空间系）
        self._l = None        # 校准时操作者水平左向（追踪空间系）

    # ---------- 校准 ----------

    def _get_hand_positions(self):
        """读 Pico 双手当前位置（已转到 MuJoCo 系，与 on_transform 同系）。

        get_transform_list() 内部加锁、拷贝后返回，且事件回调在锁外分发
        （update() 先取数据再调用回调），回调内再调用是线程安全的。
        不可用/异常时返回 None，走手柄姿态回退路径。
        """
        try:
            transforms = self.device.pico_joystick.get_transform_list()
        except Exception:
            return None
        if not transforms or len(transforms) < 2:
            return None
        try:
            p_left = np.asarray(transforms[0][0], dtype=np.float64).reshape(3)
            p_right = np.asarray(transforms[1][0], dtype=np.float64).reshape(3)
        except (IndexError, TypeError, ValueError):
            return None
        return p_left, p_right

    def _chassis_forward(self):
        """机器人当前车头水平前向（世界系，随底盘 yaw 旋转）。"""
        _, _, chassis_xquat = self.env.get_body_xpos_xmat_xquat([self.chassis_body])
        yaw_r = R.from_quat(np.asarray(chassis_xquat, dtype=np.float64)[[1, 2, 3, 0]])
        f_c = yaw_r.apply([0.0, NOSE_Y_SIGN, 0.0])
        f_c[2] = 0.0
        return f_c / np.linalg.norm(f_c)

    def _orientation_from_hand_quat(self, q_ref):
        """回退校准（v4 逻辑）：从手柄姿态提取操作者水平前向。"""
        qh = R.from_quat(np.asarray(q_ref)[[1, 2, 3, 0]])
        f = qh.apply([1.0, 0.0, 0.0])
        f[2] = 0.0
        if np.linalg.norm(f) < 1e-3:
            # 手柄恰好竖直：改用手柄局部 y（左向）反推前向 f = l×z。
            l0 = qh.apply([0.0, 1.0, 0.0])
            l0[2] = 0.0
            f = np.cross(l0 / max(np.linalg.norm(l0), 1e-9), [0.0, 0.0, 1.0])
        return f / np.linalg.norm(f)

    def _rebind(self, p_now, q_now):
        self._p_ref = np.asarray(p_now, dtype=np.float64).copy()
        self._q_ref = np.asarray(q_now, dtype=np.float64).copy()
        self._ee_p = np.asarray(self.arm.initial_ee_pos, dtype=np.float64).copy()
        self._ee_q = np.asarray(self.arm.initial_ee_quat, dtype=np.float64).copy()

        # 1) 校准朝向：优先双手连线（左手在操作者左侧 => pL-pR 指向左）。
        f = None
        source = "手柄姿态"
        hands = self._get_hand_positions()
        if hands is not None:
            l_vec = hands[0] - hands[1]
            l_vec[2] = 0.0                        # 水平投影
            norm_l = np.linalg.norm(l_vec)
            if norm_l >= 0.10:                     # 两腕至少分开 10cm 才可信
                f = np.cross(l_vec / norm_l, [0.0, 0.0, 1.0])  # f = l×z
                source = "双手连线"
            else:
                LOGGER.warning(
                    f"{self.tag} 重绑：双手水平间距 {norm_l:.3f}m < 0.10m，"
                    "连线校准不可用，回退手柄姿态校准"
                )
        if f is None:
            f = self._orientation_from_hand_quat(self._q_ref)
        self._f = f / np.linalg.norm(f)
        self._l = np.cross([0.0, 0.0, 1.0], self._f)  # z×f = 左

        # 2) 站位诊断（仅日志）：v6 映射对任意站位成立（侧向恒翻转，
        #    站位差异由校准 R_H 自动吸收），不再需要 face/same 模式判定。
        self.bound = True
        f_c = self._chassis_forward()
        dot_f = float(np.dot(self._f, f_c))
        if dot_f > 0.5:
            stance = "车尾侧(同向操作)"
        elif dot_f < -0.5:
            stance = "车头侧(面对机器人)"
        else:
            stance = "车身侧面"
        reach_txt = f"{self.reach_limit:.2f}m" if self.reach_limit else "关"
        lat_txt = "侧向翻转" if self.side_flip else "侧向正映射(配对交换)"
        LOGGER.info(
            f"{self.tag} 重绑[{source}]：手柄↔末端夹爪配对，夹爪锚点(世界系)="
            f"({self._ee_p[0]:+.3f},{self._ee_p[1]:+.3f},{self._ee_p[2]:+.3f})，"
            f"前向≈({self._f[0]:+.2f},{self._f[1]:+.2f})，"
            f"站位={stance}，{lat_txt}(手往外->臂往外)，"
            f"目标限幅：锚点球{self.goal_radius:.2f}m+肩球{reach_txt}"
        )
        # 重绑后目标基准改变，清空低通滤波器，避免把旧目标带入新映射。
        self._goal_pos_filt = None

    def _anchor_stale(self):
        """末端基准被 reset() 刷新（按下 Grip / episode 重置）则需重绑。"""
        if not self.bound:
            return True
        if not np.allclose(self._ee_p, np.asarray(self.arm.initial_ee_pos), atol=1e-9):
            return True
        q_cur = R.from_quat(np.asarray(self.arm.initial_ee_quat)[[1, 2, 3, 0]])
        q_old = R.from_quat(self._ee_q[[1, 2, 3, 0]])
        return (q_cur * q_old.inv()).magnitude() > 1e-6

    # ---------- 每帧映射 ----------

    @staticmethod
    def _clamp_to_sphere(p, center, radius):
        """把 p 投影到以 center 为心、radius 为半径的球内（在内则原样返回）。"""
        d = p - center
        dist = np.linalg.norm(d)
        if dist <= radius or dist < 1e-9:
            return p
        return center + d * (radius / dist)

    def on_transform(self, relative_position, relative_quat):
        if not self.enabled:
            return
        p_now = np.asarray(relative_position, dtype=np.float64)
        q_now = np.asarray(relative_quat, dtype=np.float64)  # wxyz
        if self._anchor_stale():
            self._rebind(p_now, q_now)

        # 机器人当前车体系水平轴（随底盘 yaw）。
        f_c = self._chassis_forward()
        l_c = np.cross([0.0, 0.0, 1.0], f_c)  # 车体左向

        # 校准坐标系 H（操作者：前 f / 左 l / 上 z，Pico 追踪世界系）
        # 与机器人车体系 C（f_c / l_c / z，机器人世界系）。
        R_H = np.column_stack([self._f, self._l, [0.0, 0.0, 1.0]])
        R_C = np.column_stack([f_c, l_c, [0.0, 0.0, 1.0]])
        # v6 位置映射：侧向符号由配对决定（side_flip），两种配对都满足
        # "手往外->臂往外"（与站位无关）：
        #   side_flip=True （L->left_elfin(-l_c 侧)）：dl -> -dl·l_c
        #   side_flip=False（L->right_elfin(+l_c 侧)，交换配对）：dl -> +dl·l_c
        # 前伸 df->f_c 沿车头、上抬 du->z 不变。
        lat = -1.0 if self.side_flip else 1.0
        M_pos = R_C @ np.diag([1.0, lat, 1.0]) @ R_H.T

        disp = self.scale * (M_pos @ (p_now - self._p_ref))
        # 位置死区：手柄位移（映射到机器人空间后）小于阈值视为静止，
        # 防止追踪噪声经 IK 放大成关节抖动。
        if np.linalg.norm(disp) < self.pos_deadzone:
            disp = np.zeros(3)
        goal_pos = self._ee_p + disp

        # 双重限幅（v5）：目标先夹进重绑锚点球，再夹进肩可达球。
        # 出可达域的目标会让 DLS 卡死/乱扭——"末端不跟手柄"的主要根因；
        # 限幅后手柄拉满时末端贴住虚拟球面，方向始终与手柄一致。
        goal_pos = self._clamp_to_sphere(goal_pos, self._ee_p, self.goal_radius)
        if self.shoulder_body is not None and self.reach_limit:
            xpos, _, _ = self.env.get_body_xpos_xmat_xquat([self.shoulder_body])
            shoulder_pos = np.asarray(xpos, dtype=np.float64).reshape(3)
            goal_pos = self._clamp_to_sphere(goal_pos, shoulder_pos, self.reach_limit)

        # 末端目标一阶低通：进一步平滑手柄抖动；首帧/重绑后直接对齐无延迟。
        if self._goal_pos_filt is None:
            self._goal_pos_filt = goal_pos.copy()
        else:
            d = goal_pos - self._goal_pos_filt
            goal_pos = self._goal_pos_filt + self.pos_smooth * d
        self._goal_pos_filt = goal_pos.copy()

        # 旋转：v6 用正映射 M_rot=R_C·R_Hᵀ（det=+1，无镜像）共轭，腕转在
        # 机器人自身车体系里同向复现（站车尾时与 v5 完全一致、1:1 跟手），
        # 不引入镜像反号；再左乘末端基准姿态。
        dq = (
            R.from_quat(q_now[[1, 2, 3, 0]])
            * R.from_quat(self._q_ref[[1, 2, 3, 0]]).inv()
        )
        # 旋转死区：手部姿态角抖动小于阈值视为无旋转。
        if dq.magnitude() < self.rot_deadzone:
            dq = R.identity()
        M_rot = R_C @ R_H.T
        dq_world = R.from_matrix(M_rot @ dq.as_matrix() @ M_rot.T)
        goal_rot = dq_world * R.from_quat(self._ee_q[[1, 2, 3, 0]])

        self.arm.action = np.concatenate([goal_pos, goal_rot.as_rotvec()])


def add_arm_ik_pico_controller(manager, env, arm_config, base_body, device, key,
                                chassis_body=None, arm_scale=1.0,
                                shoulder_body=None, goal_radius=0.25,
                                reach_limit=1.09, tag="", side_flip=True):
    """
    与 OrcaManipulation 官方 add_arm_osc_pico_controller 注册方式一致，
    区别是该 XML 使用 position 执行器，因此走 create_arm_ik_controller
    （custom_ik_pose，DLS 逆解，输出关节绝对位置目标）。

    chassis_body 非 None 时启用手柄-末端绝对绑定（见 HandEEBinding：
    侧向恒翻转"手往外->臂往外"，任意站位成立；双手连线校准 +
    锚点球/肩球双重目标限幅），否则退回 G1 原生 B 系直加。
    """
    ctrl_names = [env.actuator(name) for name in arm_config["positions_names"]]
    init_ctrl = {
        name: value
        for name, value in zip(ctrl_names, arm_config["positions_init_ctrl"])
    }
    arm_controller = create_arm_ik_controller(
        env=env,
        arm_config=arm_config,
        base_body=base_body,
        ctrl_name=ctrl_names,
        init_ctrl=init_ctrl,
    )
    # DLS 默认参数偏保守（每步只追 20% 误差），提高跟随速度。
    boost_ik_tracking(arm_controller)
    if chassis_body is not None:
        binding = HandEEBinding(env, arm_controller, device, chassis_body,
                                shoulder_body=shoulder_body,
                                scale=arm_scale,
                                goal_radius=goal_radius,
                                reach_limit=reach_limit, tag=tag,
                                side_flip=side_flip)
        device.bind_transform_event(key, binding.on_transform)
    else:
        device.bind_transform_event(key, arm_controller.update_goal)
    manager.add_controller(arm_controller)
    return arm_controller


class M20TriggerController(AbstractController):
    """一个 Pico 扳机连续映射到同侧 M20 夹爪的单个 velocity 电机。

    机构（单电机双指）：XML 里每个夹爪只有 1 个电机（velocity 执行器，
    驱动 jaw_a），jaw_b 由 equality 机械耦合跟随、相向对称闭合；XML
    不做 position 位置伺服，位置闭环在控制器里完成：
      "控制器决定夹爪位置"：扳机 t∈[0,1] 为开度位置目标，
        target = 全开 + t*(全闭-全开)（0=全开，1=全闭）；
      "关节怎么动看具体机器人"：全开/全闭从模型读 jaw_a 关节的实际
        qpos 行程（range 上限=全开、下限=全闭，本模型 [-0.0135, 0]），
        每步读当前 qpos，下发电机速度指令 v = kp*(target-qpos)，限幅
        ±GRIPPER_V_MAX；读取失败回退 XML 常量 M20_OPEN/M20_CLOSED。
    """

    def __init__(self, env, actuator_names, joint_names, base_body, device,
                 trigger_key, tag=""):
        self.tag = str(tag)
        ctrl_names = [env.actuator(name) for name in actuator_names]
        self.open_of, self.closed_of = self._read_strokes(env, joint_names, ctrl_names)
        self._qpos_joint = env.joint(joint_names[0])   # 主动 jaw（电机驱动）
        self.opening = 0.0                    # 扳机量：0=全开 1=全闭
        init_ctrl = {name: 0.0 for name in ctrl_names}   # 电机初始速度 0
        super().__init__(env, ctrl_names, init_ctrl, base_body)

        device.bind_trigger_event(trigger_key, self._on_trigger)

    def _read_strokes(self, env, joint_names, ctrl_names):
        """从模型读主动 jaw 关节的实际行程 -> ({执行器名: 全开}, {执行器名: 全闭})。"""
        open_of, closed_of = {}, {}
        try:
            resolved = [env.joint(name) for name in joint_names]
            ranges = np.asarray(
                env.gym.model.get_joint_qposrange(resolved), dtype=np.float64
            ).reshape(-1, 2)
            if ranges.shape[0] != len(ctrl_names):
                raise RuntimeError(
                    f"夹爪关节数 {ranges.shape[0]} != 执行器数 {len(ctrl_names)}"
                )
            for name, (lo, hi) in zip(ctrl_names, ranges):
                open_of[name] = float(hi)     # range 上限 = 全开
                closed_of[name] = float(lo)   # range 下限 = 全闭
            if self.tag:
                op0 = next(iter(open_of.values()))
                cl0 = next(iter(closed_of.values()))
                LOGGER.info(
                    f"{self.tag} M20 夹爪行程(来自模型): 全开 {op0:+.4f}m ~ "
                    f"全闭 {cl0:+.4f}m（单爪 {abs(cl0 - op0) * 1000:.1f}mm，"
                    f"单电机双指）"
                )
        except Exception as exc:
            LOGGER.warning(f"{self.tag} 读夹爪关节行程失败，回退 XML 常量: {exc}")
        return open_of, closed_of

    def _stroke(self, ctrl_name):
        if ctrl_name in self.open_of:
            return self.open_of[ctrl_name], self.closed_of[ctrl_name]
        return M20_OPEN, M20_CLOSED   # 回退常量（与 XML range 一致）

    def _on_trigger(self, value: float):
        self.opening = float(np.clip(value, 0.0, 1.0))

    def reset(self):
        self.opening = 0.0

    def run_controller(self):
        op, cl = self._stroke(self.ctrl_name[0])
        target = op + self.opening * (cl - op)          # 开度位置目标
        qmap = self.env.query_joint_qpos([self._qpos_joint])
        qpos = float(
            np.asarray(qmap[self._qpos_joint], dtype=np.float64).reshape(-1)[0]
        )
        # 位置误差 -> 电机速度指令（速度内环由 XML velocity 执行器完成）
        v = float(
            np.clip(GRIPPER_KP_POS * (target - qpos),
                    -GRIPPER_V_MAX, GRIPPER_V_MAX)
        )
        return {self.ctrl_index[0]: v}


class ClutchArmController(AbstractController):
    """
    带离合器的手臂控制器（toggle 持续跟随版）：

      启动 / 复位后  -> 6 个关节锁定在预备姿势（ready_q），尚未进入跟随；
      点一下同侧 Grip -> 该臂进入持续跟随：末端一直跟随 Pico 手柄位姿，
                        之后无需按住、松开也不会锁定/冻结（"开始遥操作后不停"）；
      跟随时再点 Grip -> 就地以"当前手柄位姿<->当前末端位姿"重新对中（消除
                        累积漂移），仍保持跟随，不会停下。
      右手 A 键（全局复位）-> request_reset()：双臂回预备姿势并锁定，夹爪全开、
                        手柄重新对中，底盘不动。

    按下 Grip 进入跟随的瞬间会以"当前末端位姿<->当前手柄位姿"重建绝对绑定
    （见 HandEEBinding），因此解锁瞬间不跳变。
    """

    def __init__(self, env, arm_config, base_body, device, transform_key, grip_key,
                 tag="", allow_control=True, chassis_body=None, arm_scale=1.0,
                 shoulder_body=None, goal_radius=0.25, reach_limit=1.09,
                 ready_q=None, side_flip=True):
        ctrl_names = [env.actuator(name) for name in arm_config["positions_names"]]
        init_q = list(arm_config["positions_init_ctrl"])
        init_ctrl = {name: float(value) for name, value in zip(ctrl_names, init_q)}

        # 内部真正做 DLS 逆解的控制器；锁定/复位时不调用它。
        self.arm = create_arm_ik_controller(
            env=env,
            arm_config=arm_config,
            base_body=base_body,
            ctrl_name=ctrl_names,
            init_ctrl=init_ctrl,
        )
        # DLS 默认参数偏保守（每步只追 20% 误差），提高跟随速度并加大阻尼。
        boost_ik_tracking(self.arm)
        # 手柄-末端绝对绑定：进入跟随时重绑，跟随期间持续更新。
        self.binding = None
        if allow_control:
            if chassis_body is not None:
                self.binding = HandEEBinding(env, self.arm, device, chassis_body,
                                             shoulder_body=shoulder_body,
                                             scale=arm_scale,
                                             goal_radius=goal_radius,
                                             reach_limit=reach_limit, tag=tag,
                                             side_flip=side_flip)
                self.binding.enabled = False   # 起始锁定，点一下 Grip 才持续跟随
                device.bind_transform_event(transform_key, self.binding.on_transform)
            else:
                device.bind_transform_event(transform_key, self.arm.update_goal)

        super().__init__(env, ctrl_names, init_ctrl, base_body)

        self.joint_resolved = [env.joint(name) for name in arm_config["joint_names"]]
        self.tag = tag
        # 预备姿势（A 键复位目标）：优先用传入的 ready_q，否则用当前姿势。
        if ready_q is not None:
            self.ready_qpos = np.asarray(ready_q, dtype=np.float64)
        else:
            self.ready_qpos = np.asarray(init_q, dtype=np.float64)
        self.lock_qpos = self.ready_qpos.copy()
        # allow_control=False：永久锁定（lock 模式），忽略 Grip。
        self.allow_control = allow_control
        self.following = False       # 是否处于持续跟随状态（toggle）
        self.grip_pressed = False    # 按键原始电平（用于检测按下沿）
        self._reset_pending = False  # 全局复位请求：复位运动进行中
        self._reset_frames = 0       # 复位已持续的帧数（超时兜底用）

        if allow_control:
            device.bind_grip_button_event(grip_key, self._on_grip)

    def _read_qpos(self):
        qmap = self.env.query_joint_qpos(self.joint_resolved)
        return np.array(
            [float(np.asarray(qmap[name]).reshape(-1)[0]) for name in self.joint_resolved],
            dtype=np.float64,
        )

    def _lock_angles(self):
        return {
            idx: float(self.lock_qpos[i])
            for i, idx in enumerate(self.ctrl_index)
        }

    def _start_following(self, reason: str):
        """进入持续跟随：以"当前手柄位姿 <-> 当前末端夹爪位置"重新配对。

        arm.reset() 刷新末端基准 => HandEEBinding 检测到锚点失效，下一个
        手柄回调自动以"当前手柄位姿 <-> 当前夹爪位姿"重绑，绑定的是
        末端夹爪(eef site)的世界位姿，不是关节角。
        """
        self._reset_pending = False
        self.following = True
        self.arm.reset()
        if self.binding is not None:
            self.binding.enabled = True
        if self.tag:
            LOGGER.info(
                f"{self.tag}：{reason}，开始持续跟随"
                "（手柄已与末端夹爪当前位置配对，无需按住）"
            )

    def _on_grip(self, pressed: bool):
        if not self.allow_control:
            return
        pressed = bool(pressed)
        # 只在"按下沿"动作（松开沿忽略）：toggle 持续跟随，无需一直按住。
        if pressed and not self.grip_pressed:
            if self._reset_pending:
                # 复位运动中点 Grip：立即中断复位、就地开始跟随。
                # 保证 Grip 在任何时刻按下都能进入遥操作（修复"复位后
                # 无法重新遥操"：位置伺服无重力补偿，关节可能永远到不了
                # 严苛的到位阈值，不能让复位状态吞掉 Grip）。
                self._start_following("复位中点按 Grip，就地开始跟随")
            elif not self.following:
                self._start_following("点按 Grip")
            else:
                # 跟随中再点：就地重新对中（手柄↔当前夹爪位置重新配对），
                # 消除累积漂移，仍保持跟随。
                self.arm.reset()
                if self.tag:
                    LOGGER.info(f"{self.tag}：跟随中重新对中（手柄↔末端夹爪重新配对）")
        self.grip_pressed = pressed

    def request_reset(self):
        """右手 A 键全局复位：本帧起回预备姿势并锁定（夹爪/对中由协调者处理）。"""
        if not self.allow_control:
            return
        self._reset_pending = True
        self._reset_frames = 0
        self.following = False
        if self.binding is not None:
            self.binding.enabled = False
        if self.tag:
            LOGGER.info(f"{self.tag}：收到复位指令，回预备姿势")

    def reset(self):
        # episode 重置：脱开跟随，手臂锁定到预备姿势。
        self.arm.reset()
        self.following = False
        self.grip_pressed = False
        self._reset_pending = False
        self._reset_frames = 0
        self.lock_qpos = self.ready_qpos.copy()
        if self.binding is not None:
            self.binding.enabled = False

    def run_controller(self):
        # 复位：持续下发预备关节角。到位判定绝不能依赖严苛阈值——
        # 本模型 E05 侧装，joint1 世界轴水平，预备姿势下重力矩 ~64Nm，
        # 位置伺服 kp=110 无重力补偿，稳态下沉 ~0.5 rad（实测），严阈值
        # 会永远判不到位，把 Grip 一直吞掉。因此"到位(0.05) 或超时(3s)"
        # 任一满足即结束复位；复位中点 Grip 也可立即就地跟随（_on_grip）。
        if self._reset_pending:
            self._reset_frames += 1
            self.lock_qpos = self.ready_qpos.copy()
            err = np.linalg.norm(self._read_qpos() - self.ready_qpos)
            if err < 0.05 or self._reset_frames > 300:   # 100Hz 控制 -> 3s
                self._reset_pending = False
                self.arm.reset()   # 刷新末端基准，下次 Grip 干净重绑
                if self.tag:
                    LOGGER.info(
                        f"{self.tag}：复位完成，锁定预备姿势（残差 {err:.3f} rad）"
                    )
            return self._lock_angles()

        if self.following:
            return self.arm.run_controller()
        # 未进入跟随：锁定（启动/复位后为预备姿势）。
        return self._lock_angles()


class FloatingAckermannBaseController(AbstractController):
    """
    浮动平面底盘控制器（按键与智元 G1 完全一致），且不改动原模型外观：

      左摇杆 x -> 虚拟前轮转角（只决定整车转向半径，模型轮子是固定造型不偏转）；
      右摇杆 y -> 前进/后退线速度。

    XML 中底盘为世界系 slide x / slide y + 车体 yaw 三个速度执行器（z 高度锁定）。
    本控制器每步按自行车模型把摇杆量解算为：
      v = throttle_y * MAX_SPEED
      δ = steering_x * MAX_STEER（指数手感，同 G1）
      yaw_rate = v·tan(δ)/WHEELBASE
      世界系速度 = v·R_z(yaw)·(0, NOSE_Y_SIGN, 0)
      （车头 = R_z(yaw)·(0, NOSE_Y_SIGN, 0) = 手臂工作面一侧，本机型朝世界 -y；
        推前进摇杆 => 沿车头方向行驶，车头转向与 G1 手感一致）

    执行器顺序（ctrl_index）：[base_x, base_y, base_yaw]
    """

    def __init__(self, env, actuator_names, base_body, yaw_joint, device,
                 max_speed=MAX_SPEED, max_steer=MAX_STEER, wheelbase=WHEELBASE):
        ctrl_names = [env.actuator(name) for name in actuator_names]
        init_ctrl = {name: 0.0 for name in ctrl_names}
        super().__init__(env, ctrl_names, init_ctrl, base_body)

        self.yaw_joint_resolved = env.joint(yaw_joint)
        self.max_speed = max_speed
        self.max_steer = max_steer
        self.wheelbase = wheelbase

        self.steering = 0.0   # 左摇杆 x，[-1,1]
        self.throttle = 0.0   # 右摇杆 y，[-1,1]

        device.bind_joystick_position_event(
            PicoJoystickKey.L_JOYSTICK_POSITION, self.update_steering
        )
        device.bind_joystick_position_event(
            PicoJoystickKey.R_JOYSTICK_POSITION, self.update_throttle
        )

    # ---- 摇杆回调（与 G1 ControllerSteeringDrive 相同的死区） ----
    def update_steering(self, x: float, y: float):
        self.steering = float(x) if abs(x) > STEER_DEAD_ZONE else 0.0

    def update_throttle(self, x: float, y: float):
        self.throttle = float(y) if abs(y) > THROTTLE_DEAD_ZONE else 0.0

    def reset(self):
        self.steering = 0.0
        self.throttle = 0.0

    def _read_yaw(self) -> float:
        qmap = self.env.query_joint_qpos([self.yaw_joint_resolved])
        return float(np.asarray(qmap[self.yaw_joint_resolved]).reshape(-1)[0])

    def run_controller(self):
        v = self.throttle * self.max_speed
        # 转向手感与 G1 一致：指数映射，小幅推动精细、推满到最大转角。
        k = np.e
        mag = (np.exp(k * abs(self.steering)) - 1) / (np.exp(k) - 1)
        delta = -mag * np.sign(self.steering) * self.max_steer

        yaw = self._read_yaw()
        yaw_rate = v * np.tan(delta) / self.wheelbase if abs(delta) > 1e-6 else 0.0
        # 车头 = R_z(yaw)·(0, NOSE_Y_SIGN, 0)（本机型车头朝世界 -y），
        # 因此前进时速度沿车头方向投影到世界系：
        world_vx = -NOSE_Y_SIGN * v * np.sin(yaw)
        world_vy = NOSE_Y_SIGN * v * np.cos(yaw)

        return {
            self.ctrl_index[0]: float(world_vx),
            self.ctrl_index[1]: float(world_vy),
            self.ctrl_index[2]: float(yaw_rate),
        }


# ---------------------------- 主流程 ----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Pico 双手遥操作：双 E05 臂 + 双 M20 夹爪 + 四轮移动底盘"
    )
    parser.add_argument("--addr", default="localhost:50051", help="OrcaGym gRPC 地址")
    parser.add_argument(
        "--agent",
        default="dual_arm",
        help="OrcaStudio 中的 agent 名称（env 会自动为对象名加该前缀）",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_NAMESPACE,
        help="OrcaStudio 导入模型时的命名空间前缀；若导入名不同请用此参数覆盖",
    )
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--time-step", type=float, default=0.001)
    parser.add_argument(
        "--arm-mode",
        choices=["clutch", "follow", "lock"],
        default="clutch",
        help=(
            "手臂控制模式："
            "clutch=启动锁定在当前姿势，按住同侧 Grip 才跟随（默认）；"
            "follow=手柄位姿直接持续跟随；"
            "lock=手臂永久锁定，只控夹爪"
        ),
    )
    parser.add_argument(
        "--no-ready",
        action="store_true",
        help="不摆截图预备姿势，保持 OrcaStudio 当前手臂姿态",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="关闭手柄-末端绝对绑定，退回 G1 原生 B 系直加（调试用）",
    )
    parser.add_argument(
        "--no-swap-hands",
        action="store_true",
        help=(
            "不按物理侧交换左右配对，恢复按模型命名配对"
            "（左手柄->left_elfin）。默认交换：站车尾遥操时左手柄控制"
            "操作者视角左边的臂(right_elfin)。仅站车头前面对机器人操控时"
            "才需要加此参数"
        ),
    )
    parser.add_argument(
        "--arm-scale",
        type=float,
        default=1.0,
        help=(
            "手柄位移->末端位移的比例（默认 1.0）；"
            "E05 臂展不够、感觉手一动就到极限时调小，如 0.5。仅绑定模式生效"
        ),
    )
    parser.add_argument(
        "--goal-radius",
        type=float,
        default=0.25,
        help=(
            "目标点距重绑锚点的最大半径（米，默认 0.25）：手柄拉满时末端"
            "贴住这个虚拟球面，防止目标出可达域导致 IK 卡死乱扭"
        ),
    )
    parser.add_argument(
        "--reach-limit",
        type=float,
        default=1.09,
        help=(
            "目标点距肩部的最大半径（米，默认 1.09，全展约 1.17）："
            "防止目标贴死可达边界导致 DLS 奇异；0 表示关闭肩球限幅"
        ),
    )
    args = parser.parse_args()

    names = build_robot_names(args.prefix)

    LOGGER.info(f"连接 OrcaGym: {args.addr}")
    LOGGER.info(f"Agent: {args.agent}；模型命名空间前缀: {args.prefix!r}")
    LOGGER.info(f"手臂模式: {args.arm_mode}")

    device = PicoJoystickDevice(PicoJoystick())

    # 默认（所有模式）上电先把双臂摆到截图预备角度；--no-ready 可保持当前姿态。
    use_ready_pose = not args.no_ready

    default_joint_values = {}
    if use_ready_pose:
        for name, value in zip(names["left_joints"], READY_LEFT_Q):
            default_joint_values[name] = float(value)
        for name, value in zip(names["right_joints"], READY_RIGHT_Q):
            default_joint_values[name] = float(value)
        LOGGER.info("双臂上电摆至截图预备姿势（前伸），夹爪全开")
    else:
        LOGGER.info("保持/读取机器人当前姿态作为手臂起点，夹爪全开")
    # 夹爪无论何种模式都初始化为全开。
    default_joint_values[f"{args.prefix}left_m20_jaw_a_slide"] = M20_OPEN
    default_joint_values[f"{args.prefix}left_m20_jaw_b_slide"] = M20_OPEN
    default_joint_values[f"{args.prefix}right_m20_jaw_a_slide"] = M20_OPEN
    default_joint_values[f"{args.prefix}right_m20_jaw_b_slide"] = M20_OPEN

    manager = DataCollectionManager(
        agent_name=args.agent,
        env_name="PicoDualE05Sites",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=obs_callback,
        env_index=0,
        frame_skip=args.frame_skip,
        time_step=args.time_step,
        orcagym_addr=args.addr,
        device=device,
        scene_manager=None,
        data_storage=None,
    )
    env = manager.env
    manager.render_fps = 60

    # 预检：OrcaStudio 必须加载新版模型（M20 电机执行器），否则明确提示重新导入。
    verify_gripper_motors(env, names)

    # 连接建立后读取当前关节角作为 IK 初始姿态。
    left_q = read_current_joint_values(env, names["left_joints"])
    right_q = read_current_joint_values(env, names["right_joints"])
    LOGGER.info(f"左臂当前关节角: {np.round(left_q, 4).tolist()}")
    LOGGER.info(f"右臂当前关节角: {np.round(right_q, 4).tolist()}")

    # clutch / lock 模式：把“上电预备姿势”写回 env 的默认关节值，
    # 这样 manager 运行期间每次 env.reset()（回到初始帧）都会重新恢复，
    # 手臂稳定锁定在截图前伸姿态（被 Grip 解锁移动后，松开沿会就地重锁）。
    if args.arm_mode in ("clutch", "lock"):
        hold_defaults = {}
        for name, value in zip(names["left_joints"], left_q):
            hold_defaults[name] = float(value)
        for name, value in zip(names["right_joints"], right_q):
            hold_defaults[name] = float(value)
        hold_defaults[f"{args.prefix}left_m20_jaw_a_slide"] = M20_OPEN
        hold_defaults[f"{args.prefix}left_m20_jaw_b_slide"] = M20_OPEN
        hold_defaults[f"{args.prefix}right_m20_jaw_a_slide"] = M20_OPEN
        hold_defaults[f"{args.prefix}right_m20_jaw_b_slide"] = M20_OPEN
        env.set_default_joint_values(hold_defaults)
        LOGGER.info("已把当前手臂姿势写为锁定/重置基准")

    left_arm = make_arm_config(
        names["left_joints"], names["left_positions"], names["left_site"], left_q
    )
    right_arm = make_arm_config(
        names["right_joints"], names["right_positions"], names["right_site"], right_q
    )

    base_body = names["base_body"]

    # 手柄-末端绝对绑定（默认开启）：按住 Grip 瞬间捕获"手柄位姿<->末端位姿"，
    # 之后手柄走到哪末端跟到哪；--no-mirror 时退回 G1 原生 B 系直加。
    chassis_body = None if args.no_mirror else env.body(names["chassis_body"])
    left_shoulder_body = right_shoulder_body = None
    if chassis_body is not None:
        left_shoulder_body = env.body(names["left_shoulder"])
        right_shoulder_body = env.body(names["right_shoulder"])
        LOGGER.info(
            f"手臂映射: 手柄-末端绝对绑定（侧向符号随配对：手往外->臂往外，任意站位成立），"
            f"位移比例 {args.arm_scale}，"
            f"目标限幅: 锚点球 {args.goal_radius:.2f}m + 肩球 {args.reach_limit:.2f}m；"
            "按住 Grip 重绑，双手连线自动校准朝向（不用刻意摆手柄指向）"
        )
    else:
        LOGGER.info("手臂映射: G1 原生 B 系直加（--no-mirror）")

    # 左右手柄->臂的配对（hand_left_* = 左手柄控制的臂）。
    # 本模型 left_elfin/right_elfin 命名与物理安装侧相反（FK 实测 right_elfin
    # 在车体 +x=左侧、left_elfin 在 -x=右侧），站车尾遥操时操作者视角左边
    # 的臂是 right_elfin，故默认按物理侧交换："动哪只手、视角哪边的臂就动"。
    # --no-swap-hands 恢复按模型命名配对（站车头前面对机器人操控时用）。
    # 交换配对后 HandEEBinding 侧向翻转符号随之换号（side_flip），两种配对
    # 都保证"手往外->臂往外"；ready_q/夹爪/肩body 均随臂走，不随名字走。
    if args.no_swap_hands:
        hand_left_arm, hand_left_shoulder = left_arm, left_shoulder_body
        hand_right_arm, hand_right_shoulder = right_arm, right_shoulder_body
        hand_left_ready, hand_right_ready = READY_LEFT_Q, READY_RIGHT_Q
        hand_left_gripper, hand_left_gjoints = names["left_gripper"], names["left_gripper_joints"]
        hand_right_gripper, hand_right_gjoints = names["right_gripper"], names["right_gripper_joints"]
        side_flip_pair = True
        LOGGER.info("左右配对: 按模型命名（--no-swap-hands）：左手柄->left_elfin，右手柄->right_elfin")
    else:
        hand_left_arm, hand_left_shoulder = right_arm, right_shoulder_body
        hand_right_arm, hand_right_shoulder = left_arm, left_shoulder_body
        hand_left_ready, hand_right_ready = READY_RIGHT_Q, READY_LEFT_Q
        hand_left_gripper, hand_left_gjoints = names["right_gripper"], names["right_gripper_joints"]
        hand_right_gripper, hand_right_gjoints = names["left_gripper"], names["left_gripper_joints"]
        side_flip_pair = False
        LOGGER.info("左右配对: 按物理侧交换（默认）：左手柄->right_elfin(操作者视角左侧)，右手柄->left_elfin(视角右侧)")

    if args.arm_mode == "follow":
        LOGGER.info("注册(follow): 左手柄 L_TRANSFORM -> 视角左臂 E05 IK 持续跟随")
        add_arm_ik_pico_controller(
            manager, env, hand_left_arm, base_body, device, PicoJoystickKey.L_TRANSFORM,
            chassis_body=chassis_body, arm_scale=args.arm_scale,
            shoulder_body=hand_left_shoulder,
            goal_radius=args.goal_radius, reach_limit=args.reach_limit,
            tag="左手", side_flip=side_flip_pair,
        )
        LOGGER.info("注册(follow): 右手柄 R_TRANSFORM -> 视角右臂 E05 IK 持续跟随")
        add_arm_ik_pico_controller(
            manager, env, hand_right_arm, base_body, device, PicoJoystickKey.R_TRANSFORM,
            chassis_body=chassis_body, arm_scale=args.arm_scale,
            shoulder_body=hand_right_shoulder,
            goal_radius=args.goal_radius, reach_limit=args.reach_limit,
            tag="右手", side_flip=side_flip_pair,
        )
    else:
        # clutch：点一下 Grip 即开始持续跟随（再点就地对中）；lock：永久锁定。
        allow_control = args.arm_mode == "clutch"
        LOGGER.info(
            "注册左手臂" + ("(clutch): 点一下 L_GRIP 开始持续跟随（无需按住），再点就地对中" if allow_control else "(lock): 永久锁定")
        )
        hand_left_clutch = ClutchArmController(
            env, hand_left_arm, base_body, device,
            PicoJoystickKey.L_TRANSFORM, PicoJoystickKey.L_GRIPBUTTON,
            tag="左手", allow_control=allow_control,
            chassis_body=chassis_body, arm_scale=args.arm_scale,
            shoulder_body=hand_left_shoulder,
            goal_radius=args.goal_radius, reach_limit=args.reach_limit,
            ready_q=(hand_left_ready if use_ready_pose else None),
            side_flip=side_flip_pair,
        )
        manager.add_controller(hand_left_clutch)
        LOGGER.info(
            "注册右手臂" + ("(clutch): 点一下 R_GRIP 开始持续跟随（无需按住），再点就地对中" if allow_control else "(lock): 永久锁定")
        )
        hand_right_clutch = ClutchArmController(
            env, hand_right_arm, base_body, device,
            PicoJoystickKey.R_TRANSFORM, PicoJoystickKey.R_GRIPBUTTON,
            tag="右手", allow_control=allow_control,
            chassis_body=chassis_body, arm_scale=args.arm_scale,
            shoulder_body=hand_right_shoulder,
            goal_radius=args.goal_radius, reach_limit=args.reach_limit,
            ready_q=(hand_right_ready if use_ready_pose else None),
            side_flip=side_flip_pair,
        )
        manager.add_controller(hand_right_clutch)

    LOGGER.info("注册左手夹爪: L_TRIGGER（随配对控制视角左臂的 M20）")
    hand_left_gripper = M20TriggerController(
        env, hand_left_gripper, hand_left_gjoints,
        base_body, device, PicoJoystickKey.L_TRIGGER, tag="左手"
    )
    manager.add_controller(hand_left_gripper)

    LOGGER.info("注册右手夹爪: R_TRIGGER（随配对控制视角右臂的 M20）")
    hand_right_gripper = M20TriggerController(
        env, hand_right_gripper, hand_right_gjoints,
        base_body, device, PicoJoystickKey.R_TRIGGER, tag="右手"
    )
    manager.add_controller(hand_right_gripper)

    LOGGER.info(
        "注册移动底盘(浮动阿克曼): 左摇杆x=转向, 右摇杆y=前进/后退（与 G1 一致）"
    )
    manager.add_controller(
        FloatingAckermannBaseController(
            env,
            names["base_drives"],
            names["chassis_body"],
            names["base_yaw_joint"],
            device,
        )
    )

    # 右手 A 键全局复位（仅 clutch 模式）：双臂回预备姿势并锁定、夹爪全开、
    # 手柄重新对中，底盘位置/朝向保持不动。只在按下沿触发一次。
    if args.arm_mode == "clutch":
        def _on_reset_button(pressed: bool):
            if not pressed:
                return
            LOGGER.info(">>> 右手 A 键复位：双臂回预备姿势 + 夹爪全开 + 重新对中（底盘不动）")
            hand_left_clutch.request_reset()
            hand_right_clutch.request_reset()
            hand_left_gripper.reset()
            hand_right_gripper.reset()

        device.bind_primary_button_event(PicoJoystickKey.A, _on_reset_button)
        LOGGER.info("已绑定右手 A 键 = 全局复位（双臂回预备姿势+夹爪全开+重新对中，底盘不动）")

    if args.arm_mode == "clutch":
        LOGGER.info("就绪：双臂启动锁定在预备姿势；点一下左/右 Grip 该臂即开始持续跟随（再点就地对中）；右手 A 全局复位；扳机控夹爪")
    elif args.arm_mode == "lock":
        LOGGER.info("就绪：双臂永久锁定；左右扳机仅控制 M20 夹爪")
    else:
        LOGGER.info("就绪：双手位姿持续控制双臂，左右扳机控制 M20 夹爪")
    LOGGER.info("底盘：左摇杆左右=转向、右摇杆前后=行驶（同智元 G1）；Ctrl+C 退出")

    manager.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt")
    except Exception as exc:
        LOGGER.error(f"致命错误: {exc}\n{traceback.format_exc()}")
        raise
