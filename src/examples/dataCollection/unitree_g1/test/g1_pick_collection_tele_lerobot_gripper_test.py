"""宇树 g1_pick 臂 + 智元 OmniPicker 夹爪 VR 遥操作采集 → LeRobot（uni_test2 测试副本）。

基于 g1_pick_with_gripper_collection_tele_lerobot.py：
  - 臂 / IK / 腿锁 / 夹爪控制逻辑不变（Controller2F85Reverse）
  - 默认场景改为 uni_test2.json
  - 相机端口对照 uni_test2：head=7090 / wrist_r=7080

用法：
  python -u g1_pick_collection_tele_lerobot_gripper_test.py \\
      --level default --task_config ../../common/example.yaml \\
      --scene_json uni_test2.json \\
      --agent_name g1_pick_with_gripper_usda_1 \\
      --task \"抓取测试\" \\
      --lerobot_out g1_unitree_gripper_test2 \\
      --repo_id local/g1_pick_gripper_test2 \\
      --fps 20 --clock wall --cameras head,wrist_r \\
      --orcagym_addr localhost:50051 \\
      --xr_backend televuer --tv_no_tls \\
      --tv_goal_mode rebased_tv --tv_ee_dx 0.03

默认数据集根目录: OrcaManipulation/L_dataset/unitree
"""
import argparse
import dataclasses
import gc
import os
import sys
import threading
import time
import traceback
from pathlib import Path

# IPOPT 的 MUMPS/BLAS 会为每次因式分解开一个 OMP 并行区。本问题只有 7 个决策
# 变量、函数求值 0.43ms，24 线程的同步开销反而占 20ms（实测 22.8ms → 3.4ms）。
# 必须在 numpy / casadi / pinocchio 载入前设置，否则 OMP 运行时已初始化。
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# forkserver preload("dataStorage.encoder_proc") 需要子进程也能 import dataStorage
_pp = os.environ.get("PYTHONPATH", "")
if project_root not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        project_root + (os.pathsep + _pp if _pp else "")
    )

_ORCA_MANIP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DEFAULT_LEROBOT_ROOT = os.path.join(_ORCA_MANIP_ROOT, "L_dataset", "unitree")
DEFAULT_LEROBOT_OUT = os.path.join(DEFAULT_LEROBOT_ROOT, "g1_unitree_gripper_test2")


def resolve_lerobot_out(path: str) -> str:
    """相对路径落到 DEFAULT_LEROBOT_ROOT；绝对路径原样 expand。"""
    p = os.path.expanduser(str(path).strip())
    if not os.path.isabs(p):
        p = os.path.join(DEFAULT_LEROBOT_ROOT, p)
    return os.path.abspath(p)

import numpy as np
from yaml import Loader, load

from conf import g1_omnipicker_conf, g1_pick_conf
from controllers import controllers
from controllers.controller_task import TaskStatus
from controllers.controller_task import TaskStatusController
from controllers.g1_pick_dual_arm_controller import G1PickDualArmIKController
from controllers.g1_pick_unitree_arm_ik import G1_29_ArmIK, _EL_STRAIGHT
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.g1_pick_with_gripper_data_storage import G1PickWithGripperLeRobotStorage
from dataStorage.lerobot_data_storage import (
    LeRobotDatasetWriter,
    set_logger as set_storage_logger,
)
from devices.g1_pick_device import G1PickPicoJoystickDevice
from devices.g1_pick_tv_pose_mapper import TvToOrcaPoseMapper, make_trans_x
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_pick_with_gripper_lerobot_stream"

# 左臂侧平举锁定角（Unitree G1，不参与 IK）。
# sr=+π/2 外展到左侧；el=_EL_STRAIGHT 把前臂从零位前弯~82°拉直，
# 整条臂才水平指向 +Y（左侧）。el=0 时前臂仍朝 +X，视觉上像前平举。
_L_INIT_JOINT_VALUES = [0.0, 1.5708, 0.0, float(_EL_STRAIGHT), 0.0, 0.0, 0.0]
_L_ARM_LABELS = ("sp", "sr", "sy", "el", "wr", "wp", "wy")

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="G1PickWithGripperLeRobot",
    log_file="g1_pick_with_gripper_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

# 让 dataStorage 层（含内存/NVENC 探针）的日志并入本脚本的日志文件与控制台
set_storage_logger(orca_logger)


def _fmt_arm7_deg(q7) -> str:
    arr = np.asarray(q7, dtype=np.float64).reshape(-1)
    if arr.size < 7:
        return f"(len={arr.size})"
    parts = [
        f"{_L_ARM_LABELS[i]}={np.degrees(arr[i]):+.1f}" for i in range(7)
    ]
    return "[" + " ".join(parts) + "]deg"


def _read_l_arm_q(env) -> np.ndarray | None:
    """读左臂 7 关节 qpos（rad）；失败返回 None。"""
    try:
        names = [env.joint(n) for n in g1_pick_conf.l_arm["joint_names"]]
        qdict = env.query_joint_qpos(names)
        return np.array(
            [float(np.asarray(qdict[n]).reshape(-1)[0]) for n in names],
            dtype=np.float64,
        )
    except Exception as e:
        orca_logger.warning(f"[L_ARM] 读 qpos 失败: {e}")
        return None


def _log_l_arm(env, tag: str, dual=None, target=None) -> None:
    """定位左臂平举：target / 实测 q / dual hold+cmd。"""
    tgt = list(target) if target is not None else list(_L_INIT_JOINT_VALUES)
    q = _read_l_arm_q(env)
    msg = f"[L_ARM][{tag}] target={_fmt_arm7_deg(tgt)}"
    if q is not None:
        err = q - np.asarray(tgt, dtype=np.float64)
        msg += (
            f" q={_fmt_arm7_deg(q)}"
            f" |err|_max={float(np.max(np.abs(err))):.3f}rad"
            f"({np.degrees(float(np.max(np.abs(err)))):+.1f}deg)"
            f" sr_err={np.degrees(err[1]):+.1f}deg"
        )
    else:
        msg += " q=<read_fail>"
    if dual is not None:
        try:
            hold = np.asarray(getattr(dual, "_hold_q", None), dtype=np.float64).reshape(-1)
            cmd = np.asarray(getattr(dual, "_cmd_q", None), dtype=np.float64).reshape(-1)
            if hold.size >= 7:
                msg += f" holdL={_fmt_arm7_deg(hold[:7])}"
            if cmd.size >= 7:
                msg += f" cmdL={_fmt_arm7_deg(cmd[:7])}"
            msg += f" lock_left={getattr(dual, 'lock_left', '?')} anchored={getattr(dual, '_anchored', '?')}"
        except Exception as e:
            msg += f" dual=<err:{e}>"
    orca_logger.info(msg)
    print(msg, flush=True)


def _force_dual_left_hold(dual, target=None) -> None:
    """把 dual IK 内部左臂 hold/cmd 强制写成侧平举目标（消除 reset 读实测 q 的偏差）。"""
    if dual is None:
        return
    tgt = np.asarray(
        target if target is not None else _L_INIT_JOINT_VALUES, dtype=np.float64
    ).reshape(7)
    try:
        if hasattr(dual, "_hold_q") and dual._hold_q is not None:
            hold = np.asarray(dual._hold_q, dtype=np.float64).reshape(-1).copy()
            if hold.size >= 7:
                hold[:7] = tgt
                dual._hold_q = hold
        if hasattr(dual, "_cmd_q") and dual._cmd_q is not None:
            cmd = np.asarray(dual._cmd_q, dtype=np.float64).reshape(-1).copy()
            if cmd.size >= 7:
                cmd[:7] = tgt
                dual._cmd_q = cmd
        arm_ik = getattr(dual, "arm_ik", None)
        if arm_ik is not None and hasattr(dual, "_hold_q") and dual._hold_q is not None:
            try:
                arm_ik.reset_state(np.asarray(dual._hold_q, dtype=np.float64).reshape(-1))
            except Exception:
                pass
    except Exception as e:
        orca_logger.warning(f"[L_ARM][force_hold] 失败: {e}")


def _settle_left_arm(
    env,
    manager,
    dual=None,
    max_steps: int = 40,
    tol: float = 0.05,
) -> None:
    """set_init_ctrl + 若干 env.step，直到左臂 qpos 接近 _L_INIT。"""
    try:
        manager.set_init_ctrl()
        env.set_ctrl(manager.ctrl)
        env.mj_forward()
    except Exception as e:
        orca_logger.warning(f"[L_ARM][settle] set_init_ctrl 失败: {e}")
        _force_dual_left_hold(dual)
        return

    joint_ids = [env.joint(n) for n in g1_pick_conf.l_arm["joint_names"]]
    target = np.asarray(_L_INIT_JOINT_VALUES, dtype=np.float64)
    last_err = float("inf")
    converged_at = -1
    for i in range(int(max_steps)):
        try:
            action = manager.run_controllers()
            env.step(action)
        except Exception as e:
            orca_logger.warning(f"[L_ARM][settle] step {i} 失败: {e}")
            break
        try:
            qpos = env.query_joint_qpos(joint_ids)
            q = np.array(
                [float(np.asarray(qpos[j]).reshape(-1)[0]) for j in joint_ids],
                dtype=np.float64,
            )
            last_err = float(np.max(np.abs(q - target)))
            if last_err < tol:
                converged_at = i + 1
                break
        except Exception:
            pass
    if converged_at > 0:
        orca_logger.info(
            f"[L_ARM][settle] 收敛 step={converged_at} max|err|={last_err:.4f}rad"
        )
    else:
        orca_logger.info(
            f"[L_ARM][settle] 跑满 {max_steps} 步 max|err|={last_err:.4f}rad"
        )
    _force_dual_left_hold(dual)


class LoopLagProfiler:
    """主控制环分阶段耗时：定位集内卡顿来自 ctrl / step / render / storage / stream。

    run_episode 顺序：run_controllers → env.step → env.render → collection_data(录制中)。
    每步在下一次 ctrl 开始时结算上一步 total（兼容 IDLE 不进 storage）。
    """

    def __init__(self, warn_ms: float = 80.0, log_every: int = 25,
                 diag_log=None) -> None:
        self.warn_ms = float(warn_ms)
        self.log_every = int(log_every)
        self._diag_log = diag_log   # DiagLog 对象，用于写入持久化文件
        self.n = 0
        self._stalls = 0
        self._orig: dict = {}
        self._last: dict[str, float] = {}
        self._sum_ms: dict[str, float] = {}
        self._max_ms: dict[str, float] = {}
        self._step_t0: float | None = None

    def install(self, manager, env, storage) -> None:
        self._orig["run_controllers"] = manager.run_controllers
        self._orig["step"] = env.step
        self._orig["render"] = env.render
        self._orig["collection_data"] = storage.collection_data
        writer = getattr(storage, "_lr_writer", None)
        if writer is not None:
            self._orig["stream_frame"] = writer.stream_frame
            self._orig["writer"] = writer

        def _finish_prev():
            # 先取/清空 _step_t0，再记日志，使告警 I/O 计入本步 total
            if self._step_t0 is None:
                return
            t0 = self._step_t0
            self._step_t0 = None
            self._on_step_done(t0)

        def _run_controllers():
            _finish_prev()
            self._step_t0 = time.perf_counter()
            # 新一步开始前清零本步 last，避免 IDLE 时沿用旧 storage/stream
            for k in ("ctrl", "step", "render", "storage", "stream"):
                self._last[k] = 0.0
            t0 = self._step_t0
            out = self._orig["run_controllers"]()
            self._acc("ctrl", (time.perf_counter() - t0) * 1000.0)
            return out

        def _step(action):
            t0 = time.perf_counter()
            out = self._orig["step"](action)
            self._acc("step", (time.perf_counter() - t0) * 1000.0)
            return out

        def _render(*a, **k):
            t0 = time.perf_counter()
            out = self._orig["render"](*a, **k)
            self._acc("render", (time.perf_counter() - t0) * 1000.0)
            return out

        def _collection_data(obs, env_, **kwargs):
            t0 = time.perf_counter()
            out = self._orig["collection_data"](obs, env_, **kwargs)
            total_st = (time.perf_counter() - t0) * 1000.0
            self._acc("storage", total_st)
            return out

        manager.run_controllers = _run_controllers  # type: ignore[method-assign]
        env.step = _step  # type: ignore[method-assign]
        env.render = _render  # type: ignore[method-assign]
        storage.collection_data = _collection_data  # type: ignore[method-assign]

        if writer is not None:
            def _stream_frame(frame, task):
                t0 = time.perf_counter()
                out = self._orig["stream_frame"](frame, task)
                self._acc("stream", (time.perf_counter() - t0) * 1000.0)
                return out

            writer.stream_frame = _stream_frame  # type: ignore[method-assign]

        orca_logger.info(
            f"[LAG] 探针已安装 warn>={self.warn_ms:.0f}ms every={self.log_every}"
        )

    def uninstall(self, manager, env, storage) -> None:
        if self._step_t0 is not None:
            t0 = self._step_t0
            self._step_t0 = None
            self._on_step_done(t0)
        if "run_controllers" in self._orig:
            manager.run_controllers = self._orig["run_controllers"]  # type: ignore[method-assign]
        if "step" in self._orig:
            env.step = self._orig["step"]  # type: ignore[method-assign]
        if "render" in self._orig:
            env.render = self._orig["render"]  # type: ignore[method-assign]
        if "collection_data" in self._orig:
            storage.collection_data = self._orig["collection_data"]  # type: ignore[method-assign]
        writer = self._orig.get("writer")
        if writer is not None and "stream_frame" in self._orig:
            writer.stream_frame = self._orig["stream_frame"]  # type: ignore[method-assign]
        self.log_summary(tag="uninstall")
        self._orig.clear()

    def _acc(self, key: str, ms: float) -> None:
        self._last[key] = ms
        self._sum_ms[key] = self._sum_ms.get(key, 0.0) + ms
        self._max_ms[key] = max(self._max_ms.get(key, 0.0), ms)

    def _on_step_done(self, t0: float) -> None:
        """结算一步。t0 为步起点；告警 I/O 后才记 total，避免漏计刷屏开销。"""
        self.n += 1
        provisional_ms = (time.perf_counter() - t0) * 1000.0
        if provisional_ms >= self.warn_ms:
            self._stalls += 1
            msg = (
                f"[LAG][{self.n}] total={provisional_ms:.1f}ms "
                f"ctrl={self._last.get('ctrl', 0):.1f} "
                f"step={self._last.get('step', 0):.1f} "
                f"render={self._last.get('render', 0):.1f} "
                f"storage={self._last.get('storage', 0):.1f} "
                f"cam={self._last.get('cam', 0):.1f} "
                f"stream={self._last.get('stream', 0):.1f}"
            )
            # 只输出一次：diag_log.log 内部已 print；不再 orca_logger.warning / 二次 print
            if self._diag_log is not None:
                try:
                    self._diag_log.log(msg)
                except Exception:
                    print(msg, flush=True)
            else:
                print(msg, flush=True)
        if self.log_every > 0 and self.n % self.log_every == 0:
            self.log_summary(tag=f"n={self.n}")
        # 告警 I/O 之后再累计 total
        self._acc("total", (time.perf_counter() - t0) * 1000.0)

    def log_summary(self, tag: str = "end") -> None:
        n = max(self.n, 1)
        parts = [
            f"{k}:avg={self._sum_ms.get(k, 0.0) / n:.1f}/max={self._max_ms.get(k, 0.0):.1f}"
            for k in ("ctrl", "step", "render", "storage", "stream", "total")
        ]
        msg = (
            f"[LAG][汇总 {tag}] steps={self.n} stalls(>={self.warn_ms:.0f}ms)={self._stalls} | "
            + " ".join(parts)
        )
        orca_logger.info(msg)
        print(msg, flush=True)


# =============================================================================
# OmniPicker 夹爪（TeleVuer 扳机绑定；Pico 复用 controllers 内现成 API）
# =============================================================================
def create_gripper_televuer_controller(
    manager: DataCollectionManager,
    env,
    gripper_config: dict,
    base_body: str,
    device,
    side: str,
    is_running_fn=None,
):
    """夹爪：TeleVuer 扳机 → Controller2F85Reverse（与 OmniPicker Pico 同逻辑）。"""
    ctrl_name = [
        env.actuator(n) for n in gripper_config["actuator_names"]
    ]
    init_ctrl = {
        n: v for n, v in zip(ctrl_name, gripper_config["init_ctrl"])
    }
    gc = controllers.create_gripper_2f85_reverse_controller(
        env, gripper_config, base_body, ctrl_name, init_ctrl,
    )

    def _on_trigger(value: float):
        if is_running_fn is not None and not is_running_fn():
            return
        gc.update_trigger_value(value)

    if side.upper().startswith("L"):
        device.bind_left_trigger_event(_on_trigger)
    else:
        device.bind_right_trigger_event(_on_trigger)
    manager.add_controller(gc)


# =============================================================================
# 腿部 + 腰部锁定控制器（hold 在初始 qpos）
# =============================================================================
class LeftArmFixedHoldCtrl:
    """左臂关节位置固定锁定控制器：始终输出 _L_INIT_JOINT_VALUES，覆盖双臂 IK 的左臂输出。

    排在双臂 IK 控制器之后加入 DataCollectionManager，run_controllers() 最后
    以本控制器的字典覆盖 IK 的左臂关节命令，实现关节空间硬锁（而非 EE 空间锁）。
    reset() 不读取 qpos，始终保持固定目标角度。
    """

    def __init__(self, env, target_q: list[float]) -> None:
        names = [env.actuator(n) for n in g1_pick_conf.l_arm["positions_names"]]
        self._ctrl_index = [env.model.actuator_name2id(n) for n in names]
        self._target = [float(v) for v in target_q]
        orca_logger.info(
            f"[L_HOLD] 左臂固定锁创建: actuators={names} "
            f"target_deg={[round(np.degrees(v), 1) for v in self._target]}"
        )

    # DataCollectionManager.set_init_ctrl() 会调用以下两个方法
    def init_ctrl_index(self) -> list[int]:
        return list(self._ctrl_index)

    def get_init_ctrl(self) -> dict[int, float]:
        return {self._ctrl_index[i]: self._target[i] for i in range(len(self._ctrl_index))}

    def reset(self) -> None:
        pass  # 固定目标，不随 episode reset 改变

    def run_controller(self) -> dict[int, float]:
        return {self._ctrl_index[i]: self._target[i] for i in range(len(self._ctrl_index))}


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
    orca_logger.info(
        f"[ARM-GAIN] applied n={len(applied)} kp={kp:.1f} kv={kv:.2f} "
        f"wrist_kp={wrist_kp:.1f} | " + ", ".join(applied[:4]) + (" ..." if len(applied) > 4 else "")
    )


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

    orca_logger.info(
        f"锁定下半身 {len(joint_names)} 个关节，当前 qpos: "
        f"{', '.join(f'{n}={float(init_positions[i]):.3f}' for i, n in enumerate(joint_names))}"
    )
    holder = JointHoldController(env, ctrl_name, init_positions, joint_names)
    manager.add_controller(holder)
    return holder


def pin_floating_base(env, agent_name: str) -> bool:
    """钉住浮动基座，效果接近静态刚体，但不改 nq（OrcaStudio 同步需要 nq 一致）。

    不能删 XML 里的 freejoint：本地 nq 会少 7，UpdateLocalEnv 把短 qpos 推给
    仍含 freejoint 的 OrcaStudio，关节映射错位，表现为手臂“锁死”。

    做法：包装 gym.mj_step——仅在进入前钉一次，再整批 mj_step(n)。
    骨盆/躯干/双足已有 weld 兜底；去掉后钉 + mj_forward（实测 mj_forward≈3.3ms，
    占 step 的 ~12%）。下一控制步进入时会再次钉住，漂移不跨步累积。
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
    _drift_n = [0]

    def _mj_step_pinned(nstep=1):
        n = int(nstep) if nstep is not None else 1
        n = max(n, 1)
        md.qpos[qadr:qadr + 7] = q0
        md.qvel[dadr:dadr + 6] = 0.0
        _orig_mj_step(n)
        # 子步内瞬时漂移监控（下步进入前会再钉；不跨步累积）
        drift = float(np.linalg.norm(md.qpos[qadr:qadr + 3] - q0[:3]))
        if drift > 5e-3:
            _drift_n[0] += 1
            if _drift_n[0] <= 3 or _drift_n[0] % 200 == 0:
                orca_logger.warning(
                    f"[BASE-PIN] 子步末 drift_xyz={drift:.4f}m "
                    f"(#{_drift_n[0]}；下步进入前会再钉，weld 兜底)"
                )

    gym.mj_step = _mj_step_pinned
    orca_logger.info(
        f"[BASE-PIN] freejoint pinned (pre-only, no mj_forward) qadr={qadr} dadr={dadr} "
        f"q0_xyz=({q0[0]:.3f},{q0[1]:.3f},{q0[2]:.3f}) (nq unchanged)"
    )
    return True


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
            f"lock_left={lock_left} fixed_left_q={_L_INIT_JOINT_VALUES} "
            f"dz={deadzone_pos_m}m/{deadzone_ori_deg}deg "
            f"ema={goal_ema_alpha} "
            f"max_reach={max_reach}"
        )
    ik_kwargs = {"fixed_left_q": list(_L_INIT_JOINT_VALUES)}
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
            f"dry_run={dry_run} lock_left={lock_left} "
            f"fixed_left_q={_L_INIT_JOINT_VALUES} max_dq={max_dq_step} "
            f"dz={deadzone_pos_m}m/{deadzone_ori_deg}deg "
            f"ema={goal_ema_alpha} "
            f"max_reach={max_reach} project_reachable={project_reachable}"
        )
    ik_kwargs = {"fixed_left_q": list(_L_INIT_JOINT_VALUES)}
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
    lag_warn_ms: float
    diag_log_dir: str
    tv_no_tls: bool
    time_step: float          # MuJoCo opt.timestep；场景 XML 自带 0.005
    frame_skip: int           # 每控制步物理子步数；env.dt = time_step * frame_skip
    encode_backend: str       # inproc | subproc（本脚本默认 subproc）
    enc_ring_slots: int       # subproc 共享内存环槽位数
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
            lag_warn_ms=args.lag_warn_ms,
            diag_log_dir=args.diag_log_dir,
            tv_no_tls=args.tv_no_tls,
            time_step=float(args.time_step),
            frame_skip=int(args.frame_skip),
            encode_backend=str(args.encode_backend),
            enc_ring_slots=int(args.enc_ring_slots),
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
        self._orca_log_handler = None
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

            # orca_logger 的 console handler 在 tee 之前就绑定了原始 stdout，
            # 导致 orca_logger.info 不进 --log_file。补一个指向 log_fh 的 handler。
            try:
                import logging

                _h = logging.StreamHandler(self._log_fh)
                _h.setLevel(logging.INFO)
                _h.setFormatter(
                    logging.Formatter(
                        "[%(asctime)s.%(msecs)03d] %(levelname)-11s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                orca_logger.logger.addHandler(_h)
                self._orca_log_handler = _h
            except Exception as _e:
                print(f"[日志] 挂载 orca_logger→log_file 失败: {_e}", flush=True)

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
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._dbg_fh.write(line + "\n")
        self._dbg_fh.flush()

    def close(self) -> None:
        # 先恢复 sys.stdout/stderr，再关文件，避免后续日志写入已关闭句柄
        if self._orca_log_handler is not None:
            try:
                orca_logger.logger.removeHandler(self._orca_log_handler)
            except Exception:
                pass
            self._orca_log_handler = None
        if self._log_fh is not None:
            try:
                if getattr(sys.stdout, "files", None) and self._log_fh in sys.stdout.files:
                    sys.stdout = sys.stdout.files[0]
            except Exception:
                pass
            try:
                if getattr(sys.stderr, "files", None) and self._log_fh in sys.stderr.files:
                    sys.stderr = sys.stderr.files[0]
            except Exception:
                pass
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
    """装配器：绑定双臂 IK、OmniPicker 夹爪、腿锁、任务状态与输入事件回调。

    替代 main() 内散落的 _dual_arm_holder / _is_tele_running / _on_discard 等闭包。
    持有 discard_event，供 HealthMonitor 和 EpisodeRunner 共享。
    """

    def __init__(self) -> None:
        self.dual_arm_ctrl = None
        self.discard_event = threading.Event()
        self._prev_gate_running: list = [None]
        self._diag = None   # DiagLog，由 main() 在 wire() 之前注入

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

        orca_logger.info(
            f"Adding dual-arm Unitree G1_29 CasADi IK controller "
            f"(backend={cfg.xr_backend})"
        )
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
        _force_dual_left_hold(dual)
        self.dual_arm_ctrl = dual

        # 左臂关节固定锁：紧接在双臂 IK 之后注册，覆盖执行器层左臂输出。
        # IK 已用 fixed_left_q 把左臂剔出 IPOPT（只解右臂 7 维）；本控制器作兜底，
        # 保证执行器始终收到侧平举关节角。
        l_arm_hold = LeftArmFixedHoldCtrl(env, list(_L_INIT_JOINT_VALUES))
        manager.add_controller(l_arm_hold)
        self.l_arm_hold_ctrl = l_arm_hold
        orca_logger.info(
            f"[LEFT ARM] 关节固定锁已注册 target={_L_INIT_JOINT_VALUES} "
            f"(IK fixed_left_q 右臂单独求解)"
        )

        # OmniPicker 夹爪（与 g1_omnipicker_collection_tele_lerobot 一致）
        orca_logger.info("Adding left gripper controller")
        if cfg.xr_backend == "pico":
            controllers.add_gripper_2f85_reverse_pico_controller(
                manager, env,
                g1_omnipicker_conf.gripper_l,
                g1_pick_conf.base_body,
                pico_device,
                [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
            )
            orca_logger.info("Adding right gripper controller")
            controllers.add_gripper_2f85_reverse_pico_controller(
                manager, env,
                g1_omnipicker_conf.gripper_r,
                g1_pick_conf.base_body,
                pico_device,
                [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
            )
        else:
            create_gripper_televuer_controller(
                manager, env,
                g1_omnipicker_conf.gripper_l,
                g1_pick_conf.base_body,
                xr_device, "L",
                is_running_fn=_is_tele_running,
            )
            orca_logger.info("Adding right gripper controller")
            create_gripper_televuer_controller(
                manager, env,
                g1_omnipicker_conf.gripper_r,
                g1_pick_conf.base_body,
                xr_device, "R",
                is_running_fn=_is_tele_running,
            )

        # 腿锁
        orca_logger.info("Locking lower body (legs + waist)")
        lock_lower_body(manager, env)

        # 钉住浮动基座（不删 freejoint，避免 nq 与 OrcaStudio 不一致导致手臂映射错乱）
        pin_floating_base(env, cfg.agent_name)

        # 任务状态控制器
        orca_logger.info("Setting task and task status controller")
        manager.set_task(EmptyTask(env))
        if cfg.xr_backend == "pico":
            controllers.add_task_status_pico_controller(
                manager, env, pico_device, g1_pick_conf.base_body,
            )
        else:
            tsc = TaskStatusController(env, g1_pick_conf.base_body)

            def _tv_task_toggle(_pressed: bool = True):
                tsc.update_task_status(True)

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
                    _url = (
                        xr_device.client_visit_url()
                        if hasattr(xr_device, "client_visit_url")
                        else (
                            "http://127.0.0.1:8012/"
                            if cfg.tv_no_tls
                            else "https://127.0.0.1:8012/"
                        )
                    )
                    diag.log(
                        "[TV] 未收到 XR 位姿：USB-local 请 "
                        "adb reverse tcp:8012 tcp:8012 后打开 "
                        f"{_url}"
                        + ("（明文 HTTP，勿用 https）" if cfg.tv_no_tls else "（先过证书）")
                        + "，等终端出现 websocket is connected 且右上角不再狂闪 reconnect，"
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
                # cmd = 实测执行器 ctrl（LeftArmFixedHoldCtrl 覆盖后的真值）
                cmd = np.array(
                    [float(env.ctrl[i]) for i in _arm_act_ids],
                    dtype=np.float64,
                )
                # IK 内部 _cmd_q（可能与执行器不同；右臂单独求解时左臂应等于 fixed）
                dac = wiring.dual_arm_ctrl
                if dac is not None and hasattr(dac, "_cmd_q"):
                    ik_cmd = np.asarray(dac._cmd_q, dtype=np.float64).reshape(14)
                else:
                    ik_cmd = cmd.copy()

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

                _j_n[0] += 1
                # 合并为 3 行，降低 GIL 下的日志 I/O
                diag.log(
                    f"[JOINT] #{_j_n[0]} status={status} "
                    f"max|q|={max_abs_q:.3f} max|dq|={max_abs_dq:.2f} "
                    f"max|cmd-q|={max_abs_err:.3f} ee={ee_str}"
                )
                diag.log(
                    f"[JOINT] L_q[{_fmt7_deg(q[:7])}] R_q[{_fmt7_deg(q[7:])}] | "
                    f"L_cmd[{_fmt7_deg(cmd[:7])}] R_cmd[{_fmt7_deg(cmd[7:])}]"
                )
                diag.log(
                    f"[JOINT] L_err[{_fmt7_deg(err[:7])}] R_err[{_fmt7_deg(err[7:])}] | "
                    f"L_ik[{_fmt7_deg(ik_cmd[:7])}] R_ik[{_fmt7_deg(ik_cmd[7:])}] | "
                    f"L_dq[{_fmt7_deg(dq[:7])}] R_dq[{_fmt7_deg(dq[7:])}]"
                )
                # time_step=0.005 验收：kp=150 下若震荡，dq/跟踪误差会显著抬升
                # （改前稳态 max|dq|≈0.01~0.05；>2 rad/s 或 |cmd-q|>0.25rad 视为可疑）
                if max_abs_dq > 2.0 or max_abs_err > 0.25:
                    diag.log(
                        f"[JOINT][OSC?] 可疑震荡 max|dq|={max_abs_dq:.2f} "
                        f"max|cmd-q|={max_abs_err:.3f} "
                        f"(time_step={float(cfg.time_step):.4f}；"
                        f"可试 --time_step 0.002)"
                    )
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
        default_joint_values: dict | None = None,
        wiring: "ControllerWiring | None" = None,
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
        self._default_joint_values = default_joint_values or {}
        self._wiring = wiring

    def run(self) -> None:
        cfg = self._cfg
        env = self._env
        manager = self._manager
        writer = self._writer
        storage = self._storage
        discard_event = self._discard_event

        _ep_idx = 0
        dual = self._wiring.dual_arm_ctrl if self._wiring is not None else None
        while not manager._shutdown_requested:  # noqa: SLF001
            _ep_idx += 1
            orca_logger.info(
                f"[L_ARM] ---- episode_prep #{_ep_idx} "
                f"default_keys={list(self._default_joint_values.keys())[:4]}... "
                f"n={len(self._default_joint_values)} ----"
            )
            # 打印将要写入的左臂目标（含 env.joint 前缀后的全名）
            try:
                for jn, v in zip(
                    g1_pick_conf.l_arm["joint_names"], _L_INIT_JOINT_VALUES
                ):
                    full = env.joint(jn)
                    orca_logger.info(
                        f"[L_ARM][map] {jn} -> {full} = {v:.4f}rad "
                        f"({np.degrees(v):+.1f}deg)"
                    )
            except Exception as e:
                orca_logger.warning(f"[L_ARM][map] 失败: {e}")

            env.reset()
            _log_l_arm(env, "1_after_env.reset", dual=dual)

            # 先 update_scene（会从 OrcaStudio 拉回 actor qpos，可能覆盖 reset）
            if not cfg.local_xml:
                t_us = time.perf_counter()
                if not manager.update_scene():
                    orca_logger.info("update_scene 失败，停止采集")
                    break
                orca_logger.info(
                    f"[L_ARM] update_scene 耗时 {(time.perf_counter()-t_us)*1000:.0f}ms"
                )
                _log_l_arm(env, "2_after_update_scene", dual=dual)

            # 再强制侧平举（必须在 update_scene 之后，否则会被覆盖）
            env.set_default_joint_values(self._default_joint_values)
            try:
                env.mj_forward()
            except Exception:
                pass
            _log_l_arm(env, "3_after_set_default", dual=dual)

            # settle：ctrl 驱动若干步，直到左臂 qpos 到位；再强制 dual 内部 hold
            _settle_left_arm(env, manager, dual=dual, max_steps=40, tol=0.05)
            _log_l_arm(env, "3b_after_settle", dual=dual)

            if dual is not None:
                dual.reset()
                _force_dual_left_hold(dual)
                _log_l_arm(env, "4_after_dual.reset", dual=dual)

            time.sleep(0.05)
            _log_l_arm(env, "5_after_sleep", dual=dual)

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
            orca_logger.info(f"左臂侧平举锁定 L_INIT={_L_INIT_JOINT_VALUES}")
            if cfg.xr_backend == "pico":
                print(f"\n>>> 第 {_collecting_no} 集（左Grip开始，再按保存）", flush=True)
            else:
                print(
                    f"\n>>> 第 {_collecting_no} 集（左squeeze开始，再按保存）",
                    flush=True,
                )

            # 趁等待 squeeze 的空档把 NVENC 会话开好：avcodec_open2 独占 GIL 约
            # 430ms（两路相机），留到集内首帧会把整个进程冻住。
            try:
                _hw = getattr(storage, "_lr_target_hw", None) or (480, 640)
                _dt_pe = writer.prepare_episode(int(_hw[0]), int(_hw[1]))
                if _dt_pe > 0.0:
                    print(f"[NVENC] 本集会话已就绪（{_dt_pe*1000:.0f}ms）", flush=True)
            except Exception as e:
                orca_logger.warning(f"[NVENC] 集前预建失败，退回惰性创建: {e}")

            # run_episode 开头会对所有 controller 再 reset()，再打一次 L_ARM 定位
            _dual_reset_orig = None
            if dual is not None:
                _dual_reset_orig = dual.reset

                def _reset_logged():
                    _log_l_arm(env, "5b_before_run_episode.dual.reset", dual=dual)
                    _dual_reset_orig()
                    _force_dual_left_hold(dual)
                    _log_l_arm(env, "5c_after_run_episode.dual.reset", dual=dual)

                dual.reset = _reset_logged  # type: ignore[method-assign]

            lag = LoopLagProfiler(
                warn_ms=float(cfg.lag_warn_ms),
                log_every=25,
                diag_log=self._wiring._diag if self._wiring is not None else None,
            )
            lag.install(manager, env, storage)
            _t0 = time.perf_counter()
            try:
                _ok, _start, _end, _qpos = manager.run_episode()
            finally:
                lag.uninstall(manager, env, storage)
                if dual is not None and _dual_reset_orig is not None:
                    dual.reset = _dual_reset_orig  # type: ignore[method-assign]
            _dur = time.perf_counter() - _t0
            _nframes = storage.buffered_frame_count
            _log_l_arm(env, "6_after_run_episode", dual=dual)

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
                try:
                    gc.collect()
                except Exception:
                    pass
                orca_logger.info(f"[EP {_ep_idx}] 已丢弃本集")
                print(f"[EP] discard cleared ep={_ep_idx}", flush=True)
                continue

            if manager._shutdown_requested:  # noqa: SLF001
                storage.clear_data()
                try:
                    gc.collect()
                except Exception:
                    pass
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
            try:
                gc.collect()
            except Exception:
                pass
            orca_logger.info(f"已保存，共 {writer.num_episodes} 集 / {writer.num_frames} 帧")
            print(f">>> 已保存，共 {writer.num_episodes} 集", flush=True)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="g1_pick VR 遥操作采集 → LeRobot v2.1")
    parser.add_argument("--level", type=str, default="default", help="场景名称")
    parser.add_argument("--task_config", default="../../common/example.yaml", help="任务配置 YAML")
    parser.add_argument(
        "--lerobot_out",
        default=DEFAULT_LEROBOT_OUT,
        help=(
            f"数据集输出目录（默认 {DEFAULT_LEROBOT_OUT}；"
            f"相对路径会落到 {DEFAULT_LEROBOT_ROOT}/ 下）"
        ),
    )
    parser.add_argument("--repo_id", default="local/g1_pick_with_gripper", help="LeRobot repo_id")
    parser.add_argument("--task", default="g1 pick with gripper teleoperation", help="任务描述")
    parser.add_argument("--fps", type=int, default=20, help="采集帧率")
    parser.add_argument("--clock", choices=("sim", "wall"), default="wall")
    parser.add_argument("--resume", action="store_true", help="追加到已有数据集")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument("--cameras", default="head,wrist_r",
                        help="启用的相机列表（默认 head + wrist_r）")
    parser.add_argument("--cam_resolution", default="480x640", help="分辨率 HxW")
    parser.add_argument("--camera_source", choices=("websocket", "mp4"), default="websocket")
    parser.add_argument("--local_xml", default=None,
                        help="使用本地 patched XML（跳过 gRPC 加载）")
    parser.add_argument("--scene_json", default="uni_test2.json",
                        help="场景 JSON 文件名（仅文档/提示用；实际场景以 OrcaStudio 已加载为准）")
    parser.add_argument(
        "--time_step",
        type=float,
        default=0.005,
        help=(
            "MuJoCo opt.timestep（秒）。默认 0.005=场景 XML 自带值；"
            "框架默认 0.001 会把仿真压成 12.5%% 实时。震荡时可试 0.002"
        ),
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=8,
        help=(
            "每控制步物理子步数（默认 8；env.dt=time_step×frame_skip）。"
            "加大可摊薄每圈固定的 IK/render 开销，提高仿真实时率"
        ),
    )
    parser.add_argument(
        "--agent_name",
        default="g1_pick_with_gripper_usda_1",
        help="仿真中机器人 actor 名（关节前缀）。uni_test2 请用 g1_pick_with_gripper_usda_1",
    )
    parser.add_argument("--log_file", default=None,
                        help="终端输出日志路径（同时输出到文件和终端）")
    parser.add_argument("--diag_tele", action=argparse.BooleanOptionalAction, default=True,
                        help="启用分层遥操诊断日志（默认开；--no-diag_tele 关闭）")
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
        default=0.10,
        help="每控制周期最大关节步长 (rad)；默认 0.10≈2.5rad/s@25Hz。"
             "取 0.8 时限幅几乎不触发，手柄挥快会让 IK 一次性下发十几度阶跃，"
             "实臂追不上再猛甩，表现为顿挫并触发 OSC 告警",
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
        help="目标 EMA 系数 [0,1]；1=不平滑，默认 0.95（原 0.85 偏拖尾）",
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
        help="恢复旧行为：把超程目标钳到 max_reach 球面。"
        "该球面落在各方向真实可达（0.434~0.490m）之内，会让肘部保持弯曲 15~40°",
    )
    parser.add_argument(
        "--diag_health",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="TeleVuer 分层健康心跳 [HEALTH]（默认关；交付前可关。也可用环境变量 G1_DIAG_HEALTH=1）",
    )
    parser.add_argument(
        "--diag_joints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="后台轮询双臂 14 关节 qpos/ctrl/角速度/跟踪误差（默认开；--no-diag_joints 关）",
    )
    parser.add_argument(
        "--diag_joints_hz",
        type=float,
        default=2.0,
        help="关节诊断采样频率 Hz（默认 2；跟手抖动排查可开到 10~20）",
    )
    parser.add_argument(
        "--lag_warn_ms",
        type=float,
        default=80.0,
        help="主循环单步耗时告警阈值 ms（默认 80；OMP=1 后稳态应远低于此）",
    )
    parser.add_argument(
        "--encode_backend",
        choices=("inproc", "subproc"),
        default="inproc",
        help=(
            "视频/统计JPEG 编码后端：inproc=同进程线程（默认，与 TeleVuer 兼容）；"
            "subproc=独立进程（实验：消除 GIL 冻结，勿与运行中的 fuser -k 8012 混用）"
        ),
    )
    parser.add_argument(
        "--enc_ring_slots",
        type=int,
        default=96,
        help="subproc 共享内存环槽位数（默认 96≈86MB @480x640；背压时可加大）",
    )
    parser.add_argument(
        "--diag_log_dir",
        default="/tmp/g1pick_tele_diag",
        help="调试模式下每次运行保存完整诊断日志的目录（带时间戳，默认 /tmp/g1pick_tele_diag）",
    )
    parser.add_argument(
        "--tv_no_tls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "TeleVuer USB-local 明文 HTTP/WS（免证书）。"
            "推荐：127.0.0.1 是 secure context，WebXR 可用；"
            "且避开 vuer 0.0.60 HTTPS getSocketURI 丢端口 bug。"
            "开启后请打开 http://127.0.0.1:8012/（不是 https）"
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
        f"[DIAG] tele diag={'ON' if cfg.diag_tele else 'OFF'} every={cfg.diag_every} "
        f"solver=UnitreeG1_29_CasADi",
        flush=True,
    )
    if diag.enabled:
        print(
            f"[DIAG] 本次完整日志 → {diag.log_path}\n"
            f"[DIAG] 最新软链     → {diag.latest_link}  (tail -f 可用)",
            flush=True,
        )
    # CTRL# 等 manager 调试行仅在诊断开启时挂接，避免正常采集刷屏
    if diag.enabled:
        DataCollectionManager.set_debug_log_fh(diag._dbg_fh)

    # ── 相机配置 ─────────────────────────────────────────────────────────────
    _CAM_KEY_MAP = {
        "head": "camera_head_color",
        "wrist_l": "camera_wrist_l_color",
        "wrist_r": "camera_wrist_r_color",
    }
    # 对照 uni_test2.json：head_cam=7090 / camera_right(wrist_r)=7080
    G1_PICK_WITH_GRIPPER_CAMERA_MAP = {
        "camera_head_color": ("cam_head", 7090),
        "camera_wrist_r_color": ("cam_wrist_r", 7080),
    }
    _enabled = {k.strip() for k in cfg.cameras.split(",")}
    camera_map = {
        env_name: (key, port)
        for env_name, (key, port) in G1_PICK_WITH_GRIPPER_CAMERA_MAP.items()
        if any(env_name == _CAM_KEY_MAP.get(k) for k in _enabled)
    }
    if not camera_map:
        orca_logger.warning("--cameras 未匹配到已知相机，回退 head+wrist_r")
        camera_map = dict(G1_PICK_WITH_GRIPPER_CAMERA_MAP)

    try:
        _h, _w = (int(x) for x in cfg.cam_resolution.lower().split("x"))
        cam_hw_override = (_h, _w)
    except Exception:
        cam_hw_override = DEFAULT_HW

    # ── 初始关节值（左臂侧平举锁定；右臂中性；夹爪由 gripper controller 负责）──
    default_joint_values: dict = {}
    for jn, v in zip(g1_pick_conf.l_arm["joint_names"], _L_INIT_JOINT_VALUES):
        default_joint_values[jn] = float(v)
    for jn, v in zip(
        g1_pick_conf.r_arm["joint_names"],
        g1_pick_conf.r_arm["neutral_joint_values"],
    ):
        default_joint_values[jn] = float(v)
    # 同步 positions_init，避免臂控制器/reset 又把左臂拉回 0
    g1_pick_conf.l_arm["neutral_joint_values"] = list(_L_INIT_JOINT_VALUES)
    g1_pick_conf.l_arm["positions_init_ctrl"] = list(_L_INIT_JOINT_VALUES)

    # ── VR 设备 ──────────────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("  g1_pick+OmniPicker 夹爪 LeRobot 数采启动中...", flush=True)
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
            cert_file="" if cfg.tv_no_tls else None,
            key_file="" if cfg.tv_no_tls else None,
        )
        print(
            f"  TeleVuer visit URL: {xr_device.client_visit_url()} "
            f"(tls={'off' if cfg.tv_no_tls else 'on'})",
            flush=True,
        )

    # ── 场景管理 ─────────────────────────────────────────────────────────────
    scene_config_path = os.path.join(base_dir, cfg.task_config)
    with open(scene_config_path, "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)

    scene_manager = SceneManager(cfg.orcagym_addr, config=scene_config)
    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.show_ui_message(1, "g1_pick+夹爪 遥操作启动中", "0xffff00", showtime=10)
    scene_manager.get_scene_data(script_name, "beginscene")

    # ── Storage ──────────────────────────────────────────────────────────────
    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", "g1_pick_with_gripper", cfg.level)
    storage = G1PickWithGripperLeRobotStorage(dataset_path=scratch_dir)
    _n_motor = (
        len(g1_omnipicker_conf.gripper_l["actuator_names"])
        + len(g1_omnipicker_conf.gripper_r["actuator_names"])
    )

    def _obs_callback_safe(env):
        if env.model.nu == 0:
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(_n_motor, dtype=np.float32),
                "/action/drive/ctrl": np.zeros(0, dtype=np.float32),
            }
        return storage.obs_callback(env)

    # ── DataCollectionManager ────────────────────────────────────────────────
    agent_name = cfg.agent_name
    orca_logger.info(f"agent_name={agent_name}（关节/body 前缀）")
    manager = DataCollectionManager(
        agent_name=agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=_obs_callback_safe,
        env_index=0,
        device=xr_device,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=int(cfg.frame_skip),
        time_step=float(cfg.time_step),
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
    # 诊断：确认框架没有把场景 XML 的 5ms 步长改回 1ms
    try:
        try:
            _opt_ts = float(env.gym._mjModel.opt.timestep)
        except Exception:
            _opt_ts = float(cfg.time_step)
        _dt = float(cfg.time_step) * int(cfg.frame_skip)
        orca_logger.info(
            f"[SIM-DT] time_step={cfg.time_step:.4f}s frame_skip={cfg.frame_skip} "
            f"env.dt={_dt:.4f}s real_time_step={manager.real_time_step:.4f}s "
            f"mjModel.opt.timestep={_opt_ts:.6f}s "
            f"(目标控制环≈{1.0 / max(_dt, 1e-6):.0f}Hz；"
            f"若 opt.timestep≈0.001 说明仍被框架覆盖)"
        )
        print(
            f"[SIM-DT] dt={_dt * 1000:.1f}ms/圈 (time_step={cfg.time_step * 1000:.1f}ms "
            f"× frame_skip={cfg.frame_skip})  opt.timestep={_opt_ts * 1000:.2f}ms",
            flush=True,
        )
    except Exception as _dte:
        orca_logger.warning(f"[SIM-DT] 诊断失败: {_dte}")

    # gRPC 模式：拦截 XML 加载，所有修改写入旁路副本，原始缓存文件不动
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

            # ── 2. 臂+夹爪 body gravcomp（MuJoCo 原生重力补偿，不改 nq）───────
            _arm_grip_links = [
                f"{agent_name}_{side}_{link}"
                for side in ("left", "right")
                for link in (
                    "shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link",
                    "elbow_link",
                    "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link",
                )
            ] + [
                f"{agent_name}_arm_l_end_link",
                f"{agent_name}_arm_r_end_link",
                f"{agent_name}_gripper_l_inner_link1",
                f"{agent_name}_gripper_l_outer_link1",
                f"{agent_name}_gripper_r_inner_link1",
                f"{agent_name}_gripper_r_outer_link1",
            ]
            gc_injected = 0
            if _gc > 0.0:
                for link_name in _arm_grip_links:
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

            # ── 写入旁路副本，原始文件保持不动 ──────────────────────────────
            import pathlib
            orig = pathlib.Path(orig_path)
            patched_path = str(orig.with_stem(orig.stem + "_teleop_patched"))
            with open(patched_path, "w") as f:
                f.write(xml)
            orca_logger.info(
                f"✓ XML 旁路副本: {orig.name} → {pathlib.Path(patched_path).name} "
                f"patches={patches_applied or ['none']}"
            )
            return patched_path

        env.gym.load_model_xml = _patched_load_model_xml
        orca_logger.info(
            f"gRPC 模式：XML 旁路副本（原始不动，保留 freejoint）"
            f" gravcomp={_gc} + runtime BASE-PIN"
        )

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
    wiring._diag = diag   # 注入 DiagLog，供 LoopLagProfiler 写入持久文件

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
            _log_l_arm(env, "init_after_set_default", dual=None)
            wiring.wire(cfg, env, manager, xr_device, pico_device, diag)
            _log_l_arm(env, "init_after_wire", dual=wiring.dual_arm_ctrl)

            # 初始化 settle：ctrl 驱动左臂到侧平举，再强制 dual 内部 hold
            _settle_left_arm(
                env, manager, dual=wiring.dual_arm_ctrl, max_steps=40, tol=0.05
            )
            _log_l_arm(env, "init_after_settle", dual=wiring.dual_arm_ctrl)

            # IK 预热：IPOPT 首次求解触发 JIT 编译（~90ms）会造成启动卡顿。
            # 在相机 bring_up 期间提前做一次哑解算，使后续正式采集时 solve≤30ms。
            # 注意：IDLE 状态下 run_controller() 不调用 IK（直接返回 hold_q），
            # 需要临时绕过 is_running_fn gate，并设 _was_running=True 跳过状态转换。
            if wiring.dual_arm_ctrl is not None:
                try:
                    orca_logger.info("[IK 预热] 开始哑求解以触发 CasADi JIT...")
                    t_w = time.perf_counter()
                    _dual = wiring.dual_arm_ctrl
                    _orig_fn = _dual.is_running_fn
                    # is_running_fn=None 等效 running=True（见 run_controller 第813行）
                    _dual.is_running_fn = None
                    _dual._was_running = True   # 跳过 IDLE→RUNNING，不触发 _clutch_tv
                    try:
                        _dual.run_controller()  # 真正调用 IK，触发 JIT 编译
                    finally:
                        _dual.is_running_fn = _orig_fn
                        _dual._was_running = False
                        _dual.reset()           # 恢复干净状态
                        _force_dual_left_hold(_dual)
                    orca_logger.info(
                        f"[IK 预热] 完成，耗时 {(time.perf_counter()-t_w)*1000:.0f}ms（首次 JIT）"
                    )
                    _log_l_arm(env, "init_after_ik_warmup", dual=wiring.dual_arm_ctrl)
                except Exception as _we:
                    orca_logger.warning(f"[IK 预热] 失败（不影响采集）: {_we}")

            # 长寿对象（MuJoCo / CasADi JIT）移出 GC 扫描集；主循环期间禁用自动 GC
            try:
                gc.collect()
                gc.freeze()
                gc.disable()
                orca_logger.info("[GC] freeze+disable：主循环禁用自动 GC，集边界手动 collect")
            except Exception as _gce:
                orca_logger.warning(f"[GC] 托管失败（不影响采集）: {_gce}")

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
        print("  左臂移动    已锁定（侧平举停靠，不响应手柄）", flush=True)
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
        print("  左臂移动    已锁定（侧平举停靠，不响应手柄）", flush=True)
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
    print(
        f"  仿真步长: time_step={cfg.time_step*1000:.1f}ms "
        f"frame_skip={cfg.frame_skip} "
        f"→ dt={cfg.time_step*cfg.frame_skip*1000:.1f}ms/圈 "
        f"(目标≈{1.0/max(cfg.time_step*cfg.frame_skip,1e-6):.0f}Hz)",
        flush=True,
    )
    print(
        "  验收: 看 [SIM-DT] opt.timestep≈5ms；[JOINT][OSC?] 不应刷屏；"
        "手速正常跟随时无全程慢放感",
        flush=True,
    )
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
            robot_type="g1_pick_with_gripper_q_delta",
            encode_backend=cfg.encode_backend,
            enc_ring_slots=cfg.enc_ring_slots,
        )
        storage.configure_lerobot(
            fps=cfg.fps, cameras=cameras, camera_map=camera_map,
            target_hw=cam_hw, writer=writer, task=cfg.task,
            clock=cfg.clock, camera_source=cfg.camera_source,
        )
        print(
            f"[ENCODE] backend={cfg.encode_backend} "
            f"ring_slots={cfg.enc_ring_slots}",
            flush=True,
        )
        # NVENC 冷启动预热：subproc 在子进程内完成，主进程不碰 CUDA
        try:
            _h, _w = (int(cam_hw[0]), int(cam_hw[1])) if cam_hw else (480, 640)
            _cams = list(camera_map.keys()) if camera_map else ["head", "wrist_r"]
            _dt_pw = writer._nvenc_enc.prewarm(_cams, _h, _w)
            print(
                f"[NVENC] 预热完成 backend={cfg.encode_backend} "
                f"{_w}x{_h} cams={_cams} 耗时 {_dt_pw*1000:.0f}ms",
                flush=True,
            )
        except Exception as _pwe:
            orca_logger.warning(f"[NVENC] 预热失败（不影响采集）: {_pwe}")
        with writer:
            runner = EpisodeRunner(
                cfg, env, manager, writer, storage, scene_manager,
                diag, wiring.discard_event, scratch_dir,
                init_video_started=video_started,
                default_joint_values=default_joint_values,
                wiring=wiring,
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
            f"  诊断日志: {diag.log_path}\n"
            f"{'='*60}",
            flush=True,
        )


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
