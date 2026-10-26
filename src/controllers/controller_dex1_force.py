import enum
import time
from typing import override
from orca_gym.environment import OrcaGymLocalEnv
from controllers.abstract_controller import AbstractController
import numpy as np
from orca_gym.log.orca_log import OrcaLog

orca_logger = OrcaLog.get_instance()


class ControllerDex1Force(AbstractController):
    """DEX1-1 夹爪控制器: 单 motor 力执行器 + 官方 SDK 位置阻抗律。

    真机由 1 个 M4010 电机经齿轮齿条耦合驱动两指, 官方 SDK 发送位置阻抗
    指令 (mode=1): tau = kp * (q_target - q) - kd * dq。
    本控制器在 PICO 模式把扳机映射为目标开度, 在 DATA 模式接收归一化
    开度, 统一经阻抗律换算为电机力。

    按键语义:
        扳机 (trigger)  深度 → 目标开度 (浅=张开, 深=闭合)
        主键  (X / A)   按下 → 立即全开
        副键  (Y / B)   按下 → 目标回中位
        松开所有键      保持当前目标 (抓取保持)
    """

    class ControllerType(enum.Enum):
        PICO = 0
        DATA = 1

    def __init__(
        self,
        env: OrcaGymLocalEnv,
        ctrl_name: list[str],
        init_ctrl: dict[str, float],
        gripper_config: dict,
        base_body: str,
        controller_type: ControllerType = ControllerType.PICO,
    ):
        super().__init__(env, ctrl_name, init_ctrl, base_body)
        self.actuator_range = gripper_config["actuator_ranges"]
        self.q_open = float(gripper_config.get("q_open", -0.020))
        self.q_close = float(gripper_config.get("q_close", 0.0241))
        self.kp = float(gripper_config.get("kp", 400.0))
        self.kd = float(gripper_config.get("kd", 15.0))
        self.max_force = float(
            gripper_config.get("max_force", self.actuator_range[0][1])
        )
        self.controller_type = controller_type
        self.trigger_value = 0.0
        self.primary_button = False
        self.secondary_button = False
        self.q_target = self.q_open
        self._joint_adr = None  # (qposadr, dofadr) 惰性解析
        self._print_t = 0.0  # 终端打印节流计时

    def _resolve_joint_adr(self) -> tuple[int, int] | None:
        """从执行器传动反查 joint1 的 qpos/dof 地址 (agent 前缀自适应)。"""
        if self._joint_adr is not None:
            return self._joint_adr
        import mujoco

        gym = getattr(self.env, "gym", None) or getattr(
            getattr(self.env, "unwrapped", self.env), "gym", None
        )
        mj = getattr(gym, "_mjModel", None)
        if mj is None or not self.ctrl_index:
            return None
        aid = int(self.ctrl_index[0])
        if int(mj.actuator_trntype[aid]) != mujoco.mjtTrn.mjTRN_JOINT:
            return None
        jid = int(mj.actuator_trnid[aid, 0])
        self._joint_adr = (int(mj.jnt_qposadr[jid]), int(mj.jnt_dofadr[jid]))
        return self._joint_adr

    def _read_joint_state(self) -> tuple[float, float]:
        adr = self._resolve_joint_adr()
        if adr is None:
            return self.q_target, 0.0
        gym = getattr(self.env, "gym", None) or getattr(
            getattr(self.env, "unwrapped", self.env), "gym", None
        )
        md = gym._mjData
        return float(md.qpos[adr[0]]), float(md.qvel[adr[1]])

    def _impedance_force(self, q_target: float) -> float:
        """官方阻抗律: tau = kp*(q_d - q) - kd*dq, 饱和到 ±max_force。"""
        q, dq = self._read_joint_state()
        return float(
            np.clip(self.kp * (q_target - q) - self.kd * dq,
                    -self.max_force, self.max_force)
        )

    @override
    def run_controller(self) -> dict[int, float]:
        if self.controller_type == self.ControllerType.PICO:
            if self.primary_button:
                self.q_target = self.q_open
            elif self.secondary_button:
                self.q_target = 0.5 * (self.q_open + self.q_close)
            elif self.trigger_value > 0.05:
                # 指数映射 (与 2f85_reverse 同手感): 扳机越深闭合越多
                k = np.e
                depth = (np.exp(k * self.trigger_value) - 1) / (np.exp(k) - 1)
                self.q_target = self.q_open + (self.q_close - self.q_open) * depth
            # 无输入: 保持当前目标 (抓取保持)

        ctrl = {i: self._impedance_force(self.q_target) for i in self.ctrl_index}

        # 终端节流打印夹爪输出力 (每 0.5s 一行): tau=下发给电机的力(N),
        # q=实际开度, target=目标开度; 夹住物体时 q 顶在接触位置到不了
        # target, tau 恒定非零即持续夹持力, 饱和值 = max_force。
        now = time.time()
        if now - self._print_t > 0.5:
            self._print_t = now
            side = "L" if "left" in str(self.ctrl_name[0]).lower() else "R"
            q, dq = self._read_joint_state()
            tau = ctrl[self.ctrl_index[0]]
            print(
                f"[DEX1-{side}] tau={tau:+7.2f}N  q={q:+.4f}m  "
                f"target={self.q_target:+.4f}m  dq={dq:+.3f}m/s"
            )

        return ctrl

    @override
    def reset(self):
        """每个 episode 开始时回到全开目标。"""
        self.q_target = self.q_open

    def update_trigger_value(self, trigger_value: float):
        self.trigger_value = trigger_value

    def update_primary_button(self, primary_button: bool):
        self.primary_button = primary_button

    def update_secondary_button(self, secondary_button: bool):
        self.secondary_button = secondary_button

    def update_ctrl(self, ctrl: np.ndarray):
        """DATA 模式: 接收归一化开度 [0,1] (0=张开, 1=闭合)。"""
        norm = float(np.clip(np.asarray(ctrl, dtype=np.float64).reshape(-1)[0], 0.0, 1.0))
        self.q_target = self.q_open + (self.q_close - self.q_open) * norm

    def update_target_qpos(self, ctrl: np.ndarray):
        """DATA 模式: 接收数据集记录的 DEX1 关节 qpos (joint1, joint2)。

        取 joint1 (主动指) 作为目标开度, 裁剪到行程内。回放时比直接回放
        电机力矩更稳定: 力矩是阻抗律输出, 状态偏差会累积, 而位置目标
        经同一阻抗律闭环后能自然复现夹持过程。
        """
        q = float(np.asarray(ctrl, dtype=np.float64).reshape(-1)[0])
        lo, hi = min(self.q_open, self.q_close), max(self.q_open, self.q_close)
        self.q_target = float(np.clip(q, lo, hi))


def add_dex1_force_pico_controller(
    data_collection_manager,
    env: OrcaGymLocalEnv,
    gripper_config: dict,
    base_body: str,
    device,
    keys: list,
):
    """创建 DEX1-1 夹爪 PICO 控制器并绑定按键 (对应 2f85_reverse 的工厂)。

    keys: [主键(张开), 副键(中位), 扳机(闭合)]
    """
    from orca_gym.devices.pico_joytsick import PicoJoystickKey

    ctrl_name = [
        env.actuator(actuator_name)
        for actuator_name in gripper_config["actuator_names"]
    ]
    init_ctrl = {
        name: init_val
        for name, init_val in zip(ctrl_name, gripper_config["init_ctrl"])
    }
    controller = ControllerDex1Force(
        env, ctrl_name, init_ctrl, gripper_config, base_body,
        ControllerDex1Force.ControllerType.PICO,
    )
    for key in keys:
        if key in [PicoJoystickKey.X, PicoJoystickKey.A]:
            device.bind_primary_button_event(key, controller.update_primary_button)
        elif key in [PicoJoystickKey.Y, PicoJoystickKey.B]:
            device.bind_secondary_button_event(key, controller.update_secondary_button)
        elif key in [PicoJoystickKey.L_TRIGGER, PicoJoystickKey.R_TRIGGER]:
            device.bind_trigger_event(key, controller.update_trigger_value)
        else:
            raise ValueError(f"Invalid key: {key}")
    data_collection_manager.add_controller(controller)
    return controller


def add_dex1_force_data_controller(
    data_collection_manager,
    env: OrcaGymLocalEnv,
    gripper_config: dict,
    base_body: str,
    device,
    left_gripper: bool,
):
    """创建 DEX1-1 夹爪 DATA 控制器, 从数据集 /action/effector/position 回放开度。

    数据布局: 左夹爪 2 关节 qpos 在 [0,2), 右夹爪在 [2,4)。
    """
    ctrl_name = [
        env.actuator(actuator_name)
        for actuator_name in gripper_config["actuator_names"]
    ]
    init_ctrl = {
        name: init_val
        for init_val, name in zip(gripper_config["init_ctrl"], ctrl_name)
    }
    controller = ControllerDex1Force(
        env, ctrl_name, init_ctrl, gripper_config, base_body,
        ControllerDex1Force.ControllerType.DATA,
    )
    if left_gripper:
        device.bind_dataset_event(
            "/action/effector/position", (0, 2), controller.update_target_qpos
        )
    else:
        device.bind_dataset_event(
            "/action/effector/position", (2, 4), controller.update_target_qpos
        )
    data_collection_manager.add_controller(controller)
    return controller
