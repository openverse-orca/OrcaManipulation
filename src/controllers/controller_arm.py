from typing_extensions import override
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.adapters.robosuite.controllers.base_controller import Controller

from controllers.abstract_controller import AbstractController
import numpy as np
from orca_gym.log.orca_log import OrcaLog
from scipy.spatial.transform import Rotation as R
orca_logger = OrcaLog.get_instance()

class ControllerArm(AbstractController):
    def __init__(self, env: OrcaGymLocalEnv,
            ctrl_name: list[str],
            init_ctrl: dict[str, float],
            base_body: str,
            controller: Controller):
        '''
        @param: 
            env: 环境
            ctrl_name: 控制器的名称列表
            init_ctrl: 控制器名称和初始值的对应
            base_body: 基座体
            controller: robosuite控制器，这里可以是osc控制器或者Ik控制器
        '''
        self.controller = controller
        self._integral_b = np.zeros(3, dtype=np.float64)
        self._integral_active = False
        self._integral_ki = 0.0
        self._integral_max = 0.0
        self._integral_axis_mask = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._integral_log_every = 0
        self._integral_step = 0

        super().__init__(env, ctrl_name, init_ctrl, base_body)
        self.ee_name = controller.eef_name
        ee_pos_quat_B = self.env.query_site_pos_and_quat_B([self.ee_name], [self.base_link])
        self.initial_ee_pos_B, self.initial_ee_quat_B = ee_pos_quat_B[self.ee_name]["xpos"], ee_pos_quat_B[self.ee_name]["xquat"]
        ee_pos_quat = self.env.query_site_pos_and_quat([self.ee_name])[self.ee_name]
        self.initial_ee_pos, self.initial_ee_quat = ee_pos_quat["xpos"], ee_pos_quat["xquat"]
        self.action = np.zeros(6, dtype=np.float32)
        self.action[0:3] = self.initial_ee_pos
        self.action[3:6] = R.from_quat(self.initial_ee_quat[[1, 2, 3, 0]]).as_rotvec()

    def configure_integral(
        self,
        ki: float,
        max_bias: float,
        axes: str = "z",
        log_every: int = 0,
    ) -> None:
        """配置外环积分增益、限幅与轴掩码。"""
        self._integral_ki = float(max(0.0, ki))
        self._integral_max = float(max(0.0, max_bias))
        axes_l = (axes or "z").lower()
        mask = np.zeros(3, dtype=np.float64)
        if "x" in axes_l:
            mask[0] = 1.0
        if "y" in axes_l:
            mask[1] = 1.0
        if "z" in axes_l:
            mask[2] = 1.0
        if not np.any(mask):
            mask[2] = 1.0
        self._integral_axis_mask = mask
        self._integral_log_every = max(0, int(log_every))

    def enable_integral(self, active: bool) -> None:
        self._integral_active = bool(active)

    def reset_integral(self) -> None:
        self._integral_b[:] = 0.0
        self._integral_step = 0

    def get_integral_bias_b(self) -> np.ndarray:
        return np.asarray(self._integral_b, dtype=np.float64).copy()

    def _query_actual_ee_b(self):
        try:
            ee_b = self.env.query_site_pos_and_quat_B([self.ee_name], [self.base_link])
            return np.asarray(ee_b[self.ee_name]["xpos"], dtype=np.float64)
        except Exception as exc:
            orca_logger.warning(f"query ee pose failed: {exc}")
            return None

    @override
    def reset(self):
        """重新查询当前末端位姿，更新 action 和 B 系参考，同步 IK/OSC 控制器目标。"""
        ee_pos_quat_B = self.env.query_site_pos_and_quat_B([self.ee_name], [self.base_link])
        self.initial_ee_pos_B = ee_pos_quat_B[self.ee_name]["xpos"]
        self.initial_ee_quat_B = ee_pos_quat_B[self.ee_name]["xquat"]

        ee_pos_quat = self.env.query_site_pos_and_quat([self.ee_name])[self.ee_name]
        self.initial_ee_pos = ee_pos_quat["xpos"]
        self.initial_ee_quat = ee_pos_quat["xquat"]

        self.action = np.zeros(6, dtype=np.float32)
        self.action[0:3] = self.initial_ee_pos
        self.action[3:6] = R.from_quat(self.initial_ee_quat[[1, 2, 3, 0]]).as_rotvec()

        # update() must be called before reset_goal() to ensure joint_pos / ee_pos
        # reflect the current physics state (after mj_forward), not stale values.
        self.controller.update()
        self.controller.reset_goal()
        self.reset_integral()

    @override
    def run_controller(self) -> dict[int, float]:
        self.controller.set_goal(self.action)
        ctrl = self.controller.run_controller()
        return {self.ctrl_index[i]: ctrl[i] for i in range(len(self.ctrl_index))}
    
    def update_goal(self, relative_position: np.ndarray, relative_quat: np.ndarray):
        """
        relative_position: 手柄相对其初始位置的位移，B系 (x, y, z)
        relative_quat:     手柄相对其初始朝向的旋转增量，B系 (w, x, y, z)
        """
        base_body_xpos, _, base_body_xquat = self.env.get_body_xpos_xmat_xquat([self.base_link])
        base_body_rot = R.from_quat(base_body_xquat[[1, 2, 3, 0]])

        # 位置和旋转增量均在B系下，整体转换到世界系
        goal_rot_B = R.from_quat(self.initial_ee_quat_B[[1, 2, 3, 0]]) * R.from_quat(relative_quat[[1, 2, 3, 0]])
        goal_pos_B = self.initial_ee_pos_B + relative_position

        goal_rot = base_body_rot * goal_rot_B
        goal_pos = base_body_rot.apply(goal_pos_B) + base_body_xpos

        self.action = np.concatenate([goal_pos, goal_rot.as_rotvec()])

    def update_action_position(self, position: np.array):
        '''
        @description: 更新动作的位置，数据源来于hdf5文件，里面存的是B系下的位置
        @param:
            position: 位置
        '''
        position_b = np.asarray(position, dtype=np.float64).reshape(3)
        corrected_b = position_b
        if self._integral_active and self._integral_ki > 0.0 and self._integral_max > 0.0:
            actual_b = self._query_actual_ee_b()
            if actual_b is not None:
                err_b = (position_b - actual_b) * self._integral_axis_mask
                self._integral_b = np.clip(
                    self._integral_b + self._integral_ki * err_b,
                    -self._integral_max,
                    self._integral_max,
                )
                corrected_b = position_b + self._integral_b
                self._integral_step += 1
                if self._integral_log_every > 0 and self._integral_step % self._integral_log_every == 0:
                    orca_logger.info(
                        f"integral step={self._integral_step} "
                        f"bias={self._integral_b.round(4).tolist()}"
                    )
            else:
                corrected_b = position_b + self._integral_b
        base_body_xpos, _, base_body_xquat = self.env.get_body_xpos_xmat_xquat([self.base_link])
        base_body_rot = R.from_quat(base_body_xquat[[1, 2, 3, 0]])
        self.action[:3] = base_body_rot.apply(corrected_b) + base_body_xpos

    def update_action_axisangle(self, quat: np.array):
        '''
        @description: 更新动作的轴角，数据源来于hdf5文件，里面存的是B系下的四元数(x, y, z, w)
        @param:
            quat: 四元数
        '''
        _, _, base_body_xquat = self.env.get_body_xpos_xmat_xquat([self.base_link])
        base_body_rot = R.from_quat(base_body_xquat[[1, 2, 3, 0]])
        ee_rot = base_body_rot * R.from_quat(quat)
        axisangle = ee_rot.as_rotvec()
        self.action[3:6] = axisangle
        
    @override
    def init_ctrl_index(self):
        joint_names = self.controller.joint_index
        self.controller.qpos_index, self.controller.qvel_index, _ = self.env.query_joint_offsets(joint_names)
        self.ctrl_index = [self.env.model.actuator_name2id(name) for name in self.ctrl_name]
        return self.ctrl_index