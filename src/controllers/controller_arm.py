from typing import override
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

        super().__init__(env, ctrl_name, init_ctrl, base_body)
        self.ee_name = controller.eef_name
        ee_pos_quat_B = self.env.query_site_pos_and_quat_B([self.ee_name], [self.base_link])
        self.initial_ee_pos_B, self.initial_ee_quat_B = ee_pos_quat_B[self.ee_name]["xpos"], ee_pos_quat_B[self.ee_name]["xquat"]
        ee_pos_quat = self.env.query_site_pos_and_quat([self.ee_name])[self.ee_name]
        self.initial_ee_pos, self.initial_ee_quat = ee_pos_quat["xpos"], ee_pos_quat["xquat"]
        
        self.action = np.zeros(6, dtype=np.float32)
        self.action[0:3] = self.initial_ee_pos
        self.action[3:6] = R.from_quat(self.initial_ee_quat[[1, 2, 3, 0]]).as_rotvec()
        
    @override
    def run_controller(self)-> dict[int, float]:
        self.controller.set_goal(self.action)
        ctrl = self.controller.run_controller() 
        return {self.ctrl_index[i]: ctrl[i] for i in range(len(self.ctrl_index))}
    
    def update_goal(self, relative_position: np.array, relative_quat: np.array):
        base_body_xpos, _, base_body_xquat = self.env.get_body_xpos_xmat_xquat([self.base_link])
        base_body_rot = R.from_quat(base_body_xquat[[1, 2, 3, 0]])

        goal_rot_B = R.from_quat(self.initial_ee_quat_B[[1, 2, 3, 0]]) * R.from_quat(relative_quat[[1, 2, 3, 0]])
        goal_pos_B = self.initial_ee_pos_B + relative_position

        goal_rot = base_body_rot * goal_rot_B
        goal_axisangle = goal_rot.as_rotvec()
        goal_pos = base_body_rot.apply(goal_pos_B) + base_body_xpos
        
        self.action = np.concatenate([goal_pos, goal_axisangle])

    def update_action_position(self, position: np.array):
        '''
        @description: 更新动作的位置，数据源来于hdf5文件，里面存的是B系下的位置
        @param:
            position: 位置
        '''
        base_body_xpos, _, base_body_xquat = self.env.get_body_xpos_xmat_xquat([self.base_link])
        base_body_rot = R.from_quat(base_body_xquat[[1, 2, 3, 0]])
        position = base_body_rot.apply(position) + base_body_xpos
        self.action[:3] = position

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