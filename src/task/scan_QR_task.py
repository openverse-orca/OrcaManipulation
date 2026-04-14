from typing import override
from task.abstract_task import AbstractTask
from scene.scene_manager import SceneManager
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.log import OrcaLog
import numpy as np

orca_logger = OrcaLog.get_instance()

class ScanQRTask(AbstractTask):
    '''扫描QR码任务：将扫码枪对准QR码进行扫描'''
    def __init__(self, env: OrcaGymLocalEnv):
        super().__init__(env=env)
        self.target_actor = None         # 目标QR码物体
        self.target_actor_info = None    # 目标QR码信息
        self.goal_name = None            # 扫码枪（goal）名称
        self.goal_body = None            # 扫码枪
        
    def get_goal_body_env_name(self):
        return self.goal_name + "_" + self.goal_body

    @override
    def is_success(self):
        '''判断扫码枪是否面向QR码（扫码枪射线指向QR码平面）'''
        pos, _, quat = self.env.get_body_xpos_xmat_xquat([self.target_actor_info['body_name'], self.get_goal_body_env_name()])
        target_pos, goal_pos = pos[:3], pos[3:6]
        target_quat, goal_quat = quat[:4], quat[4:8]    

        is_success = self._is_facing_(target_pos, target_quat, goal_pos, goal_quat, 75, 0.3)
        orca_logger.info(f'task is success: {is_success}')
        return is_success


    def _normalize_quaternion_(self, quat: np.ndarray) -> np.ndarray:
        """归一化四元数"""
        w, x, y, z = quat
        norm = np.sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2)
        return np.array([w / norm, x / norm, y / norm, z / norm])

    def _quaternion_forward_vector_(self, quat: np.ndarray, initial_forward: np.ndarray = np.array([0, 1, 0])) -> np.ndarray:
        """计算旋转后的前向向量，使用四元数旋转公式 v' = q * v * q^-1"""
        w, x, y, z = quat
        q = np.array([x, y, z])

        # 公式: v' = v + 2*cross(q, cross(q, v) + w*v)
        t = np.cross(q, initial_forward)
        forward = initial_forward + 2 * (w * t + np.cross(q, t))

        norm = np.linalg.norm(forward)
        return forward / norm if norm > 1e-10 else forward

    def _is_facing_(self, posA, quatA, posB, quatB, tolerance_deg=60, tolerance_distance=0.2) -> bool:
        """
        检查物体A是否面向物体B
        :param posA: 物体A的位置
        :param quatA: 物体A的四元数
        :param posB: 物体B的位置
        :param quatB: 物体B的四元数
        :param tolerance_deg: 面向的角度偏差
        :param tolerance_distance: 面向的距离偏差
        :return: 是否面向
        """
        cos_tolerance = np.cos(np.radians(tolerance_deg ))
        cos_opposite = np.cos(np.radians(180 - tolerance_deg))

        posA = np.array(posA)
        posB = np.array(posB)

        # 归一化四元数
        quatA_norm = self._normalize_quaternion_(quatA)
        quatB_norm = self._normalize_quaternion_(quatB)

        forwardA = self._quaternion_forward_vector_(quatA_norm, initial_forward=np.array([0, -1, 0]))
        forwardB = self._quaternion_forward_vector_(quatB_norm, initial_forward=np.array([0, 1, 0]))

        # 计算方向向量
        dir_A_to_B = posB - posA

        distance = np.linalg.norm(dir_A_to_B)
        if distance < 1e-10 or distance > tolerance_distance:
            orca_logger.warning(f"Distance too small or too large: {distance}")
            return False  # 避免除以零

        dir_A_to_B_normalized = dir_A_to_B / distance

        dir_B_to_A_normalized = -dir_A_to_B_normalized

        # 计算关键点积
        dot_forward = np.dot(forwardA, forwardB)
        dot_A_to_B = np.dot(forwardA, dir_A_to_B_normalized)
        dot_B_to_A = np.dot(forwardB, dir_B_to_A_normalized)

        return (
                dot_forward <= cos_opposite and
                dot_A_to_B >= cos_tolerance and
                dot_B_to_A >= cos_tolerance
        )

    @override
    def get_task_description(self):
        return f"Pick up the {self.target_actor} and scan it with the {self.goal_name}."

    @override
    def _get_task(self, scene_manager: SceneManager, task_info: dict = None) -> bool:
        if task_info is not None:
            self.target_actor = task_info.get("target_actor")
            self.target_actor_info = task_info.get("target_actor_info")
            self.goal_name = task_info.get("goal_name")
            self.goal_body = task_info.get("goal_body")
            return not self.is_success()
        
        task_config = scene_manager.get_task_config()
        self.check_task_config(task_config)

        scene_info = scene_manager.get_scene_info()
        lens = len(scene_info)
        target_index =  np.random.randint(0, lens - 1) if lens > 1 else 0
        self.target_actor = list(scene_info.keys())[target_index]
        self.target_actor_info = scene_info[self.target_actor]

        self.goal_name = task_config.get("goal").get("name")
        self.goal_body = task_config.get("goal").get("body")

        return not self.is_success()

    @override
    def get_task_info(self) -> dict:
        return {
            "target_actor": self.target_actor,
            "target_actor_info": self.target_actor_info,
            "goal_name": self.goal_name,
            "goal_body": self.goal_body,
        }

    def check_task_config(self, task_config: dict):
        help_info = '''in your task config file:
                            task:
                              type: "scan_qr"
                              goal:
                                name: "scanner_actor"
                                site: "scan_site"
                            '''
        if len(task_config) == 0:
            orca_logger.error("Task config is empty, please check your task config file")
            orca_logger.info(help_info)
            raise ValueError("Task config is empty")
        if task_config.get("type") != "scan_qr":
            orca_logger.error("Task type is not scan_qr, please check your task config file")
            orca_logger.info(help_info)
            raise ValueError("Task type is not scan_qr")
        if task_config.get("goal") is None:
            orca_logger.error("Task goal is empty, please check your task config file")
            orca_logger.info(help_info)
            raise ValueError("Task goal is empty")
        if task_config.get("goal").get("name") is None:
            orca_logger.error("Goal name (scanner) is empty, please check your task config file")
            orca_logger.info(help_info)
            raise ValueError("Goal name is empty")
        if task_config.get("goal").get("site") is None:
            orca_logger.error("Goal site (scanner site) is empty, please check your task config file")
            orca_logger.info(help_info)
            raise ValueError("Goal site is empty")
        if task_config.get("goal").get("body") is None:
            orca_logger.error("Goal body (scanner body) is empty, please check your task config file")
            orca_logger.info(help_info)
            raise ValueError("Goal body is empty")
