from typing import override
from task.abstract_task import AbstractTask
from scene.scene_manager import SceneManager
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.log import OrcaLog
import numpy as np
import mujoco

orca_logger = OrcaLog.get_instance()

class PickPlaceTask(AbstractTask):
    '''抓取目标物体，放置到目标位置'''
    def __init__(self, env: OrcaGymLocalEnv):
        super().__init__(env=env)
        self.target_actor = None
        self.target_actor_info = None
        self.goal_name = None
        self.goal_site = None
        
    def get_goal_site_env_name(self):
        return self.goal_name + "_" + self.goal_site

    @override
    def is_success(self):
        target_joint_qpos = self.env.query_joint_qpos([self.target_actor_info["joint_name"]])[self.target_actor_info["joint_name"]]
        target_pos = target_joint_qpos[:3]
        
        goal_site_info = self.env.query_site_pos_and_mat([self.get_goal_site_env_name()])[self.get_goal_site_env_name()]
        goal_site_xpos = goal_site_info['xpos']
        goal_site_xmat = goal_site_info['xmat'].reshape(3, 3)
        goal_site_size = self.env.query_site_size([self.get_goal_site_env_name()])[self.get_goal_site_env_name()]
        
        site_type = self.env.gym.query_all_sites()[self.get_goal_site_env_name()]["Type"]
        
        if site_type == mujoco.mjtGeom.mjGEOM_BOX:
            half_size = goal_site_size[:3]
            half_world = np.abs(goal_site_xmat) @ half_size
            bbox_min = goal_site_xpos - half_world
            bbox_max = goal_site_xpos + half_world
        elif site_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = goal_site_size[0]
            bbox_min = goal_site_xpos - radius
            bbox_max = goal_site_xpos + radius
        elif site_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radius = goal_site_size[0]
            half_height = goal_site_size[1]
            z_axis = goal_site_xmat[:, 2]
            half_world_xy = np.ones(3) * radius
            half_world_z = np.abs(z_axis) * half_height
            half_world = np.maximum(half_world_xy, half_world_z)
            bbox_min = goal_site_xpos - half_world
            bbox_max = goal_site_xpos + half_world
        elif site_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = goal_site_size[0]
            half_height = goal_site_size[1]
            z_axis = goal_site_xmat[:, 2]
            half_world_xy = np.ones(3) * radius
            half_world_z = np.abs(z_axis) * half_height + radius
            half_world = np.maximum(half_world_xy, half_world_z)
            bbox_min = goal_site_xpos - half_world
            bbox_max = goal_site_xpos + half_world
        elif site_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            half_size = goal_site_size[:3]
            half_world = np.abs(goal_site_xmat) @ half_size
            bbox_min = goal_site_xpos - half_world
            bbox_max = goal_site_xpos + half_world
        else:
            orca_logger.warning(f"Unsupported site type: {site_type}, using sphere approximation")
            radius = np.max(goal_site_size[:3])
            bbox_min = goal_site_xpos - radius
            bbox_max = goal_site_xpos + radius
        
        in_bbox = np.all(target_pos >= bbox_min) and np.all(target_pos <= bbox_max)
        return in_bbox

    @override
    def get_task_description(self):
        return f"Pick {self.target_actor} and place it into {self.goal_site}"

    @override
    def _get_task(self, scene_manager: SceneManager, task_info: dict = None)-> bool:
        if task_info is not None:
            self.target_actor = task_info.get("target_actor")
            self.target_actor_info = task_info.get("target_actor_info")
            self.goal_name = task_info.get("goal_name")
            self.goal_site = task_info.get("goal_site")
            return not self.is_success()
        
        task_config = scene_manager.get_task_config()
        self.check_task_config(task_config)

        scene_info = scene_manager.get_scene_info()
        lens = len(scene_info)
        target_index =  np.random.randint(0, lens - 1) if lens > 1 else 0
        self.target_actor = list(scene_info.keys())[target_index]
        self.target_actor_info = scene_info[self.target_actor]

        self.goal_name = task_config.get("goal").get("name")
        self.goal_site = task_config.get("goal").get("site")

        return not self.is_success()

    @override
    def get_task_info(self) -> dict:
        return {
            "target_actor": self.target_actor,
            "target_actor_info": self.target_actor_info,
            "goal_name": self.goal_name,
            "goal_site": self.goal_site
        }

    def check_task_config(self, task_config: dict):
        help_info = '''in your task config file:
                            task:
                              type: "pick_and_place"
                              goal:
                                site: "goal_site"
                            '''
        if len(task_config) == 0:
            orca_logger.error("Task config is empty, please check your task config file")
            orca_logger.info(help_info)
            raise ValueError("Task config is empty")
        if task_config.get("type") != "pick_and_place":
            orca_logger.error("Task type is not pick_and_place, please check your task config file")
            orca_logger.info(help_info)
            raise ValueError("Task type is not pick_and_place")
        if task_config.get("goal") is None:
            orca_logger.error("Task goal is empty, please check your task config file")
            orca_logger.info(help_info)
            raise ValueError("Task goal is empty")
        if task_config.get("goal").get("site") is None:
            orca_logger.error("Goal site is empty, please check your task config file")
            orca_logger.info(help_info)
            raise ValueError("Goal site is empty")