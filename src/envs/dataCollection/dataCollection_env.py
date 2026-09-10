from typing import Callable, Optional

import numpy as np

from orca_gym.environment.orca_gym_env import OrcaGymBaseEnv
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log.orca_log import OrcaLog

orca_logger = OrcaLog.get_instance()


class DataCollectionEnv(OrcaGymLocalEnv):
    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        default_joint_values: dict[str, float],
        obs_callback: Callable[[OrcaGymLocalEnv], dict],
        **kwargs
    ):
        self.obs_callback = obs_callback
        super().__init__(
            frame_skip = frame_skip,
            orcagym_addr = orcagym_addr,
            agent_names = agent_names,
            time_step = time_step,
            **kwargs)

        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        self._set_obs_space()
        self._set_action_space()

        self.default_joint_values = None
        self.set_default_joint_values(default_joint_values)
        self._skip_studio_render_on_reset = False

    def clear_studio_override_ctrls(self) -> None:
        """清除 Studio 侧缓存的 override_ctrls，防止覆盖本地遥操输出。"""
        if hasattr(self.gym, "clear_override_ctrls"):
            self.gym.clear_override_ctrls()

    def set_protected_override_ctrl_ids(self, actuator_ids: set[int] | list[int]) -> None:
        """注册不受 Studio override 覆盖的执行器（如夹爪 pctrl）。"""
        if hasattr(self.gym, "set_protected_override_ctrl_ids"):
            self.gym.set_protected_override_ctrl_ids(actuator_ids)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # 与 OrcaGymBaseEnv.reset 相同，但布料联调时跳过 render()，避免加载 override_ctrls
        super(OrcaGymBaseEnv, self).reset(seed=seed)
        if seed is not None:
            self.set_seed_value(seed)
        self.reset_simulation()
        obs, info = self.reset_model()
        if self._skip_studio_render_on_reset:
            self.clear_studio_override_ctrls()
        else:
            self.render_count = 1.1
            self.render()
        return obs, info

    def step(self, action):
        self.ctrl = action
        self.do_simulation(self.ctrl, self.frame_skip)
        obs = self._get_obs().copy()
        terminated = False
        truncated = False
        reward = 0.0    
        return obs, reward, terminated, truncated, {}

    def reset_model(self):
        orca_logger.info(f"reset model")
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        self.set_default_joint_values(self.default_joint_values)    
        self.mj_forward()
        self._render_time_step = 0
        obs = self._get_obs().copy()
        return obs, {}

    def init_env(self):
        self.model, self.data = self.initialize_simulation()
        self.reset()
        
    def set_default_joint_values(self, default_joint_values: dict[str, float]):
        self.default_joint_values = default_joint_values
        self._default_joint_qpos = {self.joint(joint_name): np.float32(value) for joint_name, value in default_joint_values.items()}
        self.set_joint_qpos(self._default_joint_qpos)
        
    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())
    

    def _set_action_space(self):
        low_bounds = -np.ones(self.nu, dtype=np.float32)
        high_bounds = np.ones(self.nu, dtype=np.float32)
        bound = np.array([[low_bound, high_bound] for low_bound, high_bound in zip(low_bounds, high_bounds)])
        self.action_space = self.generate_action_space(bound)

    def _get_obs(self):
        return self.obs_callback(self)


