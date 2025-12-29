from typing import Callable
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log.orca_log import OrcaLog
import numpy as np

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
        obs = self._get_obs().copy()
        return obs, {}

    def init_env(self):
        orca_logger.info(f"gym address: self.gym: {id(self.gym)}")
        self.model, self.data = self.initialize_simulation()
        self.reset()
        orca_logger.info(f"gym address: self.gym: {id(self.gym)}")
        
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


