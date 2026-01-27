from typing import Callable
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log.orca_log import OrcaLog
import numpy as np

from envs.dataCollection.conveyor_board_animator import (
    ConveyorBoardAnimator,
    ConveyorBoardAnimatorConfig,
)

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
        conveyor: dict | None = None,
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

        # Conveyor board animator (optional, swallow all failures)
        try:
            cfg = ConveyorBoardAnimatorConfig(**(conveyor or {}))
        except Exception:
            cfg = ConveyorBoardAnimatorConfig(enable=False)
        self._conveyor_animator = ConveyorBoardAnimator(self, cfg)
        self._conveyor_ready = False
        try:
            self._conveyor_animator.refresh()
        except Exception:
            pass

    def step(self, action):
        self.ctrl = action
        try:
            self._conveyor_animator.step(self.data.time)
        except Exception:
            pass
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
        # model may reload; refresh joint presence
        try:
            self._conveyor_animator.refresh()
        except Exception:
            pass
        orca_logger.info(f"gym address: self.gym: {id(self.gym)}")
        
    def bind_conveyor_device(self, device):
        """
        可选：绑定 Pico 组合键启停。吞异常，不影响主流程。
        """
        try:
            self._conveyor_animator.bind_device(device)
        except Exception:
            pass

    def set_conveyor_running(self, running: bool):
        """
        由上层（TaskStatus）控制是否启动传送带：
        - 只有在已完成摆放/settle（_conveyor_ready=True）时，running=True 才会真正 start()
        - running=False 会 stop()
        吞异常，不影响主流程。
        """
        try:
            if not running:
                self._conveyor_animator.stop(self.data.time)
                return
            if self._conveyor_ready:
                self._conveyor_animator.start(self.data.time)
        except Exception:
            pass

    def get_conveyor_start_mode(self) -> str:
        try:
            return str(getattr(self._conveyor_animator.cfg, "start_mode", "task_status"))
        except Exception:
            return "task_status"

    def _after_scene_actor_placement(self):
        """
        被 SceneManager 在“物品摆放”后调用：
        - 板子回起点
        - settle_steps 次物理步进（板子不动）
        - 启动传送带
        吞异常，不影响主流程。
        """
        try:
            self._conveyor_animator.reset_episode()
        except Exception:
            return

        try:
            settle_steps = int(getattr(self._conveyor_animator.cfg, "settle_steps", 0))
        except Exception:
            settle_steps = 0

        if settle_steps > 0:
            try:
                zero_ctrl = np.zeros(self.nu, dtype=np.float32)
                # board stays still because belt_running=False during settle
                for _ in range(settle_steps):
                    self.do_simulation(zero_ctrl, 1)
            except Exception:
                pass

        # Arm conveyor for later start (do NOT auto-start here)
        try:
            self._conveyor_ready = bool(self._conveyor_animator.enabled)
        except Exception:
            self._conveyor_ready = False

        # Optional: auto start after placement+settle
        try:
            if self._conveyor_ready and self.get_conveyor_start_mode() == "auto":
                self._conveyor_animator.start(self.data.time)
        except Exception:
            pass

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


