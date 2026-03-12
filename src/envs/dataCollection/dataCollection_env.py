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

        # Conveyor board animator(s) (optional, swallow all failures)
        self._conveyor_animators: list[ConveyorBoardAnimator] = []
        self._conveyor_ready = False
        conveyor_cfg = conveyor or {}
        try:
            # Support multi-belt in ONE env:
            # - conveyor.board_joint_names: ["Geom_Track1_Joint", ...]
            # - OR conveyor.board_joint_name: ["..."] (tolerate list)
            names = None
            if isinstance(conveyor_cfg, dict):
                if isinstance(conveyor_cfg.get("board_joint_names"), list):
                    names = conveyor_cfg.get("board_joint_names")
                elif isinstance(conveyor_cfg.get("board_joint_name"), list):
                    names = conveyor_cfg.get("board_joint_name")
            if names:
                base = dict(conveyor_cfg)
                base.pop("board_joint_names", None)
                base.pop("board_joint_name", None)
                for n in names:
                    cfg = ConveyorBoardAnimatorConfig(**{**base, "board_joint_name": str(n)})
                    self._conveyor_animators.append(ConveyorBoardAnimator(self, cfg))
            else:
                cfg = ConveyorBoardAnimatorConfig(**conveyor_cfg)
                self._conveyor_animators.append(ConveyorBoardAnimator(self, cfg))
        except Exception:
            self._conveyor_animators = [ConveyorBoardAnimator(self, ConveyorBoardAnimatorConfig(enable=False))]

        for a in self._conveyor_animators:
            try:
                a.refresh()
            except Exception:
                pass

    def step(self, action):
        self.ctrl = action
        try:
            for a in self._conveyor_animators:
                a.step(self.data.time)
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
        for a in self._conveyor_animators:
            try:
                a.refresh()
            except Exception:
                pass
        orca_logger.info(f"gym address: self.gym: {id(self.gym)}")
        
    def bind_conveyor_device(self, device):
        """
        可选：绑定 Pico 组合键启停。吞异常，不影响主流程。
        """
        try:
            for a in self._conveyor_animators:
                a.bind_device(device)
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
                for a in self._conveyor_animators:
                    a.stop()
                return
            if self._conveyor_ready:
                for a in self._conveyor_animators:
                    a.start(self.data.time)
        except Exception:
            pass

    def get_conveyor_start_mode(self) -> str:
        try:
            if self._conveyor_animators:
                return str(getattr(self._conveyor_animators[0].cfg, "start_mode", "task_status"))
            return "task_status"
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
            for a in self._conveyor_animators:
                a.reset_episode()
        except Exception:
            return

        try:
            cfg0 = self._conveyor_animators[0].cfg if self._conveyor_animators else None
            settle_steps = int(getattr(cfg0, "settle_steps", 0)) if cfg0 is not None else 0
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
            self._conveyor_ready = any(bool(a.enabled) for a in self._conveyor_animators)
        except Exception:
            self._conveyor_ready = False

        # Optional: auto start after placement+settle
        try:
            if self._conveyor_ready and self.get_conveyor_start_mode() == "auto":
                for a in self._conveyor_animators:
                    a.start(self.data.time)
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


