"""Euler 刚体数采环境：把遥操 ctrl 交给 MuJoCoFlow，不加布。"""

from typing import Callable

import numpy as np

from orca_gym.core.euler.sim_config import SimBackend
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import OrcaLog

orca_logger = OrcaLog.get_instance()


class DataCollectionEulerEnv(OrcaGymEulerEnv):
    """Manipulation 刚体通道环境。

    继承 ``OrcaGymEulerEnv``。``step(action)`` 把 action 原样交给
    ``do_simulation``（内部是 ``step_with_coupling`` → 无柔体时 MuJoCoFlow）。
    不改 CPU 的 ``DataCollectionEnv``，不加载 ESDF，不挂 CoupledGpuSim。
    """

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        default_joint_values: dict[str, float],
        obs_callback: Callable,
        **kwargs,
    ):
        self.obs_callback = obs_callback
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )
        self._require_rigid_gpu_backend()
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        self._set_obs_space()
        self._set_action_space()
        self.default_joint_values = None
        self.set_default_joint_values(default_joint_values)

    def _require_rigid_gpu_backend(self) -> None:
        """启动自检：必须是 Euler GPU 后端，且还没有 CoupledGpuSim。

        做什么：看 ``sim_config.backend`` 和 ``has_euler()``。
        为什么：本 Env 只证明手柄/回放 → ctrl → MuJoCoFlow。带布关卡或 CPU 后端都应立刻失败。
        """
        if self.sim_config.backend != SimBackend.EULER:
            raise RuntimeError(
                "刚体通道 MVP 需要 Euler GPU 后端。"
                "请通过 Manager 的 sim_device 传入 cuda* 或 hip*，不要用 cpu。"
            )
        if self.has_euler():
            raise RuntimeError(
                "刚体通道 MVP 不允许 CoupledGpuSim / 柔体相。"
                "请换无布关卡，或使用后续折布入口。"
            )

    def init_env(self) -> None:
        """场景发布后重新加载仿真。

        做什么：再次 ``initialize_simulation``，再 ``reset``。
        为什么：``SceneManager`` 发完 Studio 场景后会回调；与 CPU 数采环境同一职责。
        """
        self.initialize_simulation()
        self.reset()

    def step(self, action):
        """把遥操 ctrl 推进一宏步。

        做什么：``do_simulation(action, frame_skip)``，再取观测。
        为什么：不能沿用冒烟脚本那种丢掉 action 的 ``step``。
        """
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        if isinstance(obs, dict):
            obs = obs.copy()
        return obs, 0.0, False, False, {}

    def reset_model(self):
        """复位到默认关节，供 Mixin ``reset()`` 调用。"""
        orca_logger.info("reset model")
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.set_default_joint_values(self.default_joint_values)
        self.mj_forward()
        obs = self._get_obs()
        if isinstance(obs, dict):
            obs = obs.copy()
        return obs, {}

    def set_default_joint_values(self, default_joint_values: dict[str, float]) -> None:
        """按短名写入默认关节角。

        做什么：短名经 ``joint()`` 加上 agent 前缀，再交给 ``set_joint_qpos``。
        为什么：conf 里是短名，Studio 模型里是带前缀的全名。
        """
        self.default_joint_values = default_joint_values or {}
        prefixed = {
            self.joint(joint_name): np.float32(value)
            for joint_name, value in self.default_joint_values.items()
        }
        self.set_joint_qpos(prefixed)

    def set_joint_qpos(self, qpos) -> None:
        """写入关节位置。dict 转成整段 qpos 后再调父类。

        做什么：CPU 数采用 ``{关节名: 值}``；Euler 父类只吃整段 ndarray。
        本方法在子类里把 dict 填进当前 qpos 副本，仍走公共 ``set_joint_qpos``。
        """
        if isinstance(qpos, dict):
            full = np.array(self.data.qpos, dtype=np.float64, copy=True)
            for joint_name, value in qpos.items():
                adr = self.jnt_qposadr(joint_name)
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
                full[adr : adr + arr.size] = arr
            super().set_joint_qpos(full)
            return
        super().set_joint_qpos(np.asarray(qpos, dtype=np.float64))

    def _set_obs_space(self) -> None:
        """按当前观测生成 Gymnasium 观测空间。"""
        obs = self._get_obs()
        if isinstance(obs, dict):
            obs = obs.copy()
        self.observation_space = self.generate_observation_space(obs)

    def _set_action_space(self) -> None:
        """按执行器个数生成动作空间。"""
        low_bounds = -np.ones(self.nu, dtype=np.float32)
        high_bounds = np.ones(self.nu, dtype=np.float32)
        bound = np.array(
            [[low_bound, high_bound] for low_bound, high_bound in zip(low_bounds, high_bounds)]
        )
        self.action_space = self.generate_action_space(bound)

    def _get_obs(self):
        """调用外部 obs_callback，给 Manager / DataStorage 用。"""
        return self.obs_callback(self)
