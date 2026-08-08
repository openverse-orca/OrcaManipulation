"""为进程内 OpenPI 策略增加 RTC 采样能力的适配器。"""

# 此处有意访问被封装策略的冻结模型、随机数状态和输入输出变换。
# ruff: noqa: SLF001

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.policies import policy as _policy
from openpi.shared import nnx_utils
from openpi_rtc.config import RTCConfig
from openpi_rtc.guidance import get_prefix_weights
from openpi_rtc.pi0_sampler import Pi0RTCSampler


class RTCPolicyAdapter:
    """在保留普通推理接口的同时，增加显式的 RTC 推理路径。"""

    def __init__(self, policy: _policy.Policy, config: RTCConfig):
        if policy._is_pytorch_model:
            raise NotImplementedError("RTC currently supports only JAX Pi0/Pi0.5")
        if not isinstance(policy._model, _pi0.Pi0):
            raise TypeError(
                "RTC requires an OpenPI Pi0 model, "
                f"got {type(policy._model).__name__}"
            )

        action_horizon = policy._model.action_horizon
        if config.min_execution_horizon >= action_horizon:
            raise ValueError(
                "RTC min_execution_horizon must be smaller than the model "
                f"action horizon ({action_horizon})"
            )
        # 延迟上限同时受一半动作时域和最小执行时域约束；超过该值将无法
        # 在旧动作耗尽前持续地产生下一块动作。
        max_supported_delay = min(
            action_horizon // 2,
            action_horizon - config.min_execution_horizon,
        )
        if config.initial_delay_steps > max_supported_delay:
            raise ValueError(
                "RTC initial delay exceeds the sustainable real-time budget: "
                f"initial_delay_steps={config.initial_delay_steps}, "
                f"max_supported_delay={max_supported_delay}, "
                f"action_horizon={action_horizon}, "
                f"min_execution_horizon={config.min_execution_horizon}"
            )

        self._policy = policy
        self._config = config
        self._policy._sample_kwargs = {
            **self._policy._sample_kwargs,
            "num_steps": config.num_denoising_steps,
        }
        # 仅对 RTC 采样器做 JIT，不修改已经加载的原始 Pi0/Pi0.5 模型。
        self._sampler_module = Pi0RTCSampler(policy._model)
        self._sample_actions_rtc = nnx_utils.module_jit(
            self._sampler_module.sample_actions
        )

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        return self._policy.infer(obs, noise=noise)

    def snapshot_rng_state(self) -> Any:
        """在非 rollout 校准前保存不可变的 JAX 随机数密钥。"""

        return self._policy._rng

    def restore_rng_state(self, rng_state: Any) -> None:
        """恢复 rollout 随机数状态，避免校准改变正式采样序列。"""

        self._policy._rng = rng_state

    def infer_rtc(
        self,
        obs: dict,
        *,
        reference_actions: np.ndarray,
        inference_delay: int,
        noise: np.ndarray | None = None,
    ) -> dict[str, Any]:
        reference_actions = np.asarray(reference_actions, dtype=np.float32)
        self._validate_reference(reference_actions, inference_delay)

        # 将参考动作与观测一起送入 OpenPI 输入变换，确保归一化、填充方式和训练一致。
        inputs = jax.tree.map(lambda x: x, obs)
        inputs["actions"] = np.array(reference_actions, copy=True)
        inputs = self._policy._input_transform(inputs)
        reference_model_actions = np.asarray(
            inputs.pop("actions"),
            dtype=np.float32,
        )

        inputs = jax.tree.map(
            lambda x: jnp.asarray(x)[np.newaxis, ...],
            inputs,
        )
        observation = _model.Observation.from_dict(inputs)
        self._policy._rng, sample_rng = jax.random.split(self._policy._rng)

        sample_kwargs = dict(self._policy._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        overlap_horizon = reference_model_actions.shape[0]
        self._validate_model_reference(reference_model_actions, inference_delay)
        # 采样器要求完整 action_horizon；旧块不足的尾部补零，并由权重置零屏蔽。
        padded_reference = np.zeros(
            (self.action_horizon, self._policy._model.action_dim),
            dtype=np.float32,
        )
        padded_reference[:overlap_horizon] = reference_model_actions
        rtc_weights = get_prefix_weights(
            start=inference_delay,
            end=overlap_horizon,
            total=self.action_horizon,
            schedule=self._config.prefix_attention_schedule,
        )
        sample_kwargs.update(
            prev_actions=jnp.asarray(padded_reference)[None, ...],
            rtc_weights=rtc_weights[None, :, None],
            max_guidance_weight=self._config.max_guidance_weight,
            num_steps=self._config.num_denoising_steps,
        )

        start_time = time.monotonic()
        actions = self._sample_actions_rtc(
            sample_rng,
            observation,
            **sample_kwargs,
        )
        # JAX 默认异步派发，等待设备计算完成后计时才代表真实推理延迟。
        jax.block_until_ready(actions)
        model_time = time.monotonic() - start_time

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._policy._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    @property
    def action_horizon(self) -> int:
        return self._policy._model.action_horizon

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    @property
    def config(self) -> RTCConfig:
        return self._config

    def _validate_reference(
        self,
        reference_actions: np.ndarray,
        inference_delay: int,
    ) -> None:
        if reference_actions.ndim != 2:
            raise ValueError(
                "reference_actions must have shape [steps, action_dim], "
                f"got {reference_actions.shape}"
            )
        if not 0 < reference_actions.shape[0] <= self.action_horizon:
            raise ValueError(
                f"reference length must be in [1, {self.action_horizon}], "
                f"got {reference_actions.shape[0]}"
            )
        if not 0 <= inference_delay <= reference_actions.shape[0]:
            raise ValueError(
                "inference_delay must be in "
                f"[0, {reference_actions.shape[0]}], got {inference_delay}"
            )
        if not np.all(np.isfinite(reference_actions)):
            raise ValueError("reference_actions contains non-finite values")

    def _validate_model_reference(
        self,
        reference_actions: np.ndarray,
        inference_delay: int,
    ) -> None:
        expected_dim = self._policy._model.action_dim
        if reference_actions.ndim != 2:
            raise ValueError(
                "transformed reference actions must have shape "
                f"[steps, action_dim], got {reference_actions.shape}"
            )
        if reference_actions.shape[1] != expected_dim:
            raise ValueError(
                "transformed action dimension must be "
                f"{expected_dim}, got {reference_actions.shape[1]}"
            )
        if not 0 <= inference_delay <= reference_actions.shape[0]:
            raise ValueError(
                "inference_delay exceeds the transformed reference horizon"
            )
