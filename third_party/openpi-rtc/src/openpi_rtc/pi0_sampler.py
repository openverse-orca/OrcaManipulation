"""适配未修改的 JAX OpenPI Pi0/Pi0.5 模型的 RTC 采样器。

采样器使用旧动作块的重叠部分作为软前缀约束，并通过伪逆引导修正每一步去噪
速度。OpenPI 的采样时间从 1 递减到 0，相关引导权重也按这一时间方向计算。
"""

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.shared import array_typing as at
from openpi_rtc.guidance import get_guidance_weight


class Pi0RTCSampler(nnx.Module):
    """在不改动模型实现的前提下封装已加载的 Pi0 模型。"""

    def __init__(self, model: _pi0.Pi0):
        self.model = model

    def _decode_action_velocity(
        self,
        observation: _model.Observation,
        x_t: _model.Actions,
        time: at.Float[at.Array, ""],
        prefix_tokens: at.Float[at.Array, "b p emb"],
        prefix_mask: at.Bool[at.Array, "b p"],
        kv_cache: at.PyTree,
    ) -> _model.Actions:
        batch_size = observation.state.shape[0]
        # 前缀（图像、语言）KV 已缓存；每个去噪步只重算动作后缀。
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = (
            self.model.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
            )
        )
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(
            prefix_mask,
            "b p -> b s p",
            s=suffix_tokens.shape[1],
        )
        full_attn_mask = jnp.concatenate(
            [prefix_attn_mask, suffix_attn_mask],
            axis=-1,
        )
        assert full_attn_mask.shape == (
            batch_size,
            suffix_tokens.shape[1],
            prefix_tokens.shape[1] + suffix_tokens.shape[1],
        )

        positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1)
            - 1
        )
        (prefix_out, suffix_out), _ = self.model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return self.model.action_out_proj(
            suffix_out[:, -self.model.action_horizon :]
        )

    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        prev_actions: at.Float[at.Array, "b ah ad"],
        rtc_weights: at.Float[at.Array, "b ah 1"],
        max_guidance_weight: float | at.Float[at.Array, ""] = 5.0,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """使用 RTC 软前缀伪逆引导采样一个动作块。"""

        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        if noise is None:
            noise = jax.random.normal(
                rng,
                (
                    batch_size,
                    self.model.action_horizon,
                    self.model.action_dim,
                ),
            )
        if prev_actions.shape != noise.shape:
            raise ValueError(
                f"prev_actions must have shape {noise.shape}, got {prev_actions.shape}"
            )

        expected_weights_shape = (*noise.shape[:-1], 1)
        if rtc_weights.shape != expected_weights_shape:
            raise ValueError(
                "rtc_weights must have shape "
                f"{expected_weights_shape}, got {rtc_weights.shape}"
            )

        # 观测前缀在整个去噪循环中不变，因此提前计算一次 KV cache。
        prefix_tokens, prefix_mask, prefix_ar_mask = self.model.embed_prefix(
            observation
        )
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.model.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )

        def step(carry):
            x_t, time = carry

            def denoiser(candidate_actions):
                velocity = self._decode_action_velocity(
                    observation,
                    candidate_actions,
                    time,
                    prefix_tokens,
                    prefix_mask,
                    kv_cache,
                )
                x_0_hat = candidate_actions - time * velocity
                return x_0_hat, velocity

            # VJP 把"预测终点与旧动作的偏差"反传到当前噪声状态，形成无需
            # 显式构造雅可比矩阵的伪逆引导修正。
            x_0_hat, vjp_fn, velocity = jax.vjp(
                denoiser,
                x_t,
                has_aux=True,
            )
            # 权重为零的位置不受旧动作约束；误差本身不参与梯度计算。
            error = jax.lax.stop_gradient(
                (prev_actions - x_0_hat) * rtc_weights
            )
            correction = vjp_fn(error)[0]
            guidance_weight = get_guidance_weight(
                time,
                max_guidance_weight,
            )
            guided_velocity = velocity - guidance_weight * correction
            return x_t + dt * guided_velocity, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
