import jax
import jax.numpy as jnp

from openpi_rtc.config import PrefixAttentionSchedule


def get_prefix_weights(
    start: int | jax.Array,
    end: int | jax.Array,
    total: int,
    schedule: PrefixAttentionSchedule,
) -> jax.Array:
    """根据推理延迟和动作重叠区间生成 RTC 软前缀权重。"""

    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")

    # [0, start) 是推理期间已经执行的动作，[start, end) 是可引导的重叠区间。
    start = jnp.clip(jnp.asarray(start), 0, total)
    end = jnp.clip(jnp.asarray(end), 0, total)
    start = jnp.minimum(start, end)
    positions = jnp.arange(total, dtype=jnp.float32)

    if schedule == "ones":
        # 重叠区间内保持全权重。
        weights = jnp.ones(total, dtype=jnp.float32)
    elif schedule == "zeros":
        # 仅保留已执行前缀；主要用于关闭重叠区间的软引导。
        weights = (positions < start).astype(jnp.float32)
    elif schedule in ("linear", "exp"):
        denominator = end - start + 1
        weights = jnp.clip(
            (start - 1 - positions) / denominator + 1,
            0,
            1,
        )
        if schedule == "exp":
            weights = weights * jnp.expm1(weights) / (jnp.e - 1)
    else:
        raise ValueError(f"Unsupported prefix attention schedule: {schedule}")

    return jnp.where(positions >= end, 0, weights).astype(jnp.float32)


def get_guidance_weight(
    time: float | jax.Array,
    max_guidance_weight: float | jax.Array,
) -> jax.Array:
    """计算 OpenPI 从时间 1 到 0 去噪时使用的截断 RTC 引导系数。"""

    time = jnp.asarray(time, dtype=jnp.float32)
    max_guidance_weight = jnp.asarray(max_guidance_weight, dtype=jnp.float32)

    # OpenPI 的去噪时间从 1 递减到 0；tau 使用正向进度，因此先翻转时间轴。
    tau = 1.0 - time
    one_minus_tau = 1.0 - tau
    inv_r2 = (tau**2 + one_minus_tau**2) / one_minus_tau**2

    coefficient = jnp.nan_to_num(
        one_minus_tau / tau,
        nan=max_guidance_weight,
        posinf=max_guidance_weight,
        neginf=0.0,
    )
    guidance_weight = jnp.nan_to_num(
        coefficient * inv_r2,
        nan=max_guidance_weight,
        posinf=max_guidance_weight,
        neginf=0.0,
    )
    return jnp.minimum(guidance_weight, max_guidance_weight)
