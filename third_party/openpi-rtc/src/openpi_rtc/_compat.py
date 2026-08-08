"""openpi-rtc 与上游 openpi 版本兼容性校验。

RTC 服务端依赖 openpi 内部的若干私有 API（Policy 的模型/RNG/变换字段、
Pi0 模型的结构细节、nnx_utils.module_jit 等）。任何一处名称变动都会导致
运行时崩溃，因此在服务启动时统一做 fail-fast 校验，并在报错信息里明确指出
需要锁定的 openpi commit。

需锁定的 openpi commit：981483d
"""

from __future__ import annotations

import inspect
import sys


_REQUIRED_COMMIT = "981483d"

# Policy 私有属性：在源码字符串中扫描，因为 nnx 实例属性在类上不可见。
# 使用精确的赋值/访问模式，避免子串误命中（例如 "_model" 在 "_is_pytorch_model" 中）。
_POLICY_SRC_TOKENS = [
    "_is_pytorch_model",
    "self._model",
    "self._rng",
    "self._sample_kwargs",
    "self._input_transform",
    "self._output_transform",
]

# Pi0 源码中必须存在的字符串（涵盖实例属性与方法）
_PI0_SRC_TOKENS = [
    "action_out_proj",
    "action_horizon",
    "action_dim",
    "embed_prefix",
    "embed_suffix",
    "adarms_cond",      # 4 元组返回值标志，Pi0.5 特有
    "PaliGemma",
]

# pi0 模块级必须存在的名称（可用 hasattr 检测）
_PI0_MODULE_ATTRS = ["Pi0", "make_attn_mask"]

_MODEL_MODULE_ATTRS = ["Observation", "Actions", "preprocess_observation"]
_OBSERVATION_ATTRS = ["from_dict"]


def _fail(msg: str) -> None:
    print(
        f"\n[openpi-rtc] 兼容性校验失败：{msg}\n"
        f"  请确认使用 openpi commit {_REQUIRED_COMMIT}，\n"
        f"  并按照 openpi-rtc README 的安装步骤配置环境。\n",
        file=sys.stderr,
    )
    sys.exit(1)


def _get_src(obj: object) -> str:
    try:
        return inspect.getsource(obj)  # type: ignore[arg-type]
    except (OSError, TypeError):
        return ""


def check() -> None:
    """校验当前环境的 openpi 是否与 openpi-rtc 兼容。

    缺少任何必需的私有 API 时打印说明并以非零退出码退出。
    """

    # 1. openpi.policies.policy.Policy 私有属性（源码扫描）
    try:
        from openpi.policies import policy as _policy_mod
    except ImportError:
        _fail("无法 import openpi.policies.policy，请检查 openpi 是否已安装。")
        return

    policy_cls = getattr(_policy_mod, "Policy", None)
    if policy_cls is None:
        _fail("openpi.policies.policy 中不存在 Policy 类。")
        return

    policy_src = _get_src(policy_cls)
    for token in _POLICY_SRC_TOKENS:
        if token not in policy_src:
            _fail(
                f"openpi.policies.policy.Policy 源码中未发现 '{token}'。\n"
                f"  该属性被 RTCPolicyAdapter 使用（需要 {_REQUIRED_COMMIT}）。"
            )

    # 2. openpi.models.pi0 模块级成员
    try:
        from openpi.models import pi0 as _pi0_mod
    except ImportError:
        _fail("无法 import openpi.models.pi0，请检查 openpi 是否已安装。")
        return

    for attr in _PI0_MODULE_ATTRS:
        if not hasattr(_pi0_mod, attr):
            _fail(f"openpi.models.pi0 中缺少 '{attr}'（需要 {_REQUIRED_COMMIT}）。")

    # 3. Pi0 类源码扫描（实例属性 / 方法 / Pi0.5 特有字段）
    pi0_cls = _pi0_mod.Pi0
    pi0_src = _get_src(pi0_cls)
    for token in _PI0_SRC_TOKENS:
        if token not in pi0_src:
            _fail(
                f"openpi.models.pi0.Pi0 源码中未发现 '{token}'。\n"
                f"  该符号被 Pi0RTCSampler 或 RTCPolicyAdapter 使用"
                f"（需要 {_REQUIRED_COMMIT}）。"
            )

    # 4. PaliGemma.llm 接受 adarms_cond 关键字（在整个 pi0 模块源码中验证）
    pi0_module_src = _get_src(_pi0_mod)
    if "adarms_cond" not in pi0_module_src:
        _fail(
            "openpi.models.pi0 模块中未找到 adarms_cond，\n"
            f"  该参数被 Pi0RTCSampler 传给 PaliGemma.llm（需要 {_REQUIRED_COMMIT}）。"
        )

    # 5. openpi.models.model
    try:
        from openpi.models import model as _model_mod
    except ImportError:
        _fail("无法 import openpi.models.model。")
        return

    for attr in _MODEL_MODULE_ATTRS:
        if not hasattr(_model_mod, attr):
            _fail(f"openpi.models.model 中缺少 '{attr}'（需要 {_REQUIRED_COMMIT}）。")

    observation_cls = _model_mod.Observation
    for attr in _OBSERVATION_ATTRS:
        if not hasattr(observation_cls, attr):
            _fail(
                f"openpi.models.model.Observation 中缺少 '{attr}'（需要 {_REQUIRED_COMMIT}）。"
            )

    # 6. openpi.shared.nnx_utils.module_jit
    try:
        from openpi.shared import nnx_utils as _nnx_utils
    except ImportError:
        _fail("无法 import openpi.shared.nnx_utils。")
        return

    if not hasattr(_nnx_utils, "module_jit"):
        _fail(f"openpi.shared.nnx_utils 中缺少 'module_jit'（需要 {_REQUIRED_COMMIT}）。")

    print(
        f"[openpi-rtc] 兼容性校验通过（openpi 私有 API 完整，建议 commit {_REQUIRED_COMMIT}）。",
        file=sys.stderr,
    )


def check_or_exit() -> None:
    """与 check() 相同，用于 CLI 启动前的快捷调用。"""
    check()
