"""
从 MJCF + session 通用命名规则解析遥操布局（无机型表）。

- 掌 / 基座 / ee site：在 MJCF 全量 body/site 名上按子串匹配；
- ``mjc_agent_prefix``：由匹配到的掌 body 名反推公共前缀；
- 双臂 neutral：``mj_resetData`` 后读 arm 关节默认 ``qpos``（短名）；
- ``tele_agent_name``：仅据 MJCF 关节名命名空间启发式映射到 tele conf 模块（非刚体属性）。

XPBD 刚体跟踪仍由 ``identify_xpbd_bodies`` / session ``rigid_body_map`` 负责，与本模块无关。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco

# 左右掌：按优先级在 MJCF body 全名上匹配末段（不含机型）
_LEFT_PALM_SUFFIXES = ("arm_l_end_link", "zbll_base_link", "gripper_l_palm")
_RIGHT_PALM_SUFFIXES = ("arm_r_end_link", "zbr_base_link", "gripper_r_palm")
_BASE_BODY_SUFFIXES = ("robot_holder1", "base_link")
_EE_SITE_MARKERS = ("ee_center_site",)

# tele conf 模块选择：关节名子串 → ``data_collection_cloth_tele --agent_name``
_TELE_AGENT_JOINT_MARKERS: tuple[tuple[str, str], ...] = (
    ("idx21_arm_l_joint1", "g1_omnipicker"),
    ("J_arm_l_01", "openloong"),
)

_ARM_JOINT_RE = re.compile(r"arm_[lr].*joint", re.IGNORECASE)


@dataclass(frozen=True)
class MjcfTeleLayout:
    """MJCF 扫描得到的遥操布局（写入 session ``orcagym.tele_layout``）。"""

    mjc_agent_prefix: str
    tele_agent_name: str
    base_body: str
    left_palm_body: str
    right_palm_body: str
    left_ee_site: str
    right_ee_site: str
    tele_arm_joint_values: dict[str, float]

    def to_orcagym_dict(self) -> dict[str, Any]:
        return {
            "mjc_agent_prefix": self.mjc_agent_prefix,
            "tele_agent_name": self.tele_agent_name,
            "tele_layout": {
                "base_body": self.base_body,
                "left_palm_body": self.left_palm_body,
                "right_palm_body": self.right_palm_body,
                "left_ee_site": self.left_ee_site,
                "right_ee_site": self.right_ee_site,
            },
        }

    def shell_export(self) -> str:
        lines = [
            f"export AGENT={self.tele_agent_name}",
            f"export MJC_PREFIX={self.mjc_agent_prefix}",
            f"export ROBOT_BASE_BODY={self.base_body}",
            f"export ROBOT_PALM_L={self.left_palm_body}",
            f"export ROBOT_PALM_R={self.right_palm_body}",
        ]
        return "\n".join(lines)


def _body_names(model: mujoco.MjModel) -> list[str]:
    out: list[str] = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if name and name != "world":
            out.append(name)
    return out


def _site_names(model: mujoco.MjModel) -> list[str]:
    out: list[str] = []
    for sid in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if name:
            out.append(name)
    return out


def _joint_names(model: mujoco.MjModel) -> list[str]:
    out: list[str] = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if name:
            out.append(name)
    return out


def _strip_prefix(full: str, prefix: str) -> str:
    if prefix and full.startswith(f"{prefix}_"):
        return full[len(prefix) + 1 :]
    return full


def _prefix_from_suffixed_name(full: str, suffix: str) -> str:
    if full == suffix:
        return ""
    token = f"_{suffix}"
    if full.endswith(token):
        return full[: -len(token)]
    raise ValueError(f"cannot infer prefix from {full!r} suffix {suffix!r}")


def _find_body_by_suffixes(model: mujoco.MjModel, suffixes: tuple[str, ...]) -> str | None:
    for name in _body_names(model):
        for suf in suffixes:
            if name == suf or name.endswith(f"_{suf}"):
                return name
    return None


def _find_base_body(model: mujoco.MjModel, prefix: str) -> str:
    trials: list[str] = []
    for suf in _BASE_BODY_SUFFIXES:
        if prefix:
            trials.append(f"{prefix}_{suf}")
        trials.append(suf)
    for name in trials:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0:
            return name
    raise KeyError(f"base body not in MJCF (prefix={prefix!r})")


def _find_ee_sites(model: mujoco.MjModel, prefix: str) -> tuple[str, str]:
    sites = [s for s in _site_names(model) if any(m in s for m in _EE_SITE_MARKERS)]
    if not sites:
        raise KeyError("no ee_center_site* in MJCF")

    def _side(name: str) -> str | None:
        low = name.lower()
        if "_r" in low or low.endswith("_site_r") or "site_r" in low:
            return "R"
        if "_l" in low or low.endswith("_site_l") or "site_l" in low:
            return "L"
        if name.endswith("ee_center_site_r"):
            return "R"
        if name.endswith("ee_center_site_l"):
            return "L"
        if name.endswith("ee_center_site") and "_r" not in low:
            return "L"
        return None

    left = [s for s in sites if _side(s) == "L"]
    right = [s for s in sites if _side(s) == "R"]
    if len(left) == 1 and len(right) == 1:
        return left[0], right[0]
    if len(sites) == 2:
        return sites[0], sites[1]
    if len(sites) == 1 and "ee_center_site_r" not in sites[0]:
        alt_r = f"{prefix}_ee_center_site_r" if prefix else "ee_center_site_r"
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, alt_r) >= 0:
            return sites[0], alt_r
    raise KeyError(f"cannot pair ee sites from {sites}")


def infer_tele_agent_name(model: mujoco.MjModel) -> str:
    """据 MJCF 关节名子串选择 tele conf 模块（非机型表，仅命名空间启发式）。"""
    joints = _joint_names(model)
    for marker, agent in _TELE_AGENT_JOINT_MARKERS:
        for j in joints:
            if j == marker or j.endswith(f"_{marker}"):
                return agent
    raise RuntimeError(
        "MJCF tele agent: no known arm joint namespace "
        f"(sample joints: {joints[:12]}...)"
    )


def default_arm_qpos_from_mjcf(model: mujoco.MjModel, prefix: str) -> dict[str, float]:
    """
    ``mj_resetData`` 后读取双臂关节默认 ``qpos``，键为 tele 用短名。

    只收录名称含 ``arm_l`` / ``arm_r`` 且含 ``joint`` 的关节。
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    out: dict[str, float] = {}
    for jid in range(model.njnt):
        full = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if not full:
            continue
        short = _strip_prefix(full, prefix)
        if not _ARM_JOINT_RE.search(short):
            continue
        adr = model.jnt_qposadr[jid]
        out[short] = float(data.qpos[adr])
    if not out:
        raise RuntimeError("MJCF default arm qpos: no arm_l/arm_r joints found")
    return out


def scan_tele_layout_from_model(model: mujoco.MjModel) -> MjcfTeleLayout:
    """从 ``MjModel`` 扫描遥操布局（掌/基座/site/前缀/neutral/agent）。"""
    left_full = _find_body_by_suffixes(model, _LEFT_PALM_SUFFIXES)
    right_full = _find_body_by_suffixes(model, _RIGHT_PALM_SUFFIXES)
    if not left_full or not right_full:
        raise RuntimeError(
            f"MJCF palm bodies not found (L={left_full!r} R={right_full!r}); "
            f"bodies sample: {_body_names(model)[:10]}..."
        )

    left_suf = next(s for s in _LEFT_PALM_SUFFIXES if left_full == s or left_full.endswith(f"_{s}"))
    right_suf = next(s for s in _RIGHT_PALM_SUFFIXES if right_full == s or right_full.endswith(f"_{s}"))
    prefix_l = _prefix_from_suffixed_name(left_full, left_suf)
    prefix_r = _prefix_from_suffixed_name(right_full, right_suf)
    if prefix_l != prefix_r:
        raise RuntimeError(f"MJCF prefix mismatch: L={prefix_l!r} R={prefix_r!r}")
    prefix = prefix_l

    base_full = _find_base_body(model, prefix)
    left_ee, right_ee = _find_ee_sites(model, prefix)
    tele_agent = infer_tele_agent_name(model)
    qpos = default_arm_qpos_from_mjcf(model, prefix)

    return MjcfTeleLayout(
        mjc_agent_prefix=prefix,
        tele_agent_name=tele_agent,
        base_body=_strip_prefix(base_full, prefix),
        left_palm_body=_strip_prefix(left_full, prefix),
        right_palm_body=_strip_prefix(right_full, prefix),
        left_ee_site=_strip_prefix(left_ee, prefix),
        right_ee_site=_strip_prefix(right_ee, prefix),
        tele_arm_joint_values=qpos,
    )


def scan_tele_layout_from_mjcf(mjcf_path: str | Path) -> MjcfTeleLayout:
    path = Path(mjcf_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(path))
    return scan_tele_layout_from_model(model)


def _mjcf_path_from_session(session: dict[str, Any]) -> str:
    mjcf = (session.get("mujoco") or {}).get("model_path") or ""
    meta = session.get("_cloth_robot_session_meta") or {}
    if not mjcf:
        mjcf = meta.get("source_mjcf") or ""
    if not mjcf:
        raise FileNotFoundError("session has no mjcf path for tele layout scan")
    return str(mjcf)


def _layout_from_orcagym_cache(og: dict[str, Any]) -> MjcfTeleLayout | None:
    layout = og.get("tele_layout") or og.get("robot_scan")
    prefix = str(og.get("mjc_agent_prefix", "")).strip()
    tele = str(og.get("tele_agent_name", "")).strip()
    if not layout or not tele:
        return None
    mjcf_path = og.get("_mjcf_path_for_qpos")
    qpos: dict[str, float] = {}
    if mjcf_path and Path(mjcf_path).is_file():
        try:
            qpos = default_arm_qpos_from_mjcf(mujoco.MjModel.from_xml_path(str(mjcf_path)), prefix)
        except (OSError, RuntimeError):
            qpos = {}
    return MjcfTeleLayout(
        mjc_agent_prefix=prefix,
        tele_agent_name=tele,
        base_body=str(layout["base_body"]),
        left_palm_body=str(layout["left_palm_body"]),
        right_palm_body=str(layout["right_palm_body"]),
        left_ee_site=str(layout["left_ee_site"]),
        right_ee_site=str(layout["right_ee_site"]),
        tele_arm_joint_values=qpos,
    )


def scan_tele_layout_from_session(session: dict[str, Any]) -> MjcfTeleLayout:
    """优先 session 缓存；否则从 ``mujoco.model_path`` 扫描 MJCF。"""
    og = session.get("orcagym") or {}
    cached = _layout_from_orcagym_cache(og)
    mjcf = _mjcf_path_from_session(session)
    if cached is not None and cached.mjc_agent_prefix and cached.tele_arm_joint_values:
        return cached
    layout = scan_tele_layout_from_mjcf(mjcf)
    return layout


def load_tele_layout_from_session_path(session_path: str | Path) -> MjcfTeleLayout:
    path = Path(session_path).expanduser().resolve()
    session = json.loads(path.read_text(encoding="utf-8"))
    return scan_tele_layout_from_session(session)


def apply_tele_layout_to_session(session: dict[str, Any], layout: MjcfTeleLayout) -> dict[str, Any]:
    out = dict(session)
    og = dict(out.get("orcagym") or {})
    og.update(layout.to_orcagym_dict())
    try:
        og["_mjcf_path_for_qpos"] = _mjcf_path_from_session(session)
    except FileNotFoundError:
        pass
    out["orcagym"] = og
    return out


def resolve_palm_logical_names(session: dict[str, Any]) -> tuple[str, str]:
    """左右掌 ``logical_name``（全名）：先 ``rigid_body_map`` 子串，再 MJCF tele_layout。"""
    bodies = (
        (session.get("rigid_body_map") or [])
        + (session.get("orcalink_rigid_body_map") or [])
        + (session.get("orcagym_rigid_body_map") or [])
    )
    names = [str(b.get("logical_name") or b.get("mjc_body_name") or "") for b in bodies]
    left = next((n for n in names if any(s in n for s in _LEFT_PALM_SUFFIXES)), None)
    right = next((n for n in names if any(s in n for s in _RIGHT_PALM_SUFFIXES)), None)
    if left and right:
        return left, right

    layout = scan_tele_layout_from_session(session)
    prefix = layout.mjc_agent_prefix

    def _full(short: str) -> str:
        return f"{prefix}_{short}" if prefix else short

    return _full(layout.left_palm_body), _full(layout.right_palm_body)


def layout_as_json(layout: MjcfTeleLayout) -> str:
    return json.dumps(asdict(layout), ensure_ascii=False, indent=2)
