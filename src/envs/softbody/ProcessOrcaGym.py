"""OrcaGym 相关：MuJoCo 仿真 + 机器人（刚体映射 / 机型识别 / mocap 同步 / 执行器 / MuJoCo 访问）。

合并自原 body_map_orcagym.py、mujoco_access.py、mjcf_tele_layout.py、
gripper_mocap_sync.py 与 openloong_osc_actuators.py。
"""
from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Tuple

import mujoco
import numpy as np
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log.orca_log import OrcaLog

logger = logging.getLogger(__name__)
orca_logger = OrcaLog.get_instance()


# ---------------------------------------------------------------------------
# 刚体映射适配（原 body_map_orcagym.py）
# ---------------------------------------------------------------------------

def _ensure_logical_equals_mjc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``logical_name`` 与 ``mjc_body_name`` 完全一致（OrcaLink object_id 合同）。"""
    out: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        mjc = str(r.get("mjc_body_name", ""))
        if mjc:
            r["logical_name"] = mjc
        out.append(r)
    return out


def adapt_config_for_orcagym(
    model: mujoco.MjModel,
    config: dict[str, Any],
    *,
    scene_assets: Any,
    data: mujoco.MjData | None = None,
) -> dict[str, Any]:
    """
    按当前 Studio MJCF 构建 rigid_body_map；支持 ``xpbd_auto_discover`` 扫描优先。

    - ``xpbd_auto_discover.bodies=true``：默认 ``body_track_scan_only=true``，扫描 ``_XPBD_TRACK_GEOM``；
      ``body_track_scan_only=false`` 时用 ``body_include_substrings`` 兜底；
    - ``xpbd_auto_discover.cloth=true``：``identify_xpbd_cloth`` 合并进 ``config["cloth"]``；
    - 扫描模式下强制 ``pose_remap.enabled=false``；
    - body_track body-only：不要求 anchor SITE。

    ``scene_assets`` 为编排器预先解析好的场景/资产结果（``asset_dir`` / ``masked_cloth_block``），
    本函数不再自行查 ``studio_cloth_assets_dir`` / ``apply_masked_cloth_from_level``。
    """
    cfg = copy.deepcopy(config)
    disc = cfg.setdefault("anchor_discovery", {})
    disc["auto_from_model"] = False

    auto = cfg.get("xpbd_auto_discover") or {}
    use_bodies = bool(auto.get("bodies", False))
    use_cloth = bool(auto.get("cloth", False))

    if use_cloth:
        try:
            from modules.identify_xpbd_cloth import (  # noqa: WPS433
                enrich_cloth_discovery_pose,
                identify_xpbd_cloth,
                merge_cloth_discovery,
            )

            cloths = identify_xpbd_cloth(model)
            if data is not None and cloths:
                cloths = enrich_cloth_discovery_pose(model, data, cloths)
            level = str((cfg.get("orcagym") or {}).get("level") or "").strip()
            if cloths:
                from modules.masked_vtk_assets import (  # noqa: WPS433
                    enrich_discovered_cloths_with_masked_assets,
                )

                cloths = enrich_discovered_cloths_with_masked_assets(cloths, level=level or None)
            cfg = merge_cloth_discovery(cfg, cloths)
            if level and cfg.get("cloth"):
                cloth_blk = cfg["cloth"]
                cloth_blk.setdefault("level", level)
                cloth_blk.setdefault("asset_dir", scene_assets.asset_dir)
            if level and not (cfg.get("cloth") or {}).get("discovered"):
                if scene_assets.masked_cloth_block is not None:
                    cloth = cfg.setdefault("cloth", {})
                    cloth.update(scene_assets.masked_cloth_block)
                    cloth.setdefault("level", level)
                    cloth.setdefault("asset_dir", scene_assets.asset_dir)
        except ImportError as exc:
            logger.warning("cloth auto discover skipped: %s", exc)
            level = str((cfg.get("orcagym") or {}).get("level") or "").strip()
            if level and scene_assets.masked_cloth_block is not None:
                cloth = cfg.setdefault("cloth", {})
                cloth.update(scene_assets.masked_cloth_block)
                cloth.setdefault("level", level)
                cloth.setdefault("asset_dir", scene_assets.asset_dir)

    map_key = str(cfg.get("orcagym", {}).get("rigid_body_map_key", "rigid_body_map"))
    rows_in: list[dict[str, Any]] = []

    primary_collision_half_extents = None
    if use_bodies:
        try:
            from modules.identify_xpbd_bodies import (  # noqa: WPS433
                bodies_to_rigid_body_map,
                filter_body_names,
                identify_xpbd_bodies,
                resolve_bodies_by_name_substrings,
            )
            from modules.body_map import primary_collision_half_extents as _pce  # noqa: WPS433

            primary_collision_half_extents = _pce
        except ImportError as exc:
            logger.warning("body auto discover skipped: %s", exc)
            use_bodies = False

    if use_bodies:
        scanned = identify_xpbd_bodies(model)
        scan_only = bool(auto.get("body_track_scan_only", True))
        include_substrings: list[str] = []
        if not scan_only:
            include_substrings = list(auto.get("body_include_substrings") or [])
            geom_suffixes = list(auto.get("body_include_geom_suffixes") or [])
            if include_substrings:
                by_name = resolve_bodies_by_name_substrings(model, include_substrings)
                if by_name:
                    scanned = sorted(set(scanned) | set(by_name))
            if geom_suffixes:
                try:
                    from modules.identify_xpbd_bodies import resolve_bodies_by_geom_suffixes  # noqa: WPS433

                    for bname in resolve_bodies_by_geom_suffixes(model, geom_suffixes):
                        short = bname.rsplit("_", 1)[-1] if "_" in bname else bname
                        if short not in include_substrings:
                            include_substrings.append(short)
                except ImportError as exc:
                    logger.warning("geom suffix body resolve skipped: %s", exc)
        elif auto.get("body_include_substrings") or auto.get("body_include_geom_suffixes"):
            logger.info(
                "xpbd_auto_discover: body_track_scan_only=true, ignoring config body include lists"
            )
        if scan_only and not scanned:
            logger.warning(
                "xpbd_auto_discover: no _XPBD_TRACK_GEOM in MJCF; "
                "add EditorMjXpbdBodyTrack on MjBody entities in Studio"
            )
        scanned = filter_body_names(
            scanned,
            include_substrings=include_substrings or None,
            exclude_substrings=list(auto.get("body_exclude_substrings") or []),
            exclude_exact=list(auto.get("body_exclude_exact") or []),
        )
        scanned_rows = bodies_to_rigid_body_map(
            scanned,
            default_follow_mode=str(auto.get("default_follow_mode", "kinematic")),
            logical_name_from_body=False,
        )
        overrides_by_mjc = {
            str(r.get("mjc_body_name")): r for r in (cfg.get(map_key) or []) if r.get("mjc_body_name")
        }
        rows_in = []
        for row in scanned_rows:
            mjc = str(row["mjc_body_name"])
            merged = None
            if mjc in overrides_by_mjc:
                merged = dict(row)
                merged.update(overrides_by_mjc[mjc])
            else:
                for key, ov in overrides_by_mjc.items():
                    if mjc == key or mjc.endswith(f"_{key}"):
                        merged = dict(row)
                        merged.update(ov)
                        break
            rows_in.append(merged if merged is not None else row)
        scanned_set = {str(r["mjc_body_name"]) for r in rows_in}
        for row in cfg.get(map_key) or []:
            mjc = str(row.get("mjc_body_name", ""))
            if mjc and mjc not in scanned_set:
                rows_in.append(dict(row))
        og = cfg.setdefault("orcagym", {})
        pr = dict(og.get("pose_remap") or {})
        pr["enabled"] = False
        og["pose_remap"] = pr
        logger.info("xpbd_auto_discover: %d bodies from MJCF scan", len(rows_in))
    else:
        rows_in = list(cfg.get(map_key) or cfg.get("rigid_body_map") or [])

    rows_out: list[dict[str, Any]] = []
    publish_out: list[dict[str, Any]] = []
    pce_fn = primary_collision_half_extents
    if pce_fn is None:
        try:
            from modules.body_map import primary_collision_half_extents as pce_fn  # noqa: WPS433
        except ImportError:
            pce_fn = None
    pce_data = data
    if pce_fn is not None and pce_data is None:
        pce_data = mujoco.MjData(model)
        mujoco.mj_resetData(model, pce_data)
        mujoco.mj_forward(model, pce_data)
    for row in rows_in:
        name = str(row.get("mjc_body_name", ""))
        if not name:
            continue
        try:
            from modules.mjcf_body_resolve import resolve_mjcf_body_name  # noqa: WPS433
        except ImportError:
            resolve_mjcf_body_name = None  # type: ignore[assignment,misc]
        resolved = resolve_mjcf_body_name(model, name) if resolve_mjcf_body_name else name
        if resolved is None:
            logger.warning("Studio MJCF 无 body %s，跳过 OrcaLink 发布", name)
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, resolved)
        if bid < 0:
            logger.warning("Studio MJCF 无 body %s，跳过 OrcaLink 发布", name)
            continue
        row_copy = dict(row)
        row_copy["mjc_body_name"] = resolved
        row_copy.pop("anchor_sites", None)
        if row_copy.get("box_half_extents") is None and pce_fn is not None:
            half = pce_fn(model, bid, pce_data)
            row_copy["box_half_extents"] = [float(half[0]), float(half[1]), float(half[2])]
        rows_out.append(row_copy)
        if row_copy.get("orcalink_publish", True):
            publish_out.append(row_copy)
        else:
            logger.info("OrcaLink 不发布 %s（XPBD 保留场景初态）", row_copy.get("logical_name", name))

    rows_out = _ensure_logical_equals_mjc(rows_out)
    cfg["rigid_body_map"] = rows_out
    cfg["orcalink_rigid_body_map"] = publish_out
    return cfg


def validate_orcagym_body_map(model: mujoco.MjModel, entries: list) -> list[str]:
    """仅校验 body 是否存在（不检查 anchor SITE）。"""
    errors: list[str] = []
    for entry in entries:
        name = entry.mjc_body_name
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            errors.append(f"missing body: {name}")
    return errors


# ---------------------------------------------------------------------------
# MuJoCo model/data 访问（原 mujoco_access.py）
# ---------------------------------------------------------------------------

def get_mujoco_model_data(env: Any) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    返回 (MjModel, MjData)。

    OrcaGym 在 gym._mjModel / gym._mjData 维护本地 MuJoCo 状态；
    布料 OrcaLink 发布须使用与 mj_step 一致的句柄。
    """
    base = env
    if hasattr(base, "unwrapped"):
        base = base.unwrapped
    gym = getattr(base, "gym", None)
    if gym is None:
        raise RuntimeError("env has no gym backend (expected OrcaGymLocalEnv)")
    model = getattr(gym, "_mjModel", None)
    data = getattr(gym, "_mjData", None)
    if model is None or data is None:
        raise RuntimeError("gym._mjModel/_mjData not initialized; call env.reset() first")
    return model, data


def get_mujoco_xml_path(env: Any) -> Path:
    """
    返回当前 OrcaGym 仿真使用的 MJCF 路径（``gym._xml_path``）。

    用于写出 XPBD session 的 ``mujoco.model_path``；须先 ``reset()`` / ``mj_forward()``。
    """
    base = env
    if hasattr(base, "unwrapped"):
        base = base.unwrapped
    gym = getattr(base, "gym", None)
    if gym is None:
        raise RuntimeError("env has no gym backend (expected OrcaGymLocalEnv)")
    xml_path = getattr(gym, "_xml_path", None)
    if not xml_path:
        raise RuntimeError("gym._xml_path not set; call env.reset() after Studio Play")
    return Path(str(xml_path)).expanduser().resolve()


# ---------------------------------------------------------------------------
# 机型识别（原 mjcf_tele_layout.py）
# ---------------------------------------------------------------------------
# 左右掌：按优先级在 MJCF body 全名上匹配末段（不含机型）
_LEFT_PALM_SUFFIXES = ("arm_l_end_link", "zbll_base_link", "gripper_l_palm")
_RIGHT_PALM_SUFFIXES = ("arm_r_end_link", "zbr_base_link", "gripper_r_palm")
_BASE_BODY_SUFFIXES = ("robot_holder1", "base_link")
_EE_SITE_MARKERS = ("ee_center_site",)

# tele agent 识别：关节名子串 → agent 名
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


# ---------------------------------------------------------------------------
# 夹爪 mocap 同步（原 gripper_mocap_sync.py）
# ---------------------------------------------------------------------------
def sync_gripper_mocap_from_bodies(
    env: OrcaGymLocalEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: dict[str, Any],
) -> None:
    """
    把真实运动的夹爪 base body 位姿写入 mocap body，使 Studio / 调试与 OrcaLink 发包源一致。

    在 cloth coupling 的 mj_forward 之前调用（控制器已更新手臂与夹爪之后）。
    """
    og = config.get("orcagym", {})
    sync_cfg = og.get("sync_mocap_from_gripper", {})
    if not sync_cfg.get("enabled", True):
        return

    prefix = str(og.get("mjc_agent_prefix", "")).strip()
    pairs = sync_cfg.get("pairs") or [
        {
            "mocap_body": f"{prefix}_leftHandMocap" if prefix else "leftHandMocap",
            "palm_body": f"{prefix}_zbll_base_link" if prefix else "zbll_base_link",
        },
        {
            "mocap_body": f"{prefix}_rightHandMocap" if prefix else "rightHandMocap",
            "palm_body": f"{prefix}_zbr_base_link" if prefix else "zbr_base_link",
        },
    ]

    updates: dict[str, dict[str, np.ndarray]] = {}
    for pair in pairs:
        mocap_name = str(pair.get("mocap_body", ""))
        palm_name = str(pair.get("palm_body", ""))
        if not mocap_name or not palm_name:
            continue
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)
        pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, palm_name)
        if mid < 0 or pid < 0:
            continue
        if int(model.body_mocapid[mid]) < 0:
            continue
        updates[mocap_name] = {
            "pos": np.array(data.xpos[pid], dtype=np.float64, copy=True),
            "quat": np.array(data.xquat[pid], dtype=np.float64, copy=True),
        }

    if not updates:
        return
    if hasattr(env, "set_mocap_pos_and_quat"):
        env.set_mocap_pos_and_quat(updates)
    else:
        logger.warning("env 无 set_mocap_pos_and_quat，跳过 mocap 同步")


# ---------------------------------------------------------------------------
# 执行器设置（原 openloong_osc_actuators.py）
# ---------------------------------------------------------------------------
def disable_position_actuators_for_osc(
    env: OrcaGymLocalEnv,
    arm_conf: dict,
    *,
    dummy_joint_short_name: str = "dummy_joint",
) -> int:
    """
    将单臂 ``positions_names``（P_arm_*）的 actuator 改接到 ``dummy_joint``，与 dual_arm_env 一致。

    Studio 导出的 openloong MJCF 中 P/M 执行器均在 actuator group 0，
    ``disable_actuator(positions_group=1)`` 无效；P 执行器 kp=200 会拉住 neutral 关节角，
    与 OSC 写入的 M_arm 力矩冲突，导致 replay 时掌位失控、够不到布袖口。

    返回：成功断开的 P 执行器数量。
    """
    dummy_joint_full = env.joint(dummy_joint_short_name)
    dummy_joint_id = env.model.joint_name2id(dummy_joint_full)
    n = 0
    for pos_short in arm_conf["positions_names"]:
        act_full = env.actuator(pos_short)
        act_id = env.model.actuator_name2id(act_full)
        env.set_actuator_trnid(act_id, dummy_joint_id)
        n += 1
    return n


def setup_openloong_dual_arm_osc_actuators(
    env: OrcaGymLocalEnv,
    l_arm: dict,
    r_arm: dict,
    *,
    dummy_joint_short_name: str = "dummy_joint",
) -> None:
    """
    为双臂 openloong 断开左右 P_arm，仅保留 M_arm 供 OSC 力矩控制。

    应在 env 初次加载及 ``init_env()`` 重建仿真后各调用一次。
    """
    nl = disable_position_actuators_for_osc(
        env, l_arm, dummy_joint_short_name=dummy_joint_short_name
    )
    nr = disable_position_actuators_for_osc(
        env, r_arm, dummy_joint_short_name=dummy_joint_short_name
    )
    orca_logger.info(
        "OSC actuator setup: detached P_arm → dummy_joint "
        f"(left={nl}, right={nr}; Studio MJCF group=0, positions_group=1 无效)"
    )
