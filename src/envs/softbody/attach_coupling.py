"""将布料 MjcPBD 耦合挂载到已有 OrcaGym 环境（数据采集 / 遥操）。"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

from .base.process_utils import ProcessManager

from .ProcessOrcaGym import (
    adapt_config_for_orcagym,
    get_mujoco_model_data,
    get_mujoco_xml_path,
    validate_orcagym_body_map,
)
from .ProcessPico import write_grip_triggers
from .ProcessOrcaGym import sync_gripper_mocap_from_bodies
from .ProcessOrcaLink import OrcaLinkPoseRemapper, resolve_pose_remap, start_orcalink_if_configured
from .ProcessXPBD import start_xpbd_if_configured

from .base.paths import (
    ORCA_REPO_ROOT,
    SOFTBODY_DIR,
    SOFTBODY_DOMAIN_DIR,
    default_cloth_config_path,
    level_lower_for_hint,
    qualified_vtk_asset_path,
    resolve_cloth_data_dir,
)
from .base.masked_vtk import (
    companion_paths_for_vtk,
    load_idxmap_file,
    normalize_vtk_asset_name,
    resolve_vtk_asset_path,
)

if TYPE_CHECKING:
    from domain.anchor_frame import AnchorFrame
    from domain.body_map import BodyMapEntry

logger = logging.getLogger(__name__)

_XPBD_DEBUG_LOG = ORCA_REPO_ROOT / "XPBD" / "MjcPBD_orcalink" / "debug_log"
CLOTH_SCENE_ASSETS_BASENAME = "Config.json"


def _ensure_domain_import_path() -> None:
    root = str(SOFTBODY_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass
class ClothCouplingContext:
    """单次布料耦合会话状态。"""

    config: Dict[str, Any]
    config_path: Path
    env: OrcaGymLocalEnv
    process_manager: ProcessManager = field(default_factory=ProcessManager)
    session_timestamp: str = ""
    bridge: Any = None
    macro_frame: int = 0


@dataclass
class ClothCouplingHandle:
    """
    布料耦合句柄：每宏步向 OrcaLink 发布 POSITION，sync 模式下等待 XPBD FORCE。

    接口与 FluidCouplingHandle 一致，可挂到 DataCollectionManager.set_fluid_coupling()。
    """

    config: Dict[str, Any]
    ctx: ClothCouplingContext
    enabled: bool = True
    _grip_trigger_provider: Callable[[], tuple[float, float]] | None = field(
        default=None, repr=False
    )
    _grip_trigger_path: Path | None = field(default=None, repr=False)

    def set_grip_trigger_provider(
        self,
        provider: Callable[[], tuple[float, float]],
        trigger_path: Path | str,
    ) -> None:
        """
        PICO 实时模式：每宏步将左右 ``triggerValue`` 写入 ``trigger_path``，
        供 XPBD ``MJC_PBD_DG_TRAJ=pico`` 读取。
        """
        self._grip_trigger_provider = provider
        self._grip_trigger_path = Path(trigger_path)

    def _sync_grip_triggers_to_xpbd(self) -> None:
        if self._grip_trigger_provider is None or self._grip_trigger_path is None:
            return
        try:
            left, right = self._grip_trigger_provider()
            write_grip_triggers(self._grip_trigger_path, left, right)
        except Exception as exc:
            logger.warning("写入 grip_triggers 失败: %s", exc)

    def _base_env(self):
        base = self.ctx.env
        if hasattr(base, "unwrapped"):
            base = base.unwrapped
        return base

    def _sync_bridge_physics(self) -> bool:
        """
        将 ``ClothOrcaLinkBridge`` 的 model/data 对齐到当前 ``gym._mjData``。

        ``init_env()`` 重建仿真后旧 ``MjData`` 不再被 ``mj_step`` 更新；检测到换新实例时
        重跑 ``pose_remap.calibrate()``，使 OrcaLink 发包与 openloong 掌/指位姿一致。
        """
        bridge = self.ctx.bridge
        if bridge is None:
            return False
        model, data = get_mujoco_model_data(self._base_env())
        changed = bridge.bind_mujoco(model, data)
        if changed:
            remapper = getattr(bridge, "_pose_remapper", None)
            if remapper is not None and getattr(remapper, "enabled", False):
                remapper.calibrate(model, data, bridge.body_entries)
            logger.info("cloth: MuJoCo 句柄已刷新（MjData 实例已更换，已重校 pose_remap）")
        return changed

    def on_physics_reinitialized(self) -> None:
        """
        ``SceneManager.publish_scene`` → ``init_env()`` 之后调用。

        重置宏步计数，并绑定新建 ``MjData``，避免 XPBD 收到恒定位姿包。
        """
        self.ctx.macro_frame = 0
        self._sync_bridge_physics()

    def step(self) -> bool:
        """
        在 env.step 之前调用：mj_forward → 发布宏步 POSITION →（sync）等待 FORCE。

        返回 False 时跳过本帧 env.step（发布失败或 sync 超时）。
        """
        if not self.enabled or self.ctx.bridge is None:
            return True

        base = self._base_env()
        self._sync_bridge_physics()
        model, data = get_mujoco_model_data(base)
        sync_gripper_mocap_from_bodies(base, model, data, self.config)
        base.mj_forward()
        self._sync_grip_triggers_to_xpbd()

        bridge = self.ctx.bridge
        if bridge.should_pause():
            deadline = time.perf_counter() + 120.0
            while bridge.should_pause():
                if time.perf_counter() > deadline:
                    logger.error("sync 窗口等待超时 mf=%s", self.ctx.macro_frame)
                    return False
                time.sleep(0.002)

        mf = self.ctx.macro_frame
        ok = bridge.publish_anchor_macro_frame(mf)
        if ok:
            self.ctx.macro_frame += 1
            return True

        logger.error("cloth OrcaLink publish/wait 失败 macro_frame=%s", mf)
        return False

    def cleanup(self) -> None:
        if self.ctx.bridge is not None:
            try:
                self.ctx.bridge.close()
            except Exception as exc:
                logger.warning("bridge close: %s", exc)
            self.ctx.bridge = None
        self.ctx.process_manager.cleanup_all()


def _deep_merge(base: dict, overlay: dict) -> dict:
    """递归合并 overlay 到 base（dict 深合并，其它类型覆盖）。"""
    out = copy.deepcopy(base)
    for key, val in overlay.items():
        if key == "extends":
            continue
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _load_cloth_json_file(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cloth_config(config_path: str | Path, cloth_data_dir: Path | None = None) -> Dict[str, Any]:
    """
    加载 cloth_sim JSON；若含 \"extends\" 则递归加载基配置再深合并。

    extends 路径相对数据目录。
    """
    path = Path(config_path).expanduser().resolve()
    raw = _load_cloth_json_file(path)
    extends = raw.get("extends")
    if not extends:
        return raw
    data_dir = resolve_cloth_data_dir(cloth_data_dir)
    base_path = (data_dir / str(extends)).resolve()
    base = load_cloth_config(base_path, cloth_data_dir=cloth_data_dir)
    return _deep_merge(base, raw)


def require_orcalink_port(orcalink_cfg: dict[str, Any]) -> int:
    """读取并校验 ``orcalink.port``（缺失抛 KeyError、越界抛 ValueError）。"""
    if "port" not in orcalink_cfg:
        raise KeyError('配置缺少 orcalink.port（请在 Config.json 中设置）')
    port = int(orcalink_cfg["port"])
    if not (1 <= port <= 65535):
        raise ValueError(f"orcalink.port 无效: {port}")
    return port


def build_xpbd_session_config(base_cfg: dict[str, Any], adapted_cfg: dict[str, Any], cloth_data_dir: Path | None = None) -> dict[str, Any]:
    """
    构造供 ``MJC_PBD_CONFIG`` 使用的会话 JSON。

    P2 默认 ``xpbd.cloth_discover_only=true``：仅将 ``cloth`` discovered 段交给 XPBD，
    ``rigid_body_map`` 保持基配置短链（暂不向 XPBD 传递 Studio 扫描的 N 刚体）。
    """
    out = copy.deepcopy(adapted_cfg)
    # 归一化 debug.debug_log_dir 为绝对路径：config 里是相对 dataCollection_cook 的 ../../../../XPBD/...，
    # 但 session 落盘在 logs/<ts>/ 下，C++ 侧按 session 目录解析会错位到 OrcaManipulation/src/XPBD。
    dbg = out.get("debug") or {}
    log_dir = str(dbg.get("debug_log_dir") or "").strip()
    if log_dir and cloth_data_dir:
        p = Path(log_dir)
        if not p.is_absolute():
            dbg["debug_log_dir"] = str((resolve_cloth_data_dir(cloth_data_dir) / p).resolve())
    xpbd_blk = out.setdefault("xpbd", {})
    if bool(xpbd_blk.get("cloth_discover_only", True)):
        if "rigid_body_map" in base_cfg:
            out["rigid_body_map"] = copy.deepcopy(base_cfg["rigid_body_map"])
        if "orcalink_rigid_body_map" in base_cfg:
            out["orcalink_rigid_body_map"] = copy.deepcopy(base_cfg["orcalink_rigid_body_map"])
        elif "orcagym_rigid_body_map" in base_cfg:
            out["orcalink_rigid_body_map"] = copy.deepcopy(base_cfg["orcagym_rigid_body_map"])
    cloth = out.setdefault("cloth", {})
    level = str((out.get("orcagym") or {}).get("level") or cloth.get("level") or "").strip()
    if level:
        cloth["level"] = level
        if not str(cloth.get("asset_dir") or "").strip():
            cloth["asset_dir"] = str(studio_cloth_assets_dir(level, cloth_data_dir=cloth_data_dir))
    return out


def write_xpbd_runtime_session_config(
    config: dict[str, Any],
    *,
    session_timestamp: str,
    source_config_path: Path | None = None,
    source_mjcf_path: Path | None = None,
    log_dir: Path | None = None,
) -> Path:
    """
    将运行时 effective config 写入 ``<log_dir>/<run_dir>/cloth_sim_session_{ts}.json``，供 XPBD 子进程加载。

    ``run_dir`` 为本次运行开始时间（年-月-日-时-分-秒）。
    """
    base_dir = Path(log_dir).expanduser().resolve() if log_dir else Path.cwd()
    run_dir = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    session_dir = base_dir / run_dir
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = (session_dir / f"cloth_sim_session_{session_timestamp}.json").resolve()
    payload = copy.deepcopy(config)
    meta: dict[str, str] = {
        "session_timestamp": session_timestamp,
        "source_config": str(source_config_path.resolve()) if source_config_path else "",
    }
    if source_mjcf_path is not None and source_mjcf_path.is_file():
        meta["source_mjcf"] = str(source_mjcf_path.resolve())
    payload["_cloth_robot_session_meta"] = meta
    session_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("XPBD session config: %s", session_path)
    return session_path


def _export_xpbd_scene(config_path: Path, out_path: Path) -> Path:
    """
    从 session JSON + MJCF 导出 ``xpbd_scene_from_mjcf.json``（原 scripts/export_xpbd_scene_from_mjcf.py 归位）。

    session 须含 ``rigid_body_map`` 与 ``mujoco.model_path`` 或 ``_cloth_robot_session_meta.source_mjcf``。
    返回写出 JSON 的绝对路径。
    """
    import mujoco
    import numpy as np

    _ensure_domain_import_path()
    from domain.anchor_tetrahedron import anchor_local_positions, anchor_site_names  # noqa: WPS433
    from domain.body_map import load_body_map, validate_body_map  # noqa: WPS433
    from domain.mjc_coords import orca_quat_to_yup_link_orientation, orca_vec_to_yup  # noqa: WPS433

    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent
    meta = cfg.get("_cloth_robot_session_meta") or {}
    src_mjcf = meta.get("source_mjcf")
    mjcf_rel = cfg.get("mujoco", {}).get("model_path", "")
    if src_mjcf and Path(src_mjcf).is_file():
        mjcf_path = Path(src_mjcf).resolve()
    elif mjcf_rel:
        mjcf_path = (config_dir / mjcf_rel).resolve()
    else:
        raise RuntimeError("missing mujoco.model_path or _cloth_robot_session_meta.source_mjcf")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    entries_by_mjc = {e.mjc_body_name: e for e in load_body_map(model, cfg)}
    map_rows = cfg.get("rigid_body_map", [])
    if not map_rows:
        raise RuntimeError("rigid_body_map is empty")

    all_entries = [entries_by_mjc[row["mjc_body_name"]] for row in map_rows]
    body_only = body_track_position_packet_body_only(cfg)
    if body_only:
        for row in map_rows:
            mjc_name = row["mjc_body_name"]
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mjc_name) < 0:
                raise RuntimeError(f"body missing {mjc_name}")
    else:
        val_errs = validate_body_map(model, all_entries)
        if val_errs:
            raise RuntimeError("body_map validation: " + "; ".join(val_errs))

    def half_extents_mjc_to_yup(hx: float, hy: float, hz: float) -> list[float]:
        yx, yy, yz = orca_vec_to_yup(hx, hy, hz)
        return [abs(yx), abs(yy), abs(yz)]

    bodies_out: list[dict] = []
    for body_index, row in enumerate(map_rows):
        mjc_name = row["mjc_body_name"]
        entry = entries_by_mjc[mjc_name]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mjc_name)
        follow = row.get("follow_mode", entry.follow_mode)
        mj_mass = float(model.body_mass[bid])
        mass_kg = 0.0 if follow == "kinematic" else mj_mass

        hx, hy, hz = entry.box_half_extents
        half_yup = half_extents_mjc_to_yup(hx, hy, hz)

        xpos = data.xpos[bid].astype(np.float64)
        xquat = data.xquat[bid].astype(np.float64)
        center_yup = orca_vec_to_yup(float(xpos[0]), float(xpos[1]), float(xpos[2]))
        quat_yup = orca_quat_to_yup_link_orientation(
            float(xquat[0]), float(xquat[1]), float(xquat[2]), float(xquat[3])
        )

        anchors_out: list[dict] = []
        anchor_sites = list(entry.anchor_sites)
        if body_only and len(anchor_sites) < 4:
            anchor_sites = anchor_site_names(mjc_name)
        for ai, sname in enumerate(anchor_sites):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid >= 0:
                lp = model.site_pos[sid].astype(np.float64)
                local_mjc = [float(lp[0]), float(lp[1]), float(lp[2])]
            elif body_only:
                _, verts = anchor_local_positions(hx, hy, hz)
                vx, vy, vz = verts[ai]
                local_mjc = [float(vx), float(vy), float(vz)]
            else:
                raise RuntimeError(f"site missing {sname}")
            local_yup = list(orca_vec_to_yup(local_mjc[0], local_mjc[1], local_mjc[2]))
            anchors_out.append(
                {
                    "index": ai,
                    "site_name": sname,
                    "local_pos_mjc": local_mjc,
                    "local_pos_yup": local_yup,
                }
            )

        if len(anchors_out) != 4:
            raise RuntimeError(f"{mjc_name} expected 4 anchors, got {len(anchors_out)}")

        bodies_out.append(
            {
                "body_index": body_index,
                "logical_name": mjc_name,
                "mjc_body_name": mjc_name,
                "follow_mode": follow,
                "mass_kg": mass_kg,
                "mass_mjcf_kg": mj_mass,
                "box_half_extents_mjc": [hx, hy, hz],
                "box_half_extents_yup": list(half_yup),
                "center_mjc": [float(xpos[0]), float(xpos[1]), float(xpos[2])],
                "center_yup": list(center_yup),
                "quat_wxyz_mjc": [
                    float(xquat[0]),
                    float(xquat[1]),
                    float(xquat[2]),
                    float(xquat[3]),
                ],
                "quat_wxyz_yup": list(quat_yup),
                "anchors": anchors_out,
            }
        )

    doc = {
        "schema_version": 1,
        "coord_system_sim": "yup",
        "coord_system_mjcf": "zup",
        "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_config": str(config_path),
        "source_mjcf": str(mjcf_path),
        "body_count": len(bodies_out),
        "bodies": bodies_out,
    }

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("exported xpbd scene: %s (%d bodies)", out_path, len(bodies_out))
    return out_path


def export_xpbd_scene_for_session(
    session_path: Path,
    *,
    out_path: Path | None = None,
    cloth_data_dir: Path | None = None,
) -> Path:
    """
    从 session JSON 导出 ``xpbd_scene_from_mjcf.json``（逻辑已归位到 ``_export_xpbd_scene``，不再 subprocess）。

    session 须含 ``rigid_body_map`` 与 ``mujoco.model_path`` 或 ``_cloth_robot_session_meta.source_mjcf``。
    返回写出 JSON 的绝对路径。
    """
    session_path = session_path.resolve()
    if not session_path.is_file():
        raise FileNotFoundError(f"session config not found: {session_path}")

    if out_path is not None:
        return _export_xpbd_scene(session_path, out_path.resolve())

    cfg = json.loads(session_path.read_text(encoding="utf-8"))
    dbg = cfg.get("debug", {})
    dbg_dir = Path(str(dbg.get("debug_log_dir", _XPBD_DEBUG_LOG)))
    data_dir = resolve_cloth_data_dir(cloth_data_dir)
    if not dbg_dir.is_absolute():
        dbg_dir = (data_dir / dbg_dir).resolve()
    return _export_xpbd_scene(session_path, (dbg_dir / "xpbd_scene_from_mjcf.json").resolve())


def cloth_scene_assets_config_path(cloth_data_dir: Path | None = None) -> Path:
    """``Config.json`` 路径；可用 ``CLOTH_SCENE_ASSETS_CONFIG`` 覆盖。"""
    override = os.environ.get("CLOTH_SCENE_ASSETS_CONFIG", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    data_dir = resolve_cloth_data_dir(cloth_data_dir)
    return (data_dir / CLOTH_SCENE_ASSETS_BASENAME).resolve()


def load_template_config_for_paths(cloth_data_dir: Path | None = None) -> dict[str, Any]:
    """仅读仓库模板（场景/资产解析内部用）。"""
    path = cloth_scene_assets_config_path(cloth_data_dir)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def studio_cloth_assets_dir(level: str, cloth_data_dir: Path | None = None) -> Path:
    """
    场景权威布料资产目录：``{studio_project}/Assets/{level}/``。

    首次访问某关卡时会尝试扫描 prefab 并更新 ``~/.orcagym/cloth/scene_levels.json``。
    """
    level_name = str(level).strip()
    if not level_name:
        raise ValueError("studio_cloth_assets_dir requires non-empty level")

    import sys

    domain_dir = SOFTBODY_DOMAIN_DIR
    if str(domain_dir) not in sys.path:
        sys.path.insert(0, str(domain_dir))

    ensure_level_scene_config(level_name, cloth_data_dir=cloth_data_dir)
    level_entry(level_name, auto_sync=False, cloth_data_dir=cloth_data_dir)
    return (_studio_project_dir_from_template(cloth_data_dir=cloth_data_dir) / "Assets" / level_name).resolve()


def _studio_project_dir_from_template(cloth_data_dir: Path | None = None) -> Path:
    import sys

    domain_dir = SOFTBODY_DOMAIN_DIR
    if str(domain_dir) not in sys.path:
        sys.path.insert(0, str(domain_dir))

    return studio_project_dir(load_template_config_for_paths(cloth_data_dir=cloth_data_dir))


def level_primary_masked_stem(level: str, cloth_data_dir: Path | None = None) -> str | None:
    """
    从 ``scene_levels.json`` 读取关卡主掩码布 stem。

    当 MJCF 尚无 ``_XPBD_CLOTHSHEET_*`` 标记时，用此 stem 解析 ``cloth.mesh``，
    避免回退到基配置里的 ``shirt_v4.vtk``。
    """
    level_name = str(level).strip()
    if not level_name:
        return None
    import sys

    domain_dir = SOFTBODY_DOMAIN_DIR
    if str(domain_dir) not in sys.path:
        sys.path.insert(0, str(domain_dir))
    if str(SOFTBODY_DIR) not in sys.path:
        sys.path.insert(0, str(SOFTBODY_DIR))

    _ensure_level_scene_config(level_name, cloth_data_dir=cloth_data_dir)
    entry = level_entry(level_name, auto_sync=False, cloth_data_dir=cloth_data_dir)
    if not isinstance(entry, dict):
        return None
    if str(entry.get("cloth_mode") or "").strip() != "masked_vtk":
        return None
    stems = entry.get("masked_cloth_stems") or []
    if not stems:
        prefabs = entry.get("prefabs") or []
        if prefabs and isinstance(prefabs[0], dict):
            vtk_path = str(prefabs[0].get("vtk_asset_path") or "").strip()
            if vtk_path:
                return Path(vtk_path).stem
        return None
    return str(stems[0]).strip() or None


@dataclass
class ResolvedSceneAssets:
    """编排器对某关卡「查一次」得到的布料场景/资产结果，传给各 Process* 模块。"""

    asset_dir: str
    masked_cloth_block: dict[str, Any] | None


def enrich_cloth_entry_with_masked_assets(
    entry: dict[str, Any],
    *,
    search_roots: Sequence[Path] | None = None,
    level: str | None = None,
    asset_dir: Path | None = None,
) -> dict[str, Any]:
    """
    根据 ``vtk_asset_path`` / ``mesh`` 补全掩码资产与紧凑索引字段，写入 session ``cloth`` 块。

    同时写入 ``asset_dir``、``level``，供 XPBD ``MjcPbdConfig`` 与预制检查使用。
    """
    out = dict(entry)
    resolved_level = str(level or out.get("level") or "").strip()
    if resolved_level:
        out["level"] = resolved_level
        if asset_dir is not None:
            out["asset_dir"] = str(asset_dir)

    vtk_name = str(out.get("vtk_asset_path") or out.get("mesh") or "").strip()
    if not vtk_name or vtk_name.startswith("procedural:"):
        return out

    vtk_name = normalize_vtk_asset_name(vtk_name, level=resolved_level or None)
    if search_roots is None and asset_dir is not None:
        search_roots = [asset_dir]
    vtk_path = resolve_vtk_asset_path(vtk_name, search_roots, level=resolved_level or None)
    if vtk_path is None:
        logger.warning("enrich_cloth_entry: 未在场景资产目录找到 VTK %s (level=%s)", vtk_name, resolved_level)
        return out

    out["mesh"] = vtk_path.name
    out["vtk_asset_path"] = vtk_path.name
    out["vtk_path_resolved"] = str(vtk_path)
    out["asset_dir"] = str(vtk_path.parent.resolve())

    companions = companion_paths_for_vtk(vtk_path)
    for key, path in companions.items():
        if path.is_file():
            out[key] = str(path.resolve())

    meta_path = companions["meta_json_path"]
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                out["topo_type"] = meta.get("topo_type", "masked_sheet")
                if "nx" in meta:
                    out["cloth_nx"] = int(meta["nx"])
                if "ny" in meta:
                    out["cloth_ny"] = int(meta["ny"])
                if "spacing" in meta:
                    out["cloth_spacing_m"] = float(meta["spacing"])
                if "active_count" in meta:
                    out["compact_count"] = int(meta["active_count"])
                if "coordinate" in meta:
                    out["coordinate"] = str(meta["coordinate"])
                out["cook_y_flip"] = bool(
                    meta.get("cook_y_flip")
                    if "cook_y_flip" in meta
                    else "o3de_cook" in str(meta.get("coordinate") or "").lower()
                )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("读取 meta.json 失败 %s: %s", meta_path, exc)

    idxmap_path = companions["idxmap_path"]
    idxmap = load_idxmap_file(idxmap_path)
    if idxmap:
        out["align_mode"] = str(idxmap.get("align_mode") or "idxmap")
        compact_to_fbx = idxmap.get("compact_to_fbx")
        if isinstance(compact_to_fbx, list):
            out["compact_to_fbx"] = [int(x) for x in compact_to_fbx]
            out["compact_count"] = int(idxmap.get("compact_count") or len(compact_to_fbx))
        if idxmap.get("index_rule"):
            out["index_rule"] = str(idxmap["index_rule"])
    elif companions["mask_path"].is_file():
        out.setdefault("align_mode", "identity")

    return out


def _resolve_scene_assets(config: dict[str, Any], cloth_data_dir: Path | None = None) -> ResolvedSceneAssets:
    """解析某关卡的 asset_dir、掩码布 stem 与掩码布块（原 paths.apply_masked_cloth_from_level 的「查」部分）。"""
    level = str((config.get("orcagym") or {}).get("level") or "").strip()
    if not level:
        return ResolvedSceneAssets(asset_dir="", masked_cloth_block=None)

    asset_dir = str(studio_cloth_assets_dir(level, cloth_data_dir=cloth_data_dir))
    stem = level_primary_masked_stem(level, cloth_data_dir=cloth_data_dir)
    cloth = config.get("cloth") or {}
    masked: dict[str, Any] | None = None
    if stem and not cloth.get("discovered"):
        current_mesh = str(cloth.get("mesh") or "").strip()
        legacy_meshes = ("shirt_v4.vtk", "shirt_new.vtk", "")
        mesh_name = (
            current_mesh
            if (current_mesh and current_mesh not in legacy_meshes and stem in current_mesh)
            else f"{stem}.vtk"
        )
        masked = enrich_cloth_entry_with_masked_assets(
            {"mesh": mesh_name, "vtk_asset_path": mesh_name},
            level=level,
            asset_dir=Path(asset_dir),
        )
    return ResolvedSceneAssets(
        asset_dir=asset_dir, masked_cloth_block=masked
    )


def enrich_cloth_discovery_pose(model: Any, data: Any, discovered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    用 ``mj_forward`` 后的 body 世界位姿填充每条布片发现的 ``center_*`` / ``quat_wxyz_*``。

    坐标：MuJoCo Z-up（``center_mjc``）与 XPBD Y-up（``center_yup``，经 ``mjc_coords`` 转换）。
    """
    import math
    import mujoco
    import numpy as np

    from domain.mjc_coords import orca_quat_to_yup, orca_vec_to_yup  # noqa: WPS433

    out: list[dict[str, Any]] = []
    for entry in discovered:
        row = dict(entry)
        body_name = str(row.get("body_name", ""))
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            logger.warning("enrich_cloth_discovery_pose: missing body %s", body_name)
            out.append(row)
            continue
        xpos = data.xpos[bid].astype(np.float64)
        xquat = data.xquat[bid].astype(np.float64)
        row["center_mjc"] = [float(xpos[0]), float(xpos[1]), float(xpos[2])]
        row["quat_wxyz_mjc"] = [
            float(xquat[0]), float(xquat[1]), float(xquat[2]), float(xquat[3]),
        ]
        cy = orca_vec_to_yup(float(xpos[0]), float(xpos[1]), float(xpos[2]))
        cq = orca_quat_to_yup(
            float(xquat[0]), float(xquat[1]), float(xquat[2]), float(xquat[3]),
        )
        row["center_yup"] = [float(cy[0]), float(cy[1]), float(cy[2])]
        row["quat_wxyz_yup"] = [float(cq[0]), float(cq[1]), float(cq[2]), float(cq[3])]
        w = max(-1.0, min(1.0, float(cq[0])))
        angle_deg = math.degrees(2.0 * math.acos(abs(w)))
        logger.info(
            "enrich_cloth_discovery_pose: body=%s center_yup=%s quat_wxyz_yup=%s angle_deg=%.2f",
            body_name, row["center_yup"], row["quat_wxyz_yup"], angle_deg,
        )
        out.append(row)
    return out


def merge_cloth_discovery(config: dict[str, Any], discovered: list[dict[str, Any]]) -> dict[str, Any]:
    """
    将扫描结果合并进运行时 config["cloth"]；JSON 字段为 override（scan-first）。

    若未发现布片则保持原 config 不变。
    """
    if not discovered:
        return config
    out = dict(config)
    cloth_cfg = dict(out.get("cloth") or {})
    primary = discovered[0]
    cloth_cfg.setdefault("mesh", "shirt_v4.vtk")
    if primary.get("vtk_asset_path"):
        cloth_cfg["mesh"] = primary["vtk_asset_path"]
    for key in ("mass_kg", "thickness_m", "stretch_compliance", "shear_compliance", "bend_compliance", "lock_radius_m"):
        if key in primary and key not in cloth_cfg:
            cloth_cfg[key] = primary[key]
    cloth_cfg["body_name"] = primary.get("body_name")
    cloth_cfg["discovered"] = True
    cloth_cfg["discovered_cloths"] = discovered
    for key in ("center_mjc", "quat_wxyz_mjc", "center_yup", "quat_wxyz_yup"):
        if key in primary:
            cloth_cfg[key] = primary[key]
    out["cloth"] = cloth_cfg
    return out


_VTK_GEOM_RE = re.compile(r"^(?P<body>.+)_XPBD_CLOTHSHEET_VTK__(?P<token>.+)$")
_VTK_SITE_RE = re.compile(r"^(?P<body>.+)_XPBD_CLOTHSHEET_BOUNDS__VTK__(?P<token>.+)$")
_CLOTH_BODY_MARKERS = ("Cloth_Sheet",)


def _unsanitize_vtk_token(token: str) -> str:
    """Restore vtk filename from sanitized geom suffix (dots → underscores)."""
    if token.endswith("_vtk"):
        return token[:-4] + ".vtk"
    return token.replace("_", "/")


def _parse_site_user(model: Any, site_id: int) -> dict[str, float]:
    """Read site user[] floats: mass, thickness, stretch, shear, bend, lockRadius."""
    keys = ("mass_kg", "thickness_m", "stretch_compliance", "shear_compliance", "bend_compliance", "lock_radius_m")
    out: dict[str, float] = {}
    nuser = int(model.nsiteuser)
    if nuser <= 0:
        return out
    base = int(model.site_useradr[site_id])
    for i, key in enumerate(keys):
        idx = base + i
        if idx < nuser:
            out[key] = float(model.site_user[idx])
    return out


def _vtk_path_from_site_name(site_name: str, body_name: str) -> str | None:
    m = _VTK_SITE_RE.match(site_name)
    if m and m.group("body") == body_name:
        return _unsanitize_vtk_token(m.group("token"))
    return None


def _vtk_path_for_body(model: Any, body_id: int, body_name: str) -> str | None:
    import mujoco

    for sid in range(model.nsite):
        if int(model.site_bodyid[sid]) != body_id:
            continue
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        vtk = _vtk_path_from_site_name(sname, body_name)
        if vtk:
            return vtk
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != body_id:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        m = _VTK_GEOM_RE.match(gname)
        if m and m.group("body") == body_name:
            return _unsanitize_vtk_token(m.group("token"))
    return None


def identify_cloth_bodies_by_name(model: Any) -> list[dict[str, Any]]:
    """
    无 ``_XPBD_CLOTHSHEET_*`` site 时，按 body 名含 ``Cloth_Sheet`` 兜底发现布片。

    OrcaLab datalink 关卡可能仅有 PBDRender 实体、未写 XPBD site 标记。
    """
    import mujoco

    cloths: list[dict[str, Any]] = []
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not bname or bname == "world":
            continue
        if not any(marker in bname for marker in _CLOTH_BODY_MARKERS):
            continue
        cloths.append(
            {
                "body_name": bname,
                "discovered": True,
                "fallback_by_name": True,
            }
        )
        logger.info("identify_cloth_bodies_by_name: body=%s", bname)
    return cloths


def identify_xpbd_cloth(model: Any) -> list[dict[str, Any]]:
    """
    扫描 MJCF 中所有 ``{body}_XPBD_CLOTHSHEET_BOUNDS`` site，返回布片发现列表。

    每项包含：body_name、bounds 半轴/位置（body 局部系）、user 物理参数、可选 vtk_asset_path。
    """
    import mujoco
    import numpy as np

    cloths: list[dict[str, Any]] = []
    for sid in range(model.nsite):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if "_XPBD_CLOTHSHEET_BOUNDS" not in sname:
            continue
        body_id = int(model.site_bodyid[sid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not body_name:
            logger.warning("clothsheet site %s has no body", sname)
            continue
        half = model.site_size[sid, :3].astype(np.float64)
        pos = model.site_pos[sid, :3].astype(np.float64)
        entry: dict[str, Any] = {
            "body_name": body_name,
            "site_name": sname,
            "bounds_half_extents": tuple(float(x) for x in half),
            "bounds_pos_local": tuple(float(x) for x in pos),
            "discovered": True,
        }
        entry.update(_parse_site_user(model, sid))
        vtk = _vtk_path_for_body(model, body_id, body_name)
        if vtk:
            entry["vtk_asset_path"] = vtk
        cloths.append(entry)
        logger.info("identify_xpbd_cloth: body=%s vtk=%s", body_name, vtk)
    if not cloths:
        cloths = identify_cloth_bodies_by_name(model)
    return cloths


def _resolve_cloth_from_model(
    model: Any,
    data: Any,
    config: dict[str, Any],
    *,
    cloth_data_dir: Path | None = None,
) -> tuple[dict[str, Any], ResolvedSceneAssets]:
    """
    场景/资产解析完整流程：识别布片 + 补位姿 + 补掩码资产 + 合并 cloth + 解析资产 + 未 discovered 兜底。

    统一收在编排器（原分散在 build_p2_session_from_mjcf 与 ProcessOrcaGym.adapt_config_for_orcagym 两处，
    且 identify/enrich/merge 重复执行）。返回 (合并 cloth 后的 config, scene_assets)。
    """
    cfg = copy.deepcopy(config)
    level = str((cfg.get("orcagym") or {}).get("level") or "").strip()

    # 1) 识别布片 + 补位姿
    cloths = enrich_cloth_discovery_pose(model, data, identify_xpbd_cloth(model))
    # 2) 补掩码资产（需 asset_dir，先解析一次）
    if cloths and level:
        asset_dir = str(studio_cloth_assets_dir(level, cloth_data_dir=cloth_data_dir))
        cloths = [
            enrich_cloth_entry_with_masked_assets(row, level=level, asset_dir=Path(asset_dir))
            for row in cloths
        ]
    # 3) 合并 cloth
    cfg = merge_cloth_discovery(cfg, cloths)
    # 4) 解析资产（asset_dir + stem + masked_block，按 merged discovered 算）
    scene_assets = _resolve_scene_assets(cfg, cloth_data_dir=cloth_data_dir)
    # 5) set level/asset_dir + 未 discovered 兜底
    if level and cfg.get("cloth"):
        cfg["cloth"].setdefault("level", level)
        cfg["cloth"].setdefault("asset_dir", scene_assets.asset_dir)
    if level and not (cfg.get("cloth") or {}).get("discovered"):
        if scene_assets.masked_cloth_block is not None:
            cloth = cfg.setdefault("cloth", {})
            cloth.update(scene_assets.masked_cloth_block)
            cloth.setdefault("level", level)
            cloth.setdefault("asset_dir", scene_assets.asset_dir)
    return cfg, scene_assets


# openloong 夹爪精简：_geom_* 网格所在 link body（非 mesh 实体本身）
_GRIPPER_GEOM_SUFFIX_TO_BODY_SUFFIX: dict[str, str] = {
    "_geom_27": "zbr_base_link",
    "_geom_65": "zbll_base_link",
    "_geom_35": "r_left_spring_link",
    "_geom_38": "r_left_follower",
    "_geom_45": "r_right_spring_link",
    "_geom_47": "r_right_follower",
    "_geom_73": "l_left_spring_link",
    "_geom_76": "l_left_follower",
    "_geom_83": "l_right_spring_link",
    "_geom_85": "l_right_follower",
}


def identify_xpbd_bodies(model: Any) -> list[str]:
    """
    扫描所有 ``{body}_XPBD_TRACK_GEOM`` 几何体，返回待跟踪刚体 body 名列表（字母序）。

    仅包含 MJCF 中显式打标的 body；zbll/zbr 子树由 Studio ``EditorMjXpbdBodyTrackComponent`` 写入。
    """
    import mujoco

    bodies: set[str] = set()
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if "_XPBD_TRACK_GEOM" not in gname:
            continue
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname and bname != "world":
            bodies.add(bname)
    result = sorted(bodies)
    logger.info("identify_xpbd_bodies: found %d bodies", len(result))
    return result


def resolve_bodies_by_geom_suffixes(
    model: Any,
    geom_suffixes: list[str] | None = None,
) -> list[str]:
    """
    按 MJCF geom 名后缀（如 ``_geom_47``）解析其挂载 body 全名。

    用于夹爪精简：指尖/掌面/弹簧片 mesh 对应 link body，供 XPBD 白名单过滤。
    """
    import mujoco

    suffixes = geom_suffixes or list(_GRIPPER_GEOM_SUFFIX_TO_BODY_SUFFIX.keys())
    found: dict[str, str] = {}
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        for suf in suffixes:
            if suf not in gname:
                continue
            bid = int(model.geom_bodyid[gid])
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname and bname != "world":
                found[suf] = bname
    return [found[s] for s in suffixes if s in found]


def resolve_bodies_by_name_substrings(
    model: Any,
    substrings: list[str] | None,
) -> list[str]:
    """
    在 MJCF 全部 body 名中按子串白名单匹配（如 ``zbll_base_link``、``r_left_follower``）。

    当 Studio 未写出 ``_XPBD_TRACK_GEOM`` 时，仍可将夹爪 link 纳入 ``rigid_body_map``。
    """
    import mujoco

    if not substrings:
        return []
    found: list[str] = []
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not bname or bname == "world":
            continue
        if any(sub in bname for sub in substrings):
            found.append(bname)
    result = sorted(set(found))
    if result:
        logger.info("resolve_bodies_by_name_substrings: %d bodies", len(result))
    return result


def filter_body_names(
    body_names: list[str],
    *,
    include_substrings: list[str] | None = None,
    exclude_substrings: list[str] | None = None,
    exclude_exact: list[str] | None = None,
) -> list[str]:
    """
    按子串/精确名过滤扫描结果。

    ``include_substrings`` 非空时，仅保留名称包含任一子串的 body（白名单，用于夹爪精简 + Cube）。
    """
    include_substrings = include_substrings or []
    exclude_substrings = exclude_substrings or []
    exclude_exact = set(exclude_exact or [])
    out: list[str] = []
    for name in body_names:
        if name in exclude_exact:
            continue
        if any(sub in name for sub in exclude_substrings):
            continue
        if include_substrings and not any(sub in name for sub in include_substrings):
            continue
        out.append(name)
    return out


def bodies_to_rigid_body_map(
    body_names: list[str],
    *,
    default_follow_mode: str = "kinematic",
) -> list[dict[str, Any]]:
    """将扫描得到的 body 名列表转为 ``rigid_body_map`` 行（body_track body-only，无 anchor SITE）。"""
    rows: list[dict[str, Any]] = []
    for name in body_names:
        rows.append(
            {
                "logical_name": name,
                "mjc_body_name": name,
                "follow_mode": default_follow_mode,
                "discovered": True,
            }
        )
    return rows


def _resolve_bodies_from_model(model: Any, config: dict[str, Any]) -> dict[str, Any]:
    """
    扫描 MJCF ``_XPBD_TRACK_GEOM``（scan-first），把扫描到的刚体写入 ``config["rigid_body_map"]``。

    与 ``_resolve_cloth_from_model`` 对称：刚体识别统一收在编排器，``adapt_config_for_orcagym``
    只做 OrcaLink 富化，不再重复扫描。
    """
    out = copy.deepcopy(config)
    disc = out.setdefault("anchor_discovery", {})
    disc["auto_from_model"] = False

    auto = out.get("xpbd_auto_discover") or {}
    map_key = str((out.get("orcagym") or {}).get("rigid_body_map_key", "rigid_body_map"))
    if not auto.get("bodies", False):
        # 不扫描：沿用配置既有刚体表（优先 map_key），等价旧 adapt_config_for_orcagym 的 else 分支
        out["rigid_body_map"] = list(out.get(map_key) or out.get("rigid_body_map") or [])
        return out

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
            for bname in resolve_bodies_by_geom_suffixes(model, geom_suffixes):
                short = bname.rsplit("_", 1)[-1] if "_" in bname else bname
                if short not in include_substrings:
                    include_substrings.append(short)
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
    )
    overrides_by_mjc = {
        str(r.get("mjc_body_name")): r for r in (out.get(map_key) or []) if r.get("mjc_body_name")
    }
    rows_in: list[dict[str, Any]] = []
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
    for row in out.get(map_key) or []:
        mjc = str(row.get("mjc_body_name", ""))
        if mjc and mjc not in scanned_set:
            rows_in.append(dict(row))

    og = out.setdefault("orcagym", {})
    pr = dict(og.get("pose_remap") or {})
    pr["enabled"] = False
    og["pose_remap"] = pr
    logger.info("xpbd_auto_discover: %d bodies from MJCF scan", len(rows_in))

    out["rigid_body_map"] = rows_in
    return out


def apply_runtime_cloth_overrides(
    config: dict[str, Any],
    *,
    level: str | None = None,
    mjc_agent_prefix: str | None = None,
) -> dict[str, Any]:
    """
    将运行时关卡名、MuJoCo 机器人前缀写入配置（深拷贝）。

    关卡与机器人型号不写进 ``Config.json``，由 CLI / 环境变量注入。
    """
    out = copy.deepcopy(config)
    og = out.setdefault("orcagym", {})
    if level and str(level).strip():
        og["level"] = str(level).strip()
    prefix = (mjc_agent_prefix or "").strip()
    if prefix:
        og["mjc_agent_prefix"] = prefix
    return out


def apply_runtime_orcagym_level(config: dict[str, Any], level: str) -> dict[str, Any]:
    """将运行时关卡名写入 ``orcagym.level``（深拷贝，不改原 dict）。"""
    return apply_runtime_cloth_overrides(config, level=level)


def _ensure_level_scene_config(level: str, cloth_data_dir: Path | None = None) -> None:
    """解析关卡后自动同步本机 scene_levels（失败不阻断联调）。"""
    domain_dir = SOFTBODY_DOMAIN_DIR
    if str(domain_dir) not in sys.path:
        sys.path.insert(0, str(domain_dir))
    try:

        if ensure_level_scene_config(level, cloth_data_dir=cloth_data_dir):
            logger.debug("cloth scene config synced for level=%s", level)
    except Exception as exc:
        logger.debug("cloth scene auto-sync skipped: %s", exc)


def resolve_cloth_level(level: str | None = None, cloth_data_dir: Path | None = None) -> str:
    """委托 ``ProcessStudio.resolve_cloth_level_with_studio``，并自动同步本机关卡配置。"""
    from .ProcessStudio import resolve_cloth_level_with_studio

    resolved = resolve_cloth_level_with_studio(level)
    _ensure_level_scene_config(resolved, cloth_data_dir=cloth_data_dir)
    return resolved


def build_p2_session_from_mjcf(
    mjcf_path: Path,
    config_path: Path,
    *,
    session_timestamp: str | None = None,
    level: str | None = None,
    cloth_data_dir: Path | None = None,
    log_dir: Path | None = None,
) -> tuple[dict, dict, Path]:
    """
    从 Studio MJCF 扫描布片位姿（含 Entity3 旋转）并写出 XPBD session JSON。

    流程：``mj_forward`` → ``adapt_config_for_orcagym``（含 ``enrich_cloth_discovery_pose``）
    → ``build_xpbd_session_config`` → ``cloth_sim_session_*.json``。
    Studio 中旋转布料后须重新 Play 生成 MJCF，再调用本函数刷新 session。

    返回 ``(xpbd_session, adapted_config, session_path)``。
    """
    import mujoco

    if str(SOFTBODY_DIR) not in sys.path:
        sys.path.insert(0, str(SOFTBODY_DIR))

    base_cfg = load_cloth_config(config_path, cloth_data_dir=cloth_data_dir)
    if level:
        base_cfg = apply_runtime_orcagym_level(base_cfg, level)
    model = mujoco.MjModel.from_xml_path(str(mjcf_path.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    base_cfg, scene_assets = _resolve_cloth_from_model(model, data, base_cfg, cloth_data_dir=cloth_data_dir)
    base_cfg = _resolve_bodies_from_model(model, base_cfg)
    adapted = adapt_config_for_orcagym(model, base_cfg, data=data)
    xpbd_session = build_xpbd_session_config(base_cfg, adapted, cloth_data_dir=cloth_data_dir)
    xpbd_session.setdefault("mujoco", {})["model_path"] = str(mjcf_path.resolve())
    try:
        from .ProcessOrcaGym import (  # noqa: WPS433
            apply_tele_layout_to_session,
            scan_tele_layout_from_model,
        )

        tele = scan_tele_layout_from_model(model)
        xpbd_session = apply_tele_layout_to_session(xpbd_session, tele)
    except (ImportError, RuntimeError, KeyError, ValueError) as exc:
        logger.warning("tele_layout scan skipped: %s", exc)
    ts = session_timestamp or datetime.now().strftime("view_%Y%m%d_%H%M%S")
    session_path = write_xpbd_runtime_session_config(
        xpbd_session,
        session_timestamp=ts,
        source_config_path=config_path,
        source_mjcf_path=mjcf_path,
        log_dir=log_dir,
    )
    return xpbd_session, adapted, session_path


def body_track_position_packet_body_only(cfg: dict[str, Any]) -> bool:
    """
    是否采用 body_track 精简发包（每刚体 4 个 DataUnit：body_p/q/v/w）。

    当 body_track.enabled 且 use_anchor_sites 为 false 时返回 True；
    anchor_follow / 旧 phase1 路径仍发 12 unit/体（含 4×SITE）。
    """
    bt = cfg.get("body_track", {})
    if not bt.get("enabled", False):
        return False
    return not bool(bt.get("use_anchor_sites", False))


def frame_to_units(frame: AnchorFrame, *, body_only: bool = False) -> list[Any]:
    """
    构建 proto DataUnit 列表（延迟 import orcalink_pb2）。

    body_only=True（body_track）：每刚体 4 unit（body_p/q/v/w），不含 SITE/锚点速度。
    body_only=False（anchor_follow）：每刚体 12 unit（4×锚点 + body_*）。
    """
    from orcalink_client.protos import orcalink_pb2

    units: list[Any] = []
    for body in frame.bodies:
        ln = body.logical_name
        if not body_only:
            for i, anchor in enumerate(body.anchors):
                units.append(
                    orcalink_pb2.DataUnit(
                        object_id=f"{ln}_a{i}",
                        data_type=orcalink_pb2.DATA_TYPE_POSITION,
                        position=orcalink_pb2.PositionValue(
                            x=float(anchor.position[0]),
                            y=float(anchor.position[1]),
                            z=float(anchor.position[2]),
                            qw=1.0,
                            qx=0.0,
                            qy=0.0,
                            qz=0.0,
                        ),
                    )
                )
                units.append(
                    orcalink_pb2.DataUnit(
                        object_id=f"{ln}_a{i}_v",
                        data_type=orcalink_pb2.DATA_TYPE_VELOCITY,
                        velocity=orcalink_pb2.VelocityValue(
                            vx=float(anchor.linear_velocity[0]),
                            vy=float(anchor.linear_velocity[1]),
                            vz=float(anchor.linear_velocity[2]),
                            wx=0.0,
                            wy=0.0,
                            wz=0.0,
                        ),
                    )
                )
        units.append(
            orcalink_pb2.DataUnit(
                object_id=f"{ln}_body_q",
                data_type=orcalink_pb2.DATA_TYPE_POSITION,
                position=orcalink_pb2.PositionValue(
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    qw=float(body.quat_wxyz[0]),
                    qx=float(body.quat_wxyz[1]),
                    qy=float(body.quat_wxyz[2]),
                    qz=float(body.quat_wxyz[3]),
                ),
            )
        )
        units.append(
            orcalink_pb2.DataUnit(
                object_id=f"{ln}_body_p",
                data_type=orcalink_pb2.DATA_TYPE_POSITION,
                position=orcalink_pb2.PositionValue(
                    x=float(body.com_pos[0]),
                    y=float(body.com_pos[1]),
                    z=float(body.com_pos[2]),
                    qw=1.0,
                    qx=0.0,
                    qy=0.0,
                    qz=0.0,
                ),
            )
        )
        units.append(
            orcalink_pb2.DataUnit(
                object_id=f"{ln}_body_v",
                data_type=orcalink_pb2.DATA_TYPE_VELOCITY,
                velocity=orcalink_pb2.VelocityValue(
                    vx=float(body.com_linvel[0]),
                    vy=float(body.com_linvel[1]),
                    vz=float(body.com_linvel[2]),
                    wx=0.0,
                    wy=0.0,
                    wz=0.0,
                ),
            )
        )
        units.append(
            orcalink_pb2.DataUnit(
                object_id=f"{ln}_body_w",
                data_type=orcalink_pb2.DATA_TYPE_VELOCITY,
                velocity=orcalink_pb2.VelocityValue(
                    vx=0.0,
                    vy=0.0,
                    vz=0.0,
                    wx=float(body.ang_vel[0]),
                    wy=float(body.ang_vel[1]),
                    wz=float(body.ang_vel[2]),
                ),
            )
        )
    return units


def log_mujoco_send(frame: AnchorFrame) -> None:
    if not (os.environ.get("ORCALINK_DEBUG_ANCHOR") or os.environ.get("CLOTH_DEBUG_ANCHOR")):
        return
    print(
        f"[MUJOCO SEND] macro_frame={frame.macro_frame} sim_time={frame.sim_time:.4f} "
        f"bodies={len(frame.bodies)}",
        flush=True,
    )
    for body in frame.bodies:
        print(
            f"  body={body.logical_name} com={body.com_pos.tolist()} com_v={body.com_linvel.tolist()} "
            f"quat={body.quat_wxyz.tolist()} omega={body.ang_vel.tolist()}",
            flush=True,
        )
        for i, a in enumerate(body.anchors):
            print(
                f"    a{i} pos={a.position.tolist()} vel={a.linear_velocity.tolist()}",
                flush=True,
            )


def export_frame_jsonl(frame: AnchorFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "macro_frame": frame.macro_frame,
        "sim_time": frame.sim_time,
        "bodies": [
            {
                "logical_name": b.logical_name,
                "anchors": [
                    {"site": a.site_name, "pos": a.position.tolist(), "vel": a.linear_velocity.tolist()}
                    for a in b.anchors
                ],
                "com_pos": b.com_pos.tolist(),
                "com_linvel": b.com_linvel.tolist(),
                "quat_wxyz": b.quat_wxyz.tolist(),
                "ang_vel": b.ang_vel.tolist(),
            }
            for b in frame.bodies
        ],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class ClothOrcaLinkBridge:
    def __init__(self, config: dict[str, Any], model, data, pose_remapper=None, *, cloth_root: Path | None = None, orcalink_port: int) -> None:
        self._config = config
        self._model = model
        self._data = data
        self._ol = config["orcalink"]
        self._orcalink_port = orcalink_port
        from domain.body_map import load_body_map_ordered  # noqa: WPS433

        self._body_entries = load_body_map_ordered(model, config)
        self._pose_remapper = pose_remapper
        self._client = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = False
        dbg = config.get("debug", {})
        self._export_path = dbg.get("anchor_export_path")
        if not cloth_root:
            raise ValueError("ClothOrcaLinkBridge 需要 cloth_root（布料数据目录未指定）")
        self._cloth_root = Path(cloth_root).expanduser().resolve()

    @property
    def body_entries(self) -> list[BodyMapEntry]:
        return self._body_entries

    def bind_mujoco(self, model, data) -> bool:
        """
        绑定与 OrcaGym ``mj_step`` 一致的 ``MjModel`` / ``MjData``。

        ``SceneManager.publish_scene`` 会触发 ``init_env()`` 并重建 ``gym._mjData``；
        桥接必须在每宏步发布前指向当前实例，否则 ``sim_time`` 与刚体位姿会冻结在连接时刻。

        返回:
            bool: ``data`` 是否为与上次不同的新 ``MjData`` 对象。
        """
        changed = self._data is not data
        self._model = model
        self._data = data
        return changed

    def connect(self) -> bool:
        try:
            from orcalink_client import OrcaLinkClient
            from orcalink_client.config_loader import _build_orcalink_config_from_dict
            from orcalink_client.orcalink_client import setup_logging

            setup_logging()
        except ImportError as e:
            logger.error("orcalink_client not installed: %s", e)
            return False

        host = self._ol.get("host", "localhost")
        port = self._orcalink_port
        client_cfg = self._ol.get("client", {})
        session_cfg = dict(client_cfg.get("session", {}))
        if self._config.get("debug", {}).get("publish_only", False):
            session_cfg["expected_clients"] = 1
        pos_ch = client_cfg.get("channels", {}).get("position", {})
        force_ch = client_cfg.get("channels", {}).get("force", {})
        pos_id = int(pos_ch.get("channel_id", 20))
        force_id = int(force_ch.get("channel_id", 21))
        cfg_dict = {
            "orcalink_client": {
                "enabled": True,
                "server_address": f"{host}:{port}",
                "session_id": client_cfg.get("session_id", 101),
                "client_name": client_cfg.get("client_name", "cloth_mujoco"),
                "update_rate_hz": client_cfg.get("update_rate_hz", 50),
                "session": session_cfg,
            },
            "orcalink_bridge": {
                "coupling_mode": "force_position",
                "force_position": {
                    "channels": {
                        "force": {
                            "channel_id": force_id,
                            "publish": force_ch.get("publish", False),
                            "subscribe": force_ch.get("subscribe", True),
                        },
                        "position": {
                            "channel_id": pos_id,
                            "publish": pos_ch.get("publish", True),
                            "subscribe": pos_ch.get("subscribe", False),
                        },
                    },
                },
            },
        }
        config = _build_orcalink_config_from_dict(cfg_dict)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._client = OrcaLinkClient(config)
        ok = self._loop.run_until_complete(self._client.initialize())
        self._connected = bool(ok)
        if not self._connected:
            logger.error("OrcaLinkClient.initialize() 返回 False")
            return False
        logger.info(
            "ClothOrcaLinkBridge connected session_id=%s bodies=%d",
            client_cfg.get("session_id"),
            len(self._body_entries),
        )
        if self._sync_mode_active():
            async def _drain_stale_force(max_rounds: int = 64) -> None:
                for _ in range(max_rounds):
                    seq, _ = await self._client.subscribe_force_macro_frame(max_count=1)
                    if seq is None:
                        break

            self._loop.run_until_complete(_drain_stale_force())
            logger.info("sync: 已排空 ch21 残留 FORCE")
        return self._connected

    def should_pause(self) -> bool:
        if not self._client:
            return False
        if self._ol.get("client", {}).get("session", {}).get("control_mode") != "sync":
            return False
        return self._client.should_pause_this_cycle()

    def _sync_mode_active(self) -> bool:
        """sync 双向会话：发 POSITION 后须等 XPBD 回 FORCE 才能继续仿真。"""
        if not self._client:
            return False
        if self._ol.get("client", {}).get("session", {}).get("control_mode") != "sync":
            return False
        return bool(self._client.is_session_bidirectional)

    async def _wait_force_for_macro(
        self, macro_frame: int, timeout_sec: float = 120.0
    ) -> tuple[int | None, float]:
        """
        阻塞直到收到 sequence==macro_frame 的 FORCE，或超时。
        丢弃陈旧/乱序 FORCE，与 verify_vertex_pos_mjc_xpbd 一致。
        返回 (收到的 macro_frame 或 None, 墙钟 t4)。
        """
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            seq, t4 = await self._client.subscribe_force_macro_frame(max_count=1)
            if seq == macro_frame:
                return seq, t4
            if seq is not None:
                continue
            await asyncio.sleep(0.002)
        return None, 0.0

    def publish_anchor_macro_frame(self, macro_frame: int) -> bool:
        from domain.anchor_frame import collect_anchor_frame  # noqa: WPS433

        if not self._connected or not self._client or not self._loop:
            return False
        # 采集点 ①：MuJoCo 宏步边界物理量（Z-up）
        body_only = body_track_position_packet_body_only(self._config)
        frame = collect_anchor_frame(
            self._model, self._data, self._body_entries, macro_frame, skip_anchor_sites=body_only
        )
        if self._pose_remapper is not None and getattr(self._pose_remapper, "enabled", False):
            self._pose_remapper.apply_to_anchor_frame(frame)
        log_mujoco_send(frame)
        if self._export_path:
            export_frame_jsonl(frame, Path(self._export_path))
        # 采集点 ②：与 OrcaLink PublishFrame 一致的 DataUnit
        units = frame_to_units(frame, body_only=body_only)
        ok = self._loop.run_until_complete(
            self._client.publish_anchor_frame(units, macro_frame, frame.sim_time)
        )
        if not ok:
            return False

        if self._sync_mode_active():
            force_seq, _ = self._loop.run_until_complete(
                self._wait_force_for_macro(macro_frame)
            )
            if force_seq != macro_frame:
                logger.error(
                    "sync: 未收到 FORCE macro_frame=%s (got=%s)",
                    macro_frame,
                    force_seq,
                )
                return False

        return True

    def close(self) -> None:
        if self._client and self._loop:
            try:
                self._loop.run_until_complete(self._client.shutdown())
            except Exception:
                pass
        self._connected = False
        if self._loop:
            self._loop.close()
        self._loop = None
        self._client = None


def _connect_cloth_bridge(
    env: OrcaGymLocalEnv,
    config: dict[str, Any],
    ctx: ClothCouplingContext,
    *,
    adapted: dict[str, Any] | None = None,
    cloth_data_dir: Path | None = None,
    orcalink_port: int,
) -> bool:
    _ensure_domain_import_path()
    from domain.body_map import load_body_map_ordered  # noqa: WPS433

    model, data = get_mujoco_model_data(env)
    if adapted is None:
        config, _ = _resolve_cloth_from_model(model, data, config, cloth_data_dir=cloth_data_dir)
        config = _resolve_bodies_from_model(model, config)
        adapted = adapt_config_for_orcagym(model, config, data=data)
    publish_entries = load_body_map_ordered(model, adapted)
    errs = validate_orcagym_body_map(model, publish_entries)
    if errs:
        for e in errs:
            logger.error("body_map: %s", e)
        return False
    if not publish_entries:
        logger.error("无 OrcaLink 发布刚体；请检查 orcagym_rigid_body_map")
        return False

    remap_params = resolve_pose_remap(adapted)
    remapper = OrcaLinkPoseRemapper(
        remap_params["ref_yup"],
        enabled=remap_params["enabled"],
        logical_names=remap_params["logical_names"],
    )
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    base_env.mj_forward()
    sync_gripper_mocap_from_bodies(base_env, model, data, adapted)
    base_env.mj_forward()
    remapper.calibrate(model, data, publish_entries)

    logger.info(
        "OrcaLink 发布 %d 刚体（掌+指）: %s",
        len(publish_entries),
        [e.logical_name for e in publish_entries],
    )
    bridge = ClothOrcaLinkBridge(adapted, model, data, pose_remapper=remapper, cloth_root=cloth_data_dir, orcalink_port=orcalink_port)
    if not adapted.get("orcalink", {}).get("enabled", True):
        logger.warning("orcalink.enabled=false，跳过连接")
        ctx.bridge = bridge
        return True
    if not bridge.connect():
        logger.error("ClothOrcaLinkBridge 连接失败（请先启动 OrcaLink Server 与 XPBD）")
        return False
    ctx.bridge = bridge
    return True


def start_cloth_coupling(
    env: OrcaGymLocalEnv,
    config: Dict[str, Any],
    *,
    config_path: Optional[str | Path] = None,
    log_dir: Optional[str | Path] = None,
    auto_start_orcalink: Optional[bool] = None,
    auto_start_xpbd: Optional[bool] = None,
    cpu_affinity: Optional[str] = None,
    cloth_data_dir: Path | None = None,
) -> ClothCouplingHandle:
    """
    在已有 OrcaGym 环境上启动 OrcaLink / XPBD，并连接布料发布桥。

    典型顺序（与 MjcPBD 双进程一致）：
    1. OrcaLink Server（可选 auto_start）
    2. XPBD dual_gripper_cross_mjc（可选 auto_start，须先于 MuJoCo 凑齐 expected_clients）
    3. ClothOrcaLinkBridge 连接并等待 session_ready
    """
    _ensure_domain_import_path()

    cfg = copy.deepcopy(config)
    orcalink_port = require_orcalink_port(cfg.get("orcalink", {}))
    path = Path(config_path or default_cloth_config_path(cloth_data_dir=cloth_data_dir)).resolve()
    base_cfg = load_cloth_config(path, cloth_data_dir=cloth_data_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if auto_start_orcalink is not None:
        cfg.setdefault("orcalink", {})["auto_start"] = bool(auto_start_orcalink)
    if auto_start_xpbd is not None:
        cfg.setdefault("xpbd", {})["auto_start"] = bool(auto_start_xpbd)

    og = cfg.setdefault("orcagym", {})
    resolved_level = resolve_cloth_level(str(og.get("level") or "").strip() or None, cloth_data_dir=cloth_data_dir)
    og["level"] = resolved_level

    ctx = ClothCouplingContext(config=cfg, config_path=path, env=env, session_timestamp=ts)

    model, data = get_mujoco_model_data(env)
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    base_env.mj_forward()

    from .ProcessStudio import run_masked_vtk_prefab_check_at_startup

    cfg, scene_assets = _resolve_cloth_from_model(model, data, cfg, cloth_data_dir=cloth_data_dir)
    if not run_masked_vtk_prefab_check_at_startup(cfg, scene_assets=scene_assets):
        import logging
        logging.getLogger(__name__).warning(
            "掩码 VTK 预制检查未通过（将继续运行，仅 VTK 加载不受影响）。"
            "跳过: CLOTH_SKIP_MASKED_PREFAB_CHECK=1"
        )

    cfg = _resolve_bodies_from_model(model, cfg)
    adapted = adapt_config_for_orcagym(model, cfg, data=data)
    xpbd_session_cfg = build_xpbd_session_config(base_cfg, adapted, cloth_data_dir=cloth_data_dir)
    mjcf_path = get_mujoco_xml_path(env)
    xpbd_session_cfg.setdefault("mujoco", {})["model_path"] = str(mjcf_path)
    session_path = write_xpbd_runtime_session_config(
        xpbd_session_cfg,
        session_timestamp=ts,
        source_config_path=path,
        source_mjcf_path=mjcf_path,
        log_dir=log_dir,
    )
    ctx.config = adapted
    ctx.config_path = session_path
    run_log_dir = session_path.parent

    discover_only = bool((xpbd_session_cfg.get("xpbd") or {}).get("cloth_discover_only", True))
    if not discover_only:
        try:
            scene_path = export_xpbd_scene_for_session(session_path, cloth_data_dir=cloth_data_dir)
            logger.info("XPBD scene exported: %s", scene_path)
        except Exception as exc:
            ctx.process_manager.cleanup_all()
            raise RuntimeError(f"XPBD 场景导出失败: {exc}") from exc

    logger.info("=" * 60)
    logger.info("布料 MjcPBD 耦合挂载（OrcaGym → OrcaLink → XPBD → Studio）")
    logger.info("config: %s", path)
    logger.info("XPBD MJC_PBD_CONFIG: %s", session_path)
    cloth_blk = xpbd_session_cfg.get("cloth") or {}
    if cloth_blk.get("discovered"):
        logger.info(
            "cloth discovered: mesh=%s asset_dir=%s center_yup=%s quat_wxyz_yup=%s",
            cloth_blk.get("mesh"),
            cloth_blk.get("asset_dir"),
            cloth_blk.get("center_yup"),
            cloth_blk.get("quat_wxyz_yup"),
        )
    logger.info("=" * 60)

    start_orcalink_if_configured(
        cfg,
        process_manager=ctx.process_manager,
        log_dir=run_log_dir,
        session_timestamp=ts,
        force_restart=True,
        orcalink_port=orcalink_port,
    )
    xpbd_dg_traj = str((adapted.get("xpbd") or {}).get("dg_traj", "")).strip()
    if xpbd_dg_traj == "pico":
        trigger_path = run_log_dir / "grip_triggers.txt"
        os.environ["MJC_PBD_GRIP_TRIGGER_PATH"] = str(trigger_path)
        logger.info("PICO grip triggers → %s (MJC_PBD_DG_TRAJ=pico)", trigger_path)
    start_xpbd_if_configured(
        adapted,
        config_path=session_path,
        process_manager=ctx.process_manager,
        log_dir=run_log_dir,
        session_timestamp=ts,
        cpu_affinity=cpu_affinity,
        cloth_data_dir=cloth_data_dir,
    )

    if not _connect_cloth_bridge(env, cfg, ctx, adapted=adapted, cloth_data_dir=cloth_data_dir, orcalink_port=orcalink_port):
        ctx.process_manager.cleanup_all()
        raise RuntimeError("布料 OrcaLink 桥接初始化失败")

    handle = ClothCouplingHandle(config=adapted, ctx=ctx, enabled=True)
    return handle


def _pids_on_tcp_port(port: int) -> list[int]:
    import re
    import subprocess

    try:
        out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return []
    pids: set[int] = set()
    for line in out.stdout.splitlines():
        if f":{port}" in line:
            for m in re.finditer(r"pid=(\d+)", line):
                pids.add(int(m.group(1)))
    return sorted(pids)


def _kill_stale_cloth_processes(orcalink_port: int, pico_port: int) -> None:
    from .ProcessXPBD import kill_pids_gracefully

    orca_pids = _pids_on_tcp_port(orcalink_port)
    pico_pids = _pids_on_tcp_port(pico_port)

    if not (orca_pids or pico_pids):
        logger.info(f"无陈旧 cloth 联调进程（:{orcalink_port} / :{pico_port}）")
        return

    logger.info(f"清理陈旧 cloth 联调进程（OrcaLink :{orcalink_port}、Pico :{pico_port}）...")
    kill_pids_gracefully(f"orcalink(:{orcalink_port})", _pids_on_tcp_port(orcalink_port))
    kill_pids_gracefully(f"Pico(:{pico_port})", _pids_on_tcp_port(pico_port))

    time.sleep(1)
    if _pids_on_tcp_port(orcalink_port) or _pids_on_tcp_port(pico_port):
        logger.info(
            f"WARN: :{orcalink_port} 或 :{pico_port} 仍被占用，请手动检查: "
            f"ss -tlnp | grep -E '{orcalink_port}|{pico_port}'"
        )
    else:
        logger.info(f"陈旧进程已清理，:{orcalink_port} / :{pico_port} 已释放")


def _wait_port(port: int, label: str, max_sec: int) -> bool:
    import socket

    logger.info(f"等待 {label} localhost:{port} (最多 {max_sec}s)...")
    waited = 0
    while waited < max_sec:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                logger.info(f"OK {label} :{port}")
                return True
        except OSError:
            pass
        time.sleep(2)
        waited += 2
    logger.info(f"超时：{label} :{port} 未监听")
    return False


@dataclass
class P23cParams:
    """P23c 联调运行时参数（RunSim 组装，run_p23c 消费）。"""

    repo_root: Path
    base_dir: Path
    log_dir: Path
    level: str | None
    agent: str
    mjc_prefix: str
    config_path: str
    task_config: str | None = None
    cloth_data_dir: Path | None = None
    orcagym_port: int = 50051
    pbd_grpc_port: int = 50263
    orcalink_port: int = 50361
    pico_port: int = 8001
    wait_sec: int = 180
    kill_stale: bool = True
    auto_start_studio: bool = False
    collect_data: bool = False
    cloth_auto_start_orcalink: bool = False
    cloth_auto_start_xpbd: bool = False
    xpbd_ui: bool = True
    cloth_sync_studio_vis: bool = True
    cloth_no_realtime: bool = False
    mujoco_viewer: bool = False
    gripper_trace: bool = False
    pico_delta_trace: bool = False
    frame_skip: int = 20
    time_step: float = 0.001
    max_macro_frames: int | None = None
    max_sec: int = 120
    xpbd_auto_build: bool = True
    xpbd_build_target: str = ""
    agent_user_set: bool = False
    config_explicit: bool = False
    bench_json: str = ""


def _assemble_agent(
    agent_name: str,
    level: str,
    collect_data: bool,
    base_dir: Path,
) -> tuple[Any, Any, dict[str, Any], Any]:
    """组装机器人配置：加载 agent_conf + 建 data_storage + 算 default_joint_values + obs_callback。"""
    import numpy as np

    default_joint_values: dict[str, Any] = {}

    if agent_name == "openloong":
        from conf import openloong_conf as agent_conf

        if not collect_data:
            data_storage = None

            def obs_callback(_env):
                return {"bench_dummy": np.zeros(1, dtype=np.float32)}
        else:
            from dataStorage.openloong_data_storage import OpenLoongDataStorage

            data_storage = OpenLoongDataStorage(
                dataset_path=os.path.join(str(base_dir), "dataset", agent_name, level),
                hdf5_path="record/proprio_stats.hdf5",
            )
            obs_callback = data_storage.obs_callback
    elif agent_name == "tiangong2":
        from conf import tiangong2_conf as agent_conf

        if not collect_data:
            data_storage = None

            def obs_callback(_env):
                return {"bench_dummy": np.zeros(1, dtype=np.float32)}
        else:
            from dataStorage.tiangong_data_storage import Tiangong2DataStorage

            data_storage = Tiangong2DataStorage(
                dataset_path=os.path.join(str(base_dir), "dataset", agent_name, level),
                hdf5_path="record/proprio_stats.hdf5",
            )
            obs_callback = data_storage.obs_callback
    elif agent_name == "g1_omnipicker":
        from conf import g1_omnipicker_conf as agent_conf

        # G1 尚无专用 HDF5 storage；OpenLoongDataStorage 夹爪 actuator 名与 G1 MJCF 不兼容
        data_storage = None

        def obs_callback(_env):
            return {"bench_dummy": np.zeros(1, dtype=np.float32)}

        if not collect_data:
            logger.info("No-collect mode: skip dataset/HDF5")
        else:
            logger.warning(
                "G1 cloth tele: 暂无 G1 dataset 存储，使用 bench_dummy obs（跳过 HDF5 采集）"
            )
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")

    for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value

    return agent_conf, data_storage, default_joint_values, obs_callback


def _build_env(
    params: P23cParams,
    agent_conf: Any,
    data_storage: Any,
    default_joint_values: dict[str, Any],
    obs_callback: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """建 Pico 输入 + SceneManager + DataCollectionManager（MuJoCo env）+ env.reset + 可选 --gui。"""
    import numpy as np

    from dataCollectionManager.data_collection_manager import DataCollectionManager
    from devices.abstract_device import PicoJoystickDevice
    from orca_gym.devices.pico_joytsick import PicoJoystick
    from scene.scene_manager import SceneManager
    from scene.scene_config_util import load_scene_config

    pico_joystick = PicoJoystick()
    pico_joystick_device = PicoJoystickDevice(pico_joystick)

    config = load_scene_config(params.base_dir, params.task_config)
    scene_manager = SceneManager(f"localhost:{params.orcagym_port}", config=config)

    if params.collect_data:
        scene_manager.show_ui_message(1, "布料遥操采集，请操作手柄", "0xffff00", showtime=10)
        scene_manager.get_scene_data("RunSim", "beginscene")

    if data_storage is not None:
        data_storage.set_video_path("video")

    max_episode_steps = np.iinfo(np.int64).max
    if params.max_macro_frames is not None:
        max_episode_steps = max(1, int(params.max_macro_frames))
    elif params.max_sec is not None:
        max_episode_steps = int(params.max_sec / (params.time_step * params.frame_skip)) + 1

    mjc_prefix = (params.mjc_prefix or "").strip() or None
    if mjc_prefix is None and params.agent == "g1_omnipicker":
        mjc_prefix = "g1_omnipicker"
    elif mjc_prefix is None and params.level == "test20260508" and params.agent == "openloong":
        mjc_prefix = "openloong_gripper_2f85_fix_base_usda"

    data_collection_manager = DataCollectionManager(
        agent_name=params.agent,
        env_name="DataCollection",
        entry_point="envs.dataCollection.dataCollection_env:DataCollectionEnv",
        default_joint_values=default_joint_values,
        obs_callback=obs_callback,
        env_index=0,
        max_episode_steps=max_episode_steps,
        device=pico_joystick_device,
        scene_manager=scene_manager,
        data_storage=data_storage,
        frame_skip=params.frame_skip,
        time_step=params.time_step,
        mjc_agent_prefix=mjc_prefix,
    )
    env = data_collection_manager.env
    env.reset()

    _gui_viewer = None
    if params.mujoco_viewer:
        import mujoco
        import mujoco.viewer as mjv

        m = env.gym._mjModel
        d = env.gym._mjData
        _gui_viewer = mjv.launch_passive(m, d)
        positions = [
            d.xpos[i].copy()
            for i in range(1, m.nbody)
            if "dummy" not in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()
        ]
        if positions:
            arr = np.asarray(positions)
            center = arr.mean(axis=0)
            extent = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
            with _gui_viewer.lock():
                _gui_viewer.cam.lookat[:] = center
                _gui_viewer.cam.distance = max(4.0, extent * 2.2)
                _gui_viewer.cam.elevation = -25.0
                _gui_viewer.cam.azimuth = 135.0
                for g in range(6):
                    _gui_viewer.opt.geomgroup[g] = True

        def _sync_gui_viewer(_env):
            if _gui_viewer is not None and _gui_viewer.is_running():
                m = _env.gym._mjModel
                d = _env.gym._mjData
                mujoco.mj_forward(m, d)
                _gui_viewer.sync()

        data_collection_manager.add_post_step_callback(_sync_gui_viewer)

    return data_collection_manager, env, pico_joystick, pico_joystick_device, scene_manager, config


def _mount_cloth(
    params: P23cParams,
    env: Any,
    data_collection_manager: Any,
    pico_joystick: Any,
    level: str,
    agent: str,
    mjc_prefix: str | None,
) -> None:
    """挂布料耦合：load config + apply overrides + start_cloth_coupling + 读 Pico 扳机。"""
    cloth_data_dir = params.cloth_data_dir
    cloth_cfg_path = params.config_path
    if not cloth_cfg_path:
        cloth_cfg_path = str(default_cloth_config_path(cloth_data_dir=cloth_data_dir))
    cloth_config = load_cloth_config(cloth_cfg_path, cloth_data_dir=cloth_data_dir)
    cloth_config = apply_runtime_cloth_overrides(
        cloth_config,
        level=level,
        mjc_agent_prefix=mjc_prefix,
    )
    if agent == "openloong" and mjc_prefix:
        sync = cloth_config.setdefault("orcagym", {}).setdefault("sync_mocap_from_gripper", {})
        sync["enabled"] = True
        sync["pairs"] = [
            {"mocap_body": f"{mjc_prefix}_leftHandMocap", "palm_body": f"{mjc_prefix}_zbll_base_link"},
            {"mocap_body": f"{mjc_prefix}_rightHandMocap", "palm_body": f"{mjc_prefix}_zbr_base_link"},
        ]
    cloth_config.setdefault("xpbd", {})["dg_traj"] = "pico"

    mj_fs = int(cloth_config.get("mujoco", {}).get("frame_skip", 20))
    if mj_fs != params.frame_skip:
        logger.warning(
            f"cloth frame_skip={mj_fs} 与 frame_skip={params.frame_skip} 不一致；建议对齐 dual_gripper_cross_full"
        )

    cloth_handle = start_cloth_coupling(
        env,
        cloth_config,
        config_path=cloth_cfg_path,
        log_dir=params.log_dir,
        auto_start_orcalink=True if params.cloth_auto_start_orcalink else None,
        auto_start_xpbd=True if params.cloth_auto_start_xpbd else None,
        cloth_data_dir=cloth_data_dir,
    )
    data_collection_manager.set_cloth_coupling(cloth_handle)
    trigger_path = cloth_handle.ctx.config_path.parent / "grip_triggers.txt"

    def _read_pico_triggers() -> tuple[float, float]:
        ks = pico_joystick.get_key_state()
        if not ks:
            return 0.0, 0.0
        return (float(ks["leftHand"]["triggerValue"]), float(ks["rightHand"]["triggerValue"]))

    cloth_handle.set_grip_trigger_provider(_read_pico_triggers, trigger_path)
    data_collection_manager.set_skip_render(True)


def run_p23c(params: P23cParams) -> int:
    """P23c 完整编排：准备 → 组装 → 建 env → 挂耦合 → 装配 → 跑循环。"""
    from .base.paths import resolve_cloth_config_path
    from .ProcessOrcaGym import scan_tele_layout_from_mjcf
    from .ProcessStudio import ensure_ready, find_latest_studio_mjcf_path
    from .ProcessXPBD import cleanup, prepare

    # 下游环境变量
    os.environ["PYTHONPATH"] = (
        f"{params.repo_root / 'OrcaGym'}:"
        f"{params.repo_root / 'OrcaManipulation' / 'src'}:"
        + os.environ.get("PYTHONPATH", "")
    )
    os.environ["PBD_GRPC_ADDRESS"] = (
        os.environ.get("PBD_GRPC_ADDRESS", "").strip() or f"localhost:{params.pbd_grpc_port}"
    )
    os.environ["XPBD_UI"] = "1" if params.xpbd_ui else "0"
    os.environ.setdefault("PBDX_FORCE_GS_ONLY", "1")
    os.environ.setdefault("PBDX_SOLVER", "gs")

    # 关卡 / 配置解析
    cloth_data_dir = params.cloth_data_dir
    level = resolve_cloth_level(params.level, cloth_data_dir=cloth_data_dir)
    config_path = params.config_path
    if not config_path:
        config_path = str(resolve_cloth_config_path(level=level, agent=params.agent, cloth_data_dir=cloth_data_dir))
    agent = params.agent
    mjc_prefix = params.mjc_prefix

    # MJCF 扫描：检测 agent / mjc_prefix
    mjcf = find_latest_studio_mjcf_path()
    if mjcf is not None and mjcf.is_file():
        try:
            layout = scan_tele_layout_from_mjcf(mjcf)
            detected_agent = getattr(layout, "tele_agent_name", "") or ""
            detected_prefix = getattr(layout, "mjc_agent_prefix", "") or ""
        except Exception:
            detected_agent = detected_prefix = ""
        if not params.agent_user_set:
            if detected_agent and detected_agent != agent:
                logger.info(f"MJCF 扫描 tele_agent={detected_agent}（覆盖默认 openloong）")
                agent, mjc_prefix = detected_agent, detected_prefix
                if not params.config_explicit:
                    config_path = str(resolve_cloth_config_path(level=level, agent=agent, cloth_data_dir=cloth_data_dir))
        elif detected_prefix and detected_prefix != mjc_prefix:
            mjc_prefix = detected_prefix

    os.environ["LEVEL"] = level or ""
    os.environ["AGENT"] = agent
    os.environ["MJC_PREFIX"] = mjc_prefix
    os.environ["CFG"] = config_path

    if params.cloth_sync_studio_vis:
        os.environ["CLOTH_SYNC_STUDIO_VIS"] = "1"
        os.environ.setdefault("CLOTH_STUDIO_VIS_STRIDE", "1")
    if params.xpbd_ui:
        os.environ.pop("MJC_PBD_NO_UI", None)
        os.environ.setdefault("DISPLAY", ":0")
    else:
        os.environ["MJC_PBD_NO_UI"] = "1"

    # XPBD 二进制（清旧 + 准备）
    if params.xpbd_auto_build:
        # 默认 target/version 从 Config.json 读（单一来源）；环境变量 / 参数可覆盖
        _cfg = load_cloth_config(config_path, cloth_data_dir=cloth_data_dir)
        default_target = str(_cfg.get("xpbd_default_target") or "").strip()
        default_version = str(_cfg.get("xpbd_default_version") or "").strip()
        target = params.xpbd_build_target or default_target
        os.environ["XPBD_BUILD_TARGET"] = target
        os.environ.setdefault("ORCA_XPBD_VERSION", default_version)
        cleanup(target)
        if prepare(target) != 0:
            logger.error("XPBD 准备失败")
            return 1

    # 进程 / 端口就绪
    if params.kill_stale:
        _kill_stale_cloth_processes(params.orcalink_port, params.pico_port)
    else:
        logger.info("KILL_STALE=0：跳过陈旧进程清理")
    ensure_ready(params.auto_start_studio)
    _wait_port(params.orcagym_port, "OrcaGym", params.wait_sec)
    _wait_port(params.pbd_grpc_port, "PBDRender", params.wait_sec)

    # 同步 XPBD session
    if mjcf is None or not mjcf.is_file():
        logger.error(f"MJCF not found; Studio Play {level} first")
        return 2
    ts = f"p23c_{time.strftime('%Y%m%d_%H%M%S')}"
    _, _, session_path = build_p2_session_from_mjcf(
        mjcf, Path(config_path), session_timestamp=ts, level=level, cloth_data_dir=cloth_data_dir,
        log_dir=params.log_dir,
    )
    export_xpbd_scene_for_session(session_path, cloth_data_dir=cloth_data_dir)

    # 组装 → 建 env → 挂耦合
    agent_conf, data_storage, default_joint_values, obs_callback = _assemble_agent(
        agent, level, params.collect_data, params.base_dir
    )
    data_collection_manager, env, pico_joystick, pico_joystick_device, scene_manager, config = _build_env(
        params, agent_conf, data_storage, default_joint_values, obs_callback
    )
    _mount_cloth(params, env, data_collection_manager, pico_joystick, level, agent, mjc_prefix)

    # 遥操装配 + 任务 + 跑循环
    if params.bench_json:
        data_collection_manager.enable_bench(params.bench_json)

    setup_teleop_controllers(agent, data_collection_manager, env, agent_conf, pico_joystick_device)

    from scene.scene_config_util import create_task, should_use_empty_task

    data_collection_manager.set_task(create_task(env, config, params.task_config))
    from controllers import controllers

    controllers.add_task_status_pico_controller(
        data_collection_manager, env, pico_joystick_device, agent_conf.base_body,
    )

    data_collection_manager.save_video = params.collect_data
    if params.cloth_no_realtime:
        data_collection_manager.set_realtime_sync(False)

    if os.environ.get("CLOTH_CAMERA_MONITOR", "").strip().lower() in ("1", "true", "yes"):
        for port in (7080, 7081, 7090, 7091):
            data_collection_manager.add_monitor_port(port)

    if params.pico_delta_trace:
        from .ProcessPico import attach_pico_mjc_delta_tracer

        delta_csv = params.log_dir / "pico_mjc_delta_trace.csv"
        palm_l = palm_r = None
        if agent == "g1_omnipicker":
            palm_l, palm_r = "arm_l_end_link", "arm_r_end_link"
        attach_pico_mjc_delta_tracer(
            data_collection_manager,
            env,
            pico_joystick,
            agent_conf.base_body,
            agent_conf.l_arm,
            agent_conf.r_arm,
            delta_csv,
            arm_controllers=data_collection_manager.controllers,
            palm_l_body=palm_l,
            palm_r_body=palm_r,
        )

    data_collection_manager.run(
        max_episodes=1 if (params.max_macro_frames is not None or params.max_sec is not None) else None
    )
    return 0


def setup_teleop_controllers(
    agent_name: str,
    data_collection_manager: Any,
    env: Any,
    agent_conf: Any,
    pico_joystick_device: Any,
) -> None:
    """给指定机器人装配遥操控制器/执行器（P_arm 断开 + 夹爪 + 手臂）。"""
    from controllers import controllers
    from orca_gym.devices.pico_joytsick import PicoJoystickKey

    # 1) openloong P_arm 双控断开 / OSC 执行器
    if agent_name == "openloong":
        from .ProcessOrcaGym import setup_openloong_dual_arm_osc_actuators

        def _bind_osc_actuators() -> None:
            setup_openloong_dual_arm_osc_actuators(env, agent_conf.l_arm, agent_conf.r_arm)

        _bind_osc_actuators()
        data_collection_manager.add_physics_reinit_callback(_bind_osc_actuators)
        # 保留：若 MJCF 将 P_arm 置于 group 1 时仍生效；Studio 实场景全为 group 0，主要靠 trnid 断开
        data_collection_manager.set_disable_actuator_group([agent_conf.positions_group])
    else:
        logger.info(
            f"Skip openloong P_arm detach for agent={agent_name} "
            "(G1 仅 mctrl/pctrl，无 P_arm 双控)"
        )

    # 2) 夹爪控制器
    if agent_name == "g1_omnipicker":
        # 使用 dev_g1 的 reverse + joint2；不再做 joint1 从动断开
        logger.info("Creating left G1 reverse gripper controller")
        controllers.add_gripper_2f85_reverse_pico_controller(
            data_collection_manager,
            env,
            agent_conf.gripper_l,
            agent_conf.base_body,
            pico_joystick_device,
            [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
        )
        logger.info("Creating right G1 reverse gripper controller")
        controllers.add_gripper_2f85_reverse_pico_controller(
            data_collection_manager,
            env,
            agent_conf.gripper_r,
            agent_conf.base_body,
            pico_joystick_device,
            [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
        )
    else:
        logger.info("Creating left gripper controller")
        controllers.add_gripper_2f85_pico_controller(
            data_collection_manager, env, agent_conf.gripper_l, agent_conf.base_body,
            pico_joystick_device, [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
        )
        logger.info("Creating right gripper controller")
        controllers.add_gripper_2f85_pico_controller(
            data_collection_manager, env, agent_conf.gripper_r, agent_conf.base_body,
            pico_joystick_device, [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
        )

    # 3) 手臂控制器
    if agent_name == "g1_omnipicker":
        from controllers.g1_arm_pico_remap import (
            G1_ARM_POSITION_REMAP,
            G1_L_ARM_POSITION_FLIP,
            G1_L_ARM_ROTATION_OFFSET,
            G1_R_ARM_POSITION_FLIP,
            G1_R_ARM_ROTATION_OFFSET,
            add_g1_arm_osc_pico_controller,
        )

        logger.info("Creating left G1 arm controller (Pico remap)")
        add_g1_arm_osc_pico_controller(
            data_collection_manager,
            env,
            agent_conf.l_arm,
            agent_conf.base_body,
            pico_joystick_device,
            PicoJoystickKey.L_TRANSFORM,
            G1_L_ARM_ROTATION_OFFSET,
            G1_ARM_POSITION_REMAP,
            G1_L_ARM_POSITION_FLIP,
        )
        logger.info("Creating right G1 arm controller (Pico remap)")
        add_g1_arm_osc_pico_controller(
            data_collection_manager,
            env,
            agent_conf.r_arm,
            agent_conf.base_body,
            pico_joystick_device,
            PicoJoystickKey.R_TRANSFORM,
            G1_R_ARM_ROTATION_OFFSET,
            G1_ARM_POSITION_REMAP,
            G1_R_ARM_POSITION_FLIP,
        )
    else:
        logger.info("Creating left arm controller")
        controllers.add_arm_osc_pico_controller(
            data_collection_manager, env, agent_conf.l_arm, agent_conf.base_body,
            pico_joystick_device, PicoJoystickKey.L_TRANSFORM,
        )
        logger.info("Creating right arm controller")
        controllers.add_arm_osc_pico_controller(
            data_collection_manager, env, agent_conf.r_arm, agent_conf.base_body,
            pico_joystick_device, PicoJoystickKey.R_TRANSFORM,
        )


# ---------------------------------------------------------------------------
# 场景配置：模板 + scene_levels.json 同步（从 domain/scene_cloth_config 迁入）
# ---------------------------------------------------------------------------

_TEMPLATE_BASENAME = "Config.json"
_LEVELS_BASENAME_DEFAULT = "scene_levels.json"
_GENERATOR_NAME = "scene_cloth_config"


def template_config_path(cloth_data_dir: Path | None = None) -> Path:
    """仓库内通用模板 ``Config.json``（``cloth_data_dir`` 指定数据目录）。"""
    override = os.environ.get("CLOTH_SCENE_ASSETS_CONFIG", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if not cloth_data_dir:
        raise ValueError("template_config_path 需要 cloth_data_dir（布料数据目录未指定）")
    return (Path(cloth_data_dir).expanduser().resolve() / _TEMPLATE_BASENAME).resolve()


def orcagym_cloth_config_dir(cfg: dict[str, Any] | None = None) -> Path:
    """
    本机场景级布料配置目录，默认 ``~/.orcagym/cloth``。

    可用 ``ORCA_CLOTH_CONFIG_DIR`` 或模板 ``generation.orcagym_config_dir`` 覆盖。
    """
    cfg = cfg or load_template_config()
    gen = cfg.get("generation") or {}
    env_name = str(gen.get("orcagym_config_dir_env") or "ORCA_CLOTH_CONFIG_DIR")
    env_val = os.environ.get(env_name, "").strip()
    if env_val:
        return Path(env_val).expanduser().resolve()
    raw = str(gen.get("orcagym_config_dir") or "~/.orcagym/cloth")
    return Path(raw).expanduser().resolve()


def generated_levels_config_path(cfg: dict[str, Any] | None = None) -> Path:
    """本机生成的 ``scene_levels.json`` 路径。"""
    cfg = cfg or load_template_config()
    gen = cfg.get("generation") or {}
    env_name = str(gen.get("levels_config_env") or "CLOTH_SCENE_LEVELS_CONFIG")
    env_val = os.environ.get(env_name, "").strip()
    if env_val:
        return Path(env_val).expanduser().resolve()
    basename = str(gen.get("levels_basename") or _LEVELS_BASENAME_DEFAULT)
    return (orcagym_cloth_config_dir(cfg) / basename).resolve()


def load_template_config(cloth_data_dir: Path | None = None) -> dict[str, Any]:
    """读取仓库通用模板（无 ``levels`` 或 ``levels`` 为空）。"""
    path = template_config_path(cloth_data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"cloth scene template not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid template root in {path}")
    return data


def load_generated_levels_document(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """读取 ``~/.orcagym/cloth/scene_levels.json``；不存在时返回空文档。"""
    path = generated_levels_config_path(cfg)
    if not path.is_file():
        return {"schema_version": 1, "levels": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid generated levels root in {path}")
    return data


def is_auto_scene_sync_enabled() -> bool:
    """是否自动扫描并写入 ``~/.orcagym/cloth/scene_levels.json``（``CLOTH_NO_AUTO_SCENE_SYNC=1`` 关闭）。"""
    return os.environ.get("CLOTH_NO_AUTO_SCENE_SYNC", "0").strip().lower() not in (
        "1",
        "true",
        "yes",
    )


def _level_inputs_mtime(level: str, cfg: dict[str, Any] | None = None) -> float:
    """关卡 prefab 与 ``Assets/<level>/`` 布料相关文件的最大 mtime，用于判断是否需要重新扫描。"""
    cfg = cfg or load_template_config()
    mtimes: list[float] = []
    prefab = resolve_level_prefab_path(level, cfg)
    if prefab is not None and prefab.is_file():
        mtimes.append(prefab.stat().st_mtime)
    asset_dir = level_assets_dir(level, cfg)
    if asset_dir.is_dir():
        for pattern in ("*.vtk", "*.mask", "*.meta.json", "*.fbx"):
            for path in asset_dir.glob(pattern):
                if path.is_file():
                    mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def level_entry_is_stale(level: str, entry: dict[str, Any], cfg: dict[str, Any] | None = None) -> bool:
    """本机条目是否落后于 prefab / 资产目录修改时间。"""
    synced_at = str(entry.get("synced_at") or "").strip()
    if not synced_at:
        return True
    try:
        sync_ts = datetime.fromisoformat(synced_at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return True
    return _level_inputs_mtime(level, cfg) > sync_ts + 1.0


def load_scene_assets_config(
    *,
    sync_level: str | None = None,
    ensure_level: str | None = None,
    cloth_data_dir: Path | None = None,
) -> dict[str, Any]:
    """
    合并通用模板与本机生成的 ``levels`` 表。

    ``sync_level`` / ``ensure_level`` 非空时，先确保该关卡已扫描写入 ``~/.orcagym/cloth/scene_levels.json``。
    """
    template = load_template_config(cloth_data_dir)
    for name in (sync_level, ensure_level):
        if name and str(name).strip():
            ensure_level_scene_config(str(name).strip(), template=template)

    generated = load_generated_levels_document(template)
    merged = dict(template)
    merged["levels"] = dict(generated.get("levels") or {})
    merged["_config_sources"] = {
        "template": str(template_config_path(cloth_data_dir)),
        "generated_levels": str(generated_levels_config_path(template)),
    }
    return merged


def studio_project_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_template_config()
    pr = cfg.get("path_resolution") or {}
    env_name = str(pr.get("studio_project_env") or "ORCA_STUDIO_PROJECT")
    default = str(pr.get("studio_project_default") or "OrcaStudio_2409")
    raw = os.environ.get(env_name, "").strip() or default
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ORCA_REPO_ROOT / path).resolve()


def studio_project_rel(cfg: dict[str, Any] | None = None) -> str:
    """Studio 工程相对 ``ORCA_REPO_ROOT`` 的路径字符串（写入 prefab_rel）。"""
    root = studio_project_dir(cfg)
    repo = ORCA_REPO_ROOT
    try:
        return root.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return root.name


def level_assets_dir(level: str, cfg: dict[str, Any] | None = None) -> Path:
    return (studio_project_dir(cfg) / "Assets" / level).resolve()


def asset_catalog_hint(level: str, filename: str, cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_template_config()
    pr = cfg.get("path_resolution") or {}
    pattern = str(pr.get("asset_catalog_hint_pattern") or "assets/{level_lower}/{filename}")
    return pattern.format(level_lower=level_lower_for_hint(level), filename=filename)


def level_entry(
    level: str,
    cfg: dict[str, Any] | None = None,
    *,
    auto_sync: bool = True,
    cloth_data_dir: Path | None = None,
) -> dict[str, Any] | None:
    """
    查询某关卡的生成条目。

    ``auto_sync`` 为真时：条目缺失或 prefab/资产已更新则自动扫描并写入 ``~/.orcagym``（无需手动 sync）。
    """
    level_name = str(level).strip()
    if not level_name:
        return None
    if auto_sync:
        ensure_level_scene_config(level_name, cloth_data_dir=cloth_data_dir)
    cfg = load_scene_assets_config(cloth_data_dir=cloth_data_dir)
    entry = (cfg.get("levels") or {}).get(level_name)
    return entry if isinstance(entry, dict) else None


def resolve_level_prefab_path(level: str, cfg: dict[str, Any] | None = None) -> Path | None:
    """
    定位关卡 prefab：优先 ``Levels/{level}/{level}.prefab``，否则在关卡目录内找含布片组件的 prefab。
    """
    cfg = cfg or load_template_config()
    studio = studio_project_dir(cfg)
    primary = studio / "Levels" / level / f"{level}.prefab"
    if primary.is_file():
        return primary.resolve()

    level_dir = studio / "Levels" / level
    if not level_dir.is_dir():
        return None

    candidates: list[Path] = []
    for path in sorted(level_dir.glob("*.prefab")):
        if "_savebackup" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "EditorMjXpbdClothSheetComponent" in text:
            candidates.append(path.resolve())
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return candidates[0]
    return None


def discover_level_names(cfg: dict[str, Any] | None = None) -> list[str]:
    """枚举 Studio ``Levels/*`` 下可作为关卡的目录名。"""
    cfg = cfg or load_template_config()
    levels_root = studio_project_dir(cfg) / "Levels"
    if not levels_root.is_dir():
        return []
    names: list[str] = []
    for path in sorted(levels_root.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if name.startswith("_") or name.startswith("."):
            continue
        if (path / f"{name}.prefab").is_file() or list(path.glob("*.prefab")):
            names.append(name)
    return names


def build_level_entry_from_scan(level: str, cfg: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """
    从 prefab + ``Assets/<level>/`` 扫描生成单关卡配置条目。

    无 ``EditorMjXpbdClothSheet`` 且无掩码资产时返回 ``None``。
    """
    cfg = cfg or load_template_config()
    from .ProcessStudio import (  # noqa: WPS433
        extract_cloth_entity_with_xpbd_sheet,
        extract_cloth_sheet_mesh_asset_hint,
        extract_prefab_vtk_asset_path,
        _discover_masked_stems_in_asset_dir,
        _stem_from_vtk_asset_path,
    )
    prefab_path = resolve_level_prefab_path(level, cfg)
    asset_dir = level_assets_dir(level, cfg)
    prefab_text = ""
    cloth_entity: tuple[str, str] | None = None
    vtk_asset_path = ""

    if prefab_path is not None:
        prefab_text = prefab_path.read_text(encoding="utf-8", errors="replace")
        cloth_entity = extract_cloth_entity_with_xpbd_sheet(prefab_text)
        vtk_asset_path = str(extract_prefab_vtk_asset_path(prefab_text) or "").strip()

    masked_stems = _discover_masked_stems_in_asset_dir(asset_dir, cfg)
    if cloth_entity is None and not masked_stems and not vtk_asset_path:
        return None

    procedural = not vtk_asset_path
    if procedural:
        mesh_hint = extract_cloth_sheet_mesh_asset_hint(prefab_text) if prefab_text else None
        entry: dict[str, Any] = {
            "cloth_mode": "procedural",
            "prefabs": [
                {
                    "prefab_rel": prefab_path.relative_to(ORCA_REPO_ROOT).as_posix() if prefab_path else "",
                    "cloth_entity_name": cloth_entity[0] if cloth_entity else "",
                    "vtk_asset_path": "",
                    "mesh_asset_hint": mesh_hint or "",
                }
            ],
        }
        return entry

    stem = _stem_from_vtk_asset_path(vtk_asset_path)
    if stem and stem not in masked_stems:
        masked_stems.insert(0, stem)
    if not masked_stems and stem:
        masked_stems = [stem]
    if not masked_stems:
        return None

    primary_stem = stem or masked_stems[0]
    vtk_expected = vtk_asset_path or qualified_vtk_asset_path(level, primary_stem)
    mesh_hint = (
        extract_cloth_sheet_mesh_asset_hint(prefab_text)
        if prefab_text
        else asset_catalog_hint(level, f"{primary_stem}.fbx.azmodel", cfg)
    )
    if not mesh_hint:
        mesh_hint = asset_catalog_hint(level, f"{primary_stem}.fbx.azmodel", cfg)

    return {
        "cloth_mode": "masked_vtk",
        "masked_cloth_stems": masked_stems,
        "prefabs": [
            {
                "prefab_rel": prefab_path.relative_to(ORCA_REPO_ROOT).as_posix() if prefab_path else "",
                "cloth_entity_name": cloth_entity[0] if cloth_entity else "",
                "vtk_asset_path": vtk_expected,
                "mesh_asset_hint": mesh_hint,
            }
        ],
    }


def save_generated_levels_document(levels: dict[str, Any], *, template: dict[str, Any] | None = None) -> Path:
    """将 ``levels`` 表写入 ``~/.orcagym/cloth/scene_levels.json``。"""
    template = template or load_template_config()
    out_dir = orcagym_cloth_config_dir(template)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = generated_levels_config_path(template)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": _GENERATOR_NAME,
        "levels": levels,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def sync_level_config(level: str, *, template: dict[str, Any] | None = None) -> bool:
    """
    扫描单个关卡并合并写入本机 ``scene_levels.json``。

    成功写入或更新条目时返回 ``True``；无法识别布料场景时返回 ``False``。
    """
    template = template or load_template_config()
    entry = build_level_entry_from_scan(level, template)
    if entry is None:
        return False

    entry["synced_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = load_generated_levels_document(template)
    levels = dict(doc.get("levels") or {})
    levels[level] = entry
    save_generated_levels_document(levels, template=template)
    return True


def ensure_level_scene_config(
    level: str,
    *,
    template: dict[str, Any] | None = None,
    force: bool = False,
    cloth_data_dir: Path | None = None,
) -> bool:
    """
    确保本机 ``scene_levels.json`` 含当前关卡的最新条目（联调主入口，用户无需手跑 sync 脚本）。

    - 无条目 → 扫描写入
    - prefab / ``Assets/<level>/`` 比 ``synced_at`` 新 → 重新扫描
  - ``CLOTH_NO_AUTO_SCENE_SYNC=1`` 时跳过
    """
    level_name = str(level).strip()
    if not level_name or not is_auto_scene_sync_enabled():
        return False

    template = template or load_template_config(cloth_data_dir)
    doc = load_generated_levels_document(template)
    existing = (doc.get("levels") or {}).get(level_name)
    if isinstance(existing, dict) and not force and not level_entry_is_stale(level_name, existing, template):
        return True
    return sync_level_config(level_name, template=template)


def sync_all_level_configs(
    levels: list[str] | None = None,
    *,
    template: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """
    批量扫描关卡并写入 ``~/.orcagym/cloth/scene_levels.json``。

    返回 ``(synced_levels, skipped_levels)``。
    """
    template = template or load_template_config()
    names = levels if levels is not None else discover_level_names(template)
    doc = load_generated_levels_document(template)
    merged = dict(doc.get("levels") or {})
    synced: list[str] = []
    skipped: list[str] = []

    for level in names:
        entry = build_level_entry_from_scan(level, template)
        if entry is None:
            skipped.append(level)
            continue
        entry["synced_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        merged[level] = entry
        synced.append(level)

    if synced:
        save_generated_levels_document(merged, template=template)
    return synced, skipped
