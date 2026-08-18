"""将布料 MjcPBD 耦合挂载到已有 OrcaGym 环境（数据采集 / 遥操）。"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

from .common.process_utils import ProcessManager

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

from .common.paths import (
    ORCA_REPO_ROOT,
    SOFTBODY_DIR,
    SOFTBODY_MODULES_DIR,
    SOFTBODY_SCRIPTS_DIR,
    default_cloth_config_path,
    resolve_cloth_data_dir,
)

logger = logging.getLogger(__name__)

_XPBD_DEBUG_LOG = ORCA_REPO_ROOT / "XPBD" / "MjcPBD_orcalink" / "debug_log"
CLOTH_SCENE_ASSETS_BASENAME = "cloth_scene_assets.json"


def _ensure_cloth_modules_import_path() -> None:
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


def build_xpbd_session_config(base_cfg: dict[str, Any], adapted_cfg: dict[str, Any], cloth_data_dir: Path | None = None) -> dict[str, Any]:
    """
    构造供 ``MJC_PBD_CONFIG`` 使用的会话 JSON。

    P2 默认 ``xpbd.cloth_discover_only=true``：仅将 ``cloth`` discovered 段交给 XPBD，
    ``rigid_body_map`` 保持基配置短链（暂不向 XPBD 传递 Studio 扫描的 N 刚体）。
    """
    out = copy.deepcopy(adapted_cfg)
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
    cloth_data_dir: Path | None = None,
) -> Path:
    """
    将运行时 effective config 写入 ``<cloth_data_dir>/cloth_sim_session_{ts}.json``，供 XPBD 子进程加载。
    """
    data_dir = resolve_cloth_data_dir(cloth_data_dir)
    session_path = (data_dir / f"cloth_sim_session_{session_timestamp}.json").resolve()
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


def export_xpbd_scene_for_session(
    session_path: Path,
    *,
    out_path: Path | None = None,
    cloth_data_dir: Path | None = None,
) -> Path:
    """
    调用 ``softbody/scripts/export_xpbd_scene_from_mjcf.py``，从 session JSON 导出
    ``xpbd_scene_from_mjcf.json``（与 XPBD ``mjc_pbd_bridge`` 自动导出同源）。

    session 须含 ``rigid_body_map`` 与 ``mujoco.model_path`` 或 ``_cloth_robot_session_meta.source_mjcf``。
    返回写出 JSON 的绝对路径。
    """
    import subprocess
    import sys

    session_path = session_path.resolve()
    if not session_path.is_file():
        raise FileNotFoundError(f"session config not found: {session_path}")

    export_script = SOFTBODY_SCRIPTS_DIR / "export_xpbd_scene_from_mjcf.py"
    if not export_script.is_file():
        raise FileNotFoundError(f"export script not found: {export_script}")

    cmd = [sys.executable, str(export_script), "--config", str(session_path)]
    if out_path is not None:
        cmd.extend(["--out", str(out_path.resolve())])

    logger.info("export xpbd scene: %s", " ".join(cmd))
    data_dir = resolve_cloth_data_dir(cloth_data_dir)
    proc = subprocess.run(cmd, cwd=str(data_dir), capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stdout or "") + (proc.stderr or "")
        raise RuntimeError(f"export_xpbd_scene_from_mjcf failed (rc={proc.returncode}): {detail}")

    if out_path is not None:
        return out_path.resolve()

    cfg = json.loads(session_path.read_text(encoding="utf-8"))
    dbg = cfg.get("debug", {})
    dbg_dir = Path(str(dbg.get("debug_log_dir", _XPBD_DEBUG_LOG)))
    if not dbg_dir.is_absolute():
        dbg_dir = (data_dir / dbg_dir).resolve()
    return (dbg_dir / "xpbd_scene_from_mjcf.json").resolve()


def cloth_scene_assets_config_path(cloth_data_dir: Path | None = None) -> Path:
    """``cloth_scene_assets.json`` 路径；可用 ``CLOTH_SCENE_ASSETS_CONFIG`` 覆盖。"""
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

    modules_dir = SOFTBODY_MODULES_DIR
    if str(modules_dir) not in sys.path:
        sys.path.insert(0, str(modules_dir))
    from scene_cloth_config import ensure_level_scene_config, level_entry  # noqa: WPS433

    ensure_level_scene_config(level_name, cloth_data_dir=cloth_data_dir)
    level_entry(level_name, auto_sync=False, cloth_data_dir=cloth_data_dir)
    return (_studio_project_dir_from_template(cloth_data_dir=cloth_data_dir) / "Assets" / level_name).resolve()


def _studio_project_dir_from_template(cloth_data_dir: Path | None = None) -> Path:
    import sys

    modules_dir = SOFTBODY_MODULES_DIR
    if str(modules_dir) not in sys.path:
        sys.path.insert(0, str(modules_dir))
    from scene_cloth_config import studio_project_dir  # noqa: WPS433

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

    modules_dir = SOFTBODY_MODULES_DIR
    if str(modules_dir) not in sys.path:
        sys.path.insert(0, str(modules_dir))
    if str(SOFTBODY_DIR) not in sys.path:
        sys.path.insert(0, str(SOFTBODY_DIR))
    from scene_cloth_config import level_entry  # noqa: WPS433

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


def apply_masked_cloth_from_level(config: dict[str, Any], level: str, cloth_data_dir: Path | None = None) -> dict[str, Any]:
    """
    MJCF 未扫描到布片时，用关卡 ``scene_levels.json`` 写入掩码 ``cloth`` 块。

    已 ``discovered`` 的布片不覆盖；程序化关卡（无 masked stem）不改动。
    """
    out = copy.deepcopy(config)
    cloth = out.setdefault("cloth", {})
    if cloth.get("discovered"):
        return out

    stem = level_primary_masked_stem(level, cloth_data_dir=cloth_data_dir)
    if not stem:
        return out

    current_mesh = str(cloth.get("mesh") or "").strip()
    legacy_meshes = ("shirt_v4.vtk", "shirt_new.vtk", "")
    if current_mesh and current_mesh not in legacy_meshes and stem in current_mesh:
        mesh_name = current_mesh
    else:
        mesh_name = f"{stem}.vtk"

    import sys

    if str(SOFTBODY_DIR) not in sys.path:
        sys.path.insert(0, str(SOFTBODY_DIR))
    from modules.masked_vtk_assets import enrich_cloth_entry_with_masked_assets  # noqa: WPS433

    enriched = enrich_cloth_entry_with_masked_assets(
        {"mesh": mesh_name, "vtk_asset_path": mesh_name},
        level=level,
    )
    cloth.update(enriched)
    cloth.setdefault("level", level)
    cloth.setdefault("asset_dir", str(studio_cloth_assets_dir(level, cloth_data_dir=cloth_data_dir)))
    return out


@dataclass
class ResolvedSceneAssets:
    """编排器对某关卡「查一次」得到的布料场景/资产结果，传给各 Process* 模块。"""

    asset_dir: str
    primary_masked_stem: str | None
    masked_cloth_block: dict[str, Any] | None


def _resolve_scene_assets(config: dict[str, Any], cloth_data_dir: Path | None = None) -> ResolvedSceneAssets:
    """解析某关卡的 asset_dir、掩码布 stem 与掩码布块（原 paths.apply_masked_cloth_from_level 的「查」部分）。"""
    level = str((config.get("orcagym") or {}).get("level") or "").strip()
    if not level:
        return ResolvedSceneAssets(asset_dir="", primary_masked_stem=None, masked_cloth_block=None)

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
        if str(SOFTBODY_DIR) not in sys.path:
            sys.path.insert(0, str(SOFTBODY_DIR))
        from modules.masked_vtk_assets import enrich_cloth_entry_with_masked_assets  # noqa: WPS433

        masked = enrich_cloth_entry_with_masked_assets(
            {"mesh": mesh_name, "vtk_asset_path": mesh_name}, level=level
        )
    return ResolvedSceneAssets(
        asset_dir=asset_dir, primary_masked_stem=stem, masked_cloth_block=masked
    )


def apply_runtime_cloth_overrides(
    config: dict[str, Any],
    *,
    level: str | None = None,
    mjc_agent_prefix: str | None = None,
) -> dict[str, Any]:
    """
    将运行时关卡名、MuJoCo 机器人前缀写入配置（深拷贝）。

    关卡与机器人型号不写进 ``cloth_sim_config.json``，由 CLI / 环境变量注入。
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
    modules_dir = SOFTBODY_MODULES_DIR
    if str(modules_dir) not in sys.path:
        sys.path.insert(0, str(modules_dir))
    try:
        from scene_cloth_config import ensure_level_scene_config  # noqa: WPS433

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
    from modules.identify_xpbd_bodies import merge_body_discovery  # noqa: WPS433
    from modules.identify_xpbd_cloth import (  # noqa: WPS433
        enrich_cloth_discovery_pose,
        identify_xpbd_cloth,
        merge_cloth_discovery,
    )

    cloths = enrich_cloth_discovery_pose(model, data, identify_xpbd_cloth(model))
    base_cfg = merge_cloth_discovery(base_cfg, cloths)
    base_cfg = merge_body_discovery(base_cfg, model, data)
    scene_assets = _resolve_scene_assets(base_cfg, cloth_data_dir=cloth_data_dir)
    adapted = adapt_config_for_orcagym(model, base_cfg, scene_assets=scene_assets, data=data)
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
        cloth_data_dir=cloth_data_dir,
    )
    return xpbd_session, adapted, session_path


def _connect_cloth_bridge(
    env: OrcaGymLocalEnv,
    config: dict[str, Any],
    ctx: ClothCouplingContext,
    *,
    adapted: dict[str, Any] | None = None,
    cloth_data_dir: Path | None = None,
) -> bool:
    _ensure_cloth_modules_import_path()
    from modules.body_map import load_body_map_ordered  # noqa: WPS433
    from modules.cloth_orcalink_bridge import ClothOrcaLinkBridge  # noqa: WPS433

    model, data = get_mujoco_model_data(env)
    if adapted is None:
        scene_assets = _resolve_scene_assets(config, cloth_data_dir=cloth_data_dir)
        adapted = adapt_config_for_orcagym(model, config, scene_assets=scene_assets, data=data)
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
    bridge = ClothOrcaLinkBridge(adapted, model, data, pose_remapper=remapper, cloth_root=cloth_data_dir)
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
    _ensure_cloth_modules_import_path()
    cfg = copy.deepcopy(config)
    path = Path(config_path or default_cloth_config_path(cloth_data_dir=cloth_data_dir)).resolve()
    base_cfg = load_cloth_config(path, cloth_data_dir=cloth_data_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir).resolve() if log_dir else None

    if auto_start_orcalink is not None:
        cfg.setdefault("orcalink", {})["auto_start"] = bool(auto_start_orcalink)
    if auto_start_xpbd is not None:
        cfg.setdefault("xpbd", {})["auto_start"] = bool(auto_start_xpbd)

    og = cfg.setdefault("orcagym", {})
    resolved_level = resolve_cloth_level(str(og.get("level") or "").strip() or None, cloth_data_dir=cloth_data_dir)
    og["level"] = resolved_level
    cfg = apply_masked_cloth_from_level(cfg, resolved_level, cloth_data_dir=cloth_data_dir)

    ctx = ClothCouplingContext(config=cfg, config_path=path, env=env, session_timestamp=ts)

    model, data = get_mujoco_model_data(env)
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    base_env.mj_forward()

    from .ProcessStudio import run_masked_vtk_prefab_check_at_startup

    scene_assets = _resolve_scene_assets(cfg, cloth_data_dir=cloth_data_dir)
    if not run_masked_vtk_prefab_check_at_startup(model, cfg, scene_assets=scene_assets):
        import logging
        logging.getLogger(__name__).warning(
            "掩码 VTK 预制检查未通过（将继续运行，仅 VTK 加载不受影响）。"
            "跳过: CLOTH_SKIP_MASKED_PREFAB_CHECK=1"
        )

    adapted = adapt_config_for_orcagym(model, cfg, scene_assets=scene_assets, data=data)
    xpbd_session_cfg = build_xpbd_session_config(base_cfg, adapted, cloth_data_dir=cloth_data_dir)
    mjcf_path = get_mujoco_xml_path(env)
    xpbd_session_cfg.setdefault("mujoco", {})["model_path"] = str(mjcf_path)
    session_path = write_xpbd_runtime_session_config(
        xpbd_session_cfg,
        session_timestamp=ts,
        source_config_path=path,
        source_mjcf_path=mjcf_path,
        cloth_data_dir=cloth_data_dir,
    )
    ctx.config = adapted
    ctx.config_path = session_path

    discover_only = bool((xpbd_session_cfg.get("xpbd") or {}).get("cloth_discover_only", True))
    if not discover_only:
        try:
            scene_path = export_xpbd_scene_for_session(session_path, cloth_data_dir=cloth_data_dir)
            logger.info("XPBD scene exported: %s", scene_path)
        except Exception as exc:
            ctx.process_manager.cleanup_all()
            raise RuntimeError(f"export_xpbd_scene_from_mjcf 失败: {exc}") from exc

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
        log_dir=log_path,
        session_timestamp=ts,
        force_restart=True,
    )
    xpbd_dg_traj = str((adapted.get("xpbd") or {}).get("dg_traj", "")).strip()
    if xpbd_dg_traj == "pico" and log_path is not None:
        trigger_path = Path(log_path) / "grip_triggers.txt"
        os.environ["MJC_PBD_GRIP_TRIGGER_PATH"] = str(trigger_path)
        logger.info("PICO grip triggers → %s (MJC_PBD_DG_TRAJ=pico)", trigger_path)
    start_xpbd_if_configured(
        adapted,
        config_path=session_path,
        process_manager=ctx.process_manager,
        log_dir=log_path,
        session_timestamp=ts,
        cpu_affinity=cpu_affinity,
        cloth_data_dir=cloth_data_dir,
    )

    if not _connect_cloth_bridge(env, cfg, ctx, adapted=adapted, cloth_data_dir=cloth_data_dir):
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
    cloth_debug: bool = False
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
    if params.cloth_debug:
        cloth_config.setdefault("debug", {})["debug_mode"] = True
    else:
        cloth_config.setdefault("debug", {})["debug_mode"] = False
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
    trigger_path = params.log_dir / "grip_triggers.txt"

    def _read_pico_triggers() -> tuple[float, float]:
        ks = pico_joystick.get_key_state()
        if not ks:
            return 0.0, 0.0
        return (float(ks["leftHand"]["triggerValue"]), float(ks["rightHand"]["triggerValue"]))

    cloth_handle.set_grip_trigger_provider(_read_pico_triggers, trigger_path)
    data_collection_manager.set_skip_render(True)


def run_p23c(params: P23cParams) -> int:
    """P23c 完整编排：准备 → 组装 → 建 env → 挂耦合 → 装配 → 跑循环。"""
    from .common.paths import resolve_cloth_config_path
    from .ProcessOrcaGym import scan_tele_layout_from_mjcf
    from .ProcessStudio import ensure_ready, find_latest_studio_mjcf_path
    from .ProcessXPBD import DEFAULT_TARGET, cleanup, prepare

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
        config_path = str(resolve_cloth_config_path(level=level, agent=params.agent, debug=params.cloth_debug, cloth_data_dir=cloth_data_dir))
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
                    config_path = str(resolve_cloth_config_path(level=level, agent=agent, debug=params.cloth_debug, cloth_data_dir=cloth_data_dir))
        elif detected_prefix and detected_prefix != mjc_prefix:
            mjc_prefix = detected_prefix

    os.environ["LEVEL"] = level or ""
    os.environ["AGENT"] = agent
    os.environ["MJC_PREFIX"] = mjc_prefix
    os.environ["CFG"] = config_path
    os.environ["CLOTH_DEBUG"] = "1" if params.cloth_debug else "0"

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
        target = params.xpbd_build_target or DEFAULT_TARGET
        os.environ["XPBD_BUILD_TARGET"] = target
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
