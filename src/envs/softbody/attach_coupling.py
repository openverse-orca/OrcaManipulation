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

from ..fluid.launch.process_utils import ProcessManager
from .ProcessOrcaGym import (
    adapt_config_for_orcagym,
    get_mujoco_model_data,
    get_mujoco_xml_path,
    validate_orcagym_body_map,
)
from .ProcessPico import write_grip_triggers
from .ProcessOrcaGym import sync_gripper_mocap_from_bodies
from .ProcessOrcaLink import OrcaLinkPoseRemapper, resolve_pose_remap, start_orcalink_if_configured
from .common.paths import CLOTH_3D_DIR, ORCA_REPO_ROOT, ORCALINK_CLIENT_PYTHON, default_cloth_config_path
from .ProcessXPBD import start_xpbd_if_configured

logger = logging.getLogger(__name__)

_XPBD_DEBUG_LOG = ORCA_REPO_ROOT / "XPBD" / "MjcPBD_orcalink" / "debug_log"


def _ensure_cloth_3d_import_path() -> None:
    root = str(CLOTH_3D_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    ol_py = str(ORCALINK_CLIENT_PYTHON)
    if ol_py not in sys.path:
        sys.path.insert(0, ol_py)


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


def load_cloth_config(config_path: str | Path) -> Dict[str, Any]:
    """
    加载 cloth_sim JSON；若含 \"extends\" 则递归加载基配置再深合并。

    extends 路径相对 cloth_3d 目录。
    """
    path = Path(config_path).expanduser().resolve()
    raw = _load_cloth_json_file(path)
    extends = raw.get("extends")
    if not extends:
        return raw
    base_path = (CLOTH_3D_DIR / str(extends)).resolve()
    base = load_cloth_config(base_path)
    return _deep_merge(base, raw)


def build_xpbd_session_config(base_cfg: dict[str, Any], adapted_cfg: dict[str, Any]) -> dict[str, Any]:
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
            from envs.softbody.common.paths import studio_cloth_assets_dir  # noqa: WPS433

            cloth["asset_dir"] = str(studio_cloth_assets_dir(level))
    return out


def write_xpbd_runtime_session_config(
    config: dict[str, Any],
    *,
    session_timestamp: str,
    source_config_path: Path | None = None,
    source_mjcf_path: Path | None = None,
) -> Path:
    """
    将运行时 effective config 写入 ``cloth_3d/cloth_sim_session_{ts}.json``，供 XPBD 子进程加载。

    路径须在 ``OrcaPlayground/examples/cloth_3d/`` 下，以便 XPBD 侧脚本相对解析。
    """
    session_path = (CLOTH_3D_DIR / f"cloth_sim_session_{session_timestamp}.json").resolve()
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
) -> Path:
    """
    调用 ``cloth_3d/scripts/export_xpbd_scene_from_mjcf.py``，从 session JSON 导出
    ``xpbd_scene_from_mjcf.json``（与 XPBD ``mjc_pbd_bridge`` 自动导出同源）。

    session 须含 ``rigid_body_map`` 与 ``mujoco.model_path`` 或 ``_cloth_robot_session_meta.source_mjcf``。
    返回写出 JSON 的绝对路径。
    """
    import subprocess
    import sys

    session_path = session_path.resolve()
    if not session_path.is_file():
        raise FileNotFoundError(f"session config not found: {session_path}")

    export_script = CLOTH_3D_DIR / "scripts" / "export_xpbd_scene_from_mjcf.py"
    if not export_script.is_file():
        raise FileNotFoundError(f"export script not found: {export_script}")

    cmd = [sys.executable, str(export_script), "--config", str(session_path)]
    if out_path is not None:
        cmd.extend(["--out", str(out_path.resolve())])

    logger.info("export xpbd scene: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(CLOTH_3D_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stdout or "") + (proc.stderr or "")
        raise RuntimeError(f"export_xpbd_scene_from_mjcf failed (rc={proc.returncode}): {detail}")

    if out_path is not None:
        return out_path.resolve()

    cfg = json.loads(session_path.read_text(encoding="utf-8"))
    dbg = cfg.get("debug", {})
    dbg_dir = Path(str(dbg.get("debug_log_dir", _XPBD_DEBUG_LOG)))
    if not dbg_dir.is_absolute():
        dbg_dir = (CLOTH_3D_DIR / dbg_dir).resolve()
    return (dbg_dir / "xpbd_scene_from_mjcf.json").resolve()


def build_p2_session_from_mjcf(
    mjcf_path: Path,
    config_path: Path,
    *,
    session_timestamp: str | None = None,
    level: str | None = None,
) -> tuple[dict, dict, Path]:
    """
    从 Studio MJCF 扫描布片位姿（含 Entity3 旋转）并写出 XPBD session JSON。

    流程：``mj_forward`` → ``adapt_config_for_orcagym``（含 ``enrich_cloth_discovery_pose``）
    → ``build_xpbd_session_config`` → ``cloth_sim_session_*.json``。
    Studio 中旋转布料后须重新 Play 生成 MJCF，再调用本函数刷新 session。

    返回 ``(xpbd_session, adapted_config, session_path)``。
    """
    import mujoco

    if str(CLOTH_3D_DIR) not in sys.path:
        sys.path.insert(0, str(CLOTH_3D_DIR))

    base_cfg = load_cloth_config(config_path)
    if level:
        from .common.paths import apply_runtime_orcagym_level

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
    adapted = adapt_config_for_orcagym(model, base_cfg, data=data)
    xpbd_session = build_xpbd_session_config(base_cfg, adapted)
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
    )
    return xpbd_session, adapted, session_path


def _connect_cloth_bridge(
    env: OrcaGymLocalEnv,
    config: dict[str, Any],
    ctx: ClothCouplingContext,
    *,
    adapted: dict[str, Any] | None = None,
) -> bool:
    _ensure_cloth_3d_import_path()
    from modules.body_map import load_body_map_ordered  # noqa: WPS433
    from modules.cloth_orcalink_bridge import ClothOrcaLinkBridge  # noqa: WPS433

    model, data = get_mujoco_model_data(env)
    if adapted is None:
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
    bridge = ClothOrcaLinkBridge(adapted, model, data, pose_remapper=remapper)
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
) -> ClothCouplingHandle:
    """
    在已有 OrcaGym 环境上启动 OrcaLink / XPBD，并连接布料发布桥。

    典型顺序（与 MjcPBD 双进程一致）：
    1. OrcaLink Server（可选 auto_start）
    2. XPBD dual_gripper_cross_mjc（可选 auto_start，须先于 MuJoCo 凑齐 expected_clients）
    3. ClothOrcaLinkBridge 连接并等待 session_ready
    """
    _ensure_cloth_3d_import_path()
    cfg = copy.deepcopy(config)
    path = Path(config_path or default_cloth_config_path()).resolve()
    base_cfg = load_cloth_config(path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir).resolve() if log_dir else None

    if auto_start_orcalink is not None:
        cfg.setdefault("orcalink", {})["auto_start"] = bool(auto_start_orcalink)
    if auto_start_xpbd is not None:
        cfg.setdefault("xpbd", {})["auto_start"] = bool(auto_start_xpbd)

    from .common.paths import apply_masked_cloth_from_level, resolve_cloth_level

    og = cfg.setdefault("orcagym", {})
    resolved_level = resolve_cloth_level(str(og.get("level") or "").strip() or None)
    og["level"] = resolved_level
    cfg = apply_masked_cloth_from_level(cfg, resolved_level)

    ctx = ClothCouplingContext(config=cfg, config_path=path, env=env, session_timestamp=ts)

    model, data = get_mujoco_model_data(env)
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    base_env.mj_forward()

    from .ProcessStudio import run_masked_vtk_prefab_check_at_startup

    if not run_masked_vtk_prefab_check_at_startup(model, cfg):
        import logging
        logging.getLogger(__name__).warning(
            "掩码 VTK 预制检查未通过（将继续运行，仅 VTK 加载不受影响）。"
            "跳过: CLOTH_SKIP_MASKED_PREFAB_CHECK=1"
        )

    adapted = adapt_config_for_orcagym(model, cfg, data=data)
    xpbd_session_cfg = build_xpbd_session_config(base_cfg, adapted)
    mjcf_path = get_mujoco_xml_path(env)
    xpbd_session_cfg.setdefault("mujoco", {})["model_path"] = str(mjcf_path)
    session_path = write_xpbd_runtime_session_config(
        xpbd_session_cfg,
        session_timestamp=ts,
        source_config_path=path,
        source_mjcf_path=mjcf_path,
    )
    ctx.config = adapted
    ctx.config_path = session_path

    discover_only = bool((xpbd_session_cfg.get("xpbd") or {}).get("cloth_discover_only", True))
    if not discover_only:
        try:
            scene_path = export_xpbd_scene_for_session(session_path)
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
    )

    if not _connect_cloth_bridge(env, cfg, ctx, adapted=adapted):
        ctx.process_manager.cleanup_all()
        raise RuntimeError("布料 OrcaLink 桥接初始化失败")

    handle = ClothCouplingHandle(config=adapted, ctx=ctx, enabled=True)
    return handle
