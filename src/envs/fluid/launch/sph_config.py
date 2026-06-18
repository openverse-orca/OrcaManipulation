"""SPH 侧 JSON 配置生成与 Python 日志引导。"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..paths import FLUID_EXAMPLES_DIR, FLUID_PACKAGE_DIR

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place (override wins on conflicts)."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def _resolve_coupling_mode(fluid_config: Dict) -> str:
    """
    从 fluid JSON 解析 OrcaLink 耦合模式。

    优先 ``simulation.sync_mode``，其次 ``orcalink.bridge.coupling_mode``；
    缺省为 ``multi_point_force``（与全链路产品默认一致）。
    """
    simulation = fluid_config.get("simulation") or {}
    bridge = (fluid_config.get("orcalink") or {}).get("bridge") or {}
    return (
        simulation.get("sync_mode")
        or bridge.get("coupling_mode")
        or "multi_point_force"
    )


def _apply_fluid_bridge_to_orcasph(orcasph_config: dict, fluid_config: dict) -> None:
    """
    将 fluid JSON 中的耦合模式与 SPH 专用参数合并进 ``orcalink_bridge``。

    - 设置 ``coupling_mode``（与 Python OrcaLinkBridge 一致）；
    - **不**合并 ``orcalink.bridge.<mode>.channels``（Python 与 SPH 的 publish/subscribe 视角相反，由 SPH 模板提供）；
    - 可选 ``orcasph.position_follow`` / ``orcasph.rigid_body_dynamics`` 覆盖模板；
    - 同步 ``orcalink.client.session.sync_params`` → ``orcalink_client.session``。
    """
    coupling_mode = _resolve_coupling_mode(fluid_config)
    ob = orcasph_config.setdefault("orcalink_bridge", {})
    ob["coupling_mode"] = coupling_mode

    orcasph_cfg = fluid_config.get("orcasph") or {}
    if coupling_mode == "force_position":
        fp = ob.setdefault("force_position", {})
        pf_override = orcasph_cfg.get("position_follow")
        if pf_override:
            _deep_merge(fp, {"position_follow": pf_override})
            logger.info(
                "[OrcaSPH] position_follow override: mode=%s",
                pf_override.get("mode", "proportional"),
            )
        rbd_override = orcasph_cfg.get("rigid_body_dynamics")
        if rbd_override:
            _deep_merge(fp, {"rigid_body_dynamics": rbd_override})

    sync_params = (
        (fluid_config.get("orcalink") or {})
        .get("client", {})
        .get("session", {})
        .get("sync_params")
    )
    if sync_params:
        oc = orcasph_config.setdefault("orcalink_client", {})
        sess = oc.setdefault("session", {})
        _deep_merge(sess.setdefault("sync_params", {}), sync_params)

    logger.info("[OrcaSPH] orcalink_bridge coupling_mode=%s (from fluid JSON)", coupling_mode)


def _apply_particle_render_run_mode(orcasph_config: dict, fluid_config: dict) -> None:
    """
    Apply config['particle_render_run'] to particle_render after template + MJCF overrides.

    - live: force recording.enabled false (grpc unchanged from template).
    - record: recording on, HDF5 output_path and optional record_fps; gRPC to OrcaStudio only when
      particle_render_run.record_send_to_studio is true (CLI: run_fluid_sim.py --render-particle).
    """
    pr_run = fluid_config.get("particle_render_run") or {}
    mode = pr_run.get("mode", "live")
    if "particle_render" not in orcasph_config:
        return
    pr = orcasph_config["particle_render"]
    if mode == "live":
        _deep_merge(pr, {"recording": {"enabled": False}})
        logger.info("[ParticleRender] run mode live: recording.enabled=false")
        return
    if mode == "record":
        rec_path = pr_run.get("record_output_path") or ""
        send_to_studio = pr_run.get("record_send_to_studio", False)
        override: Dict[str, Any] = {
            "recording": {
                "enabled": True,
                "output_path": rec_path,
            },
        }
        if rec_path:
            cursor_path = str(Path(rec_path).resolve()) + ".sph_frame_cursor"
            override["recording"]["sph_frame_cursor_path"] = cursor_path
        if not send_to_studio:
            override["grpc"] = {"enabled": False}
        _deep_merge(pr, override)
        rf = pr_run.get("record_fps")
        if rf is not None:
            rf_f = float(rf)
            if "recording" not in pr:
                pr["recording"] = {}
            pr["recording"]["record_fps"] = rf_f
            if "grpc" not in pr:
                pr["grpc"] = {}
            pr["grpc"]["update_rate_hz"] = rf_f
        log_extra = ""
        if rec_path:
            log_extra = f", sph_frame_cursor_path={override['recording'].get('sph_frame_cursor_path')!r}"
        logger.info(
            "[ParticleRender] run mode record: HDF5 output_path=%r, gRPC to OrcaStudio=%s%s",
            rec_path,
            "on" if send_to_studio else "off",
            log_extra,
        )
        return


def generate_orcasph_config(
    fluid_config: Dict,
    output_path: Path,
    particle_render_override: Optional[Dict] = None,
    sphscale: float = 1.0,
) -> tuple[Path, bool]:
    """
    动态生成 orcasph 配置文件

    Args:
        fluid_config: 完整的 fluid_config.json 内容
        output_path: 输出配置文件路径
        particle_render_override: 从 MJCF bound site 计算的 particle_render 覆盖项
        sphscale: SPH 世界缩放因子（默认 1.0 = 不缩放）

    Returns:
        (生成的配置文件路径, verbose_logging配置值)
    """
    orcasph_cfg = fluid_config.get("orcasph", {})
    orcalink_cfg = fluid_config.get("orcalink", {})

    orcasph_config_template = {}

    if "config_template" in orcasph_cfg:
        template_filename = orcasph_cfg["config_template"]
        template_paths = [
            FLUID_EXAMPLES_DIR / template_filename,
            FLUID_PACKAGE_DIR / template_filename,
            Path(template_filename),
        ]

        template_path = None
        for path in template_paths:
            if path.exists():
                template_path = path
                break

        if template_path:
            with open(template_path, "r", encoding="utf-8") as f:
                orcasph_config_template = json.load(f)
            logger.info(f"✅ 从模板加载 SPH 配置: {template_path}")
        else:
            logger.warning(f"⚠️  配置模板文件未找到: {template_filename}，尝试的路径：{template_paths}")
            orcasph_config_template = {}
    elif "config" in orcasph_cfg:
        orcasph_config_template = orcasph_cfg["config"]
        logger.info("✅ 使用内嵌 SPH 配置（旧格式）")
    else:
        logger.warning("⚠️  未找到 SPH 配置模板，使用空配置")

    orcasph_config = {
        "orcalink_client": {
            "enabled": orcalink_cfg.get("enabled", True),
            "server_address": f"{orcalink_cfg.get('host', 'localhost')}:{orcalink_cfg.get('port', 50351)}",
            **orcasph_config_template.get("orcalink_client", {}),
        },
        "orcalink_bridge": orcasph_config_template.get("orcalink_bridge", {}),
        "physics": orcasph_config_template.get("physics", {}),
        "debug": orcasph_config_template.get("debug", {}),
    }

    if "particle_render" in orcasph_config_template:
        orcasph_config["particle_render"] = orcasph_config_template["particle_render"]

    if particle_render_override and "particle_render" in orcasph_config:
        _deep_merge(orcasph_config["particle_render"], particle_render_override)
        logger.info(f"particle_render config overridden from MJCF bound site: {particle_render_override}")

    _apply_particle_render_run_mode(orcasph_config, fluid_config)

    orcasph_config["orcalink_client"]["server_address"] = (
        f"{orcalink_cfg.get('host', 'localhost')}:{orcalink_cfg.get('port', 50351)}"
    )
    orcasph_config["orcalink_client"]["enabled"] = orcalink_cfg.get("enabled", True)

    pr_grpc_go = fluid_config.get("particle_render_grpc_override")
    if pr_grpc_go and "particle_render" in orcasph_config:
        _deep_merge(orcasph_config["particle_render"].setdefault("grpc", {}), pr_grpc_go)
        logger.info("particle_render.grpc 覆盖（来自 fluid_config.particle_render_grpc_override）: %s", pr_grpc_go)

    _apply_fluid_bridge_to_orcasph(orcasph_config, fluid_config)

    orcasph_config.setdefault("orcalink_bridge", {})
    orcasph_config["orcalink_bridge"]["sphscale"] = sphscale
    logger.info(f"[SPHSCALE] sphscale={sphscale} written to orcasph_config")

    if sphscale != 1.0:
        sc = orcasph_config["orcalink_bridge"].get("spring_constraint", {})
        if sc:
            original_stiffness = sc.get("stiffness", 5000.0)
            original_damping = sc.get("damping", 100.0)
            sc["stiffness"] = original_stiffness * (sphscale ** 2)
            sc["damping"] = original_damping * (sphscale ** 2)
            logger.info(f"[SPHSCALE] Spring stiffness scaling: {original_stiffness} → {sc['stiffness']} (×sphscale²)")
            logger.info(f"[SPHSCALE] Spring damping scaling: {original_damping} → {sc['damping']} (×sphscale²)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(orcasph_config, f, indent=2, ensure_ascii=False)

    verbose_logging = orcasph_config.get("debug", {}).get("verbose_logging", False)

    logger.info(f"✅ 已生成 orcasph 配置文件: {output_path}")
    return output_path, verbose_logging


def prepare_orcasph_config_from_fixed_path(
    fluid_config: Dict,
    source_path: Path,
    output_path: Path,
) -> tuple[Path, bool]:
    """
    从 scene_3chain 等固定路径加载 orcasph JSON，合并 fluid 配置中的 OrcaLink 地址与
    particle_render_run，写入本次会话输出路径（不修改源文件）。
    """
    orcalink_cfg = fluid_config.get("orcalink", {})
    with open(source_path, "r", encoding="utf-8") as f:
        orcasph_config = json.load(f)
    _apply_particle_render_run_mode(orcasph_config, fluid_config)
    orcasph_config.setdefault("orcalink_client", {})
    orcasph_config["orcalink_client"]["server_address"] = (
        f"{orcalink_cfg.get('host', 'localhost')}:{orcalink_cfg.get('port', 50351)}"
    )
    orcasph_config["orcalink_client"]["enabled"] = orcalink_cfg.get("enabled", True)
    pr_grpc_go = fluid_config.get("particle_render_grpc_override")
    if pr_grpc_go and "particle_render" in orcasph_config:
        _deep_merge(orcasph_config["particle_render"].setdefault("grpc", {}), pr_grpc_go)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(orcasph_config, f, indent=2, ensure_ascii=False)
    verbose_logging = orcasph_config.get("debug", {}).get("verbose_logging", False)
    logger.info(
        "✅ 已从固定路径准备 orcasph 配置: %s → %s",
        source_path,
        output_path,
    )
    return output_path, verbose_logging


def setup_python_logging(config: Dict) -> None:
    """根据配置设置 Python 日志级别"""
    verbose_logging = config.get("debug", {}).get("verbose_logging", False)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    if verbose_logging:
        root_logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
        logger.info("🔍 Python 日志级别: DEBUG (verbose_logging=true)")
    else:
        root_logger.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
        logger.info("ℹ️  Python 日志级别: INFO (verbose_logging=false)")

    root_logger.addHandler(console_handler)

    try:
        from orcalink_client import setup_logging as setup_orcalink_logging

        setup_orcalink_logging(verbose=verbose_logging, use_root_handler=True)
    except ImportError:
        pass
