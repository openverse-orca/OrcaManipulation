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
) -> tuple[Path, bool]:
    """
    动态生成 orcasph 配置文件

    Args:
        fluid_config: 完整的 fluid_config.json 内容
        output_path: 输出配置文件路径

    Returns:
        (生成的配置文件路径, verbose_logging配置值)
    """
    orcasph_cfg = fluid_config.get("orcasph", {})
    orcalink_cfg = fluid_config.get("orcalink", {})

    # 支持两种方式：外部模板文件（新）或内嵌配置（旧，向后兼容）
    orcasph_config_template = {}

    if "config_template" in orcasph_cfg:
        # 新方式：从外部文件加载模板
        template_filename = orcasph_cfg["config_template"]
        # 尝试多个位置查找模板文件
        template_paths = [
            FLUID_EXAMPLES_DIR / template_filename,
            FLUID_PACKAGE_DIR / template_filename,
            Path(template_filename),  # 相对于当前工作目录
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
        # 旧方式：内嵌配置（向后兼容）
        orcasph_config_template = orcasph_cfg["config"]
        logger.info("✅ 使用内嵌 SPH 配置（旧格式）")
    else:
        logger.warning("⚠️  未找到 SPH 配置模板，使用空配置")

    # 构建完整的 orcasph 配置（合并模板和动态参数）
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

    # 添加 particle_render 配置（如果模板中存在）
    if "particle_render" in orcasph_config_template:
        orcasph_config["particle_render"] = orcasph_config_template["particle_render"]

    # 用从 MJCF bound site 计算出的值覆盖 particle_render 中的 grid_resolution / origin。
    # 仅当调用方传入了计算结果时才合并（无 bound site 时 particle_render_override=None，不覆盖）。
    if particle_render_override and "particle_render" in orcasph_config:
        _deep_merge(orcasph_config["particle_render"], particle_render_override)
        logger.info(f"particle_render config overridden from MJCF bound site: {particle_render_override}")

    _apply_particle_render_run_mode(orcasph_config, fluid_config)

    # 覆盖关键参数（确保动态值生效）
    orcasph_config["orcalink_client"]["server_address"] = (
        f"{orcalink_cfg.get('host', 'localhost')}:{orcalink_cfg.get('port', 50351)}"
    )
    orcasph_config["orcalink_client"]["enabled"] = orcalink_cfg.get("enabled", True)

    # 写入文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(orcasph_config, f, indent=2, ensure_ascii=False)

    # 提取 verbose_logging 配置值
    verbose_logging = orcasph_config.get("debug", {}).get("verbose_logging", False)

    logger.info(f"✅ 已生成 orcasph 配置文件: {output_path}")
    return output_path, verbose_logging


def setup_python_logging(config: Dict) -> None:
    """根据配置设置 Python 日志级别"""
    verbose_logging = config.get("debug", {}).get("verbose_logging", False)

    # 设置根 logger 的级别
    root_logger = logging.getLogger()

    # 清除现有的 handlers（避免重复）
    root_logger.handlers.clear()

    # 创建统一的 formatter，包含模块名称
    # 格式: [模块名] 级别: 消息
    formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")

    # 创建 console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 根据配置设置日志级别
    if verbose_logging:
        root_logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
        logger.info("🔍 Python 日志级别: DEBUG (verbose_logging=true)")
    else:
        root_logger.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
        logger.info("ℹ️  Python 日志级别: INFO (verbose_logging=false)")

    # 添加 handler 到根 logger
    root_logger.addHandler(console_handler)

    # 配置 OrcaLinkClient 的日志
    try:
        from orcalink_client import setup_logging as setup_orcalink_logging

        setup_orcalink_logging(verbose=verbose_logging, use_root_handler=True)
    except ImportError:
        # 如果 orcalink_client 未安装，跳过
        pass
