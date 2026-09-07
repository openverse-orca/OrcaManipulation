#!/usr/bin/env python3
"""布料三进程联调启动入口：组装参数 → 调用编排器 Start。

编排逻辑在 envs.softbody.Start 中。本文件从 Config.json 读运行时默认值：多数参数在
run 段，level/agent/mjc_prefix（关卡/机器人身份）在 orcastudio 段；run 段参数可用环境变量覆盖。
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="[P2.3c] %(message)s")


def _repo_root() -> Path:
    env = os.environ.get("REPO_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


REPO_ROOT = _repo_root()
TELE_DIR = Path(__file__).resolve().parent

for _p in (
    str(REPO_ROOT / "OrcaGym"),
    str(REPO_ROOT / "OrcaManipulation" / "src"),
    str(TELE_DIR),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# 本次运行日志目录（精确到秒），主进程日志与 orcalink/xpbd 子进程日志共用
RUN_LOG_DIR = TELE_DIR / "logs" / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
_FILE_HANDLER = logging.FileHandler(RUN_LOG_DIR / "p23c.log", encoding="utf-8")
_FILE_HANDLER.setFormatter(logging.Formatter("[P2.3c] %(message)s"))
logging.getLogger().addHandler(_FILE_HANDLER)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """递归合并 overlay 到 base（dict 深合并，其它类型覆盖）。"""
    out = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _load_config() -> dict:
    """加载 Config.json，并按 agent_file 引用合并机器人参数文件。"""
    path = TELE_DIR / "Config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        agent_file = data.get("agent_file")
        if agent_file:
            agent_path = (path.parent / str(agent_file)).resolve()
            if agent_path.is_file():
                agent_data = json.loads(agent_path.read_text(encoding="utf-8"))
                if isinstance(agent_data, dict):
                    data = _deep_merge(data, agent_data)
        return data
    except (json.JSONDecodeError, OSError):
        return {}


CONFIG = _load_config()
RUN = CONFIG.get("run") if isinstance(CONFIG.get("run"), dict) else {}
ORCASTUDIO = CONFIG.get("orcastudio") if isinstance(CONFIG.get("orcastudio"), dict) else {}


def _env(name: str, key: str, default: str = "") -> str:
    """env 优先 → Config.json run 段 → 硬编码默认。"""
    v = os.environ.get(name, "").strip()
    if v:
        return v
    v = str(RUN.get(key, "")).strip()
    return v or default


def _env_flag(name: str, key: str, default: bool = False) -> bool:
    """布尔开关：env 优先 → Config.json run 段 → 硬编码默认。"""
    raw = os.environ.get(name)
    if raw is not None:
        return raw.strip().lower() not in ("0", "false", "no", "off")
    val = RUN.get(key)
    if val is not None:
        return bool(val)
    return default


# ----- FINGER_SPAN_MONITOR 开始（可整段删除，并删 record_thumb_index_span.py）-----
def _finger_span_monitor_enabled() -> bool:
    """是否在联调里打开拇指–食指间距检测。环境变量 COOK_FINGER_SPAN_MONITOR 可覆盖配置。"""
    return _env_flag("COOK_FINGER_SPAN_MONITOR", "finger_span_monitor", False)


def _finger_span_setup() -> Path | None:
    """给 XPBD 指定统计目录，让它写出 cloth_macro_speed.csv（含 finger_span）。"""
    if not _finger_span_monitor_enabled():
        return None
    existing = os.environ.get("MJC_PBD_CLOTH_STATS_DIR", "").strip()
    stats_dir = Path(existing) if existing else (RUN_LOG_DIR / "finger_span")
    stats_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MJC_PBD_CLOTH_STATS_DIR"] = str(stats_dir)
    logging.getLogger().info("finger_span 检测已开 → %s", stats_dir)
    return stats_dir


def _finger_span_report(stats_dir: Path | None) -> None:
    """联调进程结束后打印间距和即时合拢比例的最小/最大值。"""
    if stats_dir is None:
        return
    cloth_ratio = 0.9
    cloth = CONFIG.get("cloth") if isinstance(CONFIG.get("cloth"), dict) else {}
    try:
        cloth_ratio = float(cloth.get("finger_close_ratio", cloth_ratio))
    except (TypeError, ValueError):
        pass
    from record_thumb_index_span import print_span_summary

    print_span_summary(
        stats_dir / "cloth_macro_speed.csv",
        config_ratio=cloth_ratio,
        scene_scale=20.0,
    )
# ----- FINGER_SPAN_MONITOR 结束 -----


def main() -> int:
    # 透传给 XPBD 二进制的 env（从 run 段读，env 可覆盖）
    for env_name, key in (
        ("PBD_GRPC_SBT_ROTATION", "pbd_grpc_sbt_rotation"),
        ("MJC_PBD_CLOTH_SCALE", "mjc_pbd_cloth_scale"),
    ):
        v = _env(env_name, key)
        if v:
            os.environ[env_name] = v

    from envs.softbody import P23cParams, Start

    span_dir = _finger_span_setup()
    try:
        return Start(
            P23cParams(
                repo_root=REPO_ROOT,
                base_dir=TELE_DIR,
                log_dir=RUN_LOG_DIR,
                cloth_data_dir=TELE_DIR,
                level=str(RUN.get("level") or "").strip() or None,
                agent=str(RUN.get("agent") or "").strip(),
                mjc_prefix=str(RUN.get("mjc_prefix") or "").strip(),
                config_path=os.environ.get("CFG", "").strip() or str(TELE_DIR / "Config.json"),
                orcagym_port=int(_env("ORCAGYM_PORT", "orcagym_port")),
                pbd_grpc_port=int(_env("PBD_GRPC_PORT", "pbd_grpc_port")),
                orcalink_port=int(
                    os.environ.get("ORCALINK_PORT", "").strip()
                    or str((CONFIG.get("orcalink") or {}).get("port", 50361))
                ),
                pico_port=int(_env("PICO_PORT", "pico_port")),
                wait_sec=int(_env("WAIT_SEC", "wait_sec")),
                kill_stale=_env("KILL_STALE", "kill_stale") == "1"
                and not _env_flag("SKIP_STALE_KILL", "skip_stale_kill"),
                collect_data=_env_flag("COLLECT_DATA", "collect_data"),
                xpbd_ui=_env_flag("XPBD_UI", "xpbd_ui"),
                cloth_sync_studio_vis=_env_flag("CLOTH_SYNC_STUDIO_VIS", "cloth_sync_studio_vis"),
                cloth_no_realtime=_env_flag("CLOTH_NO_REALTIME", "cloth_no_realtime"),
                max_macro_frames=int(_env("MAX_MACRO_FRAMES", "max_macro_frames"))
                if _env("MAX_MACRO_FRAMES", "max_macro_frames") else None,
                max_sec=int(_env("MAX_SEC", "max_sec")),
                xpbd_auto_build=_env_flag("XPBD_AUTO_BUILD", "xpbd_auto_build"),
                xpbd_build_target=_env("XPBD_BUILD_TARGET", "xpbd_build_target"),
                mujoco_viewer=_env("MUJOCO_VIEWER", "mujoco_viewer", _env("GUI", "gui")) == "1",
                bench_json=_env("BENCH_JSON", "bench_json"),
            )
        )
    finally:
        _finger_span_report(span_dir)


if __name__ == "__main__":
    raise SystemExit(main())
