#!/usr/bin/env python3
"""布料三进程联调启动入口：组装参数 → 调用编排器 run_p23c。

编排逻辑在 envs.softbody.attach_coupling.run_p23c 中。本文件从 Config.json 的 run 段读运行时默认值，
环境变量可覆盖（用于临时切换关卡/采集等，无需改 Config.json）。
"""
from __future__ import annotations

import json
import logging
import os
import sys
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


def _load_run_config() -> dict:
    """加载 Config.json 的 run 段（运行时默认值）。"""
    path = TELE_DIR / "Config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        run = data.get("run") or {}
        return run if isinstance(run, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


RUN = _load_run_config()


def _env(name: str, key: str, default: str = "") -> str:
    """env 优先 → Config.json run 段 → 硬编码默认。"""
    v = os.environ.get(name, "").strip()
    if v:
        return v
    v = str(RUN.get(key, "")).strip()
    return v or default


def _env_flag(name: str, key: str, default: bool = True) -> bool:
    """布尔开关：env 优先 → Config.json run 段 → 硬编码默认。"""
    raw = os.environ.get(name)
    if raw is not None:
        return raw.strip().lower() not in ("0", "false", "no", "off")
    val = RUN.get(key)
    if val is not None:
        return bool(val)
    return default


def main() -> int:
    # 透传给 XPBD 二进制的 env（从 run 段读，env 可覆盖）
    for env_name, key in (
        ("PBD_GRPC_SBT_ROTATION", "pbd_grpc_sbt_rotation"),
        ("MJC_PBD_CLOTH_SCALE", "mjc_pbd_cloth_scale"),
    ):
        v = _env(env_name, key)
        if v:
            os.environ[env_name] = v

    from envs.softbody.attach_coupling import P23cParams, run_p23c

    return run_p23c(
        P23cParams(
            repo_root=REPO_ROOT,
            base_dir=TELE_DIR,
            log_dir=TELE_DIR / "logs",
            cloth_data_dir=TELE_DIR,
            level=_env("LEVEL", "level") or os.environ.get("ORCA_LEVEL_NAME", "").strip() or None,
            agent=_env("AGENT", "agent", "openloong"),
            mjc_prefix=_env("MJC_PREFIX", "mjc_prefix", ""),
            config_path=(os.environ.get("CFG") or os.environ.get("CLOTH_CONFIG") or "").strip(),
            orcagym_port=int(_env("ORCAGYM_PORT", "orcagym_port", "50051")),
            pbd_grpc_port=int(_env("PBD_GRPC_PORT", "pbd_grpc_port", "50263")),
            orcalink_port=int(_env("ORCALINK_PORT", "orcalink_port", "50361")),
            pico_port=int(_env("PICO_PORT", "pico_port", "8001")),
            wait_sec=int(_env("WAIT_SEC", "wait_sec", "180")),
            kill_stale=_env("KILL_STALE", "kill_stale", "1") == "1"
            and not _env_flag("SKIP_STALE_KILL", "skip_stale_kill", False),
            auto_start_studio=_env_flag("AUTO_START_STUDIO", "auto_start_studio", False),
            collect_data=_env_flag("COLLECT_DATA", "collect_data", False),
            xpbd_ui=_env_flag("XPBD_UI", "xpbd_ui", True),
            cloth_sync_studio_vis=_env_flag("CLOTH_SYNC_STUDIO_VIS", "cloth_sync_studio_vis", True),
            cloth_no_realtime=_env_flag("CLOTH_NO_REALTIME", "cloth_no_realtime", False),
            max_macro_frames=int(_env("MAX_MACRO_FRAMES", "max_macro_frames"))
            if _env("MAX_MACRO_FRAMES", "max_macro_frames") else None,
            max_sec=int(_env("MAX_SEC", "max_sec", "120")),
            xpbd_auto_build=_env_flag("XPBD_AUTO_BUILD", "xpbd_auto_build", True),
            xpbd_build_target=_env("XPBD_BUILD_TARGET", "xpbd_build_target", ""),
            agent_user_set=bool(os.environ.get("AGENT", "").strip()),
            config_explicit=bool(os.environ.get("CFG") or os.environ.get("CLOTH_CONFIG")),
            mujoco_viewer=_env("MUJOCO_VIEWER", "mujoco_viewer", _env("GUI", "gui", "0")) == "1",
            bench_json=_env("BENCH_JSON", "bench_json", ""),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
