#!/usr/bin/env python3
"""布料三进程联调启动入口：组装参数 → 调用编排器 run_p23c。

编排逻辑在 envs.softbody.attach_coupling.run_p23c 中，本文件只负责从环境变量组装参数。
"""
from __future__ import annotations

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


def _env_or(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def resolve_defaults() -> dict[str, str]:
    """复刻 shell 的 DEBUG 分支默认值（每项都尊重用户显式覆盖）。"""
    debug = _env_or("DEBUG", "0") == "1"

    def g(name: str, d1: str, d0: str) -> str:
        return _env_or(name, d1 if debug else d0)

    return {
        "CLOTH_DEBUG": g("CLOTH_DEBUG", "1", "0"),
        "COLLECT_DATA": g("COLLECT_DATA", "1", "0"),
        "XPBD_UI": g("XPBD_UI", "1", "1"),
        "CLOTH_SYNC_STUDIO_VIS": g("CLOTH_SYNC_STUDIO_VIS", "1", "1"),
        "CLOTH_NO_REALTIME": g("CLOTH_NO_REALTIME", "0", "1"),
        "MAX_MACRO_FRAMES": g("MAX_MACRO_FRAMES", "", "800"),
        "MAX_SEC": g("MAX_SEC", "120", "120"),
    }


def main() -> int:
    cfg = resolve_defaults()
    if os.environ.get("SHOW_UI", "").strip():          # 兼容旧变量
        cfg["XPBD_UI"] = os.environ["SHOW_UI"]

    from envs.softbody.attach_coupling import P23cParams, run_p23c

    return run_p23c(
        P23cParams(
            repo_root=REPO_ROOT,
            base_dir=TELE_DIR,
            log_dir=TELE_DIR / "logs",
            cloth_data_dir=TELE_DIR,
            level=_env_or("LEVEL", _env_or("ORCA_LEVEL_NAME", "")) or None,
            agent=_env_or("AGENT", "openloong"),
            mjc_prefix=_env_or("MJC_PREFIX", "openloong_gripper_2f85_fix_base_usda"),
            config_path=(os.environ.get("CFG") or os.environ.get("CLOTH_CONFIG") or "").strip(),
            orcagym_port=int(_env_or("ORCAGYM_PORT", "50051")),
            pbd_grpc_port=int(_env_or("PBD_GRPC_PORT", "50263")),
            orcalink_port=int(_env_or("ORCALINK_PORT", "50361")),
            pico_port=int(_env_or("PICO_PORT", "8001")),
            wait_sec=int(_env_or("WAIT_SEC", "180")),
            kill_stale=_env_or("KILL_STALE", "1") == "1" and not _env_flag("SKIP_STALE_KILL", False),
            auto_start_studio=_env_flag("AUTO_START_STUDIO", False),
            collect_data=cfg["COLLECT_DATA"] == "1",
            cloth_debug=cfg["CLOTH_DEBUG"] == "1",
            xpbd_ui=cfg["XPBD_UI"] == "1",
            cloth_sync_studio_vis=cfg["CLOTH_SYNC_STUDIO_VIS"] == "1",
            cloth_no_realtime=cfg["CLOTH_NO_REALTIME"] == "1",
            max_macro_frames=int(cfg["MAX_MACRO_FRAMES"]) if cfg["MAX_MACRO_FRAMES"] else None,
            max_sec=int(cfg["MAX_SEC"]),
            xpbd_auto_build=_env_flag("XPBD_AUTO_BUILD", True),
            xpbd_build_target=_env_or("XPBD_BUILD_TARGET", ""),
            agent_user_set=bool(os.environ.get("AGENT", "").strip()),
            config_explicit=bool(os.environ.get("CFG") or os.environ.get("CLOTH_CONFIG")),
            mujoco_viewer=_env_or("MUJOCO_VIEWER", _env_or("GUI", "0")) == "1",
            bench_json=os.environ.get("BENCH_JSON", "").strip(),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
