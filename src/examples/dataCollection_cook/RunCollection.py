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


def _load_config() -> dict:
    """加载整个 Config.json。"""
    path = TELE_DIR / "Config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
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
            orcalink_port=int(_env("ORCALINK_PORT", "orcalink_port")),
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


if __name__ == "__main__":
    raise SystemExit(main())
