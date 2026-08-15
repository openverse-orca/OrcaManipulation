#!/usr/bin/env python3
"""布料三进程联调编排入口（替代 run_cloth_robot_p23c.sh 的主干逻辑）。

四段主干：
  1) 解析环境/配置：Python(conda)、CFG/LEVEL/AGENT/MJC_PREFIX、XPBD 二进制
  2) 进程/端口就绪：清旧进程、确保 Studio、等端口
  3) 同步 XPBD session：refresh + export scene
  4) 启动 data_collection_cloth_tele.py

前置：需在已激活 conda orca-PriWaic 的环境下运行（4-RunClothRobotP23c.sh 已做）。
环境变量与 run_cloth_robot_p23c.sh 完全一致。
"""
from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

LOG_PREFIX = "[P2.3c]"


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


def _repo_root() -> Path:
    env = os.environ.get("REPO_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


REPO_ROOT = _repo_root()
TELE_DIR = Path(__file__).resolve().parent

for _p in (
    str(REPO_ROOT / "OrcaLink" / "Client" / "Python"),
    str(REPO_ROOT / "OrcaGym"),
    str(REPO_ROOT / "OrcaManipulation" / "src"),
    str(TELE_DIR),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 联调用 Python：默认用当前解释器（4-RunClothRobotP23c.sh 已 conda activate）。
# 若显式设置 PYTHON，需为「单个可执行文件路径」（不支持 conda run 命令串）。
PYTHON = os.environ.get("PYTHON", "").strip() or sys.executable

from envs.softbody.ProcessXPBD import (  # noqa: E402
    DEFAULT_TARGET,
    env_flag,
    kill_pids_gracefully,
    pgrep,
    pid_alive,
)


def _env_or(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


# ============================================================
# 进程 / 端口工具（复刻 shell 的 pgrep/ss/kill 语义）
# ============================================================

def _pids_on_tcp_port(port: int) -> list[int]:
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


def kill_stale_cloth_processes(orcalink_port: int, pico_port: int) -> None:
    tele_pids = pgrep(r"data_collection_cloth_tele\.py")
    orca_pids = _pids_on_tcp_port(orcalink_port)
    pico_pids = _pids_on_tcp_port(pico_port)

    if not (tele_pids or orca_pids or pico_pids):
        log(f"无陈旧 cloth 联调进程（:{orcalink_port} / :{pico_port}）")
        return

    log(f"清理陈旧 cloth 联调进程（OrcaLink :{orcalink_port}、Pico :{pico_port}）...")
    # 先停 tele，再停端口监听，避免 OrcaLink session 半开。
    if tele_pids:
        kill_pids_gracefully("data_collection_cloth_tele", tele_pids)
    kill_pids_gracefully(f"orcalink(:{orcalink_port})", _pids_on_tcp_port(orcalink_port))
    kill_pids_gracefully(f"Pico(:{pico_port})", _pids_on_tcp_port(pico_port))

    time.sleep(1)
    if _pids_on_tcp_port(orcalink_port) or _pids_on_tcp_port(pico_port):
        log(f"WARN: :{orcalink_port} 或 :{pico_port} 仍被占用，请手动检查: ss -tlnp | grep -E '{orcalink_port}|{pico_port}'")
    else:
        log(f"陈旧进程已清理，:{orcalink_port} / :{pico_port} 已释放")


def wait_port(port: int, label: str, max_sec: int) -> bool:
    log(f"等待 {label} localhost:{port} (最多 {max_sec}s)...")
    waited = 0
    while waited < max_sec:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                log(f"OK {label} :{port}")
                return True
        except OSError:
            pass
        time.sleep(2)
        waited += 2
    log(f"超时：{label} :{port} 未监听")
    return False


def _run_with_tee(cmd: list[str], log_path: Path, cwd: Path) -> int:
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, cwd=str(cwd))
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
        proc.wait()
        return proc.returncode


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
    # ---------- 1. 解析环境 / 配置 ----------
    cfg = resolve_defaults()
    if os.environ.get("SHOW_UI", "").strip():          # 兼容旧变量
        cfg["XPBD_UI"] = os.environ["SHOW_UI"]

    agent_user_set = bool(os.environ.get("AGENT", "").strip())
    agent = _env_or("AGENT", "openloong")
    mjc_prefix = _env_or("MJC_PREFIX", "openloong_gripper_2f85_fix_base_usda")

    orcagym_port = int(_env_or("ORCAGYM_PORT", "50051"))
    pbd_grpc_port = int(_env_or("PBD_GRPC_PORT", "50263"))
    orcalink_port = int(_env_or("ORCALINK_PORT", "50361"))
    pico_port = int(_env_or("PICO_PORT", "8001"))
    wait_sec = int(_env_or("WAIT_SEC", "180"))
    kill_stale = _env_or("KILL_STALE", "1") == "1"
    if env_flag("SKIP_STALE_KILL", False):
        kill_stale = False
    auto_start_studio = env_flag("AUTO_START_STUDIO", False)

    # export 环境变量（下游 tele / XPBD 读取）
    os.environ["PYTHONPATH"] = (
        f"{REPO_ROOT / 'OrcaLink' / 'Client' / 'Python'}:"
        f"{REPO_ROOT / 'OrcaGym'}:"
        f"{REPO_ROOT / 'OrcaManipulation' / 'src'}:"
        + os.environ.get("PYTHONPATH", "")
    )
    os.environ["PBD_GRPC_ADDRESS"] = _env_or("PBD_GRPC_ADDRESS", f"localhost:{pbd_grpc_port}")
    os.environ["XPBD_UI"] = cfg["XPBD_UI"]
    os.environ.setdefault("PBDX_FORCE_GS_ONLY", "1")
    os.environ.setdefault("PBDX_SOLVER", "gs")

    # LEVEL / CFG（resolve_cloth_level 内部会兜底自动检测 Studio 关卡）
    from envs.softbody.common.paths import resolve_cloth_config_path, resolve_cloth_level

    level_raw = _env_or("LEVEL", _env_or("ORCA_LEVEL_NAME", "")) or None
    level = resolve_cloth_level(level_raw)

    debug = cfg["CLOTH_DEBUG"] == "1"
    cfg_explicit = bool(os.environ.get("CFG") or os.environ.get("CLOTH_CONFIG"))
    cfg_path = (os.environ.get("CFG") or os.environ.get("CLOTH_CONFIG") or "").strip()
    if not cfg_path:
        cfg_path = str(resolve_cloth_config_path(level=level, agent=agent, debug=debug))

    # AGENT / MJC_PREFIX 检测（MJCF 扫描，直接 import 函数）
    from envs.softbody.ProcessOrcaGym import scan_tele_layout_from_mjcf
    from envs.softbody.ProcessStudio import ensure_ready, find_latest_studio_mjcf_path

    mjcf = find_latest_studio_mjcf_path()
    if mjcf is not None and mjcf.is_file():
        try:
            layout = scan_tele_layout_from_mjcf(mjcf)
            detected_agent = getattr(layout, "tele_agent_name", "") or ""
            detected_prefix = getattr(layout, "mjc_agent_prefix", "") or ""
        except Exception:
            detected_agent = detected_prefix = ""
        if not agent_user_set:
            if detected_agent and detected_agent != agent:
                log(f"MJCF 扫描 tele_agent={detected_agent}（覆盖默认 openloong）")
                agent, mjc_prefix = detected_agent, detected_prefix
                if not cfg_explicit:
                    cfg_path = str(resolve_cloth_config_path(level=level, agent=agent, debug=debug))
                    log(f"重解析 config={cfg_path}")
        elif detected_prefix and detected_prefix != mjc_prefix:
            log(f"MJCF 扫描 mjc_prefix={detected_prefix}（覆盖默认 {mjc_prefix}）")
            mjc_prefix = detected_prefix

    os.environ["LEVEL"] = level or ""
    os.environ["AGENT"] = agent
    os.environ["MJC_PREFIX"] = mjc_prefix
    os.environ["CFG"] = cfg_path
    os.environ["CLOTH_DEBUG"] = cfg["CLOTH_DEBUG"]

    # CLOTH_SYNC_STUDIO_VIS / XPBD_UI
    if cfg["CLOTH_SYNC_STUDIO_VIS"] == "1":
        os.environ["CLOTH_SYNC_STUDIO_VIS"] = "1"
        os.environ.setdefault("CLOTH_STUDIO_VIS_STRIDE", "1")
        log(f"CLOTH_SYNC_STUDIO_VIS=1：推送 qpos 到视口（stride={os.environ.get('CLOTH_STUDIO_VIS_STRIDE', '1')}）")
    if cfg["XPBD_UI"] == "1":
        os.environ.pop("MJC_PBD_NO_UI", None)
        os.environ.setdefault("DISPLAY", ":0")
        log("XPBD_UI=1：XPBD OpenGL 窗口")
    else:
        os.environ["MJC_PBD_NO_UI"] = "1"
        log("XPBD_UI=0：无 XPBD OpenGL（MJC_PBD_NO_UI=1）")

    # XPBD 二进制（清旧 + 准备）
    if env_flag("XPBD_AUTO_BUILD", True):
        from envs.softbody.ProcessXPBD import cleanup, prepare

        xpbd_target = _env_or("XPBD_BUILD_TARGET", DEFAULT_TARGET)
        os.environ["XPBD_BUILD_TARGET"] = xpbd_target
        cleanup(xpbd_target)
        if prepare(xpbd_target) != 0:
            log("FAIL: XPBD 准备失败")
            return 1

    log(f"repo={REPO_ROOT}")
    log(f"level={level}  agent={agent}  mjc_prefix={mjc_prefix}  config={cfg_path}  "
        f"DEBUG={_env_or('DEBUG', '0')}  CLOTH_DEBUG={cfg['CLOTH_DEBUG']}  COLLECT_DATA={cfg['COLLECT_DATA']}")
    log(f"XPBD_UI={cfg['XPBD_UI']}  CLOTH_SYNC_STUDIO_VIS={cfg['CLOTH_SYNC_STUDIO_VIS']}  "
        f"CLOTH_NO_REALTIME={cfg['CLOTH_NO_REALTIME']}")

    # ---------- 2. 进程 / 端口就绪 ----------
    if kill_stale:
        kill_stale_cloth_processes(orcalink_port, pico_port)
    else:
        log("KILL_STALE=0：跳过陈旧进程清理")

    ensure_ready(auto_start_studio)

    wait_port(orcagym_port, "OrcaGym", wait_sec)
    wait_port(pbd_grpc_port, "PBDRender", wait_sec)

    # ---------- 3. 同步 XPBD session ----------
    from envs.softbody.attach_coupling import build_p2_session_from_mjcf, export_xpbd_scene_for_session

    if mjcf is None or not mjcf.is_file():
        log(f"FAIL: MJCF not found; Studio Play {level} first")
        return 2
    ts = f"p23c_{time.strftime('%Y%m%d_%H%M%S')}"
    log("从 Studio MJCF 刷新 XPBD session + export scene...")
    _, _, session_path = build_p2_session_from_mjcf(
        mjcf, Path(cfg_path), session_timestamp=ts, level=level,
    )
    scene_path = export_xpbd_scene_for_session(session_path)
    log(f"session: {session_path}")
    log(f"scene: {scene_path}")

    # ---------- 4. 启动 编排器 ----------
    tele_args = [
        "--level", level or "",
        "--agent_name", agent,
        "--mjc-agent-prefix", mjc_prefix,
        "--frame-skip", "20",
        "--time-step", "0.001",
        "--cloth-coupling",
        "--cloth-config", cfg_path,
    ]
    if cfg["MAX_MACRO_FRAMES"]:
        tele_args += ["--max-macro-frames", cfg["MAX_MACRO_FRAMES"]]
    else:
        tele_args += ["--max-episode-sec", cfg["MAX_SEC"]]
    if debug:
        tele_args += ["--cloth-debug"]
    if cfg["COLLECT_DATA"] == "0":
        tele_args += ["--no-collect"]
    if cfg["CLOTH_NO_REALTIME"] == "1":
        tele_args += ["--no-realtime"]
        os.environ["CLOTH_NO_REALTIME"] = "1"
    if _env_or("MUJOCO_VIEWER", _env_or("GUI", "0")) == "1":
        tele_args += ["--gui"]
        log("MUJOCO_VIEWER=1：启动 MuJoCo 原生 passive viewer")
    if os.environ.get("BENCH_JSON", "").strip():
        tele_args += ["--bench", os.environ["BENCH_JSON"]]
        log(f"BENCH_JSON={os.environ['BENCH_JSON']}")

    log_dir = TELE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_tag = f"p23c_{time.strftime('%Y%m%d_%H%M%S')}"
    log_path = log_dir / f"{log_tag}_tele.log"

    log("启动 data_collection_cloth_tele（OrcaLink + XPBD + bridge）...")
    cmd = [PYTHON, str(TELE_DIR / "data_collection_cloth_tele.py"), *tele_args]
    start = time.monotonic()
    ret = _run_with_tee(cmd, log_path, cwd=TELE_DIR)
    wall = time.monotonic() - start
    log(f"墙钟耗时: {wall:.2f}s（CLOTH_NO_REALTIME={cfg['CLOTH_NO_REALTIME']}）")
    log(f"完成。日志: {log_path}")
    log(f"XPBD 日志: {log_dir}/xpbd_*.log")
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
