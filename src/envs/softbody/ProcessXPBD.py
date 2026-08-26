"""XPBD 相关：清理旧进程 / 准备二进制 / 启动 XPBD dual_gripper_cross_mjc 子进程。

合并自原 xpbd_process.py 与 PrepareXPBD.py。
对外接口（按生命周期顺序）：cleanup → prepare → start_xpbd_if_configured。
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..cpu_affinity import wrap_cmd_with_taskset
from .base.process_utils import ProcessManager, _subprocess_preexec
from .domain.paths import XPBD_ROOT, resolve_cloth_data_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

PYPI_INDEX = "https://pypi.org/simple/"              # 官方主源（通用依赖）
TEST_PYPI_EXTRA = "https://test.pypi.org/simple/"    # orca-* 包的 test 源


# ---------------------------------------------------------------------------
# 1) 清理旧进程（cleanup）
# ---------------------------------------------------------------------------

def pgrep(pattern: str, *, exact: bool = False) -> list[int]:
    """pgrep（默认 -f，exact=True 时 -x），返回 PID 列表（找不到则空）。"""
    flag = "-x" if exact else "-f"
    try:
        out = subprocess.run(
            ["pgrep", flag, pattern], capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return []
    if out.returncode != 0:
        return []
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 存在但无权限，视为存活
    except OSError:
        return False


def _try_kill(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def kill_pids_gracefully(label: str, pids: list[int]) -> None:
    """先 SIGTERM，等 1s 后对仍存活的补 SIGKILL。"""
    me = os.getpid()
    alive = [p for p in pids if p != me and pid_alive(p)]
    if not alive:
        return

    for pid in alive:
        print(f"[ProcessXPBD] 结束 {label} pid={pid}", flush=True)
        _try_kill(pid, signal.SIGTERM)

    time.sleep(1)
    for pid in alive:
        if pid_alive(pid):
            print(f"[ProcessXPBD] 强制结束 {label} pid={pid}", flush=True)
            _try_kill(pid, signal.SIGKILL)


def cleanup(target: str | None = None) -> None:
    """清理旧的 XPBD 进程（按 target 进程名匹配）。"""
    target = (target or os.environ.get("XPBD_BUILD_TARGET", "")).strip()
    if not target:
        raise RuntimeError("缺少 XPBD target（XPBD_BUILD_TARGET 未设置）")
    pids = pgrep(target)
    if not pids:
        print(f"[ProcessXPBD] 无陈旧 XPBD 进程（{target}）", flush=True)
        return
    print(f"[ProcessXPBD] 清理陈旧 XPBD 进程（{target}）...", flush=True)
    kill_pids_gracefully("xpbd", pids)


# ---------------------------------------------------------------------------
# 2) 准备二进制（prepare）
# ---------------------------------------------------------------------------

def env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _ensure_orca_xpbd_installed(version: str) -> str:
    """确保当前 Python 环境已安装指定版本的 orca-xpbd，返回 __version__。"""
    try:
        import orcaxpbd_client  # noqa: WPS433
        installed = str(getattr(orcaxpbd_client, "__version__", "")).strip()
        if installed and installed == version:
            return installed
    except ImportError:
        installed = ""

    if not env_flag("XPBD_PIP_AUTO_INSTALL", default=True):
        raise RuntimeError(
            f"orca-xpbd 版本不匹配或未安装（已装 {installed!r}，期望 {version!r}）；"
            f"请先: pip install -i {PYPI_INDEX} --extra-index-url {TEST_PYPI_EXTRA} "
            f"orca-xpbd=={version}"
        )

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--upgrade", "--force-reinstall",
        "-i", PYPI_INDEX, "--extra-index-url", TEST_PYPI_EXTRA,
        f"orca-xpbd=={version}",
    ]
    print(f"[ProcessXPBD] pip install orca-xpbd=={version}", flush=True)
    subprocess.run(cmd, check=True)

    import orcaxpbd_client  # noqa: WPS433
    return str(getattr(orcaxpbd_client, "__version__", version)).strip() or version


def prepare(target: str | None = None, *, debug_bin_path: str = "") -> int:
    """确保 orca-xpbd 包已安装（含 XPBD 可执行文件）；返回 0 成功。

    debug_bin_path 非空（debug 模式）时跳过 pip 安装，直接用本地编译二进制。
    """
    target = (target or os.environ.get("XPBD_BUILD_TARGET", "")).strip()
    debug_bin = (debug_bin_path or os.environ.get("XPBD_DEBUG_BIN_PATH", "")).strip()
    if not target:
        raise RuntimeError("缺少 XPBD target（XPBD_BUILD_TARGET 未设置）")
    if debug_bin:
        print(f"[ProcessXPBD] debug 模式：跳过 orca-xpbd pip 安装（本地二进制 {debug_bin}）", flush=True)
        return 0
    version = os.environ.get("ORCA_XPBD_VERSION", "").strip()
    if not version:
        raise RuntimeError("缺少 orca-xpbd 版本（ORCA_XPBD_VERSION 未设置）")
    print(f"[ProcessXPBD] 确保 orca-xpbd 已安装（{target}）...", flush=True)
    try:
        installed = _ensure_orca_xpbd_installed(version)
        if installed != version:
            print(f"[ProcessXPBD] WARN: 已装 orca-xpbd {installed}，期望 {version}", flush=True)
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[ProcessXPBD] FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


def ensure_prepared(target: str, version: str, *, debug_bin_path: str = "") -> bool:
    """清理旧 XPBD 进程 + 确保 orca-xpbd 已安装（供编排器 step 7 调用，类似 ProcessStudio.ensure_prepared）。

    Args:
        target: XPBD 可执行 target 名（如 dual_gripper_g1_cook2）。
        version: 期望的 orca-xpbd 版本。
        debug_bin_path: debug 模式指定的本地编译二进制路径；非空时跳过 pip 安装。

    Returns:
        是否准备成功（prepare == 0）。
    """
    os.environ["XPBD_BUILD_TARGET"] = target
    os.environ.setdefault("ORCA_XPBD_VERSION", version)
    cleanup(target)
    return prepare(target, debug_bin_path=debug_bin_path) == 0


# ---------------------------------------------------------------------------
# 3) 启动进程（start）
# ---------------------------------------------------------------------------

def _resolve_xpbd_executable_from_pip(target: str) -> Optional[Path]:
    """借鉴 orcaxpbd run CLI 的思想：用 orcaxpbd_client.binary_runner.get_binary_path()
    从 pip 包定位二进制路径，但保留联调原有的 Popen 启动方式（cwd=XPBD_ROOT、
    30+ 环境变量、stdout 重定向到 log_file、ProcessManager 进程管理）。

    不直接用 `orcaxpbd run` CLI 是因为：
    1. CLI 用 subprocess.run 阻塞，联调需 Popen 异步
    2. CLI 的 cwd=PACKAGE_DIR 会破坏相对路径资源加载
    3. CLI 不传 30+ 个 MJC_PBD_* / PBD_* / PBDX_* 环境变量
    4. CLI 不支持 stdout 重定向到 log_file
    5. CLI 不支持 ProcessManager 注册

    默认启用 pip 包二进制（无需环境变量）。
    通过 ORCAXPBD_USE_PIP_PACKAGE=0 显式禁用，返回 None。
    pip 包未安装或目标不存在时返回 None（由调用方抛错）。
    """
    env_val = os.environ.get("ORCAXPBD_USE_PIP_PACKAGE", "1").strip().lower()
    logger.info("[pip_diag] _resolve_xpbd_executable_from_pip called: target=%r, ORCAXPBD_USE_PIP_PACKAGE=%r", target, env_val)
    if env_val in ("0", "false", "no", "off"):
        logger.info("[pip_diag] ORCAXPBD_USE_PIP_PACKAGE=%r 显式禁用，返回 None", env_val)
        return None
    logger.info("[pip_diag] pip 包模式启用，尝试从 orcaxpbd_client 解析二进制路径")
    try:
        from orcaxpbd_client.binary_runner import get_binary_path
        logger.info("[pip_diag] orcaxpbd_client.binary_runner 导入成功")
    except ImportError as exc:
        logger.info("[pip_diag] orcaxpbd_client 未安装（import 失败: %s），返回 None", exc)
        return None
    name = Path(target).name
    logger.info("[pip_diag] 查询 pip 包内二进制: name=%r", name)
    try:
        path = get_binary_path(name)
        logger.info("[pip_diag] get_binary_path 返回: %s (exists=%s, exec=%s)",
                    path, path.is_file(), os.access(path, os.X_OK) if path.is_file() else False)
    except (FileNotFoundError, ValueError) as exc:
        logger.info("[pip_diag] pip 包内未找到目标 %s: %s，返回 None", name, exc)
        return None
    logger.info("XPBD 使用 pip 包二进制: %s", path)
    return path


def _resolve_debug_bin_path(xpbd_cfg: Dict[str, Any]) -> Optional[Path]:
    """debug 模式：xpbd.debug_bin_path（或环境变量 XPBD_DEBUG_BIN_PATH）指向本地编译的
    XPBD 二进制时，直接返回其绝对路径，跳过 pip 包解析与「发布」流程。

    相对路径按 ``XPBD_ROOT/build/`` 解析（build.sh 的 BUILD_DIR=build，产物名即 target 名）。
    未配置、文件不存在或不可执行时返回 None（回落到 pip 包解析）。
    """
    raw = os.environ.get("XPBD_DEBUG_BIN_PATH", "").strip()
    if not raw:
        raw = str(xpbd_cfg.get("debug_bin_path", "")).strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_absolute():
        p = p.resolve()
    else:
        p = (XPBD_ROOT / "build" / p).resolve()
    if not p.is_file():
        logger.warning("XPBD debug 二进制不存在，回落到 pip 包: %s", p)
        return None
    if not os.access(p, os.X_OK):
        logger.warning("XPBD debug 二进制不可执行，回落到 pip 包: %s", p)
        return None
    logger.info("XPBD debug 模式使用本地编译二进制: %s", p)
    return p


def _resolve_mjc_pbd_config(config: Dict[str, Any], config_path: Path, cloth_data_dir: Path | None = None) -> Path:
    """
    MJC_PBD_CONFIG 路径。

    debug 会话（cloth_sim_session.json / cloth_debug_*）须优先于 xpbd.config_path，
    否则 XPBD 会加载 dual_gripper_cross_full.json 的 export_csv=false。
    """
    if config_path.is_file():
        resolved = config_path.resolve()
        if resolved.name.startswith("cloth_sim_session_") or resolved.name == "cloth_sim_session.json":
            return resolved
        if "cloth_debug_" in str(resolved.parent) and resolved.name == "cloth_sim_session.json":
            try:
                meta = json.loads(resolved.read_text(encoding="utf-8"))
                sp = meta.get("session_config")
                if sp and Path(str(sp)).is_file():
                    return Path(str(sp)).resolve()
            except (OSError, json.JSONDecodeError):
                pass
    xpbd_cfg = config.get("xpbd", {})
    rel = xpbd_cfg.get("config_path")
    if rel:
        p = Path(str(rel))
        if p.is_file():
            return p.resolve()
        data_dir = resolve_cloth_data_dir(cloth_data_dir)
        candidate = (data_dir / p).resolve()
        if candidate.is_file():
            return candidate
    if config_path.is_file():
        return config_path.resolve()
    raise FileNotFoundError(f"cloth config not found for MJC_PBD_CONFIG: {config_path}")


def start_xpbd_if_configured(
    config: Dict[str, Any],
    *,
    config_path: Path,
    process_manager: ProcessManager,
    log_dir: Optional[Path] = None,
    session_timestamp: str = "cloth",
    cpu_affinity: Optional[str] = None,
    cloth_data_dir: Path | None = None,
) -> bool:
    """
    若 xpbd.enabled 且 auto_start，启动 dual_gripper_cross_mjc。

    环境变量：
    - MJC_PBD_CONFIG：cloth_sim JSON 绝对路径
    - PBD_GRPC=1：当 particle_render.enabled 时推顶点至 Studio
    """
    xpbd_cfg = config.get("xpbd", {})
    if not (xpbd_cfg.get("enabled", False) and xpbd_cfg.get("auto_start", False)):
        return False

    target = str(xpbd_cfg.get("executable", "dual_gripper_cross_mjc"))
    exe = _resolve_debug_bin_path(xpbd_cfg) or _resolve_xpbd_executable_from_pip(target)
    if exe is None:
        raise FileNotFoundError(
            f"XPBD 二进制不可用: {target!r}（pip 包 orcaxpbd_client 未安装或未含该 target；"
            f"或设置 xpbd.debug_bin_path / XPBD_DEBUG_BIN_PATH 指向本地编译的二进制）"
        )
    mjc_pbd_config = _resolve_mjc_pbd_config(config, config_path, cloth_data_dir=cloth_data_dir)

    env = os.environ.copy()
    # 全链联调须 JoinSession；勿继承 shell 冒烟遗留的 MJC_PBD_NO_ORCALINK=1
    env.pop("MJC_PBD_NO_ORCALINK", None)
    env.pop("MJC_PBD_LOCAL_PHYS_SMOKE", None)
    env["MJC_PBD_CONFIG"] = str(mjc_pbd_config)
    env["MJC_PBD_TRY_CONNECT_SEC"] = str(xpbd_cfg.get("try_connect_sec", 3))
    sim_cfg = config.get("simulation", {})
    max_sim = float(sim_cfg.get("max_sim_time", 0) or 0)
    discover_only = bool(xpbd_cfg.get("cloth_discover_only", True))
    if not discover_only:
        # 全链路：由 cloth_mujoco 宏步驱动 sync；body_track 在 C 端 unpause poll
        env["MJC_PBD_UI_SIM"] = "0"
    disable_base_phys = bool(xpbd_cfg.get("disable_base_phys", False)) or not discover_only
    if disable_base_phys:
        env["MJC_PBD_DISABLE_BASE_PHYS"] = "1"
        logger.info("XPBD MJC_PBD_DISABLE_BASE_PHYS=1 (no v4 fake table)")

    dg_traj = xpbd_cfg.get("dg_traj")
    if dg_traj:
        env["MJC_PBD_DG_TRAJ"] = str(dg_traj)
        logger.info("XPBD MJC_PBD_DG_TRAJ=%s", dg_traj)
        if str(dg_traj).strip() == "pico":
            trigger_path = env.get("MJC_PBD_GRIP_TRIGGER_PATH", "").strip()
            if trigger_path:
                logger.info("XPBD MJC_PBD_GRIP_TRIGGER_PATH=%s", trigger_path)
            else:
                logger.warning(
                    "XPBD pico 模式但 MJC_PBD_GRIP_TRIGGER_PATH 未设置（扳机文件无法读取）"
                )
    elif discover_only and max_sim >= 60.0:
        env["MJC_PBD_DG_TRAJ"] = "full"
        logger.info("XPBD MJC_PBD_DG_TRAJ=full (max_sim_time=%.1fs)", max_sim)
    elif not discover_only:
        logger.info("XPBD builtin traj off (cloth_discover_only=false, OrcaLink body_track)")

    # ClothRobot / 全链联调：默认 GS（R-PHYS-4：Jacobi 布–刚接触曾穿台）；可用 xpbd.force_gs_solver 覆盖。
    force_gs_cfg = xpbd_cfg.get("force_gs_solver")
    if force_gs_cfg is None:
        force_gs = not discover_only
    else:
        force_gs = bool(force_gs_cfg)
    if force_gs:
        env["PBDX_FORCE_GS_ONLY"] = "1"
        env["PBDX_SOLVER"] = "gs"
        env.pop("PBDX_JACOBI_CONTACT", None)
        logger.info(
            "XPBD 布求解器=GS（PBDX_FORCE_GS_ONLY=1 PBDX_SOLVER=gs；禁用 register_jacobi_cloth）"
        )

    show_ui = bool(xpbd_cfg.get("show_ui", True))
    xpbd_ui_env = os.environ.get("XPBD_UI", "").strip().lower()
    show_ui_env = os.environ.get("SHOW_UI", "").strip().lower()
    if xpbd_ui_env in ("0", "false", "no"):
        show_ui = False
    elif xpbd_ui_env in ("1", "true", "yes"):
        show_ui = True
    elif show_ui_env in ("0", "false", "no"):
        show_ui = False
    elif show_ui_env in ("1", "true", "yes"):
        show_ui = True
    if show_ui:
        env.pop("MJC_PBD_NO_UI", None)
        logger.info("XPBD XPBD_UI=1（OpenGL 窗口 + dbgdraw）")
    else:
        env["MJC_PBD_NO_UI"] = "1"
        logger.info("XPBD XPBD_UI=0（MJC_PBD_NO_UI=1，无窗口）")

    pr = config.get("particle_render", {})
    overlay = xpbd_cfg.get("overlay_mjc", True)
    overlay_env = os.environ.get("MJC_PBD_OVERLAY_MJC", "").strip().lower()
    if overlay_env in ("0", "false", "no"):
        overlay = False
    elif overlay_env in ("1", "true", "yes"):
        overlay = True
    if overlay:
        env["MJC_PBD_OVERLAY_MJC"] = "1"
    else:
        env.pop("MJC_PBD_OVERLAY_MJC", None)
        logger.info("XPBD MJC_PBD_OVERLAY_MJC=0（无 overlay 快照）")

    pbd_grpc_env = os.environ.get("PBD_GRPC", "").strip().lower()
    if pbd_grpc_env in ("0", "false", "no"):
        env["PBD_GRPC"] = "0"
        logger.info("XPBD PBD_GRPC=0（跳过 Studio 布料 gRPC UpdateMesh）")
    elif pr.get("enabled", False):
        env.setdefault("PBD_GRPC", "1")
        # 环境变量优先（test20260508 PBDRender 常为 :50261）
        grpc_addr = os.environ.get("PBD_GRPC_ADDRESS", "").strip()
        if not grpc_addr:
            grpc_addr = str(pr.get("grpc_address", "localhost:50261"))
        if not grpc_addr.startswith("localhost:") and ":" not in grpc_addr:
            grpc_addr = f"localhost:{grpc_addr}"
        env["PBD_GRPC_ADDRESS"] = grpc_addr
        mesh_id = pr.get("mesh_id")
        if mesh_id is not None and str(mesh_id).strip() != "":
            env["PBD_GRPC_MESH_ID"] = str(int(mesh_id))
            logger.info("XPBD PBD_GRPC_MESH_ID=%s", env["PBD_GRPC_MESH_ID"])
        sbt_rot = os.environ.get("PBD_GRPC_SBT_ROTATION", "").strip()
        if not sbt_rot:
            sbt_rot = str(pr.get("sbt_rotation", "from_quat")).strip()
        if sbt_rot:
            env["PBD_GRPC_SBT_ROTATION"] = sbt_rot
            logger.info("XPBD PBD_GRPC_SBT_ROTATION=%s", sbt_rot)
        logger.info("XPBD PBD_GRPC=1 -> Studio PBDRender %s", grpc_addr)

    cloth_cfg = config.get("cloth", {})
    finger_close_ratio = cloth_cfg.get("finger_close_ratio")
    if finger_close_ratio is not None:
        env["DG_FINGER_CLOSE_RATIO"] = str(finger_close_ratio)
        logger.info("XPBD DG_FINGER_CLOSE_RATIO=%s", env["DG_FINGER_CLOSE_RATIO"])
    trigger_close_thresh = cloth_cfg.get("trigger_close_thresh")
    if trigger_close_thresh is not None:
        env["DG_TRIGGER_CLOSE_THRESH"] = str(trigger_close_thresh)
        logger.info("XPBD DG_TRIGGER_CLOSE_THRESH=%s", env["DG_TRIGGER_CLOSE_THRESH"])
    lock_radius_m = cloth_cfg.get("lock_radius_m")
    if lock_radius_m is not None:
        env["DG_LOCK_RADIUS_M"] = str(lock_radius_m)
        logger.info("XPBD DG_LOCK_RADIUS_M=%s", env["DG_LOCK_RADIUS_M"])
    max_lock_approach_m = cloth_cfg.get("max_lock_approach_m")
    if max_lock_approach_m is not None:
        env["DG_MAX_LOCK_APPROACH_M"] = str(max_lock_approach_m)
        logger.info("XPBD DG_MAX_LOCK_APPROACH_M=%s", env["DG_MAX_LOCK_APPROACH_M"])

    particle_friction = os.environ.get("PARTICLE_FRICTION", "").strip()
    if not particle_friction:
        pf_cfg = cloth_cfg.get("particle_friction")
        if pf_cfg is not None:
            env["PARTICLE_FRICTION"] = str(pf_cfg)
            logger.info("XPBD PARTICLE_FRICTION=%s", env["PARTICLE_FRICTION"])

    args: list[str] = []
    for arg in xpbd_cfg.get("args", []):
        args.append(str(arg).replace("{config_path}", str(mjc_pbd_config)))

    # gdb 定位模式（XPBD_UNDER_GDB=1）：崩溃时自动打印 backtrace 到日志。
    under_gdb = os.environ.get("XPBD_UNDER_GDB", "").strip().lower() in ("1", "true", "yes")
    base_cmd = (["gdb", "-batch",
                 "-ex", "run",
                 "-ex", "p check->d",
                 "-ex", "p check->n",
                 "-ex", "p diff_c",
                 "-ex", "p check->r1",
                 "-ex", "p check->r2",
                 "-ex", "p b1->is_particle",
                 "-ex", "p b2->is_particle",
                 "-ex", "p static_friction",
                 "-ex", "p dynamic_friction",
                 "-ex", "bt",
                 "--args", str(exe)]
                if under_gdb else ["stdbuf", "-oL", "-eL", str(exe)])

    # C 端 stdout 重定向到文件时为全缓冲；stdbuf 保证联调日志实时可见。
    cmd = wrap_cmd_with_taskset(base_cmd + args, cpu_affinity)
    if cpu_affinity:
        logger.info("📌 XPBD CPU 亲和性: 核心 %s", cpu_affinity)
    logger.info("启动 XPBD: %s", " ".join(cmd))
    logger.info("MJC_PBD_CONFIG=%s", mjc_pbd_config)
    cloth_blk = config.get("cloth") or {}
    if cloth_blk.get("asset_dir"):
        logger.info("cloth.asset_dir=%s level=%s", cloth_blk.get("asset_dir"), cloth_blk.get("level"))

    log_file = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"xpbd_{session_timestamp}.log"

    # dual_gripper_cross_mjc 从 session cloth.asset_dir 加载掩码布；须在 XPBD 根目录运行
    xpbd_cwd = XPBD_ROOT if XPBD_ROOT.is_dir() else exe.parent.parent
    logger.info("XPBD cwd=%s", xpbd_cwd)

    if log_file:
        log_handle = open(log_file, "w", buffering=1)
        proc = subprocess.Popen(
            cmd,
            cwd=str(xpbd_cwd),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=_subprocess_preexec,
        )
        proc.log_file = log_handle
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=str(xpbd_cwd),
            env=env,
            preexec_fn=_subprocess_preexec,
        )

    process_manager.processes["XPBD"] = proc
    logger.info("XPBD 已启动 pid=%s", proc.pid)

    delay = float(xpbd_cfg.get("startup_delay", 4.0))
    if delay > 0:
        logger.info("等待 XPBD 初始化 %.1fs...", delay)
        time.sleep(delay)
    return True
