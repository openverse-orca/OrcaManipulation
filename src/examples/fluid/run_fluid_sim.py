#!/usr/bin/env python3
"""
Fluid-MuJoCo 耦合仿真示例

【运行前提】
1. 已启动 OrcaStudio 或 OrcaLab（实时 / 录制模式；playback 仅需 OrcaStudio 接收 gRPC）
2. 已加载包含 SPH 标记的流体仿真场景（实时 / 录制）

【运行模式】（--mode）
- live（默认）：粒子经 gRPC 发往 OrcaStudio（与 sph_sim_config.json 中 particle_render 一致；`particle_render.grpc.payload_format` 为 `quantized_packed` 时走方案三量化帧，`raw_fp32` 时走方案四原始 FP32 多批次帧）
- record：将粒子帧写入 HDF5；默认不向 OrcaStudio 推粒子流（避免 Studio 挂起；需要预览时加 --render-particle）。默认路径见下方。record 模式始终并行录制 MuJoCo 全 qpos 并在结束时合并入粒子 HDF5
- playback：不启动 MuJoCo/OrcaSPH，将已有 HDF5 通过 orca-sph 包内 API 发往 OrcaStudio；发下一帧前会等待 Orca 已呈现上一帧（GetRenderedParticleFrame 背压，与 orca-replay-particles --sync-render 一致）

【playback 依赖】
- 已安装 orca-sph（提供 orcasph_client.particle_replay）

【启动模式】
- 自动模式（推荐）：脚本自动启动 OrcaLink 和 OrcaSPH
- 手动模式：用户预先启动服务，脚本仅连接

【使用方法】
    python run_fluid_sim.py
    python run_fluid_sim.py --mode record
    python run_fluid_sim.py --mode record --render-particle     # 录制并向 OrcaStudio 推流预览
    python run_fluid_sim.py --mode record --no-record-stats-plot   # 不弹 matplotlib 统计窗
    python run_fluid_sim.py --mode playback --h5 particle_records/foo_20260101_120000.h5
    python run_fluid_sim.py --mode playback particle_records/foo.h5   # 与 --h5 等价
    python run_fluid_sim.py --config my_config.json
    python run_fluid_sim.py --manual-mode
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# 获取脚本文件所在目录，然后计算项目根目录
# 从 examples/fluid/run_fluid_sim.py 到项目根目录需要向上两级
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.fluid import run_simulation_with_config
from envs.fluid.launch.fluid_session import run_particle_playback_from_config
from envs.fluid.launch.sph_config import setup_python_logging

# Performance stats related imports
import subprocess
import sys


def load_config(config_path: str) -> dict:
    """加载 JSON 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TeeOutput:
    """同时写入多个流（如控制台 + 日志文件）。"""

    def __init__(self, *files: Any) -> None:
        self.files = files

    def write(self, obj: str) -> None:
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self) -> None:
        for f in self.files:
            f.flush()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fluid-MuJoCo 耦合仿真",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
【运行前提】
  1. 已启动 OrcaStudio/OrcaLab 并加载场景（live / record）
  2. 场景中包含带 SPH 标记的刚体（live / record）

【运行模式】
  --mode live      实时发粒子到 Orca（默认）
  --mode record    写入 HDF5（默认路径见下）；HDF5 + MuJoCo qpos ；OrcaStudio 推流需 --render-particle
  --mode playback  仅回放 HDF5 到 Orca（需 --h5 或末尾写 HDF5 路径；目标端口默认同 sph_sim_config）

【启动模式】
  自动模式: 脚本自动启动 OrcaLink 和 OrcaSPH（推荐）
  手动模式: 用户预先启动，脚本仅连接（使用 --manual-mode）

【示例】
  python run_fluid_sim.py                    # 默认 live，无 GUI
  python run_fluid_sim.py --mode record      # 录制 HDF5 + 默认 MuJoCo qpos （不向 OrcaStudio 推流）
  python run_fluid_sim.py --mode record --render-particle  # 同上并向 OrcaStudio 推粒子流预览
  python run_fluid_sim.py --mode playback --h5 particle_records/x.h5
  python run_fluid_sim.py --mode playback particle_records/x.h5
  python run_fluid_sim.py --gui              # 启用 OrcaSPH GUI
  python run_fluid_sim.py --config my.json   # 自定义配置
  python run_fluid_sim.py --manual-mode      # 手动模式
            """,
    )

    parser.add_argument(
        "--mode",
        choices=("live", "record", "playback"),
        default="live",
        help="运行模式：live=实时；record=HDF5 录制；playback=离线回放（不启动耦合仿真）",
    )
    parser.add_argument(
        "--record-output",
        default=None,
        metavar="PATH",
        help="record 模式：HDF5 输出路径（未指定则使用 脚本目录/particle_records/前缀_时间戳.h5）",
    )
    parser.add_argument(
        "--record-prefix",
        default="particle_record",
        help="record 模式默认文件名前缀（仅字母数字下划线连字符）",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=None,
        metavar="HZ",
        help="record 模式：覆盖 recording.record_fps（并与 grpc.update_rate_hz 对齐）",
    )
    parser.add_argument(
        "--h5",
        dest="playback_h5",
        default=None,
        metavar="PATH",
        help="playback 模式：录制的 HDF5 文件（也可在命令末尾写路径，与位置参数等价）",
    )
    parser.add_argument(
        "--playback-target",
        default=None,
        metavar="HOST:PORT",
        help="playback：OrcaStudio ParticleRender gRPC 地址（省略则从 sph 模板读取）",
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=0.0,
        metavar="FPS",
        help="playback 墙钟帧率（0=使用文件 record_fps 属性）",
    )
    parser.add_argument(
        "--config",
        default="fluid_sim_config.json",
        help="配置文件路径 (默认: fluid_sim_config.json)",
    )
    parser.add_argument(
        "--build-mode",
        choices=("debug", "release"),
        default=None,
        help="覆盖配置 build_mode：debug=启用监测 CSV；release=强制关闭 force_position_trace",
    )
    parser.add_argument(
        "--manual-mode",
        action="store_true",
        help="手动模式：禁用自动启动，需预先启动 orcalink 和 orcasph",
    )
    parser.add_argument(
        "--gui",
        "--sph-gui",
        action="store_true",
        help="启用 OrcaSPH（SPlisHSPlasH）原生 GUI 窗口",
    )
    parser.add_argument(
        "--mujoco-gui",
        action="store_true",
        help="启用 MuJoCo 原生被动查看器（短链无 Studio 时用于对照 SPH 同步）",
    )
    parser.add_argument(
        "--mujoco-shutdown-on-close",
        action="store_true",
        help="关闭 MuJoCo 被动查看器窗口时结束整场仿真（默认：与 --gui 同开时不结束，可后台长跑）",
    )
    parser.add_argument(
        "--use-all-cpu",
        action="store_true",
        help="不使用 CPU 亲和性（默认将 OrcaSPH 绑定至 4～末核，为 Orca Studio 保留 0-3）",
    )
    parser.add_argument(
        "--render-particle",
        dest="render_particle",
        action="store_true",
        help="record 模式：向 OrcaStudio 发送粒子 gRPC 以便预览（默认关闭，仅写 HDF5）",
    )
    parser.add_argument(
        "--no-record-stats-plot",
        action="store_true",
        help="record 模式：不启动 matplotlib 录制统计子窗口（默认会启动）",
    )
    parser.add_argument(
        "--record-stats-interval",
        type=float,
        default=5.0,
        metavar="SEC",
        help="record 统计图刷新间隔（秒），默认 5",
    )
    parser.add_argument(
        "--record-stats-window",
        type=float,
        default=5.0,
        metavar="SEC",
        help="滑动窗口 FPS 曲线的时间窗（秒），默认 5",
    )
    parser.add_argument(
        "--orcasph-log",
        default=None,
        metavar="PATH",
        help="手动指定 OrcaSPH 日志路径（用于统计图 tail；手动启动 OrcaSPH 时用）",
    )
    parser.add_argument(
        "--record-stats-skip-head",
        type=int,
        default=5,
        metavar="N",
        help="统计图：跳过开头 N 条 PARTICLE_RECORD_STATS（抑制启动异常；默认 5）",
    )
    parser.add_argument(
        "--record-stats-rolling",
        type=int,
        default=50,
        metavar="N",
        help="统计图：每条曲线最多保留最近 N 个点（默认 50）",
    )
    parser.add_argument(
        "playback_h5_positional",
        nargs="?",
        default=None,
        metavar="H5_FILE",
        help="playback 模式：HDF5 文件路径（与 --h5 等价，可写在命令行末尾）",
    )
    parser.add_argument(
        "--trajectory-record",
        action="store_true",
        help="live 模式：将人类操作（ctrl/mocap/equality 子集）写入 trajectory_records/*.h5",
    )
    parser.add_argument(
        "--trajectory-record-output",
        default=None,
        metavar="PATH",
        help="live：轨迹 HDF5 路径（默认：脚本目录/trajectory_records/前缀_时间戳.h5）",
    )
    parser.add_argument(
        "--trajectory-record-prefix",
        default="trajectory_record",
        help="live 轨迹默认文件名前缀（仅字母数字下划线连字符）",
    )
    parser.add_argument(
        "--trajectory-playback",
        default=None,
        metavar="PATH",
        help="record 模式：从该 HDF5 回放人类操作（在 bridge.step 之后叠加 mocap/eq/ctrl）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        metavar="N",
        help="主循环最大步数（达到后正常退出；0=无限）。无 --gui 时推荐与自动 OrcaLink 联调用以做短程检查",
    )
    parser.add_argument(
        "--bench",
        type=str,
        default=None,
        metavar="PATH",
        help="基准测试 JSON 输出路径（逐圈计时：fluid/step/sleep/pause_rate）",
    )
    parser.add_argument(
        "--enable-performance-stats",
        action="store_true",
        help="启用性能统计功能，每1000个timestep输出一次单行统计数据",
    )
    parser.add_argument(
        "--performance-stats-plot",
        action="store_true",
        help="启用性能统计图表实时显示",
    )
    return parser


def _validate_early_args(args: argparse.Namespace) -> Optional[int]:
    """解析后即可检查的 CLI 错误；通过则返回 None。"""
    if args.playback_h5_positional is not None and args.mode != "playback":
        print(
            f"❌ 末尾的 HDF5 路径仅在 --mode playback 时有效；当前为 {args.mode}。"
            "请改用 playback 或去掉该路径。"
        )
        return 1
    playback_h5 = args.playback_h5 or args.playback_h5_positional
    if args.mode == "playback" and not playback_h5:
        print(
            "❌ playback 模式需要 HDF5：请使用 --h5 PATH，或在命令末尾写 PATH（例如：--mode playback my.h5）"
        )
        return 1
    return None


def _resolve_cpu_affinity(use_all_cpu: bool) -> Optional[str]:
    if use_all_cpu:
        return None
    n = os.cpu_count()
    if n is not None and n > 4:
        return f"4-{n - 1}"
    if n is not None and n <= 4:
        print("⚠️ 逻辑 CPU ≤4，无法保留 0-3 与绑定 OrcaSPH 至 4+，本次不设置 CPU 亲和")
    return None


def _apply_particle_render_run_config(
    args: argparse.Namespace,
    script_dir: Path,
    session_timestamp: str,
    config: dict,
) -> None:
    """写入 config['particle_render_run']（见 sph_sim_config / generate_orcasph_config）。"""
    pr_run: dict = {"mode": args.mode}
    if args.mode == "record":
        prefix = args.record_prefix
        if not re.match(r"^[A-Za-z0-9_-]+$", prefix):
            print("⚠️  --record-prefix 仅允许字母、数字、下划线、连字符，已回退为 particle_record")
            prefix = "particle_record"
        if args.record_output:
            rp = Path(args.record_output).expanduser()
            rp.parent.mkdir(parents=True, exist_ok=True)
            record_path = str(rp.resolve())
        else:
            rec_dir = script_dir / "particle_records"
            rec_dir.mkdir(parents=True, exist_ok=True)
            record_path = str((rec_dir / f"{prefix}_{session_timestamp}.h5").resolve())
        pr_run["record_output_path"] = record_path
        pr_run["record_send_to_studio"] = bool(args.render_particle)
        if args.record_fps is not None:
            pr_run["record_fps"] = args.record_fps
        pr_run["stats_plot"] = {
            "enabled": not args.no_record_stats_plot,
            "interval": args.record_stats_interval,
            "window": args.record_stats_window,
            "orcasph_log": args.orcasph_log,
            "skip_head": args.record_stats_skip_head,
            "rolling": args.record_stats_rolling,
        }
        print("🧩 MuJoCo qpos 录制: 已启用（会话结束合并入粒子 HDF5）")
        print(f"📼 录制 HDF5: {record_path}")
        if pr_run.get("record_send_to_studio"):
            print("📡 OrcaStudio 粒子推流: 已启用（--render-particle）")
        else:
            print(
                "📡 OrcaStudio 粒子推流: 已关闭（默认；需要预览时加 --render-particle）"
            )
    elif args.mode == "playback":
        playback_h5 = args.playback_h5 or args.playback_h5_positional
        pr_run["playback_h5"] = playback_h5
        pr_run["playback_target"] = args.playback_target
        pr_run["playback_fps"] = args.playback_fps
    config["particle_render_run"] = pr_run


def _apply_mujoco_trajectory_config(
    args: argparse.Namespace,
    script_dir: Path,
    session_timestamp: str,
    config: dict,
) -> Optional[int]:
    """写入 config['mujoco_trajectory']；失败时返回退出码。"""
    traj_cfg: dict = {}
    if args.mode == "live" and args.trajectory_record:
        prefix = args.trajectory_record_prefix
        if not re.match(r"^[A-Za-z0-9_-]+$", prefix):
            print("⚠️  --trajectory-record-prefix 仅允许字母、数字、下划线、连字符，已回退为 trajectory_record")
            prefix = "trajectory_record"
        if args.trajectory_record_output:
            tp = Path(args.trajectory_record_output).expanduser()
            tp.parent.mkdir(parents=True, exist_ok=True)
            traj_path = str(tp.resolve())
        else:
            tdir = script_dir / "trajectory_records"
            tdir.mkdir(parents=True, exist_ok=True)
            traj_path = str((tdir / f"{prefix}_{session_timestamp}.h5").resolve())
        traj_cfg["enabled"] = True
        traj_cfg["output_path"] = traj_path
        traj_cfg["prefix"] = prefix
        print(f"📝 MuJoCo 轨迹录制: {traj_path}")
    elif args.mode == "live":
        traj_cfg["enabled"] = False
    if args.mode == "record" and args.trajectory_playback:
        tp = Path(args.trajectory_playback).expanduser()
        if not tp.is_file():
            print(f"❌ 轨迹回放文件不存在: {tp}")
            return 1
        traj_cfg["playback_path"] = str(tp.resolve())
        print(f"🎞️  MuJoCo 轨迹回放: {traj_cfg['playback_path']}")
    config["mujoco_trajectory"] = traj_cfg
    return None


def _apply_mujoco_gui_from_args(
    config: dict,
    mujoco_gui: bool,
    *,
    sph_gui: bool = False,
    shutdown_on_close: Optional[bool] = None,
) -> None:
    mg = config.setdefault("mujoco_gui", {})
    mg["enabled"] = bool(mujoco_gui)
    if not mujoco_gui:
        return
    if shutdown_on_close is not None:
        mg["shutdown_on_close"] = shutdown_on_close
    elif sph_gui and "shutdown_on_close" not in mg:
        mg["shutdown_on_close"] = False
    print("🖥️  MuJoCo 被动查看器已启用（mujoco.viewer.launch_passive）")
    if mg.get("shutdown_on_close") is False:
        print("   ↳ 关 MuJoCo 窗口不会结束仿真（双界面长跑）；Ctrl+C 或 --mujoco-shutdown-on-close 可改行为")


def _apply_orcasph_gui_from_args(config: dict, gui: bool) -> None:
    if "orcasph" not in config or not config["orcasph"].get("enabled", False):
        return
    if "args" not in config["orcasph"]:
        config["orcasph"]["args"] = []
    config["orcasph"]["args"] = [arg for arg in config["orcasph"]["args"] if arg != "--gui"]
    if gui:
        config["orcasph"]["args"].append("--gui")
        print("🎨 OrcaSPH GUI 已启用")

def _apply_performance_stats_from_args(config: dict, args: argparse.Namespace) -> None:
    if "orcasph" not in config or not config["orcasph"].get("enabled", False):
        return
    if "args" not in config["orcasph"]:
        config["orcasph"]["args"] = []
    # 清理现有的性能统计参数
    config["orcasph"]["args"] = [arg for arg in config["orcasph"]["args"] if not arg.startswith("--performance-stats")]
    if args.enable_performance_stats:
        config["orcasph"]["args"].append("--performance-stats")
        print("📊 性能统计功能已启用")
        if args.performance_stats_plot:
            config["orcasph"]["args"].append("--performance-stats-plot")
            print("📈 性能统计图表实时显示已启用")


def _print_sph_run_sanity_check(
    orcagym_tmp_dir: Path, session_timestamp: str, max_steps: int
) -> None:
    """
    根据当次会话日志做粗粒度「SPH 是否参与计算」检查（非物理精度证明）。
    """
    if max_steps <= 0:
        return
    run_log = orcagym_tmp_dir / f"run_fluid_sim_{session_timestamp}.log"
    olink = orcagym_tmp_dir / f"orcalink_{session_timestamp}.log"
    osph = orcagym_tmp_dir / f"orcasph_{session_timestamp}.log"
    print("\n" + "=" * 60)
    print("SPH 链路粗检（基于日志关键词）")
    print("=" * 60)
    if not run_log.is_file():
        print(f"⚠️  未找到: {run_log}")
        return
    text = run_log.read_text(encoding="utf-8", errors="replace")
    if "SPH 集成已禁用" in text or "Session ready timeout" in text:
        print("❌ MuJoCo↔OrcaLink 未在时限内凑齐双客户端，或 SPH 集成被关闭。请用无 --gui 或提高 ready_timeout。")
    elif "Session is ready" in text or "sph_wrapper.connect() returned: True" in text:
        print("✅ OrcaLink 会话已就绪，Python 侧 bridge 已连接。")
    else:
        print("⚠️  未在 run_fluid_sim 日志中明确看到 Session ready / connect True，请人工打开日志核对。")
    bad = ("NaN", "diverg", "Error: [OrcaLinkBridge]", "FATAL")
    for p in orcagym_tmp_dir.glob(f"orcasph_{session_timestamp}.log*"):
        if not p.is_file():
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        hits = [b for b in bad if b in raw]
        if hits:
            print(f"⚠️  {p.name} 含可疑关键词: {', '.join(hits)}（需结合全文判断是否影响步进）")
        else:
            print(f"✅ {p.name} 未命中常见错误关键词 {bad[:3]}…")
        break
    else:
        print(f"⚠️  未找到 orcasph_{session_timestamp}.log")
    if olink.is_file():
        ot = olink.read_text(encoding="utf-8", errors="replace")
        if "SPlisHSPlasH" in ot or "splishsplas" in ot.lower():
            print("✅ orcalink 日志中出现 SPH 侧客户端名（双端之一已注册）。")
        elif "mujoco_client" in ot and "2 clients" not in ot and "Session" in ot:
            print("ℹ️  请搜 orcalink 日志是否仅 mujoco_client（若如此为双端未齐）。")
    print("=" * 60 + "\n")


def _apply_build_mode_from_args(config: dict, args: argparse.Namespace) -> None:
    if args.build_mode is not None:
        config["build_mode"] = args.build_mode
        if args.build_mode == "release":
            print("ℹ️  build_mode=release：force_position 调试采集已关闭")


def _apply_manual_mode_from_args(config: dict, args: argparse.Namespace) -> None:
    if not args.manual_mode:
        return
    print("=" * 60)
    print("⚙️  手动模式已启用")
    print("=" * 60)
    print("请确保已手动启动以下服务：")
    print(f"  1. OrcaLink: orcalink --port {config['orcalink']['port']}")
    gui_flag = "--gui" if args.gui else ""
    print(f"  2. OrcaSPH: orcasph --scene <scene.json> {gui_flag}")
    print("=" * 60)
    config["orcalink"]["auto_start"] = False
    config["orcasph"]["auto_start"] = False


def main() -> int:
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    orcagym_tmp_dir = Path.home() / ".orcagym" / "tmp"
    orcagym_tmp_dir.mkdir(parents=True, exist_ok=True)
    log_file = orcagym_tmp_dir / f"run_fluid_sim_{session_timestamp}.log"

    log_handle = open(log_file, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeOutput(sys.stdout, log_handle)
    sys.stderr = TeeOutput(sys.stderr, log_handle)

    try:
        print(f"📝 日志文件: {log_file}")
        print("=" * 60)

        parser = _build_arg_parser()
        args = parser.parse_args()

        err = _validate_early_args(args)
        if err is not None:
            return err

        cpu_affinity = _resolve_cpu_affinity(args.use_all_cpu)

        config_path = Path(__file__).parent / args.config
        if not config_path.exists():
            print(f"❌ 错误: 配置文件不存在: {config_path}")
            return 1

        config = load_config(str(config_path))
        script_dir = Path(__file__).parent

        _apply_build_mode_from_args(config, args)
        _apply_particle_render_run_config(args, script_dir, session_timestamp, config)
        err = _apply_mujoco_trajectory_config(args, script_dir, session_timestamp, config)
        if err is not None:
            return err
        _apply_orcasph_gui_from_args(config, args.gui)
        _apply_mujoco_gui_from_args(
            config,
            args.mujoco_gui,
            sph_gui=bool(args.gui),
            shutdown_on_close=True if args.mujoco_shutdown_on_close else None,
        )
        _apply_performance_stats_from_args(config, args)
        _apply_manual_mode_from_args(config, args)

        try:
            # 启动性能统计图表显示
            if args.enable_performance_stats and args.performance_stats_plot:
                # 启动独立的性能统计图表查看器
                # 注意：OrcaSPH 的日志文件是 orcasph_{session_timestamp}.log，不是 run_fluid_sim_{session_timestamp}.log
                orcasph_log_file = orcagym_tmp_dir / f"orcasph_{session_timestamp}.log"
                print(f"📈 启动实时性能统计图表，监控日志文件: {orcasph_log_file}")
                # 构建性能统计查看器的路径
                # 使用绝对路径确保正确性
                project_root = Path(__file__).parent.parent.parent.resolve()
                stats_viewer_path = project_root / "envs" / "fluid_stats" / "performance_stats_viewer.py"
                
                if stats_viewer_path.exists():
                    # 启动独立的进程运行性能统计查看器
                    python_exe = sys.executable
                    subprocess.Popen([
                        python_exe,
                        str(stats_viewer_path),
                        str(orcasph_log_file)
                    ])
                else:
                    print(f"⚠️  性能统计查看器脚本不存在: {stats_viewer_path}")
            
            if args.mode == "playback":
                # 与 run_simulation._preflight_session 中耦合路径一致：先按配置设 Python 日志
                setup_python_logging(config)
                run_particle_playback_from_config(config)
            else:
                run_simulation_with_config(
                    config,
                    session_timestamp=session_timestamp,
                    cpu_affinity=cpu_affinity,
                    max_steps=max(0, int(args.max_steps or 0)),
                    bench_output_path=args.bench,
                )
                _print_sph_run_sanity_check(
                    orcagym_tmp_dir, session_timestamp, max(0, int(args.max_steps or 0))
                )
        except KeyboardInterrupt:
            print("\n✅ 仿真已停止")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            traceback.print_exc()
            return 1

        return 0
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    sys.exit(main())
