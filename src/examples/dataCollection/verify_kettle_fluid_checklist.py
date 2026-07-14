#!/usr/bin/env python3
"""校核水壶+SPH 演示五项 checklist（解析运行日志 + 可选 gRPC）。"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _SCRIPT_DIR.parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from examples.dataCollection.utils.kettle_trajectory_math import (  # noqa: E402
    DEFAULT_LOCAL_AXIS,
    DEFAULT_ROTATE_DEG,
)

DEFAULT_LOG = _SCRIPT_DIR / "logs" / "data_collection_fluid_auto.log"


def _read_text(paths: list[Path]) -> str:
    parts: list[str] = []
    for p in paths:
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _check_mujoco_port(port: int) -> bool:
    import socket

    try:
        s = socket.socket()
        s.settimeout(2)
        s.connect(("127.0.0.1", port))
        s.close()
        return True
    except OSError:
        return False


def _check_load_local_env(port: int) -> bool:
    try:
        import os

        proto_dir = os.path.expanduser(str(_SRC_ROOT.parent / "OrcaGym" / "orca_gym" / "protos"))
        if proto_dir not in sys.path:
            sys.path.insert(0, proto_dir)
        import grpc
        import mjc_message_pb2
        import mjc_message_pb2_grpc

        channel = grpc.insecure_channel(f"localhost:{port}")
        stub = mjc_message_pb2_grpc.GrpcServiceStub(channel)
        req = mjc_message_pb2.LoadLocalEnvRequest()
        req.req_type = mjc_message_pb2.LoadLocalEnvRequest.XML_FILE_NAME
        resp = stub.LoadLocalEnv(req, timeout=5)
        channel.close()
        return resp.status == mjc_message_pb2.LoadLocalEnvResponse.SUCCESS
    except Exception:
        return False


def verify_checklist(
    log_text: str,
    *,
    level: str,
    lift_m: float,
    rotate_deg: float,
    local_axis: str,
    fluid_hold_sec: float,
    check_grpc: bool,
    port: int,
) -> list[str]:
    errors: list[str] = []
    ok: list[str] = []

    # 1. Level + MuJoCo
    if f"[CHECKLIST-1] level={level}" in log_text or f"level={level}" in log_text:
        ok.append("1")
    elif check_grpc and _check_mujoco_port(port) and _check_load_local_env(port):
        ok.append("1")
    else:
        errors.append(f"1) FluidTest 关卡/MuJoCo 未就绪（期望 level={level}）")

    # 2. Fluid coupling
    if (
        "Starting fluid coupling (OrcaLink + OrcaSPH)" in log_text
        or "[CHECKLIST-2] fluid_coupling=ok" in log_text
        or "Fluid-MuJoCo 耦合挂载" in log_text
    ):
        ok.append("2")
    else:
        errors.append("2) 未检测到 OrcaLink + OrcaSPH 启动（fluid_sim_config.json / start_fluid_coupling）")

    # 3. Kettle trajectory params
    m3 = re.search(r"\[CHECKLIST-3\].*lift_m=([\d.]+).*rotate_deg=([-\d.]+).*local_axis=(\w+)", log_text)
    if m3:
        lm, rd, ax = float(m3.group(1)), float(m3.group(2)), m3.group(3)
        if abs(lm - lift_m) > 1e-6 or abs(rd - rotate_deg) > 1e-6 or ax != local_axis:
            errors.append(
                f"3) 轨迹参数不符: log lift={lm} rot={rd} axis={ax} "
                f"期望 lift={lift_m} rot={rotate_deg} axis={local_axis}"
            )
        elif "Kettle joint resolved" in log_text or "joint=" in log_text:
            ok.append("3")
        else:
            errors.append("3) 未解析水壶 joint")
    elif (
        f"lift_m={lift_m}" in log_text
        and f"rotate_deg={rotate_deg}" in log_text
        and f"local_axis={local_axis}" in log_text
        and "Kettle joint resolved" in log_text
    ):
        ok.append("3")
    else:
        errors.append(
            f"3) WaterJug 轨迹未确认（期望 +Z {lift_m}m, 局部{local_axis.upper()} {rotate_deg}°）"
        )

    # 4. Robot frozen
    if "[CHECKLIST-4] robot_frozen=ok" in log_text or "robot_frozen=ok" in log_text:
        ok.append("4")
    else:
        errors.append("4) 演示模式未冻结机械手（motors+positions）")

    # 5. Fluid hold
    m5 = re.search(r"\[CHECKLIST-5\].*fluid_hold_sec=([\d.]+).*total_demo_sec=([\d.]+)", log_text)
    if m5:
        fh = float(m5.group(1))
        if abs(fh - fluid_hold_sec) > 0.01:
            errors.append(f"5) fluid_hold_sec={fh} 期望 {fluid_hold_sec}")
        else:
            ok.append("5")
    elif f"fluid_hold_sec={fluid_hold_sec}" in log_text and "Demo duration" in log_text:
        ok.append("5")
    else:
        errors.append(f"5) 未配置轨迹后 SPH 续跑 {fluid_hold_sec}s")

    if "Task end" not in log_text or "Reached max_episodes=1" not in log_text:
        errors.append("演示未完整结束（缺 Task end 或 max_episodes=1）")

    if "[CHECKLIST-6] studio_sync=ok" not in log_text:
        errors.append("6) Studio 视口未启用 post_step sync（水壶在 Editor 中可能不动）")

    if "Detected" not in log_text or "FluidBlock" not in log_text:
        if "FluidBlocks from" not in log_text:
            errors.append("7) 未检测到 SPH FluidBlock（水壶内可能无水体）")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="水壶+SPH 五项 checklist 校核")
    ap.add_argument("--log", type=Path, action="append", default=None, help="日志文件（可多次）")
    ap.add_argument("--level", default="FluidTest_Hotel_Bar_robot")
    ap.add_argument("--lift-m", type=float, default=0.3)
    ap.add_argument("--rotate-deg", type=float, default=DEFAULT_ROTATE_DEG)
    ap.add_argument("--local-axis", default=DEFAULT_LOCAL_AXIS)
    ap.add_argument("--fluid-hold-sec", type=float, default=8.0)
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--skip-grpc", action="store_true")
    args = ap.parse_args()

    paths = list(args.log or [])
    if DEFAULT_LOG not in paths:
        paths.append(DEFAULT_LOG)

    log_text = _read_text(paths)
    errors = verify_checklist(
        log_text,
        level=args.level,
        lift_m=args.lift_m,
        rotate_deg=args.rotate_deg,
        local_axis=args.local_axis,
        fluid_hold_sec=args.fluid_hold_sec,
        check_grpc=not args.skip_grpc,
        port=args.port,
    )

    print("=== 水壶+SPH Checklist（5+2 扩展） ===")
    labels = [
        "1. FluidTest_Hotel_Bar_robot + MuJoCo 就绪",
        "2. OrcaLink + OrcaSPH（fluid_sim_config.json）",
        f"3. WaterJug +Z {args.lift_m}m → 局部{args.local_axis.upper()} {args.rotate_deg}°",
        "4. 冻结机械手",
        f"5. 轨迹后 SPH 续跑 {args.fluid_hold_sec}s",
        "6. Studio post_step 视口同步",
        "7. SPH FluidBlock 已生成",
    ]
    for i, label in enumerate(labels, 1):
        failed = any(e.startswith(f"{i})") for e in errors)
        print(f"  [{'✓' if not failed else '✗'}] {label}")

    demo_ok = "Task end" in log_text and "Reached max_episodes=1" in log_text
    print(f"  [{'✓' if demo_ok else '✗'}] 演示单次 episode 正常结束")
    if not demo_ok:
        errors.append("8) 演示单次 episode 未正常结束")

    if errors:
        print("\n失败项:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("\n[verify_kettle_fluid_checklist] 五项校核通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
