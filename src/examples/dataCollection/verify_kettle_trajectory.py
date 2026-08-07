#!/usr/bin/env python3
"""
校核水壶预定轨迹：离线数学采样 + 可选连接 OrcaStudio 场景实测 joint qpos。

模仿 SPH_bug/verify_sph_phys_time.py：失败时非零退出，供 one_click / CI 使用。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# 允许从仓库 src 或本目录上级运行
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _SCRIPT_DIR.parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from examples.dataCollection.utils.kettle_trajectory_math import (  # noqa: E402
    DEFAULT_LOCAL_AXIS,
    DEFAULT_PHASE1_SEC,
    DEFAULT_PHASE2_SEC,
    DEFAULT_ROTATE_DEG,
    format_speed_report_text,
    sample_pose_vel,
    trajectory_duration,
    trajectory_speed_report,
    wxyz_to_scipy,
)

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"


@dataclass
class TrajectoryParams:
    lift_m: float = 0.3
    rotate_deg: float = DEFAULT_ROTATE_DEG
    phase1_sec: float = DEFAULT_PHASE1_SEC
    phase2_sec: float = DEFAULT_PHASE2_SEC
    hold_sec: float = 1.0
    local_axis: str = DEFAULT_LOCAL_AXIS
    pos_tol_m: float = 2e-3
    angle_tol_deg: float = 1.0
    joint_tol_m: float = 5e-3
    angle_tol_joint_deg: float = 2.0


def _rotation_angle_deg(q0_wxyz: np.ndarray, q1_wxyz: np.ndarray) -> float:
    r0 = R.from_quat(wxyz_to_scipy(q0_wxyz))
    r1 = R.from_quat(wxyz_to_scipy(q1_wxyz))
    return float(np.rad2deg(np.linalg.norm((r0.inv() * r1).as_rotvec())))


def _check_samples(
    params: TrajectoryParams,
    *,
    samples: list[tuple[float, np.ndarray, np.ndarray]],
    label: str,
) -> list[str]:
    """samples: (t, expected_qpos7, actual_qpos7)"""
    errors: list[str] = []
    p0 = samples[0][1][:3]
    q0 = samples[0][1][3:7]
    t1, t2 = params.phase1_sec, params.phase2_sec

    for t, exp, act in samples:
        exp = np.asarray(exp, dtype=np.float64).ravel()
        act = np.asarray(act, dtype=np.float64).ravel()
        pos_err = float(np.linalg.norm(exp[:3] - act[:3]))
        ang_err = _rotation_angle_deg(exp[3:7], act[3:7])
        if pos_err > params.joint_tol_m or ang_err > params.angle_tol_joint_deg:
            errors.append(
                f"{label} t={t:.3f}s: |Δp|={pos_err*1e3:.2f}mm "
                f"Δθ={ang_err:.2f}° (tol {params.joint_tol_m*1e3:.1f}mm, {params.angle_tol_joint_deg:.1f}°)"
            )

    def _at(t_query: float) -> tuple[np.ndarray, np.ndarray]:
        best = min(samples, key=lambda s: abs(s[0] - t_query))
        return best[1], best[2]

    exp_end1, act_end1 = _at(t1)
    lift_err = abs(float(act_end1[2] - p0[2]) - params.lift_m)
    if lift_err > params.pos_tol_m:
        errors.append(
            f"{label} 段 1 终点 t≈{t1}s: 实际抬升 {act_end1[2]-p0[2]:.4f} m "
            f"期望 {params.lift_m:.4f} m (|Δ|={lift_err*1e3:.2f} mm)"
        )

    exp_end2, act_end2 = _at(t1 + t2)
    rot_err = _rotation_angle_deg(q0, act_end2[3:7])
    if abs(rot_err - abs(params.rotate_deg)) > params.angle_tol_deg:
        errors.append(
            f"{label} 段 2 终点 t≈{t1+t2}s: 累计转角 {rot_err:.2f}° "
            f"期望 {params.rotate_deg:.1f}° (|幅度| {abs(params.rotate_deg):.1f}°, tol {params.angle_tol_deg:.1f}°)"
        )

    return errors


def verify_offline(params: TrajectoryParams, *, n_samples: int = 200) -> list[str]:
    p0 = np.zeros(3, dtype=np.float64)
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    t_end = trajectory_duration(params.phase1_sec, params.phase2_sec, params.hold_sec)
    times = np.linspace(0.0, t_end, n_samples)
    samples: list[tuple[float, np.ndarray, np.ndarray]] = []
    for t in times:
        exp, _ = sample_pose_vel(
            float(t),
            p0,
            q0,
            lift_m=params.lift_m,
            rotate_deg=params.rotate_deg,
            phase1_sec=params.phase1_sec,
            phase2_sec=params.phase2_sec,
            local_axis=params.local_axis,
        )
        samples.append((float(t), exp.copy(), exp.copy()))
    return _check_samples(params, samples=samples, label="offline")


def _make_env(agent_name: str, level: str, kettle_joint_hint: str | None):
    from examples.dataCollection.utils.kettle_trajectory_driver import KettleTrajectoryDriver

    from dataCollectionManager.data_collection_manager import DataCollectionManager
    from scene.scene_config_util import load_scene_config
    from scene.scene_manager import SceneManager

    base_dir = str(_SCRIPT_DIR)
    orcagym_addr = "localhost:50051"
    default_joint_values: dict[str, float] = {}

    if agent_name == "tiangong2":
        from conf import tiangong2_conf as agent_conf
        from dataStorage.tiangong_data_storage import Tiangong2DataStorage

        obs_cb = Tiangong2DataStorage(
            dataset_path=os.path.join(base_dir, "dataset", agent_name, level),
            hdf5_path="record/proprio_stats.hdf5",
        ).obs_callback
    elif agent_name == "openloong":
        from conf import openloong_conf as agent_conf
        from dataStorage.openloong_data_storage import OpenLoongDataStorage

        obs_cb = OpenLoongDataStorage(
            dataset_path=os.path.join(base_dir, "dataset", agent_name, level),
            hdf5_path="record/proprio_stats.hdf5",
        ).obs_callback
    else:
        raise ValueError(agent_name)

    for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value

    config = load_scene_config(base_dir, None)
    scene_manager = SceneManager(orcagym_addr, config=config)
    last_err: Exception | None = None
    env = None
    for attempt in range(1, 6):
        try:
            manager = DataCollectionManager(
                agent_name=agent_name,
                env_name="DataCollection",
                entry_point=ENTRY_POINT,
                default_joint_values=default_joint_values,
                obs_callback=obs_cb,
                scene_manager=scene_manager,
                frame_skip=5,
                time_step=0.005,
            )
            env = manager.env
            env.reset()
            last_err = None
            break
        except Exception as e:
            last_err = e
            if attempt < 5 and "MuJoCo has not been initialized" in str(e):
                import time

                time.sleep(3.0)
                continue
            raise
    if env is None and last_err is not None:
        raise last_err
    joint = KettleTrajectoryDriver.wait_resolve_joint_name(
        env,
        kettle_joint_hint,
        timeout_sec=90.0,
        scene_manager=scene_manager,
    )
    if not joint:
        jd = env.model.get_joint_dict() or {}
        sample = sorted(jd.keys())[:30]
        raise RuntimeError(
            "场景中未解析到水壶 joint；请先 --discover-names 或传 --kettle-joint。"
            f" 当前 joint 样例({len(jd)} total): {sample}"
        )
    return env, joint, manager


def verify_scene(
    params: TrajectoryParams,
    *,
    agent_name: str,
    level: str,
    kettle_joint: str | None,
    rundir: Path | None,
    visual: bool = False,
    frame_skip: int = 5,
    time_step: float = 0.005,
    hold_after_sec: float = 3.0,
) -> list[str]:
    from examples.dataCollection.utils.kettle_trajectory_driver import KettleTrajectoryDriver

    env, joint_name, manager = _make_env(agent_name, level, kettle_joint)
    kettle = KettleTrajectoryDriver(
        joint_name,
        lift_m=params.lift_m,
        rotate_deg=params.rotate_deg,
        phase1_sec=params.phase1_sec,
        phase2_sec=params.phase2_sec,
        hold_sec=params.hold_sec,
        local_axis=params.local_axis,
    )
    kettle.reset(env)
    p0 = kettle._p0
    q0 = kettle._q0_wxyz
    assert p0 is not None and q0 is not None

    if agent_name == "tiangong2":
        import conf.tiangong2_conf as agent_conf
    else:
        import conf.openloong_conf as agent_conf

    from examples.dataCollection.utils.kettle_scene_helpers import (
        clear_studio_ctrl_overrides,
        disable_robot_actuators,
        freeze_robot_pose,
        sync_env_to_studio,
    )

    disable_robot_actuators(env, agent_conf)
    freeze_robot_pose(env, agent_conf)
    kettle.reset(env)
    from examples.dataCollection.utils.kettle_trajectory_driver import (
        _body_name_for_joint,
        _mesh_hints_for_joint,
        classify_prop_joint_blob,
    )

    jd = env.model.get_joint_dict() or {}
    jinfo = jd.get(joint_name, {})
    body_name = _body_name_for_joint(env, jinfo)
    mesh = _mesh_hints_for_joint(env, joint_name, jinfo)
    kind = classify_prop_joint_blob(f"{joint_name} {body_name} {mesh}") or "unknown"
    print(
        f"[verify_kettle_trajectory] 水壶 joint={joint_name} kind={kind} body={body_name} mesh={mesh[:100]}"
    )
    p0 = kettle._p0
    q0 = kettle._q0_wxyz
    assert p0 is not None and q0 is not None
    t0 = float(env.data.time)
    t_end = t0 + trajectory_duration(params.phase1_sec, params.phase2_sec, params.hold_sec) + 0.05
    ctrl = np.zeros(env.nu, dtype=np.float32)

    rows: list[dict[str, float]] = []
    samples: list[tuple[float, np.ndarray, np.ndarray]] = []
    real_time_step = frame_skip * time_step

    if visual:
        print(
            f"[verify_kettle_trajectory] 可视化模式：realtime={real_time_step:.3f}s/step，"
            "机械手已冻结，请在 OrcaStudio 视口观看水壶运动"
        )

    while float(env.data.time) < t_end:
        loop_start = time.time()
        clear_studio_ctrl_overrides(env)
        env.step(ctrl)
        freeze_robot_pose(env, agent_conf)
        sim_t = float(env.data.time)
        t = sim_t - t0
        kettle.apply(env)
        exp, _ = sample_pose_vel(
            t,
            p0,
            q0,
            lift_m=params.lift_m,
            rotate_deg=params.rotate_deg,
            phase1_sec=params.phase1_sec,
            phase2_sec=params.phase2_sec,
            local_axis=params.local_axis,
        )
        act = np.asarray(env.query_joint_qpos([joint_name])[joint_name], dtype=np.float64).ravel()
        samples.append((t, exp.copy(), act.copy()))
        rows.append(
            {
                "sim_time": sim_t,
                "t_traj": t,
                "exp_z": float(exp[2]),
                "act_z": float(act[2]),
                "exp_rot_deg": _rotation_angle_deg(q0, exp[3:7]),
                "act_rot_deg": _rotation_angle_deg(q0, act[3:7]),
                "pos_err_mm": float(np.linalg.norm(exp[:3] - act[:3]) * 1e3),
                "rot_err_deg": _rotation_angle_deg(exp[3:7], act[3:7]),
            }
        )
        if visual:
            sync_env_to_studio(env)
            elapsed = time.time() - loop_start
            if elapsed < real_time_step:
                time.sleep(real_time_step - elapsed)

    if visual and hold_after_sec > 0:
        hold_until = time.time() + hold_after_sec
        print(f"[verify_kettle_trajectory] 轨迹结束，保持终态 {hold_after_sec:.1f}s …")
        while time.time() < hold_until:
            kettle.apply(env)
            sync_env_to_studio(env)
            time.sleep(real_time_step)

    if rundir is not None:
        rundir.mkdir(parents=True, exist_ok=True)
        csv_path = rundir / "kettle_trajectory_verify.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        (rundir / "session_meta.txt").write_text(
            f"level={level}\nagent={agent_name}\njoint={joint_name}\n",
            encoding="utf-8",
        )

    return _check_samples(params, samples=samples, label="scene")


def main() -> int:
    ap = argparse.ArgumentParser(description="校核水壶抬升+旋转轨迹")
    ap.add_argument("--report-speeds", action="store_true", help="仅打印速度后退出 0")
    ap.add_argument("--offline-only", action="store_true", help="仅离线数学校核")
    ap.add_argument("--scene", action="store_true", help="连接 gRPC 场景实测（需 Studio Play）")
    ap.add_argument(
        "--visual",
        action="store_true",
        help="场景模式下实时 render + sleep，在 OrcaStudio 视口可见水壶运动（隐含 --scene）",
    )
    ap.add_argument(
        "--skip-offline",
        action="store_true",
        help="跳过离线数学校核（演示模式更快启动）",
    )
    ap.add_argument(
        "--hold-after-sec",
        type=float,
        default=3.0,
        help="--visual 时轨迹结束后额外保持终态秒数（默认 3）",
    )
    ap.add_argument("--level", default="FluidTest_Hotel_Bar_robot")
    ap.add_argument("--agent_name", default="tiangong2", choices=["openloong", "tiangong2"])
    ap.add_argument(
        "--kettle-joint",
        default="waterjug",
        help="水壶 joint 提示（默认 waterjug，按 mesh 匹配 WaterJug_02_fluid）",
    )
    ap.add_argument("--lift-m", type=float, default=0.3)
    ap.add_argument("--rotate-deg", type=float, default=DEFAULT_ROTATE_DEG)
    ap.add_argument("--phase1-sec", type=float, default=DEFAULT_PHASE1_SEC)
    ap.add_argument("--phase2-sec", type=float, default=DEFAULT_PHASE2_SEC)
    ap.add_argument("--hold-sec", type=float, default=1.0)
    ap.add_argument("--local-axis", default=DEFAULT_LOCAL_AXIS, choices=["x", "y", "z"])
    ap.add_argument(
        "--rundir",
        type=Path,
        default=None,
        help="场景校核 CSV 输出目录；默认 ORCA_KETTLE_VERIFY_RUNDIR 或 monitor_data/kettle_verify_*",
    )
    args = ap.parse_args()

    params = TrajectoryParams(
        lift_m=args.lift_m,
        rotate_deg=args.rotate_deg,
        phase1_sec=args.phase1_sec,
        phase2_sec=args.phase2_sec,
        hold_sec=args.hold_sec,
        local_axis=args.local_axis,
    )
    report = trajectory_speed_report(
        lift_m=params.lift_m,
        rotate_deg=params.rotate_deg,
        phase1_sec=params.phase1_sec,
        phase2_sec=params.phase2_sec,
    )
    print(format_speed_report_text(report))

    if args.report_speeds:
        return 0

    errors: list[str] = []
    if not args.skip_offline:
        errors.extend(verify_offline(params))
    scene = args.scene or args.visual
    if scene and not args.offline_only:
        rd = args.rundir
        if rd is None:
            rd_s = os.environ.get("ORCA_KETTLE_VERIFY_RUNDIR", "").strip()
            if rd_s:
                rd = Path(rd_s)
        errors.extend(
            verify_scene(
                params,
                agent_name=args.agent_name,
                level=args.level,
                kettle_joint=args.kettle_joint,
                rundir=rd,
                visual=args.visual,
                hold_after_sec=args.hold_after_sec,
            )
        )

    if errors:
        print("\n[verify_kettle_trajectory] 校核失败:", file=sys.stderr)
        for e in errors[:20]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... 另有 {len(errors)-20} 条", file=sys.stderr)
        return 1

    print("\n[verify_kettle_trajectory] 校核通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
