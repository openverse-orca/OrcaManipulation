"""纯 numpy 轨迹单测（无需 gym）。"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from kettle_trajectory_math import (
    DEFAULT_PHASE1_SEC,
    DEFAULT_PHASE2_SEC,
    sample_pose_vel,
    smoothstep,
    wxyz_to_scipy,
)


def test_smoothstep_endpoints():
    assert abs(smoothstep(0.0)) < 1e-12
    assert abs(smoothstep(1.0) - 1.0) < 1e-12


def test_phase1_lift_endpoint():
    p0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    qpos, _ = sample_pose_vel(
        DEFAULT_PHASE1_SEC,
        p0,
        q0,
        lift_m=0.3,
        rotate_deg=90.0,
        phase1_sec=DEFAULT_PHASE1_SEC,
        phase2_sec=DEFAULT_PHASE2_SEC,
    )
    assert abs(qpos[2] - 0.3) < 1e-5


def test_phase2_rotation_magnitude():
    p0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    from kettle_trajectory_math import DEFAULT_ROTATE_DEG, DEFAULT_LOCAL_AXIS

    qpos_end, _ = sample_pose_vel(
        DEFAULT_PHASE1_SEC + DEFAULT_PHASE2_SEC,
        p0,
        q0,
        lift_m=0.3,
        rotate_deg=DEFAULT_ROTATE_DEG,
        phase1_sec=DEFAULT_PHASE1_SEC,
        phase2_sec=DEFAULT_PHASE2_SEC,
        local_axis=DEFAULT_LOCAL_AXIS,
    )
    r0 = R.from_quat(wxyz_to_scipy(q0))
    r1 = R.from_quat([qpos_end[4], qpos_end[5], qpos_end[6], qpos_end[3]])
    angle = np.linalg.norm((r0.inv() * r1).as_rotvec())
    assert abs(np.rad2deg(angle) - abs(DEFAULT_ROTATE_DEG)) < 0.1


def test_joint_blob_scoring():
    import sys
    from pathlib import Path

    src_root = Path(__file__).resolve().parents[3]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from examples.dataCollection.utils.kettle_trajectory_driver import (  # noqa: E402
        classify_prop_joint_blob,
        score_joint_blob,
    )

    jug = "[2559139150351]_joint_abc [2559139150351]_bodyjoint waterjug_02_b.obj"
    cup = "[2563434117647]_joint_abc [2563434117647]_bodyjoint cup_01.obj"
    unknown = "[999]_joint_abc [999]_bodyjoint"

    assert classify_prop_joint_blob(jug) == "waterjug"
    assert classify_prop_joint_blob(cup) == "cup"
    assert classify_prop_joint_blob(unknown) is None
    assert score_joint_blob(jug) == 100
    assert score_joint_blob(cup) == -1
    assert score_joint_blob(unknown) == -1


def test_mass_heuristic_picks_heavier():
    import sys
    from pathlib import Path

    src_root = Path(__file__).resolve().parents[3]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from examples.dataCollection.utils.kettle_trajectory_driver import classify_prop_joint_blob  # noqa: E402

    jug_blob = "[1]_joint_x [1]_bodyjoint waterjug_02_b.obj"
    cup_blob = "[2]_joint_y [2]_bodyjoint cup_01.obj"
    assert classify_prop_joint_blob(jug_blob) == "waterjug"
    assert classify_prop_joint_blob(cup_blob) == "cup"


if __name__ == "__main__":
    test_smoothstep_endpoints()
    test_phase1_lift_endpoint()
    test_phase2_rotation_magnitude()
    test_joint_blob_scoring()
    test_mass_heuristic_picks_heavier()
    print("ok")
