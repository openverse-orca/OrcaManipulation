"""SONIC decoder runtime for the OrcaGym push_cart demo.

The 64 motion tokens produced by the WB-WAM bridge are decoded at 50 Hz into
29 joint position targets. The constants mirror the SONIC IsaacLab deployment
used by the HumanoidArena evaluation path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SONIC_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

SONIC_DEFAULT_POS = np.array(
    [
        -0.312, -0.312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.669, 0.669, 0.2, 0.2, -0.363, -0.363, 0.2,
        -0.2, 0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)

G1_ACTION_SCALE = np.array(
    [
        0.3506614566, 0.3506614566, 0.5475464463, 0.3506614566, 0.3506614566,
        0.4385773242, 0.5475464463, 0.5475464463, 0.4385773242, 0.3506614566,
        0.3506614566, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242,
        0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242,
        0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242,
        0.0745008737, 0.0745008737, 0.0745008737, 0.0745008737,
    ],
    dtype=np.float32,
)

SONIC_PD_KP = np.array(
    [
        99.02755, 99.02755, 40.168964, 99.02755, 99.02755, 28.445627,
        40.168964, 40.168964, 28.445627, 99.02755, 99.02755, 14.222814,
        14.222814, 48.445627, 48.445627, 34.222814, 34.222814, 28.445627,
        28.445627, 24.222814, 24.222814, 19.222814, 19.222814, 14.222814,
        14.222814, 16.77876, 16.77876, 16.77876, 16.77876,
    ],
    dtype=np.float32,
)

SONIC_PD_KD = np.array(
    [
        3.1559467, 3.1559467, 1.2792531, 3.1559467, 3.1559467, 0.90733415,
        1.2792531, 1.2792531, 0.90733415, 3.1559467, 3.1559467, 0.45366707,
        0.45366707, 0.90733415, 0.90733415, 0.45366707, 0.45366707, 0.90733415,
        0.90733415, 0.45366707, 0.45366707, 0.45366707, 0.45366707, 0.45366707,
        0.45366707, 0.53407073, 0.53407073, 0.53407073, 0.53407073,
    ],
    dtype=np.float32,
)


def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm([w, x, y, z])
    if norm <= 1e-9:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def gravity_in_base_frame(quat_wxyz: np.ndarray) -> np.ndarray:
    rot = quat_wxyz_to_rotmat(quat_wxyz)
    return (rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)


class SonicDecoder:
    """Stateful 994D -> 29D SONIC decoder."""

    def __init__(self, model_path: str | Path, *, frame_history: int = 10) -> None:
        import onnxruntime as ort

        self.model_path = str(model_path)
        self.frame_history = int(frame_history)
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        expected = tuple(self.session.get_inputs()[0].shape)
        if expected != (1, 994):
            raise ValueError(f"SONIC decoder expects [1,994], got {expected}")
        self.reset()

    def reset(self) -> None:
        self.ang_vel_hist = np.zeros((self.frame_history, 3), dtype=np.float32)
        self.qpos_hist = np.zeros((self.frame_history, 29), dtype=np.float32)
        self.qvel_hist = np.zeros((self.frame_history, 29), dtype=np.float32)
        self.last_action_hist = np.zeros((self.frame_history, 29), dtype=np.float32)
        self.grav_hist = np.zeros((self.frame_history, 3), dtype=np.float32)
        self.last_action = np.zeros(29, dtype=np.float32)
        self._history_primed = False

    def step(
        self,
        tokens: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        base_quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens = np.asarray(tokens, dtype=np.float32).reshape(64)
        joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(29)
        joint_vel = np.asarray(joint_vel, dtype=np.float32).reshape(29)
        ang_vel = np.asarray(base_ang_vel, dtype=np.float32).reshape(3)
        gravity = gravity_in_base_frame(base_quat_wxyz)

        if not self._history_primed:
            # Zero gravity/joint history is out of distribution and makes the
            # first decoded target jump. Bootstrap from the measured state.
            self.ang_vel_hist[:] = ang_vel
            self.qpos_hist[:] = joint_pos - SONIC_DEFAULT_POS
            self.qvel_hist[:] = joint_vel
            self.last_action_hist[:] = self.last_action
            self.grav_hist[:] = gravity
            self._history_primed = True

        self.ang_vel_hist = np.roll(self.ang_vel_hist, -1, axis=0)
        self.ang_vel_hist[-1] = ang_vel
        self.qpos_hist = np.roll(self.qpos_hist, -1, axis=0)
        self.qpos_hist[-1] = joint_pos - SONIC_DEFAULT_POS
        self.qvel_hist = np.roll(self.qvel_hist, -1, axis=0)
        self.qvel_hist[-1] = joint_vel
        self.last_action_hist = np.roll(self.last_action_hist, -1, axis=0)
        self.last_action_hist[-1] = self.last_action
        self.grav_hist = np.roll(self.grav_hist, -1, axis=0)
        self.grav_hist[-1] = gravity

        obs = np.concatenate(
            [
                tokens,
                self.ang_vel_hist.reshape(-1),
                self.qpos_hist.reshape(-1),
                self.qvel_hist.reshape(-1),
                self.last_action_hist.reshape(-1),
                self.grav_hist.reshape(-1),
            ]
        ).reshape(1, -1).astype(np.float32)
        raw = np.asarray(self.session.run(None, {self.input_name: obs})[0], dtype=np.float32).reshape(29)
        self.last_action = raw.copy()
        target = raw * G1_ACTION_SCALE + SONIC_DEFAULT_POS
        return target.astype(np.float32), raw
