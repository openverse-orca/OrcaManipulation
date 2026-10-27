"""Run the G1 push_cart demo inside OrcaLab/OrcaGym.

Stage 1 (--stand-only): hold the initial pose with PD and verify the robot
stands in the OrcaLab scene.
Stage 2 (default): stream 107D state + 224x224 RGB to the WB-WAM policy
server, decode the returned 64 SONIC tokens, and drive the 29 body joints
with PD.

Each 50 Hz control tick advances MuJoCo by one ``time_step`` (1 ms), so the
simulation runs 20x slower than wall clock. The action-chunk index and the
50 Hz SONIC decoder are therefore indexed by simulation time; this gives
CPU inference enough wall-clock time between chunks. Physics is paused while
waiting for a required inference result, so the initial pose and the previous
chunk cannot drift or topple before the next action arrives. The body starts
from the training mean in ``push_cart/metadata/dataset_stats.json``; the
official latent initial-motion token remains available as an option. Hands
start from the demonstrated open pre-grasp (state q01) so the policy does not
begin from a mid-grasp mean pose; ``--init-hands mean`` restores the old
behaviour.

Hand backends are auto-detected: Wuji uses 20+20 absolute policy joints and
measured feedback, while Dex3 retains the previous 7+7 retargeting and PD
control. Handle-contact measurements are logged for grasp validation only and
never override policy actions. Use ``--hand-mode wuji`` or ``--hand-mode dex3``
to override auto-detection.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv

from bridge.sonic.action_schema import WUJI_QPOS_LOWER_LIMITS, WUJI_QPOS_UPPER_LIMITS
from bridge.sonic.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from orcalab_demo.sonic_runtime import (
    SONIC_DEFAULT_POS,
    SONIC_JOINT_ORDER,
    SONIC_PD_KD,
    SONIC_PD_KP,
    SonicDecoder,
    gravity_in_base_frame,
    quat_wxyz_to_rotmat,
)

# OrcaGym / G1 joint order used by the OrcaPlayground G1 example.
G1_JOINT_SUFFIXES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_ACTUATOR_SUFFIXES = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

G1_EFFORT_LIMITS = np.array(
    [
        88, 88, 88, 139, 50, 50,
        88, 88, 88, 139, 50, 50,
        88, 50, 50,
        25, 25, 25, 25, 25, 5, 5,
        25, 25, 25, 25, 25, 5, 5,
    ],
    dtype=np.float32,
)

SONIC_TO_G1 = np.array(
    [SONIC_JOINT_ORDER.index(name) for name in G1_JOINT_SUFFIXES], dtype=np.int64
)

DEX3_JOINT_SUFFIXES = ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
WUJI_JOINT_SUFFIXES = [f"finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)]


def retarget_wuji_to_dex3(wuji_qpos: np.ndarray, dex3_ranges: np.ndarray) -> np.ndarray:
    """Map Wuji flexion/opposition to the 7-DOF Dex3 hand, then clamp to its URDF limits."""
    qpos = np.asarray(wuji_qpos, dtype=np.float32).reshape(20)
    ranges = np.asarray(dex3_ranges, dtype=np.float32).reshape(7, 2)
    normalized = np.clip(
        (qpos - WUJI_QPOS_LOWER_LIMITS) / (WUJI_QPOS_UPPER_LIMITS - WUJI_QPOS_LOWER_LIMITS),
        0.0,
        1.0,
    )
    target = np.zeros(7, dtype=np.float32)
    target[0] = qpos[1]  # Thumb opposition; the other two thumb joints handle flexion.
    flexion_sources = {
        1: (0, 2),
        2: (3,),
        3: (4,),
        4: (6, 7),
        5: (8,),
        6: (10, 11),
    }
    for output_index, source_indices in flexion_sources.items():
        closure = float(np.mean(normalized[list(source_indices)]))
        limit_index = int(np.argmax(np.abs(ranges[output_index])))
        target[output_index] = closure * ranges[output_index, limit_index]
    return np.clip(target, ranges[:, 0], ranges[:, 1]).astype(np.float32)


class G1PushCartEnv(OrcaGymEulerEnv):
    def __init__(self, frame_skip: int, orcagym_addr: str, time_step: float, **kwargs: Any) -> None:
        self._names_ready = False
        self._initial_base_qpos: np.ndarray | None = None
        self.init_pose_sonic = SONIC_DEFAULT_POS.copy()
        self.init_wuji_hand_state: np.ndarray | None = None
        self.hand_mode_override = "auto"
        self.cart_damping: dict[int, float] = {}
        self._time_step = float(time_step)
        self.start_offset_x = 0.0
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=[],
            time_step=time_step,
            **kwargs,
        )

    def _resolve_names(self) -> None:
        joint_meta = self.model.get_joint_dict()
        joint_keys = list(joint_meta.keys())
        joints = set(joint_keys)
        candidates: list[str] = []
        for key in joint_keys:
            suffix = "left_hip_pitch_joint"
            if key == suffix:
                prefix = ""
            elif key.endswith("_" + suffix):
                prefix = key[: -len("_" + suffix)]
            else:
                continue
            expected = [f"{prefix}_{s}" if prefix else s for s in G1_JOINT_SUFFIXES]
            if all(name in joints for name in expected):
                candidates.append(prefix)
        if not candidates:
            raise RuntimeError(
                "No complete G1 (29 body joints) found in the OrcaLab scene. "
                "Add a G1 actor in OrcaLab, save/publish the layout, and retry."
            )
        self.robot_prefix = candidates[0]
        self.joint_suffixes = list(G1_JOINT_SUFFIXES)
        self.joint_names = [
            f"{self.robot_prefix}_{s}" if self.robot_prefix else s for s in self.joint_suffixes
        ]
        self.base_joint = (
            f"{self.robot_prefix}_floating_base_joint"
            if self.robot_prefix
            else "floating_base_joint"
        )
        if self.base_joint not in joints:
            free_joints = [
                name
                for name, meta in self.model.get_joint_dict().items()
                if int(meta.get("Type", -1)) == 0
            ]
            named = [
                name
                for name in free_joints
                if "pelvis" in name.lower() or "floating" in name.lower()
            ]
            if named:
                self.base_joint = named[0]
            elif free_joints:
                self.base_joint = free_joints[0]
            else:
                raise RuntimeError("No free base joint found for the G1")
        actuator_meta = self.model.get_actuator_dict()
        actuator_keys = list(actuator_meta.keys())
        self.actuator_names = []
        for joint_name in self.joint_names:
            exact = [name for name in actuator_keys if name == joint_name]
            prefixed = [name for name in actuator_keys if name.startswith(joint_name + "_")]
            by_joint = [
                name
                for name, meta in actuator_meta.items()
                if str(meta.get("JointName", "")) == joint_name
            ]
            matches = exact or prefixed or by_joint
            if not matches:
                raise RuntimeError(f"No actuator found for G1 joint {joint_name!r}")
            self.actuator_names.append(matches[0])
        self.actuator_ids = np.array(
            [self.model.actuator_name2id(name) for name in self.actuator_names], dtype=np.int64
        )
        def with_prefix(name: str) -> str:
            return f"{self.robot_prefix}_{name}" if self.robot_prefix else name

        wuji_names = [
            with_prefix(f"{side}_{suffix}")
            for side in ("left", "right")
            for suffix in WUJI_JOINT_SUFFIXES
        ]
        dex3_names = [
            with_prefix(f"{side}_hand_{suffix}_joint")
            for side in ("left", "right")
            for suffix in DEX3_JOINT_SUFFIXES
        ]
        has_wuji = all(name in joints for name in wuji_names)
        has_dex3 = all(name in joints for name in dex3_names)
        self.hand_mode = self.hand_mode_override
        if self.hand_mode == "auto":
            self.hand_mode = "wuji" if has_wuji else "dex3" if has_dex3 else "auto"
        if self.hand_mode == "wuji" and not has_wuji:
            raise RuntimeError("Wuji hand mode requested, but the 40 Wuji joints are missing")
        if self.hand_mode == "dex3" and not has_dex3:
            raise RuntimeError("Dex3 hand mode requested, but the 14 Dex3 joints are missing")
        if self.hand_mode not in {"wuji", "dex3"}:
            raise RuntimeError("No supported Wuji or Dex3 hands found in the G1 model")
        self.hand_joint_names = wuji_names if self.hand_mode == "wuji" else dex3_names
        hand_actuator_names = []
        for joint_name in self.hand_joint_names:
            matches = [name for name, meta in actuator_meta.items() if str(meta.get("JointName", "")) == joint_name]
            if not matches:
                raise RuntimeError(f"No actuator found for {self.hand_mode} hand joint {joint_name!r}")
            hand_actuator_names.append(matches[0])
        self.hand_actuator_ids = np.array(
            [self.model.actuator_name2id(name) for name in hand_actuator_names], dtype=np.int64
        )
        self.hand_dof_addresses = np.array(
            [self.jnt_dofadr(name) for name in self.hand_joint_names], dtype=np.int64
        )
        self.hand_damping = 0.0
        self.hand_ranges = np.asarray(
            [joint_meta[name]["Range"] for name in self.hand_joint_names],
            dtype=np.float32,
        )
        self.hand_torque_limits = np.asarray(
            [np.max(np.abs(actuator_meta[name]["CtrlRange"])) for name in hand_actuator_names],
            dtype=np.float32,
        )
        handle_candidates = [
            (name, meta)
            for name, meta in self.model.get_geom_dict().items()
            if str(meta["BodyName"]).startswith("[") and int(meta["Type"]) == 3
        ]
        self.cart_handle_geom = None
        if handle_candidates:
            self.cart_handle_geom = max(
                handle_candidates,
                key=lambda item: (float(item[1]["Pos"][2]), float(item[1]["Size"][1])),
            )[0]
        geom_meta = self.model.get_geom_dict()
        link2_geoms = [
            name
            for name, meta in geom_meta.items()
            if str(meta.get("BodyName", "")).startswith(self.robot_prefix)
            and "finger" in str(meta["BodyName"])
            and str(meta["BodyName"]).endswith("_link2")
        ]
        mask1_geoms = [
            name
            for name in link2_geoms
            if int(geom_meta[name]["Contype"]) & 1 or int(geom_meta[name]["Conaffinity"]) & 1
        ]
        self._names_ready = True
        print(
            f"[env] bound robot prefix={self.robot_prefix!r} "
            f"nu={self.model.nu} nq={self.model.nq} nv={self.model.nv} "
            f"hand_mode={self.hand_mode} hand_actuators={len(self.hand_actuator_ids)} "
            f"cart_handle={self.cart_handle_geom!r} "
            f"finger_link2_mask1={len(mask1_geoms)}/{len(link2_geoms)}",
            flush=True,
        )
        if link2_geoms and not mask1_geoms:
            print(
                "[env] WARNING: the loaded G1 asset has no finger link2 collision geoms "
                "with mask 1; re-import orcalab_demo/assets/g1_29dof_wuji/g1_29dof_wuji.xml "
                "so a grasped handle cannot sink into the proximal phalanx.",
                flush=True,
            )

    def reset_model(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self._names_ready:
            self._resolve_names()
        qpos = np.array(self.data.qpos, copy=True)
        qvel = np.array(self.data.qvel, copy=True)
        base_address = self.jnt_qposadr(self.base_joint)
        if self._initial_base_qpos is None:
            self._initial_base_qpos = qpos[base_address : base_address + 7].copy()
        qpos[base_address : base_address + 7] = self._initial_base_qpos
        qpos[base_address] += self.start_offset_x
        base_dof = self.jnt_dofadr(self.base_joint)
        qvel[base_dof : base_dof + 6] = 0.0
        for i, joint_name in enumerate(self.joint_names):
            address = self.jnt_qposadr(joint_name)
            qpos[address] = self.init_pose_sonic[SONIC_TO_G1[i]]
            qvel[self.jnt_dofadr(joint_name)] = 0.0
        if self.init_wuji_hand_state is not None:
            if self.hand_mode == "wuji":
                hand_target = np.clip(
                    self.init_wuji_hand_state,
                    self.hand_ranges[:, 0],
                    self.hand_ranges[:, 1],
                )
            else:
                hand_target = np.concatenate(
                    [
                        retarget_wuji_to_dex3(self.init_wuji_hand_state[:20], self.hand_ranges[:7]),
                        retarget_wuji_to_dex3(self.init_wuji_hand_state[20:], self.hand_ranges[7:]),
                    ]
                )
            for joint_name, value in zip(self.hand_joint_names, hand_target, strict=True):
                qpos[self.jnt_qposadr(joint_name)] = value
                qvel[self.jnt_dofadr(joint_name)] = 0.0
        self.set_joint_qpos(qpos)
        self.set_joint_qvel(qvel)
        self.ctrl = np.zeros(self.model.nu, dtype=np.float32)
        self.mj_forward()
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> dict[str, np.ndarray]:
        all_joint_names = self.joint_names + self.hand_joint_names
        joint_qpos = self.query_joint_qpos(all_joint_names)
        joint_qvel = self.query_joint_qvel(all_joint_names)
        qpos = np.array([joint_qpos[name][0] for name in self.joint_names], dtype=np.float32)
        qvel = np.array([joint_qvel[name][0] for name in self.joint_names], dtype=np.float32)
        hand_qpos = np.array([joint_qpos[name][0] for name in self.hand_joint_names], dtype=np.float32)
        hand_qvel = np.array([joint_qvel[name][0] for name in self.hand_joint_names], dtype=np.float32)
        base_qpos = np.asarray(self.query_joint_qpos([self.base_joint])[self.base_joint], dtype=np.float64)
        base_qvel = np.asarray(self.query_joint_qvel([self.base_joint])[self.base_joint], dtype=np.float64)
        quat = np.asarray(base_qpos[3:7], dtype=np.float32)
        # MuJoCo free-joint qvel stores angular velocity in the body frame;
        # rotating it again with R^T would corrupt the WB/SONIC gyro input.
        omega_body = np.asarray(base_qvel[3:6], dtype=np.float32)
        return {
            "qpos": qpos,
            "qvel": qvel,
            "hand_qpos": hand_qpos,
            "hand_qvel": hand_qvel,
            "base_quat_wxyz": quat,
            "base_omega_body": omega_body,
            "base_height": float(base_qpos[2]),
            "base_xy": np.asarray(base_qpos[:2], dtype=np.float32),
        }

    def apply_body_torques(self, torques: np.ndarray, hand_torques: np.ndarray) -> None:
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        ctrl[self.actuator_ids] = np.asarray(torques, dtype=np.float32)
        ctrl[self.hand_actuator_ids] = np.asarray(hand_torques, dtype=np.float32)
        self.set_ctrl(ctrl)
        if self.cart_damping or self.hand_damping:
            qvel = np.array(self.data.qvel, copy=True)
            for dof, damping in self.cart_damping.items():
                qvel[dof] *= np.exp(-damping * self._time_step)
            if self.hand_damping:
                qvel[self.hand_dof_addresses] *= np.exp(-self.hand_damping * self._time_step)
            self.set_joint_qvel(qvel)
        self.mj_step(1)

    def hand_cart_contacts(self, *, handle_only: bool = False) -> list[dict[str, Any]]:
        geom_dict = self.model.get_geom_dict()
        rows = []
        for contact in self.query_contact_simple():
            geom_names = (
                self.model.geom_id2name(contact["geom1"]),
                self.model.geom_id2name(contact["geom2"]),
            )
            is_handle = self.cart_handle_geom in geom_names
            if handle_only and not is_handle:
                continue
            body_names = tuple(str(geom_dict[name]["BodyName"]) for name in geom_names)
            if not any(name.startswith("[") for name in body_names):
                continue
            side = None
            for body_name in body_names:
                if not body_name.startswith(self.robot_prefix) or not any(
                    part in body_name for part in ("palm", "finger", "hand")
                ):
                    continue
                if "_left_" in body_name:
                    side = "left"
                    break
                if "_right_" in body_name:
                    side = "right"
                    break
            if side is None:
                continue
            rows.append(
                {
                    "side": side,
                    "geoms": geom_names,
                    "bodies": body_names,
                    "handle": is_handle,
                    "pos": np.asarray(contact["pos"]).tolist(),
                    "dist": float(contact["dist"]),
                }
            )
        return rows

    def cart_handle_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.cart_handle_geom is None:
            return None
        meta = self.model.get_geom_dict()[self.cart_handle_geom]
        body_name = str(meta["BodyName"])
        body_pose = self.get_body_xpos_xmat_xquat([body_name])[body_name]
        body_rotation = np.asarray(body_pose["xmat"], dtype=np.float64).reshape(3, 3)
        center = np.asarray(body_pose["xpos"], dtype=np.float64) + body_rotation @ np.asarray(
            meta["Pos"], dtype=np.float64
        )
        rotation = body_rotation @ quat_wxyz_to_rotmat(np.asarray(meta["Quat"], dtype=np.float64))
        axis = rotation[:, 2]
        return center, axis / np.linalg.norm(axis)


class PolicyClient:
    def __init__(self, url: str, timeout_s: float = 300.0) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._pending: tuple[int, np.ndarray, np.ndarray] | None = None
        self._result: tuple[int, int, np.ndarray] | None = None
        self._request_id = 0
        self._busy = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import requests

        while not self._stop:
            with self._lock:
                pending = self._pending
                self._pending = None
            if pending is None:
                time.sleep(0.005)
                continue
            request_id, state, image = pending
            try:
                payload = {
                    "state": state.tolist(),
                    "image": {
                        "shape": list(image.shape),
                        "dtype": "uint8",
                        "data_b64": base64.b64encode(image.tobytes()).decode("ascii"),
                    },
                }
                response = requests.post(f"{self.url}/infer", json=payload, timeout=self.timeout_s)
                response.raise_for_status()
                result = response.json()
                if not result.get("dynamic_base_quat"):
                    raise RuntimeError("policy server does not provide actual_imu dynamic encoding")
                context_id = int(result["context_id"])
                chunk = np.asarray(result["action_chunk"], dtype=np.float32)
                if chunk.ndim != 2 or chunk.shape[0] < 20 or chunk.shape[1] != 104:
                    raise ValueError(f"unexpected action chunk shape {chunk.shape}")
                chunk = chunk[:20]
                if not np.isfinite(chunk).all():
                    raise ValueError("action chunk contains NaN or infinity")
                with self._lock:
                    self._result = (request_id, context_id, chunk)
            except Exception as exc:
                print(f"[policy] inference failed: {type(exc).__name__}: {exc}", flush=True)
            finally:
                with self._lock:
                    self._busy = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(self, state: np.ndarray, image: np.ndarray) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._request_id += 1
            self._pending = (self._request_id, state.copy(), image.copy())
            return True

    def take(self) -> tuple[int, int, np.ndarray] | None:
        with self._lock:
            result = self._result
            self._result = None
            return result

    def materialize(
        self,
        context_id: int,
        action_index: int,
        base_quat_wxyz: np.ndarray,
        *,
        stationary: bool,
    ) -> np.ndarray:
        import requests

        response = requests.post(
            f"{self.url}/materialize",
            json={
                "context_id": int(context_id),
                "action_index": int(action_index),
                "base_quat_wxyz": np.asarray(base_quat_wxyz, dtype=float).tolist(),
                "stationary": bool(stationary),
            },
            timeout=15.0,
        )
        response.raise_for_status()
        action = np.asarray(response.json()["action"], dtype=np.float32)
        if action.shape != (104,) or not np.isfinite(action).all():
            raise ValueError(f"invalid materialized action shape {action.shape}")
        return action

    def commit(self, context_id: int, action_index: int, action: np.ndarray) -> None:
        import requests

        response = requests.post(
            f"{self.url}/commit",
            json={
                "context_id": int(context_id),
                "action_index": int(action_index),
                "action": np.asarray(action, dtype=float).tolist(),
            },
            timeout=5.0,
        )
        response.raise_for_status()

    def reset(self) -> None:
        import requests

        with self._lock:
            self._pending = None
            self._result = None
        try:
            requests.post(f"{self.url}/reset", json={}, timeout=5.0)
        except Exception as exc:
            print(f"[policy] reset failed: {exc}", flush=True)

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=2.0)


def _quat_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def handle_envelope_scores(
    contacts: list[dict[str, Any]],
    handle_center: np.ndarray,
    handle_axis: np.ndarray,
) -> dict[str, float]:
    """Measure opposite-side handle contact; 1 is diametric opposition."""
    center = np.asarray(handle_center, dtype=np.float64)
    axis = np.asarray(handle_axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    scores = {"left": -1.0, "right": -1.0}
    for side in ("left", "right"):
        rows = [c for c in contacts if c["handle"] and c["side"] == side]
        for i, first in enumerate(rows):
            for second in rows[i + 1 :]:
                first_hand_body = next(name for name in first["bodies"] if not name.startswith("["))
                second_hand_body = next(name for name in second["bodies"] if not name.startswith("["))
                if first_hand_body == second_hand_body:
                    continue
                first_relative = np.asarray(first["pos"]) - center
                second_relative = np.asarray(second["pos"]) - center
                first_axial = float(first_relative @ axis)
                second_axial = float(second_relative @ axis)
                if abs(first_axial - second_axial) > 0.08:
                    continue
                first_radial = first_relative - first_axial * axis
                second_radial = second_relative - second_axial * axis
                if np.linalg.norm(first_radial) < 1e-6 or np.linalg.norm(second_radial) < 1e-6:
                    continue
                opposition = -float(
                    (first_radial @ second_radial)
                    / (np.linalg.norm(first_radial) * np.linalg.norm(second_radial))
                )
                scores[side] = max(scores[side], opposition)
    return scores


def camera_image(env: G1PushCartEnv, camera_name: str) -> np.ndarray | None:
    manager = env.get_recorder_manager()
    frame = manager.get_last_decoded_frame(camera_name)
    if frame is None:
        return None
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    from PIL import Image

    return np.asarray(Image.fromarray(image).resize((224, 224)), dtype=np.uint8)


def configure_orcalab_camera(url: str, actor_path: str, *, reset_scene: bool = False) -> None:
    """Restore the head-camera mounting and optionally reset the demo scene."""
    import requests

    session = requests.Session()
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    def parse(response: requests.Response) -> dict[str, Any]:
        response.raise_for_status()
        text = response.content.decode("utf-8", "replace")
        data = "".join(
            line[5:].lstrip()
            for line in text.replace("\r\n", "\n").split("\n")
            if line.startswith("data:")
        )
        return json.loads(data or text)

    response = session.post(
        url,
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "wbwam-orcalab", "version": "1.0"},
            },
        },
        timeout=10,
    )
    parse(response)
    headers["mcp-session-id"] = response.headers["mcp-session-id"]
    session.post(url, headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, timeout=10)
    next_request_id = 2
    g1_actor = None
    if actor_path == "auto" or reset_scene:
        result = parse(
            session.post(
                url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": next_request_id,
                    "method": "tools/call",
                    "params": {"name": "get_all_actors", "arguments": {}},
                },
                timeout=15,
            )
        )
        next_request_id += 1
        actors = json.loads(result["result"]["content"][0]["text"])
        g1_paths = [
            path
            for path, meta in actors.items()
            if meta.get("type") == "AssetActor" and path.startswith("/g1")
        ]
        if len(g1_paths) != 1:
            raise RuntimeError(f"expected exactly one G1 actor for auto camera setup, got {g1_paths}")
        g1_actor = g1_paths[0]
        if actor_path == "auto":
            actor_path = f"{g1_actor}/head_cam"
    values = {
        "translate.x": 0.05,
        "translate.y": 0.0,
        "translate.z": 0.43,
        "rotate.x": 0.0,
        "rotate.y": 40.0,
        "rotate.z": -90.0,
        "VerticalFieldOfViewDeg": 60.0,
        "ColorCamera": True,
        "Enable": True,
    }
    for request_id, (name, value) in enumerate(values.items(), start=next_request_id):
        result = parse(
            session.post(
                url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {
                        "name": "set_actor_properties",
                        "arguments": {
                            "asset_path": actor_path,
                            "property_name": name,
                            "property_value": value,
                        },
                    },
                },
                timeout=15,
            )
        )
        if "error" in result:
            raise RuntimeError(result["error"])
    if reset_scene:
        actors = [
            (
                g1_actor,
                [15.943731307983398, 16.561674118041992, 0.0],
                1.0,
            ),
            (
                "/blue_foldable_hand_truck_1",
                [16.872177124023438, 16.58871841430664, 0.012426406145095825],
                0.6,
            ),
        ]
        for request_id, (path, position, scale) in enumerate(actors, start=len(values) + next_request_id):
            result = parse(
                session.post(
                    url,
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "tools/call",
                        "params": {
                            "name": "set_actor_transform",
                            "arguments": {
                                "actor_path": path,
                                "position": position,
                                "rotation": [0.0, 0.0, 0.0],
                                "scale": scale,
                            },
                        },
                    },
                    timeout=15,
                )
            )
            if "error" in result:
                raise RuntimeError(result["error"])
        print("[run] demo actors reset to their initial transforms", flush=True)
    print(f"[run] camera mount restored: {actor_path} rotate=(0,40,-90)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orcagym-addr", default="127.0.0.1:50051")
    parser.add_argument("--policy-url", default="http://127.0.0.1:8765")
    parser.add_argument("--decoder", type=Path, default=Path("checkpoints/sonic_release/model_decoder.onnx"))
    parser.add_argument("--stand-only", action="store_true")
    parser.add_argument("--seconds", type=float, default=360.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--camera", default="auto")
    parser.add_argument("--camera-actor", default="auto")
    parser.add_argument("--orcalab-mcp", default="http://127.0.0.1:12345/mcp")
    parser.add_argument("--reset-scene", action="store_true", help="Reset the demo actors and cart joints before running")
    parser.add_argument("--success-distance", type=float, default=2.0, help="Cart forward distance in meters; 0 disables")
    parser.add_argument("--record", type=Path, default=Path("orcalab_demo/outputs/push_cart_demo.mp4"))
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--start-policy", action="store_true", help="Skip the initial-pose phase")
    parser.add_argument(
        "--init-pose",
        choices=["push-ready", "latent", "sonic-default", "dataset-mean"],
        default="dataset-mean",
        help="Body initialization and hold mode before policy control",
    )
    parser.add_argument("--initial-pose-seconds", type=float, default=None, help="Initial phase in simulation seconds")
    parser.add_argument("--start-offset-x", type=float, default=0.0, help="Initial base x offset in meters")
    parser.add_argument("--hand-mode", choices=["auto", "wuji", "dex3"], default="auto")
    parser.add_argument(
        "--init-hands",
        choices=["open", "mean"],
        default="open",
        help="Initial Wuji hand target: demonstrated open pre-grasp (state q01) or dataset mean (mid-grasp)",
    )
    parser.add_argument("--cart-slide", type=float, default=0.0, help="Cart slide-joint offset in meters")
    parser.add_argument("--cart-friction", type=float, default=None, help="Override cart geom friction")
    parser.add_argument("--cart-damping", type=float, default=4.0, help="Cart slide damping per second")
    parser.add_argument("--cart-yaw-damping", type=float, default=8.0, help="Cart yaw damping per second")
    parser.add_argument("--arm-inward", type=float, default=0.0, help="Inward shoulder-roll bias in radians")
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("push_cart/metadata/dataset_stats.json"),
        help="Dataset stats used for the initial pose and hand state",
    )
    args = parser.parse_args()

    if not args.stand_only and args.camera_actor:
        configure_orcalab_camera(args.orcalab_mcp, args.camera_actor, reset_scene=args.reset_scene)

    frame_skip = 20
    time_step = 0.001
    env = G1PushCartEnv(
        frame_skip=frame_skip,
        orcagym_addr=args.orcagym_addr,
        time_step=time_step,
        render_mode="none" if args.no_render else "human",
    )
    stats_path = args.stats if args.stats.is_absolute() else Path.cwd() / args.stats
    stats = json.loads(stats_path.read_text())["state"]["proprio"]
    mean_state = np.asarray(stats["global_mean"], dtype=np.float32)
    env.init_pose_sonic = SONIC_DEFAULT_POS.copy()
    if args.init_pose == "dataset-mean":
        env.init_pose_sonic += mean_state[:29]
    elif args.init_pose == "push-ready":
        env.init_pose_sonic[15:] += mean_state[15:29]
    arm_inward = float(args.arm_inward)
    if arm_inward:
        env.init_pose_sonic[15] += arm_inward
        env.init_pose_sonic[16] -= arm_inward
    env.start_offset_x = float(args.start_offset_x)
    env.hand_mode_override = args.hand_mode
    if args.init_hands == "open":
        # First percentile of demonstrated hand states: fingers extended and
        # thumb unopposed, i.e. the pre-grasp pose rather than the mean of all
        # approach/grasp/push frames.
        hand_state = np.asarray(stats["global_q01"], dtype=np.float32)[32:72].copy()
    else:
        hand_state = mean_state[32:72].copy()
    env.init_wuji_hand_state = hand_state.copy()
    env.reset()
    # q01 can exceed the URDF limits slightly (thumb opposition); clip the
    # target so policy output and logs use reachable values.
    hand_state = np.clip(hand_state, env.hand_ranges[:, 0], env.hand_ranges[:, 1])
    policy_hand_target = hand_state.copy()

    if args.reset_scene:
        qpos = np.array(env.data.qpos, copy=True)
        qvel = np.array(env.data.qvel, copy=True)
        for name in env.model.get_joint_dict():
            if name.startswith("["):
                qpos[env.jnt_qposadr(name)] = 0.0
                qvel[env.jnt_dofadr(name)] = 0.0
        env.set_joint_qpos(qpos)
        env.set_joint_qvel(qvel)
        env.mj_forward()
        print("[run] cart joints reset to zero", flush=True)

    if args.cart_slide:
        qpos = np.array(env.data.qpos, copy=True)
        for name, meta in env.model.get_joint_dict().items():
            if name.startswith("[") and int(meta["Type"]) == 2:
                qpos[env.jnt_qposadr(name)] += float(args.cart_slide)
                print(f"[run] cart slide joint {name!r} offset={args.cart_slide:.3f}", flush=True)
                break
        env.set_joint_qpos(qpos)
        env.mj_forward()

    if args.cart_friction is not None:
        cart_geoms = [name for name in env.model.get_geom_dict() if name.startswith("[")]
        friction = np.array([args.cart_friction, 0.005, 0.0001], dtype=np.float32)
        env.set_geom_friction({name: friction for name in cart_geoms})
        print(f"[run] cart friction -> {args.cart_friction} for {len(cart_geoms)} geoms", flush=True)

    camera_name = args.camera
    if not args.stand_only:
        manager = env.get_recorder_manager()
        available_cameras = manager.get_camera_names()
        if camera_name == "auto":
            matches = [name for name in available_cameras if name == "head_cam" or name.startswith("head_cam_")]
            if len(matches) != 1:
                raise RuntimeError(f"expected one head_cam backend, got {available_cameras}")
            camera_name = matches[0]
        elif camera_name not in available_cameras:
            raise RuntimeError(f"camera {camera_name!r} is not registered: {available_cameras}")
        print(f"[run] using camera backend {camera_name!r}", flush=True)
        try:
            manager.set_streaming_enabled(camera_name, False)
            manager.set_camera_properties(camera_name, capture_rgb=True)
        except Exception as exc:
            print(f"[run] camera setup warning: {exc}", flush=True)
        time.sleep(0.5)
        env.start_streaming(camera_name)
        deadline = time.monotonic() + 10.0
        initial_image = None
        while time.monotonic() < deadline:
            env.render()
            initial_image = camera_image(env, camera_name)
            if initial_image is not None:
                break
            time.sleep(0.05)
        if initial_image is None:
            raise RuntimeError(f"camera {camera_name!r} returned no RGB frame; refusing zero-image inference")
        print(f"[run] camera {camera_name!r} ready with real RGB frames", flush=True)

    decoder = SonicDecoder(args.decoder)
    policy = None if args.stand_only else PolicyClient(args.policy_url)
    if policy is not None:
        policy.reset()

    ticks = int(args.seconds * 50)
    sim_dt = float(time_step)
    initial_pose_seconds = args.initial_pose_seconds
    if initial_pose_seconds is None:
        initial_pose_seconds = {
            "latent": 2.0,
            "dataset-mean": 0.0,
            "push-ready": 0.6,
            "sonic-default": 0.6,
        }[args.init_pose]
    stand_ticks = 0 if args.start_policy else int(initial_pose_seconds / sim_dt)
    print(f"[run] init_pose={args.init_pose} phase={initial_pose_seconds:.2f}s (simulation)", flush=True)
    decoder_dt = 0.02
    action_dt = 0.05
    chunk_exec_steps = 20
    sim_time = 0.0
    chunk_context_id: int | None = None
    chunk_index = chunk_exec_steps
    current_action: np.ndarray | None = None
    current_action_index: int | None = None
    next_action_sim = 0.0
    chunk_count = 0
    target_sonic = env.init_pose_sonic.copy()
    last_decoded_target = env.init_pose_sonic.copy()
    policy_handoff_start: np.ndarray | None = None
    policy_handoff_start_sim = 0.0
    policy_handoff_seconds = 0.5
    last_decoder_sim = -decoder_dt
    task_status = "timeout"
    start = time.monotonic()
    joint_dict = env.model.get_joint_dict()
    aux_joints = [name for name in joint_dict if name.startswith("[")]
    cart_slide_joint = next((name for name in aux_joints if int(joint_dict[name]["Type"]) == 2), None)
    cart_yaw_joint = next((name for name in aux_joints if int(joint_dict[name]["Type"]) == 3), None)
    cart_slide_origin = (
        float(np.asarray(env.query_joint_qpos([cart_slide_joint])[cart_slide_joint])[0])
        if cart_slide_joint is not None
        else 0.0
    )
    if cart_slide_joint is not None and args.cart_damping > 0:
        env.cart_damping[env.jnt_dofadr(cart_slide_joint)] = float(args.cart_damping)
    if cart_yaw_joint is not None and args.cart_yaw_damping > 0:
        env.cart_damping[env.jnt_dofadr(cart_yaw_joint)] = float(args.cart_yaw_damping)
    print(f"[run] cart joint damping: {env.cart_damping}", flush=True)
    arm_bias_g1 = np.zeros(29, dtype=np.float32)
    if arm_inward:
        arm_bias_g1[env.joint_suffixes.index("left_shoulder_roll_joint")] = arm_inward
        arm_bias_g1[env.joint_suffixes.index("right_shoulder_roll_joint")] = -arm_inward

    import cv2

    writer = None
    last_video_frame = None
    record_path = args.record if args.record.is_absolute() else Path.cwd() / args.record
    if not args.stand_only and not args.no_record:
        record_path.parent.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    sonic_index = np.argsort(SONIC_TO_G1)
    initial_request_sent = False

    def request_inference(obs: dict[str, np.ndarray]) -> bool:
        if policy is None or policy.busy:
            return False
        image = camera_image(env, camera_name)
        if image is None:
            return False
        gravity = gravity_in_base_frame(obs["base_quat_wxyz"])
        policy_hand_state = obs["hand_qpos"] if env.hand_mode == "wuji" else hand_state
        state = np.concatenate(
            [
                gravity,
                obs["base_omega_body"],
                np.zeros(3, dtype=np.float32),
                obs["qpos"][sonic_index] - SONIC_DEFAULT_POS,
                obs["qvel"][sonic_index],
                policy_hand_state,
            ]
        ).astype(np.float32)
        if not policy.submit(state, image):
            return False
        print(
            f"[run] request after chunk={chunk_count} sim={sim_time:.3f}s "
            f"image_mean={float(image.mean()):.1f} gravity={np.round(gravity, 3).tolist()}",
            flush=True,
        )
        return True

    try:
        for tick in range(ticks):
            obs = env._get_obs()
            policy_active = policy is not None and (args.start_policy or tick >= stand_ticks)
            if policy is not None and not initial_request_sent:
                initial_request_sent = request_inference(obs)
            pause_for_inference = False

            if policy_active:
                assert policy is not None
                result = policy.take()
                if result is not None:
                    if chunk_context_id is None or chunk_index >= chunk_exec_steps:
                        chunk_context_id = result[1]
                        chunk_index = 0
                        current_action_index = None
                        next_action_sim = sim_time
                        if chunk_count == 0:
                            policy_handoff_start = target_sonic.copy()
                            policy_handoff_start_sim = sim_time
                        chunk_count += 1
                        last_decoder_sim = sim_time - decoder_dt
                        print(
                            f"[run] start action chunk {chunk_count} from inference {result[0]} "
                            f"at sim={sim_time:.3f}s",
                            flush=True,
                        )
                    else:
                        print("[run] discarded action chunk received during sequential execution", flush=True)

                if chunk_context_id is None or chunk_index >= chunk_exec_steps:
                    request_inference(obs)
                pause_for_inference = policy.busy and (
                    chunk_context_id is None or chunk_index >= chunk_exec_steps
                )

                if not pause_for_inference and chunk_context_id is not None and sim_time >= next_action_sim - 1e-9:
                    scheduled_hold = chunk_index >= chunk_exec_steps
                    action_index = min(chunk_index, chunk_exec_steps - 1)
                    try:
                        action = policy.materialize(
                            chunk_context_id,
                            action_index,
                            obs["base_quat_wxyz"],
                            stationary=scheduled_hold,
                        )
                    except Exception as exc:
                        print(f"[run] materialize failed; holding previous action: {exc}", flush=True)
                    else:
                        if not scheduled_hold:
                            desired_hands = action[64:104]
                            policy_hand_target = np.asarray(desired_hands, dtype=np.float32).copy()
                            hand_state = hand_state + np.clip(desired_hands - hand_state, -0.06, 0.06)
                        current_action = action.copy()
                        current_action[64:104] = hand_state
                        current_action_index = action_index
                        if not scheduled_hold:
                            chunk_index += 1
                            if chunk_index == chunk_exec_steps:
                                try:
                                    policy.commit(chunk_context_id, action_index, current_action)
                                except Exception as exc:
                                    print(f"[run] action-context commit warning: {exc}", flush=True)
                    next_action_sim = sim_time + action_dt

                if (
                    not pause_for_inference
                    and current_action is not None
                    and sim_time - last_decoder_sim >= decoder_dt - 1e-9
                ):
                    last_decoded_target, _ = decoder.step(
                        current_action[:64],
                        obs["qpos"][sonic_index],
                        obs["qvel"][sonic_index],
                        obs["base_omega_body"],
                        obs["base_quat_wxyz"],
                    )
                    last_decoder_sim = sim_time
                target_sonic = last_decoded_target
                if policy_handoff_start is not None:
                    handoff_progress = min(
                        1.0,
                        max(0.0, (sim_time - policy_handoff_start_sim) / policy_handoff_seconds),
                    )
                    target_sonic = policy_handoff_start + handoff_progress * (
                        last_decoded_target - policy_handoff_start
                    )
                    if handoff_progress >= 1.0:
                        policy_handoff_start = None
            else:
                current_action = None
                current_action_index = None
                if args.init_pose == "latent":
                    if sim_time - last_decoder_sim >= decoder_dt - 1e-9:
                        last_decoded_target, _ = decoder.step(
                            LATENT_INITIAL_MOTION_TOKEN,
                            obs["qpos"][sonic_index],
                            obs["qvel"][sonic_index],
                            obs["base_omega_body"],
                            obs["base_quat_wxyz"],
                        )
                        last_decoder_sim = sim_time
                else:
                    last_decoded_target = env.init_pose_sonic.copy()
                    last_decoder_sim = sim_time - decoder_dt
                target_sonic = last_decoded_target

            target_g1 = target_sonic[SONIC_TO_G1] + arm_bias_g1
            kp = SONIC_PD_KP[SONIC_TO_G1]
            kd = SONIC_PD_KD[SONIC_TO_G1]
            torques = kp * (target_g1 - obs["qpos"]) - kd * obs["qvel"]
            torques = np.clip(torques, -G1_EFFORT_LIMITS, G1_EFFORT_LIMITS)
            if env.hand_mode == "wuji":
                hand_target = np.clip(hand_state, env.hand_ranges[:, 0], env.hand_ranges[:, 1])
            else:
                hand_target = np.concatenate(
                    [
                        retarget_wuji_to_dex3(hand_state[:20], env.hand_ranges[:7]),
                        retarget_wuji_to_dex3(hand_state[20:], env.hand_ranges[7:]),
                    ]
                )
            hand_kp = 0.3 if env.hand_mode == "wuji" else 2.0
            hand_kd = 0.01 if env.hand_mode == "wuji" else 0.1
            hand_torques = hand_kp * (hand_target - obs["hand_qpos"]) - hand_kd * obs["hand_qvel"]
            hand_torques = np.clip(hand_torques, -env.hand_torque_limits, env.hand_torque_limits)
            if not pause_for_inference:
                env.apply_body_torques(torques, hand_torques)
                sim_time += sim_dt
            if not args.no_render and tick % 2 == 0:
                try:
                    env.render()
                except Exception as exc:
                    task_status = "failed"
                    print(f"[run] OrcaLab render connection failed: {exc}", flush=True)
                    break
                if (
                    not args.stand_only
                    and not args.no_record
                    and not pause_for_inference
                    and chunk_context_id is not None
                ):
                    frame = manager.get_last_decoded_frame(camera_name)
                    if frame is not None:
                        frame = np.asarray(frame)
                        if frame.ndim == 3 and frame.shape[2] >= 3:
                            last_video_frame = frame[..., :3]
                            if writer is None:
                                height, width = last_video_frame.shape[:2]
                                writer = cv2.VideoWriter(
                                    str(record_path),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    25.0,
                                    (width, height),
                                )
                            writer.write(cv2.cvtColor(last_video_frame, cv2.COLOR_RGB2BGR))

            if tick % 50 == 0:
                yaw = _quat_yaw(obs["base_quat_wxyz"])
                index = current_action_index
                aux = {
                    name: float(np.asarray(env.query_joint_qpos([name])[name])[0])
                    for name in aux_joints
                }
                cart_distance = (
                    aux[cart_slide_joint] - cart_slide_origin if cart_slide_joint is not None else 0.0
                )
                cart_yaw = aux[cart_yaw_joint] if cart_yaw_joint is not None else 0.0
                hand_cart_contacts = env.hand_cart_contacts()
                handle_pose = env.cart_handle_pose()
                handle_center = None
                envelope_scores = {"left": -1.0, "right": -1.0}
                if handle_pose is not None:
                    handle_center, handle_axis = handle_pose
                    envelope_scores = handle_envelope_scores(
                        hand_cart_contacts, handle_center, handle_axis
                    )
                    handle_center = handle_center.tolist()
                envelope_grasp_now = np.zeros(2, dtype=bool)
                if env.hand_mode == "wuji":
                    for side_index, side in enumerate(("left", "right")):
                        closure = float(
                            np.mean(obs["hand_qpos"][side_index * 20 : (side_index + 1) * 20])
                        )
                        envelope_grasp_now[side_index] = (
                            closure > 0.25 and envelope_scores[side] > 0.5
                        )
                print(
                    f"[run] t={time.monotonic() - start:5.1f}s sim={sim_time:5.2f}s "
                    f"height={obs['base_height']:.3f} xy=({obs['base_xy'][0]:.2f},{obs['base_xy'][1]:.2f}) "
                    f"yaw={yaw:+.3f} chunks={chunk_count} index={index} "
                    f"cart_forward={cart_distance:+.3f}m cart_yaw={cart_yaw:+.3f} "
                    f"hand_contacts={len(hand_cart_contacts)} "
                    f"envelope={envelope_grasp_now.tolist()} "
                    f"opposition=({envelope_scores['left']:.2f},{envelope_scores['right']:.2f})",
                    flush=True,
                )
                metrics.append(
                    {
                        "t": time.monotonic() - start,
                        "sim_t": sim_time,
                        "height": float(obs["base_height"]),
                        "x": float(obs["base_xy"][0]),
                        "y": float(obs["base_xy"][1]),
                        "yaw": float(yaw),
                        "chunk_count": chunk_count,
                        "has_chunk": float(chunk_context_id is not None),
                        "action_index": current_action_index,
                        "hand_mode": env.hand_mode,
                        "handle_center": handle_center,
                        "envelope_opposition": envelope_scores,
                        "envelope_grasp": envelope_grasp_now.tolist(),
                        "arm_inward": arm_inward,
                        "hand_qpos": obs["hand_qpos"].tolist(),
                        "hand_target": np.asarray(hand_target).tolist(),
                        "hand_policy_target": policy_hand_target.tolist(),
                        "hand_cart_contacts": hand_cart_contacts,
                        "cart_forward": cart_distance,
                        "cart_yaw": cart_yaw,
                        "cart": aux,
                    }
                )
                if (
                    args.success_distance > 0
                    and cart_slide_joint is not None
                    and cart_distance >= args.success_distance
                    and obs["base_height"] >= 0.35
                ):
                    task_status = "success"
                    print(f"[run] push-cart SUCCESS: forward={cart_distance:.3f}m", flush=True)
                    break
            elapsed = time.monotonic() - start
            if elapsed < (tick + 1) / 50.0:
                time.sleep((tick + 1) / 50.0 - elapsed)

            if obs["base_height"] < 0.35:
                task_status = "failed"
                print(f"[run] robot fell (height={obs['base_height']:.3f}); stopping", flush=True)
                break
    finally:
        print(f"[run] task status: {task_status}", flush=True)
        if writer is not None:
            writer.release()
            print(f"[run] video saved: {record_path}", flush=True)
        if metrics:
            metrics[-1]["task_status"] = task_status
            metrics_path = record_path.with_suffix(".json")
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2))
            print(f"[run] metrics saved: {metrics_path}", flush=True)
        if policy is not None:
            policy.close()
        try:
            env.close()
        except Exception as exc:
            print(f"[run] OrcaLab close warning: {exc}", flush=True)


if __name__ == "__main__":
    main()
