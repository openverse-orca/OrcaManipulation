"""TeleVuer XR backend for g1_pick (does not fake PicoJoystickKey).

Provides backend-agnostic bind APIs for dual SE3 poses, triggers, and
task/discard/shutdown events with squeeze hysteresis + chord arbitration.
"""
from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    from devices.abstract_device import AbstractDevice
except ImportError:  # unit tests without orca_gym
    class AbstractDevice:  # type: ignore[no-redef]
        def update(self):
            raise NotImplementedError

from devices.g1_pick_tv_pose_mapper import (
    TvToOrcaPoseMapper,
    normalize_televuer_trigger,
)


@dataclass
class _HysteresisButton:
    # Pico WebXR often reports squeeze as bool with squeezeValue≈0; keep thr modest.
    press_thr: float = 0.5
    release_thr: float = 0.2
    pressed: bool = False

    def update(self, value: float) -> bool:
        """Return True on rising edge only."""
        v = float(value)
        rising = False
        if not self.pressed and v >= self.press_thr:
            self.pressed = True
            rising = True
        elif self.pressed and v <= self.release_thr:
            self.pressed = False
        return rising


def _squeeze_analog(pressed: bool, value: float) -> float:
    """Merge bool squeeze with analog value (Pico may only set the bool)."""
    v = float(value)
    if pressed:
        return max(v, 1.0)
    return v


def _port_listening(port: int, host: str = "127.0.0.1") -> bool:
    """True if something accepts TCP on host:port (best-effort)."""
    try:
        with socket.create_connection((host, int(port)), timeout=0.15):
            return True
    except OSError:
        return False


class SqueezeChordArbitrator:
    """Arbitrate L/R squeeze rising edges with a short chord window.

    Priority:
      both within window → shutdown
      right only         → discard
      left only          → task_toggle
    """

    def __init__(self, chord_window_s: float = 0.08):
        self.chord_window_s = float(chord_window_s)
        self._l_btn = _HysteresisButton()
        self._r_btn = _HysteresisButton()
        self._pending_l_t: float | None = None
        self._pending_r_t: float | None = None

    def update(
        self, left_squeeze: float, right_squeeze: float, now: float | None = None
    ) -> str | None:
        """Return one of: 'shutdown' | 'discard' | 'task_toggle' | None."""
        t = time.monotonic() if now is None else float(now)
        l_rise = self._l_btn.update(left_squeeze)
        r_rise = self._r_btn.update(right_squeeze)
        if l_rise:
            self._pending_l_t = t
        if r_rise:
            self._pending_r_t = t

        # Both currently pressed after a chord → shutdown (even if edges staggered)
        if self._l_btn.pressed and self._r_btn.pressed:
            if self._pending_l_t is not None or self._pending_r_t is not None:
                self._pending_l_t = None
                self._pending_r_t = None
                return "shutdown"

        # Resolve pending single-side edges after window
        action = None
        if self._pending_l_t is not None and (t - self._pending_l_t) >= self.chord_window_s:
            if not self._r_btn.pressed:
                action = "task_toggle"
            self._pending_l_t = None
        if self._pending_r_t is not None and (t - self._pending_r_t) >= self.chord_window_s:
            if not self._l_btn.pressed:
                # right takes priority over a simultaneous pending left if both expire
                action = "discard"
            self._pending_r_t = None
        return action


def _selfcheck_trigger_normalize() -> None:
    """D7: assert tv_wrapper-style inverted trigger → HandController [0,1].

    TeleVuerWrapper exposes left_ctrl_triggerValue = 10 - raw*10 (raw in [0,1]).
    normalize_televuer_trigger maps that back to close amount in [0,1].
    """
    # raw=0 (released) → wrapper 10 → close 0
    # raw=1 (pressed)  → wrapper 0  → close 1
    cases = (
        (10.0, 0.0),
        (0.0, 1.0),
        (5.0, 0.5),
    )
    for tv_val, expect in cases:
        got = normalize_televuer_trigger(tv_val)
        if abs(got - expect) > 1e-6:
            raise RuntimeError(
                f"trigger normalize broken: in={tv_val} got={got} expect={expect}"
            )


class TeleVuerDevice(AbstractDevice):
    """Live TeleVuer XR device. Requires optional `televuer` package."""

    VUER_PORT = 8012

    def __init__(
        self,
        pose_mapper: TvToOrcaPoseMapper | None = None,
        disconnect_timeout_s: float = 1.0,
        sustained_timeout_s: float = 3.0,
        chord_window_s: float = 0.08,
        display_mode: str = "pass-through",
        img_shape: tuple[int, int] = (480, 640),
        binocular: bool = False,
        run_trigger_selfcheck: bool = True,
        cert_file: str | None = None,
        key_file: str | None = None,
    ):
        self.pose_mapper = pose_mapper or TvToOrcaPoseMapper()
        self.disconnect_timeout_s = float(disconnect_timeout_s)
        self.sustained_timeout_s = float(sustained_timeout_s)
        self._arbitrator = SqueezeChordArbitrator(chord_window_s=chord_window_s)

        self._dual_pose_cb: Callable | None = None
        self._left_trigger_cb: Callable | None = None
        self._right_trigger_cb: Callable | None = None
        self._task_toggle_cb: Callable | None = None
        self._discard_cb: Callable | None = None
        self._shutdown_cb: Callable | None = None
        self._disconnect_cb: Callable | None = None
        self._reconnect_cb: Callable | None = None

        self._event_seq = 0
        # Must start at 0: TeleVuer shared counter begins at 0, and `-1 → 0`
        # on the first update would falsely look like a controller event
        # (spurious reconnect / disconnect).
        self._last_event_seq = 0
        self._last_cam_seq = 0
        self._last_ctrl_seq = 0
        self._cam_seq = 0
        self._ctrl_seq = 0
        self._last_event_monotonic = 0.0
        self._last_cam_monotonic = 0.0
        self._last_ctrl_monotonic = 0.0
        self._connected = False
        self._was_connected = False
        self._closed = False
        self._child_was_alive = True
        # Empty string "" disables TLS (Vuer skips cert resolution + SSL site).
        # None = TeleVuer default path (~/.config/xr_teleoperate or package cert).
        self._cert_file = cert_file
        self._key_file = key_file
        self._tls_enabled = not (
            isinstance(cert_file, str)
            and isinstance(key_file, str)
            and cert_file == ""
            and key_file == ""
        )

        # Rate samples for heartbeat (cam/ctrl events over last window)
        self._rate_t0 = time.monotonic()
        self._rate_cam0 = 0
        self._rate_ctrl0 = 0

        self._tv_wrapper = None
        self._counting_tv = None
        self._vuer_process = None

        if run_trigger_selfcheck:
            _selfcheck_trigger_normalize()

        self._init_televuer(
            display_mode=display_mode,
            img_shape=img_shape,
            binocular=binocular,
        )

    @property
    def tls_enabled(self) -> bool:
        return self._tls_enabled

    def client_visit_url(self) -> str:
        """USB-local URL Pico should open (overrides Vuer's misleading Visit: vuer.ai).

        vuer 0.0.60 getSocketURI drops the port on the HTTPS branch, so TLS mode
        must pass ?ws=wss://127.0.0.1:8012. Plain HTTP branch keeps the port.
        """
        port = self.VUER_PORT
        if self._tls_enabled:
            return f"https://127.0.0.1:{port}/?ws=wss://127.0.0.1:{port}"
        return f"http://127.0.0.1:{port}/"

    def _init_televuer(self, display_mode: str, img_shape: tuple[int, int], binocular: bool):
        try:
            from multiprocessing import Value
            from televuer.televuer import TeleVuer
            from televuer.tv_wrapper import TeleVuerWrapper
        except ImportError as e:
            raise ImportError(
                "televuer is not installed. Please install the xr_teleoperate "
                "televuer package before running Unitree teleoperation."
            ) from e

        class CountingTeleVuer(TeleVuer):
            def __init__(self_inner, *args, **kwargs):
                # Shared counters must exist before Process.start inside TeleVuer.__init__
                # NOTE: monkeypatch depends on TeleVuerWrapper looking up
                # `televuer.tv_wrapper.TeleVuer` at call time (module attribute),
                # not a bind-at-import alias. If upstream changes that, this breaks.
                self_inner._event_seq = Value("Q", 0)
                self_inner._cam_seq = Value("Q", 0)
                self_inner._ctrl_seq = Value("Q", 0)
                super().__init__(*args, **kwargs)

            def _bump(self_inner, which: str = "any"):
                with self_inner._event_seq.get_lock():
                    self_inner._event_seq.value += 1
                if which == "cam":
                    with self_inner._cam_seq.get_lock():
                        self_inner._cam_seq.value += 1
                elif which == "ctrl":
                    with self_inner._ctrl_seq.get_lock():
                        self_inner._ctrl_seq.value += 1

            async def on_controller_move(self_inner, event, session, fps=60):
                await super().on_controller_move(event, session, fps)
                self_inner._bump("ctrl")

            async def on_cam_move(self_inner, event, session, fps=60):
                await super().on_cam_move(event, session, fps)
                # Head tracking proves WSS uplink; does NOT prove controller stream.
                self_inner._bump("cam")

        # Ensure TeleVuerWrapper constructs our counting subclass.
        import televuer.tv_wrapper as wrap_mod

        self._orig_televuer_cls = wrap_mod.TeleVuer
        wrap_mod.TeleVuer = CountingTeleVuer
        wrap_kwargs = dict(
            use_hand_tracking=False,
            binocular=binocular,
            img_shape=img_shape,
            display_mode=display_mode,
            zmq=False,
            webrtc=False,
            arm_reference_mode="head_yaw",
        )
        # Only pass cert args when caller set them (None = TeleVuer default search).
        if self._cert_file is not None:
            wrap_kwargs["cert_file"] = self._cert_file
        if self._key_file is not None:
            wrap_kwargs["key_file"] = self._key_file
        try:
            self._tv_wrapper = TeleVuerWrapper(**wrap_kwargs)
            self._counting_tv = self._tv_wrapper.tvuer
            # Child Process started inside TeleVuer.__init__ as .process
            self._vuer_process = getattr(self._counting_tv, "process", None)
        except Exception:
            wrap_mod.TeleVuer = self._orig_televuer_cls
            if self._tv_wrapper is not None:
                try:
                    self._tv_wrapper.close()
                except Exception:
                    pass
                self._tv_wrapper = None
            raise
        finally:
            wrap_mod.TeleVuer = self._orig_televuer_cls

    # ----- bind API -----
    def bind_dual_pose_event(self, callback: Callable):
        self._dual_pose_cb = callback

    def bind_left_trigger_event(self, callback: Callable[[float], None]):
        self._left_trigger_cb = callback

    def bind_right_trigger_event(self, callback: Callable[[float], None]):
        self._right_trigger_cb = callback

    def bind_task_toggle_event(self, callback: Callable):
        self._task_toggle_cb = callback

    def bind_discard_event(self, callback: Callable):
        self._discard_cb = callback

    def bind_shutdown_event(self, callback: Callable):
        self._shutdown_cb = callback

    def bind_disconnect_event(self, callback: Callable):
        self._disconnect_cb = callback

    def bind_reconnect_event(self, callback: Callable):
        self._reconnect_cb = callback

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def event_seq(self) -> int:
        return int(self._event_seq)

    @property
    def cam_seq(self) -> int:
        return int(self._cam_seq)

    @property
    def ctrl_seq(self) -> int:
        return int(self._ctrl_seq)

    def _read_shared_seq(self, attr: str, fallback: int) -> int:
        tv = self._counting_tv
        if tv is None or not hasattr(tv, attr):
            return fallback
        shared = getattr(tv, attr)
        with shared.get_lock():
            return int(shared.value)

    def _read_event_seq(self) -> int:
        return self._read_shared_seq("_event_seq", self._event_seq)

    def _read_cam_seq(self) -> int:
        return self._read_shared_seq("_cam_seq", self._cam_seq)

    def _read_ctrl_seq(self) -> int:
        return self._read_shared_seq("_ctrl_seq", self._ctrl_seq)

    def child_alive(self) -> bool:
        """True if Vuer multiprocessing child is still running."""
        p = self._vuer_process
        if p is None:
            return False
        try:
            return bool(p.is_alive())
        except Exception:
            return False

    def child_exitcode(self):
        p = self._vuer_process
        if p is None:
            return None
        return getattr(p, "exitcode", None)

    def port_bound(self, port: int | None = None) -> bool:
        return _port_listening(port if port is not None else self.VUER_PORT)

    def health_snapshot(self) -> dict:
        """Structured health dict (cam/ctrl liveness, child process, port)."""
        now = time.monotonic()
        cam = self._cam_seq
        ctrl = self._ctrl_seq
        dt = max(1e-3, now - self._rate_t0)
        cam_rate = (cam - self._rate_cam0) / dt
        ctrl_rate = (ctrl - self._rate_ctrl0) / dt
        self._rate_t0 = now
        self._rate_cam0 = cam
        self._rate_ctrl0 = ctrl

        uplink_age = (
            now - self._last_event_monotonic if self._last_event_monotonic > 0 else -1.0
        )
        cam_age = (
            now - self._last_cam_monotonic if self._last_cam_monotonic > 0 else -1.0
        )
        ctrl_age = (
            now - self._last_ctrl_monotonic if self._last_ctrl_monotonic > 0 else -1.0
        )
        motion_latched = False
        try:
            if self._tv_wrapper is not None:
                motion_latched = bool(
                    self._tv_wrapper.get_tele_data().motion_data_ready
                )
        except Exception:
            pass

        alive = self.child_alive()
        cam_only = cam > 0 and ctrl == 0
        seq_rising_no_motion = (cam + ctrl) > 30 and not motion_latched
        backpressure = (
            cam_rate > 20.0 and ctrl_rate >= 0.0 and ctrl_rate < cam_rate * 0.25
            and cam_rate - ctrl_rate > 10.0
        )

        return {
            "child": "alive" if alive else "DEAD",
            "child_exitcode": self.child_exitcode(),
            "port8012": "bound" if self.port_bound() else "free",
            "tls": self._tls_enabled,
            "visit": self.client_visit_url(),
            "uplink_age": uplink_age,
            "cam_age": cam_age,
            "ctrl_age": ctrl_age,
            "cam_only": cam_only,
            "connected": self._connected,
            "motion": motion_latched,
            "seq": self._event_seq,
            "cam_seq": cam,
            "ctrl_seq": ctrl,
            "cam_rate": cam_rate,
            "ctrl_rate": ctrl_rate,
            "seq_rising_no_motion": seq_rising_no_motion,
            "backpressure": backpressure,
        }

    def update(self):
        if self._closed or self._tv_wrapper is None:
            return

        now = time.monotonic()

        alive = self.child_alive()
        self._child_was_alive = alive

        seq = self._read_event_seq()
        cam_seq = self._read_cam_seq()
        ctrl_seq = self._read_ctrl_seq()

        # ONLY advancing seq refreshes liveness — never sticky
        # motion_data_ready (which stays True forever after first CONTROLLER_MOVE).
        seq_advanced = seq > self._last_event_seq
        if seq_advanced:
            self._last_event_seq = seq
            self._last_event_monotonic = now
            self._event_seq = seq
        if cam_seq > self._last_cam_seq:
            self._last_cam_seq = cam_seq
            self._last_cam_monotonic = now
            self._cam_seq = cam_seq
        if ctrl_seq > self._last_ctrl_seq:
            self._last_ctrl_seq = ctrl_seq
            self._last_ctrl_monotonic = now
            self._ctrl_seq = ctrl_seq

        tele = self._tv_wrapper.get_tele_data()

        age = now - self._last_event_monotonic if self._last_event_monotonic > 0 else 1e9
        connected = self._last_event_monotonic > 0 and age < self.disconnect_timeout_s
        if connected and not self._was_connected:
            self._connected = True
            if self._reconnect_cb is not None:
                self._reconnect_cb()
        if (not connected) and self._was_connected:
            self._connected = False
            sustained = age >= self.sustained_timeout_s
            if self._disconnect_cb is not None:
                self._disconnect_cb(sustained)
        self._was_connected = connected
        self._connected = connected

        # Fresh controller traffic required to push goals / buttons.
        # Sticky motion_data_ready alone is NOT enough (avoids stale pose after drop).
        ctrl_fresh = (
            self._last_ctrl_monotonic > 0
            and (now - self._last_ctrl_monotonic) < self.disconnect_timeout_s
        )
        if not tele.motion_data_ready or not ctrl_fresh:
            return

        # Push arm goals whenever TeleVuer reports ready + fresh controller poses.
        if self._dual_pose_cb is not None:
            Tl, Tr = self.pose_mapper.map_dual(
                tele.left_wrist_pose, tele.right_wrist_pose
            )
            self._dual_pose_cb(Tl, Tr, now)

        if self._left_trigger_cb is not None:
            self._left_trigger_cb(
                normalize_televuer_trigger(tele.left_ctrl_triggerValue)
            )
        if self._right_trigger_cb is not None:
            self._right_trigger_cb(
                normalize_televuer_trigger(tele.right_ctrl_triggerValue)
            )

        # Pico/WebXR often sets squeeze=True while squeezeValue stays 0.0.
        # Gating only on the analog value makes grip look completely dead.
        l_sq = _squeeze_analog(tele.left_ctrl_squeeze, tele.left_ctrl_squeezeValue)
        r_sq = _squeeze_analog(tele.right_ctrl_squeeze, tele.right_ctrl_squeezeValue)

        action = self._arbitrator.update(l_sq, r_sq, now=now)
        if action == "shutdown" and self._shutdown_cb is not None:
            self._shutdown_cb()
        elif action == "discard" and self._discard_cb is not None:
            self._discard_cb()
        elif action == "task_toggle" and self._task_toggle_cb is not None:
            self._task_toggle_cb(True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._tv_wrapper is not None:
            try:
                self._tv_wrapper.close()
            except Exception:
                pass
            self._tv_wrapper = None
            self._counting_tv = None
            self._vuer_process = None


class FakeTeleVuerDevice(AbstractDevice):
    """Replay / unit-test device with the same bind API as TeleVuerDevice."""

    def __init__(
        self,
        pose_mapper: TvToOrcaPoseMapper | None = None,
        chord_window_s: float = 0.08,
        disconnect_timeout_s: float = 1.0,
    ):
        self.pose_mapper = pose_mapper or TvToOrcaPoseMapper()
        self.disconnect_timeout_s = float(disconnect_timeout_s)
        self._arbitrator = SqueezeChordArbitrator(chord_window_s=chord_window_s)
        self._dual_pose_cb = None
        self._left_trigger_cb = None
        self._right_trigger_cb = None
        self._task_toggle_cb = None
        self._discard_cb = None
        self._shutdown_cb = None
        self._disconnect_cb = None
        self._reconnect_cb = None
        self._connected = False
        self._was_connected = False
        self._closed = False
        self._event_seq = 0
        self._cam_seq = 0
        self._ctrl_seq = 0
        self._last_event_monotonic = 0.0
        # Pending frame injected by tests / replay
        self._frame = None

    def bind_dual_pose_event(self, callback):
        self._dual_pose_cb = callback

    def bind_left_trigger_event(self, callback):
        self._left_trigger_cb = callback

    def bind_right_trigger_event(self, callback):
        self._right_trigger_cb = callback

    def bind_task_toggle_event(self, callback):
        self._task_toggle_cb = callback

    def bind_discard_event(self, callback):
        self._discard_cb = callback

    def bind_shutdown_event(self, callback):
        self._shutdown_cb = callback

    def bind_disconnect_event(self, callback):
        self._disconnect_cb = callback

    def bind_reconnect_event(self, callback):
        self._reconnect_cb = callback

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def cam_seq(self) -> int:
        return int(self._cam_seq)

    @property
    def ctrl_seq(self) -> int:
        return int(self._ctrl_seq)

    def push_frame(
        self,
        *,
        T_l: np.ndarray | None = None,
        T_r: np.ndarray | None = None,
        left_trigger: float = 10.0,
        right_trigger: float = 10.0,
        left_squeeze: float = 0.0,
        right_squeeze: float = 0.0,
        already_mapped: bool = False,
    ):
        """Queue one frame. Triggers use TeleVuer 10→0 convention unless already normalized."""
        self._frame = {
            "T_l": None if T_l is None else np.asarray(T_l, dtype=np.float64),
            "T_r": None if T_r is None else np.asarray(T_r, dtype=np.float64),
            "left_trigger": float(left_trigger),
            "right_trigger": float(right_trigger),
            "left_squeeze": float(left_squeeze),
            "right_squeeze": float(right_squeeze),
            "already_mapped": bool(already_mapped),
        }
        self._event_seq += 1
        self._ctrl_seq += 1
        self._last_event_monotonic = time.monotonic()

    def update(self):
        if self._closed:
            return
        now = time.monotonic()
        age = (
            now - self._last_event_monotonic
            if self._last_event_monotonic > 0
            else 1e9
        )
        connected = self._last_event_monotonic > 0 and age < self.disconnect_timeout_s
        if connected and not self._was_connected and self._reconnect_cb:
            self._reconnect_cb()
        if (not connected) and self._was_connected and self._disconnect_cb:
            self._disconnect_cb(True)
        self._was_connected = connected
        self._connected = connected

        if self._frame is None:
            return
        fr = self._frame
        self._frame = None

        if fr["T_l"] is not None and fr["T_r"] is not None and self._dual_pose_cb:
            if fr["already_mapped"]:
                Tl, Tr = fr["T_l"], fr["T_r"]
            else:
                Tl, Tr = self.pose_mapper.map_dual(fr["T_l"], fr["T_r"])
            self._dual_pose_cb(Tl, Tr, now)

        if self._left_trigger_cb:
            self._left_trigger_cb(normalize_televuer_trigger(fr["left_trigger"]))
        if self._right_trigger_cb:
            self._right_trigger_cb(normalize_televuer_trigger(fr["right_trigger"]))

        action = self._arbitrator.update(fr["left_squeeze"], fr["right_squeeze"], now=now)
        if action == "shutdown" and self._shutdown_cb:
            self._shutdown_cb()
        elif action == "discard" and self._discard_cb:
            self._discard_cb()
        elif action == "task_toggle" and self._task_toggle_cb:
            self._task_toggle_cb(True)

    def close(self):
        self._closed = True
