#!/usr/bin/env python3
"""
Pico 手柄信号诊断工具
====================
独立运行，无需 OrcaGym 仿真环境。
打印每一帧变化的按键状态、连接/断开事件，
帮助确认各按键是否正常上报。

用法：
    python pico_diag.py            # 默认端口 8001
    python pico_diag.py --port 8001
"""
import argparse
import signal
import sys
import threading
import time

from orca_gym.devices.pico_joytsick import PicoJoystick


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v > 0.5
    if isinstance(v, str):
        return v.lower() == "true"
    return False


def main():
    parser = argparse.ArgumentParser(description="Pico 手柄信号诊断")
    parser.add_argument("--port", type=int, default=8001, help="PicoJoystick TCP 端口（默认 8001）")
    args = parser.parse_args()

    stop = threading.Event()

    def _on_sigint(_sig, _frame):
        print("\n[退出]", flush=True)
        stop.set()

    signal.signal(signal.SIGINT, _on_sigint)

    print(f"[INFO] 监听 0.0.0.0:{args.port}，等待 Pico 连接 …（Ctrl+C 退出）", flush=True)
    print(f"[INFO] 按各个按键，观察是否有对应的 [CHANGE] 行输出\n", flush=True)

    pico = PicoJoystick(port=args.port)

    # 上一帧状态，用于变化检测
    prev = {}
    prev_n = 0
    last_ptr = None

    KEYS_L = [
        ("L_joystickPressed", lambda h: _as_bool(h.get("joystickPressed"))),
        ("L_gripButton",      lambda h: _as_bool(h.get("gripButtonPressed"))),
        ("L_trigger",         lambda h: round(float(h.get("triggerValue") or 0), 2)),
        ("L_X(primary)",      lambda h: _as_bool(h.get("primaryButtonPressed"))),
        ("L_Y(secondary)",    lambda h: _as_bool(h.get("secondaryButtonPressed"))),
        ("L_joystick_x",      lambda h: round(float((h.get("joystickPosition") or [0,0])[0]), 2)),
        ("L_joystick_y",      lambda h: round(float((h.get("joystickPosition") or [0,0])[1]), 2)),
    ]
    KEYS_R = [
        ("R_joystickPressed", lambda h: _as_bool(h.get("joystickPressed"))),
        ("R_gripButton",      lambda h: _as_bool(h.get("gripButtonPressed"))),
        ("R_trigger",         lambda h: round(float(h.get("triggerValue") or 0), 2)),
        ("R_A(primary)",      lambda h: _as_bool(h.get("primaryButtonPressed"))),
        ("R_B(secondary)",    lambda h: _as_bool(h.get("secondaryButtonPressed"))),
        ("R_joystick_x",      lambda h: round(float((h.get("joystickPosition") or [0,0])[0]), 2)),
        ("R_joystick_y",      lambda h: round(float((h.get("joystickPosition") or [0,0])[1]), 2)),
    ]

    try:
        while not stop.is_set():
            n = len(pico.clients)
            if n != prev_n:
                ts = time.strftime("%H:%M:%S")
                if n > prev_n:
                    print(f"[{ts}] [CONN] Pico 已连接（clients: {prev_n}→{n}）", flush=True)
                else:
                    print(f"[{ts}] [DISC] Pico 断开（clients: {prev_n}→{n}）", flush=True)
                    prev.clear()
                prev_n = n

            if n == 0:
                time.sleep(0.05)
                continue

            # 轮询新帧
            raw = pico.get_key_state()
            if raw is None:
                time.sleep(0.02)
                continue

            # 用 id(raw) 作简单新帧判断（deepcopy 每次返回新对象，比较内容）
            lh = raw.get("leftHand") or {}
            rh = raw.get("rightHand") or {}

            changes = []
            for name, extractor in KEYS_L:
                val = extractor(lh)
                if prev.get(name) != val:
                    changes.append(f"{name}: {prev.get(name, '?')} → {val}")
                    prev[name] = val
            for name, extractor in KEYS_R:
                val = extractor(rh)
                if prev.get(name) != val:
                    changes.append(f"{name}: {prev.get(name, '?')} → {val}")
                    prev[name] = val

            if changes:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] [CHANGE] " + "  |  ".join(changes), flush=True)

            time.sleep(0.02)
    finally:
        pico.close()


if __name__ == "__main__":
    main()
