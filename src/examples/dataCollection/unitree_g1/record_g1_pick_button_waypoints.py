"""record_g1_pick_button_waypoints.py — 宇树 g1_pick 四色按钮接触关节角录制。

遥操作连接模块与数采 ``g1_pick_collection_tele_lerobot.py`` 对齐：
  - 同一套 ``TeleopConfig`` / ``ControllerWiring.wire``（IK/手/锁腿/pin/闸）
  - 同一套 TeleVuer ``task_toggle`` / ``disconnect→END`` / ``reconnect→rebase``
  - 同一套 ``HealthMonitor``（televuer）
  - 主循环按 ``run_episode`` 会话边界：``tsc.reset`` → 跑 → ``END`` 后开下一轮

仅覆盖数采的两枚业务键（连接逻辑不改）：
  ▶ 左 squeeze           → 数采三态闸 NOT_STARTED↔RUNNING↔END（clutch）
  ★ 左右 squeeze 同按  → 记录 28 维关节角 q（数采此处为退出）
  ↺ 单右 squeeze       → 重置场景并开下一轮（数采此处为丢弃本集）
  Ctrl+C               → 写出 YAML 并退出

录制顺序引导（默认每色 1 点）：红 → 绿 → 黄 → 蓝。
输出 YAML 对齐 OmniPicker ``pose_g1_button_candidates.yaml``，字段为 q（28 维）。

端口（与 XR_TELEVUER.md / 数采相同，勿混）：
  50051 = orcalab 仿真 gRPC
  8012  = TeleVuer 头显网页 + WebSocket（本脚本）
  8001  = 旧 Pico App（--xr_backend pico 才用）
  8010  = 策略评测（本脚本不用）

────────────────────────────────────────────────────────
完整联调次序（USB-local + --tv_no_tls，与数采同款）
────────────────────────────────────────────────────────

### A. 终端 1 — 仿真（先开，保持不关）

```bash
conda activate orcalab_lerobot
orcalab
```

在 OrcaStudio 中：
  - 打开场景 ``unitree_button``（JSON 名 ``unitree_button.json``）
  - 确认 actor = ``unitree_humanoid_robot_1``
  - 仿真处于可连接状态（本机 gRPC ``localhost:50051``）

### B. 终端 2 — USB 预检（强烈建议每次开跑前）

```bash
cd /home/dht/orca_m/OrcaManipulation/src/examples/dataCollection
bash preflight_usb_local.sh --no-tls
# 若 8012 被残留占用：
# bash preflight_usb_local.sh --no-tls --kill-8012
```

全绿再继续。自检也可：

```bash
ss -tnp | grep ':8012' || true
adb reverse --list
adb devices
```

### C. 终端 2 — 启动本录制脚本（先启脚本，再开头显页）

```bash
conda activate orcalab_lerobot
cd /home/dht/orca_m/OrcaManipulation/src/examples/dataCollection

python -u record_g1_pick_button_waypoints.py \\
  --level default \\
  --task_config example.yaml \\
  --scene_json unitree_button.json \\
  --agent_name unitree_humanoid_robot_1 \\
  --orcagym_addr localhost:50051 \\
  --xr_backend televuer \\
  --tv_no_tls \\
  --tv_goal_mode rebased_tv \\
  --tv_ee_dx 0.03 \\
  --tv_max_pos_jump 0.50 \\
  --tv_max_ori_jump 90.0 \\
  --tv_max_dq_step 0.8 \\
  --tv_deadzone_pos 0.006 \\
  --tv_deadzone_ori 2.0 \\
  --tv_goal_ema 0.95 \\
  --arm_kp 150 \\
  --arm_kv_ratio 0.11 \\
  --arm_gravcomp 1.0 \\
  --per_color 1 \\
  --output pose_g1_pick_button_candidates.yaml
```

等到终端打印类似：
  ``TeleVuer visit URL: http://127.0.0.1:8012/``
  以及 ``✓ 遥操作已就绪``

### D. 终端 3（或同机另开）— adb reverse + 打开头显浏览器

```bash
adb reverse tcp:8012 tcp:8012
adb shell am force-stop com.pico.browser   # 可选，清旧页
adb shell am start -a android.intent.action.VIEW \\
  -d 'http://127.0.0.1:8012/'
```

注意：
  - 必须与脚本一致：``--tv_no_tls`` → 用 **http**（不要用 https / 不要裸开 vuer.ai）
  - 只插 USB、不做 reverse → Pico 打不开 127.0.0.1:8012

### E. 头显内操作（连接/开闸与数采相同）

1. 电脑日志出现 ``websocket is connected``
2. 进 XR（ENTER VR / Pass Through），双手拿柄晃动
3. **左 squeeze** → ``[EP] transition …→RUNNING`` + ``[DUAL-IK] clutch``
4. 右臂跟手；接触按钮后 **左右 squeeze 同按** 记 q
5. 再按 **左 squeeze** → END，本轮结束并自动准备下一轮（再左 squeeze 开闸）
6. **右 squeeze** → 重置场景并准备下一轮
7. 顺序：红 → 绿 → 黄 → 蓝；Ctrl+C 写出 YAML

排障可加：``--diag_health``（与数采相同的 ``[HEALTH]`` 心跳）
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from yaml import Loader, load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from conf import g1_pick_conf
from controllers.controller_task import TaskStatus
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.g1_pick_data_storage import G1PickLeRobotStorage, _build_g1pick_q_names
from orca_gym.log.orca_log import get_orca_logger
from scene.scene_manager import SceneManager

import g1_pick_collection_tele_lerobot as tele  # noqa: E402

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

log_dir = os.path.join(base_dir, "logs")
orca_logger = get_orca_logger(
    name="RecordG1PickButton",
    log_file="record_g1_pick_button_waypoints.log",
    max_bytes=5 * 1024 * 1024,
    backup_count=3,
    console_level="INFO",
    file_level="DEBUG",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

_COLOR_ORDER = ["red", "green", "yellow", "blue"]
_COLOR_CN = {
    "red": "红色",
    "green": "绿色",
    "yellow": "黄色",
    "blue": "蓝色",
}
_COLOR_TASK = {
    "red": "按红色按钮",
    "green": "按绿色按钮",
    "yellow": "按黄色按钮",
    "blue": "按蓝色按钮",
}
STATE_DIM = 28


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------

def _fmt_float_list(vals, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in vals) + "]"


def write_candidates_yaml(
    by_color: dict[str, list[dict]],
    output_path: str,
    joint_names: list[str],
    *,
    quiet: bool = False,
) -> None:
    lines: list[str] = []
    lines.append(
        f"# 由 record_g1_pick_button_waypoints.py 自动生成  "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append("# 机器人：宇树 g1_pick（关节角控制）")
    lines.append("# q 顺序与 G1PickLeRobotStorage / observation.state 一致：")
    lines.append("#   [0:7] L臂  [7:14] R臂  [14:21] L手  [21:28] R手")
    lines.append("# 触发：左右 squeeze 同按边沿采样实测 q（接触瞬间）")
    lines.append("# 颜色顺序：红 → 绿 → 黄 → 蓝")
    lines.append("# 遥操作：复用 g1_pick_collection_tele_lerobot.ControllerWiring")
    lines.append("")
    lines.append(f"joint_dim: {len(joint_names)}")
    lines.append("joint_names:")
    for n in joint_names:
        lines.append(f"  - {n}")
    lines.append("")
    lines.append("buttons:")
    lines.append("")

    for color in _COLOR_ORDER:
        cands = by_color.get(color, [])
        lines.append(f"  {color}:")
        lines.append(f'    task: "{_COLOR_TASK[color]}"')
        if not cands:
            lines.append("    candidates: []  # 本色尚未录制")
            lines.append("")
            continue
        lines.append("    candidates:")
        for i, wp in enumerate(cands):
            q = wp["q"]
            ts = wp.get("timestamp", "")
            r = q[7:14]
            lines.append(f"      # {_COLOR_CN[color]}候选 {i + 1}  时间: {ts}")
            lines.append(f"      - q: {_fmt_float_list(q)}")
            lines.append(
                f"        # R_arm≈ pitch={r[0]:.3f} roll={r[1]:.3f} "
                f"yaw={r[2]:.3f} elbow={r[3]:.3f}"
            )
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    n_total = sum(len(by_color.get(c, [])) for c in _COLOR_ORDER)
    orca_logger.info(f"已写出 {n_total} 个接触点 → {output_path}")
    if not quiet:
        print(f"\n[完成] 已写出 {n_total} 个接触点 → {output_path}", flush=True)
        for c in _COLOR_ORDER:
            print(f"  {_COLOR_CN[c]}: {len(by_color.get(c, []))} 点", flush=True)


def _guide_slots(per_color: int) -> list[tuple[str, int]]:
    return [(c, i) for c in _COLOR_ORDER for i in range(1, per_color + 1)]


def _next_prompt(by_color: dict[str, list], per_color: int) -> str | None:
    for color, idx in _guide_slots(per_color):
        if len(by_color.get(color, [])) < idx:
            return (
                f"下一步：{_COLOR_CN[color]}按钮 候选 {idx}/{per_color}  "
                f"（已录 {len(by_color.get(color, []))}/{per_color}）"
            )
    return None


def _neutral_joint_values() -> dict[str, float]:
    out: dict[str, float] = {}
    for conf in (
        g1_pick_conf.l_arm,
        g1_pick_conf.r_arm,
        g1_pick_conf.l_hand,
        g1_pick_conf.r_hand,
    ):
        for jn, v in zip(conf["joint_names"], conf["neutral_joint_values"]):
            out[jn] = float(v)
    return out


def _install_xml_patch_like_tele(env, agent_name: str, arm_gravcomp: float) -> None:
    """与 g1_pick_collection_tele_lerobot 相同的 weld + gravcomp 旁路补丁。"""
    _orig_load = env.gym.load_model_xml
    _gc = float(arm_gravcomp)

    async def _patched_load_model_xml():
        orig_path = await _orig_load()
        with open(orig_path, "r") as f:
            xml = f.read()

        welds_xml = ""
        for name, body in [
            (f"{agent_name}_pelvis_weld", f"{agent_name}_pelvis"),
            (f"{agent_name}_torso_weld", f"{agent_name}_torso_link_rev_1_0"),
            (f"{agent_name}_left_foot_weld", f"{agent_name}_left_ankle_roll_link"),
            (f"{agent_name}_right_foot_weld", f"{agent_name}_right_ankle_roll_link"),
        ]:
            if name not in xml:
                welds_xml += (
                    f'        <weld active="true" name="{name}" body1="{body}" '
                    f'body2="world" solref="0.02 1" solimp="0.9 0.95 0.001"/>\n'
                )
        if welds_xml:
            if "</equality>" in xml:
                xml = xml.replace("</equality>", welds_xml + "</equality>", 1)
            else:
                xml = xml.replace(
                    "</actuator>",
                    "</actuator>\n    <equality>\n" + welds_xml + "    </equality>",
                    1,
                )

        _arm_hand_links = [
            f"{agent_name}_{side}_{link}"
            for side in ("left", "right")
            for link in (
                "shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link",
                "elbow_link",
                "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link",
                "hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link",
                "hand_index_0_link", "hand_index_1_link",
                "hand_middle_0_link", "hand_middle_1_link",
            )
        ]
        if _gc > 0.0:
            for link_name in _arm_hand_links:
                marker = f'name="{link_name}"'
                if marker not in xml:
                    continue
                idx = xml.index(marker)
                body_start = xml.rfind("<body", 0, idx)
                if "gravcomp=" in xml[body_start:idx + len(marker)]:
                    continue
                xml = xml.replace(marker, f'gravcomp="{_gc}" {marker}', 1)

        orig = Path(orig_path)
        patched_path = str(orig.with_stem(orig.stem + "_record_btn_patched"))
        with open(patched_path, "w") as f:
            f.write(xml)
        return patched_path

    env.gym.load_model_xml = _patched_load_model_xml


def _build_tele_cfg(args) -> tele.TeleopConfig:
    """用与数采相同的 TeleopConfig；录制不落盘，lerobot 字段填占位。"""
    ns = SimpleNamespace(
        level=args.level,
        task_config=args.task_config,
        lerobot_out="/tmp/_g1_pick_record_btn_unused",
        repo_id="local/g1_pick_record_btn_unused",
        task="record_button_contact_q",
        fps=20,
        clock="wall",
        resume=False,
        orcagym_addr=args.orcagym_addr,
        cameras="head,wrist_r",
        cam_resolution="480x640",
        camera_source="websocket",
        local_xml=None,
        scene_json=args.scene_json,
        agent_name=args.agent_name,
        log_file=None,
        diag_tele=bool(getattr(args, "diag_tele", False)),
        diag_every=int(getattr(args, "diag_every", 50)),
        xr_backend=args.xr_backend,
        tv_goal_mode=args.tv_goal_mode,
        tv_ee_dx=args.tv_ee_dx,
        tv_position_scale=args.tv_position_scale,
        dry_run_tele=False,
        tv_max_pos_jump=args.tv_max_pos_jump,
        tv_max_ori_jump=args.tv_max_ori_jump,
        tv_max_dq_step=args.tv_max_dq_step,
        tv_deadzone_pos=args.tv_deadzone_pos,
        tv_deadzone_ori=args.tv_deadzone_ori,
        tv_goal_ema=args.tv_goal_ema,
        ik_max_reach=args.ik_max_reach,
        ik_project_reachable=args.ik_project_reachable,
        diag_health=bool(getattr(args, "diag_health", False)),
        diag_joints=bool(getattr(args, "diag_joints", False)),
        diag_joints_hz=float(getattr(args, "diag_joints_hz", 5.0)),
        diag_log_dir=str(getattr(args, "diag_log_dir", "/tmp/g1pick_record_btn_diag")),
        tv_no_tls=args.tv_no_tls,
        arm_kp=args.arm_kp,
        arm_kv=args.arm_kv,
        arm_kv_ratio=args.arm_kv_ratio,
        arm_gravcomp=args.arm_gravcomp,
    )
    return tele.TeleopConfig.from_args(ns)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="宇树 g1_pick 四色按钮接触关节角录制（遥操接线=数采 ControllerWiring）"
    )
    # 与数采对齐的场景 / 地址
    parser.add_argument("--level", default="default")
    parser.add_argument("--task_config", default="../common/example.yaml")
    parser.add_argument(
        "--scene_json",
        default="unitree_button.json",
        help="场景 JSON 提示名（实际以 OrcaStudio 已加载为准；与数采一致）",
    )
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--agent_name",
        default="unitree_humanoid_robot_1",
        help="仿真 actor 名（unitree_button 用 unitree_humanoid_robot_1）",
    )
    # 录制专用
    parser.add_argument(
        "--output",
        default="pose_g1_pick_button_candidates.yaml",
        help="输出候选 YAML（相对脚本目录或绝对路径）",
    )
    parser.add_argument("--per_color", type=int, default=1,
                        help="每色引导候选点数（默认 1：红→绿→黄→蓝各录一点；要多样本再加大）")
    parser.add_argument("--debounce", type=float, default=0.5,
                        help="双 squeeze 防抖秒数（默认 0.5）")
    parser.add_argument(
        "--autosave", action=argparse.BooleanOptionalAction, default=True,
        help="每录一点覆盖写出 YAML（默认开）",
    )
    # XR / TeleVuer（与数采同名同默认）
    parser.add_argument(
        "--xr_backend", choices=("televuer", "pico"), default="televuer",
    )
    parser.add_argument(
        "--tv_no_tls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="TeleVuer 明文 HTTP/WS（默认开，与常用数采命令一致；关：--no-tv_no_tls）",
    )
    parser.add_argument(
        "--tv_goal_mode", choices=("rebased_tv", "absolute_tv"), default="rebased_tv",
    )
    parser.add_argument("--tv_ee_dx", type=float, default=0.03)
    parser.add_argument("--tv_position_scale", type=float, default=1.0)
    parser.add_argument("--tv_max_pos_jump", type=float, default=0.50)
    parser.add_argument("--tv_max_ori_jump", type=float, default=90.0)
    parser.add_argument("--tv_max_dq_step", type=float, default=0.8)
    parser.add_argument("--tv_deadzone_pos", type=float, default=0.006)
    parser.add_argument("--tv_deadzone_ori", type=float, default=2.0)
    parser.add_argument("--tv_goal_ema", type=float, default=0.95)
    parser.add_argument("--ik_max_reach", type=float, default=0.44)
    parser.add_argument("--ik_project_reachable", action="store_true")
    parser.add_argument("--arm_kp", type=float, default=150.0)
    parser.add_argument("--arm_kv", type=float, default=None)
    parser.add_argument("--arm_kv_ratio", type=float, default=0.11)
    parser.add_argument("--arm_gravcomp", type=float, default=1.0)
    # 与数采同名的诊断开关
    parser.add_argument("--diag_tele", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diag_every", type=int, default=50)
    parser.add_argument("--diag_health", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diag_joints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diag_joints_hz", type=float, default=5.0)
    parser.add_argument("--diag_log_dir", default="/tmp/g1pick_record_btn_diag")
    args = parser.parse_args()

    if args.per_color < 1:
        parser.error("--per_color must be >= 1")

    cfg = _build_tele_cfg(args)
    diag = tele.DiagLog(cfg)

    output_path = (
        args.output if os.path.isabs(args.output)
        else os.path.join(base_dir, args.output)
    )
    joint_names = _build_g1pick_q_names()
    assert len(joint_names) == STATE_DIM

    by_color: dict[str, list[dict]] = {c: [] for c in _COLOR_ORDER}
    _record_event = threading.Event()
    _reset_event = threading.Event()
    _shutdown = threading.Event()
    _last_record_t = [0.0]

    default_joint_values = _neutral_joint_values()

    # ── XR 设备（与数采相同构造）──────────────────────────────────────────
    print("=" * 60, flush=True)
    print("  宇树 g1_pick 四色按钮接触关节角录制", flush=True)
    print(f"  场景提示: {cfg.scene_json}  agent: {cfg.agent_name}", flush=True)
    print(f"  XR: {cfg.xr_backend}  tv_no_tls={cfg.tv_no_tls}  mode={cfg.tv_goal_mode}", flush=True)
    print(f"  arm_kp={cfg.arm_kp} gravcomp={cfg.arm_gravcomp} ee_dx={cfg.tv_ee_dx}", flush=True)
    print(f"  输出: {output_path}", flush=True)
    print(f"  每色引导: {args.per_color} 点（红→绿→黄→蓝）", flush=True)
    print("=" * 60, flush=True)

    xr_device = None
    pico_device = None
    if cfg.xr_backend == "pico":
        from orca_gym.devices.pico_joytsick import PicoJoystick
        from devices.g1_pick_device import G1PickPicoJoystickDevice

        print("  等待 Pico 连接...", flush=True)
        pico_device = G1PickPicoJoystickDevice(PicoJoystick())
        xr_device = pico_device
    else:
        from devices.g1_pick_televuer_device import TeleVuerDevice
        from devices.g1_pick_tv_pose_mapper import TvToOrcaPoseMapper, make_trans_x

        print("  初始化 TeleVuer（pass-through，与数采一致）...", flush=True)
        pose_mapper = TvToOrcaPoseMapper(
            T_ee_correction=make_trans_x(cfg.tv_ee_dx),
            position_scale=cfg.tv_position_scale,
        )
        xr_device = TeleVuerDevice(
            pose_mapper=pose_mapper,
            display_mode="pass-through",
            img_shape=(480, 640),
            binocular=False,
            cert_file="" if cfg.tv_no_tls else None,
            key_file="" if cfg.tv_no_tls else None,
            log_buttons=False,
            evt_log=False,
        )
        print(
            f"  TeleVuer visit URL: {xr_device.client_visit_url()} "
            f"(tls={'off' if cfg.tv_no_tls else 'on'})",
            flush=True,
        )
        if cfg.tv_no_tls:
            print("  请在头显打开 http://127.0.0.1:8012/ （需先 adb reverse tcp:8012 tcp:8012）", flush=True)

    # ── 场景 / manager（对齐数采）──────────────────────────────────────────
    with open(os.path.abspath(os.path.join(base_dir, cfg.task_config)), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(cfg.orcagym_addr, config=scene_config)
    scene_manager.show_ui_message(1, "g1_pick 按钮路点录制启动中", "0xffff00", showtime=10)
    scene_manager.get_scene_data(os.path.basename(__file__), "beginscene")

    scratch = os.path.join(base_dir, "_lerobot_scratch", "g1_pick_record_btn", cfg.level)
    storage = G1PickLeRobotStorage(dataset_path=scratch)

    def _obs_callback(env):
        if env.model.nu == 0:
            return {
                "/action/joint/position": np.zeros(14, dtype=np.float32),
                "/action/effector/position": np.zeros(14, dtype=np.float32),
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(14, dtype=np.float32),
                "/action/drive/ctrl": np.zeros(0, dtype=np.float32),
            }
        return storage.obs_callback(env)

    manager = DataCollectionManager(
        agent_name=cfg.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback,
        env_index=0,
        device=xr_device,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=5,
        orcagym_addr=cfg.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False
    manager.mode = DataCollectionManager.DataCollectionMode.TELECONTROL
    _install_xml_patch_like_tele(env, cfg.agent_name, cfg.arm_gravcomp)

    wiring = tele.ControllerWiring()
    dual = None
    monitor = None

    try:
        env.reset()
        time.sleep(0.1)
        if not manager.update_scene():
            orca_logger.error("update_scene 失败（请确认 OrcaLab 已加载 unitree_button 场景）")
            return

        env.set_default_joint_values(default_joint_values)
        tele.limit_cabinet_button_slides(
            env, cfg.agent_name, toward_robot_m=0.0, into_panel_m=0.05
        )
        # ★ 与数采相同：ControllerWiring.wire（含 task_toggle / disconnect / reconnect）
        dual = wiring.wire(cfg, env, manager, xr_device, pico_device, diag)

    except Exception as e:
        orca_logger.error(f"初始化失败: {e}\n{traceback.format_exc()}")
        try:
            env.close()
        except Exception:
            pass
        return

    # ── 仅覆盖业务键；连接/开闸/断连保持 wire() 数采原样 ───────────────────
    def _request_record():
        now = time.perf_counter()
        if (now - _last_record_t[0]) < float(args.debounce):
            return
        _last_record_t[0] = now
        _record_event.set()
        orca_logger.info("[Squeeze] 左右同按 → 记录接触 q（录制）")
        print("[EP] RECORD reason=both_squeeze", flush=True)

    def _request_reset():
        orca_logger.info("[Squeeze] 右 squeeze → 重置场景（录制）")
        print("[EP] RESET reason=right_squeeze", flush=True)
        wiring.discard_event.set()
        _reset_event.set()
        # 不设 manager._shutdown_requested（数采丢弃会结束 episode；此处只重置）

    if cfg.xr_backend == "televuer":
        # wire 已绑：task_toggle / disconnect→END / reconnect→rebase
        # 仅改：shutdown=记q，discard=重置（不退出进程）
        xr_device.bind_shutdown_event(_request_record)
        xr_device.bind_discard_event(_request_reset)
        orca_logger.info(
            "[GATE] televuer=数采连接逻辑；业务键：双squeeze=记q 右squeeze=重置"
        )
        monitor = tele.HealthMonitor(
            cfg, env, xr_device, pico_device, manager, wiring, diag
        )
        monitor.start()
    else:
        # Pico：数采 HealthMonitor 会把双Grip/右Grip 映射成退出；录制改用自建监控
        _mon_stop = threading.Event()

        def _pico_record_monitor():
            both_prev = False
            r_only_count = 0
            r_only_triggered = False
            while not _mon_stop.wait(0.02):
                try:
                    pj = pico_device.pico_joystick
                    raw = pj.current_key_state
                    if not pj.clients or raw is None:
                        both_prev = False
                        r_only_count = 0
                        r_only_triggered = False
                        continue
                    l_grip = bool((raw.get("leftHand") or {}).get("gripButtonPressed", False))
                    r_grip = bool((raw.get("rightHand") or {}).get("gripButtonPressed", False))
                    both = l_grip and r_grip
                    r_only = r_grip and not l_grip
                    if both and not both_prev:
                        _request_record()
                        r_only_count = 0
                        r_only_triggered = True
                    if r_only:
                        if not r_only_triggered:
                            r_only_count += 1
                            if r_only_count >= 8:
                                _request_reset()
                                r_only_triggered = True
                    else:
                        r_only_count = 0
                        if not r_grip:
                            r_only_triggered = False
                    both_prev = both
                except Exception:
                    pass

        threading.Thread(target=_pico_record_monitor, daemon=True).start()

    def _sigint(signum, frame):
        print("\n[退出] Ctrl+C，正在保存 YAML...", flush=True)
        _shutdown.set()
        manager._shutdown_requested = True  # noqa: SLF001

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    print("", flush=True)
    print("=" * 60, flush=True)
    print("  ✓ 遥操作已就绪（连接模块=数采 ControllerWiring + HealthMonitor）", flush=True)
    print("-" * 60, flush=True)
    if cfg.xr_backend == "televuer":
        print("  【连接/开闸 · 与数采相同】", flush=True)
        print("  左臂        锁定", flush=True)
        print("  左 squeeze  NOT_STARTED→RUNNING→END（clutch）", flush=True)
        print("  断连 sustained → END（hold），下一轮再开闸", flush=True)
        print("  右臂跟随    右手柄（RUNNING 后）", flush=True)
        print("  双手        左右扳机", flush=True)
        print("-" * 60, flush=True)
        print("  【录制业务键 · 相对数采改写】", flush=True)
        print("  ★ 双 squeeze  记录接触关节角 q", flush=True)
        print("  ↺ 右 squeeze  重置场景并开下一轮", flush=True)
        ui_msg = "左squeeze=开闸  双squeeze=记q  右squeeze=重置  左squeeze再按=本轮结束"
    else:
        print("  【操作 · Pico】", flush=True)
        print("  左 Grip     开闸（数采 task status）", flush=True)
        print("  ★ 双 Grip   记关节角", flush=True)
        print("  ↺ 右 Grip   重置场景", flush=True)
        ui_msg = "左Grip=开闸  双Grip=记q  右Grip=重置"
    print("  退出保存    Ctrl+C", flush=True)
    print("-" * 60, flush=True)
    print(f"  【顺序】每色 {args.per_color} 点：红 → 绿 → 黄 → 蓝", flush=True)
    for color in _COLOR_ORDER:
        print(f"    {_COLOR_CN[color]} × {args.per_color}  → {_COLOR_TASK[color]}", flush=True)
    if cfg.diag_health:
        print(f"  [HEALTH] 已开启 → {diag.log_path}", flush=True)
    print("=" * 60, flush=True)
    try:
        scene_manager.show_ui_message(1, ui_msg, "0x00ff00", showtime=0)
    except Exception:
        pass

    def _assign_color_for_new_point() -> str:
        for color in _COLOR_ORDER:
            if len(by_color[color]) < args.per_color:
                return color
        return "blue"

    def _handle_record_once() -> None:
        if not _record_event.is_set():
            return
        _record_event.clear()
        try:
            q = storage.build_state(storage.obs_callback(env))
            if q.shape[0] != STATE_DIM:
                raise ValueError(f"q 维异常: {q.shape}")
            color = _assign_color_for_new_point()
            ts = datetime.now().strftime("%H:%M:%S")
            by_color[color].append({
                "q": [float(x) for x in q.tolist()],
                "timestamp": ts,
            })
            idx = len(by_color[color])
            r = q[7:14]
            print(
                f"\n  ✓ 已记录 [{_COLOR_CN[color]} #{idx}] @ {ts}  "
                f"R_arm pitch={r[0]:+.3f} elbow={r[3]:+.3f}",
                flush=True,
            )
            orca_logger.info(
                f"[RECORD] {color}#{idx} q_R_arm={np.round(r, 4).tolist()}"
            )
            try:
                scene_manager.show_ui_message(
                    1, f"已录 {_COLOR_CN[color]} #{idx}", "0x00ff88", showtime=2
                )
            except Exception:
                pass
            if args.autosave:
                write_candidates_yaml(by_color, output_path, joint_names, quiet=True)
            nxt = _next_prompt(by_color, args.per_color)
            if nxt:
                print(f"  → {nxt}\n", flush=True)
            else:
                print(
                    f"  → 引导 {args.per_color * 4} 点已满；"
                    f"可继续多录（追加到蓝色）或 Ctrl+C 结束\n",
                    flush=True,
                )
        except Exception as e:
            orca_logger.error(f"记录失败: {e}\n{traceback.format_exc()}")
            print(f"  ⚠ 记录失败: {e}", flush=True)

    def _reset_scene() -> None:
        print("\n  ↺ 重置场景...", flush=True)
        env.reset()
        time.sleep(0.1)
        if not manager.update_scene():
            raise RuntimeError("update_scene 失败")
        env.set_default_joint_values(default_joint_values)
        tele.limit_cabinet_button_slides(
            env, cfg.agent_name, toward_robot_m=0.0, into_panel_m=0.05
        )

    try:
        session = 0
        while not _shutdown.is_set() and not manager._shutdown_requested:  # noqa: SLF001
            session += 1
            # ── 对齐 run_episode 开头：init ctrl + controller/tsc reset ──
            manager.set_init_ctrl()
            env.set_ctrl(manager.ctrl)
            env.mj_forward()
            for c in manager.controllers:
                c.reset()
            tsc = manager.task_status_controller
            if tsc is not None:
                tsc.reset()

            hint = _next_prompt(by_color, args.per_color)
            print(
                f"\n>>> 第 {session} 轮（左squeeze开闸；双squeeze记q；"
                f"右squeeze重置；左squeeze再按=本轮END）",
                flush=True,
            )
            if hint:
                print(f"  → {hint}", flush=True)

            need_scene_reset = False
            while not _shutdown.is_set() and not manager._shutdown_requested:  # noqa: SLF001
                start_time = time.time()
                action = manager.run_controllers()
                env.step(action)
                env.render()

                _handle_record_once()

                if _reset_event.is_set() or wiring.discard_event.is_set():
                    _reset_event.clear()
                    wiring.discard_event.clear()
                    need_scene_reset = True
                    break

                # 与数采 run_episode 相同：读 task status；END → 结束本轮
                if tsc is not None:
                    task_status = tsc.run_controller()
                    if task_status == TaskStatus.END:
                        print(
                            f"[EP] session={session} END → 准备下一轮，请左 squeeze 开闸",
                            flush=True,
                        )
                        try:
                            scene_manager.show_ui_message(
                                1, "本轮结束，左squeeze开闸", "0xff8800", showtime=2
                            )
                        except Exception:
                            pass
                        break

                elapsed = time.time() - start_time
                if elapsed < manager.real_time_step:
                    time.sleep(manager.real_time_step - elapsed)

            if need_scene_reset:
                try:
                    _reset_scene()
                    print("  ↺ 场景重置完成。", flush=True)
                    try:
                        scene_manager.show_ui_message(
                            1, "已重置，左squeeze开闸", "0xffaa00", showtime=2
                        )
                    except Exception:
                        pass
                except Exception as e:
                    orca_logger.warning(f"重置失败: {e}")
                    print(f"  ⚠ 重置失败: {e}", flush=True)

    except KeyboardInterrupt:
        _shutdown.set()
    finally:
        if monitor is not None:
            try:
                monitor.stop()
            except Exception:
                pass
        write_candidates_yaml(by_color, output_path, joint_names)
        try:
            if cfg.xr_backend == "televuer" and xr_device is not None:
                xr_device.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        try:
            diag.close()
        except Exception:
            pass
        orca_logger.info("Exiting record_g1_pick_button_waypoints")
        os._exit(0)


if __name__ == "__main__":
    main()
