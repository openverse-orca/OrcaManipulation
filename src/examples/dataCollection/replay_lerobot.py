"""LeRobot parquet 格式数据集回放脚本。

从 --dataset_dir 指定的 LeRobot 数据集中读取 action 轨迹（parquet），
用 OSC 控制器（末端位姿 B 系四元数 + 夹爪）进行回放验证。

action 布局（由 data_collection_tele_lerobot.py 输出，按 agent_name 不同维度不同）：
  openloong (16 维)：
    [0:3]   l_pos_b        左臂末端位置（B 系）
    [3:7]   l_quat_b       左臂末端四元数 xyzw
    [7:10]  r_pos_b        右臂末端位置
    [10:14] r_quat_b       右臂末端四元数 xyzw
    [14]    l_gripper      左夹爪 [0,1]（回放时还原电机量程）
    [15]    r_gripper      右夹爪 [0,1]

  tiangong2 (38 维)：
    [0:3]   l_pos_b
    [3:7]   l_quat_b
    [7:10]  r_pos_b
    [10:14] r_quat_b
    [14:26] l_hand_norm    左灵巧手 12 个 actuator（各自归一化 [0,1]）
    [26:38] r_hand_norm    右灵巧手 12 个 actuator

运行环境：orcalab_lerobot

用法：
  cd src/examples/dataCollection
  # 回放第 1 集（1-indexed）
  python replay_lerobot.py --dataset_dir /path/to/out --episode 01 \\
      --agent_name openloong --task_config example.yaml

  # 按 30fps 节奏回放
  python replay_lerobot.py --dataset_dir /path/to/out --episode 01 \\
      --agent_name openloong --task_config example.yaml --sleep

  # 复原场景（需采集时记录了 scene_info，此处预留接口）
  # python replay_lerobot.py ... --restore_scene
"""
import argparse
import json
import os
import sys
import time
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from yaml import Loader, load

from controllers.controller_2f85 import Controller2F85
from controllers.controllers import (
    create_arm_osc_controller,
    create_gripper_2f85_controller,
)
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import AbstractDevice
from scene.scene_manager import SceneManager

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
CHUNKS_SIZE = 1000
DATA_PATH_TMPL = "data/chunk-{chunk:03d}/episode_{index:06d}.parquet"

# 每个 30fps action 保持多少个 5ms 控制步（给 OSC 收敛时间）
DEFAULT_STEPS_PER_FRAME = 20
SIM_STEP_DT = 0.005  # frame_skip=5, time_step=0.001


# ---------------------------------------------------------------------------
# EEFDevice：把解析后的末端位姿推给 OSC 臂控制器
# ---------------------------------------------------------------------------

class EEFDevice(AbstractDevice):
    """将末端位姿目标直接推给 OSC 控制器的轻量设备。"""

    def __init__(self, l_arm=None, r_arm=None, l_grip=None, r_grip=None):
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.l_grip = l_grip
        self.r_grip = r_grip
        self._l_pos = None
        self._l_quat = None
        self._r_pos = None
        self._r_quat = None
        self._l_grip_ctrl = None
        self._r_grip_ctrl = None

    def set_target(
        self,
        l_pos_b=None,
        l_quat_b=None,
        r_pos_b=None,
        r_quat_b=None,
        l_grip_ctrl=None,
        r_grip_ctrl=None,
    ) -> None:
        if l_pos_b is not None:
            self._l_pos = np.asarray(l_pos_b, dtype=np.float32)
        if l_quat_b is not None:
            self._l_quat = np.asarray(l_quat_b, dtype=np.float32)
        if r_pos_b is not None:
            self._r_pos = np.asarray(r_pos_b, dtype=np.float32)
        if r_quat_b is not None:
            self._r_quat = np.asarray(r_quat_b, dtype=np.float32)
        if l_grip_ctrl is not None:
            self._l_grip_ctrl = np.asarray(l_grip_ctrl, dtype=np.float32).flatten()
        if r_grip_ctrl is not None:
            self._r_grip_ctrl = np.asarray(r_grip_ctrl, dtype=np.float32).flatten()

    def update(self) -> None:
        if self.l_arm is not None and self._l_pos is not None and self._l_quat is not None:
            self.l_arm.update_action_position(self._l_pos)
            self.l_arm.update_action_axisangle(self._l_quat)
        if self.r_arm is not None and self._r_pos is not None and self._r_quat is not None:
            self.r_arm.update_action_position(self._r_pos)
            self.r_arm.update_action_axisangle(self._r_quat)
        if self.l_grip is not None and self._l_grip_ctrl is not None:
            self.l_grip.update_ctrl(self._l_grip_ctrl)
        if self.r_grip is not None and self._r_grip_ctrl is not None:
            self.r_grip.update_ctrl(self._r_grip_ctrl)


# ---------------------------------------------------------------------------
# 数据集读取
# ---------------------------------------------------------------------------

def load_episode_actions(dataset_dir: str, episode_index: int) -> np.ndarray:
    """读取指定 episode（0-indexed）的 action 数组，返回 (N, state_dim) float32。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "需要 pyarrow 读取 LeRobot parquet 文件。"
            "请在 orcalab_lerobot 环境中运行。"
        )
    root = os.path.expanduser(dataset_dir)
    chunk = episode_index // CHUNKS_SIZE
    path = os.path.join(root, DATA_PATH_TMPL.format(chunk=chunk, index=episode_index))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet 文件不存在: {path}")
    table = pq.read_table(path)
    actions = np.array(table["action"].to_pylist(), dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"action 形状异常: {actions.shape}")
    print(
        f"已加载 episode {episode_index:06d}: {len(actions)} 步  ({path})"
    )
    return actions


def episode_index_from_arg(episode_str: str, dataset_dir: str) -> int:
    """把 1-indexed 序号字符串（如 '01'）转为 0-indexed episode_index。"""
    try:
        order = int(episode_str)
    except ValueError as e:
        raise ValueError(
            f"--episode 必须是整数，收到 {episode_str!r}"
        ) from e
    root = os.path.expanduser(dataset_dir)
    info_path = os.path.join(root, "meta", "info.json")
    total = None
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            total = json.load(f).get("total_episodes")
    if total is not None and not (1 <= order <= total):
        raise ValueError(f"--episode {episode_str} 超出范围 (1..{total})")
    idx = order - 1
    print(f"Episode {order:02d}/{total or '?'} -> episode_{idx:06d}.parquet")
    return idx


# ---------------------------------------------------------------------------
# action 解析（按 agent_name 取 conf 得到各段维度）
# ---------------------------------------------------------------------------

def _get_gripper_max(conf) -> tuple[np.ndarray, np.ndarray]:
    """从 conf 取左右夹爪/手各 actuator 量程最大值。"""
    l_max = np.array(
        [r[1] for r in conf.gripper_l["actuator_ranges"]], dtype=np.float32
    )
    r_max = np.array(
        [r[1] for r in conf.gripper_r["actuator_ranges"]], dtype=np.float32
    )
    return l_max, r_max


def parse_action(raw: np.ndarray, l_grip_max: np.ndarray, r_grip_max: np.ndarray) -> dict:
    """将 action 向量拆分为末端位姿 + 夹爪控制量（还原量程）。

    布局约定（与 lerobot_data_storage.build_state 对称）：
        [0:3]  l_pos_b
        [3:7]  l_quat_b  (xyzw)
        [7:10] r_pos_b
        [10:14] r_quat_b (xyzw)
        [14:14+n_l] l_grip_norm   n_l = len(gripper_l.actuator_ranges)
        [14+n_l:]   r_grip_norm   n_r = len(gripper_r.actuator_ranges)
    """
    n_l = len(l_grip_max)
    l_grip_ctrl = np.clip(raw[14:14 + n_l], 0.0, 1.0) * l_grip_max
    r_grip_ctrl = np.clip(raw[14 + n_l:], 0.0, 1.0) * r_grip_max
    return {
        "l_pos_b": raw[0:3],
        "l_quat_b": raw[3:7],
        "r_pos_b": raw[7:10],
        "r_quat_b": raw[10:14],
        "l_grip_ctrl": l_grip_ctrl,
        "r_grip_ctrl": r_grip_ctrl,
    }


# ---------------------------------------------------------------------------
# env / controller 构建
# ---------------------------------------------------------------------------

def build_default_joint_values(conf) -> dict:
    d = {}
    for n, v in zip(conf.l_arm["joint_names"], conf.l_arm["neutral_joint_values"]):
        d[n] = v
    for n, v in zip(conf.r_arm["joint_names"], conf.r_arm["neutral_joint_values"]):
        d[n] = v
    return d


def create_arm(env, conf, arm_conf):
    ctrl_names = [env.actuator(n) for n in arm_conf["motors_names"]]
    init_ctrl = {n: v for n, v in zip(ctrl_names, arm_conf["motors_init_ctrl"])}
    return create_arm_osc_controller(env, arm_conf, conf.base_body, ctrl_names, init_ctrl)


def create_gripper(env, conf, grip_conf):
    ctrl_names = [env.actuator(n) for n in grip_conf["actuator_names"]]
    init_ctrl = {n: v for n, v in zip(ctrl_names, grip_conf["init_ctrl"])}
    return create_gripper_2f85_controller(
        env, grip_conf, conf.base_body, ctrl_names, init_ctrl,
        Controller2F85.ControllerType.DATA,
    )


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeRobot parquet 数据集回放（OSC 末端位姿控制）"
    )
    parser.add_argument(
        "--agent_name", required=True, choices=["openloong", "tiangong2"],
        help="机器人型号（决定 state 布局与控制器配置）"
    )
    parser.add_argument("--dataset_dir", required=True, help="LeRobot 数据集根目录")
    parser.add_argument(
        "--episode", required=True,
        help="回放的集序号（1-indexed），如 01 表示第一集"
    )
    parser.add_argument("--task_config", default="example.yaml", help="场景配置 YAML")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--sleep", action="store_true",
        help="按 30fps 节奏回放（每步 sleep 至 SIM_STEP_DT）"
    )
    parser.add_argument("--max_steps", type=int, default=None, help="最多执行步数")
    parser.add_argument(
        "--steps_per_frame", type=int, default=DEFAULT_STEPS_PER_FRAME,
        help=f"每个 30fps action 保持的 5ms 控制步数（默认 {DEFAULT_STEPS_PER_FRAME}）"
    )
    args = parser.parse_args()

    # ── agent conf ──────────────────────────────────────────────────
    if args.agent_name == "openloong":
        from conf import openloong_conf as agent_conf
    else:
        from conf import tiangong2_conf as agent_conf

    l_grip_max, r_grip_max = _get_gripper_max(agent_conf)

    # ── 数据加载 ────────────────────────────────────────────────────
    episode_index = episode_index_from_arg(args.episode, args.dataset_dir)
    actions = load_episode_actions(args.dataset_dir, episode_index)
    steps_per_frame = int(args.steps_per_frame)
    num_frames = len(actions) if args.max_steps is None else min(
        len(actions), args.max_steps
    )

    base_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    # ── env / manager ───────────────────────────────────────────────
    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=build_default_joint_values(agent_conf),
        obs_callback=lambda env: {"time": np.array([env.data.time], dtype=np.float32)},
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.set_disable_actuator_group([agent_conf.positions_group])

    # ── 控制器 ──────────────────────────────────────────────────────
    l_arm = create_arm(env, agent_conf, agent_conf.l_arm)
    r_arm = create_arm(env, agent_conf, agent_conf.r_arm)
    l_grip = create_gripper(env, agent_conf, agent_conf.gripper_l)
    r_grip = create_gripper(env, agent_conf, agent_conf.gripper_r)
    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    device = EEFDevice(l_arm=l_arm, r_arm=r_arm, l_grip=l_grip, r_grip=r_grip)
    manager.set_device(device)

    total_sim_steps = num_frames * steps_per_frame
    print(
        f"帧数: {num_frames}  |  fps=30  |  steps_per_frame={steps_per_frame}  "
        f"|  总仿真步数: {total_sim_steps}  |  预计时长: {num_frames / 30.0:.1f}s"
    )

    try:
        env.reset()
        time.sleep(0.1)
        if not manager.update_scene():
            print("update_scene 失败，退出")
            return

        manager.set_init_ctrl()
        env.set_ctrl(manager.ctrl)
        env.mj_forward()
        for ctrl in manager.controllers:
            ctrl.reset()

        for sim_step in range(total_sim_steps):
            start_time = time.time()
            frame_idx = sim_step // steps_per_frame
            parsed = parse_action(actions[frame_idx], l_grip_max, r_grip_max)
            device.set_target(**parsed)

            ctrl_out = manager.run_controllers()
            obs, _reward, terminated, truncated, _ = env.step(ctrl_out)
            env.render()

            if sim_step % steps_per_frame == 0:
                cmd_l = parsed["l_pos_b"]
                cmd_r = parsed["r_pos_b"]
                print(
                    f"frame={frame_idx:04d}/{num_frames}  "
                    f"t={env.data.time:.3f}  "
                    f"cmd_L=[{cmd_l[0]:+.3f},{cmd_l[1]:+.3f},{cmd_l[2]:+.3f}]  "
                    f"cmd_R=[{cmd_r[0]:+.3f},{cmd_r[1]:+.3f},{cmd_r[2]:+.3f}]  "
                    f"term={terminated} trunc={truncated}"
                )

            if terminated or truncated:
                print("仿真提前终止")
                break

            if args.sleep:
                remain = SIM_STEP_DT - (time.time() - start_time)
                if remain > 0:
                    time.sleep(remain)

    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Unexpected error: {e}\n{traceback.format_exc()}")
