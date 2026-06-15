"""LeRobot parquet 格式数据集回放脚本。

从 competition_warehouse_v2_50_quat（LeRobot 格式，parquet）读取 action 轨迹，
用 OSC 控制器（与 replay.py 完全相同的控制方式）进行回放。

action 格式（16-dim，分组布局，与 convert_waic_ee_quat_to_lerobot.py 输出一致）：
  [0:3]   l_pos_b     左臂末端位置（机器人基坐标系）
  [3:7]   l_quat_b    左臂末端姿态（xyzw 四元数）
  [7:10]  r_pos_b     右臂末端位置
  [10:14] r_quat_b    右臂末端姿态（xyzw 四元数）
  [14]    l_grip      左夹爪（归一化 [0,1]，回放时 ×255 还原电机值）
  [15]    r_grip      右夹爪（归一化 [0,1]）

用法：
  python replay_lerobot.py --dataset_dir /path/to/out_dataset --episode 01          # 第 1 条
  python replay_lerobot.py --dataset_dir /path/to/out_dataset --episode 05 --sleep  # 按 30fps 节奏回放

  # 复原采集时的 C12C 随机位姿（需采集时加 --record-scene 生成 c12c_poses.json）
  python replay_lerobot.py --dataset_dir /path/to/out_dataset --episode 01 --restore-scene
"""

import argparse
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
from controllers.controllers import create_arm_osc_controller, create_gripper_2f85_controller
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import AbstractDevice
from scene.scene_manager import SceneManager

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
DEFAULT_DATASET_DIR = "~/openpi-orca/lerobot_out/hangzhou2026/competition_warehouse_v2_50_quat"
DATA_PATH_TMPL = "data/chunk-{chunk:03d}/file-{index:03d}.parquet"
CHUNKS_SIZE = 1000


class EEFDevice(AbstractDevice):
    def __init__(self, l_arm=None, r_arm=None, l_grip=None, r_grip=None):
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.l_grip = l_grip
        self.r_grip = r_grip
        self.l_pos_b = None
        self.l_quat_b = None
        self.r_pos_b = None
        self.r_quat_b = None
        self.l_grip_ctrl = None
        self.r_grip_ctrl = None

    def set_target(self, l_pos_b=None, l_quat_b=None, r_pos_b=None, r_quat_b=None,
                   l_grip_ctrl=None, r_grip_ctrl=None):
        if l_pos_b is not None:
            self.l_pos_b = np.asarray(l_pos_b, dtype=np.float32)
        if l_quat_b is not None:
            self.l_quat_b = np.asarray(l_quat_b, dtype=np.float32)
        if r_pos_b is not None:
            self.r_pos_b = np.asarray(r_pos_b, dtype=np.float32)
        if r_quat_b is not None:
            self.r_quat_b = np.asarray(r_quat_b, dtype=np.float32)
        if l_grip_ctrl is not None:
            self.l_grip_ctrl = np.asarray([l_grip_ctrl], dtype=np.float32)
        if r_grip_ctrl is not None:
            self.r_grip_ctrl = np.asarray([r_grip_ctrl], dtype=np.float32)

    def update(self):
        if self.l_arm is not None and self.l_pos_b is not None and self.l_quat_b is not None:
            self.l_arm.update_action_position(self.l_pos_b)
            self.l_arm.update_action_axisangle(self.l_quat_b)
        if self.r_arm is not None and self.r_pos_b is not None and self.r_quat_b is not None:
            self.r_arm.update_action_position(self.r_pos_b)
            self.r_arm.update_action_axisangle(self.r_quat_b)
        if self.l_grip is not None and self.l_grip_ctrl is not None:
            self.l_grip.update_ctrl(self.l_grip_ctrl)
        if self.r_grip is not None and self.r_grip_ctrl is not None:
            self.r_grip.update_ctrl(self.r_grip_ctrl)


def build_default_joint_values(agent_conf):
    d = {}
    for n, v in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        d[n] = v
    for n, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        d[n] = v
    return d


def create_arm(env, agent_conf, arm_conf):
    ctrl_names = [env.actuator(name) for name in arm_conf["motors_names"]]
    init_ctrl = {name: value for name, value in zip(ctrl_names, arm_conf["motors_init_ctrl"])}
    return create_arm_osc_controller(env, arm_conf, agent_conf.base_body, ctrl_names, init_ctrl)


def create_gripper(env, agent_conf, grip_conf):
    ctrl_names = [env.actuator(name) for name in grip_conf["actuator_names"]]
    init_ctrl = {name: value for name, value in zip(ctrl_names, grip_conf["init_ctrl"])}
    return create_gripper_2f85_controller(
        env, grip_conf, agent_conf.base_body, ctrl_names, init_ctrl,
        Controller2F85.ControllerType.DATA,
    )


def load_episode_actions(dataset_dir: str, episode_index: int) -> np.ndarray:
    """读取指定 episode（0-indexed）的 action 数组，返回 (N, 16) float32。"""
    root = os.path.expanduser(dataset_dir)
    chunk = episode_index // CHUNKS_SIZE
    rel = DATA_PATH_TMPL.format(chunk=chunk, index=episode_index)
    parquet_path = os.path.join(root, rel)
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required to read LeRobot parquet files. "
            "Install with: pip install pyarrow"
        )

    t = pq.read_table(parquet_path)
    actions = np.array(t["action"].to_pylist(), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 16:
        raise ValueError(f"Expected action shape (N, 16), got {actions.shape}")
    print(f"Loaded episode {episode_index:06d}: {len(actions)} steps  ({parquet_path})")
    return actions


GRIPPER_MAX = 255.0
# 每个 30fps 动作保持多少个 5ms 控制步，给 OSC 收敛时间。
# 默认 20（≈100ms）；过小会导致快速段跟踪滞后、夹爪提前闭合撞物体。
STEPS_PER_FRAME = 20


def _restore_c12c_pose(env, dataset_dir: str, episode_index: int) -> None:
    """从 <dataset_dir>/c12c_poses.json 复原采集时记录的 C12C 初始位姿。

    采集脚本加 --record-scene 时逐集随机化 C12C 并记录；回放仅 update_scene 到默认位置，
    故此处按 episode_index 复原 C12C 自由关节 qpos，使场景与采集时一致。
    缺失文件/记录时仅告警跳过，不影响回放。
    """
    import json
    path = os.path.join(os.path.expanduser(dataset_dir), "c12c_poses.json")
    if not os.path.exists(path):
        print(f"[WARN] 未找到 C12C 位姿文件: {path}，跳过场景恢复（C12C 使用默认位置）")
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            poses = json.load(f)
    except Exception as e:
        print(f"[WARN] 读取 c12c_poses.json 失败: {e}，跳过场景恢复")
        return
    rec = poses.get(str(episode_index))
    if rec is None:
        print(f"[WARN] c12c_poses.json 无 episode {episode_index} 记录，跳过场景恢复")
        return
    joint = rec["joint"]
    qpos = np.asarray(rec["qpos"], dtype=np.float64)
    env.set_joint_qpos({joint: qpos})
    env.mj_forward()
    print(f"[✓] 已恢复 C12C 初始位姿 (episode {episode_index}, joint={joint}, "
          f"pos={qpos[:3].round(3).tolist()})")


def parse_action(raw: np.ndarray) -> dict:
    # 16 维分组布局：[l_pos(3), l_quat(4), r_pos(3), r_quat(4), l_grip, r_grip]
    # 与 WAIC replay_lerobot_ee_quat.py 一致。原实现误用交错布局（夹爪当成在 7/15），
    # 导致右臂被喂错位垃圾、左夹爪被喂成 r_pos_x → 回放右臂乱动。
    return {
        "l_pos_b":     raw[0:3],
        "l_quat_b":    raw[3:7],
        "r_pos_b":     raw[7:10],
        "r_quat_b":    raw[10:14],
        "l_grip_ctrl": float(np.clip(raw[14], 0.0, 1.0)) * GRIPPER_MAX,
        "r_grip_ctrl": float(np.clip(raw[15], 0.0, 1.0)) * GRIPPER_MAX,
    }


def episode_index_from_arg(episode_str: str, dataset_dir: str) -> int:
    """把 1-indexed 的序号字符串（如 "01"）转为 0-indexed episode_index。"""
    try:
        order = int(episode_str)
    except ValueError as exc:
        raise ValueError(f"--episode must be an integer, got {episode_str!r}") from exc

    # 统计数据集中实际存在的 episode 数量
    root = os.path.expanduser(dataset_dir)
    info_path = os.path.join(root, "meta", "info.json")
    total = None
    if os.path.exists(info_path):
        import json
        with open(info_path) as f:
            total = json.load(f).get("total_episodes")

    if total is not None and (order < 1 or order > total):
        raise ValueError(f"--episode {episode_str} out of range (1..{total})")

    idx = order - 1
    print(f"Episode {order:02d}/{total or '?'} -> episode_{idx:06d}.parquet")
    return idx


def main():
    parser = argparse.ArgumentParser(
        description="Replay LeRobot parquet trajectory via OSC EEF control."
    )
    parser.add_argument("--task_config", type=str, default="competition_warehouse.yaml")
    parser.add_argument(
        "--dataset_dir", type=str, default=DEFAULT_DATASET_DIR,
        help="LeRobot 格式数据集根目录"
    )
    parser.add_argument(
        "--episode", type=str, required=True,
        help="数据集序号（1-indexed），如 01 表示第一条"
    )
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--sleep", action="store_true",
                        help="按 30fps 节奏回放（每步 ~33ms）")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="最多执行步数，默认回放全部")
    parser.add_argument("--steps_per_frame", type=int, default=STEPS_PER_FRAME,
                        help=f"每个 30fps 动作保持的 5ms 控制步数（默认 {STEPS_PER_FRAME}）。"
                             "调大可减小 OSC 跟踪滞后，避免夹爪提前闭合撞物体")
    parser.add_argument("--restore-scene", action="store_true",
                        help="从 c12c_poses.json 复原采集时的 C12C 随机位姿（需采集时 --record-scene）")
    args = parser.parse_args()

    from conf import d12_conf as agent_conf

    episode_index = episode_index_from_arg(args.episode, args.dataset_dir)
    actions = load_episode_actions(args.dataset_dir, episode_index)
    num_steps = len(actions) if args.max_steps is None else min(len(actions), args.max_steps)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)

    manager = DataCollectionManager(
        agent_name="humanoid_industrial_robot_1",
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

    l_arm  = create_arm(env, agent_conf, agent_conf.l_arm)
    r_arm  = create_arm(env, agent_conf, agent_conf.r_arm)
    l_grip = create_gripper(env, agent_conf, agent_conf.gripper_l)
    r_grip = create_gripper(env, agent_conf, agent_conf.gripper_r)
    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    device = EEFDevice(l_arm=l_arm, r_arm=r_arm, l_grip=l_grip, r_grip=r_grip)
    manager.set_device(device)

    # LeRobot 数据集 fps=30，仿真控制步 5ms。每个 30fps 动作保持 steps_per_frame 个控制步，
    # 给 OSC 足够时间收敛到该目标位姿（默认 20≈100ms）。
    SIM_STEP_DT = 0.005
    steps_per_frame = int(args.steps_per_frame)

    try:
        env.reset()
        time.sleep(0.1)
        if not manager.update_scene():
            return

        if args.restore_scene:
            _restore_c12c_pose(env, args.dataset_dir, episode_index)

        manager.set_init_ctrl()
        env.set_ctrl(manager.ctrl)
        env.mj_forward()
        for controller in manager.controllers:
            controller.reset()

        total_sim_steps = num_steps * steps_per_frame
        print(f"Frames: {num_steps}  |  fps=30  |  STEPS_PER_FRAME={steps_per_frame}  "
              f"|  sim_steps={total_sim_steps}  |  est. duration={num_steps/30.0:.1f}s")

        for sim_step in range(total_sim_steps):
            start_time = time.time()
            frame_idx = sim_step // steps_per_frame
            parsed = parse_action(actions[frame_idx])
            device.set_target(**parsed)

            ctrl = manager.run_controllers()
            obs, reward, terminated, truncated, _ = env.step(ctrl)
            env.render()
            if sim_step % steps_per_frame == 0:
                cmd_l = parsed["l_pos_b"]
                cmd_r = parsed["r_pos_b"]
                print(
                    f"frame={frame_idx:04d}/{num_steps}  t={env.data.time:.3f}  "
                    f"cmd_L=[{cmd_l[0]:+.3f},{cmd_l[1]:+.3f},{cmd_l[2]:+.3f}]  "
                    f"cmd_R=[{cmd_r[0]:+.3f},{cmd_r[1]:+.3f},{cmd_r[2]:+.3f}]  "
                    f"grip(L,R)=({parsed['l_grip_ctrl']:.0f},{parsed['r_grip_ctrl']:.0f})  "
                    f"term={terminated} trunc={truncated}"
                )
            if terminated or truncated:
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
