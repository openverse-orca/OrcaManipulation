"""G1 OmniPicker LeRobot v2.1 parquet 数据集回放脚本（fallback 版本）。

通过 G1ParquetReplayDevice + manager.run_episode() 完全复用采集脚本的驱动链路，
彻底绕开自定义主循环时 OSC 不跟踪的问题。

回放目标：g1_omnipicker_collection_scripted_button_lerobot.py 生成的数据集
  action shape: (N, 18) float32，B 系（base_body = body_link1）

用法：
  cd src/examples/dataCollection

  # 回放全部集
  python g1_omnipicker_replay_lerobot.py \\
      --dataset_dir /path/to/lerobot_dataset \\
      --task_config example.yaml

  # 仅回放第 1 集
  python g1_omnipicker_replay_lerobot.py \\
      --dataset_dir /path/to/lerobot_dataset \\
      --task_config example.yaml \\
      --episode 1

  # 循环回放（Ctrl+C 退出）
  python g1_omnipicker_replay_lerobot.py \\
      --dataset_dir /path/to/lerobot_dataset \\
      --task_config example.yaml \\
      --loop
"""
import os
import sys
import time
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import numpy as np
from yaml import Loader, load
from typing import override

from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controllers import (
    create_arm_osc_controller,
    create_gripper_2f85_reverse_controller,
)
from controllers.controller_task import TaskStatusController, TaskStatus
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import AbstractDevice
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="G1LeRobotReplay",
    log_file="g1_omnipicker_replay_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="DEBUG",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


# ---------------------------------------------------------------------------
# Parquet 读取 & 反归一化
# ---------------------------------------------------------------------------

def _scan_parquet_files(dataset_dir: str) -> list[str]:
    """扫描 data/chunk-*/episode_*.parquet，按文件名升序返回。"""
    data_dir = os.path.join(dataset_dir, "data")
    files: list[str] = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data 目录不存在: {data_dir}")
    for chunk in sorted(os.listdir(data_dir)):
        chunk_path = os.path.join(data_dir, chunk)
        if not os.path.isdir(chunk_path):
            continue
        for fname in sorted(os.listdir(chunk_path)):
            if fname.startswith("episode_") and fname.endswith(".parquet"):
                files.append(os.path.join(chunk_path, fname))
    return files


def _load_episode(parquet_path: str, agent_conf) -> dict:
    """
    读取 parquet，解析 18 维 action → 反归一化末端位姿 + 电机绝对值。

    返回 dict：
      l_pos_b:   (N, 3) float32，左臂末端 B 系位置
      l_quat_b:  (N, 4) float32，左臂末端 B 系四元数 xyzw
      r_pos_b:   (N, 3) float32
      r_quat_b:  (N, 4) float32
      l_grip:    (N, 2) float32，左夹爪 [inner, outer] 绝对电机值
      r_grip:    (N, 2) float32，右夹爪 [inner, outer] 绝对电机值
      n_frames:  int
    """
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    actions = np.array(table["action"].to_pylist(), dtype=np.float32)  # (N, 18)

    if actions.ndim != 2 or actions.shape[1] < 18:
        raise ValueError(f"action 形状异常: {actions.shape}，期望 (N, >=18)  文件: {parquet_path}")

    # 臂末端（B 系）
    l_pos_b  = actions[:, 0:3]
    l_quat_b = actions[:, 3:7]
    r_pos_b  = actions[:, 7:10]
    r_quat_b = actions[:, 10:14]

    # 夹爪反归一化：val = clip(norm,0,1) * (max-min) + min；G1 量程 (-1, 2)
    def _denorm(norm_col: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip(norm_col, 0.0, 1.0) * (hi - lo) + lo

    l_ranges = agent_conf.gripper_l["actuator_ranges"]  # [(lo, hi), (lo, hi)]
    r_ranges = agent_conf.gripper_r["actuator_ranges"]

    l_inner = _denorm(actions[:, 14], *l_ranges[0])
    l_outer = _denorm(actions[:, 15], *l_ranges[1])
    r_inner = _denorm(actions[:, 16], *r_ranges[0])
    r_outer = _denorm(actions[:, 17], *r_ranges[1])
    l_grip  = np.stack([l_inner, l_outer], axis=1)   # (N, 2)
    r_grip  = np.stack([r_inner, r_outer], axis=1)   # (N, 2)

    return {
        "l_pos_b":  l_pos_b,
        "l_quat_b": l_quat_b,
        "r_pos_b":  r_pos_b,
        "r_quat_b": r_quat_b,
        "l_grip":   l_grip,
        "r_grip":   r_grip,
        "n_frames": len(actions),
    }


# ---------------------------------------------------------------------------
# Fallback Device：用 run_episode() 驱动，复用采集脚本的驱动链路
# ---------------------------------------------------------------------------

class G1ParquetReplayDevice(AbstractDevice):
    """每次 update() 被调用时按 steps_per_frame 节奏推进 parquet 数据帧。

    steps_per_frame: 每帧保持的仿真步数。每 steps_per_frame 次 update() 切换下一帧。
    data: _load_episode() 返回的 dict
    task_status: TaskStatusController（is_controller=False，无速率限制）
    """

    def __init__(
        self,
        l_arm,
        r_arm,
        l_grip: Controller2F85Reverse,
        r_grip: Controller2F85Reverse,
        task_status: TaskStatusController,
        data: dict,
        steps_per_frame: int,
    ):
        super().__init__()
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.l_grip = l_grip
        self.r_grip = r_grip
        self.task_status = task_status
        self.data = data
        self.steps_per_frame = max(1, steps_per_frame)
        self.n_frames = data["n_frames"]
        self._call_count = 0   # 总 update() 调用次数
        self._frame_idx = -1   # 当前正在持续的帧号

    @override
    def update(self):
        call = self._call_count
        self._call_count += 1

        # 计算当前应播放的帧号（每 steps_per_frame 次 update 推进一帧）
        frame = call // self.steps_per_frame
        if frame >= self.n_frames:
            # 所有帧播完，发出 END 信号
            if self.task_status.current_status == TaskStatus.RUNNING:
                self.task_status.update_task_status(True)
            return

        # 第一帧第一次：启动任务
        if call == 0:
            self.task_status.update_task_status(True)  # NOT_STARTED → RUNNING

        # 只在帧号变化时更新目标（避免重复 B→W 变换消耗）
        if frame != self._frame_idx:
            self._frame_idx = frame
            l_pos  = self.data["l_pos_b"][frame]
            l_quat = self.data["l_quat_b"][frame]
            r_pos  = self.data["r_pos_b"][frame]
            r_quat = self.data["r_quat_b"][frame]
            l_g    = self.data["l_grip"][frame]
            r_g    = self.data["r_grip"][frame]

            self.l_arm.update_action_position(l_pos)
            self.l_arm.update_action_axisangle(l_quat)
            self.r_arm.update_action_position(r_pos)
            self.r_arm.update_action_axisangle(r_quat)
            self.l_grip.update_ctrl(l_g)
            self.r_grip.update_ctrl(r_g)

            # 每 10% 帧打印进度
            if frame % max(1, self.n_frames // 10) == 0:
                orca_logger.info(
                    f"  frame={frame:04d}/{self.n_frames}  "
                    f"L=[{l_pos[0]:+.3f},{l_pos[1]:+.3f},{l_pos[2]:+.3f}]  "
                    f"R=[{r_pos[0]:+.3f},{r_pos[1]:+.3f},{r_pos[2]:+.3f}]"
                )


# ---------------------------------------------------------------------------
# 工具：创建控制器（镜像 eval_g1_omnipicker_lerobot.py）
# ---------------------------------------------------------------------------

def _create_arm(env, arm_conf, agent_conf):
    ctrl_names = [env.actuator(n) for n in arm_conf["motors_names"]]
    init_ctrl  = {n: v for n, v in zip(ctrl_names, arm_conf["motors_init_ctrl"])}
    return create_arm_osc_controller(env, arm_conf, agent_conf.base_body, ctrl_names, init_ctrl)


def _create_gripper(env, grip_conf, agent_conf):
    ctrl_names = [env.actuator(n) for n in grip_conf["actuator_names"]]
    init_ctrl  = {n: v for n, v in zip(ctrl_names, grip_conf["init_ctrl"])}
    return create_gripper_2f85_reverse_controller(
        env, grip_conf, agent_conf.base_body, ctrl_names, init_ctrl,
        Controller2F85Reverse.ControllerType.DATA,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="G1 OmniPicker LeRobot v2.1 parquet 数据集回放"
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True,
        help="LeRobot 数据集根目录（含 data/ meta/ 子目录）",
    )
    parser.add_argument(
        "--task_config", type=str, required=True,
        help="场景配置 YAML（与数据集对应的任务配置）",
    )
    parser.add_argument(
        "--episode", type=int, default=None,
        help="仅回放指定集（1-indexed）；缺省则按顺序播完全部集",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="循环回放（播完全部后从头再来；配合 --episode 可单集循环）",
    )
    parser.add_argument(
        "--steps_per_frame", type=int, default=3,
        help=(
            "每个 action 帧保持的 5ms 控制步数（默认 3）。"
            "传 1 最快；传 10 约 1× 实时速度"
        ),
    )
    parser.add_argument(
        "--render_every", type=int, default=5,
        help="每隔多少控制步渲染一次（默认 5）；设为 1 每步都渲染（慢），设为 0 禁用渲染",
    )
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    args = parser.parse_args()

    from conf import g1_omnipicker_conf as agent_conf

    dataset_dir = os.path.abspath(os.path.expanduser(args.dataset_dir))
    orca_logger.info(f"数据集: {dataset_dir}")

    # ── 扫描 parquet 文件 ────────────────────────────────────────────
    all_files = _scan_parquet_files(dataset_dir)
    if not all_files:
        raise FileNotFoundError(f"未找到 parquet 文件: {dataset_dir}/data/")

    if args.episode is not None:
        idx = args.episode - 1
        if not (0 <= idx < len(all_files)):
            raise ValueError(f"--episode {args.episode} 超出范围 (1..{len(all_files)})")
        playlist = [all_files[idx]]
    else:
        playlist = list(all_files)

    orca_logger.info(
        f"共 {len(all_files)} 集，待回放 {len(playlist)} 集  "
        f"steps_per_frame={args.steps_per_frame}  loop={args.loop}"
    )

    # ── 关节初值 ─────────────────────────────────────────────────────
    default_joint_values: dict = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = v

    # ── 场景管理 ─────────────────────────────────────────────────────
    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    # ── DataCollectionManager ──────────────────────────────────────────
    def obs_callback(env) -> dict:
        return {"replay": np.zeros(env.nu, dtype=np.float32)}

    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=obs_callback,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.set_disable_actuator_group([agent_conf.positions_group])
    manager.set_task(EmptyTask(env))

    # ── 跳帧渲染补丁（run_episode 每步都调 env.render，用计数器控制频率）──
    _render_counter = [0]
    _render_every = max(0, args.render_every)
    _orig_render = env.render

    def _patched_render():
        if _render_every == 0:
            return None
        _render_counter[0] += 1
        if _render_counter[0] % _render_every == 0:
            return _orig_render()
        return None

    env.render = _patched_render
    orca_logger.info(f"render_every={_render_every}  steps_per_frame={args.steps_per_frame}")

    # ── 创建控制器 ────────────────────────────────────────────────────
    orca_logger.info("Creating arm OSC controllers")
    l_arm  = _create_arm(env, agent_conf.l_arm, agent_conf)
    r_arm  = _create_arm(env, agent_conf.r_arm, agent_conf)
    orca_logger.info("Creating gripper controllers (2F85Reverse)")
    l_grip = _create_gripper(env, agent_conf.gripper_l, agent_conf)
    r_grip = _create_gripper(env, agent_conf.gripper_r, agent_conf)

    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    # TaskStatusController（is_controller=False：不使用速率限制，精确按帧控制）
    task_status = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    manager.set_task_status_controller(task_status)

    # ── 主回放循环 ────────────────────────────────────────────────────
    orca_logger.info("Starting G1 replay loop (Ctrl+C to stop)")

    try:
        ep_files = list(playlist)
        ep_total = len(ep_files)
        ep_idx   = 0

        while True:
            if manager._shutdown_requested:
                break
            if ep_idx >= len(ep_files):
                if args.loop:
                    ep_files = list(playlist)
                    ep_idx   = 0
                    orca_logger.info("所有集回放完毕，循环重头")
                else:
                    orca_logger.info("所有集回放完毕，退出")
                    break

            parquet_path = ep_files[ep_idx]
            ep_idx += 1

            ep_name = os.path.basename(parquet_path)
            orca_logger.info(f"=== 回放 {ep_name}  ({ep_idx}/{ep_total}) ===")

            # 加载 parquet
            ep_data  = _load_episode(parquet_path, agent_conf)
            n_frames = ep_data["n_frames"]
            steps_per_frame = max(1, args.steps_per_frame)
            orca_logger.info(
                f"  {n_frames} 帧 × {steps_per_frame} 步/帧 = {n_frames * steps_per_frame} 控制步  "
                f"预计仿真时长={n_frames * steps_per_frame * 0.005:.1f}s"
            )

            # 重置场景（对齐采集脚本时序）
            env.reset()
            time.sleep(0.1)
            manager.update_scene()
            # spawn_scene 后再 apply 一次关节初值，确保 controllers.reset() 抓到正确起始位姿
            env.set_default_joint_values(default_joint_values)

            # 创建本集的 replay device 并注入 manager
            device = G1ParquetReplayDevice(
                l_arm=l_arm,
                r_arm=r_arm,
                l_grip=l_grip,
                r_grip=r_grip,
                task_status=task_status,
                data=ep_data,
                steps_per_frame=steps_per_frame,
            )
            manager.set_device(device)

            # 走 run_episode()：与采集脚本完全相同的驱动链路
            task_status.reset()
            manager.run_episode()

            orca_logger.info(f"  episode 播完: {device._call_count} 控制步")

    except Exception as e:
        OrcaLog.get_instance().error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        try:
            env.close()
        except Exception:
            pass
        orca_logger.info("Exiting program")
        os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as e:
        OrcaLog.get_instance().error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        orca_logger.info("Exiting program")
        os._exit(0)
