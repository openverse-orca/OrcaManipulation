"""LeRobot v2.1 parquet 数据集回放脚本（结构对齐 data_collection_replay.py）。

走 manager.run() + Task + DataDevice 事件驱动机制，
数据源从 HDF5(DataDevice) 换成 LeRobot parquet(LeRobotDataDevice)。

与 data_collection_replay.py 的区别：
  - --data_root / --agent_name 换成 --dataset_dir（LeRobot 根目录）
  - --replay_mode 只支持 osc（仅 openloong 双臂 2F85）
  - 用 LeRobotReplayManager 子类重写 update_scene()，绕过 AUGMENTATION 模式
    对 DataDevice 的严格类型检查与 scene_info restore 依赖
  - save_video=False（不录制回放视频）
  - 支持 --episode N 单集回放（缺省播完全部）

运行环境：orcalab_lerobot（含 orca_gym + pyarrow）

用法：
  cd src/examples/dataCollection

  # 回放全部集
  python data_collection_replay_lerobot.py \\
      --dataset_dir ~/lerobot_data/scripted_shop \\
      --task_config scripted-example.yaml

  # 仅回放第 1 集
  python data_collection_replay_lerobot.py \\
      --dataset_dir ~/lerobot_data/scripted_shop \\
      --task_config scripted-example.yaml \\
      --episode 1

  # 循环回放（按 Ctrl+C 退出）
  python data_collection_replay_lerobot.py \\
      --dataset_dir ~/lerobot_data/scripted_shop \\
      --task_config scripted-example.yaml \\
      --loop
"""
import os
import sys
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import numpy as np
from yaml import Loader, load

from controllers import controllers
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.lerobot_data_device import LeRobotDataDevice
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="LeRobotReplay",
    log_file="data_collection_replay_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="DEBUG",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


# ---------------------------------------------------------------------------
# 局部 Manager 子类：重写 update_scene() 绕过 AUGMENTATION 硬约束
# ---------------------------------------------------------------------------

class LeRobotReplayManager(DataCollectionManager):
    """重写 update_scene() 以支持 LeRobotDataDevice。

    原 AUGMENTATION 模式有两处硬约束：
      1. `if type(self.device) != DataDevice: raise ValueError`（严格类型检查）
      2. `update_actor_qpos(restore=True, scene_info=...)`（parquet 无 scene_info）

    本子类用「加载 parquet → 随机/默认场景 → task.get_task()」替换这两处，
    其余 run() / run_episode() / run_controllers() 全部继承复用。
    只访问公共属性，不触碰任何 _ 前缀内部成员。
    """

    def update_scene(self) -> bool:
        if self.scene_manager is None:
            return True

        # 1. 生成场景 Actor
        self.scene_manager.spawn_scene()

        # 2. 从 parquet 加载下一集数据（队列空 → False → run() 退出）
        load_ret = self.device.load_data()
        if not load_ret:
            orca_logger.info("LeRobot 数据集回放完毕，退出")
            return False

        unit_path = self.device.get_current_unit_path()
        if unit_path:
            ep_name = os.path.basename(unit_path)
            orca_logger.info(f"回放 parquet: {ep_name}")

        # 3. 场景随机化/默认位（parquet 无 scene_info，等效于 TELECONTROL 的新集初始化）
        if self.task is not None:
            self.scene_manager.update_actor_qpos()
            self.task.get_task(self.scene_manager)
            orca_logger.info(f"Task description: {self.task.get_task_description()}")

        # 4. 禁用位置执行器（与 DataCollectionManager.update_scene 末尾一致）
        self.env.disable_actuator(self.disable_actuator_group)
        return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeRobot v2.1 parquet 数据集回放（OSC 末端位姿，openloong，走 manager.run()）"
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
        "--level", type=str, default="replay",
        help="场景名称标识（仅用于日志，默认 replay）",
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
        "--steps_per_frame", type=int, default=10,
        help=(
            "每个 30fps 动作帧保持的 5ms 控制步数（默认 10 ≈ 50ms，约 1.5x 实时）。"
            "传 7 接近实时；传 20 最稳但较慢"
        ),
    )
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    args = parser.parse_args()

    from conf import openloong_conf as agent_conf

    dataset_dir = os.path.abspath(os.path.expanduser(args.dataset_dir))
    orca_logger.info(f"数据集: {dataset_dir}")

    # ── 关节初值 ─────────────────────────────────────────────────────
    default_joint_values: dict = {}
    for n, v in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[n] = v
    for n, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[n] = v

    # ── LeRobotDataDevice ────────────────────────────────────────────
    orca_logger.info("Creating LeRobotDataDevice")
    device = LeRobotDataDevice(
        dataset_dir=dataset_dir,
        agent_conf=agent_conf,
        loop_playback=args.loop,
        steps_per_frame=args.steps_per_frame,
    )

    # 记录总集数（select_episode 前，unit_datasets_path 包含全部 episode）
    n_eps = len(device.unit_datasets_path)

    if args.episode is not None:
        device.select_episode(args.episode)

    orca_logger.info(
        f"[Replay] 数据集 episodes={n_eps}  steps_per_frame={args.steps_per_frame}  "
        f"env.dt=5ms → 每帧保持 {args.steps_per_frame * 5}ms  "
        f"(实时帧 dt=33.3ms，当前约 {args.steps_per_frame * 5 / 33.3:.1f}x 实时)"
    )

    # ── 场景管理 ─────────────────────────────────────────────────────
    orca_logger.info("Creating scene manager")
    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    # ── DataCollectionManager（使用局部子类）────────────────────────
    orca_logger.info("Creating LeRobotReplayManager")

    def obs_callback(env) -> dict:
        return {"replay": np.zeros(env.nu, dtype=np.float32)}

    manager = LeRobotReplayManager(
        agent_name="openloong",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=obs_callback,
        env_index=0,
        device=device,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env

    # AUGMENTATION 模式：使 run() 不进入 TELECONTROL 分支的录制逻辑
    # （update_scene 已重写，mode 仅影响 run_episode 里 show_ui_message 的条件判断）
    manager.mode = DataCollectionManager.DataCollectionMode.AUGMENTATION
    manager.save_video = False

    # ── 禁用位置执行器 ───────────────────────────────────────────────
    manager.set_disable_actuator_group([agent_conf.positions_group])

    # ── OSC 双臂控制器（复用现有工厂，不做任何修改）────────────────
    orca_logger.info("Adding arm OSC controllers")
    controllers.add_arm_osc_openloong_data_controller(
        manager, env, agent_conf.l_arm, agent_conf.base_body, device, left_arm=True
    )
    controllers.add_arm_osc_openloong_data_controller(
        manager, env, agent_conf.r_arm, agent_conf.base_body, device, left_arm=False
    )

    # ── 2F85 夹爪控制器 ──────────────────────────────────────────────
    orca_logger.info("Adding gripper controllers")
    controllers.add_gripper_2f85_openloong_data_controller(
        manager, env, agent_conf.gripper_l, agent_conf.base_body, device, left_gripper=True
    )
    controllers.add_gripper_2f85_openloong_data_controller(
        manager, env, agent_conf.gripper_r, agent_conf.base_body, device, left_gripper=False
    )

    # ── TaskStatus 控制器（device 末帧触发 END）──────────────────────
    orca_logger.info("Adding task status controller")
    controllers.add_task_status_openloong_data_controller(
        manager, env, device, agent_conf.base_body
    )

    # EmptyTask：is_success() 恒 True，回放不落数据所以成功判定无实际意义
    manager.set_task(EmptyTask(env))

    # ── 启动回放循环 ─────────────────────────────────────────────────
    orca_logger.info("Starting replay loop (Ctrl+C to stop)")
    manager.run()


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
