import argparse
import os
import sys
import shutil
import tempfile

import numpy as np


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from devices.data_device import DataDevice
from scene.scene_manager import SceneManager
from task.pick_place_task import PickPlaceTask
from task.abstract_task import EmptyTask
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log.orca_log import get_orca_logger
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers import controllers
from controllers.abstract_controller import AbstractController
from conf import d12_conf as openloong_conf
from yaml import load, Loader
from dataStorage.openloong_data_storage import OpenLoongDataStorage

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "data_replay.log"

orca_logger = get_orca_logger(name="DataReplay",
                              log_file=log_file,
                              max_bytes=10*1024*1024,
                              backup_count=5,
                              console_level="INFO",
                              file_level="DEBUG",
                              log_dir=log_dir,
                              use_colors=True,
                              force_reinit=True)


class JointReplayController(AbstractController):
    '''
    @description: 严格一比一回放控制器。
        采集时 hdf5 记录的是机器人实际状态：action/joint/position 为
        15 个关节 qpos（左臂7 + 右臂7 + 腰1），action/effector/motor 为
        左右夹爪执行器的原始 ctrl 值（0-255）。
        当前 d12 模型手臂只有力矩电机 M_arm_*（无 P_arm_* 位置执行器），
        因此手臂用 robosuite JOINT_POSITION 控制器做关节空间 PD 跟踪
        （惯量解耦 + 重力补偿，即位置执行器的内部实现），目标是记录的
        绝对 qpos；腰部 P_waist 是位置执行器，直接写入记录的腰角；
        手指 l/r_fingers_actuator 直接写入记录的原始 ctrl 值。
        （对比：data_collection_aug.py 走 OSC 逆解，是末端目标轨迹的
        重新执行，接触扰动下会偏离原始动作。）
    '''

    def _create_joint_position_controller(self, env: OrcaGymLocalEnv, arm_config: dict, kp_list: list):
        '''
        @description: 按 create_arm_osc_controller 的方式创建 robosuite
            JOINT_POSITION 控制器（同一套 joint_indexes 构造）。
            注意：须在 super().__init__ 之前调用，故显式传入 env。
            kp 支持按关节设置：该控制器力矩 = M(惯量)×kp×误差 + 重力补偿，
            有效刚度为 kp×M —— 腕部惯量极小，须用高 kp 才能扛住负载。
        '''
        from orca_gym.adapters.robosuite.controllers import controller_config, controller_factory

        arm_joint_names = [env.joint(joint_name) for joint_name in arm_config["joint_names"]]
        qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(arm_joint_names)
        joint_indexes = {
            "joints": arm_joint_names,
            "qpos": qpos_offsets,
            "qvel": qvel_offsets,
        }
        motors_ranges = [[range_[0] for range_ in arm_config["motors_ranges"]],
                         [range_[1] for range_ in arm_config["motors_ranges"]]]

        jnt_config = controller_config.load_config("joint_position")
        jnt_config["sim"] = env.gym
        jnt_config["eef_name"] = env.site(arm_config["ee_site_name"])
        jnt_config["joint_indexes"] = joint_indexes
        jnt_config["actuator_range"] = motors_ranges
        jnt_config["policy_freq"] = 1.0 / env.dt
        jnt_config["ndim"] = len(arm_joint_names)
        jnt_config["kp"] = kp_list
        jnt_config["damping_ratio"] = 1.0
        jnt_config["interpolation"] = None
        return controller_factory(jnt_config["type"], jnt_config)

    def __init__(self, env: OrcaGymLocalEnv, agent_conf, base_body: str, kp: float = 300.0, right_ki: float = 5.0):
        self._kp = kp
        self._arm_motor_names = (
            [env.actuator(name) for name in agent_conf.l_arm["motors_names"]]
            + [env.actuator(name) for name in agent_conf.r_arm["motors_names"]]
        )
        self._n_arm = len(self._arm_motor_names)
        # 均匀 kp：上一版腕部单独拉高至 3000 会引起高频抽动（低控制频率下
        # 高增益失稳），故回退为全关节统一增益。
        kp_list = [kp] * 7
        self._left_controller = self._create_joint_position_controller(env, agent_conf.l_arm, kp_list)
        self._right_controller = self._create_joint_position_controller(env, agent_conf.r_arm, kp_list)

        # 右臂抬升补偿（积分项）：持瓶时腕部因负载下垂，PD 无法消除稳态
        # 误差；对右臂误差积分产生附加力矩顶住负载，无负载时误差≈0、
        # 积分不动，不影响空载段。左臂不加。
        self._dt = float(env.dt)
        self._right_ki = right_ki
        self._right_integ = np.zeros(7, dtype=np.float64)
        self._integ_torque_limit = 2.5  # 每关节附加力矩上限 Nm（电机 ±20 Nm，安全）

        ctrl_name = list(self._arm_motor_names)
        if hasattr(agent_conf, "waist"):
            ctrl_name.append(env.actuator(agent_conf.waist["position_name"]))
        self._gripper_ctrl_names = (
            [env.actuator(name) for name in agent_conf.gripper_2f85_l["actuator_names"]]
            + [env.actuator(name) for name in agent_conf.gripper_2f85_r["actuator_names"]]
        )
        ctrl_name += self._gripper_ctrl_names

        init_ctrl = {name: 0.0 for name in ctrl_name}

        super().__init__(env, ctrl_name, init_ctrl, base_body)

        self._n_gripper = len(self._gripper_ctrl_names)
        # 数据未加载时的保持值：初始关节角 / 夹爪全开
        self._left_targets = np.array(agent_conf.l_arm["neutral_joint_values"], dtype=np.float32)
        self._right_targets = np.array(agent_conf.r_arm["neutral_joint_values"], dtype=np.float32)
        self._waist_target = float(agent_conf.waist.get("neutral_joint_value", 0.0)) if hasattr(agent_conf, "waist") else 0.0
        self._gripper_targets = np.zeros(self._n_gripper, dtype=np.float32)
        # 跟踪误差统计步数
        self._step = 0

    def update_joint_positions(self, values: np.array):
        '''数据事件：一帧 15 维关节 qpos（左臂7 + 右臂7 + 腰1）'''
        values = np.asarray(values, dtype=np.float32).flatten()
        if len(values) != 15:
            return
        self._left_targets = values[0:7]
        self._right_targets = values[7:14]
        self._waist_target = float(values[14])

    def update_gripper_motors(self, values: np.array):
        '''数据事件：一帧 2 维夹爪执行器原始 ctrl 值（左、右）'''
        values = np.asarray(values, dtype=np.float32).flatten()
        if len(values) == self._n_gripper:
            self._gripper_targets = values

    def run_controller(self) -> dict[int, float]:
        # 手臂：JOINT_POSITION 关节空间 PD（含重力补偿），目标为记录的绝对 qpos
        self._left_controller.set_goal(np.zeros(7), set_qpos=self._left_targets)
        left_torques = self._left_controller.run_controller()
        self._right_controller.set_goal(np.zeros(7), set_qpos=self._right_targets)
        right_torques = np.asarray(self._right_controller.run_controller(), dtype=np.float64)

        # 右臂抬升补偿：误差积分 → 附加力矩（带抗饱和限幅）
        r_actual = self._right_controller.joint_pos
        if r_actual is not None:
            r_err = np.asarray(self._right_targets, dtype=np.float64) - np.asarray(r_actual, dtype=np.float64)
            self._right_integ += r_err * self._dt
            comp = np.clip(self._right_ki * self._right_integ, -self._integ_torque_limit, self._integ_torque_limit)
            right_torques = right_torques + comp

        ctrl = list(np.asarray(left_torques, dtype=np.float64)) + list(right_torques)
        # 腰部：P_waist 位置执行器直接写绝对角
        ctrl.append(self._waist_target)
        # 手指：直接写记录的原始 ctrl 值
        ctrl += list(self._gripper_targets)

        # 跟踪误差监控：目标(记录值) vs 当前实际关节角
        self._step += 1
        if self._step == 1 or self._step % 25 == 0:
            l_actual = self._left_controller.joint_pos
            r_actual = self._right_controller.joint_pos
            if l_actual is not None and r_actual is not None:
                l_err = np.abs(np.asarray(self._left_targets, dtype=np.float64) - np.asarray(l_actual, dtype=np.float64))
                r_err = np.abs(np.asarray(self._right_targets, dtype=np.float64) - np.asarray(r_actual, dtype=np.float64))
                orca_logger.info(
                    f"[跟踪] step={self._step} "
                    f"左臂 max_err={l_err.max():.3f}rad (关节{int(l_err.argmax())}) "
                    f"右臂 max_err={r_err.max():.3f}rad (关节{int(r_err.argmax())})"
                )

        return {self.ctrl_index[i]: float(ctrl[i]) for i in range(len(self.ctrl_index))}


def parse_args():
    parser = argparse.ArgumentParser(description="一比一回放已采集的遥操作数据（不做增强、不落盘）")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(base_dir, "dataset", "d12"),
        help="数据集目录（其下每个 uuid 子目录为一条 episode），默认 dataset/d12",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="只回放指定的一条 episode（uuid 目录名），不指定则按顺序回放全部",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="回放完最后一条后退出（默认循环回放）",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=300.0,
        help="手臂关节 PD 刚度增益（统一增益，过高会抽动），默认 300",
    )
    parser.add_argument(
        "--right-ki",
        type=float,
        default=5.0,
        help="右臂抬升补偿积分增益（持瓶下垂改善不明显加大此值，出现慢振荡减小），默认 5",
    )
    return parser.parse_args()


def build_replay_dir(dataset_path: str, episode: str | None) -> str:
    '''
    @description: 确定回放目录。指定单条 episode 时，创建临时目录并软链该
        episode（DataDevice 把 dataset_path 下的子目录作为回放队列）。
    '''
    if episode is None:
        return dataset_path
    episode_path = os.path.join(dataset_path, episode)
    if not os.path.isdir(episode_path):
        episodes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        raise FileNotFoundError(
            f"episode '{episode}' 不存在于 {dataset_path}。可用 episode: {episodes}"
        )
    replay_dir = os.path.join(tempfile.gettempdir(), f"orca_replay_{episode}")
    if os.path.exists(replay_dir):
        shutil.rmtree(replay_dir)
    os.makedirs(replay_dir)
    os.symlink(episode_path, os.path.join(replay_dir, episode))
    orca_logger.info(f"单条回放: {episode_path}")
    return replay_dir


def main():
    args = parse_args()
    orca_logger.info(f"log file: {log_file}")
    orca_logger.info(f"log dir: {log_dir}")

    orcagym_addr = "localhost:50051"
    env_name = "DataCollection"
    env_index = 0
    # 须与采集时使用的 agent_name 一致（参见 data_collection_tele.py）
    agent_name = "d12_waist_motor_usda"
    default_joint_values = {}

    for joint_name, value in zip(openloong_conf.l_arm["joint_names"], openloong_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(openloong_conf.r_arm["joint_names"], openloong_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    if hasattr(openloong_conf, "waist"):
        default_joint_values[openloong_conf.waist["joint_name"]] = openloong_conf.waist.get("neutral_joint_value", 0.0)

    replay_dir = build_replay_dir(args.dataset, args.episode)
    episodes = sorted(os.listdir(replay_dir)) if os.path.isdir(replay_dir) else []
    orca_logger.info(f"回放目录: {replay_dir}（共 {len(episodes)} 条 episode）")

    # 一比一回放：不使用插值器（那是数据增强的用途）
    data_device = DataDevice(replay_dir, "record/proprio_stats.hdf5", interpolator=None, loop_playback=not args.no_loop)

    # 须与 data_collection_tele.py 使用的 yaml 一致（避免 example.yaml 内 spawnable 在引擎侧不存在）
    with open(os.path.join(base_dir, "conveyor_collect.yaml"), "r") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(orcagym_addr, config=config)
    conveyor_config = config.get("conveyor")

    # 纯回放不落盘，不创建 data_storage；obs_callback 只复用观测格式。
    data_storage = None
    obs_callback = lambda env: OpenLoongDataStorage.obs_callback(None, env)

    data_collection_manager = DataCollectionManager(
        agent_name=agent_name,
        env_name=env_name,
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=obs_callback,
        env_index=env_index,
        device=data_device,
        scene_manager=scene_manager,
        data_storage=data_storage,
        conveyor=conveyor_config,
    )
    env = data_collection_manager.env
    env.reset()

    data_collection_manager.mode = DataCollectionManager.DataCollectionMode.AUGMENTATION

    # 与 aug 保持一致：禁用位置执行器组（本模型中该组无手臂执行器，
    # 腰部 P_waist 与手指执行器不受影响）。
    orca_logger.info("Disabling position controller")
    data_collection_manager.set_disable_actuator_group([openloong_conf.positions_group])

    orca_logger.info(f"Creating joint replay controller（记录 qpos 一比一驱动, kp={args.kp}, 右臂补偿ki={args.right_ki}）")
    replay_controller = JointReplayController(env, openloong_conf, openloong_conf.base_body, kp=args.kp, right_ki=args.right_ki)

    # 关节：15 维 = 左臂7 + 右臂7 + 腰1；夹爪：2 维 = 左右执行器原始 ctrl
    data_device.bind_dataset_event("/action/joint/position", (0, 15), replay_controller.update_joint_positions)
    data_device.bind_dataset_event("/action/effector/motor", (0, 2), replay_controller.update_gripper_motors)

    data_collection_manager.add_controller(replay_controller)

    # 须与采集任务一致：collect_only 等模式下 EmptyTask 写入的 task_info 为空
    orca_logger.info("Creating task")
    if config.get("type") in ["collect_only", "collection", "manual_record"]:
        data_collection_manager.set_task(EmptyTask(env))
    else:
        data_collection_manager.set_task(PickPlaceTask(env))
    controllers.add_task_status_openloong_data_controller(data_collection_manager, env, data_device, openloong_conf.base_body)

    data_collection_manager.save_video = False

    orca_logger.info("开始一比一回放（Ctrl+C 退出）")
    data_collection_manager.run()


if __name__ == "__main__":
    main()
