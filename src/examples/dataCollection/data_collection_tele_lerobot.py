"""VR 遥操作采集 → LeRobot v3.0 格式（parquet + 视频）。

与 data_collection_tele.py 的区别：
  - data_storage 换成 OpenLoongLeRobotStorage / Tiangong2LeRobotStorage
  - 相机走 OrcaGym 客户端 PyAV remux 录制（``setup_cameras`` 触发推流）
  - manager.save_video=False（不用 OrcaStudio 服务端录像）
  - collection_data() 流式写帧：降频门控 + ``SingleFrameTask`` 逐帧回调，
    回调内调用 ``manager.decode_frame_at`` 解码真实图像，直接 ``add_frame``
    传给 LeRobot（流式编码 ``streaming_encoding=True``，边采集边编码）
  - 数据直接写入 LeRobot v3.0 parquet 格式，可直接用于 lerobot 训练

架构设计：
  使用 manager.run() + EpisodeLifecycleCallback 回调机制，与 data_collection_tele.py
  保持一致的框架调用方式。LeRobot 特有的初始化和清理逻辑通过回调注入：

    on_run_start: setup_cameras 启动推流 → configure_lerobot 注入依赖 → with writer
    on_episode_start: 计数
    on_episode_end: 帧数统计
    on_run_end: writer.__exit__（在 env.close() 之前执行）

相机命名映射：
  env 相机名（cameras_conf 的 key）→ LeRobot feature key 后缀。
  例：``camera_head`` → ``cam_head``，``observation.images.cam_head``。

退出清理顺序（由 on_run_end 回调保证，在 manager.run() 的 env.close() 之前执行）：
    writer.__exit__()  → stop_image_writer + VEM 清理
    env.close()        → manager.run() finally 执行（关闭 viewer + 录制器 + gRPC）

运行环境：需安装 orca_gym 26.6.x + lerobot 0.3.x + av + pyarrow

用法：
  cd src/examples/dataCollection
  python data_collection_tele_lerobot.py \\
      --level tele \\
      --agent_name openloong \\
      --task_config example.yaml \\
      --lerobot_out /path/to/out_dataset \\
      --repo_id your_org/my_dataset \\
      --fps 20
"""
import argparse
import os
import sys
import time
import traceback

# conda run / 管道环境下 stdout/stderr 默认全缓冲，导致终端长时间看不到日志。
# 在脚本最早处强制行缓冲，确保每行输出立即刷到终端。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.append(os.path.join("/home/orca/Projects", "OrcaGym"))

import numpy as np
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from yaml import Loader, load

from controllers import controllers
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_data_storage import LeRobotDatasetWriter
from devices.abstract_device import PicoJoystickDevice
from scene.scene_manager import SceneManager
from task.pick_place_task import PickPlaceTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="TeleLerobot",
    log_file="data_collection_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


def _load_agent_conf(agent_name: str):
    """根据 agent_name 加载对应的机器人配置。"""
    if agent_name == "openloong":
        from conf import openloong_conf
        return openloong_conf
    elif agent_name == "tiangong2":
        from conf import tiangong2_conf
        return tiangong2_conf
    raise ValueError(f"不支持的 agent_name: {agent_name!r}（支持 openloong / tiangong2）")


def _make_storage(agent_name: str, scratch_dir: str):
    """根据 agent_name 创建对应的 LeRobot storage 实例。

    具体子类已分离到独立模块：
        - OpenLoongLeRobotStorage  → lerobot_openloong_storage.py
        - Tiangong2LeRobotStorage → lerobot_tiangong_storage.py
    """
    if agent_name == "openloong":
        from dataStorage.lerobot_openloong_storage import OpenLoongLeRobotStorage
        return OpenLoongLeRobotStorage(dataset_path=scratch_dir)
    elif agent_name == "tiangong2":
        from dataStorage.lerobot_tiangong_storage import Tiangong2LeRobotStorage
        return Tiangong2LeRobotStorage(dataset_path=scratch_dir)
    raise ValueError(f"不支持的 agent_name: {agent_name!r}")


def _filter_cameras_by_name(cameras_conf: dict, enabled: set[str] | None = None) -> dict:
    """按相机短名过滤 cameras_conf。

    ``enabled`` 中的短名（如 ``"head"``）匹配 ``camera_head``；
    也接受完整名（如 ``"camera_head"``）。为 None 时返回全部。
    """
    if enabled is None:
        return dict(cameras_conf)
    result: dict = {}
    for env_cam, props in cameras_conf.items():
        short = env_cam[len("camera_"):] if env_cam.startswith("camera_") else env_cam
        if short in enabled or env_cam in enabled:
            result[env_cam] = props
    return result


class LeRobotEpisodeCallback:
    """LeRobot 采集的 episode 生命周期回调。

    通过 EpisodeLifecycleCallback 协议注入 DataCollectionManager，
    实现 LeRobot 特有的初始化和有序清理逻辑，无需修改 manager.run() 核心流程。

    生命周期调用顺序：
        on_run_start:  setup_cameras 启动推流 → configure_lerobot 注入依赖 → with writer
        on_episode_start: episode 计数
        on_episode_end:   帧数统计
        on_run_end:       writer.__exit__（在 manager.run() 的 env.close() 之前执行）
    """

    def __init__(
        self,
        manager: DataCollectionManager,
        storage,
        cameras_conf: dict,
        fps: int,
        task: str,
        lerobot_out: str,
        repo_id: str,
        resume: bool,
        agent_name: str,
    ) -> None:
        self._manager = manager
        self._storage = storage
        self._cameras_conf = cameras_conf
        self._fps = fps
        self._task = task
        self._lerobot_out = lerobot_out
        self._repo_id = repo_id
        self._resume = resume
        self._agent_name = agent_name

        self._writer: LeRobotDatasetWriter | None = None
        self._writer_entered: bool = False
        self._ep_idx = 0

    def on_run_start(self) -> None:
        """run() 主循环启动前：启动相机推流、创建 writer、配置 storage。

        异常安全：任何步骤失败都会设置 ``_writer_entered=False``，确保
        ``on_run_end`` 不会对未 ``__enter__`` 的 writer 调用 ``__exit__``。
        """
        try:
            self._storage.setup_cameras(self._manager.env, self._cameras_conf, show_viewer=False)
            self._writer = LeRobotDatasetWriter.create(
                repo_id=self._repo_id,
                root=self._lerobot_out,
                fps=self._fps,
                cameras_conf=self._cameras_conf,
                state_dim=self._storage.state_dim,
                state_names=self._storage.state_names,
                resume=self._resume,
                robot_type=self._agent_name,
            )
            self._storage.configure_lerobot(
                env=self._manager.env,
                cameras_conf=self._cameras_conf,
                writer=self._writer,
                task=self._task,
            )
            self._writer.__enter__()
            self._writer_entered = True
        except Exception:
            self._writer_entered = False
            raise

    def on_episode_start(self) -> None:
        """每集开始前：计数。"""
        self._ep_idx += 1

    def on_episode_end(self, task_is_success: bool) -> None:
        """每集结束后：帧数统计（在 save_data/clear_data 清空缓冲区之前读取）。"""
        _ep_frames = self._storage.buffered_frame_count
        orca_logger.info(f"[EP {self._ep_idx}] 捕获 {_ep_frames} 帧")

    def on_run_end(self) -> None:
        """run() 结束时：释放 writer 资源（在 env.close() 之前）。

        异常安全：无论采集过程中是否抛异常（包括 KeyboardInterrupt），
        ``manager.run()`` 的 ``finally`` 块都会调用本方法，保证 writer 被正确关闭。

        清理逻辑：
            - 若 ``__enter__`` 成功过：调 ``__exit__``（VEM 清理 + finalize 写 parquet footer）
            - 若 ``__enter__`` 未成功但 writer 已创建：调 ``close()``（停写图线程 + finalize）
            - 若 writer 创建失败（None）：跳过
        manager.run() 的 finally 随后执行 env.close()（关闭 viewer + 录制器 + gRPC）。
        """
        if self._writer is None:
            return
        try:
            if self._writer_entered:
                self._writer.__exit__(None, None, None)
            else:
                # __enter__ 未成功（on_run_start 中途失败），但仍需停写图线程 + finalize
                self._writer.close()
        except Exception as e:
            orca_logger.error(f"[LeRobot] writer 清理异常: {e}")
        finally:
            try:
                orca_logger.info(
                    f"采集结束，共 {self._writer.num_episodes} 集 / "
                    f"{self._writer.num_frames} 帧"
                )
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VR 遥操作采集 → LeRobot v3.0 格式（parquet + 视频）"
    )
    parser.add_argument(
        "--level", required=True,
        help="关卡标识，仅用于 scratch 暂存目录命名与日志归档，不实际切换 OrcaStudio 场景"
    )
    parser.add_argument(
        "--agent_name", required=True, choices=["openloong", "tiangong2"],
        help="机器人型号"
    )
    parser.add_argument("--task_config", required=True, help="场景配置 YAML 文件名")
    parser.add_argument("--lerobot_out", required=True, help="LeRobot 数据集输出根目录")
    parser.add_argument(
        "--repo_id", default="local/manipulation",
        help="LeRobot repo_id（默认 local/manipulation）"
    )
    parser.add_argument(
        "--task", default="robot arm manipulation",
        help="任务语言描述（写入 LeRobot 元数据）"
    )
    parser.add_argument("--fps", type=int, default=30, help="采集帧率，默认 30")
    parser.add_argument(
        "--resume", action="store_true",
        help="追加到已有数据集（断点续采）"
    )
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--cameras", default="head,wrist_l,wrist_r",
        help="启用的相机列表（逗号分隔，可选 head/wrist_l/wrist_r），默认三路全开。"
             "显存不足时可只留 head 减轻 GPU 压力，例如 --cameras head",
    )
    args = parser.parse_args()

    agent_conf = _load_agent_conf(args.agent_name)
    lerobot_out = os.path.abspath(os.path.expanduser(args.lerobot_out))

    cameras_conf_all = getattr(agent_conf, "cameras_conf", None)
    if not cameras_conf_all:
        orca_logger.error(
            f"agent_conf '{args.agent_name}' 缺少 cameras_conf 属性，无法启动 LeRobot 采集"
        )
        return

    _enabled = {k.strip() for k in args.cameras.split(",") if k.strip()}
    cameras_conf = _filter_cameras_by_name(cameras_conf_all, enabled=_enabled)
    if not cameras_conf:
        orca_logger.error(
            f"--cameras '{args.cameras}' 未匹配到任何有效相机（cameras_conf keys="
            f"{list(cameras_conf_all.keys())}），退出"
        )
        return

    # ── 关节初值 ──────────────────────────────────────────────────────
    default_joint_values: dict = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = v

    # ── VR 设备 ───────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("  LeRobot 数采启动中...", flush=True)
    print(f"  机器人: {args.agent_name}  场景: {args.level}", flush=True)
    print(f"  相机: {list(cameras_conf.keys())}", flush=True)
    print(f"  输出目录: {lerobot_out}", flush=True)
    print("  等待 Pico 连接（请确认 Pico 端 OrcaLab App 已启动）...", flush=True)
    print("=" * 60, flush=True)
    orca_logger.info("Creating VR device")
    pico_device = PicoJoystickDevice(PicoJoystick())

    # ── 场景管理 ──────────────────────────────────────────────────────
    print("[场景] 正在连接 OrcaStudio，加载场景配置...", flush=True)
    orca_logger.info("Creating scene manager")
    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.show_ui_message(
        1, "开始仿真程序，请按左右遥杆进行操作 ", "0xffff00", showtime=10
    )
    scene_manager.get_scene_data(script_name, "beginscene")

    # ── storage ──────────────────────────────────────────────────────
    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", args.agent_name, args.level)
    storage = _make_storage(args.agent_name, scratch_dir)

    # OrcaStudio 的 beginscene 是异步的：env.__init__ 时机器人可能尚未 spawn
    # （nu=0，模型里只有场景物件）。用包装函数在 nu=0 时返回正确形状的占位零向量，
    # 保证 DataCollectionEnv._set_obs_space 能顺利构建观测空间；
    # init_env() 重载含机器人模型后，后续调用自动委托给 storage.obs_callback。
    def _obs_callback_safe(env):
        if env.model.nu == 0:
            _n_motor = (
                len(agent_conf.gripper_l["actuator_names"])
                + len(agent_conf.gripper_r["actuator_names"])
            )
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(_n_motor, dtype=np.float32),
            }
        return storage.obs_callback(env)

    # ── DataCollectionManager ────────────────────────────────────────
    print("[场景] 正在初始化仿真环境（等待机器人 spawn）...", flush=True)
    orca_logger.info("Creating DataCollectionManager")
    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback_safe,
        env_index=0,
        device=pico_device,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=20,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    # 视频由 LeRobot 流式编码（streaming_encoding=True），不用 env.begin_save_video
    manager.save_video = False

    # ── 场景就绪后初始化控制器 ────────────────────────────────────────
    try:
        env.reset()
        time.sleep(0.1)
        # ── 补设真实关节初值（机器人已 spawn，关节存在）─────────
        env.set_default_joint_values(default_joint_values)

        # ── 控制器 ─────────────────────────────────────────────
        orca_logger.info("Disabling position actuator group")
        manager.set_disable_actuator_group([agent_conf.positions_group])

        orca_logger.info("Adding gripper controllers")
        controllers.add_gripper_2f85_pico_controller(
            manager, env, agent_conf.gripper_l, agent_conf.base_body,
            pico_device,
            [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
        )
        controllers.add_gripper_2f85_pico_controller(
            manager, env, agent_conf.gripper_r, agent_conf.base_body,
            pico_device,
            [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
        )

        orca_logger.info("Adding arm controllers")
        controllers.add_arm_osc_pico_controller(
            manager, env, agent_conf.l_arm, agent_conf.base_body,
            pico_device, PicoJoystickKey.L_TRANSFORM,
        )
        controllers.add_arm_osc_pico_controller(
            manager, env, agent_conf.r_arm, agent_conf.base_body,
            pico_device, PicoJoystickKey.R_TRANSFORM,
        )

        orca_logger.info("Adding task status controller")
        manager.set_task(PickPlaceTask(env))
        controllers.add_task_status_pico_controller(
            manager, env, pico_device, agent_conf.base_body
        )
    except KeyboardInterrupt:
        orca_logger.info("初始化阶段收到 Ctrl+C，退出")
        return
    except Exception as e:
        orca_logger.error(f"初始化失败: {e}\n{traceback.format_exc()}")
        return

    # ── 创建 LeRobot 回调 ────────────────────────────────────────────
    lerobot_callback = LeRobotEpisodeCallback(
        manager=manager,
        storage=storage,
        cameras_conf=cameras_conf,
        fps=args.fps,
        task=args.task,
        lerobot_out=lerobot_out,
        repo_id=args.repo_id,
        resume=args.resume,
        agent_name=args.agent_name,
    )
    manager.register_episode_callback(lerobot_callback)
    manager.render_fps = args.fps

    orca_logger.info(
        f"开始采集，LeRobot 输出: {lerobot_out}  "
        f"相机: {list(cameras_conf.keys())}  {args.fps}fps"
    )
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
