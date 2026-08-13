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
      --fps 20 \\
      --clock wall
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

# env 相机名（cameras_conf 的 key）→ LeRobot feature key 后缀。
# 例：``camera_head`` → ``observation.images.cam_head``。
CAMERA_LEROBOT_KEY_MAP = {
    "camera_head": "cam_head",
    "camera_wrist_l": "cam_wrist_l",
    "camera_wrist_r": "cam_wrist_r",
}

# 默认采集帧分辨率（H, W），与 cameras_conf 中的 Width/Height 对齐
DEFAULT_HW = (720, 1080)

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


def _camera_short_name(env_cam: str) -> str:
    """env 相机名 → 短名（去掉 ``camera_`` 前缀）。

    例：``camera_head`` → ``head``，``camera_wrist_l`` → ``wrist_l``。
    """
    prefix = "camera_"
    return env_cam[len(prefix):] if env_cam.startswith(prefix) else env_cam


def _build_camera_map(cameras_conf: dict, enabled: set[str] | None = None) -> dict:
    """从 cameras_conf 构建 ``{env_camera_name: lerobot_key}`` 映射。

    Args:
        cameras_conf: 机器人配置中的相机配置字典（key 是 env 相机名）。
        enabled: 可选的相机启用集合（短名，如 ``{"head", "wrist_l"}``，
            也可传完整 env 名 ``{"camera_head"}``）。为 None 时启用全部。

    Returns:
        ``{env_camera_name: lerobot_key}``，仅包含在 CAMERA_LEROBOT_KEY_MAP
        中有映射且（若指定 enabled）被启用的相机。
    """
    camera_map: dict = {}
    for env_cam in cameras_conf.keys():
        # 短名过滤（同时接受短名 "head" 和完整名 "camera_head"）
        if enabled is not None:
            short = _camera_short_name(env_cam)
            if short not in enabled and env_cam not in enabled:
                continue
        lerobot_key = CAMERA_LEROBOT_KEY_MAP.get(env_cam)
        if lerobot_key is None:
            orca_logger.warning(f"相机 {env_cam!r} 不在 CAMERA_LEROBOT_KEY_MAP 中，跳过")
            continue
        camera_map[env_cam] = lerobot_key
    return camera_map


def _filter_cameras_conf(cameras_conf: dict, camera_map: dict) -> dict:
    """按 camera_map 过滤 cameras_conf，仅保留映射中的相机。"""
    return {env_cam: cameras_conf[env_cam] for env_cam in camera_map.keys() if env_cam in cameras_conf}


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
        camera_map: dict,
        cam_hw: tuple,
        fps: int,
        clock: str,
        task: str,
        lerobot_out: str,
        repo_id: str,
        resume: bool,
        agent_name: str,
    ) -> None:
        self._manager = manager
        self._storage = storage
        self._cameras_conf = cameras_conf
        self._camera_map = camera_map
        self._cam_hw = cam_hw
        self._fps = fps
        self._clock = clock
        self._task = task
        self._lerobot_out = lerobot_out
        self._repo_id = repo_id
        self._resume = resume
        self._agent_name = agent_name

        self._writer: LeRobotDatasetWriter | None = None
        self._writer_entered: bool = False  # 标记 writer.__enter__ 是否成功（用于 __exit__ 配对）
        self._ep_idx = 0

    def on_run_start(self) -> None:
        """run() 主循环启动前：启动相机推流、探测真实帧尺寸、创建 writer、配置 storage。

        尺寸探测：``setup_cameras`` 后轮询等待首帧到达，用 ``decode_frame_at``
        解码获取真实帧尺寸。若与 ``cameras_conf`` 声明不符（引擎端可能忽略
        Width/Height 参数或被场景配置覆盖），以真实尺寸为准创建数据集，
        避免后续 ``add_frame`` 因 shape 不符被 LeRobot 拒绝。

        异常安全：任何步骤失败都会设置 ``_writer_entered=False``，确保
        ``on_run_end`` 不会对未 ``__enter__`` 的 writer 调用 ``__exit__``。
        """
        try:
            # 1. 启动相机推流 + viewer（OrcaGym 客户端 PyAV remux 录制）
            self._storage.setup_cameras(self._manager.env, self._cameras_conf, show_viewer=False)

            # 2. 探测真实帧尺寸（阻塞等待首帧，最多 ~5 秒）
            cam_hw = self._probe_actual_cam_hw()

            # 3. 创建 writer（用探测到的真实尺寸）
            cam_shape = (3, cam_hw[0], cam_hw[1])
            self._writer = LeRobotDatasetWriter.create(
                repo_id=self._repo_id,
                root=self._lerobot_out,
                fps=self._fps,
                camera_map=self._camera_map,
                state_dim=self._storage.state_dim,
                state_names=self._storage.state_names,
                cam_shape=cam_shape,
                resume=self._resume,
                robot_type=self._agent_name,
            )
            # 4. 注入 LeRobot 依赖（fps / env / camera_map / writer / task / clock）
            self._storage.configure_lerobot(
                fps=self._fps,
                env=self._manager.env,
                cameras_conf=self._cameras_conf,
                camera_map=self._camera_map,
                target_hw=cam_hw,
                writer=self._writer,
                task=self._task,
                clock=self._clock,
            )
            # 5. 进入 writer context（管理 VideoEncodingManager 生命周期）
            self._writer.__enter__()
            self._writer_entered = True
        except Exception:
            # setup_cameras/create/configure/__enter__ 任一步失败，标记未 entered，
            # on_run_end 仍会尝试 close()（停写图线程 + finalize），但不会 __exit__。
            self._writer_entered = False
            raise

    def _probe_actual_cam_hw(self, timeout: float = 5.0, poll_interval: float = 0.1) -> tuple:
        """探测相机实际渲染尺寸（阻塞等待首帧并解码）。

        ``setup_cameras`` 后引擎端可能不会立即推流，需轮询等待首帧到达。
        首帧到达后用 ``decode_frame_at`` 解码获取真实 (H, W)。若与
        ``self._cam_hw`` 声明不符，打印警告并以真实尺寸为准。

        Args:
            timeout: 等待首帧的最长时间（秒）。
            poll_interval: 轮询间隔（秒）。

        Returns:
            ``(height, width)`` 真实帧尺寸。若超时未收到首帧，回退到
            ``self._cam_hw`` 声明尺寸（后续 ``add_frame`` 会因 shape 不符报错，
            提示用户检查相机推流是否正常）。
        """
        manager = self._manager.env.get_recorder_manager()
        primary_cam = next(iter(self._camera_map.keys()))
        deadline = time.monotonic() + timeout
        probed_hw: tuple | None = None

        while time.monotonic() < deadline:
            latest_idx = manager.get_latest_frame_simulate_index(primary_cam, "color")
            if latest_idx is not None and latest_idx >= 0:
                img = manager.decode_frame_at(primary_cam, latest_idx, "color")
                if img is not None and img.ndim == 3 and img.shape[0] >= 2:
                    probed_hw = (int(img.shape[0]), int(img.shape[1]))  # HWC
                    break
            time.sleep(poll_interval)

        if probed_hw is None:
            orca_logger.warning(
                f"[LeRobot] 探测首帧尺寸超时（{timeout}s 未收到 {primary_cam} color 流），"
                f"回退到 cameras_conf 声明尺寸 {self._cam_hw}。"
                f"若后续 add_frame 报 shape 不符，请检查相机推流是否正常。"
            )
            return self._cam_hw

        if probed_hw != self._cam_hw:
            orca_logger.warning(
                f"[LeRobot] 实际帧尺寸 {probed_hw[0]}x{probed_hw[1]} 与 "
                f"cameras_conf 声明 {self._cam_hw[0]}x{self._cam_hw[1]} 不符，"
                f"以实际尺寸为准创建数据集。"
            )
        else:
            orca_logger.info(
                f"[LeRobot] 探测帧尺寸: {probed_hw[0]}x{probed_hw[1]}（与声明一致）"
            )
        return probed_hw

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
        "--clock", choices=("sim", "wall"), default="sim",
        help="采帧门控时钟源。sim=仿真时间（默认）；wall=墙钟，用于 VR 遥操作，"
             "让录到的机械臂速度=人真实操作速度、视频时长≈操作墙钟时长。"
             "wall 模式下建议 --fps 20（循环频率约 18~25Hz），过高会欠采。",
    )
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

    # ── 相机映射 / 配置过滤 ────────────────────────────────────────────
    cameras_conf_all = getattr(agent_conf, "cameras_conf", None)
    if not cameras_conf_all:
        orca_logger.error(
            f"agent_conf '{args.agent_name}' 缺少 cameras_conf 属性，无法启动 LeRobot 采集"
        )
        return

    _enabled = {k.strip() for k in args.cameras.split(",") if k.strip()}
    camera_map = _build_camera_map(cameras_conf_all, enabled=_enabled)
    if not camera_map:
        orca_logger.error(
            f"--cameras '{args.cameras}' 未匹配到任何有效相机（cameras_conf keys="
            f"{list(cameras_conf_all.keys())}），退出"
        )
        return
    cameras_conf = _filter_cameras_conf(cameras_conf_all, camera_map)

    # 从 cameras_conf 读取目标分辨率（与渲染分辨率一致）
    first_cam_props = next(iter(cameras_conf.values()))
    cam_hw = (
        int(first_cam_props.get("Height", DEFAULT_HW[0])),
        int(first_cam_props.get("Width", DEFAULT_HW[1])),
    )

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
    print(f"  相机: {list(camera_map.keys())}  分辨率: {cam_hw[0]}x{cam_hw[1]}", flush=True)
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
        camera_map=camera_map,
        cam_hw=cam_hw,
        fps=args.fps,
        clock=args.clock,
        task=args.task,
        lerobot_out=lerobot_out,
        repo_id=args.repo_id,
        resume=args.resume,
        agent_name=args.agent_name,
    )
    manager.register_episode_callback(lerobot_callback)
    manager.render_fps = 30

    orca_logger.info(
        f"开始采集，LeRobot 输出: {lerobot_out}  "
        f"相机: {list(camera_map.keys())}  {cam_hw[0]}x{cam_hw[1]}  {args.fps}fps"
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
