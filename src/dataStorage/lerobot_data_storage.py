"""LeRobot v3.0 格式数据存储层。

设计要点（多相机同步 + 回调直接 add_frame 真实图像）：
    默认主时钟 = 仿真时间（由 ``collection_data`` 的调用方传入 simulate_index）。
    ``collection_data`` 在每个控制步被调用，用当前 ``simulate_index`` 提交
    ``SingleFrameTask``（每个相机一个，目标 sim_idx 相同）：
        - 各 recorder 的 save_worker 内部 FIFO 保证 ``DecodeTask`` 先于
          ``SingleFrameTask`` 执行，因此每个 ``SingleFrameTask`` 的 future
          result 就是该相机在该 sim_idx 的解码帧（由持久 CodecContext 解码）。
        - 主相机的 ``on_frame`` 回调内等待所有副相机 future 完成，收集各相机
          decoded_frame，合并成 ``images`` 字典后调用 ``add_frame``。
        - 副相机的 ``on_frame`` 为 no-op（结果通过 future 由主相机收集）。
        - 最后一帧（无 next-state 作 action）在 save_data 时丢弃。
    episode 成功后 ``save_data`` 调用 ``flush_episode``（同步 ``save_episode``，
    LeRobot 内部完成 parquet + meta + ffmpeg 编码）。

多相机并发安全（修复跨 recorder 读共享状态问题）：
    旧方案在主相机回调内通过 ``manager.get_last_decoded_frame`` 读副相机的
    ``_last_decoded_frame``，存在跨线程无锁访问 + sim_idx 不可控（读到"过新"
    或"过旧"帧）问题。新方案对每个相机提交独立 ``SingleFrameTask``，通过
    future 同步保证 sim_idx 严格对齐，且不跨 recorder 读共享状态。

线程职责：
    T0（主线程）：控制→物理→渲染→提交 SingleFrameTask（每个相机一个，非阻塞）
    T-recv（各 recorder 接收线程）：收帧，提交 DecodeTask，触发 SingleFrameTask
    T-save（各 recorder save_worker 线程）：执行 DecodeTask（解码）+
        SingleFrameTask 回调（主相机回调内等待副相机 future + add_frame）
    T0（主线程，episode 结束）：flush_episode（同步 save_episode）

State/action 布局（各机器人不同，由 RobotProfile.build_state() 决定）：
    openloong (16 维)：l_pos(3) + l_quat_xyzw(4) + r_pos(3) + r_quat_xyzw(4) + l_grip(1) + r_grip(1)
    tiangong2 (38 维)：l_pos(3) + l_quat_xyzw(4) + r_pos(3) + r_quat_xyzw(4) + effector_motor_norm(24)

    action[i] = state[i+1]（next-step 移位约定，与 v5 及 openpi 训练约定一致）。

运行环境：
    本模块依赖 lerobot>=0.3.0 + orca_gym 26.6.x + av。若当前环境缺失
    lerobot 库，动态导入会抛出 ImportError 并提示安装，不会影响 HDF5 采集流程。
    其他开发人员如果不需要 LeRobot 数据格式功能，可忽略本模块。
"""
from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from orca_gym.recorder import SingleFrameTask
from dataStorage.abstract_data_storage import AbstractDataStorage
from dataStorage.robot_profile import RobotProfile

if TYPE_CHECKING:
    from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
    from orca_gym.recorder import FrameEntry

# 模块级 logger，避免使用 root logger（与项目 orca_logger 风格一致）
_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 动态导入 lerobot（缺失时抛出 ImportError 并提示安装，不影响其他模块）
# ---------------------------------------------------------------------------

def _import_lerobot_dataset():
    """动态导入 LeRobotDataset，兼容新旧模块路径。"""
    for mod in (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    ):
        try:
            import importlib
            return importlib.import_module(mod).LeRobotDataset
        except Exception:
            pass
    raise ImportError(
        "无法导入 LeRobotDataset。当前环境缺失 lerobot 库，"
        "请执行 `pip install lerobot>=0.3.0` 安装。"
        "如不需要 LeRobot 格式功能，可忽略本模块，使用现有的 HDF5 采集流程即可。"
    )


def _import_video_encoding_manager():
    """动态导入 VideoEncodingManager，作为 with 上下文管理器使用。"""
    for mod in (
        "lerobot.datasets.video_utils",
        "lerobot.common.datasets.video_utils",
    ):
        try:
            import importlib
            return importlib.import_module(mod).VideoEncodingManager
        except Exception:
            pass
    raise ImportError(
        "无法导入 VideoEncodingManager，请检查 lerobot 版本是否 >=0.3.0。"
        "如缺失，请执行 `pip install lerobot>=0.3.0` 安装。"
    )


def camera_keys(camera_map: dict) -> list[str]:
    """返回 LeRobot 相机键列表（写入数据集 features 时用）。

    ``camera_map`` 结构：``{env_camera_sensor_name: lerobot_key}``，
    其中 ``lerobot_key`` 是写入 LeRobot features 的 key 后缀（如
    ``observation.images.<lerobot_key>``）。
    """
    return list(camera_map.values())


def _validate_dataset_root(root: str) -> None:
    """校验 LeRobot 数据集根目录 ``root`` 的安全性。

    在非 resume 模式下，``LeRobotDatasetWriter.create`` 会执行
    ``shutil.rmtree(root)`` 清空旧目录。若 ``root`` 指向当前工作目录或其
    祖先目录，会误删用户的工作目录（含脚本自身、源码、.git 等）。

    本函数拒绝以下危险 ``root``：
        - 空字符串或 None
        - ``"."`` / ``".."`` 或解析后为当前目录
        - 当前工作目录（cwd）本身或其任意祖先目录
        - 用户 home 目录

    Args:
        root: LeRobot 数据集根目录路径（相对或绝对）。

    Raises:
        ValueError: ``root`` 不安全时抛出，含明确的修复建议。
    """
    if not root or not root.strip():
        raise ValueError(
            "LeRobot 数据集 root 为空。请用 --lerobot_out 指定一个专用的数据集输出"
            "子目录（如 ./lerobot_out/my_dataset），不要用当前目录 '.'。"
        )

    root_path = Path(root).resolve()
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()

    def _is_ancestor(maybe_ancestor: Path, descendant: Path) -> bool:
        """``maybe_ancestor`` 是否是 ``descendant`` 的祖先（含自身）。"""
        try:
            descendant.relative_to(maybe_ancestor)
            return True
        except ValueError:
            return False

    # 拒绝 root 是 cwd 本身或其祖先（会删光 cwd 及其全部内容）
    if _is_ancestor(root_path, cwd):
        raise ValueError(
            f"LeRobot 数据集 root {root_path} 是当前工作目录 {cwd} 本身或其祖先目录！"
            f"非 --resume 模式下会执行 shutil.rmtree 清空该目录，"
            f"导致脚本自身和源码被删除。请用 --lerobot_out 指定一个专用的"
            f"子目录（如 ./lerobot_out/my_dataset），不要用 '.' 或 '..'。"
        )

    # 拒绝 root == home（删光整个 home 目录）
    if root_path == home:
        raise ValueError(
            f"LeRobot 数据集 root 指向用户 home 目录 {home}！"
            f"shutil.rmtree 会删除 home 下全部文件，极其危险。"
            f"请用 --lerobot_out 指定一个专用的子目录。"
        )


# ---------------------------------------------------------------------------
# LeRobotDatasetWriter：封装 LeRobotDataset 的创建/写入/生命周期
# ---------------------------------------------------------------------------

class LeRobotDatasetWriter:
    """封装 LeRobotDataset 的创建、流式帧写入与 episode 存盘生命周期。

    使用流式编码（``streaming_encoding=True``）：``add_frame`` 时立即送入
    编码线程（不写 PNG 临时文件），``save_episode`` 时只需 flush 编码器
    （近瞬时），适合实时采集。

    ``add_frame_with_timestamp``：供 ``SingleFrameTask`` 回调使用，
    显式传入 timestamp（来自 ``FrameEntry.timestamp_ns / 1e9``）。

    使用方式：
        writer = LeRobotDatasetWriter.create(...)
        with writer:
            ...  # 若干集采集，调用 add_frame_with_timestamp() / flush_episode() / discard_episode()
        # with 块退出后 close() 自动停写图线程 + VEM 清理残留资源
    """

    def __init__(self, dataset) -> None:
        self._dataset = dataset
        self._vem_ctx = None
        # 下一个待采集 episode 的 index（主线程独占写）
        self._next_ep_idx: int = dataset.num_episodes

    # -- 工厂方法 --

    @classmethod
    def create(
        cls,
        repo_id: str,
        root: str,
        fps: int,
        camera_map: dict,
        state_dim: int,
        state_names: list[str],
        cam_shape: tuple,
        resume: bool = False,
        robot_type: str = "humanoid",
    ) -> "LeRobotDatasetWriter":
        """创建（或恢复）一个 LeRobotDataset，返回已包装的 writer。

        安全护栏：``root`` 不允许是当前工作目录或其祖先目录，否则在非 resume
        模式下的 ``shutil.rmtree(root)`` 会清空用户的工作目录（含脚本自身）。
        """
        _validate_dataset_root(root)
        LeRobotDataset = _import_lerobot_dataset()
        cams = camera_keys(camera_map)

        if resume and Path(root).exists():
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                download_videos=False,
                tolerance_s=0.0001,
            )
            dataset.start_image_writer(0, 4 * len(cams))
            dataset.episode_buffer = dataset.create_episode_buffer()
            _logger.info(
                f"[resume] 已加载 {dataset.num_episodes} 集 / "
                f"{dataset.num_frames} 帧 (root={root})"
            )
        else:
            if Path(root).exists() and not resume:
                shutil.rmtree(root)

            features: dict = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (state_dim,),
                    "names": [state_names],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (state_dim,),
                    "names": [state_names],
                },
            }
            for cam_key in cams:
                features[f"observation.images.{cam_key}"] = {
                    "dtype": "video",
                    "shape": cam_shape,
                    "names": ["channels", "height", "width"],
                }

            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=int(fps),
                robot_type=robot_type,
                features=features,
                root=root,
                use_videos=True,
                tolerance_s=0.0001,
                image_writer_processes=0,
                image_writer_threads=4 * len(cams),
                # 流式编码：add_frame 时立即送入编码线程（不写 PNG 临时文件），
                # save_episode 时只需 flush 编码器（近瞬时）。适合实时采集。
                # 每个摄像头一个独立编码线程（daemon），队列满时丢弃帧 + warning。
                streaming_encoding=True,
            )

        return cls(dataset)

    # -- context manager（管理 VideoEncodingManager 生命周期）--

    def __enter__(self) -> "LeRobotDatasetWriter":
        VideoEncodingManager = _import_video_encoding_manager()
        self._vem_ctx = VideoEncodingManager(self._dataset)
        self._vem_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(exc_type, exc_val, exc_tb)
        return False

    def close(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        """有序关闭：停写图线程 → VEM 清理 → finalize 写 parquet footer。

        幂等：多次调用安全（image_writer / VEM / finalize 均有内部保护）。
        退出清理顺序：
            1. stop_image_writer()：发 None 令牌停写图线程
            2. VEM.__exit__()：清理残留 PNG 目录（异常中断时）+ 批编码剩余 episode
            3. dataset.finalize()：关闭 parquet writer，写入 footer
               （不调 finalize 会导致 parquet 文件缺少 footer，无法读取）
        """
        try:
            self.stop_image_writer()
        finally:
            if self._vem_ctx is not None:
                self._vem_ctx.__exit__(exc_type, exc_val, exc_tb)
            try:
                self._dataset.finalize()
            except Exception:
                _logger.exception("[LeRobot] finalize 异常")

    # -- 流式写帧（SingleFrameTask 回调内调用）--

    def add_frame(
        self,
        state: np.ndarray,
        action: np.ndarray,
        images: dict,
        task: str,
    ) -> None:
        """流式写入一帧（timestamp 由 LeRobot 自动维护）。

        供 ``SingleFrameTask.on_frame`` 回调使用：真实图像进 ``add_frame``，
        LeRobot 自动计算 ``timestamp = frame_index / fps``（从 0 开始的相对
        时间，float32 精度足够），视频编码与 parquet 行 1:1 对应。

        之所以不用引擎系统时间戳：LeRobot ``timestamp`` 的 dtype 是
        ``float32``（7 位有效数字），而系统时间戳约 ``1.78e9`` 秒，float32
        精度约 128 秒——相邻帧间隔 0.033 秒会被完全丢失，导致同一集内所有
        帧的 timestamp 相同。用相对时间 ``frame_index / fps`` 才适合 float32。

        Args:
            state: ``observation.state``（float32 向量）。
            action: ``action``（float32 向量，next-step 移位）。
            images: ``{cam_key: np.ndarray}``，每个值是真实图像（与 cam_shape 一致）。
                cam_key 是 LeRobot feature key 后缀（如 ``"cam_head"``），
                内部自动拼成 ``observation.images.<cam_key>``。
            task: 任务描述字符串。
        """
        frame: dict = {
            "observation.state": state.astype(np.float32),
            "action": action.astype(np.float32),
            "task": task,
        }
        for cam_key, img in images.items():
            frame[f"observation.images.{cam_key}"] = img
        self._dataset.add_frame(frame)

    # -- episode 生命周期管理 --

    def flush_episode(self) -> int:
        """同步保存本集（``save_episode``），返回本集 episode_index。

        使用流式编码时，``add_frame`` 已实时编码到临时 MP4，``save_episode``
        只需 flush 编码器并收集结果（近瞬时），不阻塞主线程。

        Returns:
            本集 episode_index（供日志使用）。
        """
        ep_idx = self._next_ep_idx
        # save_episode 是 lerobot 公共 API：流式编码下只需 flush 编码器
        # + 写 parquet + 更新 meta（近瞬时，不阻塞）
        self._dataset.save_episode()
        self._next_ep_idx += 1
        return ep_idx

    def discard_episode(self) -> None:
        """丢弃本集缓存（任务失败时调用）。

        同步 flush PNG（保证 rmtree 安全）→ 删 PNG 目录 → 用 _next_ep_idx 重置缓冲。
        _next_ep_idx 不递增（下次重试同一 episode_index）。
        """
        # clear_episode_buffer() 是 lerobot 公共 API：flush PNG + rmtree + 重置 buffer
        self._dataset.clear_episode_buffer()
        # clear_episode_buffer 内部用 meta.total_episodes 创建新缓冲；若
        # save_episode 尚未执行，meta.total_episodes 可能落后于 _next_ep_idx，需修正。
        self._dataset.episode_buffer = self._dataset.create_episode_buffer(
            episode_index=self._next_ep_idx
        )

    def stop_image_writer(self) -> None:
        """停止 AsyncImageWriter 线程池（发 None 令牌）。"""
        try:
            if getattr(self._dataset, "image_writer", None) is not None:
                self._dataset.stop_image_writer()
        except Exception:
            _logger.exception("[LeRobot] stop_image_writer 异常")

    @property
    def features(self) -> dict:
        """LeRobot 数据集 features 定义（含 dtype / shape / names）。

        用于查询各 feature 的 shape（如黑屏占位图 shape）。
        """
        return self._dataset.features

    @property
    def num_episodes(self) -> int:
        return self._dataset.num_episodes

    @property
    def num_frames(self) -> int:
        return self._dataset.num_frames

    @property
    def episode_frame_count(self) -> int:
        """当前 episode buffer 中已写入的帧数。

        从 LeRobot ``episode_buffer["timestamp"]`` 的长度获取，反映实际
        ``add_frame`` 成功的次数（被 ``SingleFrameTask`` 丢弃的帧不计入）。
        """
        buf = self._dataset.episode_buffer
        if buf is None:
            return 0
        ts = buf.get("timestamp")
        return len(ts) if ts is not None else 0


# ---------------------------------------------------------------------------
# LerobotDataStorage（取代 LeRobotSimSyncMixin）
# ---------------------------------------------------------------------------

class LerobotDataStorage(AbstractDataStorage):
    """LeRobot v3.0 格式数据存储中间类。

    继承 ``AbstractDataStorage``，委托 ``RobotProfile`` 的
    ``build_state`` / ``state_dim`` / ``state_names``。

    流程（每集）：
        1. 外部调用 ``configure_lerobot`` 注入 fps / env / camera_map / writer 等
        2. ``start_episode_recording`` 记录起始 sim_idx
        3. 每步 ``collection_data(obs, env, simulate_index=...)``：
           - 对每个相机提交 ``SingleFrameTask``（目标 sim_idx 相同，各 recorder
             内部 FIFO 保证 DecodeTask 先于 SingleFrameTask 执行，future result
             为该相机在该 sim_idx 的解码帧）
           - 主相机 ``on_frame`` 回调等待所有副相机 future，合并图像后 add_frame
        4. ``stop_episode_recording`` 记录结束 sim_idx
        5. 任务成功 → ``save_data``：
           - ``flush_episode`` 同步 ``save_episode``（LeRobot 内部完成
             parquet + meta + ffmpeg 编码真实视频）
        6. 任务失败 → ``clear_data``：``discard_episode`` + 重置状态

    真实图像策略：
        - 各 recorder 的 save_worker 维护持久 CodecContext，``DecodeTask`` 实时
          解码每帧 NAL 并更新 ``_last_decoded_frame``（P 帧无需回溯到 IDR）
        - ``SingleFrameTask.execute`` 从 ``last_decoded_frame`` 取解码帧，
          作为 future result 返回（主相机回调内等待副相机 future 收集图像）
        - 直接 ``add_frame`` 传真实图像给 LeRobot，由 LeRobot API 负责编码
          视频和写 parquet，无需事后替换
    """

    def __init__(
        self,
        dataset_path: str,
        robot_profile: RobotProfile,
    ):
        super().__init__(dataset_path=dataset_path, robot_profile=robot_profile)
        self._lr_writer: LeRobotDatasetWriter | None = None
        self._lr_env: OrcaGymLocalEnv | None = None
        self._lr_fps: float = 0.0
        self._lr_task: str = "robot manipulation"
        self._lr_camera_map: dict = {}
        self._lr_cam_keys: list[str] = []
        self._lr_primary_camera: str | None = None
        self._lr_ep_start_sim_idx: int | None = None
        self._lr_ep_end_sim_idx: int | None = None
        self._lr_pending_futures: list[Future] = []
        self._reset_episode()

    # -- RobotProfile 委托（LeRobot 特有）--

    def build_state(self, obs: dict) -> np.ndarray:
        """从 obs 组装 LeRobot state 向量（委托给 profile）。"""
        return self._robot_profile.build_state(obs)

    @property
    def state_dim(self) -> int:
        """LeRobot state 维度（委托给 profile）。"""
        return self._robot_profile.state_dim

    @property
    def state_names(self) -> list[str]:
        """LeRobot state 列名（委托给 profile）。"""
        return self._robot_profile.state_names

    @property
    def buffered_frame_count(self) -> int:
        """当前集已写帧数（从 LeRobot episode_buffer 实时读取）。

        反映实际 ``add_frame`` 成功的次数（被 ``SingleFrameTask`` 丢弃的帧
        不计入）。在 ``save_data`` / ``clear_data``（会清空状态）之前读取。

        会先等待所有已提交的异步 ``SingleFrameTask`` 完成，确保计数准确
        （``on_episode_end`` 在 ``save_data`` 之前调用时，可能还有任务未落地）。
        """
        self._wait_pending_frames()
        if self._lr_writer is not None:
            return self._lr_writer.episode_frame_count
        return 0

    # -- 配置注入 --

    def configure_lerobot(
        self,
        fps: float,
        env: OrcaGymLocalEnv,
        cameras_conf: dict,
        camera_map: dict,
        target_hw: tuple,
        writer: LeRobotDatasetWriter,
        task: str = "robot manipulation",
        clock: str = "sim",
    ) -> None:
        """在 __init__ 完成后、run() 开始前调用，注入 LeRobot 相关依赖。

        Args:
            fps: LeRobot 数据集 fps（用于降频门控参考，实际帧率以渲染为准）。
            env: OrcaGym 环境（用于 ``get_recorder_manager``）。
            cameras_conf: 相机配置（由 ``setup_cameras`` 注入到 env，
                ``configure_lerobot`` 仅用于校验 primary_camera 存在）。
            camera_map: ``{env_camera_name: lerobot_key}``。
            target_hw: ``(height, width)`` 目标分辨率（保留参数，用于校验
                与 writer 创建时的 cam_shape 一致）。
            writer: ``LeRobotDatasetWriter`` 实例。
            task: 任务描述字符串。
            clock: 时钟源（保留参数，当前实现以 simulate_index 为准）。
        """
        if clock not in ("sim", "wall"):
            raise ValueError(f"clock 只能是 'sim' 或 'wall'，收到: {clock!r}")
        self._lr_fps = float(fps)
        self._lr_env = env
        self._lr_camera_map = dict(camera_map)
        self._lr_cam_keys = camera_keys(camera_map)
        self._lr_writer = writer
        self._lr_task = task
        # primary_camera：camera_map 的第一个 env 相机名（用于降频门控查询）
        # 所有相机 simulate_index 一致，查任一即可
        self._lr_primary_camera = next(iter(self._lr_camera_map.keys())) if self._lr_camera_map else None

    # -- 覆盖 AbstractDataStorage 的接口 --

    def collection_data(self, data: dict, env: OrcaGymLocalEnv, **kwargs) -> None:
        """直接提交 SingleFrameTask 回调写帧（回调内解码真实图像 + add_frame）。

        每个控制步直接用当前 ``simulate_index`` 提交 ``SingleFrameTask``，
        由 WebSocket 接收线程在帧到达时触发，worker 线程执行回调。
        若帧不存在（渲染跳号），``SingleFrameTask.execute`` 自动丢弃。

        不需要主线程做降频门控查询——``simulate_index`` 与渲染帧 1:1 对应
        （每个控制步都会 ``env.render(simulate_index)``），``SingleFrameTask``
        的触发条件 ``current_simulate_index >= task.simulate_index`` 正好
        匹配"帧到达后触发"的语义。

        Args:
            data: ``obs_callback`` 返回的观测字典。
            env: OrcaGym 环境。
            **kwargs: 必须含 ``simulate_index``（当前物理仿真步索引）。
        """
        if self._lr_writer is None or self._lr_env is None or self._lr_primary_camera is None:
            return
        simulate_index = kwargs.get("simulate_index")
        if simulate_index is None:
            return

        state_cur = self.build_state(data)

        # 第一帧没有 prev_state（无 action 可移位），只缓存不提交
        if self._lr_prev_state is not None:
            self._submit_frame_task(self._lr_prev_state, state_cur, simulate_index)
        self._lr_prev_state = state_cur

    def _submit_frame_task(
        self,
        state_prev: np.ndarray,
        action: np.ndarray,
        simulate_index: int,
    ) -> None:
        """对每个相机提交 SingleFrameTask，主相机回调等待所有副相机 future
        后合并图像并 ``add_frame``。**非阻塞**。

        对每个相机（含主相机）提交一个 ``SingleFrameTask``，目标
        ``simulate_index`` 相同。各 recorder 内部 save_worker FIFO 保证
        ``DecodeTask`` 先于同帧的 ``SingleFrameTask`` 执行，因此每个 future
        的 result 就是该相机在该 sim_idx 的解码帧。主相机回调内等待所有副相机
        future 完成，合并图像后 ``add_frame``。若某相机解码失败，该帧被丢弃。

        Args:
            state_prev: 上一帧的 state（作为 ``observation.state``）。
            action: 当前帧的 state（作为 ``action``，next-step 移位）。
            simulate_index: 目标帧的 simulate_index。
        """
        writer = self._lr_writer
        camera_map = self._lr_camera_map
        task_desc = self._lr_task
        env = self._lr_env
        primary_cam = self._lr_primary_camera
        manager = env.get_recorder_manager()

        secondary_cams = [c for c in camera_map if c != primary_cam]

        # 副相机回调返回 decoded_frame，供主相机回调通过 future.result() 收集
        secondary_futures: dict[str, Future] = {}
        for env_cam in secondary_cams:
            sec_task = SingleFrameTask(
                simulate_index=simulate_index,
                on_frame=lambda _entry, decoded: decoded,
            )
            secondary_futures[env_cam] = manager.submit_task(env_cam, sec_task, "color")

        def on_frame(frame_entry: FrameEntry, decoded_frame) -> None:
            """主相机回调：等待所有副相机 future，合并图像后 add_frame。"""
            images: dict = {}
            if decoded_frame is None:
                _logger.warning(
                    f"[LeRobot] 主相机解码帧为 None "
                    f"(cam={primary_cam}, sim_idx={simulate_index})，丢弃该帧"
                )
                return
            images[camera_map[primary_cam]] = decoded_frame
            for env_cam in secondary_cams:
                fut = secondary_futures[env_cam]
                try:
                    sec_decoded = fut.result()
                except Exception as e:
                    _logger.warning(
                        f"[LeRobot] 副相机 future 异常 "
                        f"(cam={env_cam}, sim_idx={simulate_index}): {e}，丢弃该帧"
                    )
                    return
                if sec_decoded is None:
                    _logger.warning(
                        f"[LeRobot] 副相机解码帧为 None "
                        f"(cam={env_cam}, sim_idx={simulate_index})，丢弃该帧"
                    )
                    return
                images[camera_map[env_cam]] = sec_decoded
            writer.add_frame(
                state=state_prev,
                action=action,
                images=images,
                task=task_desc,
            )

        # 提交主相机任务（on_frame 内会等待副相机 future）
        primary_task = SingleFrameTask(simulate_index=simulate_index, on_frame=on_frame)
        primary_future = manager.submit_task(primary_cam, primary_task, "color")
        # 只跟踪主相机 future（副相机 future 由主相机回调内部等待）
        self._lr_pending_futures.append(primary_future)

    def save_data(self, env: OrcaGymLocalEnv | None = None, **kwargs) -> None:
        """任务成功：flush_episode（同步 save_episode，LeRobot 内部完成编码）。

        使用流式编码（``streaming_encoding=True``）时，``add_frame`` 已实时
        编码到临时 MP4，``save_episode`` 只需 flush 编码器并收集结果（近瞬时）。

        Args:
            env: OrcaGym 环境（保留参数，与基类签名一致）。
            **kwargs: 保留（与基类签名一致，LeRobot 不使用 task_info / scene_info）。
        """
        if self._lr_writer is None:
            return
        # 等待所有已提交的 SingleFrameTask 完成（异步回调 add_frame 全部落地）
        self._wait_pending_frames()
        # 从 LeRobot episode_buffer 获取实际写入的帧数
        # （被 SingleFrameTask 丢弃的帧不计入）
        actual_frames = self._lr_writer.episode_frame_count
        if actual_frames < 1:
            _logger.warning(
                f"[LeRobot] 帧数不足（{actual_frames}），丢弃本集"
            )
            self._lr_writer.discard_episode()
            self._reset_episode()
            return

        # flush_episode：LeRobot 内部完成 parquet + meta + 视频编码
        ep_idx = self._lr_writer.flush_episode()
        _logger.info(
            f"[LeRobot] ✓ Episode {ep_idx} flush 完成（{actual_frames} 帧，流式编码）"
        )

        self._reset_episode()

    def clear_data(self) -> None:
        """任务失败：等待未完成任务落地，丢弃帧缓存，重置 episode 状态。"""
        # 等待已提交的 SingleFrameTask 完成（避免 discard_episode 时回调仍在执行）
        self._wait_pending_frames()
        if self._lr_writer is not None:
            self._lr_writer.discard_episode()
        self._reset_episode()

    def start_episode_recording(
        self, env: OrcaGymLocalEnv, episode_index: int, start_simulate_index: int
    ) -> None:
        """记录 episode 起始 sim_idx（供 save_data 时 save_streaming 用）。"""
        super().start_episode_recording(env, episode_index, start_simulate_index)
        self._lr_ep_start_sim_idx = start_simulate_index

    def stop_episode_recording(
        self, env: OrcaGymLocalEnv, end_simulate_index: int
    ) -> None:
        """记录 episode 结束 sim_idx（供 save_data 时 save_streaming 用）。"""
        super().stop_episode_recording(env, end_simulate_index)
        self._lr_ep_end_sim_idx = end_simulate_index

    def _save_episode_videos(self, env: OrcaGymLocalEnv) -> dict:
        """LeRobot 不用基类的 _save_episode_videos（由 save_data 内部处理）。"""
        return {}

    def _reset_episode(self) -> None:
        """重置本集所有流式状态。"""
        self._lr_prev_state: np.ndarray | None = None
        self._lr_ep_start_sim_idx: int | None = None
        self._lr_ep_end_sim_idx: int | None = None
        self._lr_pending_futures: list[Future] = []

    def _wait_pending_frames(self) -> None:
        """等待所有已提交的主相机 SingleFrameTask 完成。

        ``_lr_pending_futures`` 只跟踪主相机 future（副相机 future 由主相机
        ``on_frame`` 回调内部等待）。主相机 future 完成意味着副相机 future 也
        已完成，``add_frame`` 已全部执行（或已丢弃）。
        """
        for fut in self._lr_pending_futures:
            try:
                fut.result()
            except Exception as e:
                _logger.warning(f"[LeRobot] SingleFrameTask 异常: {e}")
        self._lr_pending_futures = []


# ---------------------------------------------------------------------------
# 具体子类已分离到独立模块：
#   - lerobot_openloong_storage.py  → OpenLoongLeRobotStorage
#   - lerobot_tiangong_storage.py  → Tiangong2LeRobotStorage
#
# 继承结构（组合形式，避免 MRO 陷阱）：
#   OpenLoongLeRobotStorage(LerobotDataStorage)
#       __init__ 内部自动创建 OpenLoongRobotProfile()
#   Tiangong2LeRobotStorage(LerobotDataStorage)
#       __init__ 内部自动创建 Tiangong2RobotProfile()
#
#   RobotProfile 提供 obs_callback / build_state / state_dim / state_names，
#   LerobotDataStorage 提供 collection_data / save_data / clear_data
#   （格式相关写入逻辑）。两者通过组合注入，职责正交。
# ---------------------------------------------------------------------------
