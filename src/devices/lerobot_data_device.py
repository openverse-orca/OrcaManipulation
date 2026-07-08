"""LeRobotDataDevice — 从 LeRobot v2.1 parquet 数据集提供回放数据。

继承 DataDevice，重写 load_unit_dataset() / load_data() / update()：
- load_unit_dataset / load_data：把 parquet 16 维 action 重整为 HDF5-layout 嵌套字典；
- update()：每帧保持 steps_per_frame 个控制步（帧保持机制），解决原版 1:1 导致
  OSC 无法收敛（每帧只有 5ms）的问题；同时保留 task_status 触发语义。
  bind_dataset_event() / bind_task_status_event() 继续复用父类实现。

parquet 16 维布局（与 OpenLoongLeRobotStorage.build_state 对称）：
    [0:3]   l_pos_b       左臂末端位置（基座系，米）
    [3:7]   l_quat_b      左臂末端四元数 xyzw
    [7:10]  r_pos_b       右臂末端位置
    [10:14] r_quat_b      右臂末端四元数 xyzw
    [14]    l_gripper     左夹爪归一化 [0,1]
    [15]    r_gripper     右夹爪归一化 [0,1]

重整后的 HDF5-layout（与 add_arm_osc_openloong_data_controller 绑定路径一致）：
    /action/end/position    每帧 (6,) = [l_pos(3), r_pos(3)]
    /action/end/orientation 每帧 (8,) = [l_quat(4), r_quat(4)]
    /action/effector/motor  每帧 (2,) = [l_motor, r_motor]（反归一化回电机量程）

运行环境：orcalab_lerobot（含 pyarrow）
"""
import json
import os

import numpy as np
from typing_extensions import override

from devices.data_device import DataDevice
from orca_gym.log.orca_log import OrcaLog

orca_logger = OrcaLog.get_instance()


class LeRobotDataDevice(DataDevice):
    """从 LeRobot v2.1 parquet 数据集提供 DataDevice 兼容的事件驱动回放数据。

    参数
    ----
    dataset_dir : str
        LeRobot 数据集根目录（含 data/ meta/ videos/ 子目录）。
    agent_conf : module
        机器人配置模块（openloong_conf），用于读取夹爪量程做反归一化。
    loop_playback : bool
        True 时播完全部 episode 后循环重头播放。
    """

    def __init__(
        self,
        dataset_dir: str,
        agent_conf,
        loop_playback: bool = False,
        steps_per_frame: int = 20,
    ) -> None:
        # 必须在 super().__init__() **之前** 设置子类属性，
        # 因为 super().__init__() 会调用 self.load_unit_dataset()（多态）。
        self._dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
        self._l_grip_max = float(agent_conf.gripper_l["actuator_ranges"][0][1])
        self._r_grip_max = float(agent_conf.gripper_r["actuator_ranges"][0][1])
        # 帧保持：每帧重复 steps_per_frame 个控制步，给 OSC 充分收敛时间
        # 采集 30fps → frame_dt=33.3ms；env.dt=5ms；steps_per_frame≈7 为实时，10 为默认（约1.5x实时）
        self._steps_per_frame: int = max(1, int(steps_per_frame))
        self._play_step: int = 0      # 当前集已执行的控制步数
        self._episode_len: int = 0    # 当前集总帧数（load_data 时更新）

        # 用 dataset_dir 作为 dataset_path；hdf5_path=None（我们重写 load_data，不走 HDF5）
        super().__init__(
            dataset_path=self._dataset_dir,
            hdf5_path=None,
            interpolator=None,
            loop_playback=loop_playback,
        )

    # ------------------------------------------------------------------
    # 重写：扫描 parquet 文件，建待回放队列
    # ------------------------------------------------------------------

    @override
    def load_unit_dataset(self) -> None:
        """扫描 dataset_dir/data/chunk-*/episode_*.parquet，排序后建待回放队列。"""
        self.unit_datasets_path = []

        data_dir = os.path.join(self._dataset_dir, "data")
        if not os.path.isdir(data_dir):
            orca_logger.warning(
                f"[LeRobotDataDevice] data 目录不存在: {data_dir}"
            )
            self._all_unit_datasets_path = []
            return

        parquet_files: list[str] = []
        for chunk_name in sorted(os.listdir(data_dir)):
            chunk_path = os.path.join(data_dir, chunk_name)
            if not os.path.isdir(chunk_path):
                continue
            for fname in sorted(os.listdir(chunk_path)):
                if fname.startswith("episode_") and fname.endswith(".parquet"):
                    parquet_files.append(os.path.join(chunk_path, fname))

        self.unit_datasets_path = parquet_files
        self._all_unit_datasets_path = list(parquet_files)

        # 读取 meta/info.json 输出统计
        info_path = os.path.join(self._dataset_dir, "meta", "info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, encoding="utf-8") as f:
                    info = json.load(f)
                total_ep = info.get("total_episodes", "?")
                total_fr = info.get("total_frames", "?")
                orca_logger.info(
                    f"[LeRobotDataDevice] 数据集: {self._dataset_dir}  "
                    f"total_episodes={total_ep}  total_frames={total_fr}  "
                    f"已扫描 {len(parquet_files)} 个 parquet 文件"
                )
            except Exception as e:
                orca_logger.warning(f"[LeRobotDataDevice] 读取 meta/info.json 失败: {e}")

    # ------------------------------------------------------------------
    # 重写：加载单个 parquet → 重整为 HDF5-layout self.data
    # ------------------------------------------------------------------

    @override
    def load_data(self) -> bool:
        """弹出下一个 parquet，重整为 DataDevice HDF5-layout 格式。

        返回 False 表示队列已空（播完所有 episode 或 loop_playback 被中断）。
        """
        if not self.unit_datasets_path:
            if self.loop_playback and self._all_unit_datasets_path:
                self._reset_unit_dataset_queue()
                orca_logger.info("[LeRobotDataDevice] 数据集已播完，从头循环")
            else:
                self.data = None
                self.current_unit_path = None
                return False

        # 正向播放（pop(0)），与 sort 后的 episode 顺序一致
        parquet_path = self.unit_datasets_path.pop(0)
        self.current_unit_path = parquet_path

        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "需要 pyarrow 读取 LeRobot parquet 文件。"
                "请在 orcalab_lerobot 环境中运行。"
            ) from e

        table = pq.read_table(parquet_path)
        actions = np.array(table["action"].to_pylist(), dtype=np.float32)  # (N, 16)

        if actions.ndim != 2 or actions.shape[1] < 16:
            raise ValueError(
                f"[LeRobotDataDevice] action 形状异常: {actions.shape}，期望 (N, >=16)  "
                f"文件: {parquet_path}"
            )

        n_frames = len(actions)

        # ── 重整为 HDF5-layout ──────────────────────────────────────────
        # /action/end/position:    (N, 6) = [l_pos(3), r_pos(3)]
        positions = np.concatenate(
            [actions[:, 0:3], actions[:, 7:10]], axis=1
        )  # (N, 6)

        # /action/end/orientation: (N, 8) = [l_quat(4), r_quat(4)]
        orientations = np.concatenate(
            [actions[:, 3:7], actions[:, 10:14]], axis=1
        )  # (N, 8)

        # /action/effector/motor:  (N, 2) = 反归一化回电机量程
        l_motor = np.clip(actions[:, 14:15], 0.0, 1.0) * self._l_grip_max
        r_motor = np.clip(actions[:, 15:16], 0.0, 1.0) * self._r_grip_max
        motors = np.concatenate([l_motor, r_motor], axis=1)  # (N, 2)

        # DataDevice.get_data() 按 "/" 分割路径逐层索引，每层是 dict 或 list
        # list[cursor] 取当前帧，raw_data.flatten()[i:j] 取子段
        self.data = {
            "action": {
                "end": {
                    "position": list(positions),      # list of (6,) float32
                    "orientation": list(orientations),  # list of (8,) float32
                },
                "effector": {
                    "motor": list(motors),  # list of (2,) float32
                },
            },
        }

        # 与 DataDevice.load_data() 末尾行为一致
        self.task_info = {}
        self.scene_info = {}
        self.dataset_cursor = {path: 0 for path in self.dataset_event.keys()}
        self.update_task_status = True  # 触发第一步 NOT_STARTED → RUNNING

        # 帧保持计数器复位
        self._play_step = 0
        self._episode_len = n_frames

        ep_name = os.path.basename(parquet_path)
        remaining = len(self.unit_datasets_path)
        total_ctrl_steps = n_frames * self._steps_per_frame
        orca_logger.info(
            f"[LeRobotDataDevice] 加载 {ep_name}  ({n_frames} 帧)  "
            f"steps_per_frame={self._steps_per_frame}  "
            f"预计控制步={total_ctrl_steps}  预计仿真时长={total_ctrl_steps * 0.005:.1f}s  "
            f"剩余队列 {remaining} 集"
        )
        return True

    # ------------------------------------------------------------------
    # 重写 update()：帧保持机制（每帧重复 steps_per_frame 个控制步）
    # ------------------------------------------------------------------

    @override
    def update(self) -> bool:
        """每帧保持 steps_per_frame 个控制步，给 OSC 充分收敛时间。

        任务状态机对齐父类语义：
          - 首步（_play_step==0）：帧 0 派发后触发 NOT_STARTED → RUNNING
          - 末步（frame >= _episode_len）：触发 RUNNING → END，data=None
        """
        if self.data is None:
            return True

        frame = self._play_step // self._steps_per_frame

        # ── 集结束：触发 RUNNING → END ─────────────────────────────────
        if frame >= self._episode_len:
            if self.task_status_event is not None:
                self.task_status_event(True)
            self.data = None
            orca_logger.info(
                f"[LeRobotReplay] Episode 播完 "
                f"({self._episode_len} 帧 × {self._steps_per_frame} 步 "
                f"= {self._play_step} 控制步)"
            )
            return True

        # ── 每帧首步打印关键日志 ────────────────────────────────────────
        if self._play_step % self._steps_per_frame == 0:
            self._log_frame(frame)

        # ── 派发当前帧到所有绑定事件 ────────────────────────────────────
        for dataset_path, events in self.dataset_event.items():
            seq = self.get_data(dataset_path)
            raw = seq[frame]
            for index, event in events:
                event(raw.flatten()[index[0]:index[1]])

        self._play_step += 1

        # ── 首步触发 NOT_STARTED → RUNNING（消费 load_data 设置的标志）──
        if self.update_task_status:
            if self.task_status_event is not None:
                self.task_status_event(True)
            self.update_task_status = False

        return True

    def _log_frame(self, frame: int) -> None:
        """每帧首步打印末端位置与夹爪状态，便于调试。"""
        try:
            pos_seq = self.get_data("/action/end/position")
            motor_seq = self.get_data("/action/effector/motor")
            pos = pos_seq[frame].flatten()
            motor = motor_seq[frame].flatten()
            l_pos, r_pos = pos[0:3], pos[3:6]
            l_grip, r_grip = float(motor[0]), float(motor[1])
            orca_logger.info(
                f"[LeRobotReplay] frame={frame:04d}/{self._episode_len}  "
                f"step={self._play_step}  "
                f"L=[{l_pos[0]:+.3f},{l_pos[1]:+.3f},{l_pos[2]:+.3f}]  "
                f"R=[{r_pos[0]:+.3f},{r_pos[1]:+.3f},{r_pos[2]:+.3f}]  "
                f"grip(L,R)=({l_grip:.0f},{r_grip:.0f})"
            )
        except Exception:
            orca_logger.info(
                f"[LeRobotReplay] frame={frame:04d}/{self._episode_len}  step={self._play_step}"
            )

    # ------------------------------------------------------------------
    # 单集模式：裁剪队列为仅指定集
    # ------------------------------------------------------------------

    def select_episode(self, episode_order: int) -> None:
        """裁剪待回放队列为仅第 episode_order 集（1-indexed）。

        同时更新 _all_unit_datasets_path，使 loop_playback 也只循环该集。
        """
        total = len(self._all_unit_datasets_path)
        if total == 0:
            raise ValueError("[LeRobotDataDevice] 数据集为空，无法选取指定集")
        if not (1 <= episode_order <= total):
            raise ValueError(
                f"[LeRobotDataDevice] --episode {episode_order} 超出范围 (1..{total})"
            )
        ep_path = self._all_unit_datasets_path[episode_order - 1]
        self.unit_datasets_path = [ep_path]
        self._all_unit_datasets_path = [ep_path]
        orca_logger.info(
            f"[LeRobotDataDevice] 单集模式: episode {episode_order}/{total}  "
            f"→ {os.path.basename(ep_path)}"
        )
