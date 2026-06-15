"""collect_lerobot.py — 直采 LeRobot v2.1 数据集（构造时对齐 + 后台异步编码）。

设计要点（构造时对齐，alignment by construction）：
    唯一主时钟 = 仿真时间。采集循环里按 env.data.time 跨过 k/fps 边界时，
    才把「同一拍」的 state + 3 路相机图像一起抓下来缓存；episode 成功后逐帧写入
    LeRobotDataset。state、图像、action 来自同一仿真时刻，零对齐误差。

    - 图像：CameraWrapper websocket 内存流 get_frame() 直取最新帧（~ms 级，不落盘）。
      参照 OrcaGym/examples/so101/so101_leader_sim_record.py 已验证的取图路径。
    - action = state[i+1]（next-step 移位，沿用现有 replay/训练约定）。
    - 边采边保存（仿照 so101 lerobot_sim_record）：
        * PNG 写盘走异步 image_writer（image_writer_threads>0），add_frame 不阻塞主循环。
        * 每集仅同步落 parquet + 元数据（save_episode_data_only），立即进入下一集。
        * av1 视频编码（~2min/集）丢到后台单线程（BackgroundVideoEncoder）并行执行。
        * 退出前 VideoEncodingManager 收尾并清理临时 PNG 目录。

State/Action 16 维：
    0-2   左臂末端位置 (x,y,z)         B 系
    3-6   左臂末端四元数 (x,y,z,w)      B 系
    7-9   右臂末端位置 (x,y,z)
    10-13 右臂末端四元数 (x,y,z,w)
    14    左夹爪 [0,1]（归一化自 [0,255]）
    15    右夹爪 [0,1]

运行环境需同时含 orca_gym 与 lerobot 0.3.x（如 conda env: orcalab_lerobot）。

用法：
  python collect_lerobot.py \\
      --pose_file pose_mp_no_barcode.yaml \\
      --rand_file rand_no_barcode.yaml \\
      --lerobot_out /path/to/out_dataset \\
      --repo_id hangzhou2026/competition_warehouse_quat \\
      --episodes 50 --fps 30

  # 断点续采（追加到已有数据集）
  python collect_lerobot.py ... --lerobot_out <已有目录> --repo_id <同名> --resume

  # 可选：记录 C12C 随机位姿供回放复原（--record-scene）
  # 可选：逐帧诊断日志（--verbose）
"""

import argparse
import json
import os
import sys
import shutil
import time
import logging
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import cv2

base_dir = os.path.dirname(os.path.realpath(__file__))

# ── LeRobot 数据集布局常量 ──────────────────────────────────────────────────────
EE_NAMES = [
    "l_pos_x", "l_pos_y", "l_pos_z",
    "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
    "r_pos_x", "r_pos_y", "r_pos_z",
    "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
    "l_gripper", "r_gripper",
]
STATE_DIM = 16
GRIPPER_MAX = 255.0
CAMERAS = ["cam_head", "cam_wrist_l", "cam_wrist_r"]
ENV_CAM_TO_KEY = {
    "camera_head_color": "cam_head",
    "camera_wrist_l_color": "cam_wrist_l",
    "camera_wrist_r_color": "cam_wrist_r",
}
CAMERA_PORTS = {
    "camera_head_color": 7070,
    "camera_wrist_l_color": 7080,
    "camera_wrist_r_color": 7090,
}
ROBOT_TYPE = "d12_waist"
TASK_DEFAULT = "pick up the electronic component and place it into the box"
DEFAULT_HW = (480, 640)
STREAM_TRIGGER_PATH = "/tmp/collect_lerobot_stream"


def setup_cameras() -> dict:
    """启动三路相机 WebSocket 内存流，返回 {env_cam_name: CameraWrapper}。"""
    from orca_gym.sensor.rgbd_camera import CameraWrapper
    cameras = {}
    for name, port in CAMERA_PORTS.items():
        try:
            cam = CameraWrapper(name=name, port=port)
            cam.start()
            cameras[name] = cam
            print(f"✓ 相机 {name} 已启动（端口 {port}）", flush=True)
        except Exception as e:
            print(f"✗ 相机 {name} 启动失败（端口 {port}）: {e}", flush=True)
    return cameras


def wait_for_cameras(cameras: dict, timeout: float = 30.0) -> None:
    """等待所有相机收到首帧（最长 timeout 秒）。"""
    print(f"等待相机首帧就绪（最长 {timeout:.0f}s）...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        pending = [n for n, c in cameras.items() if not c.is_first_frame_received()]
        if not pending:
            print("✓ 所有相机首帧已就绪", flush=True)
            return
        print(f"  等待: {pending}", flush=True)
        time.sleep(1.0)
    print("⚠️  超时：部分相机首帧未就绪，继续运行", flush=True)


def close_cameras(cameras: dict) -> None:
    """停止相机后台线程。"""
    for cam in cameras.values():
        try:
            cam.running = False
        except Exception:
            pass
    for cam in cameras.values():
        thread = getattr(cam, "thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


def capture_frame_images(cameras: dict, target_hw) -> dict:
    """从内存流直取当前三路相机最新 RGB 帧，返回 {cam_key: (H,W,C) uint8}。"""
    H, W = target_hw
    images = {}
    for env_name, key in ENV_CAM_TO_KEY.items():
        cam = cameras[env_name]
        frame, _ = cam.get_frame(format="rgb24")
        if frame.shape[0] != H or frame.shape[1] != W:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        images[key] = np.ascontiguousarray(frame, dtype=np.uint8)
    return images


def probe_camera_hw(cameras: dict):
    """从首帧探测相机真实分辨率 (H,W)，失败回退 DEFAULT_HW。"""
    try:
        cam = cameras["camera_head_color"]
        frame, _ = cam.get_frame(format="rgb24")
        if frame is not None and frame.ndim == 3 and frame.size > 0:
            return (int(frame.shape[0]), int(frame.shape[1]))
    except Exception as e:
        print(f"[WARN] 相机分辨率探测失败，回退 {DEFAULT_HW}: {e}")
    return DEFAULT_HW


def _import_lerobot_dataset():
    last_err = None
    for mod in ("lerobot.datasets.lerobot_dataset",
                "lerobot.common.datasets.lerobot_dataset"):
        try:
            import importlib
            return importlib.import_module(mod).LeRobotDataset
        except Exception as e:
            last_err = e
    raise ImportError(
        "无法导入 lerobot 的 LeRobotDataset。请在同时装有 orca_gym 与 lerobot "
        f"的环境中运行（如 conda env orcalab_lerobot）。原始错误: {last_err}"
    )


def _import_video_encoding_manager():
    last_err = None
    for mod in ("lerobot.datasets.video_utils",
                "lerobot.common.datasets.video_utils"):
        try:
            import importlib
            return importlib.import_module(mod).VideoEncodingManager
        except Exception as e:
            last_err = e
    raise ImportError(f"无法导入 lerobot 的 VideoEncodingManager。原始错误: {last_err}")


class BackgroundVideoEncoder:
    """单 worker 线程顺序编码各集视频，主线程不阻塞。"""

    def __init__(self, dataset) -> None:
        self._dataset = dataset
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bg_video_enc")
        self._futures: list[Future] = []
        self._ep0_done = False

    def submit(self, episode_index: int) -> None:
        self._futures.append(self._executor.submit(self._encode, episode_index))

    def _encode(self, episode_index: int) -> None:
        try:
            logging.info(f"⏳ [后台编码] Episode {episode_index} 开始编码…")
            t0 = time.perf_counter()
            self._dataset.encode_episode_videos(episode_index)
            logging.info(f"✓  [后台编码] Episode {episode_index} 完成（{time.perf_counter()-t0:.1f}s）")
        except Exception:
            logging.exception(f"✗  [后台编码] Episode {episode_index} 编码失败")

    def ensure_ep0_done(self) -> None:
        if self._ep0_done:
            return
        if self._futures:
            try:
                self._futures[0].result()
            except Exception:
                pass
        self._ep0_done = True

    def wait_all(self) -> None:
        pending = sum(1 for f in self._futures if not f.done())
        if pending:
            logging.info(f"⏳ 等待 {pending} 个后台视频编码任务完成，请勿关闭程序…"
                         "（再次按 Ctrl+C 可强制跳过）")
        interrupted = False
        for future in self._futures:
            if interrupted:
                future.cancel()
                continue
            try:
                future.result()
            except KeyboardInterrupt:
                logging.warning("⚠  强制跳过剩余编码任务，部分视频可能不完整")
                self._executor.shutdown(wait=False, cancel_futures=True)
                interrupted = True
            except Exception:
                logging.exception("后台编码任务异常")
        if not interrupted:
            self._executor.shutdown(wait=False)
            if self._futures:
                logging.info("✓  所有后台视频编码任务已完成")


def create_empty_dataset(repo_id: str, root: str, fps: int, cam_shape,
                         mode: Literal["video", "image"] = "video"):
    LeRobotDataset = _import_lerobot_dataset()
    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": [EE_NAMES]},
        "action": {"dtype": "float32", "shape": (STATE_DIM,), "names": [EE_NAMES]},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": mode, "shape": cam_shape,
            "names": ["channels", "height", "width"],
        }
    if root and Path(root).exists():
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        robot_type=ROBOT_TYPE,
        features=features,
        root=root,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=0,
        image_writer_threads=4 * len(CAMERAS),
    )


def load_existing_dataset(repo_id: str, root: str):
    LeRobotDataset = _import_lerobot_dataset()
    dataset = LeRobotDataset(
        repo_id=repo_id, root=root,
        download_videos=False, tolerance_s=0.0001,
    )
    dataset.start_image_writer(0, 4 * len(CAMERAS))
    dataset.episode_buffer = dataset.create_episode_buffer()
    print(f"[resume] 已加载 {dataset.num_episodes} 集 / {dataset.num_frames} 帧 (root={root})")
    return dataset


def _make_sim_sync_storage(D12DataStorage):
    """工厂：构造按仿真时钟 fps 门控同步采集的存储子类。"""

    class D12SimSyncStorage(D12DataStorage):
        def configure(self, fps: float, cameras: dict, target_hw,
                      mp_mod=None, agent_conf=None, c12c_name="C12C_3c_c12c",
                      verbose: bool = False):
            self.fps = float(fps)
            self.cameras = cameras
            self.target_hw = target_hw
            self.frames = []
            self._next_cap = None
            self._mp = mp_mod
            self._agent_conf = agent_conf
            self._c12c_name = c12c_name
            self._verbose = verbose
            self._cap_idx = 0
            self._base_body = None
            self._c12c_body = None

        def begin_episode(self):
            self.frames = []
            self._next_cap = None
            self._cap_idx = 0

        def _resolve_diag_bodies(self, env):
            if self._mp is None or self._agent_conf is None:
                return
            try:
                self._base_body = env.body(self._agent_conf.base_body)
                self._c12c_body = self._mp._resolve_body_name(env, self._c12c_name)
            except Exception as e:
                print(f"[WARN] C12C 诊断初始化失败: {e}", flush=True)
                self._mp = None

        def _query_c12c_in_base(self, env):
            if self._mp is None or self._base_body is None or self._c12c_body is None:
                return None
            try:
                pos_b, _ = self._mp._query_body_pose_B(env, self._c12c_body, self._base_body)
                return np.asarray(pos_b, dtype=np.float64).reshape(3)
            except Exception:
                return None

        def collection_data(self, data: dict, env, **kwargs):
            pos = np.asarray(data["/action/end/position"], dtype=np.float32)
            quat = np.asarray(data["/action/end/orientation"], dtype=np.float32)
            motor = np.asarray(data["/action/effector/motor"], dtype=np.float32).flatten()
            state = np.concatenate([
                pos[0], quat[0], pos[1], quat[1],
                np.array([motor[0] / GRIPPER_MAX, motor[1] / GRIPPER_MAX], dtype=np.float32),
            ]).astype(np.float32)

            t = float(env.data.time)
            if self._next_cap is None:
                self._next_cap = t
            if t + 1e-9 >= self._next_cap:
                images = capture_frame_images(self.cameras, self.target_hw)
                self.frames.append((state, images))
                self._next_cap += 1.0 / self.fps

                if self._verbose:
                    if self._cap_idx == 0:
                        self._resolve_diag_bodies(env)
                    l_pos, r_pos = state[0:3], state[7:10]
                    l_quat = state[3:7]
                    gl, gr = float(motor[0]), float(motor[1])
                    c12c_b = self._query_c12c_in_base(env)
                    if c12c_b is not None:
                        d_lc = float(np.linalg.norm(l_pos.astype(np.float64) - c12c_b))
                        d_rc = float(np.linalg.norm(r_pos.astype(np.float64) - c12c_b))
                        c12c_str = (f"  C12C_B=[{c12c_b[0]:+.3f},{c12c_b[1]:+.3f},{c12c_b[2]:+.3f}]"
                                    f"  |L-C12C|={d_lc:.3f}  |R-C12C|={d_rc:.3f}")
                    else:
                        c12c_str = ""
                    print(
                        f"[采集] f={self._cap_idx:04d}  t={t:.3f}  "
                        f"L_pos=[{l_pos[0]:+.3f},{l_pos[1]:+.3f},{l_pos[2]:+.3f}]  "
                        f"R_pos=[{r_pos[0]:+.3f},{r_pos[1]:+.3f},{r_pos[2]:+.3f}]  "
                        f"L_quat=[{l_quat[0]:+.3f},{l_quat[1]:+.3f},{l_quat[2]:+.3f},{l_quat[3]:+.3f}]  "
                        f"grip(L,R)=({gl:.0f},{gr:.0f}){c12c_str}",
                        flush=True,
                    )
                self._cap_idx += 1

    return D12SimSyncStorage


def _add_episode_frames(dataset, frames, task: str) -> int:
    """next-step 移位写入帧到 episode_buffer，返回写入帧数。"""
    num = len(frames)
    for i in range(num - 1):
        state_i, images_i = frames[i]
        state_next, _ = frames[i + 1]
        frame = {
            "observation.state": state_i.astype(np.float32),
            "action": state_next.astype(np.float32),
        }
        for cam in CAMERAS:
            frame[f"observation.images.{cam}"] = images_i[cam]
        frame["task"] = task
        dataset.add_frame(frame)
    return num - 1


def _record_scene_c12c_pose(mp, env) -> dict | None:
    """查询随机化后 C12C 自由关节位姿，供 --record-scene 落盘。"""
    try:
        c12c_body = mp._resolve_body_name(env, "C12C_3c_c12c")
        c12c_joint = mp._find_free_joint_for_body(env, c12c_body)
        qpos = np.asarray(
            env.query_joint_qpos([c12c_joint])[c12c_joint], dtype=np.float64
        ).reshape(-1).tolist()
        return {"joint": c12c_joint, "qpos": qpos, "body": c12c_body}
    except Exception as e:
        print(f"[WARN] 记录 C12C 位姿失败: {e}")
        return None


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(">>> collect_lerobot 启动，正在初始化环境（加载模型约需数秒）...", flush=True)

    parser = argparse.ArgumentParser(
        description="直采 LeRobot v2.1 数据集（构造时对齐 + 后台异步编码，末端四元数 16 维）。"
    )
    parser.add_argument("--pose_file", type=str, default="pose_mp_no_barcode.yaml")
    parser.add_argument("--rand_file", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1, help="目标成功集数")
    parser.add_argument("--level", type=str, default="competition_warehouse")
    parser.add_argument("--task_config", type=str, default="competition_warehouse.yaml")
    parser.add_argument("--lerobot_out", type=str, required=True,
                        help="LeRobot 数据集输出根目录")
    parser.add_argument("--repo_id", type=str, default="hangzhou2026/competition_warehouse_quat",
                        help="LeRobot repo_id")
    parser.add_argument("--task", type=str, default=TASK_DEFAULT, help="任务语言描述")
    parser.add_argument("--fps", type=int, default=30, help="采集帧率（仿真时间），默认 30")
    parser.add_argument("--resume", action="store_true",
                        help="追加到 --lerobot_out 指向的已有数据集")
    parser.add_argument("--record-scene", action="store_true",
                        help="记录每集 C12C 随机位姿到 c12c_poses.json，供回放 --restore-scene 使用")
    parser.add_argument("--verbose", action="store_true",
                        help="打印逐帧采集诊断日志（默认关闭）")
    args = parser.parse_args()

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    import data_collection_mp as mp
    import run_collection as rc
    from conf import d12_conf as agent_conf
    from dataStorage.d12_data_storage import D12DataStorage

    VideoEncodingManager = _import_video_encoding_manager()

    lerobot_out = os.path.abspath(os.path.expanduser(args.lerobot_out))

    inner_args = argparse.Namespace(
        level=args.level, task_config=args.task_config,
        pose_file=args.pose_file, rand_file=args.rand_file,
        record_data=True, episodes=args.episodes,
        steps=None, delta_b=None, l_target_b=None, r_target_b=None,
        l_quat_b=None, r_quat_b=None, gripper_open=None, gripper_close=None,
        dump_pose=None, resolve_pose_only=False,
    )

    spec = mp.load_pose_spec_from_file(os.path.join(base_dir, args.pose_file)) if args.pose_file else {}
    rand_spec = mp.load_yaml_dict(os.path.join(base_dir, args.rand_file)) if args.rand_file else {}
    steps, g_open, g_close = mp._resolve_trajectory_args(inner_args, spec)

    manager, env, l_arm, r_arm, l_grip, r_grip, task_status = \
        mp.create_manager_and_controllers(inner_args, agent_conf)

    StorageCls = _make_sim_sync_storage(D12DataStorage)
    storage = StorageCls(
        dataset_path=os.path.join(base_dir, "dataset", "_sim_sync_scratch"),
        hdf5_path="record/proprio_stats.hdf5",
    )
    manager.set_data_storage(storage)
    manager.save_video = False

    gidx = rc._build_grasp_index(env, mp)

    cameras = {}
    cam_hw = DEFAULT_HW
    try:
        env.reset()
        if manager.update_scene():
            os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
            env.begin_save_video(STREAM_TRIGGER_PATH)
            print("✓ begin_save_video 已调用，相机推流已触发", flush=True)
            cameras = setup_cameras()
            wait_for_cameras(cameras)
            cam_hw = probe_camera_hw(cameras)
    except Exception as e:
        print(f"[WARN] 相机初始化阶段异常，使用 {DEFAULT_HW}: {e}")
    if not cameras:
        print("✗ 没有可用相机，退出", flush=True)
        env.close()
        return
    storage.configure(
        fps=args.fps, cameras=cameras, target_hw=cam_hw,
        mp_mod=mp if args.verbose else None,
        agent_conf=agent_conf if args.verbose else None,
        verbose=args.verbose,
    )
    cam_shape = (3, cam_hw[0], cam_hw[1])
    print(f"[INFO] 相机分辨率 = {cam_hw[0]}x{cam_hw[1]}，fps = {args.fps}", flush=True)

    if args.resume:
        if not Path(lerobot_out).exists():
            print(f"[ERROR] --resume 需要已存在的 --lerobot_out: {lerobot_out}")
            return
        dataset = load_existing_dataset(args.repo_id, lerobot_out)
    else:
        dataset = create_empty_dataset(args.repo_id, lerobot_out, args.fps, cam_shape)
    print(f"[INFO] LeRobot 数据集输出: {lerobot_out}  (repo_id={args.repo_id})")

    print(f">>> 初始化完成，开始采集，目标成功集数 = {args.episodes}", flush=True)

    n_success = 0
    episode_index = 0
    target_episodes = args.episodes
    manager._shutdown_requested = False

    c12c_poses = {}
    c12c_poses_path = os.path.join(lerobot_out, "c12c_poses.json")

    try:
        with VideoEncodingManager(dataset):
            bg_encoder = BackgroundVideoEncoder(dataset)
            try:
                while n_success < target_episodes:
                    if manager._shutdown_requested:
                        print("\n[STOP] 收到中断信号，停止采集。")
                        break
                    attempt_label = f"Episode {n_success + 1}/{target_episodes} (seed={episode_index})"
                    print(f"\n=== {attempt_label} ===")

                    env.reset()
                    if not manager.update_scene():
                        print("update_scene failed, exit")
                        break

                    try:
                        c12c_body = mp._resolve_body_name(env, "C12C_3c_c12c")
                        c12c_joint = mp._find_free_joint_for_body(env, c12c_body)
                        env.set_joint_qpos({c12c_joint: np.array(rc.C12C_DEFAULT_QPOS, dtype=np.float64)})
                        env.mj_forward()
                    except Exception as e:
                        print(f"[WARN] C12C 复位默认位失败: {e}")

                    episode_rand_spec = mp.advance_rand_spec_seed(rand_spec, episode_index)
                    if episode_rand_spec:
                        mp.apply_object_randomization(env, episode_rand_spec)

                    scene_c12c_pose = (
                        _record_scene_c12c_pose(mp, env) if args.record_scene else None
                    )

                    rc._print_episode_init(env, agent_conf)

                    resolved_spec = mp.resolve_pose_spec_for_current_scene(env, agent_conf, spec)
                    l_pos, l_quat, r_pos, r_quat, l_gm, r_gm = mp.build_trajectory_from_resolved_spec(
                        env, agent_conf, inner_args, resolved_spec, g_open, g_close, steps
                    )

                    device = rc._make_grasp_device(
                        mp, env, gidx,
                        l_arm=l_arm, r_arm=r_arm, l_grip=l_grip, r_grip=r_grip,
                        task_status=task_status,
                        l_pos=l_pos, l_quat_xyzw=l_quat,
                        r_pos=r_pos, r_quat_xyzw=r_quat,
                        l_grip_motor=l_gm, r_grip_motor=r_gm,
                    )
                    manager.set_device(device)
                    storage.begin_episode()
                    manager.run_episode()

                    if manager._shutdown_requested:
                        print("\n[STOP] 收到中断信号，丢弃当前集并停止采集。")
                        break

                    grasp_ok = device.grasp_confirmed
                    episode_index += 1

                    if grasp_ok:
                        frames = storage.frames
                        if len(frames) < 2:
                            print(f"[!] {attempt_label} 抓取确认但采样帧不足({len(frames)})，丢弃换 seed 重试")
                            dataset.clear_episode_buffer()
                            continue

                        if dataset.num_episodes > 0:
                            bg_encoder.ensure_ep0_done()

                        n_frames = _add_episode_frames(dataset, frames, args.task)
                        ep_idx = dataset.num_episodes
                        dataset.save_episode()
                        bg_encoder.submit(ep_idx)

                        if args.record_scene and scene_c12c_pose is not None:
                            c12c_poses[str(int(ep_idx))] = scene_c12c_pose
                            try:
                                with open(c12c_poses_path, "w", encoding="utf-8") as f:
                                    json.dump(c12c_poses, f, indent=2)
                                print(f"[rec] C12C 位姿已记录 → {c12c_poses_path} (ep {int(ep_idx)})")
                            except Exception as e:
                                print(f"[WARN] 写 c12c_poses.json 失败: {e}")

                        n_success += 1
                        print(f"[✓] {attempt_label} 抓取成功，已写入 LeRobot {n_frames} 帧 "
                              f"（视频编码后台进行中，Episode {ep_idx}）"
                              f"({n_success}/{target_episodes})")
                    else:
                        peak_mm = getattr(device, "_peak_lift", 0.0) * 1000
                        print(f"[✗] {attempt_label} 抓取失败（峰值抬升仅 {peak_mm:.0f}mm，未达 "
                              f"{rc.GRASP_LIFT_THRESHOLD * 1000:.0f}mm），数据已丢弃，换 seed 重试")

                print(f"\n全部采集完成: 成功 {n_success}/{target_episodes}")
            finally:
                bg_encoder.wait_all()

        print(f"[INFO] LeRobot 数据集: {lerobot_out}  共 {dataset.num_episodes} 集 / {dataset.num_frames} 帧")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt, stopping.")
    except Exception as e:
        print(f"Error: {e}\n{traceback.format_exc()}")
    finally:
        try:
            if getattr(dataset, "image_writer", None) is not None:
                dataset.stop_image_writer()
        except Exception:
            pass
        try:
            env.stop_save_video()
        except Exception:
            pass
        close_cameras(cameras)
        env.close()


if __name__ == "__main__":
    main()
