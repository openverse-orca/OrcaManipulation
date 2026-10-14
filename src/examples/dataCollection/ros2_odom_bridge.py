"""
FreeJoint 里程计 ROS2 桥接工具 — 将 g1_omnipicker_free_joint 的位姿/速度发布为 ROS2 里程计

发布 Topic:
  /odom  — nav_msgs/Odometry (位姿 + 速度)
  TF     — odom -> base_link (默认开启, --no-tf 关闭)

背景:
  遥操脚本运行在 orcalab9 (Python 3.12) 环境, 而 ROS2 Humble 的 rclpy 只有
  Python 3.10 的 C 扩展 (/opt/ros/humble/.../_rclpy_pybind11.cpython-310-*.so),
  无法在 3.12 解释器中 import。因此本工具采用双进程架构:

    遥操进程 (orcalab9)                      桥接进程 (/usr/bin/python3 + ROS2)
    ┌────────────────────────┐    UDP JSON   ┌──────────────────────────────┐
    │ freejoint qpos/qvel    │ ────────────► │ nav_msgs/Odometry -> /odom     │
    │ (本地 MuJoCo 数据)      │  127.0.0.1   │ TF: odom -> base_link         │
    └────────────────────────┘               └──────────────────────────────┘

用法 (推荐, 遥操脚本中自动拉起桥接子进程):
    from ros2_odom_bridge import FreeJointOdomPublisher

    odom_publisher = FreeJointOdomPublisher()
    ...
    odom_publisher.publish_env_freejoint(env)   # 每个控制循环调用
    ...
    odom_publisher.close()

手动单独运行桥接进程 (效果相同):
    python3 ros2_odom_bridge.py --port 9870

依赖 (无需 conda 环境, 使用系统 python3):
  sudo apt install -y ros-humble-ros-base ros-humble-nav-msgs ros-humble-tf2-ros ros-humble-geometry-msgs

RViz2 验证:
  rviz2 -> Fixed Frame: odom -> Add -> Odometry -> Topic: /odom
  该话题/TF 可供 rtabmap 订阅, 作为建图的里程计输入
"""

import argparse
import glob
import json
import os
import socket
import subprocess
import sys
import threading
import time

DEFAULT_PORT = 9870
DEFAULT_FRAME_ID = "odom"
DEFAULT_CHILD_FRAME_ID = "base_link"
DEFAULT_JOINT_NAME = "g1_omnipicker_free_joint"


# ---------------------------------------------------------------------------
# ROS2 环境检测 (参考 demo.py)
# ---------------------------------------------------------------------------

def _detect_ros2_root():
    """扫描 /opt/ros/ 下已安装的 ROS2 发行版, 按字母序取首个有效安装。"""
    for candidate in sorted(glob.glob("/opt/ros/*")):
        if os.path.isfile(os.path.join(candidate, "setup.bash")):
            return candidate
    return None


def _detect_py_version(ros2_root):
    """从 ROS2 安装目录探测 Python 版本目录名 (如 python3.10)。"""
    for pattern in ("local/lib/python3.*", "lib/python3.*"):
        matches = sorted(glob.glob(os.path.join(ros2_root, pattern)))
        if matches:
            return os.path.basename(matches[0])
    return f"python{sys.version_info.major}.{sys.version_info.minor}"


def _ros2_env_paths(ros2_root):
    """返回 ROS2 python 包路径列表 (humble: local/lib/.../dist-packages + lib/.../site-packages)。"""
    py_ver = _detect_py_version(ros2_root)
    candidates = [
        f"{ros2_root}/local/lib/{py_ver}/dist-packages",
        f"{ros2_root}/lib/{py_ver}/site-packages",
    ]
    return [p for p in candidates if os.path.isdir(p)]


def _ensure_ros2_env():
    """确保当前进程能 import rclpy; 必要时通过 os.execv 重启自身。仅桥接模式使用。"""
    ros2_root = _detect_ros2_root()
    if ros2_root is None:
        print("[ERROR] No ROS2 installation found under /opt/ros/")
        sys.exit(1)

    py_paths = _ros2_env_paths(ros2_root)
    ros2_py_path = ":".join(py_paths)
    need_restart = False
    if ros2_py_path and ros2_py_path not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = ros2_py_path + ":" + os.environ.get("PYTHONPATH", "")
        need_restart = True
    if ros2_root + "/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = ros2_root + "/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
        need_restart = True
    # LD_LIBRARY_PATH 在进程启动时被 ld.so 缓存, 运行时修改 os.environ 不影响当前进程的 dlopen。
    # 通过 os.execv 重启自身, 让新进程以正确的环境变量启动。
    if need_restart and "_ROS2_ODOM_BRIDGE_RESTARTED" not in os.environ:
        os.environ["_ROS2_ODOM_BRIDGE_RESTARTED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)


# ---------------------------------------------------------------------------
# 四元数 / 向量工具 (纯 python, 无 numpy 依赖, 保证桥接进程可运行)
# ---------------------------------------------------------------------------

def _quat_wxyz_to_matrix(q):
    """四元数 (w,x,y,z) -> 3x3 旋转矩阵 (行嵌套元组)。"""
    w, x, y, z = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def _mat_vec(m, v):
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


# ---------------------------------------------------------------------------
# 发送端: 在遥操进程中使用 (orcalab9 / 任意 Python3)
# ---------------------------------------------------------------------------

class FreeJointOdomPublisher:
    """
    读取 freejoint 的 qpos/qvel, 转换为里程计数据并通过 UDP 发给桥接进程。

    首次构造时自动用系统 python3 (/usr/bin/python3) 拉起桥接子进程;
    若 ROS2 未安装则退化为 no-op, 不影响遥操主流程。
    桥接子进程通过 --parent-pid 监测父进程, 即使父进程 os._exit 也能自动退出。
    """

    def __init__(
        self,
        joint_name=DEFAULT_JOINT_NAME,
        host="127.0.0.1",
        port=DEFAULT_PORT,
        frame_id=DEFAULT_FRAME_ID,
        child_frame_id=DEFAULT_CHILD_FRAME_ID,
        spawn_bridge=True,
    ):
        self.joint_name = joint_name
        self.frame_id = frame_id
        self.child_frame_id = child_frame_id
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._bridge_proc = None
        self._closed = False
        self._warned = False

        if spawn_bridge:
            self._spawn_bridge(host, port)

    def _log(self, msg):
        print(f"[OdomPublisher] {msg}")

    def _spawn_bridge(self, host, port):
        ros2_root = _detect_ros2_root()
        if ros2_root is None:
            self._log("ROS2 not found under /opt/ros/, odometry publishing disabled")
            return
        bridge_python = "/usr/bin/python3"
        if not os.path.isfile(bridge_python):
            self._log(f"{bridge_python} not found, odometry publishing disabled")
            return
        env = os.environ.copy()
        # 给桥接进程预置 ROS2 环境变量, 避免其再次 execv 重启
        py_paths = _ros2_env_paths(ros2_root)
        if py_paths:
            env["PYTHONPATH"] = ":".join(py_paths)
        env["LD_LIBRARY_PATH"] = ros2_root + "/lib"
        self._bridge_proc = subprocess.Popen(
            [
                bridge_python,
                os.path.abspath(__file__),
                "--host", host,
                "--port", str(port),
                "--frame-id", self.frame_id,
                "--child-frame-id", self.child_frame_id,
                "--parent-pid", str(os.getpid()),
            ],
            env=env,
        )
        self._log(f"ROS2 odom bridge spawned (pid={self._bridge_proc.pid}, port={port})")

    def publish_env_freejoint(self, env):
        """从环境中查询 freejoint 的 qpos/qvel 并发布, 供每个控制循环调用。"""
        try:
            qpos = env.query_joint_qpos([self.joint_name]).get(self.joint_name)
            if qpos is None:
                self._warn_once(f"joint '{self.joint_name}' not found, skip odom")
                return
            qvel = env.query_joint_qvel([self.joint_name]).get(self.joint_name)
            self.publish_freejoint(qpos, qvel)
        except Exception as e:
            self._warn_once(f"publish_env_freejoint failed: {e}")

    def publish_freejoint(self, qpos, qvel=None):
        """
        发布 MuJoCo freejoint 状态。

        qpos: [x, y, z, qw, qx, qy, qz]  世界系位姿
        qvel: [vx, vy, vz, wx, wy, wz]
              线速度为世界系, 角速度为本体系 (MuJoCo freejoint 约定),
              此处统一转换到本体坐标系 (nav_msgs/Odometry 的 twist 定义在本体系)。
        """
        if self._closed:
            return
        pos = [float(v) for v in qpos[:3]]
        quat_wxyz = [float(v) for v in qpos[3:7]]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]

        lin_body = [0.0, 0.0, 0.0]
        ang_body = [0.0, 0.0, 0.0]
        if qvel is not None and len(qvel) >= 6:
            # 世界系线速度 -> 本体系: v_body = R^T * v_world
            rot = _quat_wxyz_to_matrix(quat_wxyz)
            rot_t = tuple(zip(*rot))
            lin_body = list(_mat_vec(rot_t, [float(v) for v in qvel[:3]]))
            ang_body = [float(v) for v in qvel[3:6]]  # MuJoCo 角速度已在本体系

        packet = {
            "pos": pos,
            "quat_xyzw": quat_xyzw,
            "lin": lin_body,
            "ang": ang_body,
        }
        try:
            self._sock.sendto(json.dumps(packet).encode("utf-8"), self._addr)
        except OSError as e:
            self._warn_once(f"sendto failed: {e}")

    def _warn_once(self, msg):
        if not self._warned:
            self._warned = True
            self._log(f"WARNING: {msg} (only logged once)")

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._sock.close()
        except OSError:
            pass
        if self._bridge_proc is not None:
            self._bridge_proc.terminate()
            try:
                self._bridge_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._bridge_proc.kill()
            self._bridge_proc = None
            self._log("ROS2 odom bridge stopped")


# ---------------------------------------------------------------------------
# 桥接端: 使用系统 python3 运行本文件
# ---------------------------------------------------------------------------

def _run_bridge(args):
    import rclpy
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import TransformBroadcaster

    rclpy.init()
    node = rclpy.create_node("freejoint_odom_bridge")
    odom_pub = node.create_publisher(Odometry, args.topic, 10)
    tf_broadcaster = None if args.no_tf else TransformBroadcaster(node)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((args.host, args.port))
    except OSError as e:
        node.get_logger().error(f"bind udp://{args.host}:{args.port} failed: {e}")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    sock.settimeout(0.5)

    node.get_logger().info(
        f"Odom bridge listening on udp://{args.host}:{args.port}, "
        f"publishing {args.topic} ({args.frame_id} -> {args.child_frame_id}), "
        f"tf={'off' if args.no_tf else 'on'}"
    )

    # 父进程存活监测: 父进程退出后自动退出, 兼容 os._exit 等不走 atexit 的场景
    def _parent_watchdog():
        while True:
            try:
                os.kill(args.parent_pid, 0)
            except OSError:
                node.get_logger().info("Parent process exited, shutting down bridge")
                sock.close()
                return
            time.sleep(1.0)

    if args.parent_pid is not None:
        threading.Thread(target=_parent_watchdog, daemon=True).start()

    # 里程计协方差 (对角线), 供 rtabmap 等下游参考
    cov = [0.0] * 36
    for i in (0, 7, 14, 21, 28, 35):
        cov[i] = 0.01

    frame_count = 0
    try:
        while True:
            try:
                data, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = json.loads(data.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue

            stamp = node.get_clock().now().to_msg()
            pos = packet["pos"]
            quat = packet["quat_xyzw"]
            lin = packet["lin"]
            ang = packet["ang"]

            msg = Odometry()
            msg.header.stamp = stamp
            msg.header.frame_id = args.frame_id
            msg.child_frame_id = args.child_frame_id
            msg.pose.pose.position.x = float(pos[0])
            msg.pose.pose.position.y = float(pos[1])
            msg.pose.pose.position.z = float(pos[2])
            msg.pose.pose.orientation.x = float(quat[0])
            msg.pose.pose.orientation.y = float(quat[1])
            msg.pose.pose.orientation.z = float(quat[2])
            msg.pose.pose.orientation.w = float(quat[3])
            msg.pose.covariance = cov[:]
            msg.twist.twist.linear.x = float(lin[0])
            msg.twist.twist.linear.y = float(lin[1])
            msg.twist.twist.linear.z = float(lin[2])
            msg.twist.twist.angular.x = float(ang[0])
            msg.twist.twist.angular.y = float(ang[1])
            msg.twist.twist.angular.z = float(ang[2])
            msg.twist.covariance = cov[:]
            odom_pub.publish(msg)

            if tf_broadcaster is not None:
                t = TransformStamped()
                t.header.stamp = stamp
                t.header.frame_id = args.frame_id
                t.child_frame_id = args.child_frame_id
                t.transform.translation.x = float(pos[0])
                t.transform.translation.y = float(pos[1])
                t.transform.translation.z = float(pos[2])
                t.transform.rotation.x = float(quat[0])
                t.transform.rotation.y = float(quat[1])
                t.transform.rotation.z = float(quat[2])
                t.transform.rotation.w = float(quat[3])
                tf_broadcaster.sendTransform(t)

            frame_count += 1
            if frame_count % 500 == 0:
                node.get_logger().info(f"Published {frame_count} odom messages")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sock.close()
        except OSError:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("[OdomBridge] exited")


def main():
    parser = argparse.ArgumentParser(description="FreeJoint Odometry ROS2 Bridge")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="UDP 监听地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="UDP 监听端口")
    parser.add_argument("--topic", type=str, default="/odom",
                        help="nav_msgs/Odometry 话题名")
    parser.add_argument("--frame-id", type=str, default=DEFAULT_FRAME_ID,
                        help="里程计父坐标系 (默认 odom)")
    parser.add_argument("--child-frame-id", type=str, default=DEFAULT_CHILD_FRAME_ID,
                        help="里程计子坐标系 (默认 base_link)")
    parser.add_argument("--no-tf", action="store_true",
                        help="不发布 odom -> base_link TF")
    parser.add_argument("--parent-pid", type=int, default=None,
                        help="父进程 PID, 父进程退出后桥接进程自动退出")
    args = parser.parse_args()

    _ensure_ros2_env()
    _run_bridge(args)


if __name__ == "__main__":
    main()
