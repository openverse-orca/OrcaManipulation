# 相机与视频编码

OrcaStudio 按端口推流相机。通用 API 在 `src/sensor/camera_stream.py`，NVENC 编码进程在 `src/sensor/encoder_proc.py`。

## 相机映射

映射形状为 `{环境相机名: (数据集键, 端口)}`。端口与环境名属于机器人/部署，不写在通用模块里：

```python
from conf import g1_omnipicker_conf
from sensor.camera_stream import bring_up_cameras, probe_camera_hw, close_cameras

camera_map = g1_omnipicker_conf.camera_map(enable_wrist_l=False)
cameras = bring_up_cameras(camera_map, port_timeout=30.0, frame_timeout=30.0)
hw = probe_camera_hw(cameras, camera_map)  # 默认 (480, 640)
```

`bring_up_cameras` 会先 `wait_ports_open` 再连 WebSocket，并等待首帧。对 OrcaGym `CameraWrapper` 的 `running` / `thread` 访问收在 `stop_camera_stream` / `camera_thread`，业务代码不要直接改这些属性。

## 视频暂存目录

`env.begin_save_video(path)` 的 `path` 是**视频输出目录**，不是触发文件。各脚本用 `--stream_scratch_dir` 传入（默认 `/tmp/<任务>_stream`），不要再硬编码 `STREAM_TRIGGER_PATH`。

两种相机源：

| `--camera_source` | 行为 |
|-------------------|------|
| `websocket`（默认） | 内存流边采边写 |
| `mp4` | 每集写入暂存目录，集末从服务端 MP4 抽帧 |

## NVENC

`encoder_proc.py` 在独立进程里用 FFmpeg NVENC 压 MP4。硬件前提：NVIDIA GPU + 可用的 `h264_nvenc`。兼容层会替换 LeRobot 默认的逐帧 PNG 写入。

`resource_tracker.unregister` 等标准库/第三方私有调用只允许出现在该兼容层，禁止散落到采集脚本。
