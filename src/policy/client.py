"""策略服务客户端。封装 OpenPI WebSocket 调用与相机观测组装。"""
from __future__ import annotations

import numpy as np

from policy.schema import PolicySchema


class CameraObservationBuilder:
    """从已配置的相机数据流构建策略图像观测（CHW）。"""

    def __init__(
        self,
        cameras: dict,
        camera_name_map: dict[str, str],
        target_hw: tuple = (480, 640),
    ):
        self.cameras = cameras
        self.camera_name_map = camera_name_map
        self.target_hw = target_hw

    def build_images(self) -> dict:
        import cv2

        height, width = self.target_hw
        images = {}
        for env_camera_name, policy_camera_name in self.camera_name_map.items():
            cam = self.cameras.get(env_camera_name)
            if cam is None:
                rgb = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                try:
                    frame, _ = cam.get_frame(format="rgb24")
                    if frame is None or frame.size == 0:
                        rgb = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        if frame.shape[0] != height or frame.shape[1] != width:
                            frame = cv2.resize(
                                frame, (width, height), interpolation=cv2.INTER_AREA
                            )
                        rgb = np.ascontiguousarray(frame, dtype=np.uint8)
                except Exception:
                    rgb = np.zeros((height, width, 3), dtype=np.uint8)
            images[policy_camera_name] = np.transpose(rgb, (2, 0, 1))
        return images


class PolicyClient:
    """封装 openpi_client WebSocket 策略调用。

    控制循环内手动遍历 action chunk，不使用 ActionChunkBroker，以保持原有时序。
    """

    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
        camera_name_map: dict[str, str],
        cameras: dict,
        schema: PolicySchema | None = None,
        target_hw: tuple = (480, 640),
        use_images: bool = True,
    ):
        from openpi_client import websocket_client_policy

        self.policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.policy.get_server_metadata()
        self.prompt = prompt
        self.schema = schema
        self.use_images = use_images
        self.cam_builder = (
            CameraObservationBuilder(
                cameras=cameras,
                camera_name_map=camera_name_map,
                target_hw=target_hw,
            )
            if use_images
            else None
        )

    def build_observation(self, state: np.ndarray) -> dict:
        images = self.cam_builder.build_images() if self.use_images else {}
        return {"state": state, "images": images, "prompt": self.prompt}

    def infer_action_chunk(self, state: np.ndarray) -> np.ndarray:
        observation = self.build_observation(state)
        result = self.policy.infer(observation)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        min_dim = self.schema.action_dim if self.schema is not None else 1
        if actions.shape[-1] < min_dim:
            raise ValueError(
                f"Expected policy action dim >= {min_dim}, got {actions.shape}"
            )
        return actions
