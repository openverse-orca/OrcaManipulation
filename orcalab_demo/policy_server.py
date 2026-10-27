"""Small HTTP wrapper around the WB-WAM -> SONIC encoder bridge adapter.

The OrcaGym simulation process sends one 107D gravity-layout state and one
224x224 RGB frame, and receives the first ``--replan`` rows of the 104D SONIC
action (64 motion tokens + 20 left-hand + 20 right-hand joint positions).

Run from the repository root with the bridge environment:
    bridge/.venv-wam/bin/python -m orcalab_demo.policy_server \
        --config orcalab_demo/deploy/deploy_push_cart.yaml --port 8765
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bridge.cli import _make_wam_adapter
from bridge.common.observation_builder import ACTION_DIM
from bridge.sonic.action_schema import split_action_step
from bridge.wbwam.text_cache import ensure_prompt_embedding

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STATE_DIM = 107
EXPECTED_IMAGE_HW = (224, 224)


def load_deployment_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"deployment config must be a mapping: {path}")
    return config


def decode_image(payload: dict[str, Any]) -> np.ndarray:
    item = payload["image"]
    shape = tuple(int(v) for v in item["shape"])
    if len(shape) != 3 or shape[0] != EXPECTED_IMAGE_HW[0] or shape[1] != EXPECTED_IMAGE_HW[1] or shape[2] != 3:
        raise ValueError(f"expected uint8 HWC RGB image {EXPECTED_IMAGE_HW + (3,)}, got {shape}")
    if item.get("dtype") != "uint8":
        raise ValueError("image dtype must be uint8")
    raw = base64.b64decode(item.get("data_b64", item.get("data")), validate=True)
    if len(raw) != int(np.prod(shape)):
        raise ValueError("image byte count mismatch")
    return np.frombuffer(raw, np.uint8).reshape(shape).copy()


class Service:
    def __init__(self, config_path: Path, replan: int, record_captures: bool) -> None:
        self.config_path = config_path
        self.config = load_deployment_config(config_path)
        self.replan = int(replan)
        self.inference_lock = threading.Lock()
        self.materialize_lock = threading.Lock()
        self.context_lock = threading.Lock()
        self.contexts: dict[int, Any] = {}
        self.context_count = 0
        self.count = 0
        self.capture_dir = REPO_ROOT / "orcalab_demo" / "captures"
        if record_captures:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.record_captures = bool(record_captures)

        policy_config = dict(self.config.get("policy") or {})
        prompt = str((self.config.get("task") or {}).get("prompt", "")).strip()
        if not prompt:
            raise ValueError("task.prompt is required")
        self.prompt = prompt

        embedding = ensure_prompt_embedding(
            config_path,
            prompt,
            policy_config=policy_config,
        )
        policy_config["prompt_embedding_path"] = str(embedding)
        self.config["policy"] = policy_config

        self.adapter = _make_wam_adapter(
            self.config,
            dict(self.config.get("wam_input") or {}),
            mock_policy=False,
        )
        if self.adapter.info.action_dim != ACTION_DIM:
            raise ValueError(f"expected {ACTION_DIM}D SONIC action, got {self.adapter.info.action_dim}")
        print(
            "[policy_server] ready "
            f"prompt={prompt!r} state_dim={self.adapter.info.state_dim} "
            f"action_dim={self.adapter.info.action_dim} horizon={self.adapter.info.action_horizon}",
            flush=True,
        )

    def reset(self) -> dict[str, Any]:
        with self.inference_lock:
            with self.materialize_lock:
                self.adapter.reset_execution_context()
                with self.context_lock:
                    self.contexts.clear()
        return {"status": "ok"}

    def checked_action(self, values: np.ndarray) -> np.ndarray:
        runtime = self.config.get("runtime") or {}
        max_abs_token = float(runtime.get("max_abs_token", 1.25))
        checked = split_action_step(
            values,
            snap_token_grid=bool(runtime.get("snap_token_grid", True)),
            token_min=-max_abs_token,
            token_max=max_abs_token,
        )
        return np.concatenate(
            [checked.token_state, checked.left_wuji_qpos, checked.right_wuji_qpos]
        ).astype(np.float32)

    def materialize(self, payload: dict[str, Any]) -> dict[str, Any]:
        context_id = int(payload["context_id"])
        action_index = int(payload["action_index"])
        base_quat = np.asarray(payload["base_quat_wxyz"], dtype=np.float32).reshape(4)
        if not np.isfinite(base_quat).all() or np.linalg.norm(base_quat) < 1e-6:
            raise ValueError("base_quat_wxyz must be finite and non-zero")
        with self.context_lock:
            context = self.contexts.get(context_id)
        if context is None:
            raise KeyError(f"unknown inference context {context_id}")
        with self.materialize_lock:
            try:
                values = self.adapter.materialize_action(
                    context,
                    action_index=action_index,
                    base_quat_wxyz=base_quat,
                    stationary=bool(payload.get("stationary", False)),
                )
                if values is None:
                    raise RuntimeError("adapter has no dynamic base-quat materializer")
                action = self.checked_action(np.asarray(values, dtype=np.float32))
                self.adapter.commit_action_materialization(context)
            except Exception:
                self.adapter.rollback_action_materialization(context)
                raise
        return {"action": action.tolist()}

    def commit(self, payload: dict[str, Any]) -> dict[str, Any]:
        context_id = int(payload["context_id"])
        with self.context_lock:
            context = self.contexts.get(context_id)
        if context is None:
            raise KeyError(f"unknown inference context {context_id}")
        action = np.asarray(payload["action"], dtype=np.float32).reshape(ACTION_DIM)
        with self.materialize_lock:
            self.adapter.commit_executed_action(
                context,
                action_index=int(payload["action_index"]),
                published_action=action,
            )
        return {"status": "ok"}

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(payload["state"], dtype=np.float32).reshape(-1)
        if state.size != EXPECTED_STATE_DIM or not np.isfinite(state).all():
            raise ValueError(f"expected finite state[{EXPECTED_STATE_DIM}], got {state.shape}")
        image = decode_image(payload)
        prompt = str(payload.get("prompt") or self.prompt)

        start = time.perf_counter()
        with self.inference_lock:
            action = np.asarray(
                self.adapter.infer(
                    {
                        "states": state,
                        "observation/image": image,
                        "prompt": prompt,
                    }
                ),
                dtype=np.float32,
            )
            context = self.adapter.take_inference_context()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if action.shape != (32, ACTION_DIM) or not np.isfinite(action).all():
            raise ValueError(f"expected finite action[32,{ACTION_DIM}], got {action.shape}")
        action = np.asarray([self.checked_action(row) for row in action], dtype=np.float32)
        context_id = None
        if self.adapter.needs_live_base_quat:
            if context is None:
                raise RuntimeError("actual_imu inference did not produce a dynamic execution context")
            with self.context_lock:
                self.context_count += 1
                context_id = self.context_count
                self.contexts[context_id] = context

        if self.record_captures:
            np.savez_compressed(
                self.capture_dir / f"inference_{self.count:06d}.npz",
                state=state,
                action=action,
                image=image,
            )
        self.count += 1
        print(
            json.dumps(
                {
                    "inference": self.count,
                    "latency_ms": round(elapsed_ms, 2),
                    "token_min": float(action[:, :64].min()),
                    "token_max": float(action[:, :64].max()),
                    "left_hand_min": float(action[:, 64:84].min()),
                    "left_hand_max": float(action[:, 64:84].max()),
                    "right_hand_min": float(action[:, 84:104].min()),
                    "right_hand_max": float(action[:, 84:104].max()),
                }
            ),
            flush=True,
        )
        return {
            "action_chunk": action[: self.replan].tolist(),
            "latency_ms": elapsed_ms,
            "context_id": context_id,
            "dynamic_base_quat": bool(self.adapter.needs_live_base_quat),
        }


def make_server(service: Service, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                self.reply(
                    200,
                    {
                        "status": "ok",
                        "state_dim": service.adapter.info.state_dim,
                        "action_dim": service.adapter.info.action_dim,
                        "action_horizon": service.adapter.info.action_horizon,
                        "replan": service.replan,
                        "prompt": service.prompt,
                        "dynamic_base_quat": bool(service.adapter.needs_live_base_quat),
                        "num_inference_steps": service.config["policy"].get("num_inference_steps"),
                    },
                )
            else:
                self.reply(404, {})

        def do_POST(self) -> None:
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 16 << 20:
                    raise ValueError("invalid request body size")
                payload = json.loads(self.rfile.read(size))
                if self.path == "/reset":
                    result = service.reset()
                elif self.path == "/infer":
                    result = service.infer(payload)
                elif self.path == "/materialize":
                    result = service.materialize(payload)
                elif self.path == "/commit":
                    result = service.commit(payload)
                else:
                    self.reply(404, {})
                    return
                self.reply(200, result)
            except Exception as exc:
                import traceback

                traceback.print_exc()
                self.reply(500, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--replan", type=int, default=20)
    parser.add_argument("--no-captures", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    service = Service(config_path, replan=args.replan, record_captures=not args.no_captures)
    server = make_server(service, args.port)
    print(f"[policy_server] Native push_cart ready port={args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
