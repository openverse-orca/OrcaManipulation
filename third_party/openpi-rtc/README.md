# openpi-rtc

OpenPI Pi0/Pi0.5 的 RTC（Real-Time Control）服务端扩展包。

安装本包后，无需修改任何 openpi 源文件，即可为 Pi0/Pi0.5 策略服务器
启用软前缀伪逆引导（soft-prefix pseudo-inverse guidance）的实时控制推理能力。

---

## 目录

1. [前置条件](#1-前置条件)
2. [安装](#2-安装)
3. [启动推理服务](#3-启动推理服务)
4. [参数说明](#4-参数说明)
5. [WebSocket 协议契约](#5-websocket-协议契约)
6. [版本兼容声明](#6-版本兼容声明)
7. [已知问题：本地 LeRobot 数据集训练](#7-已知问题本地-lerobot-数据集训练)

---

## 1. 前置条件

### 1.1 拉取上游 openpi 并锁定版本

> **必须锁定到指定 commit，否则 openpi 内部 API 变动会导致 RTC 运行时崩溃。**

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout 981483d
```

### 1.2 按上游 README 安装 openpi 环境

```bash
# 上游推荐使用 uv；确保已安装 uv >= 0.5
pip install uv

# 在 openpi 目录下安装所有依赖（含 JAX CUDA12 后端）
uv sync
```

安装完成后确认 JAX 可见 GPU：

```bash
uv run python -c "import jax; print(jax.devices())"
```

---

## 2. 安装

在已激活的 openpi 虚拟环境（`.venv`）中安装本包：

```bash
# 从目录安装（交付时将 openpi-rtc/ 目录拷贝到任意位置）
cd /path/to/openpi-rtc
uv pip install .

# 或开发模式（editable）安装，适合在 openpi workspace 内使用
uv pip install -e /path/to/openpi/packages/openpi-rtc
```

安装后可运行兼容性自检：

```bash
uv run python -c "from openpi_rtc import check_openpi_compat; check_openpi_compat()"
```

输出 `兼容性校验通过` 即表示 openpi 版本与本包匹配。

---

## 3. 启动推理服务

```bash
# 基本用法
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
openpi-rtc-serve \
  --port 8010 \
  --rtc-min-execution-horizon 25 \
  --rtc-initial-delay-steps 4 \
  policy:checkpoint \
    --policy.config=<TrainConfig 名称> \
    --policy.dir=<checkpoint 路径>
```

等价于之前在 openpi 源码中使用的：

```bash
uv run scripts/serve_policy.py \
  --port 8010 \
  --rtc \
  --rtc-min-execution-horizon 25 \
  policy:checkpoint \
    --policy.config=<TrainConfig 名称> \
    --policy.dir=<checkpoint 路径>
```

服务器启动后会打印如下日志，确认 RTC 已生效：

```
[openpi-rtc] 兼容性校验通过 ...
RTC 已启用：H=50 s_min=25 d_init=4 guidance=10.0 steps=10 protocol_v=1
启动服务器（host: ..., ip: ..., port: 8010）
```

---

## 4. 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--port` | `8000` | WebSocket 监听端口 |
| `--default-prompt` | `None` | obs 缺省 prompt 时的语言提示 |
| `--skip-compat-check` | `False` | 跳过启动时的 API 兼容性校验（不建议） |
| `--rtc-num-denoising-steps` | `10` | RTC 去噪步数（与 Pi0.5 标准推理一致） |
| `--rtc-max-guidance-weight` | `10.0` | 软前缀伪逆引导最大权重 |
| `--rtc-prefix-attention-schedule` | `exp` | 前缀权重衰减方式：`exp` / `linear` / `ones` / `zeros` |
| `--rtc-control-frequency-hz` | `20.0` | 策略控制频率（Hz），用于延迟估计 |
| `--rtc-min-execution-horizon` | `25` | 每个动作块至少执行多少步后才允许发起下一次 RTC 请求 |
| `--rtc-initial-delay-steps` | `4` | 初始推理延迟先验值（策略 action 步数） |
| `--rtc-delay-history-size` | `10` | 延迟历史窗口大小（用于指数平滑估计） |

`policy:checkpoint` 子命令参数：

| 参数 | 说明 |
|---|---|
| `--policy.config` | TrainConfig 名称，需在 openpi 的 `training/config.py` 中注册 |
| `--policy.dir` | checkpoint 目录，例如 `checkpoints/my_exp/29999` |

---

## 5. WebSocket 协议契约

### 5.1 握手 metadata

服务器启动后，客户端建立 WebSocket 连接时会收到 msgpack 编码的 metadata 字典。
RTC 服务器在其中额外包含 `rtc` 段：

```json
{
  "norm_stats": { ... },
  "rtc": {
    "protocol_version": 1,
    "action_horizon": 50,
    "num_denoising_steps": 10,
    "max_guidance_weight": 10.0,
    "prefix_attention_schedule": "exp",
    "control_frequency_hz": 20.0,
    "min_execution_horizon": 25,
    "initial_delay_steps": 4,
    "delay_history_size": 10
  }
}
```

客户端应校验 `rtc.protocol_version == 1`，不匹配时应拒绝连接并报错。

### 5.2 普通推理请求（向后兼容）

不携带 `prev_actions` 的请求走普通推理路径，行为与标准 openpi 服务器完全一致：

```python
obs = {
    "state": ...,        # [action_dim] float32
    "images": { ... },   # 相机图像
    "prompt": "...",     # 语言提示（可选，有 default_prompt 时可省略）
}
response = client.infer(obs)
# response["actions"]: [action_horizon, action_dim] float32
```

### 5.3 RTC 推理请求

携带 `prev_actions` 和 `inference_delay` 时走 RTC 软前缀引导路径：

```python
obs = {
    "state": ...,
    "images": { ... },
    "prompt": "...",
    "prev_actions": prev_chunk_tail,   # [n, action_dim] float32, 1 <= n <= action_horizon
    "inference_delay": delay_steps,    # int, 推理期间已执行的动作步数估计
}
response = client.infer(obs)
# response["actions"]: [action_horizon, action_dim] float32
# response["policy_timing"]["infer_ms"]: float, 实际推理耗时（毫秒）
```

**字段说明：**

- `prev_actions`：当前正在执行的旧动作块的尾段（从推理发起时刻起，
  还剩多少步未执行）。形状 `[n, action_dim]`，`n` 满足 `1 ≤ n ≤ action_horizon`。
- `inference_delay`：发起本次推理时，预计当推理完成后已执行了多少步旧动作。
  满足 `0 ≤ inference_delay ≤ n`。
  客户端可用 `min_execution_horizon` 和上一次推理的实测延迟来估算此值。

---

## 6. 版本兼容声明

openpi-rtc 依赖 openpi 的若干**私有** API（`Policy._model`、`Policy._rng`、
`Pi0.embed_suffix` 的 4 元组返回值等），上游 API 变动会导致运行时崩溃。

**必须锁定 openpi 到 commit `981483d`。**

服务器启动时会自动运行兼容性校验（`_compat.check()`），检查所有依赖的私有
API 是否存在；若不匹配会打印明确错误信息并退出，不会出现难以定位的运行时崩溃。

如果你在 openpi workspace 内以 editable 模式安装本包，`src/openpi/rtc/` 下的
服务端文件均为 re-export shim，真实实现在本包内；你之前使用的
`scripts/serve_policy.py --rtc` 路径**无需修改**，继续可用。

---

## 7. 已知问题：本地 LeRobot 数据集训练

上游 openpi 的 `src/openpi/training/data_loader.py` 在加载数据集时默认走
HuggingFace Hub 解析元信息，对**纯本地**（从未上传 Hub）的 LeRobot 数据集会报错。

原因：上游代码直接调用：

```python
dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
```

不传 `root=` 时会尝试访问 Hub；而 OrcaManipulation 采集产出的数据集只存在于本地。

**修复方式**：在 openpi 工作区手动应用如下补丁（锁定 commit 981483d 的基础上）：

```diff
--- a/src/openpi/training/data_loader.py
+++ b/src/openpi/training/data_loader.py
@@ -137,16 +137,51 @@ def create_torch_dataset(
     if repo_id == "fake":
         return FakeDataset(model_config, num_samples=1024)
 
-    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
-    dataset = lerobot_dataset.LeRobotDataset(
-        data_config.repo_id,
-        delta_timestamps={
-            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
-        },
-    )
+    from lerobot.common.constants import HF_LEROBOT_HOME
+    _local_path = HF_LEROBOT_HOME / repo_id
+    _root = str(_local_path) if _local_path.exists() else None
+
+    if _root is not None:
+        import json as _json
+        from pathlib import Path as _Path
+        _info_path = _Path(_root) / "meta" / "info.json"
+        _local_fps = _json.loads(_info_path.read_text()).get("fps", 30) if _info_path.exists() else 30
+        dataset = lerobot_dataset.LeRobotDataset(
+            data_config.repo_id,
+            root=_root,
+            delta_timestamps={
+                key: [t / _local_fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
+            },
+        )
+        dataset_meta = None
+    else:
+        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
+        dataset = lerobot_dataset.LeRobotDataset(
+            data_config.repo_id,
+            delta_timestamps={
+                key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
+            },
+        )
 
     if data_config.prompt_from_task:
-        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
+        if dataset_meta is not None:
+            tasks = dataset_meta.tasks
+        else:
+            import json as _json2
+            from pathlib import Path as _Path2
+            _tasks_path = _Path2(_root) / "meta" / "tasks.jsonl"
+            if _tasks_path.exists():
+                tasks = {
+                    obj["task_index"]: obj["task"]
+                    for obj in (_json2.loads(line) for line in _tasks_path.read_text().splitlines() if line.strip())
+                }
+            else:
+                tasks = {}
+        if tasks:
+            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(tasks)])
 
     return dataset
```

**说明：**

- 补丁优先检查 `HF_LEROBOT_HOME/<repo_id>/` 是否存在本地数据集；
  存在则从 `meta/info.json` 读取真实帧率（OrcaManipulation 采集默认 20 FPS，
  而上游代码假设 30 FPS 会导致 delta_timestamps 错误）；
- 任务列表从 `meta/tasks.jsonl` 读取，与 LeRobot v2.1 格式一致；
- 若本地目录不存在则回退到 Hub 路径，不影响原有用法。

此修复不包含在 openpi-rtc 包内，因为它属于训练数据加载路径，与 RTC 推理服务无关。
