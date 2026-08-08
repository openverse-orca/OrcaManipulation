# OpenPI 部署指南

本文档面向需要基于本仓库采集数据、训练策略并部署 RTC 推理服务的工程师。本仓库（OrcaManipulation）负责机器人数据采集与在线推理客户端，策略训练与服务端推理由 [Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi) 框架承担。

---

## 概述

```
OrcaManipulation           openpi（独立环境）
─────────────              ────────────────────
数据采集（LeRobot v2.1）
        │
        ▼ HF_LEROBOT_HOME/<dataset>/
                          训练 train.py
                                │
                                ▼ checkpoints/
                          openpi-rtc-serve（RTC 推理服务）
                                │ WebSocket
        ▼
eval 推理脚本（本仓库）
```

两套环境**不共享** Python 虚拟环境：采集与推理使用 `orcalab_lerobot`（Conda），训练与服务端使用 openpi 的 uv 环境。

---

## 1. 硬件要求

| 使用场景 | 最低 GPU 显存 | 参考机型 |
|---|---|---|
| 推理（服务端） | 8 GB | RTX 4090 |
| LoRA 微调 | 22.5 GB | RTX 4090 |
| 全量微调 | 70 GB | A100-80G / H100 |

操作系统：Ubuntu 22.04，NVIDIA 驱动已安装。

---

## 2. 部署 openpi 环境

### 2.1 拉取并锁定版本

> 请务必锁定到以下 commit，RTC 扩展包与该版本的内部 API 严格绑定，升级上游版本会导致服务启动失败。

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout 981483d
```

### 2.2 安装 uv

```bash
pip install uv
```

或参考 [uv 官方安装文档](https://docs.astral.sh/uv/getting-started/installation/)。

### 2.3 安装依赖

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

> `GIT_LFS_SKIP_SMUDGE=1` 用于跳过 LeRobot 的 LFS 文件拉取。

验证 JAX 可见 GPU：

```bash
uv run python -c "import jax; print(jax.devices())"
```

### 2.4 安装 openpi-rtc RTC 扩展包

将本仓库 `third_party/openpi-rtc/` 复制到 openpi 目录下，然后安装：

```bash
cp -r /path/to/OrcaManipulation/third_party/openpi-rtc /path/to/openpi/packages/openpi-rtc
cd /path/to/openpi
uv pip install -e packages/openpi-rtc
```

验证安装与兼容性：

```bash
uv run python -c "from openpi_rtc import check_openpi_compat; check_openpi_compat()"
```

输出 `兼容性校验通过` 则表示环境正确。

---

## 3. 环境变量说明

在 openpi 环境中运行训练或服务时，按需配置以下环境变量。

| 变量 | 作用 | 示例值 |
|---|---|---|
| `OPENPI_DATA_HOME` | checkpoint 下载缓存目录（默认 `~/.cache/openpi`） | `/data/openpi_cache` |
| `HF_LEROBOT_HOME` | 本地 LeRobot 数据集根目录，训练时读取数据集 | `/data/lerobot_datasets` |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | JAX 训练时可占用的 GPU 显存比例（默认 0.75） | `0.9` |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | 推理时关闭 JAX 显存预分配（推荐设为 `false`） | `false` |
| `XLA_PYTHON_CLIENT_ALLOCATOR` | 推理时使用平台分配器（推荐与上条同时使用） | `platform` |
| `CUDA_VISIBLE_DEVICES` | 指定使用的 GPU 编号 | `0` 或 `1` |
| `WANDB_MODE` | 控制 Weights & Biases 日志；离线训练时可设为 `offline` 或 `disabled` | `offline` |

建议在 shell 配置或训练启动脚本中统一设置，避免每次手动指定。

---

## 4. 数据准备

本仓库采集产出的 LeRobot v2.1 格式数据集，目录结构如下：

```
<数据集名称>/
├── meta/
│   ├── info.json          # fps、action_dim、相机键等元信息
│   ├── episodes.jsonl     # 每集的 index / length / task
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl        # 语言指令列表（task_index → task）
├── data/chunk-000/
│   └── episode_XXXXXX.parquet
└── videos/chunk-000/
    ├── observation.images.cam_head/
    └── observation.images.cam_wrist_r/
```

请将数据集放置在 `HF_LEROBOT_HOME/<数据集名称>/` 下，训练时以 `<数据集名称>` 作为 `repo_id`。

### 已知问题：本地数据集加载

上游 openpi（`981483d`）的 `src/openpi/training/data_loader.py` 默认从 HuggingFace Hub 解析数据集元信息，对纯本地数据集会报错。

请在 openpi 工作区手动应用以下补丁（一次性操作）：

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

**补丁说明**：优先检查 `HF_LEROBOT_HOME/<repo_id>/` 是否存在本地数据集；存在时从 `meta/info.json` 读取真实帧率（本仓库采集默认 20 FPS，上游代码假设 30 FPS 会导致时间步错位），任务列表从 `meta/tasks.jsonl` 读取。不影响从 Hub 加载数据集的原有路径。

---

## 5. 配置训练参数（config.py）

openpi 的所有训练配置均集中在 `src/openpi/training/config.py` 的 `_CONFIGS` 列表中。为自己的机器人任务接入 openpi，需要完成以下三步。

### 5.1 编写数据变换（Inputs / Outputs transforms）

创建一个策略文件（参照 `src/openpi/policies/libero_policy.py`），定义两个数据变换类：

**`XxxInputs`（训练与推理共用）**：把 LeRobot 数据集或推理时的 obs 字典，映射到模型期望的标准键：

```python
import dataclasses
from openpi.policies import transforms

@dataclasses.dataclass(frozen=True)
class MyRobotInputs(transforms.DataTransformFn):
    action_dim: int           # 与 TrainConfig.model.action_dim 一致
    action_horizon: int = 50  # 与 TrainConfig.model.action_horizon 一致

    def __call__(self, data: dict) -> dict:
        # 从数据集键名映射到模型输入键名
        # 模型标准输入键：state, images, prompt, (actions - 仅训练时)
        # 'state': 关节角等本体感知信息，shape [action_dim]
        # 'images': dict，键为相机名称，值为 uint8 RGB 图像
        # 'prompt': 字符串，任务语言指令
        return {
            "state": data["observation.state"],
            "images": {
                "cam_head": data["observation.images.cam_head"],
                "cam_wrist_r": data["observation.images.cam_wrist_r"],
            },
            "prompt": data.get("prompt", ""),
            # 训练时需要提供 actions
            "actions": data["action"][:self.action_horizon],
        }
```

**`XxxOutputs`（推理时使用）**：把模型输出的 `actions` 数组，映射回机器人控制格式：

```python
@dataclasses.dataclass(frozen=True)
class MyRobotOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # data["actions"]: shape [action_horizon, action_dim]
        # 按机器人协议返回所需格式
        return {"actions": data["actions"]}
```

### 5.2 定义数据集配置（DataConfigFactory）

在 `config.py` 中继承 `DataConfigFactory`，指定数据集 repo_id 和变换：

```python
@dataclasses.dataclass(frozen=True)
class MyRobotDataConfig(DataConfigFactory):
    repo_id: str = "my_robot_task_dataset"  # HF_LEROBOT_HOME 下的目录名

    def create(self, model_config: BaseModelConfig) -> DataConfig:
        return DataConfig(
            repo_id=self.repo_id,
            input_transform=MyRobotInputs(
                action_dim=model_config.action_dim,
                action_horizon=model_config.action_horizon,
            ),
            output_transform=MyRobotOutputs(),
            prompt_from_task=True,  # 从 tasks.jsonl 读取语言指令
        )
```

### 5.3 定义并注册 TrainConfig

`TrainConfig` 的关键字段：

| 字段 | 说明 | 典型值 |
|---|---|---|
| `name` | 配置名称，全局唯一，训练和推理时用 `--config` 指定 | `"my_robot_task_lora"` |
| `model` | 模型结构配置，含 `action_dim`（动作维度）、`action_horizon`（动作块长度） | 见下方 |
| `data` | DataConfigFactory 实例 | `MyRobotDataConfig(repo_id="...")` |
| `batch_size` | 训练批大小（LoRA 推荐 32，全量推荐 16 或更小） | `32` |
| `num_train_steps` | 总训练步数 | `30_000` |
| `freeze_filter` | 冻结的参数过滤器（全量微调时留空，LoRA 时通过模型 config 获取） | 见下方 |
| `exp_name` | 实验名，checkpoints 保存在 `checkpoints/<name>/<exp_name>/` 下 | CLI 传入 |

**π₀.₅ LoRA 微调示例**：

```python
TrainConfig(
    name="pi05_my_robot_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=28,          # 根据实际关节数修改
        action_horizon=50,      # 根据任务节奏修改
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=MyRobotDataConfig(repo_id="my_robot_task_dataset"),
    batch_size=32,
    num_train_steps=30_000,
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        action_dim=28,
        action_horizon=50,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    # LoRA 微调关闭 EMA
    ema_decay=None,
)
```

最后把该 `TrainConfig` 对象追加到 `_CONFIGS` 列表（文件末尾，`if len(...)` 校验之前）：

```python
_CONFIGS = [
    ...
    # 你的自定义配置
    TrainConfig(name="pi05_my_robot_lora", ...),
]
```

---

## 6. 计算归一化统计

在开始训练之前，必须先计算数据集的归一化统计，以便模型对 state 和 action 正确归一化：

```bash
cd /path/to/openpi
HF_LEROBOT_HOME=/path/to/datasets \
uv run scripts/compute_norm_stats.py --config-name pi05_my_robot_lora
```

统计文件保存在 `assets/<config_name>/` 目录下。

---

## 7. 启动训练

```bash
cd /path/to/openpi

CUDA_VISIBLE_DEVICES=0 \
HF_LEROBOT_HOME=/path/to/datasets \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
WANDB_MODE=offline \
uv run scripts/train.py pi05_my_robot_lora \
    --exp-name=my_experiment \
    --overwrite
```

- `--overwrite`：若已存在同名实验目录则覆盖，重跑时加此参数。
- checkpoint 自动保存在 `checkpoints/pi05_my_robot_lora/my_experiment/<step>/`。
- 训练进度默认同步到 Weights & Biases，离线环境请设置 `WANDB_MODE=offline`。
- 显存使用量超出时，可降低 `batch_size` 或缩减 `action_horizon`。

---

## 8. 启动推理服务

训练完成后，启动策略服务器等待来自本仓库推理脚本的连接。

### 8.1 普通推理模式

```bash
cd /path/to/openpi

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run scripts/serve_policy.py \
    --port 8010 \
    policy:checkpoint \
    --policy.config=pi05_my_robot_lora \
    --policy.dir=checkpoints/pi05_my_robot_lora/my_experiment/29999
```

### 8.2 RTC 异步推理模式（推荐）

RTC（Real-Time Control）模式通过软前缀伪逆引导，实现当前动作块执行期间提前计算下一块，大幅降低实际控制延迟。需要先按第 2.4 节安装 `openpi-rtc`。

```bash
cd /path/to/openpi

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run openpi-rtc-serve \
    --port 8010 \
    --rtc-control-frequency-hz 20.0 \
    --rtc-min-execution-horizon 25 \
    --rtc-initial-delay-steps 4 \
    policy:checkpoint \
    --policy.config=pi05_my_robot_lora \
    --policy.dir=checkpoints/pi05_my_robot_lora/my_experiment/29999
```

#### RTC 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--rtc-num-denoising-steps` | `10` | 去噪步数，与标准 π₀.₅ 推理一致 |
| `--rtc-max-guidance-weight` | `10.0` | 软前缀伪逆引导最大权重 |
| `--rtc-prefix-attention-schedule` | `exp` | 前缀权重衰减方式：`exp` / `linear` / `ones` / `zeros` |
| `--rtc-control-frequency-hz` | `20.0` | 策略控制频率（Hz），用于推理延迟估计 |
| `--rtc-min-execution-horizon` | `25` | 每个动作块至少执行多少步后才发起下一次推理 |
| `--rtc-initial-delay-steps` | `4` | 初始推理延迟先验值（策略 action 步数） |
| `--rtc-delay-history-size` | `10` | 延迟历史窗口大小 |

服务器成功启动后，日志会显示 `RTC 已启用` 并列出当前参数，随后进入等待状态：

```
RTC 已启用：H=50 s_min=25 d_init=4 guidance=10.0 steps=10 protocol_v=1
启动服务器（host: ..., ip: ..., port: 8010）
```

---

## 9. 与本仓库 eval 脚本对接

本仓库的 eval 脚本（`src/examples/inference/`）使用 `third_party/openpi-client` 通过 WebSocket 与上述推理服务通信。

### 9.1 普通推理协议

发送标准 obs 字典（含 `state`、`images`、`prompt`），服务器返回 `actions`：

```python
from openpi_client import websocket_client_policy as _wsp
client = _wsp.WebsocketClientPolicy(host="localhost", port=8010)
response = client.infer(obs)
# response["actions"]: shape [action_horizon, action_dim]
```

### 9.2 RTC 推理协议

在 obs 中额外携带 `prev_actions` 和 `inference_delay` 字段，服务器走 RTC 软前缀引导路径：

```python
obs_rtc = {
    **obs,
    "prev_actions": prev_chunk_tail,  # shape [n, action_dim], 1 ≤ n ≤ action_horizon
    "inference_delay": delay_steps,   # int，推理期间预计已执行的步数
}
response = client.infer(obs_rtc)
```

连接建立时，服务器握手 metadata 中包含 `rtc` 段，可用于校验版本与参数：

```python
metadata = client.get_server_metadata()
rtc_meta = metadata["rtc"]
# rtc_meta["protocol_version"] == 1
# rtc_meta["action_horizon"]、rtc_meta["min_execution_horizon"] 等
```

`prev_actions` 字段为当前正在执行的旧动作块中**尚未执行**的尾段；`inference_delay` 为发出推理请求时估计的延迟步数（满足 `0 ≤ inference_delay ≤ len(prev_actions)`）。

---

## 10. 故障排查

**显存不足（OOM）**

- 训练时：降低 `batch_size`（LoRA 从 32 降到 16）；或降低 `XLA_PYTHON_CLIENT_MEM_FRACTION`（如 0.85）。
- 推理时：确认 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 已设置；检查是否有其他进程占用 GPU 显存（`nvidia-smi`）。

**找不到 TrainConfig 名称**

```
ValueError: Config 'xxx' not found.
```

请确认：①配置已追加到 `_CONFIGS` 列表；②配置名称拼写与 `--config` 参数完全一致（区分大小写）；③已重新安装或重新运行（`uv run` 会自动使用更新后的源码）。

**本地数据集报 Hub 连接错误**

```
requests.exceptions.ConnectionError: ...
```

请确认已按第 4 节应用 `data_loader.py` 补丁，并且数据集目录存在于 `HF_LEROBOT_HOME/<repo_id>/`。

**RTC 兼容性校验失败**

```
[openpi-rtc] 兼容性校验失败：...
```

表示当前 openpi 版本与 `openpi-rtc` 不匹配。请确认：

```bash
cd /path/to/openpi
git rev-parse --short HEAD  # 应输出 981483d
```

若不匹配，请执行 `git checkout 981483d` 并重新 `uv sync`。

**推理时机械臂动作抖动或跟踪发散**

请确认 eval 脚本使用开环积分（每步以上一步输出的绝对位置为基础叠加 Δq），而非每步用实测关节角重基。若使用 RTC 模式，请检查 `inference_delay` 的估计逻辑——过大的估计值会使引导权重异常。
