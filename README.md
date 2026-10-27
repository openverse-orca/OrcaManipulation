# OrcaLab G1 Push-Cart Demo

在 [OrcaLab / OrcaGym](https://github.com/openverse-orca/OrcaLab) 仿真环境中，用 **WB-WAM 世界模型 + SONIC 动作解码器** 闭环驱动机器人推平板推车的演示项目。

机器人模型为 **Unitree G1（29 DoF）+ 五指灵巧手（Wuji / Dex3）**。策略服务接收 107 维本体状态与 224×224 RGB 图像，输出 104 维 SONIC 动作（64 个 motion tokens + 20+20 手部关节），再由 SONIC decoder 以 50 Hz 解码为 29 个身体关节目标，通过 PD 控制写入 OrcaGym。

> 本项目是一个**部署/闭环推理**工程，不含 WB-WAM 训练代码与模型权重。运行前需要准备好上游 WB-WAM 仓库、bridge 环境、SONIC ONNX 资源与 OrcaLab 场景。

## 演示视频

- [灵活手推车（第三人称）](videos/灵活手推车1.mp4)
- [第一人称视角](videos/第一人称.mp4)

---

## 功能特性

- **闭环仿真推理**：`run_push_cart.py` 通过 gRPC 连接 OrcaGym（默认 `127.0.0.1:50051`），每个控制周期上传状态与图像、取回动作块并执行。
- **HTTP 策略服务**：`policy_server.py` 将 WB-WAM bridge 适配器封装为本地 HTTP 服务，支持推理上下文、动态 base quaternion 物化与动作提交。
- **SONIC 50 Hz 解码**：`sonic_runtime.py` 将 64 个 motion tokens 解码为 29 个身体关节目标，内置关节顺序、默认位姿、动作缩放与 PD 增益。
- **手部后端自动识别**：支持 Wuji（20+20 绝对关节）与 Dex3（7+7 重定向），可通过 `--hand-mode` 覆盖。
- **仿真时间对齐**：50 Hz 控制周期推进 1 ms 物理步长（仿真约比真实时间慢 20 倍），推理等待期间暂停物理，避免动作块到达前姿态漂移。
- **可视化与数据采集**：支持录制 MP4、保存每次推理的 `npz`（状态/图像/动作）、输出推车距离与手部接触指标。
- **站立自检**：`--stand-only` 可先用 PD 保持初始姿态，验证机器人在场景中稳定站立。

## 系统架构

```
┌──────────────────────────── OrcaLab / OrcaGym ────────────────────────────┐
│  factory 场景：G1 机器人 + 平板推车（物理仿真，gRPC 127.0.0.1:50051）      │
└───────────────┬──────────────────────────────────────────▲───────────────┘
                │ 107D state + 224×224 RGB                 │ 29 关节目标 (PD)
                ▼                                          │
      ┌───────────────────┐        HTTP         ┌──────────┴──────────┐
      │ run_push_cart.py  │ ──────────────────► │  policy_server.py   │
      │ （闭环控制/渲染） │ ◄────────────────── │  （WB-WAM 推理）    │
      └─────────┬─────────┘   20×104 action    └──────────┬──────────┘
                │                                         │
                │                                  bridge (WB-WAM)
                ▼                                         │
      ┌───────────────────┐                      ┌────────▼──────────┐
      │ sonic_runtime.py  │  64 tokens @ 50 Hz   │ WB-WAM checkpoint │
      │ 解码 + PD 控制    │ ◄──────────────────── │ + SONIC encoder   │
      └───────────────────┘                      └───────────────────┘
```

- 推理频率约 2.5 Hz，每次返回 `replan`（默认 20）行动作块；
- 解码器以 50 Hz 将动作块转换为关节目标，控制周期与物理步长按仿真时间对齐。

## 目录结构

```
.
├── README.md
├── videos/                       # 演示视频（第三人称 / 第一人称）
└── orcalab_demo/
    ├── __init__.py
    ├── policy_server.py          # WB-WAM → SONIC 的 HTTP 策略服务
    ├── run_push_cart.py          # OrcaGym 闭环控制入口（gRPC + 渲染 + 录制）
    ├── sonic_runtime.py          # SONIC decoder 运行时（50 Hz，29 关节 + PD）
    ├── run_closed_loop.sh        # 一键启动：检查环境 → 启动策略服务 → 运行闭环
    ├── deploy/
    │   ├── deploy_push_cart.yaml       # GPU 部署配置（cuda:0）
    │   ├── deploy_push_cart_cpu.yaml   # CPU 部署配置（闭环脚本默认）
    │   └── push_cart_config.yaml       # 训练配置的部署副本（fastwam.* → wbwam.*）
    ├── assets/g1_29dof_wuji/     # G1 29 DoF + Wuji 手 MJCF 与 STL 网格
    ├── captures/                 # 每次推理保存的 npz（state/action/image）
    ├── outputs/                  # 推车指标 JSON、录制视频
    └── reference/                # 上游迁移说明、公开训练/部署配置参考
```

## 环境依赖

| 组件 | 说明 |
| --- | --- |
| OrcaLab / OrcaGym | 打开 factory 场景（G1 + 推车）并启动仿真，gRPC 监听 `127.0.0.1:50051` |
| WB-WAM 仓库 | 提供 `bridge` 包；本项目需放在其仓库根目录下（与 `bridge/` 同级） |
| bridge 环境 | `bridge/.venv-wam/bin/python`，包含 `bridge.cli`、`bridge.sonic`、`bridge.wbwam` 等模块 |
| SONIC 资源 | `checkpoints/sonic_release/model_encoder.onnx`（编码）与 `model_decoder.onnx`（解码） |
| WB-WAM 权重 | `push_cart/weights/step_004850.pt`、`push_cart/metadata/dataset_stats.json` |
| OrcaLab Python | 默认 `/home/user/miniconda3/envs/orcalab-26.10.1.16/bin/python`（可用 `ORCALAB_PYTHON` 覆盖） |
| Python 包 | `numpy`、`PyYAML`、ONNX Runtime（CPU/GPU 视配置）等，随 bridge 环境提供 |

部署目录布局示例：

```
<WB-WAM 仓库根目录>/
├── bridge/                      # 上游 bridge 与 .venv-wam
├── checkpoints/sonic_release/   # SONIC ONNX 资源
├── push_cart/                   # 权重与 dataset_stats.json
└── orcalab_demo/                # 本项目
```

## 快速开始

### 1. 启动仿真

在 OrcaLab 中打开 factory 场景（G1 + 平板推车），确认仿真运行且 `127.0.0.1:50051` 可访问。

### 2. 一键闭环

```bash
# 在 WB-WAM 仓库根目录执行
bash orcalab_demo/run_closed_loop.sh

# 指定时长、重置场景并录制视频
bash orcalab_demo/run_closed_loop.sh --seconds 360 --reset-scene \
  --record orcalab_demo/outputs/push_cart.mp4

# 不录制，只跑 30 秒
bash orcalab_demo/run_closed_loop.sh --no-record --seconds 30

# 调整初始位置与推车初始滑动量
bash orcalab_demo/run_closed_loop.sh --start-offset-x 0.2 --cart-slide -0.15
```

脚本会复用已在 `POLICY_URL` 上运行的策略服务；若没有则按 `POLICY_CONFIG` 启动一个，并在退出时关闭。

### 3. 手动启动策略服务

```bash
bridge/.venv-wam/bin/python -m orcalab_demo.policy_server \
  --config orcalab_demo/deploy/deploy_push_cart_cpu.yaml \
  --port 8766 --replan 20
```

然后运行闭环控制器：

```bash
PYTHONPATH=. /path/to/orcalab/python -m orcalab_demo.run_push_cart \
  --policy-url http://127.0.0.1:8766
```

### 4. 站立自检

```bash
python -m orcalab_demo.run_push_cart --stand-only --seconds 30
```

## 配置说明

`deploy/deploy_push_cart.yaml`（GPU）与 `deploy_push_cart_cpu.yaml`（CPU）的主要字段：

| 字段 | 说明 |
| --- | --- |
| `policy.checkpoint_path` | WB-WAM 权重路径，默认 `push_cart/weights/step_004850.pt` |
| `policy.config_path` | 部署用训练配置副本 `deploy/push_cart_config.yaml` |
| `policy.dataset_stats_path` | 归一化统计 `push_cart/metadata/dataset_stats.json` |
| `policy.device` | `cuda:0` 或 `cpu` |
| `policy.num_inference_steps` / `seed` | 推理步数与固定噪声种子 |
| `task.prompt` | 任务指令，默认 `Push the chair or cart forward.` |
| `wam_input` | 107 维状态布局（gravity）与 224×224 RGB 输入 |
| `sonic_action` | SONIC encoder 路径、时间步长（0.05 s）、坐标系与动作表示 |
| `runtime.max_abs_token` | motion token 限幅（默认 1.25） |

`run_closed_loop.sh` 支持的环境变量：

| 变量 | 默认值 |
| --- | --- |
| `POLICY_CONFIG` | `orcalab_demo/deploy/deploy_push_cart_cpu.yaml` |
| `POLICY_PORT` / `POLICY_URL` | `8766` / `http://127.0.0.1:8766` |
| `POLICY_THREADS` | `8`（CPU 推理线程数） |
| `POLICY_CAPTURES` | `1`（保存推理 npz，设为 `0` 关闭） |
| `POLICY_LOG` | `/tmp/wbwam_policy_server.log` |
| `ORCALAB_PYTHON` | `/home/user/miniconda3/envs/orcalab-26.10.1.16/bin/python` |

## 策略服务接口

`policy_server.py` 默认监听 `127.0.0.1:8765`，请求体上限 16 MiB。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 返回状态维度、动作维度、动作时域、`replan`、提示词与推理步数 |
| POST | `/reset` | 重置适配器与推理上下文 |
| POST | `/infer` | 输入 `state`（107D）、`image`（224×224×3 uint8，base64）、可选 `prompt`；返回 `action_chunk`（默认 20×104）及 `context_id` |
| POST | `/materialize` | 根据 `context_id`、`action_index`、`base_quat_wxyz` 物化动态 base quaternion 动作 |
| POST | `/commit` | 提交实际执行的动作，供适配器滚动上下文 |

动作布局：`[0:64]` motion tokens，`[64:84]` 左手 20 关节，`[84:104]` 右手 20 关节。服务端会做有限性检查与 token 限幅。

## 主要命令行参数（run_push_cart.py）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--orcagym-addr` | `127.0.0.1:50051` | OrcaGym gRPC 地址 |
| `--policy-url` | `http://127.0.0.1:8765` | 策略服务地址 |
| `--decoder` | `checkpoints/sonic_release/model_decoder.onnx` | SONIC 解码器 |
| `--seconds` | `360` | 运行时长（仿真秒） |
| `--stand-only` | 关闭 | 仅保持初始姿态自检 |
| `--start-policy` | 关闭 | 跳过初始姿态阶段，直接推理 |
| `--hand-mode` | `auto` | `auto` / `wuji` / `dex3` |
| `--success-distance` | `2.0` | 推车前进判定距离（米，0 关闭） |
| `--record` / `--no-record` | 录制 | 视频输出路径 |
| `--reset-scene` | 关闭 | 运行前重置场景与推车关节 |
| `--start-offset-x` | `0.0` | 机器人初始 x 偏移（米） |
| `--cart-slide` | `0.0` | 推车滑轨初始偏移（米） |
| `--cart-friction` / `--cart-damping` | 自动 / `4.0` | 推车摩擦与阻尼 |

## 输出

- `orcalab_demo/outputs/push_cart_demo.mp4`：默认录制视频（可用 `--record` 修改）；
- `orcalab_demo/outputs/wuji_push.json`：运行指标（推车距离、手部接触等）；
- `orcalab_demo/captures/inference_XXXXXX.npz`：每次推理的 `state`、`action` 与 `image`，便于离线回放与分析。

## 注意事项

- 本项目默认**仿真时间与真实时间不同步**（推进 1 ms 物理步长），推理延迟会被仿真暂停吸收，便于 CPU 推理。
- 策略服务与闭环控制器需在**同一仓库根目录**下运行（`PYTHONPATH` 指向 WB-WAM 仓库根），否则 `bridge` 与 `orcalab_demo` 无法导入。
- 若手动调整过场景或机器人资产，运行 `run_closed_loop.sh` 时请去掉 `--reset-scene` 以保留变换。
- 更换权重时，请确保 **权重、训练配置副本与 `dataset_stats.json` 来自同一 checkpoint 版本**。
- `reference/` 中保留了上游迁移说明、训练配置与部署配置参考，仅用于对照。

## 上游与致谢

- **OrcaLab / OrcaGym**：仿真环境与 gRPC 接口；
- **WB-WAM**：视频-动作世界模型与 bridge 适配器；
- **NVIDIA GEAR-SONIC**：SONIC encoder/decoder 与动作 token 表示；
- **Unitree G1**：29 DoF 人形机器人模型与网格资源。

本仓库仅包含部署与闭环推理代码，模型权重与上游框架需自行获取。
