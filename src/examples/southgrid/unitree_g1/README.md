# Unitree G1 双臂 OSC 抓取 — 采集与回放

本目录包含 Unitree G1 双臂抓取任务的**脚本化数据采集**和 **LeRobot 格式回放**两个入口。  
控制方式为 OSC（Operational Space Control），数据格式为 LeRobot v2.1（18D 末端位姿 + 归一化夹爪指令）。

---

## 目录

- [环境要求](#环境要求)
- [安装](#安装)
- [启动 OrcaLab 仿真](#启动-orcalab-仿真)
- [数据采集](#数据采集)
- [数据回放](#数据回放)
- [关键参数速查](#关键参数速查)
- [常见问题](#常见问题)

---

## 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Ubuntu 22.04 / 24.04 (x86_64) |
| Python | 3.12.13（由 Conda 固定） |
| Conda | Miniconda 或 Anaconda |
| GPU | NVIDIA，视频编码需要 RTX 40 系（AV1 NVENC） |
| OrcaLab | 需已安装并持有 G1 抓取场景订阅 |
| gRPC 端口 | `localhost:50051`（仿真服务默认） |

---

## 安装

**从仓库根目录执行，不要在已有环境上更新。**

```bash
# 1. 克隆仓库
git clone https://github.com/openverse-orca/OrcaManipulation.git
cd OrcaManipulation

# 2. 创建并激活 Conda 环境（Python 3.12.13）
conda env create -f docs/lerobot/environment.yml
conda activate orcalab_lerobot

# 3. 安装 pip 依赖（哈希锁定）+ orca-gym/lab + lerobot + openpi-client
bash docs/lerobot/install_runtime.sh

# 4. 验证安装
python docs/lerobot/verify_environment.py
```

> **注意**：`verify_environment.py` 会测试 AV1 NVENC 编码。若没有 RTX 40 系 GPU，验证会失败，
> 但不录视频的采集与回放仍可正常运行（去掉 `--cameras` 参数即可）。

---

## 启动 OrcaLab 仿真

在另一个终端中：

```bash
conda activate orcalab_lerobot
orcalab
```

1. 在 OrcaLab 界面中选择 **G1 抓取场景**，加载对应布局（JSON 在 `configs/` 目录）。
2. 点击右上角绿色 **启动** 按钮，选择 **无仿真程序（手动启动）**。
3. 等待仿真就绪（状态栏显示 gRPC 已连接）。

---

## 数据采集

切换到脚本目录，激活环境后运行：

```bash
cd /path/to/OrcaManipulation/src/examples/southgrid/unitree_g1
conda activate orcalab_lerobot
export DISPLAY=:1   # 若在无头服务器上运行

python g1_pick_osc_scripted_dual.py \
    --lerobot_out ~/data/g1_pick_dataset \
    --waypoint my_waypoint_dual.yaml \
    --max_episodes 20 \
    --grasp_stiff on \
    --log_txt ~/data/collect01.txt
```

### 常用采集参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lerobot_out` | *必填* | LeRobot 数据集输出目录 |
| `--waypoint` | `my_waypoint_dual.yaml` | 路点 YAML 文件 |
| `--max_episodes` | `1` | 采集回合数 |
| `--fps` | `20` | 数据集帧率（Hz） |
| `--kp` | *(osc_pose.json)* | OSC 刚度；不指定时读配置文件（约 150） |
| `--grasp_stiff` | `off` | `on`：加硬杯子/夹爪接触，去掉深度穿透锁死；**新采集推荐开** |
| `--cameras` | `head,wrist_r,wrist_l` | 启用相机；去掉则不录视频 |
| `--log_txt` | *(不写文件)* | 同步写日志到指定 txt；首行记录完整命令参数 |
| `--orcagym_addr` | `localhost:50051` | gRPC 地址 |
| `--task_config` | `example.yaml` | 场景任务配置 YAML |

---

## 数据回放

回放某一回合（验证采集质量或调参）：

```bash
cd /path/to/OrcaManipulation/src/examples/southgrid/unitree_g1
conda activate orcalab_lerobot
export DISPLAY=:1

python g1_pick_osc_replay_dual.py \
    --lerobot_out ~/data/g1_pick_dataset \
    --episode_index 0 \
    --log_txt ~/data/replay01.txt
```

> **默认配置**（已针对跟踪精度调优）：  
> `--interp on`、`--lead 1.0`（速度前馈）、`--track_ki 0.0`、`--grasp_x_bias 0.012`

### 常用回放参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lerobot_out` | *必填* | 与采集时相同的数据集目录 |
| `--episode_index` | `0` | 回放的回合编号 |
| `--steps` | *(自动推算)* | 每帧保持的控制步数；`10` ≈ 20 Hz，`50` ≈ 4 Hz |
| `--kp` | *(osc_pose.json)* | OSC 刚度 |
| `--track_ki` | `0.0` | 右臂外环积分增益；`--lead > 0` 时自动关闭 |
| `--interp` | `on` | 帧间线性插值，平滑台阶输入 |
| `--lead` | `1.0` | 速度前馈系数；`1.0` 补偿 OSC 的结构性速度滞后 |
| `--grasp_x_bias` | `0.012` | 接近杯子到首次合爪期间右臂 x 补偿（米） |
| `--grasp_stiff` | `off` | 与采集保持一致；**旧数据回放保持 off** |
| `--log_txt` | *(不写文件)* | 回放日志写入 txt，含 EEF 误差、接触力、夹爪状态 |
| `--diag` | *(关闭)* | 开启后逐帧输出奇异值/零空间/力矩削顶诊断 |
| `--freeze_frame` | `-1` | 到该帧后冻结目标，用于区分带宽问题与静态偏差 |

---

## 关键参数速查

### 采集与回放一致性

| 场景 | 推荐设置 |
|------|---------|
| 首次采集新数据 | `--grasp_stiff on`（新采集），回放时也须加 `--grasp_stiff on` |
| 回放旧数据集 | `--grasp_stiff off`（默认），保持与采集时物理一致 |
| 提升回放跟踪精度 | `--interp on --lead 1.0 --track_ki 0.0`（均为默认，无需额外传入） |
| 加快采集速度 | 调小 `--steps`（例如 `--steps 10`）或加大 `--kp` |

### 控制频率说明

| 模式 | 控制步频 | 目标点频率 |
|------|---------|-----------|
| 采集（路点插值） | 200 Hz | 200 Hz（连续插值） |
| 回放 `--steps 10` | 200 Hz | 20 Hz（每帧换一次目标） |
| 回放 `--steps 50` | 200 Hz | 4 Hz（每帧换一次目标） |
| 回放 `--interp on` | 200 Hz | 200 Hz（帧内再次插值，信息量仍 20 Hz） |

---

## 常见问题

**Q: `gRPC connect failed` / 仿真无响应？**  
→ 确认 OrcaLab 已启动且仿真状态为运行中；检查 `--orcagym_addr`（默认 `localhost:50051`）。

**Q: `verify_environment.py` 报 `av1_nvenc GPU encode failed`？**  
→ 需要 NVIDIA RTX 40 系或更高 GPU。不录视频时删去 `--cameras` 参数，脚本依然可用。

**Q: 回放时杯子在 180~200 帧附近掉落？**  
→ 原因通常是接触力不足。检查：  
  1. 旧数据回放是否误加了 `--grasp_stiff on`（会破坏软接触的夹持）；  
  2. `--grasp_x_bias` 是否足够（默认 12mm）；  
  3. 尝试适当增大 `--kp`（如 200~300）。

**Q: EEF 跟踪误差大（Y 轴 ±16mm）？**  
→ 确认 `--interp on --lead 1.0`（已为默认）。若误差仍大，可减小 `--steps`（减少台阶输入）。

**Q: `--log_txt` 日志为空或只有部分输出？**  
→ 脚本使用 `_Tee` 同时写 stdout/stderr 和 `orca_logger`。若日志文件已存在会**追加**而非覆盖，
  换一个新路径或 `rm` 旧文件后再运行。

**Q: `[geom扫描] 夹爪 geom 0 个` 警告？**  
→ 仿真中夹爪 body 名不含 `gripper/finger/2f85/pad`，需在 `g1_pick_osc_scripted_dual.py` / 
  `g1_pick_osc_replay_dual.py` 的 `_GRIP_BODY_KW` 中补充关键词。

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `g1_pick_osc_scripted_dual.py` | 脚本化双臂采集入口 |
| `g1_pick_osc_replay_dual.py` | LeRobot 数据集回放入口 |
| `g1_pick_osc_record_waypoints_dual.py` | 交互式路点录制工具 |
| `my_waypoint_dual.yaml` | 采集路点示例 |
| `g1_pick_constraints.py` | 关节限位 / `--grasp_stiff` / `apply_grasp_stiff` |
| `mj_joint_strip.py` | MuJoCo 关节剥离与接触加硬工具函数 |
| `../configs/example.yaml` | 场景任务配置示例 |
| `../../../../docs/lerobot/environment.yml` | Conda 环境（Python 3.12.13） |
| `../../../../docs/lerobot/install_runtime.sh` | pip 依赖安装脚本 |
| `../../../../docs/lerobot/verify_environment.py` | 环境验证脚本 |
