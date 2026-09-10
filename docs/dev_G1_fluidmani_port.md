# G1 手臂/夹爪控制移植说明（fluidmani ← dev_g1_omnipicker）

分支：`dev_G1_fluidmani`  
基准：`fluidmani`  
对照来源：`dev_g1_omnipicker`（G1 专用入口：`src/examples/dataCollection/g1_omnipicker_collection_tele.py`）

---

## 1. 修改目标

把 `dev_g1_omnipicker` 上已调通的 **G1 手臂 + Omnipicker 夹爪** 控制方式，接到 `fluidmani` 的布料遥操入口上，使 NursingHome 等场景下 G1 手感与数采分支一致。

### 1.1 具体目标

| # | 项 | 内容 |
|---|-----|------|
| 1 | 影响面 | **只改 G1**；`openloong` / `tiangong2` 保持原样 |
| 2 | 夹爪 | `Controller2F85Reverse`，驱动 **joint2** `pctrl`（弃用 joint1 + `ControllerOmnipicker`） |
| 3 | 手臂 | dev 工作位中立角；`base_body=body_link1`；接入 Pico **remap**（步骤 ④） |
| 4 | 场景 | 默认 `mjc-agent-prefix=g1_omnipicker`（对齐 NursingHome MJCF，避免 `g1_omnipicker_usda`） |
| 5 | 布料 | 入口仍为 `data_collection_cloth_tele.py`；布料耦合主链路尽量不动 |

### 1.2 非目标

- 不整分支 merge `dev_g1_omnipicker`
- 本轮不强行保留/重写依赖 `ControllerOmnipicker` 的 gripper-trace
- 本轮不强制接入底盘 `front_drive`（dev 有，布料 tele 可选后续）
- **主路线不冻结手臂关节**（见 §9 潜在路线）

### 1.3 完成度说明（避免误解）

| 阶段 | 含义 |
|------|------|
| ①②③ 已完成 | G1 能启动；夹爪对齐 dev；前缀/配置修正 |
| ④ 完成后 | **手臂 Pico 跟随**才与 dev 数采分支接近 |
| ⑤ 完成后 | NursingHome 纯遥操 + 可选 cloth-coupling 验收 |

**夹爪改动不直接修复手臂反关节问题**；手臂改善来自 `base_body`、中立位、remap、前缀（见 §2.4）。

---

## 2. 背景与关键概念

### 2.1 两条控制链路（必须分开理解）

```text
链路 A — 手臂（手掌+整臂，跟 Pico 位姿）
  L/R_TRANSFORM → ControllerArm.update_goal → OSC → 7×mctrl → arm joint1~7 qpos

链路 B — 夹爪（仅手指四连杆，跟 Pico 扳机）
  L/R_TRIGGER → Controller2F85Reverse → 2×pctrl(joint2) → 手指 qpos（约束带动 joint1/3/4）
```

- **扳机（trigger）只走链路 B**，不参与手臂 OSC。
- 之前「四连杆示意图」仅描述链路 B。

### 2.2 `base_body` 与 Pico 位移

Pico 上报的 `position` 是 **相对 VR 追踪零点**（开遥操时手柄初始位姿），不是相对 `base_body` 直接测量。

`ControllerArm.update_goal` 把它 **解释成** `base_body` 坐标系（B 系）下的增量：

```text
goal_pos_B = initial_ee_pos_B + relative_position   # B 系末端目标
goal_pos_world = R(base_body) @ goal_pos_B + pos(base_body)
```

| 方案 | `base_body` | 影响 |
|------|-------------|------|
| fluidmani 旧 | `robot_holder1`（底盘支架） | B 系原点在地面侧，与「相对身体遥操」直觉不符 |
| dev / 本分支 | `body_link1`（躯干） | B 系原点在肩附近，世界系目标方向正确 |

### 2.3 夹爪：joint1 / joint2 与从动断开

Omnipicker 单指为四连杆；MJCF 中每铰链各有 `pctrl`。

| 方案 | 主动驱动 | 控制器 | 从动断开 |
|------|----------|--------|----------|
| fluidmani 原 | joint1 `pctrl` | `ControllerOmnipicker`（线性 open/close） | joint2/3/4 的 `pctrl` → `dummy_joint` |
| dev / 本分支 | joint2 `pctrl` | `Controller2F85Reverse`（指数扳机） | **不调用**断开逻辑 |

**从动断开**：把非主驱动 `pctrl` 的 `trnid` 改接到 `dummy_joint`，力矩不再推真实手指关节，避免与主驱动顶牛。  
joint2 方案下 **不能**沿用 joint1 方案的 follower 列表（会把 joint2 也卸掉）。

若 joint2 方案出现四连杆顶牛/单侧锁死，回退应为 **新 follower 列表（断开 joint1/3/4，保留 joint2）**，而非恢复 joint1 主驱动。

### 2.4 手臂：OSC 末端、7 关节、remap

**OSC 末端（eef）**：跟踪 `ee_center_site_l/r`（挂在 `gripper_base_link` 上的 site），即 **掌座/腕法兰控制点**，不是指尖。手指开合不在 OSC 的 `joint_indexes` 内。

**7×mctrl**：G1 每臂 `joint1~7` 各一个力矩执行器，标准 7-DOF 冗余臂，非「三臂多出一个废关节」。

**remap（坐标重映射）**：Pico 位移进 `update_goal` **之前**，做轴重排、翻转、固定旋转偏移，使「手柄前/上/右」对齐 G1 安装方向。

| 入口 | G1 专用 remap |
|------|----------------|
| fluidmani `data_collection_cloth_tele.py` | **无**（通用 `add_arm_osc_pico_controller`） |
| dev `g1_omnipicker_collection_tele.py` | **有** |
| 本分支 cloth tele | **待步骤 ④ 接入** |

### 2.5 `mjc-agent-prefix`

配置写短名（如 `idx21_arm_l_joint1`），运行时拼成 `前缀_短名`。  
NursingHome 场景实体前缀为 **`g1_omnipicker`**（无 `_usda`）；fluidmani 旧默认 `g1_omnipicker_usda` 会导致关节名错误。

`--cloth-coupling` 时 `apply_runtime_cloth_overrides` 会覆盖 `orcagym.mjc_agent_prefix`，但 `cloth_sim_config.g1_omnipicker.json` 中 `orcagym_rigid_body_map` 仍可能残留 `_usda_` 硬编码行，联调时需核对扫描结果（见 §7.2）。

---

## 3. 修改路线（分层移植）

| 步骤 | 内容 | 状态 |
|------|------|------|
| ① | `g1_omnipicker_conf.py`：工作位、`body_link1`、夹爪 joint2 | 已完成 |
| ② | `controller_2f85_reverse.py` + `controllers.py` 注册 | 已完成 |
| ③ | `data_collection_cloth_tele.py` G1 分支：前缀、reverse、去从动断开、跳过旧 trace | 已完成 |
| ④ | G1 手臂 Pico remap（见 §4） | **待做** |
| ⑤ | NursingHome 联调（§7） | 待做 |

原则：**只动 `if agent_name == "g1_omnipicker"` 内逻辑**；`else` 中 openloong/tiangong2 不变。

---

## 4. 步骤 ④ 细化：手臂 remap 移植

### 4.1 移植源

`origin/dev_g1_omnipicker:src/examples/dataCollection/g1_omnipicker_collection_tele.py`

不要从通用 `data_collection_tele.py` 抄（该文件不含 G1 remap）。

### 4.2 常量（与 dev 保持一致）

左臂：

```python
L_ARM_ROTATION_OFFSET = np.array([np.pi / 2, 0, 0])
L_ARM_POSITION_REMAP = [0, 2, 1]          # 新 (x,y,z) = 旧 (x,z,y)
L_ARM_POSITION_FLIP = np.array([1.0, 1.0, -1.0])
```

右臂：

```python
R_ARM_ROTATION_OFFSET = np.array([-3 * np.pi / 2, 0, 0])
R_ARM_POSITION_REMAP = [0, 2, 1]
R_ARM_POSITION_FLIP = np.array([1.0, 1.0, -1.0])
```

### 4.3 接入方式

在 `data_collection_cloth_tele.py` 的 G1 分支中：

1. 复用 dev 的 `make_rotated_callback` + `add_arm_osc_pico_controller_with_rotation`（或抽到 `src/controllers/` / `src/envs/` 小模块，避免复制三份）。
2. **替换**当前对左右臂的 `controllers.add_arm_osc_pico_controller(...)` 调用。
3. 依赖：`scipy.spatial.transform.Rotation`、`create_arm_osc_controller`（`controllers.py` 已有）。

### 4.4 数据流（一帧，链路 A）

```text
Pico position/rotation（相对 VR 零点）
  → abstract_device：Unity → MuJoCo 轴交换
  → [G1 remap] 轴重排 + 翻转 + 旋转偏移
  → ControllerArm.update_goal（B 系 = body_link1）
  → OSC.set_goal（世界系 ee_center_site 目标）
  → 7×mctrl
  → mj_step → qpos[joint1~7]
```

### 4.5 步骤 ④ 验收

- 手柄前推 → 掌座大致前伸（非横漂/反拧）
- 与 dev `g1_omnipicker_collection_tele.py` 同场景下方向体感接近
- 可选：`--pico-delta-trace` / `CLOTH_PICO_DELTA_TRACE=1` 对比 `pico_mjc_delta_trace.csv`

---

## 5. fluidmani 旧方案 vs 本分支对照

### 5.1 配置与入口

| 项 | fluidmani 旧 | 本分支 |
|----|-------------|--------|
| `base_body` | `robot_holder1` | `body_link1` |
| 左臂中立角 joint1~4 | `0, 0, 0, -0.87` | `-1.42, 0.88, 1.54, -1.48` |
| 夹爪主驱动 | joint1 `pctrl` | joint2 `pctrl` |
| 夹爪控制器 | `ControllerOmnipicker` | `Controller2F85Reverse` |
| 从动断开 | 调用 `setup_g1_dual_gripper_actuators` | 不调用 |
| 默认前缀 | `g1_omnipicker_usda` | `g1_omnipicker` |
| 手臂 remap | 无 | 待步骤 ④ |

### 5.2 夹爪控制器差异

| 维度 | `ControllerOmnipicker` | `Controller2F85Reverse` |
|------|------------------------|-------------------------|
| 扳机曲线 | 线性 0→1 | 指数 `(e^(k·t)-1)/(e^k-1)` |
| 目标 ctrl | `open_ctrl`↔`close_ctrl` 插值 | 从 `actuator_range` 下界向上推 |
| 配置字段 | 需 open/close | 仅需 ranges + init_ctrl |

### 5.3 配置字段（`g1_omnipicker_conf.py`）

| 字段 | 作用 |
|------|------|
| `actuator_names` | 遥操实际写入的夹爪 `pctrl` |
| `actuator_ranges` / `init_ctrl` | reverse 算 ctrl |
| `joint_names` | 结构/诊断；主路径不靠它下指令 |
| `motors_names` | 手臂 7×`mctrl` |
| `ee_site_name` | OSC 末端 site |
| `open_ctrl` / `close_ctrl` / `follower_actuator_names` | 已删除（仅旧方案） |

---

## 6. 已改文件清单

| 文件 | 改动摘要 |
|------|----------|
| `src/conf/g1_omnipicker_conf.py` | 工作位；`base_body=body_link1`；夹爪 joint2 |
| `src/controllers/controller_2f85_reverse.py` | 新增，自 dev 引入 |
| `src/controllers/controllers.py` | 注册 reverse 夹爪 API |
| `src/examples/dataCollection_cloth/data_collection_cloth_tele.py` | G1：前缀、reverse、跳过旧 trace |

**待改（步骤 ④）**：`data_collection_cloth_tele.py`（或新建 G1 arm 辅助模块）接入 remap。

---

## 7. 联调与验收

### 7.1 命令（Docker / OrcaLab 已 Play）

```bash
source /etc/profile.d/conda.sh
conda activate orca-apr24
cd /opt/OrcaManipulation

git rev-parse --abbrev-ref HEAD
ss -lntp | grep 50051

# 阶段 1：纯遥操（步骤 ①②③）
python src/examples/dataCollection_cloth/data_collection_cloth_tele.py \
  --level NursingHome \
  --agent_name g1_omnipicker \
  --mjc-agent-prefix g1_omnipicker \
  --no-collect

# 阶段 2：步骤 ④ 完成后重复上式，对比手臂方向

# 阶段 3：布料耦合（步骤 ⑤）
python src/examples/dataCollection_cloth/data_collection_cloth_tele.py \
  --level NursingHome \
  --agent_name g1_omnipicker \
  --mjc-agent-prefix g1_omnipicker \
  --cloth-coupling \
  --no-collect
```

### 7.2 验收标准

| # | 项 | 阶段 |
|---|-----|------|
| 1 | 启动无 `g1_omnipicker_usda_idx...` 关节名错误 | ①②③ |
| 2 | 左右夹爪可开合，无单侧锁死 | ①②③ |
| 3 | `openloong` / `tiangong2` 路径不变 | ①②③ |
| 4 | Pico 手臂方向与 dev 体感接近 | **④ 后** |
| 5 | `--cloth-coupling` 能启动；`rigid_body_map` 无错误 `_usda_` 残留 | ⑤ |
| 6 | （可选）`grip_triggers.txt` 随扳机更新；布料交互正常 | ⑤ |

### 7.3 联调顺序建议

1. `--no-collect` 纯遥操 → 验启动、夹爪、前缀  
2. 完成 ④ → 验手臂跟随  
3. 加 `--cloth-coupling` → 验 XPBD / body track / grip lock  

**说明**：跳过 gripper-trace **不影响**布料抓取；XPBD 走 `grip_triggers.txt` + Body Track，与 trace 无关。

---

## 8. 后续（主路线）

1. 完成步骤 ④（remap 移植 + 小模块去重）。  
2. 步骤 ⑤ NursingHome 联调并记录与 dev 差异。  
3. 若 joint2 四连杆顶牛：评估 **joint1/3/4 follower 断开**（保留 joint2），勿回退 joint1 主驱动。  
4. 可选：gripper-trace 适配 `Controller2F85Reverse`；`front_drive` 底盘。  
5. 可选：更新 `cloth_sim_config.g1_omnipicker.json` 中 `_usda_` 硬编码行。

---

## 9. 潜在路线：冻结「中臂–小臂」关节（实验性，非主路线）

### 9.1 动机

G1 左臂从躯干到末端可粗分为三段连杆：

```text
大臂：arm_l_link1 ~ link3  （joint1~3，肩+上臂）
中臂：arm_l_link4          （joint4，肘段）
小臂：arm_l_link5 ~ link6 + arm_l_end_link（joint5~7，前臂+腕+法兰）
```

口语上像「大臂 + 中臂 + 小臂」三段，对应 7 个转动关节。若遥操中前臂某轴冗余导致路径别扭，可尝试 **冻结中臂与小臂之间的关节**，使运动学上更接近「大臂 + 小臂」两段感。**dev 与当前主路线均未采用**；仅当步骤 ④ 完成后手臂仍不理想时再试。

### 9.2 候选关节（联调时确认）

| 候选 | 关节名（左臂短名） | 连接 | 语义 |
|------|-------------------|------|------|
| **A（优先试）** | `idx25_arm_l_joint5` | link4 → link5 | 肘后前臂起始轴 |
| B | `idx26_arm_l_joint6` | link5 → link6 | 前臂中段 / 腕前 |

右臂对应：`idx65_arm_r_joint5`、`idx66_arm_r_joint6`。

### 9.3 实现思路（择一，需实测）

| 方式 | 做法 | 优点 | 风险 |
|------|------|------|------|
| OSC 降维 | 从 `arm_config["joint_names"]` / OSC `joint_indexes` 去掉该关节 | 不改 MJCF | 末端少 1 DOF，可达空间缩小 |
| 锁定 qpos | 每步将目标关节角固定为中立值 | 实现简单 | 与 OSC 其他 6 轴可能冲突 |
| 停用 mctrl | 该关节 `mctrl` 恒 0 或从控制器列表剔除 | 改动小 | 重力下可能下垂 |
| MJCF 等式约束 | `equality/weld` 或锁定 joint range=0 | 物理一致 | 需改场景资产、影响面大 |

推荐实验顺序：**OSC 降维（软件）→ 停用 mctrl → 再考虑 MJCF**。

### 9.4 验收与回退

- **试**：步骤 ④ 已完成前提下，对比冻结前后「反关节感」、可达范围、布料抓取姿态。  
- **通过**：反关节减轻且 NursingHome 任务可达性可接受。  
- **不通过**：回退冻结，保持 7-DOF 全控；主路线仍以 remap + `body_link1` + 工作位为准。

### 9.5 与主路线关系

```text
主路线（默认）：7 关节全控 + remap + body_link1 + 工作位
潜在路线（备选）：在上述基础上，可选冻结 joint5（或 joint6）做对比实验
```

**不得**用冻结关节替代步骤 ④ remap；不得在未完成 ④ 时把手臂问题归因于「关节太多」。

---

## 10. 附录：手臂关节与连杆对照（左臂）

| 关节 | 连杆 | 执行器 | OSC |
|------|------|--------|-----|
| joint1 | link1 | `idx21_arm_l_joint1_mctrl` | 是 |
| joint2 | link2 | `idx22_arm_l_joint2_mctrl` | 是 |
| joint3 | link3 | `idx23_arm_l_joint3_mctrl` | 是 |
| joint4 | link4 | `idx24_arm_l_joint4_mctrl` | 是 |
| joint5 | link5 | `idx25_arm_l_joint5_mctrl` | 是（潜在路线可冻结） |
| joint6 | link6 | `idx26_arm_l_joint6_mctrl` | 是（潜在路线备选） |
| joint7 | end_link | `idx27_arm_l_joint7_mctrl` | 是 |
| — | `ee_center_site_l` on gripper_base | — | **OSC 跟踪点** |
| finger* | gripper 四连杆 | `*_pctrl` | **否**（扳机 / reverse） |
