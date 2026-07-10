# G1 手臂/夹爪控制移植说明（fluidmani ← dev_g1_omnipicker）

分支：`dev_G1_fluidmani`  
基准：`fluidmani`  
对照来源：`dev_g1_omnipicker`

---

## 1. 修改目标

把 `dev_g1_omnipicker` 上已调通的 **G1 手臂 + Omnipicker 夹爪** 控制方式，接到 `fluidmani` 的布料遥操入口上，使 NursingHome 等场景下 G1 手感与数采分支一致。

具体目标：

1. **只改 G1**：`openloong` / `tiangong2` 路径保持原样。
2. **夹爪**：使用 `Controller2F85Reverse`，驱动 **joint2** 的 `pctrl`（不再用 fluidmani 原 joint1 + `ControllerOmnipicker`）。
3. **手臂**：使用 dev 的工作位中立姿态，基座参考改为 `body_link1`；Pico 坐标重映射按计划接入（见路线第 ④ 步）。
4. **场景适配**：默认 `mjc-agent-prefix` 改为 `g1_omnipicker`（与 NursingHome MJCF 一致，避免错误的 `g1_omnipicker_usda`）。
5. **不破坏布料主链路**：入口仍是 `data_collection_cloth_tele.py`；布料耦合相关代码尽量不动，仅切换 G1 臂/爪接线。

非目标：

- 不整分支 merge `dev_g1_omnipicker`
- 本轮不强行保留/重写依赖 `ControllerOmnipicker` 的 gripper-trace
- 本轮不强制接入底盘 `front_drive`（可选后续）

---

## 2. 背景与关键概念

### 2.1 控制链路（简图）

```text
Pico 手柄
  → 控制器（算目标）
  → 执行器 ctrl（pctrl / mctrl）
  → MuJoCo 关节运动
  → OrcaLab 画面
```

配置文件只回答「写哪些执行器、范围多少」；控制器回答「扳机/位姿怎么变成数字」；入口脚本负责把二者接上。

### 2.2 `robot_holder1` 与 `body_link1`

NursingHome 模型中大致为：

```text
robot_holder1   ← 底盘/支架根
  └── body_link1  ← 躯干（抬高约 0.65 m）
        └── 双臂 / 头 / 夹爪
```

`base_body` 决定手臂目标相对哪块刚体换算。dev 使用 **`body_link1`（躯干）**。

### 2.3 joint1 / joint2 与从动断开

Omnipicker 单指是四连杆，多个铰链编号为 joint1、joint2、…

| 方案 | 主动驱动 | 控制器 |
|------|----------|--------|
| fluidmani 原 G1 | joint1 的 `pctrl` | `ControllerOmnipicker`（线性 open/close） |
| dev_g1（本分支采用） | joint2 的 `pctrl` | `Controller2F85Reverse`（指数扳机） |

**从动断开**：把四连杆上其它 `pctrl` 改接到 `dummy_joint`，避免与主驱动顶牛。这是 joint1 方案的配套；改用 joint2 + reverse 后，入口不再调用该逻辑。

---

## 3. 修改路线（分层移植）

按层推进，每层可单独验证：

| 步骤 | 内容 | 状态 |
|------|------|------|
| ① | 改 `g1_omnipicker_conf.py`：工作位姿态、`body_link1`、夹爪 joint2 配置 | 已完成 |
| ② | 引入 `controller_2f85_reverse.py`，在 `controllers.py` 注册 `add_gripper_2f85_reverse_pico_controller` | 已完成 |
| ③ | 改 `data_collection_cloth_tele.py` 的 G1 分支：前缀、换 reverse、关掉从动断开、跳过旧 gripper-trace | 已完成 |
| ④ | G1 手臂接入 Pico remap/flip/rotation（与 dev tele 一致） | **待做** |
| ⑤ | NursingHome 联调（纯遥操 → 可选 cloth-coupling） | 待做 |

原则：**只动 G1 分支判断内的逻辑**；`else` 中的 openloong/tiangong2 保持原调用。

---

## 4. 思路说明

### 4.1 为什么不整包替换 fluidmani 的 G1？

fluidmani 已有布料耦合、XPBD、诊断工具等。整包替换成本高、易伤 openloong。  
因此采用 **「手感层用 dev，入口仍挂在 cloth tele」**：

- 夹爪算法与配置对齐 dev
- 布料进程编排尽量保留
- 通过 `if agent_name == "g1_omnipicker"` 隔离影响面

### 4.2 为什么夹爪选 joint2 + reverse，而不是继续 joint1？

需求明确为：手臂和机器手都使用 **dev_g1 已专门优化的控制方式**。  
dev 验证路径是 reverse + joint2；继续 joint1 无法复现该手感，且会与新配置字段（无 open/close）冲突。

### 4.3 配置字段怎么理解？

| 字段 | 作用 |
|------|------|
| `actuator_names` | 真正写入的执行器（最关键） |
| `actuator_ranges` | ctrl 允许范围；reverse 用它算开合 |
| `init_ctrl` | 初始 ctrl |
| `joint_names` | 结构/记录用；夹爪主路径几乎不靠它下指令 |
| `open_ctrl` / `close_ctrl` | 仅旧 Omnipicker 使用 → 已删除 |
| `follower_actuator_names` | 仅旧从动断开使用 → 已删除 |

### 4.4 入口为何要改三处（步骤 ③）？

1. **前缀**：短名会拼成 `前缀_短名`，必须与场景一致。  
2. **换 `add_gripper_2f85_reverse_pico_controller`**：点新控制器。  
3. **去掉 `setup_g1_dual_gripper_actuators`**：避免 joint1 从动断开干扰 joint2 方案。

---

## 5. 已改文件清单

| 文件 | 改动摘要 |
|------|----------|
| `src/conf/g1_omnipicker_conf.py` | 工作位中立角；`base_body=body_link1`；夹爪改为 joint2 + ranges/init |
| `src/controllers/controller_2f85_reverse.py` | **新增**，自 dev 引入 |
| `src/controllers/controllers.py` | 注册 create/add reverse 夹爪 API（保留旧 omnipicker API） |
| `src/examples/dataCollection_cloth/data_collection_cloth_tele.py` | G1：前缀、reverse 夹爪、跳过旧 trace；openloong/tiangong2 未改 |

---

## 6. 联调命令（Docker / OrcaLab 已 Play）

```bash
source /etc/profile.d/conda.sh
conda activate orca-apr24
cd /opt/OrcaManipulation

# 确认分支与 50051
git rev-parse --abbrev-ref HEAD
ss -lntp | grep 50051

python src/examples/dataCollection_cloth/data_collection_cloth_tele.py \
  --level NursingHome \
  --agent_name g1_omnipicker \
  --mjc-agent-prefix g1_omnipicker \
  --no-collect
```

（默认前缀已改为 `g1_omnipicker`，显式传入更稳妥。）

---

## 7. 验收标准

1. NursingHome 下 G1 能启动，不再出现 `g1_omnipicker_usda_idx...` 关节名错误。  
2. 左右夹爪能开合，不出现单侧锁死。  
3. `openloong` / `tiangong2` 启动路径与改前一致。  
4. （步骤 ④ 完成后）Pico 手臂方向与 `dev_g1_omnipicker` 体感接近。  
5. （可选）`--cloth-coupling` 仍能启动。

---

## 8. 后续

1. 完成步骤 ④：G1 专用 Pico 坐标重映射。  
2. 实机/仿真联调并记录手感差异。  
3. 若需要，再适配 gripper-trace 到 `Controller2F85Reverse`，或按需加底盘。
