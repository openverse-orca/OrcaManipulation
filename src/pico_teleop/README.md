# 双臂移动机器人 Pico VR 手柄遥操作

双 E05 六轴机械臂 + 双 M20 平行夹爪 + 轮式浮动底盘（浮动阿克曼转向）的
Pico VR 双手柄遥操作系统。

- **文件**
  - `orcastudio_pico_dual_e05_m20_sites.py` — 遥操作主脚本（数据采集 Manager + 全部控制器）
  - `dual_arm_mujoco_mobile_m20_pico_sites.xml` — MuJoCo 模型（双臂 + 双夹爪 + 浮动底盘）
  - `dual_arm_mujoco.xml` — **原始模型参照**（由 `dual_arm.urdf` 转换的第一版 MJCF：
    无执行器、无轮式底盘、无 M20 夹爪、无 eef site，双臂 16 DOF），供对照下述改动
  - `../meshes/`（body / e05 / gripper 共 16 个文件）— 模型引用的全部 mesh，
    XML 内相对路径 `../meshes/...`，克隆本分支后模型可直接打开
    （含原始车身 `body/base_link.obj`，原始 MJCF 也因此可加载）
  - 本目录是**自包含快照**；实际运行用的文件在 `~/dual_robot/dual_arm_robot/` 下
    （OrcaStudio 从其 `urdf/` 导入、脚本也从那里运行）。

## 相较原始模型，XML 做了哪些改动

原始模型 `dual_arm.urdf`（其第一版 MJCF 转换 `dual_arm_mujoco.xml` 已随包提供，
可直接打开对照）：固定基座，双 E05 装在车身上，车身升降四连杆
（XT/DT/XB/TB）为**活动 revolute 关节**，末端仅 dummy 空夹爪，**无执行器、
无轮式底盘**。本 MJCF（`dual_arm_mujoco_mobile_m20_pico_sites.xml`）相对它的改动：

1. **URDF → MJCF 重构**：radian 单位、obj/stl mesh 引用、inertial 按 mesh 几何重算；
2. **轮式浮动底盘**：固定基座 → 3 个自由关节（世界 x/y 平移 + 车体偏航，高度锁定
   0.58m）；4 个轮子为视觉模型（2 前轮带转向铰链 + 4 滚动铰链），
   后轮滚动角用 equality 与同侧前轮耦合；3 个 motor 执行器（gear=600，
   ±2m/s / ±3rad/s）供遥操驱动底盘，关节 damping=600 承担速度阻尼；
3. **车身升降四连杆 bake 固定**：XT/DT/XB/TB 四个 revolute 按固定角度焊死
   （XT -0.39 / DT 0.02 / XB 0.08 / TB -0.28 rad），无关节无执行器，防止遥操中车身晃动；
4. **双臂 12 个 motor 执行器**（原始无执行器，演化 position → velocity → motor）：
   gear joint1-3=60 / joint4=13 / joint5-6=1，actuatorfrcrange 按电机规格限力；
   速度阻尼配平在关节 damping（J1-3=60 / J4=13 / J5-6=3，MuJoCo 隐式积分、
   无条件稳定），脚本以 100Hz 位置外环下发速度命令，等效位置刚度
   = gear×Kp 逐关节复现旧 position 增益（3000 / ~104 / 55）；
5. **真实 M20 夹爪替换 dummy 末端**：每爪 jaw_a/jaw_b 双 slide 关节
   （行程 [-0.0135, 0]，0=全开），equality 对称耦合，每爪 1 个 motor
   执行器（gear=80，jaw_a damping=82），开度由脚本速度闭环决定
   （不用位置伺服）；夹持失速力 4N/指；pad 摩擦 μ=5（摩擦锥 20N/指，
   治提升/晃动中的动态滑移）+ 全局 impratio=50（治静态爬滑，见 `<option>`）；
6. **新增末端 TCP site**：`left_eef_site` / `right_eef_site`（M20 法兰前 0.155m），
   IK 与遥操绑定直接以 site 世界位姿为目标；
7. **腕部数值稳定**（9-30）：腕部惯量极小，OrcaStudio 1ms 步长下执行器显式
   速度项 kv=7 会自持振荡（现象：夹爪自己转动）；改为执行器 kv=1 + 关节
   damping（现 motor 化后为 damping=3=旧 kv1+原关节2，MuJoCo 隐式积分、
   无条件稳定）；
8. **底盘 mesh 去外露方块**：`base_link_mobile.obj` 为删除两个外露连通域的
   版本（几何层面修改，XML 引用名不变）。

注意：OrcaStudio 导入会**丢弃 armature、重算 inertial**，但**保留 gear/damping**
（position 时代则是保留 kp/kv/damping）——因此物理调参只用这几类属性
（详见常见问题）。XML 的 `impratio` 属性疑似也被导入管道丢弃，脚本启动时
会运行时强制 impratio=50 + noslip_iterations=3 双保险。

## 前置条件

1. OrcaStudio 已打开含本机器人的场景，且**仿真已启动**（gRPC 50051 就绪）；
2. conda 环境：`~/miniconda3/envs/orcalab`（含 `orca_gym`、`gymnasium`、`grpc` 等）。

## 启动

```bash
# 1. 先在 OrcaStudio 里启动仿真，等场景 就绪
# 2. 再运行脚本：
cd ~/dual_robot/dual_arm_robot/urdf
~/miniconda3/envs/orcalab/bin/python orcastudio_pico_dual_e05_m20_sites.py
```

启动后机器人自动摆到**预备姿势**（双臂肘部微弯，夹爪水平指向车头正前方、
左右对称，距肩约 0.70m，预留各方向 ~0.39m 可达余量），随后锁定等待遥操作。

## 手柄操作

| 手柄 | 功能 |
| --- | --- |
| **Grip（侧键）点按** | 该侧手臂开始持续跟随手柄（无需按住）；跟随中再点一下 = 就地重新对中 |
| **扳机（模拟量）** | 该侧 M20 夹爪开度：全松=全开，按到底=全闭，行程连续可调 |
| **右手 A 键** | 全局复位：双臂回预备姿势并锁定、夹爪全开、手柄重新对中（底盘位置/朝向不动）；复位中点 Grip 可立即就地跟随 |
| **左摇杆 x** | 底盘转向（指数手感，小幅精细、推满到最大转角） |
| **右摇杆 y** | 底盘前进/后退 |

配对规则（默认按物理侧）：**动哪只手、操作者视角哪边的臂就动**。
本模型 `left_elfin`/`right_elfin` 命名与物理安装侧相反，站在车尾遥操时
左手柄控制视角左侧的臂（`right_elfin`）；若站在车头**面对**机器人操控，
加 `--no-swap-hands` 恢复按模型命名配对。

### 跟随原理（手柄 ↔ 夹爪绑定）

- 绑定对象是**末端夹爪 TCP（eef site）的世界位姿**，不是关节角；
- 点按 Grip 时以"当前手柄位姿 ↔ 当前夹爪位姿"配对，之后手柄的**位移/旋转
  增量**直接映射到夹爪目标位姿——手往外移，夹爪就往外走；
- 目标限幅（防 IK 卡死乱扭）：锚点球 `--goal-radius`（默认 0.25m）+ 肩球
  `--reach-limit`（默认 1.09m）。够不到远处时：松手 → 摇杆开底盘靠近 → 再点 Grip 重绑。

## 常用命令行参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--arm-mode` | `clutch` | `clutch`=点按 Grip 跟随（推荐）；`follow`=手柄持续跟随；`lock`=手臂锁定只控夹爪 |
| `--no-ready` | 关 | 不摆预备姿势，保持 OrcaStudio 当前姿态 |
| `--no-swap-hands` | 关 | 不按物理侧交换左右配对（站车头前面操控时用） |
| `--arm-scale` | `1.0` | 手柄位移→末端位移比例；臂展不够、手一动就到极限时调小（如 0.5） |
| `--goal-radius` | `0.25` | 单次绑定可移动半径（米） |
| `--reach-limit` | `1.09` | 目标距肩最大半径（米）；`0` 关闭肩球限幅 |
| `--addr` | `localhost:50051` | OrcaGym gRPC 地址 |
| `--agent` / `--prefix` | `dual_arm` / 模型命名空间 | OrcaStudio 导入名不同时覆盖 |

## 常见问题

**`Connection refused (111)`** — OrcaStudio 仿真没启动/未就绪。先启动仿真，
`ss -tln | grep 50051` 确认监听后再跑脚本。

**`MuJoCo has not been initialized`** — 仿真刚启动还在初始化，等几秒重试。

**启动后夹爪自己转动（腕部摆动/旋转）** — 腕部关节惯量极小，执行器显式
速度项 `kv=7` 在 1ms 步长下数值失稳导致自持振荡。修复：速度阻尼改到关节
damping（现 J5/6 `damping="3"`，motor 执行器无速度项；MuJoCo 对 joint
damping 隐式积分，无条件稳定）。**注意 OrcaStudio 导入特性**：

- 修改 XML 后，仅"重新导入文件"只更新**资产库**，**场景里的实体不会自动换新**
  ——需要"删除场景中的机器人实体 → 重新把资产放入场景 → 重启仿真"；
- 导入管道会**丢弃** `armature`、按 mesh 几何**重算** `inertial`，但**保留**
  `gear/damping`（position 时代为 kp/kv/damping）——调物理参数只用这几类；
- 可从 `~/Orca/OrcaStudio/<工程>/tmp/out.xml` 查看当前仿真实际生效的参数。

**把夹爪往外推不跟随/关节乱转** — 旧版预备姿势顶在可达边界（距肩 1.086m，
余量 4mm）近全伸展奇异。已更换为肘弯预备姿势（预备角见脚本
`READY_LEFT_Q/READY_RIGHT_Q`，可自行调整）。

## 已调好的内部参数（改前先看注释）

- IK：DLS 阻尼 λ=0.03、步长 α=0.5、单步关节增量上限 0.18（配合肘弯预备姿势，
  多轮遥操实测调好的"跟手且稳"值，改动需实测回归）；
- 手臂位置外环（motor 速度命令）：`v = ARM_POS_KP·(q_des - q)`，
  Kp=[50,50,50,8,55,55]，限幅 ±2/±3 rad/s；上电初始速度命令为 0，
  锁定/复位/跟随统一走此外环；
- 夹爪电机（两段式堵转）：`v = 22.0 * (目标开度 - 当前开度)`；接近段限速
  0.02 m/s（碰物只轻推不撞飞），检测到堵转（~30ms 不动）锁存后放开到
  0.05 m/s 全力夹紧（4N/指），松扳机自动解锁；开爪方向不限速；
- 防滑三保险：XML `<option> impratio=50`（静态爬滑）+ pad μ=5（动态滑移
  摩擦锥 20N/指）+ 脚本运行时强制 impratio=50、noslip_iterations=3（XML
  属性疑被导入管道丢弃，双保险）；
- A 键复位：按住期间每帧回调 pressed=True，已做边沿检测防复位每 10ms 重启；
- 目标死区/低通：位置死区 5mm、姿态死区 0.02rad、平滑 0.5（防手柄噪声抖动）。
