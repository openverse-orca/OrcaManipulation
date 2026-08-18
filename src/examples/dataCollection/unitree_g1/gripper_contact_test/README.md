# 宇树 G1 臂 + OmniPicker 夹爪 · 遥操作采集与接触力诊断（测试脚本）

本目录用于把 `g1_pick_collection_tele_lerobot_gripper_test.py` 交给开发复现。脚本在原
`g1_pick_with_gripper_collection_tele_lerobot.py` 基础上加了三样东西：

1. `--sim_actor_col off`：运行时关闭 OrcaStudio `ActorManipulator` 拖拽球的碰撞并把它归位到 mocap 锚点。
2. `--contact_probe`：工具 ↔ 桌面 / 工具箱 / 夹爪 / 腕部的逐接触点力学采样，落 CSV + 周期打摘要。
3. `--time_step` / `--frame_skip`：物理步长与子步数暴露到命令行，用于对比不同控制周期下的物理表现。

`artifacts/` 里是两轮真实运行的产物（视频 + 诊断日志 + 接触 CSV），可直接对照。

---

## 1. 环境

与仓库既有宇树采集流程完全一致，详见 [`docs/unitree_g1_collection.md`](../../../../../docs/unitree_g1_collection.md)
的「环境」一节。简版：

```bash
conda activate orcalab_lerobot
bash scripts/install_runtime.sh
```

请勿自行安装 `pin` / `pinocchio`，Conda 已提供 Pinocchio 3.9.0 + CasADi 3.7.2。
首次启动会在 `src/examples/dataCollection/_ik_cache/` 生成 IK 缓存，耗时数秒。

---

## 2. 场景与相机

**注意：本脚本不读布局 JSON**，以 OrcaStudio 当前已加载的场景为准。

1. 启动 OrcaLab 7.1，加载含 g1_pick + OmniPicker 的场景（本次测试用 `uni_test`）。
2. 确认机器人 actor 名为 `g1_pick_with_gripper_usda_1`（即 `--agent_name` 的值）。
3. 配两路相机，**端口与既有宇树文档不同**：

   | 相机 | Color Port |
   |------|-----------|
   | 头部 `head` | 7090 |
   | 右腕 `wrist_r` | 7080 |

   每路勾选 **UseNvEnc**、**Color Camera**、**Enable**；两路都配完后再统一勾 **Recording**。
4. 点 OrcaLab 运行按钮，等界面显示仿真已运行。gRPC 默认 `localhost:50051`。

---

## 3. 最小化完整操作流程

顺序不能颠倒：**先端口转发 → 再启动脚本 → 最后在头显开页面**。

### 3.1 连接头显并做端口转发

```bash
adb devices
adb reverse tcp:8012 tcp:8012
```

多设备时用 `adb -s <序列号> reverse tcp:8012 tcp:8012`。

### 3.2 启动采集脚本

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection/unitree_g1/gripper_contact_test

OMP_NUM_THREADS=1 python -u g1_pick_collection_tele_lerobot_gripper_test.py \
  --xr_backend televuer --tv_no_tls --tv_goal_mode rebased_tv \
  --agent_name g1_pick_with_gripper_usda_1 \
  --task "抓取测试" \
  --lerobot_out g1_unitree_gripper_test5 \
  --repo_id local/g1_pick_gripper_test5 \
  --fps 20 --clock wall --cameras head,wrist_r \
  --orcagym_addr localhost:50051 \
  --sim_actor_col off \
  --time_step 0.001 --frame_skip 5 \
  --contact_probe --contact_every 5
```

`OMP_NUM_THREADS=1` 是必要的：MuJoCo 单步在多线程下反而更慢，会拖垮实时率。

等终端打印出访问地址后再往下走：

```text
http://127.0.0.1:8012/
```

如需无线连接（头显直连本机 IP），必须同时给 `--tv_host <本机IP>`、`--tv_cert_file`、
`--tv_key_file`；非回环地址不启用 TLS 时头显无法进入沉浸式会话。

### 3.3 在头显里开始采集

```bash
adb reverse tcp:8012 tcp:8012
adb shell am start -a android.intent.action.VIEW -d 'http://127.0.0.1:8012/'
```

1. 头显页面里点 **Virtual Reality**（页面出现网格背景说明**没有**进入沉浸式，需重开）。
2. 轻按**左手 squeeze（侧握键）**开始采集，终端出现「正在采集第 N 集」。
3. 每次重跑前，先关掉头显当前浏览器页面，再按 3.1 → 3.2 → 3.3 重来一遍。

断点续采追加 `--resume`。

### 3.4 产物位置

| 内容 | 路径 |
|------|------|
| LeRobot 数据集 | `L_dataset/unitree/<--lerobot_out>/` |
| 分层遥操诊断日志 | `/tmp/g1pick_tele_diag/debug_tele_g1pick_<时间戳>.txt` |
| 接触力 CSV（每集一份） | `/tmp/g1pick_tele_diag/contact/contact_ep<NNN>_<时间戳>.csv` |

诊断目录可用 `--diag_log_dir` 改。

---

## 4. 关键日志怎么看

### 4.1 `[CONTACT]` 周期摘要

每 50 个控制步打一行 top-6（按饱和度排序）；每集结束打一次完整汇总（含按连杆对的 top20）。
取自 `artifacts/run_B_dt5ms_test5/diag_tele.txt` 的真实输出：

```text
[CONTACT][汇总 #7710] 按类别:
  tool|table: n=29347 Fn均=0.8N Fn峰=38.3N 贯穿=-25.69mm 饱和=25.69 滑移=1.00 隧穿=5.95 阻抗=0.97
  grip|table: n=142   Fn均=11.9N Fn峰=33.3N 贯穿=-15.15mm 饱和=15.15 滑移=1.00 隧穿=2.75 阻抗=0.95
  tool|grip:  n=1704  Fn均=8.3N  Fn峰=80.8N 贯穿=-10.35mm 饱和=10.35 滑移=1.00 隧穿=2.35 阻抗=0.97
  grip|grip:  n=18469 Fn均=20.8N Fn峰=177.2N 贯穿=-10.15mm 饱和=10.15 滑移=1.00 隧穿=0.84 阻抗=0.95
  按连杆对（饱和度 top20）:
    tool|table Screwdriver~Static_Phy:      n=805 Fn峰=16.6N 贯穿=-25.69mm 饱和=25.69 隧穿=3.15
    tool|table Screwdriver~Table_green_03:  n=670 Fn峰=34.2N 贯穿=-17.14mm 饱和=17.14 隧穿=5.95
```

字段读法：

| 字段 | 含义 | 异常判据 |
|------|------|----------|
| `Fn均` / `Fn峰` | 单接触点法向力均值 / 峰值 | `Fn峰` 远超物体自重（如 0.5kg 工具出现上百 N）说明有寄生约束或冗余接触点 |
| `贯穿` | 最深穿透 `dist`（负值为穿透，mm） | 超过几何壁厚即为穿模 |
| `饱和` | `\|dist\| / solimp[2]`，接触刚度饱和度 | `>1` 表示已进入非线性饱和区，力不再随穿透线性增长，会「楔住」 |
| `滑移` | `Ft / (mu·Fn)` | `≈1.00` 表示已到摩擦极限，正在滑 |
| `隧穿` | `\|vn\|·dt / solimp[2]` | `>1` 表示单步位移超过接触带宽，可能直接穿过去不产生接触 |
| `阻抗` | 求解器 `efc_KBIP` 的 impedance | 逼近 `solimp[1]`（0.95/0.99）表示阻抗饱和 |

### 4.2 接触力 CSV

表头（每行一个接触点、一个采样步）：

```text
step,sim_time,cls,part1,part2,body1,body2,geom1,geom2,
dist,Fn,Ft,mu,slip_ratio,solref_t,solref_d,solimp_w,sat_ratio,
vn,vt,vdt_mm,tunnel_ratio,efc_imp,cz
```

`cls` 为接触类别（`tool|table`、`tool|box`、`tool|grip`、`grip|grip`、`tool|wrist` 等），
`part1/part2` 是归一化后的短名（`Screwdriver`、`Table_green_03`、`gr_in4`、`gr_ou2`、`pad`…）。
按 `cls` + `part` 过滤即可定位到具体连杆对。

### 4.3 已知可复现现象

对比 `artifacts/` 两轮（同一场景、同一任务，只改 `dt`）：

| | run_A（`--time_step 0.002 --frame_skip 10`，dt=20ms） | run_B（`--time_step 0.001 --frame_skip 5`，dt=5ms） |
|---|---|---|
| `tool\|box` 单点 `Fn峰` | **1324.5 N** | 77.3 N |
| 单几何对同时接触点数 | 9（螺丝刀楔进工具箱卡槽） | 显著减少 |
| 主观手感 | 工具明显「变重」、被挤出、遥操作卡顿 | 明显缓解 |

复现该数字：

```bash
zcat artifacts/run_A_dt20ms_test4/contact_ep001.csv.gz \
  | awk -F, '$3=="tool|box"{if($11>m)m=$11}END{printf "Fn峰=%.1fN\n", m}'
# → Fn峰=1324.5N
```

run_A 的 ep001 里 `tool|box` 出现过 ΣFn ≈ 6577 N 持续 10s 以上，而物体自重仅 5.2 N、
法向速度为 0、穿透固定在 −6.6 mm —— 即约 6572 N 是矢量相互抵消的内部自平衡力，物体被
「冻」在原地不动。成因是螺丝刀楔入卡槽后，网格碰撞在同一几何对上生成 9 个各自独立按穿透量
出力的接触点。大 `dt` 下物体单步「传送」进楔住状态是触发条件，所以缩小 `dt` 能大幅缓解。

同一份日志里还能看到 `tool|table` 的 `Screwdriver~Static_Phy` 贯穿 −25.69 mm、饱和度 25.69、
隧穿 5.95 —— 接触带宽 `solimp[2]` 未随几何壁厚缩放，薄壁件会直接饱和/穿模。

---

## 5. 命令行参数说明

只列与本次测试相关的；其余参数 `--help` 可查。

### 物理 / 诊断（本脚本新增或需重点关注）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--time_step` | `0.005` | MuJoCo `opt.timestep`（秒）。`0.005` 是场景 XML 自带值；`0.001` 精度更高但实时率下降 |
| `--frame_skip` | `8` | 每控制步物理子步数，`env.dt = time_step × frame_skip`。加大可摊薄每圈固定的 IK/render 开销 |
| `--sim_actor_col` | `off` | `off` = 关闭 `ActorManipulator` 球碰撞并归位到 mocap 锚点，消除埋在地下约 1000m 的寄生接触约束；`keep` = 保留原行为（用于 A/B） |
| `--contact_probe` | 关 | 开启接触力采样，写 `contact_*.csv` 并周期打 `[CONTACT]` 摘要 |
| `--contact_every` | `10` | 接触力采样间隔（控制步）。测试用 `5` |
| `--diag_tele` / `--no-diag_tele` | 开 | 分层遥操诊断日志 |
| `--diag_every` | `50` | 诊断日志节流间隔（步） |
| `--diag_log_dir` | `/tmp/g1pick_tele_diag` | 诊断日志与接触 CSV 的输出根目录 |

### 遥操作 / XR

| 参数 | 默认 | 说明 |
|------|------|------|
| `--xr_backend` | `pico` | 本测试必须用 `televuer` |
| `--tv_no_tls` | 关 | USB + `adb reverse` 场景下必开：走明文 HTTP/WS，免证书，且规避 vuer 0.0.60 HTTPS 丢端口的 bug |
| `--tv_goal_mode` | `rebased_tv` | TeleVuer 目标模式（clutch 相对模式） |
| `--tv_ee_dx` | `0.03` | TeleVuer→Orca 末端 TransX 校正（m） |
| `--tv_position_scale` | `1.0` | 位置尺度 |
| `--dry_run_tele` | 关 | 只算并打印 IK 目标与误差，不驱动臂部 actuator（排查用） |

### 采集 / 数据集

| 参数 | 默认 | 说明 |
|------|------|------|
| `--agent_name` | `g1_pick_with_gripper_usda_1` | 仿真中机器人 actor 名（关节前缀），须与场景一致 |
| `--orcagym_addr` | `localhost:50051` | OrcaGym gRPC 地址 |
| `--cameras` | `head,wrist_r` | 启用的相机；端口 head=7090 / wrist_r=7080 |
| `--cam_resolution` | `480x640` | `HxW` |
| `--fps` | `20` | 采集帧率 |
| `--clock` | `wall` | `wall` 按墙钟节流，`sim` 按仿真时间 |
| `--lerobot_out` | — | 数据集输出目录；相对路径落到 `L_dataset/unitree/` 下 |
| `--repo_id` | `local/g1_pick_with_gripper` | LeRobot `repo_id` |
| `--task` | — | 任务描述文本 |
| `--resume` | 关 | 追加到已有数据集 |
| `--level` / `--task_config` | `default` / `example.yaml` | 场景名与任务配置（本脚本不读布局 JSON） |

---

## 6. 本目录改动了哪些文件

除本目录外，脚本依赖以下已随本次提交更新的模块（都在原位，无需额外配置）：

```text
src/controllers/g1_pick_controllers.py          （新增）
src/controllers/controller_task.py
src/controllers/controller_arm.py
src/controllers/g1_pick_dual_arm_controller.py
src/dataCollectionManager/data_collection_manager.py
src/dataStorage/lerobot_camera.py
src/dataStorage/lerobot_data_storage.py
src/dataStorage/encoder_proc.py
```

脚本靠自身位置推导 `sys.path`（`../../../..` → `src/`），放在本目录下即可直接运行，不要单独挪走。

---

## 7. artifacts 目录

```text
artifacts/
├── run_A_dt20ms_test4/          # --time_step 0.002 --frame_skip 10
│   ├── videos/                  # cam_head / cam_wrist_r 的 mp4，各 2 集
│   ├── meta/                    # LeRobot info/episodes/tasks 元数据
│   ├── diag_tele.txt            # 完整分层诊断日志
│   └── contact_ep00{1,2,3}.csv.gz
└── run_B_dt5ms_test5/           # --time_step 0.001 --frame_skip 5
    └── （同上结构）
```

说明：

- 按要求**未包含 parquet**（`data/` 目录），只有视频与元数据；如需完整数据集请自行重跑。
- 每轮采了 3 集、落盘保留 2 集，所以 CSV 有 3 份而视频只有 2 集。
- CSV 解压：`gunzip -k contact_ep001.csv.gz`（原始约 4–6 MB/集）。
