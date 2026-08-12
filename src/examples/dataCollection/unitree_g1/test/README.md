# G1 + OmniPicker 夹爪遥操作数据采集（测试版）

本目录为 **支链测试脚本**，对应 G1 机械臂 + OmniPicker 夹爪（双指夹爪）的 VR 遥操作数据采集流程。  
主链路共用模块（`lerobot_data_storage`、`g1_pick_unitree_arm_ik` 等）位于上级目录。

---

## 目录结构

```
test/
├── g1_pick_collection_tele_lerobot_gripper_test.py   # 采集主脚本
├── uni_test2.json                                     # OrcaLab 场景布局
├── run.sh                                             # 一键运行示例
└── README.md                                          # 本文件
```

---

## 前置条件

### 1. 订阅 USDA —— G1 + 夹爪机器人资产

在 **OrcaLab 资产草稿箱**中订阅以下 USDA：

| 字段 | 值 |
|------|-----|
| Actor 名 | `g1_pick_with_gripper_usda_1` |
| 资产路径 | `assets/13951baeb514b4b9/default_project/prefabs/g1_pick_with_gripper_usda` |

订阅步骤：OrcaLab → 资产中心 → 草稿箱 → 搜索 `g1_pick_with_gripper_usda` → 订阅并等待同步完成。

### 2. 加载场景

在 OrcaLab 中加载本目录的 `uni_test2.json`（包含工具架、抓取物体及 G1+夹爪机器人）。  
相机端口映射：
- `head` → 7090
- `wrist_r` → 7080

### 3. 连接 Pico VR 头显

通过 USB 连接 Pico，确认 `adb devices` 可见设备。

---

## 快速运行

```bash
# 在本目录（test/）下执行
bash run.sh /path/to/your/output_dataset
```

或手动分步执行：

```bash
# 步骤 1：ADB 端口转发（Pico → PC）
adb reverse tcp:8012 tcp:8012

# 步骤 2：在 Pico 端打开 TeleVuer 网页
adb shell am start -a android.intent.action.VIEW -d 'http://127.0.0.1:8012/'

# 步骤 3：启动采集脚本
python -u g1_pick_collection_tele_lerobot_gripper_test.py \
    --level default \
    --task_config ../../common/example.yaml \
    --scene_json uni_test2.json \
    --agent_name g1_pick_with_gripper_usda_1 \
    --task "抓取测试" \
    --lerobot_out /YOUR/OUTPUT/PATH/g1_unitree_gripper \
    --repo_id local/g1_pick_gripper \
    --fps 20 \
    --clock wall \
    --cameras head,wrist_r \
    --cam_resolution 480x640 \
    --orcagym_addr localhost:50051 \
    --xr_backend televuer \
    --tv_no_tls \
    --tv_goal_mode rebased_tv \
    --tv_ee_dx 0.03 \
    --log_file /tmp/gripper_collect.txt
```

---

## 控制说明

| 操作 | 效果 |
|------|------|
| 右手 **Squeeze**（握紧） | 开始 / 停止本集采集 |
| 左手 **Squeeze** | 丢弃当前集（不保存） |
| 右手移动 | 控制右臂末端 EE（遥操作） |
| 左臂 | 侧平举锁死，不参与 IK 计算 |

---

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tv_ee_dx` | 0.03 | EE 每步位移缩放（增大 → 动作更灵敏） |
| `--tv_goal_mode` | rebased_tv | IK 目标模式 |
| `--encode_backend` | inproc | 编码后端；`subproc` 可消除 GIL 卡顿 |
| `--cam_resolution` | 480x640 | H×W，需与场景相机分辨率一致 |
| `--fps` | 20 | 采集帧率 |

---

## 数据集依赖（主链路）

本脚本使用以下主链路模块（位于 `src/dataStorage/` 和 `src/controllers/`）：

- `lerobot_data_storage.py` — LeRobot v2.1 写入器（含 NVENC 编码、subproc 后端）
- `g1_pick_with_gripper_data_storage.py` — G1+夹爪 state/action schema
- `lerobot_camera.py` — 相机帧采集工具
- `encoder_proc.py` — forkserver 子进程编码器
- `g1_pick_unitree_arm_ik.py` — 宇树 G1 双臂 IK（含 `fixed_left_q` 左臂锁定）
- `g1_pick_dual_arm_controller.py` — 双臂控制器
