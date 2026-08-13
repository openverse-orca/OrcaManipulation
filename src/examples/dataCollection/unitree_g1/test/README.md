# G1 + OmniPicker 夹爪遥操作（测试目录）

本目录提供 G1 机械臂 + OmniPicker 双指夹爪的 VR 遥操作：纯跟手（不落盘）与数据采集两套入口。共用模块位于仓库 `src/`。

---

## 目录结构

```
test/
├── g1_pick_teleop_gripper_test.py                     # 纯遥操作（不落盘）
├── uni_test.json                                      # 纯遥操作场景布局
├── g1_pick_collection_tele_lerobot_gripper_test.py    # 遥操作采集（落盘）
├── uni_test2.json                                     # 采集场景布局
├── run.sh                                             # 采集一键运行示例
├── run_teleop.sh                                      # 纯遥操作一键运行示例
└── README.md
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

纯遥操作请加载本目录的 `uni_test.json`。采集请加载 `uni_test2.json`。  
两份布局均包含工具架、抓取物体及 G1+夹爪机器人。

采集时的相机端口：
- `head` → 7090
- `wrist_r` → 7080

纯遥操作不采相机，无需配置 Recording。

### 3. 连接 Pico VR 头显

通过 USB 连接 Pico，确认 `adb devices` 可见设备。首次使用请按仓库 `docs/unitree_g1_collection.md` 禁用房间标定向导，并做 `adb reverse tcp:8012 tcp:8012`。

---

## 纯遥操作（不落盘）

OrcaLab 加载 `uni_test.json` 并运行仿真后，在本目录执行：

```bash
bash run_teleop.sh
```

或：

```bash
adb reverse tcp:8012 tcp:8012
adb shell am start -a android.intent.action.VIEW -d 'http://127.0.0.1:8012/'

python -u g1_pick_teleop_gripper_test.py \
    --level default \
    --task_config ../../common/example.yaml \
    --agent_name g1_pick_with_gripper_usda_1 \
    --task "抓取测试" \
    --orcagym_addr localhost:50051 \
    --xr_backend televuer \
    --tv_no_tls \
    --tv_goal_mode rebased_tv \
    --tv_ee_dx 0.03
```

头显请打开 `http://127.0.0.1:8012/`（明文，不要用 https）。

| 操作 | 效果 |
|------|------|
| 左 squeeze | 开始 / 结束本轮跟手 |
| 右 squeeze | 重置本轮 |
| 左右 squeeze 同按 | 退出 |
| 右臂 | 跟随右手柄位姿 |
| 左臂 | 侧平举锁定 |
| 左 / 右扳机 | 对应夹爪开合 |

---

## 遥操作采集（落盘）

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

## 采集脚本依赖

落盘采集还会用到 `src/dataStorage/` 下的 LeRobot 写入与相机模块；纯遥操作不依赖这些模块。
