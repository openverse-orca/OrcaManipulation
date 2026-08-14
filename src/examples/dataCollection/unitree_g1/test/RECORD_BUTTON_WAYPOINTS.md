# 宇树 G1 四色按钮接触点标定

`record_g1_pick_button_waypoints.py` — 遥操到按钮、双 squeeze 记 28 维关节角，写出 `../pose_g1_pick_button/{color}.yaml`，供脚本化采集使用。

---

## 前置条件

1. OrcaLab 已启动并加载 `unitree_button.json` 场景，仿真运行中（`localhost:50051`）。
2. TeleVuer 头显通过 USB 连接，`adb devices` 可见。
3. conda 环境 `orcalab_lerobot` 已激活。

---

## 录制步骤

每次只录一种颜色，换色重跑。四色均需录制完才能用于采集。

```bash
cd /home/dht/orca_m/_g1_lerobot_push/src/examples/dataCollection/unitree_g1/test

python -u record_g1_pick_button_waypoints.py \
  --color red \
  --per_color 3 \
  --orcagym_addr localhost:50051
```

`--color` 取值：`red` / `green` / `yellow` / `blue`。

---

## 操控键位

| 操作 | 效果 |
|---|---|
| 左 squeeze | 开闸 / 关闸（进入 / 结束 RUNNING） |
| 左 + 右 squeeze 同按 | **记录当前关节角**（接触按钮后触发） |
| 右 squeeze（单按） | 重置场景，准备下一轮 |
| Ctrl+C | 写出 YAML 并退出 |

---

## 启动头显连接

```bash
# 另开终端，在脚本打印 TeleVuer URL 后执行
adb reverse tcp:8012 tcp:8012
adb shell am start -a android.intent.action.VIEW -d 'http://127.0.0.1:8012/'
```

---

## 输出文件

```
unitree_g1/pose_g1_pick_button/
  red.yaml      # 28 维 q，3 个接触 waypoint
  green.yaml
  yellow.yaml
  blue.yaml
```

---

## 常用可选参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--per_color` | 3 | 每色录几个点 |
| `--output_dir` | `../pose_g1_pick_button` | 输出目录 |
| `--approach_back` | 0.12 | 写入 YAML 的回退距离（米） |
| `--arm_kp` | 150 | 臂阻抗刚度 |
| `--diag_health` | 关 | 打开 `[HEALTH]` 心跳日志 |

---

## 标完后运行脚本化采集

```bash
cd /home/dht/orca_m/_g1_lerobot_push/src/examples/dataCollection/unitree_g1

python -u g1_pick_collection_scripted_button_lerobot.py \
  --pose_candidates pose_g1_pick_button \
  --lerobot_out /path/to/output \
  --repo_id local/g1_pick_button_scripted \
  --orcagym_addr localhost:50051
```
