#!/usr/bin/env bash
# G1 + OmniPicker 夹爪 VR 纯遥操作（不落盘）
# 运行前请确认：
#   1. OrcaLab 已启动且已加载 uni_test.json
#   2. Pico 已通过 USB 连接并已授权 adb
#   3. 已完成 adb 端口转发（见下方 ADB 步骤）

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[ADB] 转发 Pico 8012 端口..."
adb reverse tcp:8012 tcp:8012

echo "[ADB] 在 Pico 端打开 TeleVuer 页面..."
adb shell am start -a android.intent.action.VIEW -d 'http://127.0.0.1:8012/'

echo "[遥操作] 启动 gripper 纯遥操作..."
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
