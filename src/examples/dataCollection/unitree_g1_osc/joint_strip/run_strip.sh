#!/usr/bin/env bash
# 与实测采集 g1_osc_strip2 相同的启动命令。
# 请先在 OrcaLab 打开本目录 uni_osc.json 并运行仿真（localhost:50051）。
set -euo pipefail
cd "$(dirname "$0")"

adb reverse tcp:8001 tcp:8001 || true

OMP_NUM_THREADS=1 python g1_pick_osc_collection_tele_lerobot_strip.py \
    --task_config example.yaml \
    --agent_name g1_pick_southgrid_usda_1 \
    --lerobot_out "${1:-$HOME/orca_m/OrcaManipulation/L_dataset/g1_osc_strip}" \
    --repo_id local/g1_pick_osc_strip \
    --task "按红色按钮" \
    --fps 20 --clock wall \
    --cameras head,wrist_r --camera_source websocket \
    --dls_lambda 0.2 \
    --joint_strip on --strip_col off \
    --time_step 0.001 --frame_skip 5
