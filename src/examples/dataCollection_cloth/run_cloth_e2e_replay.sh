#!/usr/bin/env bash
# 端到端冒烟：Pico 轨迹回放 + OrcaLink + XPBD + Studio 布料 gRPC
#
# 前置（另开终端）：
#   1. OrcaStudio Play（含 PBDRender Gem，cloth_demo，gRPC :50251）
#   2. OrcaGym / Studio 关卡须含 dual_gripper 刚体（gripper_l_palm 等）
#
# 本脚本自动：OrcaLink Server、XPBD dual_gripper_cross_mjc（PBD_GRPC=1）
set -euo pipefail
REPO="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "./dual_gripper_cross_v4_replay.json" ]]; then
  python3 generate_pico_replay_data.py
fi

export PYTHONPATH="${REPO}/OrcaLink/Client/Python:${REPO}/OrcaManipulation/src:${PYTHONPATH:-}"

LEVEL="${LEVEL:-dual_gripper_cross}"
AGENT="${AGENT:-openloong}"

python3 data_collection_cloth_tele.py \
  --level "$LEVEL" \
  --agent_name "$AGENT" \
  --replay \
  --replay_data ./dual_gripper_cross_v4_replay.json \
  --max-episode-sec 67 \
  --frame-skip 20 \
  --time-step 0.001 \
  --cloth-coupling \
  --cloth-config "${REPO}/OrcaPlayground/examples/cloth_3d/cloth_sim_config.orcagym_e2e.json"
