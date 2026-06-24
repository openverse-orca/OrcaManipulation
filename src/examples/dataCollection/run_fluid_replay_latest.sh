#!/usr/bin/env bash
# 一键：回放 FluidTest_Hotel_Bar_RobotReplay 最新 HDF5 + SPH 流体（50Hz 对齐）
set -euo pipefail

ROOT="/home/hjadmin/OrcaApr24"
DC="${ROOT}/OrcaManipulation/src/examples/dataCollection"
LEVEL="FluidTest_Hotel_Bar_RobotReplay"
SPH_PATCH="${ROOT}/SPH_bug/Robot_replay_waterShale/orcasph_position_follow.json"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate orca-apr24

echo "=== 前置：OrcaStudio Play 关卡 ${LEVEL} ==="
echo "=== 回放：dataset 下最新回合 | 50Hz (frame_skip=20, dt=0.001) | vel_uniform+rot_slerp ==="

cd "${DC}"
python data_collection_fluid_replay.py \
  --level "${LEVEL}" \
  --agent_name tiangong2 \
  --data_root dataset \
  --latest \
  --replay_mode osc \
  --frame-skip 20 \
  --time-step 0.001 \
  --sph-follow-config "${SPH_PATCH}" \
  "$@"
