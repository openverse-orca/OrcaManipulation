#!/usr/bin/env bash
# 一键：VR 流体遥操采集（OrcaLink + OrcaSPH，50Hz 宏步对齐）
set -euo pipefail

ROOT="/home/hjadmin/OrcaApr24"
DC="${ROOT}/OrcaManipulation/src/examples/dataCollection"
LEVEL="FluidTest_Hotel_Bar_RobotReplay"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate orca-apr24

echo "=== 前置：OrcaStudio Play 关卡 ${LEVEL} ==="
echo "=== 流体 VR 遥操 | 50Hz (frame_skip=20, dt=0.001) | conda: orca-apr24 ==="

cd "${DC}"
python data_collection_fluid_tele.py \
  --level "${LEVEL}" \
  --agent_name tiangong2 \
  --frame-skip 20 \
  --time-step 0.001 \
  --build-mode release \
  "$@"
