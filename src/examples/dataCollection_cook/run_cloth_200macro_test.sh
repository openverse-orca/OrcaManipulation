#!/usr/bin/env bash
# 200 宏步 debug 联调：跑满宏步 → 自动分析 CSV（刚体同步 + 布料变形）
set -euo pipefail
REPO="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LEVEL="${LEVEL:-test20260508}"
AGENT="${AGENT:-openloong}"
MJC_PREFIX="${MJC_PREFIX:-}"
if [[ "$LEVEL" == "test20260508" && "$AGENT" == "openloong" && -z "$MJC_PREFIX" ]]; then
  MJC_PREFIX="openloong_gripper_2f85_fix_base_usda"
fi
MAX_MACRO_FRAMES="${MAX_MACRO_FRAMES:-200}"
ORCAGYM_PORT="${ORCAGYM_PORT:-50051}"
PBD_GRPC_PORT="${PBD_GRPC_PORT:-50261}"
WAIT_SEC="${WAIT_SEC:-120}"

export PYTHONPATH="${REPO}/OrcaLink/Client/Python:${REPO}/OrcaGym:${REPO}/OrcaManipulation/src:${PYTHONPATH:-}"
export PBD_GRPC_ADDRESS="${PBD_GRPC_ADDRESS:-localhost:${PBD_GRPC_PORT}}"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v conda >/dev/null 2>&1 && conda env list | grep -qE '^\s*orca-apr24\s'; then
    PYTHON="conda run --no-capture-output -n orca-apr24 python3"
  else
    PYTHON="python3"
  fi
fi

wait_port() {
  local port=$1 label=$2 max=$3 waited=0
  echo "[200MF] 等待 ${label} localhost:${port} (最多 ${max}s)..."
  while [ "$waited" -lt "$max" ]; do
    if timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
      echo "[200MF] OK ${label} :${port}"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "[200MF] 超时：${label} :${port} 未监听"
  return 1
}

echo "[200MF] level=${LEVEL} agent=${AGENT} macro_frames=${MAX_MACRO_FRAMES}"
wait_port "$ORCAGYM_PORT" "OrcaGym" "$WAIT_SEC"
wait_port "$PBD_GRPC_PORT" "PBDRender" "$WAIT_SEC"

if [[ ! -f "./dual_gripper_cross_v4_replay.json" ]]; then
  echo "[200MF] 生成 Pico replay JSON..."
  $PYTHON generate_pico_replay_data.py
fi

CFG="${REPO}/OrcaPlayground/examples/cloth_3d/cloth_sim_config.debug.json"
if [[ ! -f "$CFG" ]]; then
  CFG="${REPO}/OrcaPlayground/examples/cloth_3d/cloth_sim_config.json"
fi

LOG_TAG="macro${MAX_MACRO_FRAMES}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

EXTRA_ARGS=()
if [[ -n "$MJC_PREFIX" ]]; then
  EXTRA_ARGS+=(--mjc-agent-prefix "$MJC_PREFIX")
fi

echo "[200MF] 启动 tele (日志 logs/${LOG_TAG}_tele.log)..."
$PYTHON data_collection_cloth_tele.py \
  --level "$LEVEL" \
  --agent_name "$AGENT" \
  "${EXTRA_ARGS[@]}" \
  --replay \
  --replay_data ./dual_gripper_cross_v4_replay.json \
  --max-macro-frames "$MAX_MACRO_FRAMES" \
  --frame-skip 20 \
  --time-step 0.001 \
  --cloth-coupling \
  --cloth-debug \
  --cloth-config "$CFG" \
  --cloth-auto-start-orcalink \
  --cloth-auto-start-xpbd \
  2>&1 | tee "logs/${LOG_TAG}_tele.log"

echo ""
echo "[200MF] 分析最新 cloth_debug_* CSV..."
$PYTHON analyze/analyze_cloth_debug_session.py --watch-latest --target-macro-frames "$MAX_MACRO_FRAMES" \
  | tee "logs/${LOG_TAG}_analysis.txt"

echo "[200MF] 完成。见 logs/${LOG_TAG}_* 与 logs/cloth_debug_*/analysis_rigid_cloth.png"
