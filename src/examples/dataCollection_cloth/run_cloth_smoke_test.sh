#!/usr/bin/env bash
# 端到端冒烟：等待 Studio gRPC → 生成 replay → 短轨迹联调
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
MAX_SEC="${MAX_SEC:-12}"
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
  echo "[SMOKE] 等待 ${label} localhost:${port} (最多 ${max}s)..."
  while [ "$waited" -lt "$max" ]; do
    if timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
      echo "[SMOKE] OK ${label} :${port}"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "[SMOKE] 超时：${label} :${port} 未监听（Studio 须 Game/Play 且含 OrcaGym + PBDRender）"
  return 1
}

echo "[SMOKE] level=${LEVEL} agent=${AGENT} max_episode=${MAX_SEC}s"
wait_port "$ORCAGYM_PORT" "OrcaGym" "$WAIT_SEC"
wait_port "$PBD_GRPC_PORT" "PBDRender" "$WAIT_SEC"

if [[ ! -f "./dual_gripper_cross_v4_replay.json" ]]; then
  echo "[SMOKE] 生成 Pico replay JSON..."
  $PYTHON generate_pico_replay_data.py
fi

CLOTH_DEBUG="${CLOTH_DEBUG:-0}"
if [[ "$CLOTH_DEBUG" == "1" ]]; then
  CFG="${REPO}/OrcaPlayground/examples/cloth_3d/cloth_sim_config.debug.json"
  if [[ ! -f "$CFG" ]]; then
    CFG="${REPO}/OrcaPlayground/examples/cloth_3d/cloth_sim_config.json"
  fi
else
  CFG="${REPO}/OrcaPlayground/examples/cloth_3d/cloth_sim_config.json"
fi
LOG_TAG="smoke_$(date +%Y%m%d_%H%M%S)"
echo "[SMOKE] 启动 data_collection_cloth_tele (日志 logs/${LOG_TAG}_tele.log)..."
MONITOR_PID=""
if [[ "$CLOTH_DEBUG" == "1" ]]; then
  echo "[SMOKE] CLOTH_DEBUG=1：后台启动 analyze/run_cloth_debug_monitor.py --watch-latest"
  $PYTHON analyze/run_cloth_debug_monitor.py --watch-latest --interval 3 &
  MONITOR_PID=$!
fi

EXTRA_ARGS=()
if [[ -n "$MJC_PREFIX" ]]; then
  EXTRA_ARGS+=(--mjc-agent-prefix "$MJC_PREFIX")
fi

DEBUG_ARGS=()
if [[ "$CLOTH_DEBUG" == "1" ]]; then
  DEBUG_ARGS+=(--cloth-debug)
fi

$PYTHON data_collection_cloth_tele.py \
  --level "$LEVEL" \
  --agent_name "$AGENT" \
  "${EXTRA_ARGS[@]}" \
  --replay \
  --replay_data ./dual_gripper_cross_v4_replay.json \
  --max-episode-sec "$MAX_SEC" \
  --frame-skip 20 \
  --time-step 0.001 \
  --cloth-coupling \
  --cloth-config "$CFG" \
  "${DEBUG_ARGS[@]}" \
  2>&1 | tee "logs/${LOG_TAG}_tele.log"

if [[ -n "$MONITOR_PID" ]]; then
  kill "$MONITOR_PID" 2>/dev/null || true
fi

echo "[SMOKE] 完成。检查 logs/${LOG_TAG}_tele.log 与 logs/xpbd_*.log / orcalink_*.log"
if [[ "$CLOTH_DEBUG" == "1" ]]; then
  echo "[SMOKE] Debug CSV 目录: logs/cloth_debug_* （与 tele 同 session 时间戳）"
fi
