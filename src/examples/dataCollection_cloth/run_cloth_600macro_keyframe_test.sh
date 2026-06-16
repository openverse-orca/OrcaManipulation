#!/usr/bin/env bash
# dual_gripper 关键帧短链：600 宏步 → 分析 CSV（无 Pico / OrcaGym 50051）
set -euo pipefail
REPO="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MAX_MACRO_FRAMES="${MAX_MACRO_FRAMES:-600}"
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

CHECK_PBD_MESH="${REPO}/XPBD/build/check_pbd_mesh_count"

wait_port() {
  local port=$1 label=$2 max=$3 waited=0
  echo "[KF600] 等待 ${label} localhost:${port} (最多 ${max}s)..."
  while [ "$waited" -lt "$max" ]; do
    if timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
      echo "[KF600] OK ${label} :${port}"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "[KF600] 超时：${label} :${port} 未监听（Studio PBDRender Play?）"
  return 1
}

wait_pbd_meshes() {
  local addr="${PBD_GRPC_ADDRESS}" max="${1:-$WAIT_SEC}" waited=0
  if [[ ! -x "$CHECK_PBD_MESH" ]]; then
    echo "[KF600] WARN: 未找到 $CHECK_PBD_MESH，跳过 mesh 预检（仅检查端口）"
    return 0
  fi
  echo "[KF600] 等待 PBDRender 注册布料 mesh（GetMeshCount>0，最多 ${max}s）..."
  while [ "$waited" -lt "$max" ]; do
    if out="$("$CHECK_PBD_MESH" "$addr" 2>&1)"; then
      echo "[KF600] OK $out"
      return 0
    fi
    echo "[KF600]   $out"
    sleep 2
    waited=$((waited + 2))
  done
  echo "[KF600] 超时：${addr} GetMeshCount=0（请确认 test20260508 已 Play 且布料实体已激活）"
  return 1
}

CFG="${REPO}/OrcaPlayground/examples/cloth_3d/cloth_sim_config.dual_gripper_cross_shortchain.debug.json"
LOG_TAG="keyframe_macro${MAX_MACRO_FRAMES}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

echo "[KF600] dual_gripper keyframe short chain, macro_frames=${MAX_MACRO_FRAMES}"
echo "[KF600] 请确认 OrcaStudio 已 Play test20260508（PBDRender 注册布料 mesh 后 XPBD 才能 UpdateMesh）"
wait_port "$PBD_GRPC_PORT" "PBDRender" "$WAIT_SEC"
wait_pbd_meshes "$WAIT_SEC"

echo "[KF600] 启动短链 (日志 logs/${LOG_TAG}_shortchain.log)..."
$PYTHON run_cloth_keyframe_shortchain.py \
  --cloth-config "$CFG" \
  --max-macro-frames "$MAX_MACRO_FRAMES" \
  --log-dir logs \
  --no-realtime \
  2>&1 | tee "logs/${LOG_TAG}_shortchain.log"

echo ""
echo "[KF600] 分析最新 cloth_debug_* CSV..."
$PYTHON analyze_cloth_debug_session.py --watch-latest --target-macro-frames "$MAX_MACRO_FRAMES" \
  | tee "logs/${LOG_TAG}_analysis.txt"

echo "[KF600] 完成。见 logs/${LOG_TAG}_* 与 logs/cloth_debug_*/"
