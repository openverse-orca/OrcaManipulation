#!/usr/bin/env bash
# Closed-loop WB-WAM push-cart inference in OrcaLab / OrcaGym.
#
# Preconditions:
#   1. OrcaLab is open with the factory scene (G1 + flatbed trolley) and the
#      simulation is running (OrcaGym gRPC on 127.0.0.1:50051).
#   2. The bridge environment exists: bridge/.venv-wam
#   3. SONIC assets exist: checkpoints/sonic_release/model_decoder.onnx
#
# Usage (from anywhere; the script resolves the repository root):
#   orcalab_demo/run_closed_loop.sh [run_push_cart.py options]
#
# Examples:
#   orcalab_demo/run_closed_loop.sh
#   orcalab_demo/run_closed_loop.sh --seconds 360 --reset-scene --record orcalab_demo/outputs/push_cart.mp4
#   orcalab_demo/run_closed_loop.sh --no-record --seconds 30
#   orcalab_demo/run_closed_loop.sh --start-offset-x 0.2 --cart-slide -0.15
#
# Hand backends: auto (default), wuji, or dex3 via --hand-mode.
# After manual scene/asset adjustments, omit --reset-scene to preserve transforms.
#
# Environment overrides:
#   POLICY_CONFIG   deployment YAML   (default: orcalab_demo/deploy/deploy_push_cart_cpu.yaml)
#   POLICY_URL      policy endpoint   (default: http://127.0.0.1:8766)
#   POLICY_PORT     policy port       (default: 8766)
#   ORCALAB_PYTHON  OrcaLab python    (default: /home/user/miniconda3/envs/orcalab-26.10.1.16/bin/python)
#   POLICY_LOG      server log file   (default: /tmp/wbwam_policy_server.log)
#   POLICY_THREADS  CPU inference threads (default: 8)
#   POLICY_CAPTURES save per-inference npz captures (default: 1; 0 disables)
#
# The script reuses an already-running policy server on POLICY_URL. If none is
# running it starts one from POLICY_CONFIG and stops it on exit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

POLICY_CONFIG="${POLICY_CONFIG:-orcalab_demo/deploy/deploy_push_cart_cpu.yaml}"
POLICY_PORT="${POLICY_PORT:-8766}"
POLICY_URL="${POLICY_URL:-http://127.0.0.1:${POLICY_PORT}}"
ORCALAB_PYTHON="${ORCALAB_PYTHON:-/home/user/miniconda3/envs/orcalab-26.10.1.16/bin/python}"
POLICY_LOG="${POLICY_LOG:-/tmp/wbwam_policy_server.log}"
POLICY_THREADS="${POLICY_THREADS:-8}"
POLICY_CAPTURES="${POLICY_CAPTURES:-1}"
BRIDGE_PYTHON="$ROOT/bridge/.venv-wam/bin/python"
capture_flag="--no-captures"
if [[ "$POLICY_CAPTURES" == "1" ]]; then
  capture_flag=""
fi

if ! timeout 2 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/50051" 2>/dev/null; then
  echo "[closed-loop] OrcaGym 127.0.0.1:50051 is not reachable." >&2
  echo "[closed-loop] Open OrcaLab, load the factory scene and start the simulation first." >&2
  exit 1
fi
if [[ ! -x "$ORCALAB_PYTHON" ]]; then
  echo "[closed-loop] OrcaLab python not found: $ORCALAB_PYTHON" >&2
  exit 1
fi
if [[ ! -x "$BRIDGE_PYTHON" ]]; then
  echo "[closed-loop] bridge python not found: $BRIDGE_PYTHON" >&2
  exit 1
fi

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    echo "[closed-loop] policy server stopped"
  fi
}
trap cleanup EXIT INT TERM

if curl -sf --max-time 2 "$POLICY_URL/health" > /dev/null 2>&1; then
  echo "[closed-loop] reusing policy server at $POLICY_URL"
else
  echo "[closed-loop] starting policy server ($POLICY_CONFIG) on port $POLICY_PORT"
  OMP_NUM_THREADS="$POLICY_THREADS" OMP_WAIT_POLICY=passive MKL_NUM_THREADS="$POLICY_THREADS" \
    PYTHONPATH="$ROOT" "$BRIDGE_PYTHON" -u -m orcalab_demo.policy_server \
    --config "$POLICY_CONFIG" --port "$POLICY_PORT" --replan 20 $capture_flag \
    > "$POLICY_LOG" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 60); do
    if curl -sf --max-time 2 "$POLICY_URL/health" > /dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "[closed-loop] policy server exited; see $POLICY_LOG" >&2
      tail -n 40 "$POLICY_LOG" >&2 || true
      exit 1
    fi
    sleep 3
  done
  if ! curl -sf --max-time 2 "$POLICY_URL/health" > /dev/null 2>&1; then
    echo "[closed-loop] policy server did not become ready; see $POLICY_LOG" >&2
    exit 1
  fi
  curl -s "$POLICY_URL/health"; echo
fi

echo "[closed-loop] launching runner"
PYTHONPATH="$ROOT" "$ORCALAB_PYTHON" -m orcalab_demo.run_push_cart \
  --policy-url "$POLICY_URL" "$@"
