#!/usr/bin/env bash
# Hardware-free checks only. No motion publishers or robot services are started.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${WBWAM_REPO:-/home/xaai/wb_wam_review}"
MODE="${1:-check}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export WBWAM_BRIDGE_ENCODER_MODEL="${WBWAM_BRIDGE_ENCODER_MODEL:-$REPO/checkpoints/sonic_release/model_encoder.onnx}"
export WBWAM_BRIDGE_TEXT_CACHE_DIR="${WBWAM_BRIDGE_TEXT_CACHE_DIR:-$SCRIPT_DIR/text_embeds}"
# Loopback values satisfy config expansion; fake inputs never contact them.
export WBWAM_BRIDGE_CAMERA_ENDPOINT=tcp://127.0.0.1:5560
export WBWAM_BRIDGE_CAMERA_KEY=d455
export WBWAM_BRIDGE_STATE_ACTION_ENDPOINT=tcp://127.0.0.1:5558
export WBWAM_BRIDGE_HAND_STATUS_ENDPOINT=tcp://127.0.0.1:5559
case "$MODE" in
  check) args=(--check-config) ;;
  mock) args=(--dry-run --mock-policy --fake-camera --fake-state --smoke-warmup-iters 0 --smoke-iters 1) ;;
  dry-run) args=(--dry-run --fake-camera --fake-state --smoke-warmup-iters 0 --smoke-iters 1) ;;
  *) echo "Usage: $0 [check|mock|dry-run]" >&2; exit 2 ;;
esac
exec "$REPO/bridge/scripts/run_bridge.sh" --config "$SCRIPT_DIR/deploy.push_cart.yaml" "${args[@]}"
