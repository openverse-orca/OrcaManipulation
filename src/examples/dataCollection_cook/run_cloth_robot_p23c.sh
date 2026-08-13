#!/usr/bin/env bash
# ClothRobot P2.3c 三进程联调（对齐 SPH 流体链路）：
#   1) OrcaStudio Game/Play（OrcaGym :50051 + PBDRender :50261）
#      推荐: bash XPBD/Cloth_robot/run_cloth_studio.sh  （taskset -c 0-3 强制绑核）
#   2) 本脚本：refresh session + export scene → data_collection_cloth_tele --cloth-coupling
#      自动拉起 OrcaLink + XPBD + ClothOrcaLinkBridge
#
# 用法（默认：XPBD OpenGL 窗口、Studio 刚体跟随、无 debug/采集、无实时 sleep）:
#   bash XPBD/Cloth_robot/run_cloth_studio.sh    # 第一步：Studio 绑 0-3
#   bash XPBD/Cloth_robot/run_cloth_robot_p23c.sh # 第二步：Python+XPBD 绑 4～末核（默认 XPBD_UI=1）
#   AUTO_START_STUDIO=1 bash XPBD/Cloth_robot/run_cloth_robot_p23c.sh  # 一步：先启 Studio 再联调
#   XPBD_UI=0 bash XPBD/Cloth_robot/run_cloth_robot_p23c.sh          # 关闭 XPBD OpenGL
#   CLOTH_SYNC_STUDIO_VIS=0 bash ...                                 # 关闭 Studio 刚体跟随
#   USE_ALL_CPU=1 bash ...                                           # 关闭 CPU 绑核（默认 XPBD+Python→4～末核）
#   DEBUG=1 bash XPBD/Cloth_robot/run_cloth_robot_p23c.sh            # debug CSV + 采集 + 分析
#   bash XPBD/Cloth_robot/run_cloth_robot_p23c.sh # 第二步：自动读 Studio Play 关卡名
#   CLOTH_HOST=auto|studio|orcalab bash XPBD/Cloth_robot/run_cloth_robot_p23c.sh  # 双宿主（默认 auto）
#   CLOTH_HOST=orcalab LEVEL=NursingHome AGENT=g1_omnipicker bash ...  # OrcaLab Play 后联调
#   LEVEL=test20260508_RobotFold bash ...          # 可选：手动覆盖自动检测
#   SKIP_STALE_KILL=1 bash ...                     # 跳过启动前清理（默认会结束陈旧 tele/XPBD/OrcaLink）
#   CLOTH_CAMERA_MONITOR=1 bash ...              # 开启 4 路相机 figure（默认关闭）
set -euo pipefail

REPO="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
CLOTH_3D="${REPO}/OrcaPlayground/examples/cloth_3d"
TELE_DIR="${REPO}/OrcaManipulation/src/examples/dataCollection_cook"

# DEBUG=1：恢复全量 debug 联调（CSV、dataset、实时墙钟、自动分析）
DEBUG="${DEBUG:-0}"

if [[ "$DEBUG" == "1" ]]; then
  CLOTH_DEBUG="${CLOTH_DEBUG:-1}"
  COLLECT_DATA="${COLLECT_DATA:-1}"
  XPBD_RELEASE_BUILD="${XPBD_RELEASE_BUILD:-0}"
  XPBD_UI="${XPBD_UI:-1}"
  CLOTH_SYNC_STUDIO_VIS="${CLOTH_SYNC_STUDIO_VIS:-1}"
  CLOTH_NO_REALTIME="${CLOTH_NO_REALTIME:-0}"
  MAX_MACRO_FRAMES="${MAX_MACRO_FRAMES:-}"
  MAX_SEC="${MAX_SEC:-120}"
else
  CLOTH_DEBUG="${CLOTH_DEBUG:-0}"
  COLLECT_DATA="${COLLECT_DATA:-0}"
  XPBD_RELEASE_BUILD="${XPBD_RELEASE_BUILD:-1}"
  XPBD_UI="${XPBD_UI:-1}"
  CLOTH_SYNC_STUDIO_VIS="${CLOTH_SYNC_STUDIO_VIS:-1}"
  CLOTH_NO_REALTIME="${CLOTH_NO_REALTIME:-1}"
  MAX_MACRO_FRAMES="${MAX_MACRO_FRAMES:-800}"
  MAX_SEC="${MAX_SEC:-120}"
fi

# 兼容旧变量：SHOW_UI 等同 XPBD_UI（若用户仍传 SHOW_UI）
if [[ -n "${SHOW_UI:-}" ]]; then
  XPBD_UI="$SHOW_UI"
fi

AGENT_USER_SET=0
if [[ -n "${AGENT+x}" && -n "${AGENT:-}" ]]; then
  AGENT_USER_SET=1
fi
LEVEL_USER_SET=0
if [[ -n "${LEVEL+x}" && -n "${LEVEL:-}" ]] || [[ -n "${ORCA_LEVEL_NAME+x}" && -n "${ORCA_LEVEL_NAME:-}" ]]; then
  LEVEL_USER_SET=1
fi

AGENT="${AGENT:-openloong}"
# LEVEL：省略时由 detect_studio_level（Editor.log / lastLoadPath.preset）自动解析
LEVEL="${LEVEL:-${ORCA_LEVEL_NAME:-}}"
export LEVEL AGENT
MJC_PREFIX="${MJC_PREFIX:-openloong_gripper_2f85_fix_base_usda}"
ORCAGYM_PORT="${ORCAGYM_PORT:-50051}"
# PBD_GRPC_PORT="${PBD_GRPC_PORT:-50261}"
PBD_GRPC_PORT="${PBD_GRPC_PORT:-50263}"
ORCALINK_PORT="${ORCALINK_PORT:-50361}"
PICO_PORT="${PICO_PORT:-8001}"
WAIT_SEC="${WAIT_SEC:-180}"
KILL_STALE="${KILL_STALE:-1}"
if [[ "${SKIP_STALE_KILL:-0}" == "1" ]]; then
  KILL_STALE=0
fi
STUDIO_CPU_AFFINITY="${STUDIO_CPU_AFFINITY:-0-3}"
AUTO_START_STUDIO="${AUTO_START_STUDIO:-0}"
AUTO_START_ORCALAB="${AUTO_START_ORCALAB:-0}"
CLOTH_HOST="${CLOTH_HOST:-auto}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 集中配置：XPBD 默认 target / pip 版本（新增场景只改 xpbd_scene.conf）
if [[ -f "${SCRIPT_DIR}/xpbd_scene.conf" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/xpbd_scene.conf"
fi

export PYTHONPATH="${REPO}/OrcaLink/Client/Python:${REPO}/OrcaGym:${REPO}/OrcaManipulation/src:${PYTHONPATH:-}"
export PBD_GRPC_ADDRESS="${PBD_GRPC_ADDRESS:-localhost:${PBD_GRPC_PORT}}"
export XPBD_UI
# 布物理：GS（与 MjcPBD implement_log R-PHYS-4 一致；xpbd_process 在 force_gs_solver 时也会注入）
export PBDX_FORCE_GS_ONLY="${PBDX_FORCE_GS_ONLY:-1}"
export PBDX_SOLVER="${PBDX_SOLVER:-gs}"

# 联调 Python 必须为 conda orca-apr24（OrcaGym）；见 resolve_orca_conda_python.sh
eval "$(bash "${SCRIPT_DIR}/resolve_orca_conda_python.sh")"

if [[ -z "${CFG:-}" && -z "${CLOTH_CONFIG:-}" ]]; then
  if [[ "$LEVEL_USER_SET" == "0" && -z "${LEVEL:-}" ]]; then
    LEVEL="$(bash "${SCRIPT_DIR}/detect_studio_level.sh")"
    echo "[P2.3c] Auto-detected Studio level: ${LEVEL}"
    export LEVEL
  fi
  CFG="$(bash "${SCRIPT_DIR}/resolve_cloth_config.sh")"
else
  CFG="${CFG:-${CLOTH_CONFIG}}"
  if [[ "$LEVEL_USER_SET" == "0" && -z "${LEVEL:-}" ]]; then
    LEVEL="$(bash "${SCRIPT_DIR}/detect_studio_level.sh")"
    echo "[P2.3c] Auto-detected Studio level: ${LEVEL}"
    export LEVEL
  fi
fi

if [[ "$AGENT_USER_SET" == "0" ]]; then
  if SYNC_OUT="$($PYTHON "${REPO}/XPBD/Cloth_robot/sync_tele_env_from_mjcf.py" 2>/dev/null)"; then
    read -r DETECTED_AGENT DETECTED_PREFIX <<< "$SYNC_OUT"
    if [[ -n "$DETECTED_AGENT" && "$DETECTED_AGENT" != "$AGENT" ]]; then
      echo "[P2.3c] MJCF 扫描 tele_agent=${DETECTED_AGENT}（覆盖默认 openloong）"
      AGENT="$DETECTED_AGENT"
      MJC_PREFIX="$DETECTED_PREFIX"
      export AGENT MJC_PREFIX
      if [[ -z "${CFG:-}" || -z "${CLOTH_CONFIG:-}" ]]; then
        CFG="$(bash "${SCRIPT_DIR}/resolve_cloth_config.sh")"
        echo "[P2.3c] 重解析 config=${CFG}"
      fi
    fi
  fi
elif SYNC_OUT="$($PYTHON "${REPO}/XPBD/Cloth_robot/sync_tele_env_from_mjcf.py" 2>/dev/null)"; then
  read -r _DETECTED_AGENT DETECTED_PREFIX <<< "$SYNC_OUT"
  if [[ -n "$DETECTED_PREFIX" && "$DETECTED_PREFIX" != "$MJC_PREFIX" ]]; then
    echo "[P2.3c] MJCF 扫描 mjc_prefix=${DETECTED_PREFIX}（覆盖默认 ${MJC_PREFIX}）"
    MJC_PREFIX="$DETECTED_PREFIX"
    export MJC_PREFIX
  fi
fi

export LEVEL AGENT CFG CLOTH_DEBUG MJC_PREFIX
export XPBD_RELEASE_BUILD XPBD_FORCE_REBUILD XPBD_AUTO_BUILD

if [[ "$CLOTH_SYNC_STUDIO_VIS" == "1" ]]; then
  export CLOTH_SYNC_STUDIO_VIS
  CLOTH_STUDIO_VIS_STRIDE="${CLOTH_STUDIO_VIS_STRIDE:-1}"
  export CLOTH_STUDIO_VIS_STRIDE
  echo "[P2.3c] CLOTH_SYNC_STUDIO_VIS=1：推送 qpos 到视口（Studio/OrcaLab，stride=${CLOTH_STUDIO_VIS_STRIDE}）"
fi

if [[ "$XPBD_UI" == "1" ]]; then
  unset MJC_PBD_NO_UI
  export DISPLAY="${DISPLAY:-:0}"
  echo "[P2.3c] XPBD_UI=1：XPBD OpenGL 窗口（MJC_PBD_OVERLAY_MJC 灰线框）"
else
  export MJC_PBD_NO_UI=1
  echo "[P2.3c] XPBD_UI=0：无 XPBD OpenGL（MJC_PBD_NO_UI=1）"
fi

if [[ "${XPBD_AUTO_BUILD:-1}" != "0" ]]; then
  XPBD_TARGET="${XPBD_BUILD_TARGET:-${XPBD_DEFAULT_TARGET:-dual_gripper_g1_cook2}}"
  # PrepareXPBD.py 内部根据 ORCAXPBD_USE_PIP_PACKAGE 分派：pip 同步 或 源码编译
  XPBD_BUILD_TARGET="${XPBD_TARGET}" \
    ORCAXPBD_USE_PIP_PACKAGE="${ORCAXPBD_USE_PIP_PACKAGE:-0}" \
    python3 "${SCRIPT_DIR}/PrepareXPBD.py"
fi

pin_studio_cpus() {
  if [[ "${USE_ALL_CPU:-0}" == "1" ]]; then
    echo "[P2.3c] USE_ALL_CPU=1：跳过 Studio 绑核"
    return 0
  fi
  local spec="${STUDIO_CPU_AFFINITY}"
  local pids pinned=0
  pids=$(pgrep -x OrcaEditor 2>/dev/null || true)
  if [[ -z "$pids" ]]; then
    return 1
  fi
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    if taskset -cp "$spec" "$pid" 2>/dev/null; then
      echo "[P2.3c] OrcaEditor pid=${pid} → CPU ${spec}"
      pinned=$((pinned + 1))
    fi
  done <<< "$pids"
  [[ "$pinned" -gt 0 ]]
}

ensure_studio_ready() {
  if [[ "${AUTO_START_STUDIO:-0}" == "1" ]]; then
    echo "[P2.3c] AUTO_START_STUDIO=1：启动 Studio（CPU ${STUDIO_CPU_AFFINITY}）..."
    bash "${SCRIPT_DIR}/run_cloth_studio.sh"
    return $?
  fi
  if pin_studio_cpus; then
    echo "[P2.3c] 已对运行中的 OrcaEditor 绑核 ${STUDIO_CPU_AFFINITY}"
  else
    echo "[P2.3c] WARN: OrcaEditor 未运行；请先: bash ${SCRIPT_DIR}/run_cloth_studio.sh"
  fi
  return 0
}

_resolve_cloth_host() {
  bash "${SCRIPT_DIR}/detect_cloth_runtime_host.sh"
}

_print_host_play_hint() {
  local host=$1
  case "$host" in
    studio)
      echo "  Studio: bash ${SCRIPT_DIR}/run_cloth_studio.sh"
      echo "  （taskset -c ${STUDIO_CPU_AFFINITY} 启动 ${LEVEL:-<LEVEL>} Game/Play）"
      ;;
    orcalab)
      echo "  OrcaLab: orcalab --scene ${LEVEL:-NursingHome}"
      echo "  进入 Play，确认 :${ORCAGYM_PORT} / :${PBD_GRPC_PORT} 监听"
      ;;
    *)
      echo "  Studio: bash ${SCRIPT_DIR}/run_cloth_studio.sh"
      echo "  OrcaLab: orcalab --scene ${LEVEL:-NursingHome}  # 进入 Play"
      ;;
  esac
}

ensure_runtime_ready() {
  local host
  host="$(_resolve_cloth_host)"
  echo "[P2.3c] CLOTH_HOST=${CLOTH_HOST} → runtime=${host}"

  case "$host" in
    studio)
      ensure_studio_ready
      ;;
    orcalab)
      if [[ "${AUTO_START_ORCALAB:-0}" == "1" ]]; then
        echo "[P2.3c] AUTO_START_ORCALAB=1：请手动确认 orcalab 已启动（本脚本不自动启 GUI）"
        echo "  建议: orcalab --scene ${LEVEL:-NursingHome}"
      elif ! pgrep -f 'orcalab\.main|/bin/orcalab' >/dev/null 2>&1; then
        echo "[P2.3c] WARN: OrcaLab 未运行；请先: orcalab --scene ${LEVEL:-NursingHome} 并 Play"
      else
        echo "[P2.3c] 已检测到 OrcaLab 进程（跳过 OrcaEditor 绑核）"
      fi
      ;;
    *)
      echo "[P2.3c] WARN: 未检测到 Studio/OrcaLab；请先启动其一并进入 Play"
      _print_host_play_hint unknown
      ;;
  esac
  return 0
}

wait_port() {
  local port=$1 label=$2 max=$3 waited=0
  echo "[P2.3c] 等待 ${label} localhost:${port} (最多 ${max}s)..."
  while [ "$waited" -lt "$max" ]; do
    if timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
      echo "[P2.3c] OK ${label} :${port}"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "[P2.3c] 超时：${label} :${port} 未监听"
  local host
  host="$(_resolve_cloth_host)"
  _print_host_play_hint "$host"
  return 1
}

# 结束占用 cloth 联调端口的陈旧进程，避免 8001/OrcaLink 冲突与 XPBD 宏步错位。
_pids_on_tcp_port() {
  local port=$1
  ss -tlnp 2>/dev/null | grep -E ":${port}[[:space:]]" \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u || true
}

_kill_pids_gracefully() {
  local label=$1
  shift
  local pid killed=0
  for pid in "$@"; do
    [[ -z "$pid" || "$pid" == "$$" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "[P2.3c] 结束 ${label} pid=${pid}"
      kill -TERM "$pid" 2>/dev/null || true
      killed=1
    fi
  done
  if [[ "$killed" == "1" ]]; then
    sleep 1
    for pid in "$@"; do
      [[ -z "$pid" || "$pid" == "$$" ]] && continue
      if kill -0 "$pid" 2>/dev/null; then
        echo "[P2.3c] 强制结束 ${label} pid=${pid}"
        kill -KILL "$pid" 2>/dev/null || true
      fi
    done
  fi
}

kill_stale_cloth_processes() {
  local tele_pids orca_pids pico_pids tele_pid
  tele_pids=$(pgrep -f "data_collection_cloth_tele\\.py" 2>/dev/null || true)
  orca_pids=$(_pids_on_tcp_port "$ORCALINK_PORT")
  pico_pids=$(_pids_on_tcp_port "$PICO_PORT")

  if [[ -z "$tele_pids$orca_pids$pico_pids" ]]; then
    echo "[P2.3c] 无陈旧 cloth 联调进程（:${ORCALINK_PORT} / :${PICO_PORT}）"
    return 0
  fi

  echo "[P2.3c] 清理陈旧 cloth 联调进程（OrcaLink :${ORCALINK_PORT}、Pico :${PICO_PORT}）..."

  # 先停 tele，再停端口监听，避免 OrcaLink session 半开。
  # （XPBD 旧进程由 PrepareXPBD.py 在准备阶段清理，不在此处处理。）
  if [[ -n "$tele_pids" ]]; then
    while read -r tele_pid; do
      [[ -z "$tele_pid" ]] && continue
      _kill_pids_gracefully "data_collection_cloth_tele" "$tele_pid"
    done <<< "$tele_pids"
  fi

  orca_pids=$(_pids_on_tcp_port "$ORCALINK_PORT")
  if [[ -n "$orca_pids" ]]; then
    while read -r tele_pid; do
      [[ -z "$tele_pid" ]] && continue
      _kill_pids_gracefully "orcalink(:${ORCALINK_PORT})" "$tele_pid"
    done <<< "$orca_pids"
  fi

  pico_pids=$(_pids_on_tcp_port "$PICO_PORT")
  if [[ -n "$pico_pids" ]]; then
    while read -r tele_pid; do
      [[ -z "$tele_pid" ]] && continue
      _kill_pids_gracefully "Pico(:${PICO_PORT})" "$tele_pid"
    done <<< "$pico_pids"
  fi

  sleep 1
  if ss -tln 2>/dev/null | grep -qE ":${ORCALINK_PORT}[[:space:]]|:${PICO_PORT}[[:space:]]"; then
    echo "[P2.3c] WARN: :${ORCALINK_PORT} 或 :${PICO_PORT} 仍被占用，请手动检查: ss -tlnp | grep -E '${ORCALINK_PORT}|${PICO_PORT}'"
  else
    echo "[P2.3c] 陈旧进程已清理，:${ORCALINK_PORT} / :${PICO_PORT} 已释放"
  fi
  return 0
}

echo "[P2.3c] repo=${REPO}"
echo "[P2.3c] cloth_host=${CLOTH_HOST}  level=${LEVEL}  agent=${AGENT}  mjc_prefix=${MJC_PREFIX}  config=${CFG}  DEBUG=${DEBUG}  CLOTH_DEBUG=${CLOTH_DEBUG}  COLLECT_DATA=${COLLECT_DATA}"
echo "[P2.3c] XPBD_UI=${XPBD_UI}  CLOTH_SYNC_STUDIO_VIS=${CLOTH_SYNC_STUDIO_VIS}  XPBD_RELEASE_BUILD=${XPBD_RELEASE_BUILD:-0}  CLOTH_NO_REALTIME=${CLOTH_NO_REALTIME:-0}"
echo "[P2.3c] python=${PYTHON}"
if [[ -n "$MAX_MACRO_FRAMES" ]]; then
  echo "[P2.3c] MAX_MACRO_FRAMES=${MAX_MACRO_FRAMES}（无实时 sleep 下尽快跑满）"
else
  echo "[P2.3c] MAX_SEC=${MAX_SEC}（实时同步下约 ${MAX_SEC}/0.2≈$((MAX_SEC / 5)) 宏步；15s 轨迹建议 MAX_SEC>=120）"
fi

if [[ "$KILL_STALE" == "1" ]]; then
  kill_stale_cloth_processes
else
  echo "[P2.3c] KILL_STALE=0：跳过陈旧进程清理"
fi

ensure_runtime_ready

wait_port "$ORCAGYM_PORT" "OrcaGym" "$WAIT_SEC"
wait_port "$PBD_GRPC_PORT" "PBDRender" "$WAIT_SEC"

echo "[P2.3c] 从 Studio MJCF 刷新 XPBD session + export scene..."
$PYTHON "${SCRIPT_DIR}/refresh_xpbd_session_from_mjcf.py" \
  --config "$CFG" \
  --export-scene \
  --session-tag p23c

LOG_TAG="p23c_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${TELE_DIR}/logs"

cd "$TELE_DIR"
TELE_ARGS=(
  --level "$LEVEL"
  --agent_name "$AGENT"
  --mjc-agent-prefix "$MJC_PREFIX"
  --frame-skip 20
  --time-step 0.001
  --cloth-coupling
  --cloth-config "$CFG"
)
if [[ -n "$MAX_MACRO_FRAMES" ]]; then
  TELE_ARGS+=(--max-macro-frames "$MAX_MACRO_FRAMES")
else
  TELE_ARGS+=(--max-episode-sec "$MAX_SEC")
fi
if [[ "$CLOTH_DEBUG" == "1" ]]; then
  TELE_ARGS+=(--cloth-debug)
fi
if [[ "$COLLECT_DATA" == "0" ]]; then
  TELE_ARGS+=(--no-collect)
fi
if [[ "${CLOTH_NO_REALTIME:-0}" == "1" ]]; then
  TELE_ARGS+=(--no-realtime)
  export CLOTH_NO_REALTIME
fi
if [[ "${USE_ALL_CPU:-0}" == "1" ]]; then
  TELE_ARGS+=(--use-all-cpu)
  echo "[P2.3c] USE_ALL_CPU=1：MuJoCo/Python 与 XPBD 不绑核"
else
  echo "[P2.3c] CPU 亲和性：Studio→${STUDIO_CPU_AFFINITY}；MuJoCo/Python+XPBD→4～末核"
fi
if [[ "${MUJOCO_VIEWER:-${GUI:-0}}" == "1" ]]; then
  TELE_ARGS+=(--gui)
  echo "[P2.3c] MUJOCO_VIEWER=1：启动 MuJoCo 原生 passive viewer"
fi
if [[ -n "${BENCH_JSON:-}" ]]; then
  TELE_ARGS+=(--bench "$BENCH_JSON")
  echo "[P2.3c] BENCH_JSON=${BENCH_JSON}"
fi


echo "[P2.3c] 启动 data_collection_cloth_tele（OrcaLink + XPBD + bridge）..."
RUN_START=$(date +%s.%N)
$PYTHON data_collection_cloth_tele.py "${TELE_ARGS[@]}" \
  2>&1 | tee "logs/${LOG_TAG}_tele.log"
RUN_END=$(date +%s.%N)
RUN_WALL=$(python3 -c "print(f'{(float('${RUN_END}') - float('${RUN_START}')):.2f}')")
echo "[P2.3c] 墙钟耗时: ${RUN_WALL}s（CLOTH_NO_REALTIME=${CLOTH_NO_REALTIME:-0}）"

echo "[P2.3c] 完成。日志: ${TELE_DIR}/logs/${LOG_TAG}_tele.log"
echo "[P2.3c] XPBD 日志: ${TELE_DIR}/logs/xpbd_*.log"
