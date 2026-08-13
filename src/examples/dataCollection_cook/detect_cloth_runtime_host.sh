#!/usr/bin/env bash
# 判定布料联调运行时宿主：studio | orcalab | unknown
# 供 run_cloth_robot_p23c.sh 在 CLOTH_HOST=auto 时调用。
set -euo pipefail

host="${CLOTH_HOST:-auto}"

if [[ "$host" != "auto" ]]; then
  case "$host" in
    studio|orcalab|unknown) echo "$host" ;;
    *)
      echo "unknown"
      echo "[detect_cloth_runtime_host] WARN: invalid CLOTH_HOST=${host}" >&2
      ;;
  esac
  exit 0
fi

if pgrep -x OrcaEditor >/dev/null 2>&1; then
  echo studio
  exit 0
fi

if pgrep -f 'orcalab\.main|/bin/orcalab' >/dev/null 2>&1; then
  echo orcalab
  exit 0
fi

echo unknown
