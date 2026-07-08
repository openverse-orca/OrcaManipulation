#!/usr/bin/env bash
# 解析 ClothRobot 联调用 Python：必须为 conda orca-apr24（含 OrcaGym）。
# 用法: eval "$(bash XPBD/Cloth_robot/resolve_orca_conda_python.sh)"
set -euo pipefail

if [[ -n "${PYTHON:-}" ]]; then
  printf 'export PYTHON=%q\n' "$PYTHON"
  exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
  if [[ -x "${HOME}/miniconda3/envs/orca-apr24/bin/python3" ]]; then
    printf 'export PYTHON=%q\n' "${HOME}/miniconda3/envs/orca-apr24/bin/python3"
    exit 0
  fi
  echo "[orca-apr24] FAIL: 未找到 conda；请先安装并创建环境 orca-apr24" >&2
  exit 1
fi
if ! conda env list | grep -qE '^\s*orca-apr24\s'; then
  echo "[orca-apr24] FAIL: 未找到 conda 环境 orca-apr24" >&2
  echo "  联调依赖 OrcaGym，不可使用系统 python3 绕过" >&2
  exit 1
fi

printf 'export PYTHON=%q\n' "conda run --no-capture-output -n orca-apr24 python3"
