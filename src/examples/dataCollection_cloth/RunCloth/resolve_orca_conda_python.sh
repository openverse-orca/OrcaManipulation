#!/usr/bin/env bash
# 解析 ClothRobot 联调用 Python
# 优先级：$PYTHON > 当前激活的 conda 环境
# 用法: eval "$(bash XPBD/Cloth_robot/resolve_orca_conda_python.sh)"
set -euo pipefail

if [[ -n "${PYTHON:-}" ]]; then
  printf 'export PYTHON=%q\n' "$PYTHON"
  exit 0
fi

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]] && [[ "$CONDA_DEFAULT_ENV" != "base" ]]; then
  echo "[orca-conda] 使用已激活的 conda 环境: ${CONDA_DEFAULT_ENV}" >&2
  printf 'export PYTHON=%q\n' "conda run --no-capture-output -n ${CONDA_DEFAULT_ENV} python3"
  exit 0
fi

echo "[orca-conda] FAIL: 未检测到激活的 conda 环境（当前: ${CONDA_DEFAULT_ENV:-无}）" >&2
echo "  请先创建并激活 conda 环境，例如:" >&2
echo "    conda create -n CondaEnviromentName python=3.12" >&2
echo "    conda activate CondaEnviromentName" >&2
exit 1