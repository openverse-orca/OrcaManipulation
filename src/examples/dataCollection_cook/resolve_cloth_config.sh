#!/usr/bin/env bash
# 解析 cloth_sim_config 路径（供 bash 联调脚本调用）。
#
# 环境变量：LEVEL、ORCA_LEVEL_NAME、AGENT、CLOTH_DEBUG、CFG / CLOTH_CONFIG
# 输出：绝对路径（stdout）
set -euo pipefail

REPO="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
export PYTHONPATH="${REPO}/OrcaLink/Client/Python:${REPO}/OrcaGym:${REPO}/OrcaManipulation/src:${PYTHONPATH:-}"

eval "$(bash "$(dirname "$0")/resolve_orca_conda_python.sh")"

export LEVEL="${LEVEL:-${ORCA_LEVEL_NAME:-}}"
export AGENT="${AGENT:-openloong}"
export CLOTH_DEBUG="${CLOTH_DEBUG:-0}"

$PYTHON - <<'PY'
import os
from envs.cloth.paths import resolve_cloth_config_path, resolve_cloth_level

explicit = os.environ.get("CFG") or os.environ.get("CLOTH_CONFIG") or None
if explicit:
    explicit = explicit.strip() or None

debug = os.environ.get("CLOTH_DEBUG", "0") == "1"
level = resolve_cloth_level(os.environ.get("LEVEL") or None)
path = resolve_cloth_config_path(
    level=level,
    agent=os.environ.get("AGENT", "openloong"),
    debug=debug,
    explicit=explicit,
)
print(path)
PY
