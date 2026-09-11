#!/usr/bin/env bash
# 安装 LeRobot 采集 / 回放 / 推理运行时。不要对 HDF5 旧入口使用本脚本。
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
cd "$repo_root"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "error: 请先激活 docs/lerobot/environment.yml 创建的环境：conda activate orcalab_lerobot" >&2
    exit 2
fi

python - <<'PY'
import sys

if sys.version_info[:3] != (3, 12, 13):
    raise SystemExit(
        f"error: 需要 Python 3.12.13，当前是 {sys.version.split()[0]}；"
        "请用 docs/lerobot/environment.yml 新建环境"
    )
PY

# NumPy / SciPy 由 Conda 提供，pip 用 --no-deps 避免覆盖。
python -m pip install --no-deps --require-hashes -r "$here/requirements.txt"
python -m pip install --no-deps "orca-gym==26.7.3"
python -m pip install --no-deps "orca-lab==26.7.3"
python -m pip install --no-deps --no-build-isolation ./third_party/lerobot
python -m pip install --no-deps --no-build-isolation ./third_party/openpi-client

python "$here/verify_environment.py"
