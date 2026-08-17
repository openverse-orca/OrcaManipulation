#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "error: activate the environment created from environment-unitree.yml first" >&2
    exit 2
fi

python - <<'PY'
import sys

if sys.version_info[:3] != (3, 12, 13):
    raise SystemExit(
        f"error: expected Python 3.12.13, got {sys.version.split()[0]}; "
        "create a fresh environment from environment-unitree.yml"
    )
PY

# Guard the Conda-owned ABI before invoking pip. This also detects an environment
# polluted by the unrelated PyPI `pinocchio` package or the cmeel `pin` wheel.
python - <<'PY'
import pathlib

import casadi
import numpy
import pinocchio
import pinocchio.casadi
import scipy

expected = {
    "numpy": (numpy.__version__, "2.4.6"),
    "scipy": (scipy.__version__, "1.15.3"),
    "pinocchio": (pinocchio.__version__, "3.9.0"),
    "casadi": (casadi.__version__, "3.7.2"),
}
for name, (actual, wanted) in expected.items():
    if actual != wanted:
        raise SystemExit(f"error: {name} must be {wanted}, got {actual}")

pin_path = pathlib.Path(pinocchio.__file__).resolve()
cpin_path = pathlib.Path(pinocchio.casadi.__file__).resolve()
if "cmeel.prefix" in str(pin_path) or "cmeel.prefix" in str(cpin_path):
    raise SystemExit(
        "error: pip/cmeel Pinocchio shadows the Conda build; recreate the environment"
    )
print("Conda ABI preflight OK")
PY

# requirements.txt is a complete resolved closure and carries its own --index-url /
# --extra-index-url lines, so the CPU-only PyTorch wheels resolve without extra flags.
# --no-deps keeps pip from re-resolving: NumPy and SciPy belong to Conda, and
# params-proto declares the PyPI argparse backport, which on Python 3.12 would shadow
# the standard library module and drop argparse.BooleanOptionalAction.
python -m pip install --no-deps --require-hashes -r requirements.txt

# --no-deps is intentional: the validated Conda NumPy/SciPy versions differ
# from stale dependency metadata published by orca-gym 26.7.1.
python -m pip install --no-deps "orca-gym==26.7.1"

# OrcaLab is the desktop application; the PyPI package is the whole thing, so
# there is no separate installer to run. --no-deps applies for the same reason as
# above, plus its pyparsing==3.2.5 pin, which would downgrade the version
# matplotlib is validated against. Every dependency it actually needs is in
# requirements.txt. orca-lab and orca-gym must stay on the same version: orca-lab
# declares an exact orca-gym pin, and a mismatch breaks the gRPC handshake.
python -m pip install --no-deps "orca-lab==26.7.1"

# Repository-owned sources are installed last. No developer-machine path or
# unpinned Git checkout is consulted.
python -m pip install --no-deps --no-build-isolation ./third_party/lerobot
python -m pip install --no-deps --no-build-isolation ./third_party/televuer
python -m pip install --no-deps --no-build-isolation ./third_party/openpi-client

python scripts/verify_environment.py

# OrcaLab pulls the Qt platform libraries it is missing on first launch. It can do
# that unprivileged for libxcb-cursor0 (downloaded and copied next to PySide6), but
# libvdpau1 goes through "sudo apt install", which stalls on a password prompt in a
# non-interactive session. Report it here instead of letting the first launch block.
if command -v dpkg-query >/dev/null 2>&1; then
    if ! dpkg-query -W -f='${Status}' libvdpau1 2>/dev/null | grep -q '^install ok installed$'; then
        echo
        echo "warning: libvdpau1 is missing; OrcaLab would prompt for a sudo password on first launch."
        echo "         install it now with: sudo apt install -y libvdpau1"
    fi
fi
