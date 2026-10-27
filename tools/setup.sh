#!/usr/bin/env bash
# Install Python dependencies and validate bundled resources; never open hardware.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
EXTRAS=vision
if [[ $# -gt 0 ]]; then
    if [[ $# -ne 1 || "$1" != --trajectory ]]; then
        printf 'Usage: ./tools/setup.sh [--trajectory]\n' >&2
        exit 2
    fi
    EXTRAS=vision,trajectory
    export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
fi
if [[ ! -x .venv/bin/python ]]; then
    "${VIVIS_PYTHON:-python3}" -m venv .venv
fi
.venv/bin/python -m pip install -e ".[$EXTRAS]"
.venv/bin/python tools/check_assets.py
for example in examples/01_face_control examples/02_face_eye_tracking examples/03_face_eye_neck_tracking; do
    "$example/run.sh" --check > /dev/null
    printf 'Offline check passed: %s\n' "$example"
done
if ! command -v amixer > /dev/null 2>&1; then
    printf '默认USB音箱音量控制需要amixer：sudo apt-get install alsa-utils\n' >&2
fi
printf 'Ready. Run the three examples individually; setup has not opened hardware.\n'
