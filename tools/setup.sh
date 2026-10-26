#!/usr/bin/env bash
# Install Python dependencies and validate bundled resources; never open hardware.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
    "${VIVIS_PYTHON:-python3}" -m venv .venv
fi
.venv/bin/python -m pip install -e '.[vision]'
.venv/bin/python tools/check_assets.py
for example in examples/01_face_control examples/02_face_eye_tracking examples/03_face_eye_neck_tracking; do
    "$example/run.sh" --check > /dev/null
    printf 'Offline check passed: %s\n' "$example"
done
printf 'Ready. Run the three examples individually; setup has not opened hardware.\n'
