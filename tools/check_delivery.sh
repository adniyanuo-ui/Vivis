#!/usr/bin/env bash
# Offline delivery verification only. Never start a hardware example here.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
PYTHON="${VIVIS_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]] && ! command -v -- "$PYTHON" > /dev/null 2>&1; then
    printf '请先执行 ./tools/setup.sh 安装独立环境。\n' >&2
    exit 1
fi
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONNOUSERSITE=1
"$PYTHON" -m pip check
"$PYTHON" tools/check_assets.py
"$PYTHON" tools/check_calibration.py
"$PYTHON" -m compileall -q src examples tools
for entry in examples/*/run.sh; do
    bash -n "$entry"
    for work in 0001 wz2; do
        VIVIS_PYTHON="$PYTHON" "$entry" --work "$work" --check > /dev/null
        printf 'Offline check passed: %s / %s\n' "$entry" "$work"
    done
done
if "$PYTHON" -c "import importlib.util; raise SystemExit(importlib.util.find_spec('ruckig') is None)"; then
    for work in 0001 wz2; do
        VIVIS_PYTHON="$PYTHON" ./examples/03_face_eye_neck_tracking/run.sh --work "$work" --upstream-trajectory on --check > /dev/null
        printf 'Optional upstream trajectory check passed: %s\n' "$work"
    done
else
    printf 'Optional trajectory dependency absent; checking default/off paths only.\n'
fi
"$PYTHON" -m unittest discover -s tests -p 'test_*.py'
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    git diff --check
fi
printf '离线交付自检通过；未连接机器头，也不代表完成现场验收。\n'
