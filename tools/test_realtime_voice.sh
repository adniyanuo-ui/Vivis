#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/outputs/speech/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    printf '请先执行 ./tools/setup_realtime_voice.sh\n' >&2
    exit 1
fi
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONNOUSERSITE=1
exec "$PYTHON" -m vivis.speech.realtime "$@"
