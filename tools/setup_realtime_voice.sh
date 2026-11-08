#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/outputs/speech/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    python3 -m venv "$PROJECT_DIR/outputs/speech/venv"
fi
"$PYTHON" -m pip install -r "$PROJECT_DIR/tools/requirements/realtime_voice.txt"
printf '语音测试依赖已准备；AEC还需要系统已有的libspeexdsp1。未打开音频或连接云端。\n'
