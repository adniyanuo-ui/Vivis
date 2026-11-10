#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 1 || $# -eq 1 && "$1" != --voice-preserve ]]; then
    printf '用法：./tools/setup_realtime_voice.sh [--voice-preserve]\n' >&2
    exit 2
fi
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
PYTHON="$PROJECT_DIR/outputs/speech/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    python3 -m venv "$PROJECT_DIR/outputs/speech/venv"
fi
"$PYTHON" -m pip install -r "$PROJECT_DIR/tools/requirements/realtime_voice.txt"
printf '语音测试依赖已准备（含AEC3）；Speex对照还需libspeexdsp1。未打开音频或连接云端。\n'
if [[ $# -eq 1 ]]; then
    "$PROJECT_DIR/tools/build_voice_preserve.sh"
fi
