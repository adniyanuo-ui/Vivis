#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONNOUSERSITE=1
exec "$PROJECT_DIR/outputs/speech/venv/bin/python" -m vivis.speech.cloud_tts "$@"
