#!/usr/bin/env bash
# Preparation calls Qwen and native A2F; --check only verifies cached files.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONNOUSERSITE=1
exec "$PROJECT_DIR/outputs/speech/venv/bin/python" -m vivis.speech.prepare_fixed_replies "$@"
