#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$EXAMPLE_DIR/../.." && pwd)"
PYTHON="${VIVIS_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$PROJECT_DIR"
exec "$PYTHON" "$EXAMPLE_DIR/main.py" "$@"
