#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$EXAMPLE_DIR/../.." && pwd)"
PYTHON="${VIVIS_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]] && ! command -v -- "$PYTHON" > /dev/null 2>&1; then
    printf '缺少Python环境，请在仓库根目录先执行 ./tools/setup.sh\n' >&2
    exit 1
fi
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONNOUSERSITE=1
cd -- "$PROJECT_DIR"
exec "$PYTHON" "$EXAMPLE_DIR/main.py" "$@"
