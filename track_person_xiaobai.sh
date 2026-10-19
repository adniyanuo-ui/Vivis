#!/usr/bin/env bash
# 仅运行视觉检测和网页预览，不打开舵机串口。
set -euo pipefail
XIAOBAI_TRACK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$XIAOBAI_TRACK_ROOT/.venv/bin/python" "$XIAOBAI_TRACK_ROOT/tools/xiaobai_track_person.py" "$@"
