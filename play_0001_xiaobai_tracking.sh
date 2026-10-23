#!/usr/bin/env bash
# 2026-10-19 现场认可效果：60fps、水平眼球跟随、ID32 ±160、转头3倍速。
set -eo pipefail
XIAOBAI_TRACKING_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$XIAOBAI_TRACKING_ROOT/play_0001_xiaobai.sh" \
  --track-neck --fps 60 --neck-follow-offset-counts 160 \
  --neck-speed-multiplier 3 "$@"
