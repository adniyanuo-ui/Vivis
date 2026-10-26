#!/usr/bin/env bash
# Compatibility entry: wz2, one independent full round by default.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec ./play_mojia_full_show.sh --work wz2 --repeat 1 "$@"
