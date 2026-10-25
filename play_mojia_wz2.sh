#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec ./play_mojia_full_show.sh \
    --profile mojia/wz2_expression15_MOJIA_MOTOR_TARGETS_60fps_BLINK_NATURAL1000_PROFILE.json \
    --targets mojia/wz2_expression15_MOJIA_MOTOR_TARGETS_60fps_BLINK_NATURAL1000.json \
    --audio mojia/wz2_expression15.wav --repeat 1 "$@"
