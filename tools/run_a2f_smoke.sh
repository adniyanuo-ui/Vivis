#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$EVA_PROJECT_ROOT/third_party/Audio2Face-3D-SDK/_build/local/eva-a2f-smoke" "${1:-$EVA_PROJECT_ROOT/assets/audio/official_4sec.wav}"
