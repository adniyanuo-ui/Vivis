#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
EVA_VARIANT_DIR="$1"
EVA_VARIANT_VIDEO="$2"
EVA_VARIANT_AUDIO="${3:-$EVA_PROJECT_ROOT/5.wav}"
EVA_AUDIO_GAIN="${4:-4}"
python "$EVA_PROJECT_ROOT/tools/validate_variant.py" "$EVA_VARIANT_DIR" "$EVA_VARIANT_VIDEO"
ffmpeg -hide_banner -y -framerate 30 -start_number 0 -i "$EVA_VARIANT_DIR/frames/%04d.png" -i "$EVA_VARIANT_AUDIO" -map 0:v:0 -map 1:a:0 -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -af "volume=$EVA_AUDIO_GAIN" -c:a aac -b:a 192k -movflags +faststart "$EVA_VARIANT_VIDEO"
python "$EVA_PROJECT_ROOT/tools/validate_variant.py" "$EVA_VARIANT_DIR" "$EVA_VARIANT_VIDEO"
