#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
python - "$EVA_PROJECT_ROOT" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
folder=root/'outputs/5_frames'
missing=[i for i in range(830) if not (folder/f'{i:04d}.png').is_file()]
if missing:raise SystemExit(f'Missing render frames: {missing[:10]} ({len(missing)} total)')
PY
ffmpeg -hide_banner -y -framerate 30 -start_number 0 -i "$EVA_PROJECT_ROOT/outputs/5_frames/%04d.png" -i "$EVA_PROJECT_ROOT/5.wav" -map 0:v:0 -map 1:a:0 -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -af volume=4 -c:a aac -b:a 192k -movflags +faststart "$EVA_PROJECT_ROOT/outputs/5_A2F_MARK_PREVIEW.mp4"
ffmpeg -hide_banner -i "$EVA_PROJECT_ROOT/outputs/5_A2F_MARK_PREVIEW.mp4" -f null -
