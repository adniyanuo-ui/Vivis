#!/usr/bin/env bash
# 指定歌曲的真机观察播放；可从任意工作目录启动。
set -eo pipefail
XIAOBAI_PLAY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$XIAOBAI_PLAY_ROOT"
source "$XIAOBAI_PLAY_ROOT/tools/xiaobai_env.sh"

XIAOBAI_PLAY_FPS=60
XIAOBAI_PLAY_EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fps)
      [[ $# -ge 2 ]] || { echo '用法：--fps 30 或 --fps 60' >&2; exit 2; }
      XIAOBAI_PLAY_FPS="$2"; shift 2 ;;
    --check)
      XIAOBAI_PLAY_EXTRA+=(--check); shift ;;
    --max-seconds)
      [[ $# -ge 2 ]] || { echo '用法：--max-seconds 秒数' >&2; exit 2; }
      XIAOBAI_PLAY_EXTRA+=(--max-seconds "$2"); shift 2 ;;
    -h|--help)
      echo '用法：./play_0001_xiaobai.sh [--fps 30|60] [--check] [--max-seconds 秒数]'
      echo '默认：指定0001 JSON + 0001.wav，60fps、100%幅度、反馈只记录、颈部保持。'
      echo '--check 只检查资料、串口权限与已归位颈部，不播放或驱动。'
      exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done
case "$XIAOBAI_PLAY_FPS" in 30|60) ;; *) echo '帧率只能是30或60' >&2; exit 2 ;; esac

if [[ -e "$XIAOBAI_PLAY_ROOT/outputs/xiaobai/STOP" ]]; then
  echo '存在 outputs/xiaobai/STOP；请确认现场可以启动后再移除该停止文件。' >&2
  exit 2
fi
if ! "$XIAOBAI_PLAY_ROOT/.venv/bin/python" - <<'PY'
import os
import sys
from pathlib import Path
sys.path.insert(0,'tools')
from xiaobai_switch_usb_driver import find_interface,driver,tty_devices
device,interface=find_interface()
port=Path('/dev/ttyCH343USB0')
ready=(driver(interface)=='usb_ch343' and str(port) in tty_devices(device)
       and port.exists() and os.access(port,os.R_OK|os.W_OK))
raise SystemExit(0 if ready else 1)
PY
then
  echo '串口未就绪。USB拔插后，请先恢复驱动与权限：' >&2
  echo "sudo python3 $XIAOBAI_PLAY_ROOT/tools/xiaobai_switch_usb_driver.py vendor" >&2
  exit 2
fi

XIAOBAI_PLAY_OUTPUT="$XIAOBAI_PLAY_ROOT/outputs/xiaobai/0001_direct_$(date +%Y%m%d_%H%M%S_%N)"
if [[ " ${XIAOBAI_PLAY_EXTRA[*]} " != *" --check "* ]]; then
  mkdir -p "$XIAOBAI_PLAY_ROOT/outputs/xiaobai"
  printf '%s\n' "$XIAOBAI_PLAY_OUTPUT" > "$XIAOBAI_PLAY_ROOT/outputs/xiaobai/latest_0001_direct.txt"
fi
echo "指定JSON：0001_CLAIRE_SINGING_GAZE_v3.0-Claire_BLENDSHAPES_61_60fps.json"
echo "播放：${XIAOBAI_PLAY_FPS}fps，100%幅度，反馈只记录，颈部保持2047。"
echo "日志：$XIAOBAI_PLAY_OUTPUT"
echo '按 Ctrl+C 停止。'
exec "$XIAOBAI_PLAY_ROOT/.venv/bin/python" "$XIAOBAI_PLAY_ROOT/tools/xiaobai_observation_player.py" \
  --profile "$XIAOBAI_PLAY_ROOT/xiaobai/playback_profile_0001_observation.json" \
  --clip "$XIAOBAI_PLAY_ROOT/outputs/0001_CLAIRE_SINGING_GAZE_v3.0-Claire_BLENDSHAPES_61_60fps.json" \
  --audio "$XIAOBAI_PLAY_ROOT/0001.wav" \
  --fps "$XIAOBAI_PLAY_FPS" --monitor-neck-hold \
  --feedback-polls-per-frame "$((60 / XIAOBAI_PLAY_FPS))" \
  --output "$XIAOBAI_PLAY_OUTPUT" "${XIAOBAI_PLAY_EXTRA[@]}"
