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
    --track-person)
      XIAOBAI_PLAY_EXTRA+=(--track-person); shift ;;
    --track-neck)
      XIAOBAI_PLAY_EXTRA+=(--track-neck); shift ;;
    --neck-follow-trial|--neck-follow-sign|--neck-follow-offset-counts|--neck-speed-multiplier)
      [[ $# -ge 2 ]] || { echo "颈部参数缺少值：$1" >&2; exit 2; }
      XIAOBAI_PLAY_EXTRA+=("$1" "$2"); shift 2 ;;
    --gaze-direction-trial)
      XIAOBAI_PLAY_EXTRA+=(--gaze-direction-trial); shift ;;
    --gaze-trial-offset-counts)
      [[ $# -ge 2 ]] || { echo '用法：--gaze-trial-offset-counts 40 或 80' >&2; exit 2; }
      XIAOBAI_PLAY_EXTRA+=(--gaze-trial-offset-counts "$2"); shift 2 ;;
    --tracking-config)
      [[ $# -ge 2 ]] || { echo '用法：--tracking-config 配置路径' >&2; exit 2; }
      XIAOBAI_PLAY_EXTRA+=(--tracking-config "$2"); shift 2 ;;
    --max-seconds)
      [[ $# -ge 2 ]] || { echo '用法：--max-seconds 秒数' >&2; exit 2; }
      XIAOBAI_PLAY_EXTRA+=(--max-seconds "$2"); shift 2 ;;
    -h|--help)
      echo '用法：./play_0001_xiaobai.sh [--fps 30|60] [--check] [--max-seconds 秒数] [--track-person|--track-neck|--gaze-direction-trial] [--tracking-config 路径]'
      echo '默认：指定0001 JSON + 0001.wav，60fps、100%幅度、反馈只记录、颈部保持。'
      echo '--check 只检查资料、串口权限与已归位颈部，不播放或驱动。'
      echo '--track-person 使用本地视觉服务，仅接管眼球水平轴8、23；须先确认跟踪配置。'
      echo '--track-neck 完整歌曲中眼球8、23先响应，颈部32平滑跟随；默认颈部±160计数，末尾缓慢回中。'
      echo '--neck-follow-trial 32 眼球先动、颈轴延迟跟随；中位±20计数，默认30秒，末尾缓慢回中。ID34已确认为侧弯。'
      echo '--neck-follow-sign -1|1 试验颈轴的编码器方向，默认1，实际物理方向待现场确认。'
      echo '--neck-follow-offset-counts 20|80|160 颈部范围，与现有设备限位取交集，回中时间相应延长。'
      echo '--neck-speed-multiplier 1|2|3 调整颈部速度；3倍保持2倍档的加减速，眼球、音频和行程不变。'
      echo '--gaze-direction-trial 首次方向试验：眼球±40计数、限速60计数/秒，默认20秒；其他表情沿用原片。'
      echo '--gaze-trial-offset-counts 80 将试验范围增至±80计数，限速120计数/秒。'
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
if [[ " ${XIAOBAI_PLAY_EXTRA[*]} " == *" --track-neck "* ]]; then
  echo "播放：${XIAOBAI_PLAY_FPS}fps，100%面部幅度，眼球8、23与颈部32平滑跟随，反馈只记录。"
elif [[ " ${XIAOBAI_PLAY_EXTRA[*]} " == *" --neck-follow-trial "* ]]; then
  echo "播放：${XIAOBAI_PLAY_FPS}fps，100%面部幅度，反馈只记录，单颈轴小幅延迟跟随试验。"
else
  echo "播放：${XIAOBAI_PLAY_FPS}fps，100%幅度，反馈只记录，颈部保持2047。"
fi
echo "日志：$XIAOBAI_PLAY_OUTPUT"
echo '按 Ctrl+C 停止。'
exec "$XIAOBAI_PLAY_ROOT/.venv/bin/python" "$XIAOBAI_PLAY_ROOT/tools/xiaobai_observation_player.py" \
  --profile "$XIAOBAI_PLAY_ROOT/xiaobai/playback_profile_0001_observation.json" \
  --clip "$XIAOBAI_PLAY_ROOT/outputs/0001_CLAIRE_SINGING_GAZE_v3.0-Claire_BLENDSHAPES_61_60fps.json" \
  --audio "$XIAOBAI_PLAY_ROOT/0001.wav" \
  --fps "$XIAOBAI_PLAY_FPS" --monitor-neck-hold \
  --feedback-polls-per-frame "$((60 / XIAOBAI_PLAY_FPS))" \
  --output "$XIAOBAI_PLAY_OUTPUT" "${XIAOBAI_PLAY_EXTRA[@]}"
