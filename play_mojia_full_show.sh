#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="$PWD/.venv/bin/python"
COMMON=(--config mojia/person_gaze.json
        --profile mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK_PROFILE.json
        --targets mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK.json
        --audio exmple/0001.wav --authorized-tongue-observation
        --usb-volume-times-three --hold-vertical-eyes --repeat 3)
for arg in "$@"; do
    if [[ "$arg" == --check ]]; then
        exec "$PYTHON" tools/mojia_performance.py "${COMMON[@]}" --output /tmp/vivis-show-offline "$@"
    fi
done
RUN_DIR="outputs/mojia/manual_show_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_DIR"
if ! curl --silent --fail --max-time 2 http://127.0.0.1:8765/state >/dev/null; then
    "$PYTHON" -u tools/xiaobai_track_person.py --config mojia/person_gaze.json \
        --output "$RUN_DIR/vision" >"$RUN_DIR/vision.log" 2>&1 &
    vision_pid=$!
    ready=0
    for ((i=0; i<30; i++)); do
        if curl --silent --fail --max-time 1 http://127.0.0.1:8765/state >/dev/null; then
            ready=1; break
        fi
        if ! kill -0 "$vision_pid" 2>/dev/null; then
            cat "$RUN_DIR/vision.log"; exit 1
        fi
        sleep 0.5
    done
    if [[ "$ready" != 1 ]]; then
        echo "视觉服务未就绪，请查看 $RUN_DIR/vision.log" >&2
        exit 1
    fi
fi
printf '完整歌曲＋横向眼球/颈部跟踪；ID7/22保持原位；音量为原始基准的3倍幅度。\n监控：http://127.0.0.1:8765/\n日志：%s/performance\nCtrl+C 可停止，停止后保持颈部扭矩。\n' "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > outputs/mojia/latest_full_show.txt
exec "$PYTHON" -u tools/mojia_performance.py "${COMMON[@]}" --output "$RUN_DIR/performance" "$@"
