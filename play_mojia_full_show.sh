#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="$PWD/.venv/bin/python"
COMMON=(--config mojia/person_gaze.json
        --profile mojia/0001_MOJIA_MOTOR_TARGETS_60fps_BLINK_NATURAL1000_PROFILE.json
        --targets mojia/0001_MOJIA_MOTOR_TARGETS_60fps_BLINK_NATURAL1000.json
        --audio exmple/0001.wav --authorized-tongue-observation
        --usb-volume-times-three --hold-vertical-eyes --face-fps 60 --mouth-gain 1.0 --repeat 3)
for arg in "$@"; do
    if [[ "$arg" == --check ]]; then
        exec "$PYTHON" tools/mojia_performance.py "${COMMON[@]}" --output /tmp/vivis-show-offline "$@"
    fi
done
home_only=0
for arg in "$@"; do
    if [[ "$arg" == --home-only ]]; then home_only=1; fi
done
if command -v systemctl >/dev/null; then
    if systemctl is-active --quiet dms.service; then
        echo 'dms.service正在运行，可能占用串口并覆盖舵机参数。请先执行：sudo systemctl stop dms.service' >&2
        exit 1
    fi
    if [[ "$home_only" == 0 ]] && systemctl is-active --quiet ies.service; then
        echo 'ies.service正在运行，其face_track可能占用右眼相机。请先执行：sudo systemctl stop ies.service' >&2
        exit 1
    fi
fi
RUN_DIR="outputs/mojia/manual_show_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_DIR"
if [[ "$home_only" == 0 ]] && ! curl --silent --fail --max-time 2 http://127.0.0.1:8765/state >/dev/null; then
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
if [[ "$home_only" == 1 ]]; then
    printf '仅执行35轴缓慢回中，不启动相机或音频。\n日志：%s/performance\n' "$RUN_DIR"
else
    printf '每遍独立准备、35轴缓慢回中，再从头完整播放音频与表情＋横向眼球/颈部跟踪；定时控制默认60 Hz，对照试验可指定30 Hz。\n回中后ID7/22保持原位；音量为原始基准的3倍幅度。\n监控：http://127.0.0.1:8765/\n日志：%s/performance\nCtrl+C 可停止，停止后保持颈部扭矩。\n' "$RUN_DIR"
fi
printf '%s\n' "$RUN_DIR" > outputs/mojia/latest_full_show.txt
exec "$PYTHON" -u tools/mojia_performance.py "${COMMON[@]}" --output "$RUN_DIR/performance" "$@"
