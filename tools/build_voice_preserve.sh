#!/usr/bin/env bash
# Build one optional extension in the isolated speech environment, never system Python.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
PYTHON="$PROJECT_DIR/outputs/speech/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    printf '请先执行 ./tools/setup_realtime_voice.sh\n' >&2
    exit 1
fi
SOURCE_META="$PROJECT_DIR/native/speech/voice_preserve/sources.json"
COMMIT="$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["commit"])' "$SOURCE_META")"
REPOSITORY="$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["repository"])' "$SOURCE_META")"
PYBIND_VERSION="$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["pybind11"])' "$SOURCE_META")"
SOURCE_DIR="$PROJECT_DIR/outputs/speech/source/pywebrtc-audio-0.2.0"
PATCH="$PROJECT_DIR/native/speech/voice_preserve/profile.patch"
PATCH_HASH="$(sha256sum "$PATCH" | cut -d ' ' -f 1)"
BUILD_ROOT="$PROJECT_DIR/outputs/speech/voice_preserve/${COMMIT:0:12}-${PATCH_HASH:0:12}"
BUILD_PYTHON="$PROJECT_DIR/outputs/speech/build-venv/bin/python"
if [[ ! -x "$BUILD_PYTHON" ]]; then
    "$PYTHON" -m venv "$PROJECT_DIR/outputs/speech/build-venv"
fi
"$BUILD_PYTHON" -m pip install "pybind11==$PYBIND_VERSION"
mkdir -p "$(dirname -- "$SOURCE_DIR")"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    git clone --no-checkout "$REPOSITORY" "$SOURCE_DIR"
fi
if ! git -C "$SOURCE_DIR" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
    git -C "$SOURCE_DIR" fetch origin "$COMMIT"
fi
if [[ ! -d "$BUILD_ROOT/source" ]]; then
    mkdir -p "$BUILD_ROOT"
    TEMP_DIR="$(mktemp -d "$BUILD_ROOT/source.XXXXXX")"
    trap 'rm -rf -- "$TEMP_DIR"' EXIT
    git -C "$SOURCE_DIR" archive "$COMMIT" | tar -x -C "$TEMP_DIR"
    "$PYTHON" - "$TEMP_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
meta=json.loads(Path('native/speech/voice_preserve/sources.json').read_text())
source=Path(sys.argv[1])
patch=Path('native/speech/voice_preserve/profile.patch').read_bytes()
if hashlib.sha256(patch).hexdigest()!=meta['patch_sha256']:raise SystemExit('Patch hash mismatch')
for name,expected in meta['upstream_files_sha256'].items():
    if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:raise SystemExit('Upstream hash mismatch: '+name)
PY
    patch --batch --forward -d "$TEMP_DIR" -p1 -i "$PATCH"
    mv -- "$TEMP_DIR" "$BUILD_ROOT/source"
    trap - EXIT
fi
PYBIND_DIR="$("$BUILD_PYTHON" -m pybind11 --cmakedir)"
SITE_PACKAGES="$("$PYTHON" -c 'import sysconfig;print(sysconfig.get_path("platlib"))')"
printf '编译可选人声保护扩展；原AEC3安装包保持独立。日志：%s/build.log\n' "$BUILD_ROOT"
cmake -S "$BUILD_ROOT/source" -B "$BUILD_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$PYBIND_DIR" \
    -DPYTHON_EXECUTABLE="$PYTHON" > "$BUILD_ROOT/build.log" 2>&1
cmake --build "$BUILD_ROOT/build" --parallel 4 >> "$BUILD_ROOT/build.log" 2>&1
cmake --install "$BUILD_ROOT/build" --prefix "$SITE_PACKAGES" >> "$BUILD_ROOT/build.log" 2>&1
# Keep upstream notices beside the local build; no binaries are added to Git.
mkdir -p "$BUILD_ROOT/licenses"
cp -- "$BUILD_ROOT/source/LICENSE" "$BUILD_ROOT/licenses/WRAPPER_LICENSE"
cp -- "$BUILD_ROOT/source/NOTICE" "$BUILD_ROOT/licenses/NOTICE"
cp -- "$BUILD_ROOT/source/vendor/webrtc_audio/PATENTS" "$BUILD_ROOT/licenses/WEBRTC_PATENTS"
cp -- "$BUILD_ROOT/source/vendor/webrtc_audio/MODIFICATIONS.md" "$BUILD_ROOT/licenses/UPSTREAM_MODIFICATIONS.md"
cp -- "$BUILD_ROOT/source/vendor/webrtc_audio/abseil/LICENSE" "$BUILD_ROOT/licenses/ABSEIL_LICENSE"
cp -- "$BUILD_ROOT/source/vendor/webrtc_audio/jsoncpp/LICENSE" "$BUILD_ROOT/licenses/JSONCPP_LICENSE"
"$PYTHON" - "$SITE_PACKAGES" <<'PY'
import hashlib,json,platform,sys
from pathlib import Path
import _vivis_aec3,numpy as np
if _vivis_aec3.vivis_profile_api_version!=1:raise SystemExit('Unexpected profile API')
engine=_vivis_aec3.AudioProcessor(sample_rate=16000,echo_cancellation=True,
    noise_suppression=True,auto_gain_control=False,nearend_threshold=1.)
result=engine.process(np.zeros(320,np.float32),np.zeros(320,np.float32))
if result.shape!=(320,) or not np.isfinite(result).all():raise SystemExit('Native smoke check failed')
meta=json.loads(Path('native/speech/voice_preserve/sources.json').read_text())
meta.update(python=sys.version,platform=platform.platform(),
    binary_sha256=hashlib.sha256(Path(_vivis_aec3.__file__).read_bytes()).hexdigest())
(Path(sys.argv[1])/'_vivis_aec3_build.json').write_text(json.dumps(meta,indent=2)+'\n')
print('可选人声保护扩展构建完成；未打开音频、云端或机器头。')
PY
