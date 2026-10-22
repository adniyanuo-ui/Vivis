#!/usr/bin/env bash
# Prerecorded playback needs Python and PortAudio, not the A2F CUDA/TRT layout.
EVA_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$EVA_PROJECT_ROOT/.venv/bin:$PATH"
case "$(uname -m)" in
  x86_64) XIAOBAI_AUDIO_ARCH=x86_64-linux-gnu ;;
  aarch64) XIAOBAI_AUDIO_ARCH=aarch64-linux-gnu ;;
  *) XIAOBAI_AUDIO_ARCH= ;;
esac
if [[ -n "$XIAOBAI_AUDIO_ARCH" && -d "$EVA_PROJECT_ROOT/tools/audio-deps/usr/lib/$XIAOBAI_AUDIO_ARCH" ]]; then
  export LD_LIBRARY_PATH="$EVA_PROJECT_ROOT/tools/audio-deps/usr/lib/$XIAOBAI_AUDIO_ARCH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
