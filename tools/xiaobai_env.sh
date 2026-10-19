#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export LD_LIBRARY_PATH="$EVA_PROJECT_ROOT/tools/audio-deps/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
