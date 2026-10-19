#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
python "$EVA_PROJECT_ROOT/tools/prepare_trt_layout.py"
cmake -S "$EVA_PROJECT_ROOT/tools/a2f-build" -B "$EVA_PROJECT_ROOT/third_party/Audio2Face-3D-SDK/_build/local" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCUDAToolkit_ROOT="$CUDA_PATH" -DTENSORRT_ROOT_DIR="$TENSORRT_ROOT_DIR" -DTEST_DATA_DIR="$EVA_PROJECT_ROOT/third_party/Audio2Face-3D-SDK"
cmake --build "$EVA_PROJECT_ROOT/third_party/Audio2Face-3D-SDK/_build/local" --target eva-a2f-smoke eva-a2f-export eva-a2f-solve --parallel 8
