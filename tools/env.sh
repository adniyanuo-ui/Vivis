#!/usr/bin/env bash
EVA_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_PATH="$EVA_PROJECT_ROOT/tools/cuda-12.9"
export CUDACXX="$CUDA_PATH/bin/nvcc"
export TENSORRT_ROOT_DIR="$EVA_PROJECT_ROOT/tools/tensorrt"
export PATH="$EVA_PROJECT_ROOT/.venv/bin:$CUDA_PATH/bin:$EVA_PROJECT_ROOT/tools/bin:$PATH"
export LD_LIBRARY_PATH="$TENSORRT_ROOT_DIR/lib:$CUDA_PATH/lib:$EVA_PROJECT_ROOT/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="$EVA_PROJECT_ROOT/tools/tensorrt-debs/usr/lib/python3.10/dist-packages:$EVA_PROJECT_ROOT/tools/tensorrt-debs/usr/lib/python3/dist-packages:${PYTHONPATH:-}"
