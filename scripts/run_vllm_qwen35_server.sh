#!/usr/bin/env bash
set -euo pipefail
export CC=/project/scripts/gcc-with-python-headers.sh
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export C_INCLUDE_PATH="/project/local/python312deb/extract/usr/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="/project/local/python312deb/extract/usr/include:${CPLUS_INCLUDE_PATH:-}"

exec /project/.venv-vllm/bin/vllm serve Qwen/Qwen3.5-2B \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1
