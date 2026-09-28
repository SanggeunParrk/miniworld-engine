#!/usr/bin/env bash
# Run inside the activated Vast engine environment, normally via vast-sync.sh run.
set -euo pipefail
blas=/workspace/envs/engine/lib/python3.10/site-packages/nvidia/cublas/lib
export LD_PRELOAD="$blas/libcublasLt.so.12:$blas/libcublas.so.12"
exec "$@"
