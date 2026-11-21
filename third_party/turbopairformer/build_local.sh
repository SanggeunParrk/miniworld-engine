#!/bin/bash
# Build the TurboPairFormer CUDA extensions for the engine environment (python 3.12, torch 2.13+cu129, sm_90a) with the library's own development JIT
# (TURBOPAIRFORMER_ALLOW_JIT=1): the PyPI wheel is cp314 + torch 2.10 only. Needs an H100 (the JIT refuses another capability) and nvcc (the engine env's);
# run inside a Slurm job (10 min, 1 GPU) with the card's engine env on PATH. Resumable: a finished extension is cached in $TORCH_EXTENSIONS_DIR.
#   build_local.sh [forward delta dkdv dq_dbias]      (default: all four; one per job if a build does not fit the time limit)
set -euo pipefail
ROOT=${PAPER_EXT_ATTN:-$HOME/paper-ext-attn}
export PYTHONPATH=$ROOT/turbopairformer${PYTHONPATH:+:$PYTHONPATH} TURBOPAIRFORMER_ALLOW_JIT=1 TORCH_EXTENSIONS_DIR=$ROOT/torch_ext TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=${MAX_JOBS:-4}
mkdir -p "$TORCH_EXTENSIONS_DIR"
for k in "${@:-forward delta dkdv dq_dbias}"; do
  for key in $k; do
    echo "== build turbopairformer_$key $(date +%T)"
    python - "$key" <<'PY'
import importlib, sys
m = importlib.import_module({"forward": "turbopairformer.triangle_forward", "delta": "turbopairformer.triangle_delta", "dkdv": "turbopairformer.triangle_dkdv", "dq_dbias": "turbopairformer.triangle_dq_dbias"}[sys.argv[1]])
print(m.load_extension(verbose=True))
PY
  done
done
