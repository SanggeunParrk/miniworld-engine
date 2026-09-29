#!/bin/bash
# A100 environment (same as ../a100_anthropic_baseline): FoldForge venv (torch 2.10 cu128, triton 3.6, cuEq 0.11.1), this repo's src,
# and Anthropic's opt_core at the pinned revision (refs/uplifting-biomolecular-modeling @ f4f62fa6).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
# the FoldForge venv (what FoldForge's former scripts/activate_env.sh did): GCC 14 C++ runtime for the FA2 wheel, GCC 12.4 for the JIT
source "${FOLDFORGE_ENV:-/home/psk6950/practice/FoldForge/.venv}/bin/activate"
RT=${MINIWORLD_CXX_RUNTIME:-/home/psk6950/.cache/miniworld-runtime/gcc14}
[[ -f "$RT/libstdc++.so.6" ]] && export LD_LIBRARY_PATH="$RT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -x /opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++ ]]; then
  export CC=${CC:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/gcc} CXX=${CXX:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++}
fi
export ANTHROPIC_ROOT=${ANTHROPIC_ROOT:-/home/psk6950/practice/refs/uplifting-biomolecular-modeling}
export PYTHONPATH=$REPO/src:$ANTHROPIC_ROOT/common/opt_core
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
CACHE=${A100_BASELINE_CACHE:-$HOME/.cache/miniworld-a100}
export TRITON_CACHE_DIR=$CACHE/triton MODEL_OPT_JIT_ROOT=$CACHE/jit OPT_CORE_VERDICT_DIR=$CACHE/verdict
export TORCH_EXTENSIONS_DIR=$CACHE/ext TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
export OPT_CORE_CELL_CENSUS_PRINT=0 PYTHONNOUSERSITE=1
mkdir -p "$CACHE"
exec "$@"
