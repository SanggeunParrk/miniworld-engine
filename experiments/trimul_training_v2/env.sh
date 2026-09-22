#!/usr/bin/env bash
set -euo pipefail
research_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MINIWORLD_RESEARCH_ROOT="$research_root"
export MINIWORLD_ANTHROPIC_ROOT="$research_root/runs/trimul_sm90_parity_20260917/engine/third_party/anthropic/upstream"
export PYTHONPATH="$research_root/runs/trimul_sm90_parity_20260917/engine/src:$MINIWORLD_ANTHROPIC_ROOT/common/opt_core:$research_root/runs${PYTHONPATH:+:$PYTHONPATH}"
export TRIMUL_NATIVE_BUILD_DIR="$research_root/build/native"
export MODEL_OPT_JIT_ROOT="$research_root/build/jit"
export TORCH_EXTENSIONS_DIR="$research_root/build/extensions"
export TRITON_CACHE_DIR="$research_root/build/triton"
export OPT_CORE_CELL_CENSUS_PRINT=0
export MAX_JOBS=${MAX_JOBS:-4}
exec "$@"
