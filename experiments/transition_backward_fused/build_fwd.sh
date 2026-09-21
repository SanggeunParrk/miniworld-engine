#!/bin/bash
# [OUT=<name>] [NVCC=<path>] build_fwd.sh [extra nvcc flags]  -> build/<OUT>.cubin (default OUT=transition_fwd). Compute node only.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
csrc="$here/../trimul_b7b12/vendor/anthropic_v5/csrc"
out=${OUT:-transition_fwd}
nvcc=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}
mkdir -p "$here/build"
"$nvcc" -std=c++17 -O3 -arch=sm_90a --cubin -lineinfo -Xptxas=-v -I"$csrc" "$@" \
  "$here/src/transition_fwd.cu" -o "$here/build/$out.cubin" 2> "$here/build/$out.ptxas.log" \
  || { cat "$here/build/$out.ptxas.log" >&2; exit 1; }
grep -E "Used|spill|stack" "$here/build/$out.ptxas.log" | tail -3
