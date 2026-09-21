#!/bin/bash
# Build the fused Transition backward kernel.
#   [OUT=<name>] [NVCC=<path>] build.sh [extra nvcc flags]      -> build/<OUT>.cubin   (default OUT=transition_bwd_r8)
# The only include it needs is the Anthropic v5 device header set vendored for the sibling experiment.
# Compute node only: nvcc is not available on the login node.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
csrc="$here/../trimul_b7b12/vendor/anthropic_v5/csrc"
[ -d "$csrc" ] || { echo "vendored Anthropic csrc not found at $csrc" >&2; exit 2; }
out=${OUT:-transition_bwd_r8}
nvcc=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}
mkdir -p "$here/build"
"$nvcc" -std=c++17 -O3 -arch=sm_90a --cubin -lineinfo -Xptxas=-v -I"$csrc" "$@" \
  "$here/src/transition_bwd.cu" -o "$here/build/$out.cubin" 2> "$here/build/$out.ptxas.log" \
  || { cat "$here/build/$out.ptxas.log" >&2; exit 1; }
grep -E "Used|spill|stack" "$here/build/$out.ptxas.log" | tail -3
