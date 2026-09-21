#!/bin/bash
# [OUT=<name>] [NVCC=<path>] build_fwd.sh [<src-stem>] [extra nvcc flags]
#   -> build/<OUT|src-stem>.cubin from src/<src-stem>.cu (default stem and OUT: transition_fwd). Compute node only.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
csrc="$here/../trimul_b7b12/vendor/anthropic_v5/csrc"
stem=transition_fwd
if [ $# -gt 0 ] && [ -f "$here/src/$1.cu" ]; then stem=$1; shift; fi
out=${OUT:-$stem}
nvcc=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}
mkdir -p "$here/build"
"$nvcc" -std=c++17 -O3 -arch=sm_90a --cubin -lineinfo -Xptxas=-v -I"$csrc" "$@" \
  "$here/src/$stem.cu" -o "$here/build/$out.cubin" 2> "$here/build/$out.ptxas.log" \
  || { cat "$here/build/$out.ptxas.log" >&2; exit 1; }
grep -E "Used|spill|stack" "$here/build/$out.ptxas.log" | tail -3
