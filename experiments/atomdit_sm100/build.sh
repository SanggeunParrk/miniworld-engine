#!/bin/bash
# [OUT=<name>] build.sh <src-stem> [extra nvcc flags]  ->  build/<OUT|stem>.cubin (sm_100a). Run on the B200 box.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stem=$1; shift
out=${OUT:-$stem}
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
mkdir -p "$here/build"
"$nvcc" -std=c++17 -O3 -arch=sm_100a --cubin -lineinfo -Xptxas=-v "$@" \
  "$here/src/$stem.cu" -o "$here/build/$out.cubin" 2> "$here/build/$out.ptxas.log" \
  || { cat "$here/build/$out.ptxas.log" >&2; exit 1; }
grep -E "Used|spill|stack" "$here/build/$out.ptxas.log" | tail -3
