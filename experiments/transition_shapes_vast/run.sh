#!/usr/bin/env bash
set -euo pipefail
out=${1:?result directory}
shift
mkdir -p "$out"
for d in "$@"; do
  for length in 384 768; do
    python experiments/transition_shapes_vast/bench.py --width "$d" --length "$length" --out "$out" > "$out/D$d-L$length.log" 2>&1
  done
done
