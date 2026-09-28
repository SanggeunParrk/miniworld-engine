#!/usr/bin/env bash
# Invoke through scripts/vast-sync.sh run GPU bash <this file>.
set -euo pipefail
out=/workspace/vast-results/trimul-large-d
blas=/workspace/envs/engine/lib/python3.10/site-packages/nvidia/cublas/lib
sanitizer=${VAST_SANITIZER:-/workspace/tools/sanitizer132/bin/compute-sanitizer}
[[ -x "$sanitizer" ]] || { echo "Install the isolated CUDA 13.2 sanitizer; see README.md" >&2; exit 2; }
mkdir -p "$out"
"$sanitizer" --version > "$out/sanitizer-version.txt"
for tool in memcheck racecheck synccheck; do
  set +e
  "$sanitizer" --tool "$tool" --error-exitcode 86 \
    --preload-library "$blas/libcublasLt.so.12" \
    --preload-library "$blas/libcublas.so.12" \
    python experiments/trimul_large_d_vast/qualify_input_split.py --sanitize \
    > "$out/split4-$tool.log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$out/split4-$tool.exit"
  if (( rc != 0 )); then tail -n 8 "$out/split4-$tool.log"; exit "$rc"; fi
  grep -E 'SANITIZER_DONE|SUMMARY' "$out/split4-$tool.log"
done
