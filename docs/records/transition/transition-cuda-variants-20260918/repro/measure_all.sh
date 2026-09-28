#!/bin/bash
set -euo pipefail
root=/home/psk6950/MiniWorld/runs/transition_cuda_variants_20260918
for d in "$@"; do
 for l in 384 768; do
  if ! bash "$root/env.sh" "$root/is_complete.py" module "$d" "$l"; then
   bash "$root/env.sh" "$root/measure_module.py" --d "$d" --length "$l" > "$root/module-D$d-L$l.log" 2>&1
  fi
  bash "$root/env.sh" "$root/breakdown.py" --d "$d" --length "$l" > "$root/breakdown-D$d-L$l.log" 2>&1
 done
 if [ "$d" -ge 384 ]; then
  for variant in streamed_k full_k; do
   bash "$root/profile_selected.sh" "$d" "$variant" > "$root/profile-selected-$variant-D$d.log" 2>&1
  done
 fi
done
