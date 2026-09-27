#!/bin/bash
set -euo pipefail
root=/home/psk6950/MiniWorld/runs/transition_cuda_variants_20260918
for d in "$@"; do
 for l in 384 768; do
  bash "$root/env.sh" "$root/measure_pytorch.py" --d "$d" --length "$l" > "$root/pytorch-D$d-L$l.log" 2>&1
 done
done
