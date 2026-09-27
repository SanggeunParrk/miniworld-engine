#!/bin/bash
set -euo pipefail
root=/home/psk6950/MiniWorld/runs/transition_cuda_variants_20260918
variant=$1
for d in 128 256 384 512; do
 echo "START $variant D$d $(date --iso-8601=seconds)"
 bash "$root/env.sh" "$root/smoke.py" --variant "$variant" --d "$d" > "$root/final-smoke-$variant-D$d.log" 2>&1
 echo "DONE $variant D$d $(date --iso-8601=seconds)"
done
