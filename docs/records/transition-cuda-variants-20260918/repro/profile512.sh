#!/bin/bash
set -euo pipefail
root=/home/psk6950/MiniWorld/runs/transition_cuda_variants_20260918
for variant in streamed_k full_k; do
 bash "$root/profile_selected.sh" 512 "$variant" > "$root/profile-selected-$variant-D512.log" 2>&1
done
