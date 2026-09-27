#!/bin/bash
set -euo pipefail
root=/home/psk6950/MiniWorld/runs/trimul_pytorch_compare_20260919
envscript=/home/psk6950/MiniWorld/runs/transition_cuda_variants_20260918/env.sh
for round in 0 1; do
 if [ ! -f "$root/current-L$1-r$round.json" ]; then
  bash "$envscript" "$root/measure.py" --version current --length "$1" --round "$round" > "$root/training-L$1-r$round.log" 2>&1
 fi
 bash "$envscript" "$root/inference.py" --version current --length "$1" --round "$round" > "$root/inference-L$1-r$round.log" 2>&1
done
