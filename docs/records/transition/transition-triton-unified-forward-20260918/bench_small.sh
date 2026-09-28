#!/bin/bash
set -euo pipefail
cd /home/psk6950/MiniWorld
for width in 128 256; do
 for length in 384 768; do
  bash runs/transition_triton_wide_20260918/env.sh runs/transition_triton_next_20260918/measure.py --width "$width" --length "$length"
 done
done
bash runs/transition_triton_wide_20260918/env.sh --memcheck -m pytest runs/trimul_sm90_parity_20260917/engine/tests/numerics/test_transition_b2b_residual_gpu.py -q
