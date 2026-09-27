#!/bin/bash
set -euo pipefail
width="$1"
for length in 384 768; do
 bash runs/transition_triton_wide_20260918/env.sh runs/transition_segmented_small_20260918/measure_default.py --width "$width" --length "$length"
done
bash runs/transition_triton_wide_20260918/env.sh --memcheck -m pytest runs/trimul_sm90_parity_20260917/engine/tests/numerics/test_transition_segmented_small_gpu.py -q -k "autotuned_wrapper and $width"
