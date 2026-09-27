#!/bin/bash
set -euo pipefail
width="$1"
for length in 384 768; do
 bash runs/transition_triton_wide_20260918/env.sh runs/transition_segmented_small_20260918/measure_default.py --width "$width" --length "$length"
done
bash runs/transition_triton_wide_20260918/env.sh -m pytest runs/trimul_sm90_parity_20260917/engine/tests/numerics/test_transition_b2b_residual_gpu.py -q -k "module_ops and $width"
if [ "$width" = 128 ]; then
 bash runs/transition_triton_wide_20260918/env.sh -m pytest runs/trimul_sm90_parity_20260917/engine/tests/numerics/test_transition_wide_b2b_gpu.py -q
 bash runs/transition_triton_next_20260918/check_registry.sh
fi
