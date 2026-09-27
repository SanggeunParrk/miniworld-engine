#!/bin/bash
set -euo pipefail
bash runs/transition_triton_wide_20260918/env.sh runs/transition_triton_next_20260918/tune_segmented.py --width 384
bash runs/transition_triton_wide_20260918/env.sh runs/transition_triton_next_20260918/tune_segmented.py --width 512
