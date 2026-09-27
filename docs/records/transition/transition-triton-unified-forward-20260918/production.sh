#!/bin/bash
set -euo pipefail
bash runs/transition_triton_wide_20260918/env.sh runs/transition_triton_next_20260918/smoke.py
bash runs/transition_triton_wide_20260918/env.sh runs/transition_triton_next_20260918/tune_production.py --width 128
bash runs/transition_triton_wide_20260918/env.sh runs/transition_triton_next_20260918/tune_production.py --width 256
