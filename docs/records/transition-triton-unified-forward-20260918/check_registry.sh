#!/bin/bash
set -euo pipefail
cd /home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine
bash /home/psk6950/MiniWorld/runs/transition_triton_wide_20260918/env.sh -m pytest tests/registry/test_config_axes_match_the_kernel.py tests/registry/test_driver_imports_resolve.py tests/registry/test_every_gemm_orders_its_tiles_or_says_why.py tests/registry/test_no_kernel_launch_skips_the_autotuner.py -q
