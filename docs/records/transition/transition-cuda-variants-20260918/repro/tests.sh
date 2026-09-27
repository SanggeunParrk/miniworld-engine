#!/bin/bash
set -euo pipefail
root=/home/psk6950/MiniWorld/runs/transition_cuda_variants_20260918
test=/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/tests/numerics/test_transition_cuda_variants_gpu.py
variant=$1
bash "$root/env.sh" -m pytest -q "$test" -k "$variant" --disable-warnings > "$root/pytest-$variant.log" 2>&1
bash "$root/env.sh" --memcheck -m pytest -q "$test" -k "$variant and gradients" --disable-warnings > "$root/memcheck-$variant.log" 2>&1
