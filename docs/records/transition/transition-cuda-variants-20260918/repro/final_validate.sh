#!/bin/bash
set -uo pipefail
root=/home/psk6950/MiniWorld/runs/transition_cuda_variants_20260918
test=/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/tests/numerics/test_transition_cuda_variants_gpu.py
bash "$root/env.sh" "$root/validation_manifest.py"
bash "$root/env.sh" -m pytest -q "$test" --disable-warnings > "$root/pytest-final.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/pytest-final.exit"
if [ "$status" -ne 0 ]; then exit "$status"; fi
bash "$root/env.sh" --memcheck -m pytest -q "$test" -k gradients --disable-warnings > "$root/memcheck-final.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/memcheck-final.exit"
exit "$status"
