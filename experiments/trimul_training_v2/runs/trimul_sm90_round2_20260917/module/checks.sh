#!/usr/bin/env bash
set -euo pipefail
source /home/psk6950/MiniWorld/runs/transition_triton_audit_20260917/env.sh
export PYTHONPATH=/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/src:/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine
export PYTHONNOUSERSITE=1
export CUTE_DSL_DUMP_DIR=/home/psk6950/MiniWorld/runs/trimul_sm90_round2_20260917/module/compile
mkdir -p "$CUTE_DSL_DUMP_DIR"
cd /home/psk6950/MiniWorld/runs/trimul_sm90_round2_20260917
case "$1" in
  validate)
    for task_length in 384 768; do
      python module/validate.py --configs "module/measured-configs-L${task_length}.json" --length "$task_length" --compile --output "module/validation-L${task_length}.json" > "module/validation-L${task_length}.log" 2>&1
    done
    ;;
  benchmark)
    for task_length in 384 768; do
      python module/benchmark.py --configs "module/measured-configs-L${task_length}.json" --triton-configs "module/triton-configs-L${task_length}.json" --length "$task_length" > "module/benchmark-L${task_length}.log" 2>&1
    done
    ;;
  *) exit 2 ;;
esac
