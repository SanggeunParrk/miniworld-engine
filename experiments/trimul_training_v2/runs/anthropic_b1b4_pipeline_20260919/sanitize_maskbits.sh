#!/bin/bash
set -uo pipefail
P=runs/anthropic_b1b4_pipeline_20260919
E=runs/anthropic_adoption_20260919/env.sh
for tool in memcheck racecheck synccheck; do
  for length in 64 384; do
    timeout -k 10s 480s bash "$E" compute-sanitizer --tool "$tool" \
      --kernel-name kns=dual_b1b4 --error-exitcode 3 \
      python -u -B "$P/sanitize_experiment.py" --source dual_maskbits \
      --length "$length" --count 132 --part 2 \
      > "$P/maskbits-$tool-L$length.log" 2>&1
    printf 'RESULT %s L%s exit=%s\n' "$tool" "$length" "$?"
  done
done
