#!/bin/bash
set -euo pipefail
P=runs/anthropic_b1b4_pipeline_20260919
E=runs/anthropic_adoption_20260919/env.sh
timeout -k 5s 300s bash "$E" python -u -B "$P/check_experiment.py" --source dual_balanced > "$P/balanced-strict.log" 2>&1
printf 'RESULT strict24 passed\n'
timeout -k 5s 300s bash "$E" python -u -B "$P/measure_experiment.py" --sources dual_optimized dual_balanced --parts 1 2 --full --output balanced-final-results.json > "$P/balanced-final.log" 2>&1
printf 'RESULT final-benchmark passed\n'
for tool in memcheck racecheck synccheck; do
 for length in 64 384; do
  timeout -k 10s 480s bash "$E" compute-sanitizer --tool "$tool" --kernel-name kns=dual_b1b4 --error-exitcode 3 python -u -B "$P/sanitize_experiment.py" --source dual_balanced --length "$length" --count 132 --part 2 > "$P/balanced-$tool-L$length.log" 2>&1
  printf 'RESULT %s L%s passed\n' "$tool" "$length"
 done
done
for length in 64 384; do
 timeout -k 10s 180s bash "$E" compute-sanitizer --tool memcheck --error-exitcode 3 python -u -B "$P/sanitize_experiment.py" --source dual_balanced --length "$length" --count 132 --part 1 > "$P/balanced-split-memcheck-L$length.log" 2>&1
 printf 'RESULT split-memcheck L%s passed\n' "$length"
done
for length in 384 768; do
 timeout -k 5s 300s bash "$E" ncu --set full --import-source yes -k regex:dual_b1b4 --profile-from-start off --force-overwrite -o "$P/balanced-L$length" python -u -B "$P/profile_experiment.py" --source dual_balanced --length "$length" > "$P/balanced-L$length-ncu.log" 2>&1
 bash "$E" ncu --import "$P/balanced-L$length.ncu-rep" --csv --page raw > "$P/balanced-L$length.csv" 2> "$P/balanced-L$length-export.log"
 printf 'RESULT NCU L%s passed\n' "$length"
done
