#!/bin/bash
set -uo pipefail
P=runs/anthropic_b1b4_pipeline_20260919
E=runs/anthropic_adoption_20260919/env.sh
timeout -k 5s 120s bash "$E" python -u -B "$P/check_dspref.py" > "$P/dspref.log" 2>&1
timeout -k 5s 180s bash "$E" compute-sanitizer --tool memcheck --kernel-name kns=dual_b1b4 --error-exitcode 3 python -u -B "$P/sanitize_dual.py" > "$P/memcheck-persistent.log" 2>&1
printf 'memcheck exit: %s\n' "$?"
timeout -k 5s 180s bash "$E" compute-sanitizer --tool racecheck --kernel-name kns=dual_b1b4 --racecheck-report analysis --error-exitcode 3 python -u -B "$P/sanitize_dual.py" > "$P/racecheck-persistent.log" 2>&1
printf 'racecheck exit: %s\n' "$?"
timeout -k 5s 180s bash "$E" ncu --profile-from-start off --target-processes all --set full --force-overwrite -o "$P/dual-profile" python -u -B "$P/profile_dual.py" > "$P/ncu-dual.log" 2>&1
printf 'ncu exit: %s\n' "$?"
bash "$E" ncu --import "$P/dual-profile.ncu-rep" --page raw --csv > "$P/dual-profile.csv" 2> "$P/ncu-import-dual.log"
