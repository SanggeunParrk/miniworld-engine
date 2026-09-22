#!/bin/bash
set -euo pipefail
R=runs/trimul_fused_recompute_20260920
E=runs/anthropic_adoption_20260919/env.sh
timeout 300 bash "$E" compute-sanitizer --tool memcheck --kernel-name 'regex=b1_fused|front_b7b12' --error-exitcode 42 python -u -B "$R/bench.py" --length 768 --check-only > "$R/memcheck-L768.log" 2>&1
timeout 300 bash "$E" ncu --set full --cache-control none --clock-control none --profile-from-start off --kernel-name 'regex:b1_fused|front_b7b12' --force-overwrite -o "$R/ncu-both-L768" python -u -B "$R/bench.py" --length 768 --profile > "$R/ncu-both-L768.log" 2>&1
bash "$E" ncu --import "$R/ncu-both-L768.ncu-rep" --page raw --csv > "$R/ncu-both-L768.csv" 2> "$R/ncu-both-export.log"
