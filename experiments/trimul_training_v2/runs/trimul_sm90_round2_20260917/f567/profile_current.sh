set -euo pipefail
source /home/psk6950/MiniWorld/runs/transition_triton_audit_20260917/env.sh
cd /home/psk6950/MiniWorld/runs/trimul_sm90_round2_20260917/f567
export PYTHONPATH=/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/src
export PYTHONNOUSERSITE=1 CUTE_DSL_LINEINFO=1
export CUTE_DSL_DUMP_DIR=/home/psk6950/MiniWorld/runs/trimul_sm90_round2_20260917/f567/compile
mkdir -p compile
for length in 384 768; do
 /usr/local/cuda/bin/ncu --target-processes all --profile-from-start off --cache-control none --set full --import-source yes --launch-count 1 --force-overwrite --export "current-L$length" python ncu_current.py "$length" > "ncu-current-L$length.log" 2>&1
 /usr/local/cuda/bin/ncu --import "current-L$length.ncu-rep" --page details > "ncu-current-L$length-details.txt"
 /usr/local/cuda/bin/ncu --import "current-L$length.ncu-rep" --page raw --csv > "ncu-current-L$length-raw.csv"
done
