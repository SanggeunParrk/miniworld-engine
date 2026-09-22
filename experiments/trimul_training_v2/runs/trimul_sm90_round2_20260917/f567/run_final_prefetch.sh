set -euo pipefail
source /home/psk6950/MiniWorld/runs/transition_triton_audit_20260917/env.sh
cd /home/psk6950/MiniWorld/runs/trimul_sm90_round2_20260917/f567
export PYTHONPATH=/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/src PYTHONNOUSERSITE=1
unset CUTE_DSL_KEEP CUTE_DSL_KEEP_PTX CUTE_DSL_KEEP_CUBIN CUTE_DSL_LINEINFO
export CUTE_DSL_DUMP_DIR=$PWD/compile
python final_prefetch.py > prefetch_final.log 2>&1
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode=86 python -m pytest -q test_prefetch_initial.py > prefetch_memcheck.log 2>&1
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode=86 python -m pytest -q test_prefetch_initial.py > prefetch_racecheck.log 2>&1
