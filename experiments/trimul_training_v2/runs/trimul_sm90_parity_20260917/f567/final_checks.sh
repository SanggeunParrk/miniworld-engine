set -e
source /home/psk6950/MiniWorld/runs/transition_triton_audit_20260917/env.sh
export PYTHONPATH=/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/src
out=/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/f567
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode=77 python -m pytest -q /home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/tests/numerics/test_trimul_parity_f567_gpu.py > "$out/memcheck_final.log" 2>&1
python "$out/final_benchmark.py" > "$out/final_benchmark.log" 2>&1
CUTE_DSL_KEEP=ptx,cubin CUTE_DSL_DUMP_DIR="$out/ptx" CUTE_DSL_NO_CACHE=1 python "$out/probe.py" > "$out/ptx.log" 2>&1
