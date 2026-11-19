#!/bin/bash
# PWA FP8 training step-time benchmark vs the default route (same process, interleaved CUDA graphs, cool / sustained, same-cubin control).
# usage (from the repo root, with an env that has the engine on PYTHONPATH and a warm extension cache):
#   sbatch -J fp8t3-pwa-bench tools/fp8_train/run_pwa_step_bench.sbatch
# Cases and variants are overridable: CASES="1024:768 2048:768" VARIANTS="bwd=dv,dw/fused/e4m3;f8e4=fwd,dv,dw/fused/e4m3"
#SBATCH -p h100 -A cssb --qos=cssb_h100 --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=00:30:00
export DEADLINE=$(( $(date +%s) + 1700 ))
export PWA_BENCH_OUT=${PWA_BENCH_OUT:-pwa_step_bench_$(date +%Y%m%d_%H%M)}
python tools/fp8_train/pwa_step_bench.py ${CASES:-1024:768 2048:768 4096:768 1024:512 1024:384 2048:384 1024:768:256}
