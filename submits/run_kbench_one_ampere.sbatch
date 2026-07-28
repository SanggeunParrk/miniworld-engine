#!/bin/bash
# ONE kernel target per job (1 Ampere GPU), 2h cap. Isolates the monolithic kbench hang: the
# shared per-batch `wait` in kernel_targets_bench.sh blocks forever if any task hangs; per-target
# jobs let the other targets finish and make the hung target obvious (its job won't COMPLETE).
#
#   BENCH_TARGET=tri_attn sbatch --gres=gpu:A6000:1 submits/run_kbench_one_ampere.sbatch
#   BENCH_TARGET=tri_attn sbatch --gres=gpu:A5000:1 submits/run_kbench_one_ampere.sbatch
#SBATCH --job-name=mwk_kbench1_ampere
#SBATCH --partition=gpu
#SBATCH --account=cssb
#SBATCH --qos=normal
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/kernels/kbench1_ampere_%x_%j.out
set -uo pipefail
set -f
cd /home/psk6950/practice/miniworld-kernels
export HYDRA_FULL_ERROR=1
echo "host=$(hostname) target=${BENCH_TARGET} date=$(date)"
pixi run --frozen python submits/_assert_device.py --capability 8.6 --name A5000 --name A6000 || exit 3
TARGETS_OVERRIDE="${BENCH_TARGET}" bash submits/kernel_targets_bench.sh
echo "[exit=$?] KBENCH1 DONE target=${BENCH_TARGET}"
