#!/bin/bash
# ONE kernel target per job (1 A100), 2h cap. Isolates the monolithic kbench hang: the shared
# per-batch `wait` in kernel_targets_bench.sh blocks forever if any task hangs; per-target jobs
# let the other targets finish and make the hung target obvious (its job won't COMPLETE).
#SBATCH --job-name=mwk_kbench1
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/kernels/kbench1_%x_%j.out
set -uo pipefail
set -f
cd /home/psk6950/miniworld-kernels
export HYDRA_FULL_ERROR=1
echo "host=$(hostname) target=${BENCH_TARGET} date=$(date)"
pixi run --frozen python -c "import torch; cap=torch.cuda.get_device_capability(0); print(torch.cuda.get_device_name(0),'sm',cap); assert cap[0]==8, 'need A100 (sm80)'"
TARGETS_OVERRIDE="${BENCH_TARGET}" bash submits/kernel_targets_bench.sh
echo "[exit=$?] KBENCH1 DONE target=${BENCH_TARGET}"
