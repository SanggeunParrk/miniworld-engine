#!/bin/bash
# A100 KERNEL-level benchmark sweep (detached). Wraps submits/kernel_targets_bench.sh
# (which fans the per-kernel targets across the node's GPUs) inside a single A100
# allocation holding 4 GPUs. Runs inference-only, cudagraph=manual, L + d sweeps, then
# renders. Baseline pass on the CURRENT code -> surfaces A100 crashes / OOM / arch
# misroutes per kernel, exactly like the module sweep does for the modules.
#
# TARGETS_OVERRIDE (env) narrows the kernel list; default = all in kernel_targets_bench.sh.
#SBATCH --job-name=mwk_kbench_a100
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=benchmarks/kernels/kbench_a100_%x_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-kernels

echo "host=$(hostname)  date=$(date)"
pixi run --frozen python -c "import torch; cap=torch.cuda.get_device_capability(0); print('torch', torch.__version__, torch.cuda.get_device_name(0), 'sm', cap, 'ngpu', torch.cuda.device_count()); assert cap[0]==8, f'expected sm_80 (A100), got sm_{cap[0]}{cap[1]}'"

# kernel_targets_bench.sh reads TARGETS_OVERRIDE and fans across $(nvidia-smi -L).
bash submits/kernel_targets_bench.sh
echo "[kernel sweep exit=$?]"
echo "KERNEL BENCH DONE"
