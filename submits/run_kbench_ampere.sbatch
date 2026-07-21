#!/bin/bash
# Ampere workstation GPU (RTX A5000 / A6000, sm_86) KERNEL-level benchmark sweep for the
# CSSB cluster (partition=gpu). Wraps submits/kernel_targets_bench.sh (fans the per-kernel
# targets across the allocation's GPUs) inside one gpu-partition allocation. Inference-only,
# cudagraph=manual, L + d sweeps, then renders. Baseline pass on the CURRENT code -> surfaces
# any sm_86 crash / OOM / arch misroute per kernel, exactly like the A100 kernel sweep.
#
# TRITON-ONLY like the A100: the per-kernel IMPLS lists in kernel_targets_bench.sh still
# include cute/cuda rows, but on sm_86 those either skip (arch-gated) or record status=failed;
# the triton rows are the meaningful comparison here.
#
# Pick the card + GPU count with --gres on the command line (default: 4x A6000). A5000 node
# (gpu02) and A6000 nodes (gpu01/03/04/05) each hold 8 GPUs:
#   sbatch --gres=gpu:A6000:4 submits/run_kbench_ampere.sbatch
#   sbatch --gres=gpu:A5000:4 submits/run_kbench_ampere.sbatch
# TARGETS_OVERRIDE (env) narrows the kernel list; default = all in kernel_targets_bench.sh.
#SBATCH --job-name=mwk_kbench_ampere
#SBATCH --partition=gpu
#SBATCH --account=cssb
#SBATCH --qos=normal
#SBATCH --gres=gpu:A6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=benchmarks/kernels/kbench_ampere_%x_%j.out

set -uo pipefail
set -f
cd /home/psk6950/practice/miniworld-kernels

echo "host=$(hostname)  date=$(date)"
pixi run --frozen python -c "
import torch
name = torch.cuda.get_device_name(0); cap = torch.cuda.get_device_capability(0)
print('torch', torch.__version__, name, 'sm', cap, 'ngpu', torch.cuda.device_count())
assert cap == (8, 6), f'expected sm_86 (RTX A5000/A6000), got sm_{cap[0]}{cap[1]} ({name})'
assert ('A5000' in name) or ('A6000' in name), f'unexpected Ampere card: {name}'
"

# kernel_targets_bench.sh reads TARGETS_OVERRIDE and fans across $(nvidia-smi -L).
bash submits/kernel_targets_bench.sh
echo "[kernel sweep exit=$?]"
echo "KERNEL BENCH DONE"
