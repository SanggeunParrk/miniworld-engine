#!/bin/bash
# H100 (sm_90) KERNEL-level benchmark sweep. Holds node22's 8 H100s in ONE allocation and
# wraps submits/kernel_targets_bench.sh, which fans the per-kernel targets across the GPUs
# (per-GPU TRITON_CACHE_DIR). inference-only, cudagraph=manual, L + d sweeps, then renders.
# This exercises the sm90 cute kernel variants directly (per-kernel rows incl cute paths).
# TARGETS_OVERRIDE (env) narrows the kernel list; default = all in kernel_targets_bench.sh.
#SBATCH --job-name=mwk_kbench_h100
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=benchmarks/kernels/%x_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-kernels

echo "host=$(hostname)  date=$(date)"
# FAIL-FAST CUDA/sm90 gate with retry (HGX nodes can throw CUDA error 802 right after alloc).
ok=0
for try in 1 2 3 4 5; do
  if pixi run --frozen python submits/_assert_device.py --major 9; then
    ok=1; break
  fi
  echo "[cuda check try $try failed — retrying in 20s]"; sleep 20
done
[ "$ok" = 1 ] || { echo "FATAL: CUDA not usable on $(hostname) after retries — aborting"; exit 1; }

# kernel_targets_bench.sh reads TARGETS_OVERRIDE and fans across $(nvidia-smi -L).
bash submits/kernel_targets_bench.sh
echo "[kernel sweep exit=$?]"
echo "KERNEL BENCH DONE (H100)"
