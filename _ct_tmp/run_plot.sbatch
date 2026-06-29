#!/bin/bash
#SBATCH --job-name=ct_plot
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:06:00
#SBATCH --output=/home/psk6950/miniworld-kernels/_ct_tmp/ct_plot_%j.out
set -uo pipefail
cd /home/psk6950/miniworld-kernels
ENV=/home/psk6950/miniworld-kernels/.pixi/envs/default
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=src
$ENV/bin/python /home/psk6950/miniworld-kernels/_ct_tmp/plot_ct.py \
  /home/psk6950/miniworld-kernels/_ct_tmp/ct_bench_9903.out \
  /home/psk6950/miniworld-kernels/src/miniworld_kernels/kernels/conditioned_transition/benchmark 2>&1
echo "[exit=$?] CT PLOT DONE"
