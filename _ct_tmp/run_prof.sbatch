#!/bin/bash
#SBATCH --job-name=ct_prof
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/home/psk6950/miniworld-kernels/_ct_tmp/ct_prof_%j.out
set -uo pipefail
cd /home/psk6950/miniworld-kernels
ENV=/home/psk6950/miniworld-kernels/.pixi/envs/default
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}; export PYTHONPATH=src
echo "===== ROOFLINE (per-kernel) ====="
$ENV/bin/python /home/psk6950/miniworld-kernels/_ct_tmp/prof.py
echo ""; echo "===== ncu availability ====="
NCU=$(which ncu 2>/dev/null || echo "$ENV/bin/ncu")
ls -la "$NCU" 2>/dev/null && "$NCU" --version 2>&1 | head -3 || echo "ncu NOT found"
echo "[exit=$?] PROF DONE"
