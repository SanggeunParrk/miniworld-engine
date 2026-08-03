#!/bin/bash
#SBATCH --job-name=ct_full
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/psk6950/miniworld-engine/_ct_cutlass/full.out

set -e
ENV=/home/psk6950/miniworld-engine/.pixi/envs/default
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/psk6950/miniworld-engine/src
export TORCH_CUDA_ARCH_LIST="9.0a"
cd /home/psk6950/miniworld-engine/_ct_cutlass
echo "host=$(hostname) date=$(date)"
$ENV/bin/python bench_full.py 2>&1
echo "=== FULL E2E DONE ==="
