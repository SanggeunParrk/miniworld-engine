#!/bin/bash
#SBATCH --job-name=ct_m0v
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/psk6950/miniworld-kernels/_ct_cutlass/m0v.out
set -e
ENV=/home/psk6950/miniworld-kernels/.pixi/envs/default
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0a"
cd /home/psk6950/miniworld-kernels/_ct_cutlass
rm -rf build
$ENV/bin/python setup_m0.py build_ext --inplace 2>&1 | grep -iE "error|warning #|assert|static_assert" | head -40
echo "=== BUILD EXIT ${PIPESTATUS[0]} ==="
