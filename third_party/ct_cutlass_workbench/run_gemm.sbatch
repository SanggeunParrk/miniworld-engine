#!/bin/bash
#SBATCH --job-name=ct_gemm
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/home/psk6950/miniworld-engine/_ct_cutlass/gemm.out

set -e
ENV=/home/psk6950/miniworld-engine/.pixi/envs/default
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/psk6950/miniworld-engine/src
export TORCH_CUDA_ARCH_LIST="9.0a"
export MAX_JOBS=8
cd /home/psk6950/miniworld-engine/_ct_cutlass
echo "=== BUILD ==="
$ENV/bin/python setup_gemm.py build_ext --inplace 2>&1 | grep -vE "^ptxas|wgmma.mma_async|Potential Performance" | tail -40
echo "=== BUILD EXIT ${PIPESTATUS[0]} ==="
echo "=== PICK ==="
$ENV/bin/python gemm_pick.py 2>&1
echo "=== DONE ==="
