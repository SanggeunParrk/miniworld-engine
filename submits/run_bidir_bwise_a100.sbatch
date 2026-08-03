#!/bin/bash
# Verify the TRITON bidirectional trimul generalised to B>1 (batched grid axis + channel-outer
# preact). Correctness vs pytorch reference + batch-independence + B=1 regression timing.
# On A100 (sm80) the trimul miniworld backend IS triton, so this exercises the live path.
#SBATCH --job-name=mwk_bidir_bwise_a100
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/bidir_bwise_a100_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-engine
export HYDRA_FULL_ERROR=1
echo "host=$(hostname) date=$(date)"
pixi run --frozen python -c "import torch; assert torch.cuda.get_device_capability(0)[0]==8, 'need A100'; print('A100 ok')"

pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python submits/_bidir_bwise_verify.py'
echo "[exit=$?]"
echo "BIDIR_BWISE DONE"
