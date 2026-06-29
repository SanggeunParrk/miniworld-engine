#!/bin/bash
#SBATCH --job-name=ct_imp
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/home/psk6950/miniworld-kernels/_ct_tmp/ct_imp_%j.out
set -uo pipefail
cd /home/psk6950/miniworld-kernels
ENV=/home/psk6950/miniworld-kernels/.pixi/envs/default
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=src
$ENV/bin/python -c "
from miniworld_kernels.kernels.conditioned_transition import cond_transition_fwd_12_345
from miniworld_kernels.kernels.conditioned_transition.triton import cond_transition_fwd_12_345 as t2
print('import OK', cond_transition_fwd_12_345 is t2)
"
echo "[exit=$?] CT IMP DONE"
