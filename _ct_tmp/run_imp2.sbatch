#!/bin/bash
#SBATCH --job-name=ct_imp2
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/home/psk6950/miniworld-kernels/_ct_tmp/ct_imp2_%j.out
set -uo pipefail
cd /home/psk6950/miniworld-kernels
ENV=/home/psk6950/miniworld-kernels/.pixi/envs/default
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=src
$ENV/bin/python -c "
from miniworld_kernels.kernels.conditioned_transition import cond_transition_train_12_345, ConditionedTransitionTail12345Function
from miniworld_kernels.kernels import conditioned_transition as ct
print('import OK'); print([n for n in ct.__all__ if '12_345' in n or '12345' in n])
"
echo "[exit=$?] CT IMP2 DONE"
