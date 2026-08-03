#!/bin/bash
#SBATCH --job-name=mwk_b2b_ktiled_exp
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/b2b_ktiled_exp_a100_%j.out
set -uo pipefail
cd /home/psk6950/miniworld-engine
echo "host=$(hostname) date=$(date)"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python submits/_b2b_ktiled_exp.py'
echo "[exit=$?]"
