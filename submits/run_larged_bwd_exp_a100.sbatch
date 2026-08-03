#!/bin/bash
#SBATCH --job-name=mwk_larged_bwd_exp
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/larged_bwd_exp_a100_%j.out
set -uo pipefail
cd /home/psk6950/miniworld-engine
echo "host=$(hostname) date=$(date)"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python submits/_larged_bwd_exp.py'
echo "[exit=$?]"
