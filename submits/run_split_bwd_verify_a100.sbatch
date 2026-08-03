#!/bin/bash
#SBATCH --job-name=mwk_split_bwd_verify
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=benchmarks/split_bwd_verify_a100_%j.out
set -uo pipefail
cd /home/psk6950/miniworld-engine
echo "host=$(hostname) date=$(date)"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python submits/_split_bwd_verify.py'
echo "[exit=$?]"
