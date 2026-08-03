#!/bin/bash
# Backend-agnostic autotune framework API mechanics check (standalone .py, no quoting hell).
#SBATCH --job-name=mwk_autotune_api
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=benchmarks/autotune_api_a100_%j.out

set -uo pipefail
cd /home/psk6950/miniworld-engine
echo "host=$(hostname) date=$(date)"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; MINIWORLD_ENGINE_CACHE_DIR=$(mktemp -d) PYTHONPATH=src python submits/_fw_api_check.py'
echo "[api exit=$?]"
echo "AUTOTUNE API DONE"
