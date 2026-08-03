#!/bin/bash
# Demonstrate the cache works as intended on the real GPU for both (A) known GPU -> HIT +
# prune narrows, and (B) unknown GPU -> miss + warn + full grid.
#SBATCH --job-name=mwk_cache_hit
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=benchmarks/cache_hit_a100_%j.out

set -uo pipefail
cd /home/psk6950/miniworld-engine
echo "host=$(hostname) date=$(date)"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python submits/_cache_hit_check.py'
echo "[exit=$?]"
echo "CACHE HIT CHECK DONE"
