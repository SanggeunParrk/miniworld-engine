#!/bin/bash
#SBATCH --job-name=mwk_explicit_build
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/explicit_build_a100_%j.out
set -uo pipefail
cd /home/psk6950/miniworld-kernels
echo "host=$(hostname) date=$(date) cache=$MINIWORLD_KERNELS_CACHE_DIR"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python -m miniworld_kernels.autotune.build --op all'
echo "[build exit=$?]"
find "$MINIWORLD_KERNELS_CACHE_DIR/autotune" -name '*.json' 2>/dev/null | sed -E 's#.*/autotune/##' | sort
echo "EXPLICIT BUILD DONE"
