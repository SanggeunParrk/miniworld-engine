#!/bin/bash
# Build the shipped autotune-config cache on A100 for all adopted ops, then copy the runtime
# cache into the committed package-data dir (src/miniworld_engine/autotune/data/<op>/<gpu>.json).
# Commit those JSONs so A100 gets tuned configs out of the box. (H100/B200 run this on their box.)
#SBATCH --job-name=mwk_autotune_build
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=benchmarks/autotune_build_a100_%j.out

set -uo pipefail
cd /home/psk6950/miniworld-kernels
echo "host=$(hostname) date=$(date)"
pixi run --frozen python -c "import torch; assert torch.cuda.get_device_capability(0)[0]==8, 'need A100'; print('A100 ok')"

pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python -m miniworld_engine.autotune.build --op all'
echo "[build exit=$?]"

# Copy runtime cache -> shipped package data (committed).
RT="${MINIWORLD_KERNELS_CACHE_DIR:-$HOME/.cache/miniworld_engine}/autotune"
DST="src/miniworld_engine/autotune/data"
echo "=== copying runtime cache $RT -> $DST ==="
if [ -d "$RT" ]; then
  cp -rv "$RT/." "$DST/"
  echo "=== shipped cache files ==="; find "$DST" -name '*.json' -print
else
  echo "NO runtime cache produced at $RT"
fi
echo "AUTOTUNE BUILD DONE"
