#!/bin/bash
# Quick H100 (sm_90) smoke: confirm the env builds, we land on an H100, the sm90 cute
# dispatch runs, and bench.py writes a CSV — BEFORE spending hours on the full sweep.
# Tiny sweep (transition, inference, L={384,512}, d=128). ~10 min.
#SBATCH --job-name=mwk_smoke_h100
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/modules/%x_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-kernels
export HYDRA_FULL_ERROR=1

echo "host=$(hostname)  date=$(date)"
pixi run --frozen python submits/_assert_device.py --major 9 || exit 3

echo ""
echo "=== smoke: transition inference L-sweep (tiny) ==="
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/runners/bench.py "$@"' _ \
  kernel=transition implementations='[pytorch,miniworld]' metric=time compile=true \
  cudagraph=manual mode=inference sweep_axis=seq_len name_suffix=smoke \
  mask_prob=0.0 min_seq_len=384 max_seq_len=512 seq_len_step=128 d_pair_values=[128] sweep_seq_len=384
echo "[bench exit=$?]"

echo ""
echo "=== CSVs produced under benchmarks/modules/transition/artifacts ==="
find benchmarks/modules/transition/artifacts -name '*smoke*.csv' -printf '%p\n' 2>/dev/null | tail -5
echo "SMOKE DONE"
