#!/bin/bash
#SBATCH --job-name=ct_ncu2
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/home/psk6950/miniworld-kernels/_ct_tmp/ct_ncu2_%j.out
set -uo pipefail
cd /home/psk6950/miniworld-kernels
ENV=/home/psk6950/miniworld-kernels/.pixi/envs/default
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}; export PYTHONPATH=src
echo "=== sanity: script runs WITHOUT ncu ==="
"$ENV/bin/python" /home/psk6950/miniworld-kernels/_ct_tmp/prof_single.py
echo "=== ncu (full output, squeeze+expand, basic sections) ==="
"$ENV/bin/ncu" --target-processes application-only -k "regex:expand_swiglu|squeeze_gate" -c 2 \
  --section SpeedOfLight --section Occupancy --section WarpStateStats \
  "$ENV/bin/python" /home/psk6950/miniworld-kernels/_ct_tmp/prof_single.py 2>&1 | tail -140
echo "[exit=${PIPESTATUS[0]}] NCU2 DONE"
