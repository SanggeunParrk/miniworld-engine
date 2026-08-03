#!/bin/bash
#SBATCH -J ct_stats -p h100 -A cssb -q cssb_h100 --gres=gpu:h100:1 -c 8 --mem=128G -t 00:40:00
#SBATCH --output=/home/psk6950/miniworld-engine/_ct_cutlass/stats_%j.out
set -uo pipefail; cd /home/psk6950/miniworld-engine/_ct_cutlass
ENV=/home/psk6950/miniworld-engine/.pixi/envs/default
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/psk6950/miniworld-engine/src:/home/psk6950/miniworld-engine/_ct_cutlass
"$ENV/bin/python" bench_stats_A48.py 2>&1
echo "[exit=$?] STATS DONE2"
