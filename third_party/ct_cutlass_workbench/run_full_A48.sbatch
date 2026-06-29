#!/bin/bash
#SBATCH -J ct_fullA48 -p h100 -A cssb -q cssb_h100 --gres=gpu:h100:1 -c 8 --mem=128G -t 00:40:00
#SBATCH --output=/home/psk6950/miniworld-kernels/_ct_cutlass/full_A48_%j.out
set -uo pipefail
cd /home/psk6950/miniworld-kernels/_ct_cutlass
ENV=/home/psk6950/miniworld-kernels/.pixi/envs/default
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/psk6950/miniworld-kernels/src:/home/psk6950/miniworld-kernels/_ct_cutlass
"$ENV/bin/python" bench_full_A48.py 2>&1
echo "[exit=$?] FULLA48 DONE"
